from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    __package__ = "sg_eval"


CLIP_MODEL = "openai/clip-vit-large-patch14-336"
PREDICATE_MODEL = "jinaai/jina-embeddings-v2-base-en"

OBJ_RECALL_K = [1, 5, 10]
PRED_RECALL_K = [1, 3, 5, 20, 50, 100]
TRIPLET_RECALL_K = [1, 20, 50, 100]

NONE_LABEL = "none"


def _load_vocab(
    txt_path: Optional[str],
    train_path: Optional[str],
    val_path: Optional[str],
    extractor: Callable[[str], Set[str]],
    fallback: Set[str],
) -> Set[str]:
    from .io_utils import load_vocab_from_txt

    if txt_path:
        return load_vocab_from_txt(txt_path)
    vocab: Set[str] = set()
    for path in (train_path, val_path):
        if path:
            vocab.update(extractor(path))
    return vocab or fallback


def _labels_from_ground_truths(
    ground_truths: Dict[str, SceneGraph],
) -> Tuple[Set[str], Set[str]]:
    object_labels = {obj.label for sg in ground_truths.values() for obj in sg.objects}
    predicate_labels = {
        rel.predicate for sg in ground_truths.values() for rel in sg.relationships
    }
    return object_labels, predicate_labels


def _predicate_counts(scene_graphs: Dict[str, SceneGraph]) -> Counter:
    counts: Counter = Counter()
    for scene_graph in scene_graphs.values():
        counts.update(rel.predicate for rel in scene_graph.relationships)
    return counts


def _select_tail_predicates(
    validation_labels: Set[str],
    train_counts: Counter,
    fraction: float = 0.30,
) -> List[str]:
    if not validation_labels:
        return []
    count = max(1, math.ceil(len(validation_labels) * fraction))
    return sorted(
        validation_labels,
        key=lambda label: (train_counts.get(label, 0), label),
    )[:count]


def _split_predicates_by_frequency(
    validation_labels: Set[str],
    train_counts: Counter,
    fraction: float = 0.30,
) -> Tuple[List[str], List[str], List[str]]:
    """Return disjoint head, body, and tail predicate groups."""
    if not validation_labels:
        return [], [], []

    ordered = sorted(
        validation_labels,
        key=lambda label: (train_counts.get(label, 0), label),
    )
    tail_count = max(1, math.ceil(len(ordered) * fraction))
    head_count = min(tail_count, len(ordered) - tail_count)
    tail_labels = ordered[:tail_count]
    head_labels = ordered[len(ordered) - head_count:] if head_count else []
    body_labels = ordered[tail_count:len(ordered) - head_count]
    return head_labels, body_labels, tail_labels


def _concat_rank_arrays(rank_arrays: List[np.ndarray]) -> np.ndarray:
    non_empty = [np.asarray(arr, dtype=float) for arr in rank_arrays if len(arr) > 0]
    if not non_empty:
        return np.array([], dtype=float)
    return np.concatenate(non_empty)


def _scene_average_recall(rank_arrays: List[np.ndarray], k: int) -> float:
    recalls = [
        float(np.sum(arr <= k)) / float(len(arr))
        for arr in rank_arrays
        if len(arr) > 0
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def _summarize_recall(
    rank_arrays: List[np.ndarray],
    labels: List[str],
    k_values: Iterable[int],
) -> Dict[int, Tuple[float, float]]:
    from .metrics import compute_mrecall_at_k

    ranks = _concat_rank_arrays(rank_arrays)
    return {
        k: (
            _scene_average_recall(rank_arrays, k),
            compute_mrecall_at_k(ranks, labels, k),
        )
        for k in k_values
    }


def _format_table(rows: List[List[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(row: List[str]) -> str:
        cells = [f" {value:<{widths[i]}} " for i, value in enumerate(row)]
        return "|" + "|".join(cells) + "|"

    lines = [border, format_row(rows[0]), border]
    lines.extend(format_row(row) for row in rows[1:])
    lines.append(border)
    return "\n".join(lines)


def _print_recall_table(title: str, metrics: Dict[int, Tuple[float, float]]) -> None:
    rows = [["K", "R@K", "mR@K"]]
    for k, (recall, mean_recall) in metrics.items():
        rows.append([str(k), f"{recall:.4f}", f"{mean_recall:.4f}"])
    print(f"\n{title}")
    print(_format_table(rows))


def _print_predicate_table(
    labels: List[str],
    details: Dict[str, Tuple[int, Dict[int, float]]],
    validation_counts: Counter,
) -> None:
    rows = [
        ["Predicate", "Val occurrences", *(f"R@{k}" for k in PRED_RECALL_K)]
    ]
    for label in labels:
        _, recalls = details.get(label, (0, {}))
        rows.append([
            label,
            str(validation_counts.get(label, 0)),
            *(f"{recalls.get(k, 0.0):.4f}" for k in PRED_RECALL_K),
        ])
    mean_recalls = {
        k: (
            float(np.mean([
                details.get(label, (0, {}))[1].get(k, 0.0)
                for label in labels
            ]))
            if labels
            else 0.0
        )
        for k in PRED_RECALL_K
    }
    rows.append([
        "Mean recall",
        str(sum(validation_counts.get(label, 0) for label in labels)),
        *(f"{mean_recalls[k]:.4f}" for k in PRED_RECALL_K),
    ])
    print("\nValidation Predicate Per-Class Recall@K")
    print(_format_table(rows))


def _precompute_matchers(
    predictions: Dict[str, SceneGraph],
    ground_truths: Dict[str, SceneGraph],
    obj_vocab: List[str],
    pred_vocab: List[str],
) -> Tuple[CLIPObjectMatcher, JinaPredicateMatcher, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    from .matchers import CLIPObjectMatcher, JinaPredicateMatcher

    pred_obj_labels = {obj.label for sg in predictions.values() for obj in sg.objects}
    pred_rel_labels = {rel.predicate for sg in predictions.values() for rel in sg.relationships}
    gt_obj_labels = {obj.label for sg in ground_truths.values() for obj in sg.objects}

    obj_matcher = CLIPObjectMatcher(CLIP_MODEL, verbose=False)
    pred_matcher = JinaPredicateMatcher(PREDICATE_MODEL, verbose=False)

    object_labels = sorted(set(obj_vocab) | pred_obj_labels | gt_obj_labels)
    obj_matcher.precompute(object_labels)
    pred_matcher.precompute(sorted(set(pred_vocab) | pred_rel_labels | {NONE_LABEL}))

    obj_vocab_embs = obj_matcher.get_embeddings(obj_vocab)
    pred_vocab_embs = pred_matcher.get_embeddings(pred_vocab)
    label_embeddings = {label: obj_matcher.encode(label) for label in object_labels}
    return obj_matcher, pred_matcher, obj_vocab_embs, pred_vocab_embs, label_embeddings


def evaluate(
    predictions: Dict[str, SceneGraph],
    ground_truths: Dict[str, SceneGraph],
    obj_labels: Set[str],
    pred_labels: Set[str],
    tail_predicate_labels: Optional[Set[str]] = None,
    tail_predicate_details: Optional[
        Dict[str, Tuple[int, Dict[int, float]]]
    ] = None,
    predicate_details: Optional[
        Dict[str, Tuple[int, Dict[int, float]]]
    ] = None,
    head_predicate_labels: Optional[Set[str]] = None,
    body_predicate_labels: Optional[Set[str]] = None,
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    from .metrics import (
        evaluate_recall_metrics,
        evaluate_recall_metrics_open3dsg_parity,
    )

    common_scenes = sorted(set(predictions) & set(ground_truths))
    if not common_scenes:
        raise ValueError("No overlapping scene ids between predictions and ground truth.")
    if not obj_labels:
        raise ValueError("Object vocabulary is empty.")
    if not pred_labels:
        raise ValueError("Predicate vocabulary is empty.")

    obj_vocab = sorted(obj_labels)
    pred_vocab = sorted(pred_labels)
    (
        obj_matcher,
        pred_matcher,
        obj_vocab_embs,
        pred_vocab_embs,
        label_embeddings,
    ) = _precompute_matchers(predictions, ground_truths, obj_vocab, pred_vocab)

    object_rank_arrays: List[np.ndarray] = []
    object_gt_labels: List[str] = []
    predicate_rank_arrays: List[np.ndarray] = []
    predicate_label_arrays: List[List[str]] = []
    predicate_gt_labels: List[str] = []
    triplet_rank_arrays: List[np.ndarray] = []
    triplet_gt_labels: List[str] = []

    for scene_id in common_scenes:
        recall_result = evaluate_recall_metrics(
            predictions[scene_id],
            ground_truths[scene_id],
            obj_matcher,
            pred_matcher,
            obj_vocab,
            obj_vocab_embs,
            pred_vocab,
            pred_vocab_embs,
            label_embeddings=label_embeddings,
            allowed_classes=obj_labels,
            scene_id=scene_id,
        )
        object_rank_arrays.append(recall_result["obj_ranks"])
        object_gt_labels.extend(recall_result["obj_gt_labels"])

        parity_result = evaluate_recall_metrics_open3dsg_parity(
            predictions[scene_id],
            recall_result["gt_sg_eval"],
            recall_result["geo_matching"],
            obj_matcher,
            pred_matcher,
            obj_vocab,
            obj_vocab_embs,
            pred_vocab,
            none_label=NONE_LABEL,
        )
        predicate_rank_arrays.append(parity_result["parity_pred_ranks"])
        scene_predicate_labels = parity_result["parity_pred_gt_labels"]
        predicate_label_arrays.append(scene_predicate_labels)
        predicate_gt_labels.extend(scene_predicate_labels)
        triplet_rank_arrays.append(parity_result["parity_triplet_ranks"])
        triplet_gt_labels.extend(parity_result["parity_triplet_gt_labels"])

    results = {
        "Object Recall@K": _summarize_recall(
            object_rank_arrays,
            object_gt_labels,
            OBJ_RECALL_K,
        ),
        "Predicate Recall@K": _summarize_recall(
            predicate_rank_arrays,
            predicate_gt_labels,
            PRED_RECALL_K,
        ),
        "Triplet Recall@K": _summarize_recall(
            triplet_rank_arrays,
            triplet_gt_labels,
            TRIPLET_RECALL_K,
        ),
    }
    for group_name, group_labels in (
        ("Head", head_predicate_labels),
        ("Body", body_predicate_labels),
    ):
        if group_labels is None:
            continue
        group_rank_arrays: List[np.ndarray] = []
        group_gt_labels: List[str] = []
        for ranks, labels in zip(predicate_rank_arrays, predicate_label_arrays):
            mask = np.asarray(
                [label in group_labels for label in labels],
                dtype=bool,
            )
            group_rank_arrays.append(np.asarray(ranks)[mask])
            group_gt_labels.extend(label for label in labels if label in group_labels)
        results[f"{group_name} Predicate Recall@K"] = _summarize_recall(
            group_rank_arrays,
            group_gt_labels,
            PRED_RECALL_K,
        )
    if tail_predicate_labels is not None:
        tail_rank_arrays: List[np.ndarray] = []
        tail_gt_labels: List[str] = []
        for ranks, labels in zip(predicate_rank_arrays, predicate_label_arrays):
            mask = np.asarray(
                [label in tail_predicate_labels for label in labels],
                dtype=bool,
            )
            tail_rank_arrays.append(np.asarray(ranks)[mask])
            tail_gt_labels.extend(
                label for label in labels if label in tail_predicate_labels
            )
        results["Tail Predicate Recall@K"] = _summarize_recall(
            tail_rank_arrays,
            tail_gt_labels,
            PRED_RECALL_K,
        )
        if tail_predicate_details is not None:
            all_tail_ranks = _concat_rank_arrays(tail_rank_arrays)
            labels_array = np.asarray(tail_gt_labels, dtype=object)
            for label in sorted(tail_predicate_labels):
                label_ranks = all_tail_ranks[labels_array == label]
                tail_predicate_details[label] = (
                    int(label_ranks.shape[0]),
                    {
                        k: (
                            float(np.sum(label_ranks <= k)) / float(len(label_ranks))
                            if len(label_ranks) > 0
                            else 0.0
                        )
                        for k in PRED_RECALL_K
                    },
                )
    if predicate_details is not None:
        all_predicate_ranks = _concat_rank_arrays(predicate_rank_arrays)
        labels_array = np.asarray(predicate_gt_labels, dtype=object)
        validation_predicate_labels = {
            rel.predicate
            for scene_graph in ground_truths.values()
            for rel in scene_graph.relationships
        } | {NONE_LABEL}
        for label in sorted(validation_predicate_labels):
            label_ranks = all_predicate_ranks[labels_array == label]
            predicate_details[label] = (
                int(label_ranks.shape[0]),
                {
                    k: (
                        float(np.sum(label_ranks <= k)) / float(len(label_ranks))
                        if len(label_ranks) > 0
                        else 0.0
                    )
                    for k in PRED_RECALL_K
                },
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser("Scene graph evaluation")
    parser.add_argument("-p", "--predictions", required=True, help="Prediction JSON file or directory.")
    parser.add_argument("-g", "--ground_truth", required=True, help="Validation ground-truth JSON file.")
    parser.add_argument("-d", "--dataset_dir", default=None, help="Dataset root for GT empty-bbox filtering.")
    parser.add_argument("--classes_txt", default=None, help="Optional object class vocabulary txt file.")
    parser.add_argument("--pred_vocab_txt", default=None, help="Optional predicate vocabulary txt file.")
    parser.add_argument("--train_scene_graphs", default=None, help="Optional training scene-graph JSON for vocab extraction.")
    parser.add_argument("--val_scene_graphs", default=None, help="Optional validation scene-graph JSON for vocab extraction.")
    parser.add_argument(
        "--tail_predicate_metrics",
        action="store_true",
        help=(
            "Report soft predicate Recall@K/mR@K for head, body, and tail "
            "training-frequency groups, plus per-class recall for every "
            "validation predicate."
        ),
    )
    parser.add_argument(
        "--predicate_frequency_source",
        default=None,
        help=(
            "Actual training scene-graph JSON used to rank predicate classes by "
            "frequency for --tail_predicate_metrics."
        ),
    )
    args = parser.parse_args()

    if args.tail_predicate_metrics and not args.predicate_frequency_source:
        parser.error("--tail_predicate_metrics requires --predicate_frequency_source")

    from .io_utils import (
        extract_object_labels_from_scene_graph_json,
        extract_predicates_from_scene_graph_json,
        filter_gt_with_point_clouds,
        load_ground_truth,
        load_predictions,
    )

    ground_truths, pcd_paths = load_ground_truth(args.ground_truth)
    if args.dataset_dir:
        ground_truths = filter_gt_with_point_clouds(
            ground_truths,
            pcd_paths,
            args.dataset_dir,
            verbose=False,
        )

    predictions, _, _, _ = load_predictions(args.predictions)
    gt_obj_labels, gt_pred_labels = _labels_from_ground_truths(ground_truths)
    obj_labels = _load_vocab(
        args.classes_txt,
        args.train_scene_graphs,
        args.val_scene_graphs,
        extract_object_labels_from_scene_graph_json,
        gt_obj_labels,
    )
    pred_labels = _load_vocab(
        args.pred_vocab_txt,
        args.train_scene_graphs,
        args.val_scene_graphs,
        extract_predicates_from_scene_graph_json,
        gt_pred_labels,
    )

    tail_predicate_labels = None
    head_predicate_labels = None
    body_predicate_labels = None
    tail_predicate_details = None
    predicate_details = None
    validation_predicate_counts = None
    selected_tail_labels: List[str] = []
    if args.tail_predicate_metrics:
        frequency_graphs, _ = load_ground_truth(args.predicate_frequency_source)
        train_counts = _predicate_counts(frequency_graphs)
        (
            selected_head_labels,
            selected_body_labels,
            selected_tail_labels,
        ) = _split_predicates_by_frequency(gt_pred_labels, train_counts)
        selected_body_labels.append(NONE_LABEL)
        head_predicate_labels = set(selected_head_labels)
        body_predicate_labels = set(selected_body_labels)
        tail_predicate_labels = set(selected_tail_labels)
        tail_predicate_details = {}
        predicate_details = {}
        validation_predicate_counts = _predicate_counts(ground_truths)
        print("\nPredicate classes by training frequency")
        for group_name, labels in (
            ("Head (top 30%)", selected_head_labels),
            ("Body (middle 40%)", selected_body_labels),
            ("Tail (bottom 30%)", selected_tail_labels),
        ):
            print(f"  {group_name}")
            for label in labels:
                if label == NONE_LABEL:
                    print(f"    {label}: synthesized class")
                else:
                    print(
                        f"    {label}: {train_counts.get(label, 0)} "
                        "training occurrences"
                    )

    metrics = evaluate(
        predictions,
        ground_truths,
        obj_labels,
        pred_labels,
        tail_predicate_labels=tail_predicate_labels,
        head_predicate_labels=head_predicate_labels,
        body_predicate_labels=body_predicate_labels,
        tail_predicate_details=tail_predicate_details,
        predicate_details=predicate_details,
    )
    for title, table_metrics in metrics.items():
        _print_recall_table(title, table_metrics)
    if predicate_details is not None:
        validation_predicate_counts[NONE_LABEL] = predicate_details[
            NONE_LABEL
        ][0]
        _print_predicate_table(
            sorted(gt_pred_labels | {NONE_LABEL}),
            predicate_details,
            validation_predicate_counts,
        )


if __name__ == "__main__":
    main()
