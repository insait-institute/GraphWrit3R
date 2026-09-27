from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data_types import Object3D, Relationship, SceneGraph
from .iou import compute_iou_3d
from .matchers import CLIPObjectMatcher, PredicateMatcher


LAYOUT_OBJECT_LABELS: Set[str] = {"ceiling", "floor", "wall"}


def compute_mrecall_at_k(
    ranks: np.ndarray,
    gt_labels: List[str],
    k: int,
) -> float:
    """Per-class Recall@K, then mean across classes."""
    class_hits: Dict[str, List[float]] = defaultdict(list)
    for rank, label in zip(ranks, gt_labels):
        class_hits[label].append(1.0 if rank <= k else 0.0)
    per_class = [np.mean(v) for v in class_hits.values()]
    return float(np.mean(per_class)) if per_class else 0.0


def _adaptive_centroid_fallback_max_distance(
    gt_obj: Object3D,
    base_max_distance: float,
    gt_diagonal_fraction: float,
    layout_max_distance: float,
) -> float:
    if gt_obj.label in LAYOUT_OBJECT_LABELS:
        return float(layout_max_distance)
    gt_diagonal = float(np.linalg.norm(np.asarray(gt_obj.size, dtype=float)))
    return float(max(base_max_distance, gt_diagonal_fraction * gt_diagonal))


def _normalize_predicate_label(label: str, none_label: str, none_aliases: Set[str]) -> str:
    value = str(label).lower().strip()
    return none_label if value in none_aliases else value


def _build_parity_pred_vocab(
    pred_vocab: List[str],
    pred_matcher: PredicateMatcher,
    none_label: str,
    none_aliases: Set[str],
) -> Tuple[List[str], np.ndarray]:
    normalized_vocab: List[str] = []
    seen: Set[str] = set()
    for pred in pred_vocab:
        pred_norm = _normalize_predicate_label(pred, none_label, none_aliases)
        if pred_norm not in seen:
            normalized_vocab.append(pred_norm)
            seen.add(pred_norm)

    normalized_vocab = [none_label] + [pred for pred in normalized_vocab if pred != none_label]
    pred_matcher.precompute(normalized_vocab)
    return normalized_vocab, pred_matcher.get_embeddings(normalized_vocab)


def get_geometric_matching(
    pred_sg: SceneGraph,
    gt_sg: SceneGraph,
    iou_threshold: float = 0.1,
    semantic_cost_weight: float = 2.0,
    label_embeddings: Optional[Dict[str, np.ndarray]] = None,
    centroid_fallback: bool = True,
    centroid_fallback_threshold: float = 1.0,
    centroid_fallback_semantic_threshold: Optional[float] = None,
    centroid_fallback_max_distance: float = 4.0,
    centroid_fallback_gt_diagonal_fraction: float = 0.75,
    centroid_fallback_layout_max_distance: float = 6.0,
    centroid_fallback_iou_cost_weight: float = 1.0,
) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
    """Match predicted objects to GT objects using IoU, semantics, and centroid fallback."""
    pred_objs = pred_sg.objects
    gt_objs = gt_sg.objects
    if not pred_objs or not gt_objs:
        return {}, {}

    large_cost = 1e6
    num_pred = len(pred_objs)
    num_gt = len(gt_objs)
    iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float32)
    center_distance_matrix = np.full((num_pred, num_gt), np.inf, dtype=np.float32)
    cost_matrix = np.full((num_pred, num_gt), large_cost, dtype=np.float32)

    semantic_sim = None
    if semantic_cost_weight > 0.0 and label_embeddings:
        first_emb = next(iter(label_embeddings.values()))
        zero_emb = np.zeros_like(first_emb)
        pred_label_embs = np.stack([label_embeddings.get(obj.label, zero_emb) for obj in pred_objs])
        gt_label_embs = np.stack([label_embeddings.get(obj.label, zero_emb) for obj in gt_objs])
        semantic_sim = pred_label_embs @ gt_label_embs.T

    for i, pred_obj in enumerate(pred_objs):
        for j, gt_obj in enumerate(gt_objs):
            iou = float(compute_iou_3d(pred_obj, gt_obj))
            center_distance = float(
                np.linalg.norm(
                    np.asarray(pred_obj.position, dtype=float)
                    - np.asarray(gt_obj.position, dtype=float)
                )
            )
            iou_matrix[i, j] = iou
            center_distance_matrix[i, j] = center_distance
            if iou >= iou_threshold:
                sem_term = semantic_cost_weight * float(semantic_sim[i, j]) if semantic_sim is not None else 0.0
                cost_matrix[i, j] = -iou - sem_term

    matching: Dict[int, int] = {}
    match_details: Dict[int, Dict[str, Any]] = {}
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] >= large_cost:
            continue
        pred_id = int(pred_objs[i].id)
        gt_id = int(gt_objs[j].id)
        matching[pred_id] = gt_id
        match_details[pred_id] = {
            "gt_id": gt_id,
            "match_type": "iou",
            "iou": float(iou_matrix[i, j]),
            "center_distance": float(center_distance_matrix[i, j]),
        }

    if not centroid_fallback or centroid_fallback_threshold <= 0.0:
        return matching, match_details

    matched_pred_ids = set(matching.keys())
    matched_gt_ids = set(matching.values())
    unmatched_pred_indices = [
        i for i, pred_obj in enumerate(pred_objs)
        if int(pred_obj.id) not in matched_pred_ids
    ]
    unmatched_gt_indices = [
        j for j, gt_obj in enumerate(gt_objs)
        if int(gt_obj.id) not in matched_gt_ids
    ]
    if not unmatched_pred_indices or not unmatched_gt_indices:
        return matching, match_details

    fallback_cost = np.full(
        (len(unmatched_pred_indices), len(unmatched_gt_indices)),
        large_cost,
        dtype=np.float32,
    )
    base_max_distance = float(max(0.0, centroid_fallback_max_distance))
    layout_max_distance = float(max(base_max_distance, centroid_fallback_layout_max_distance))
    gt_diagonal_fraction = float(max(0.0, centroid_fallback_gt_diagonal_fraction))

    for fi, i in enumerate(unmatched_pred_indices):
        for fj, j in enumerate(unmatched_gt_indices):
            center_distance = float(center_distance_matrix[i, j])
            max_distance = _adaptive_centroid_fallback_max_distance(
                gt_objs[j],
                base_max_distance,
                gt_diagonal_fraction,
                layout_max_distance,
            )
            if center_distance > max_distance:
                continue

            sem_raw = float(semantic_sim[i, j]) if semantic_sim is not None else 0.0
            if (
                centroid_fallback_semantic_threshold is not None
                and sem_raw < centroid_fallback_semantic_threshold
            ):
                continue

            distance_cost = center_distance / centroid_fallback_threshold
            sem_term = semantic_cost_weight * sem_raw
            iou_term = centroid_fallback_iou_cost_weight * float(iou_matrix[i, j])
            fallback_cost[fi, fj] = 1.0 + distance_cost - sem_term - iou_term

    f_row_ind, f_col_ind = linear_sum_assignment(fallback_cost)
    for fi, fj in zip(f_row_ind, f_col_ind):
        if fallback_cost[fi, fj] >= large_cost:
            continue
        i = unmatched_pred_indices[fi]
        j = unmatched_gt_indices[fj]
        pred_id = int(pred_objs[i].id)
        gt_id = int(gt_objs[j].id)
        if pred_id in matching or gt_id in matching.values():
            continue
        matching[pred_id] = gt_id
        match_details[pred_id] = {
            "gt_id": gt_id,
            "match_type": "centroid_fallback",
            "iou": float(iou_matrix[i, j]),
            "center_distance": float(center_distance_matrix[i, j]),
        }

    return matching, match_details


def evaluate_recall_metrics(
    pred_sg: SceneGraph,
    gt_sg: SceneGraph,
    obj_matcher: CLIPObjectMatcher,
    pred_matcher: Optional[PredicateMatcher],
    obj_vocab: List[str],
    obj_vocab_embs: np.ndarray,
    pred_vocab: List[str],
    pred_vocab_embs: Optional[np.ndarray],
    iou_threshold: float = 0.1,
    semantic_cost_weight: float = 1.0,
    label_embeddings: Optional[Dict[str, np.ndarray]] = None,
    allowed_classes: Optional[Set[str]] = None,
    scene_id: str = "",
    centroid_fallback: bool = True,
    centroid_fallback_threshold: float = 1.0,
    centroid_fallback_semantic_threshold: Optional[float] = None,
    centroid_fallback_max_distance: float = 4.0,
    centroid_fallback_gt_diagonal_fraction: float = 0.75,
    centroid_fallback_layout_max_distance: float = 6.0,
    centroid_fallback_iou_cost_weight: float = 1.0,
    evaluate_relationships: bool = True,
) -> Dict:
    """Compute object Recall@K inputs and the geometry grounding used by relation eval."""
    pred_vocab_set = set(pred_vocab) if evaluate_relationships else set()
    gt_eval_objects = [
        obj for obj in gt_sg.objects
        if allowed_classes is None or obj.label in allowed_classes
    ]
    gt_eval_ids = {obj.id for obj in gt_eval_objects}
    gt_eval_relationships: List[Relationship] = []
    if evaluate_relationships:
        for rel in gt_sg.relationships:
            if rel.subject_id not in gt_eval_ids or rel.object_id not in gt_eval_ids:
                continue
            if rel.predicate not in pred_vocab_set:
                continue
            gt_eval_relationships.append(rel)

    gt_sg_eval = SceneGraph(objects=gt_eval_objects, relationships=gt_eval_relationships)
    geo_matching, geo_match_details = get_geometric_matching(
        pred_sg,
        gt_sg_eval,
        iou_threshold=iou_threshold,
        semantic_cost_weight=semantic_cost_weight,
        label_embeddings=label_embeddings,
        centroid_fallback=centroid_fallback,
        centroid_fallback_threshold=centroid_fallback_threshold,
        centroid_fallback_semantic_threshold=centroid_fallback_semantic_threshold,
        centroid_fallback_max_distance=centroid_fallback_max_distance,
        centroid_fallback_gt_diagonal_fraction=centroid_fallback_gt_diagonal_fraction,
        centroid_fallback_layout_max_distance=centroid_fallback_layout_max_distance,
        centroid_fallback_iou_cost_weight=centroid_fallback_iou_cost_weight,
    )
    gt_to_pred = {gt_id: pred_id for pred_id, gt_id in geo_matching.items()}
    pred_obj_map = {obj.id: obj for obj in pred_sg.objects}

    obj_ranks: List[float] = []
    obj_gt_labels: List[str] = []
    for gt_obj in gt_sg_eval.objects:
        if gt_obj.id in gt_to_pred:
            pred_obj = pred_obj_map[gt_to_pred[gt_obj.id]]
            rank = obj_matcher.rank_in_vocab(pred_obj.label, gt_obj.label, obj_vocab, obj_vocab_embs)
        else:
            rank = float("inf")
        obj_ranks.append(rank)
        obj_gt_labels.append(gt_obj.label)

    return {
        "obj_ranks": np.array(obj_ranks, dtype=float),
        "obj_gt_labels": obj_gt_labels,
        "geo_matching": geo_matching,
        "geo_match_details": geo_match_details,
        "gt_sg_eval": gt_sg_eval,
    }


def evaluate_recall_metrics_open3dsg_parity(
    pred_sg: SceneGraph,
    gt_sg_eval: SceneGraph,
    geo_matching: Dict[int, int],
    obj_matcher: CLIPObjectMatcher,
    pred_matcher: PredicateMatcher,
    obj_vocab: List[str],
    obj_vocab_embs: np.ndarray,
    pred_vocab: List[str],
    none_label: str = "none",
    none_distance_threshold: float = 0.5,
    rel_topk: int = 100,
    implicit_none_for_missing_edges: bool = True,
) -> Dict:
    """Open3DSG-style predicate and triplet recall with dense none-pair synthesis."""
    none_label = str(none_label).lower().strip()
    none_aliases: Set[str] = {
        none_label,
        "no relation",
        "no relationship",
        "unrelated",
        "not related",
        "not",
    }
    parity_pred_vocab, parity_pred_vocab_embs = _build_parity_pred_vocab(
        pred_vocab,
        pred_matcher,
        none_label,
        none_aliases,
    )

    gt_to_pred = {gt_id: pred_id for pred_id, gt_id in geo_matching.items()}
    pred_obj_map = {obj.id: obj for obj in pred_sg.objects}
    gt_obj_map = {obj.id: obj for obj in gt_sg_eval.objects}

    gt_edge_map: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for rel in gt_sg_eval.relationships:
        gt_edge_map[(rel.subject_id, rel.object_id)].append(
            _normalize_predicate_label(rel.predicate, none_label, none_aliases)
        )

    pred_edge_map: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for rel in pred_sg.relationships:
        pred_edge_map[(rel.subject_id, rel.object_id)].append(
            _normalize_predicate_label(rel.predicate, none_label, none_aliases)
        )

    dense_gt_relations: List[Tuple[int, int, str]] = []
    gt_ids = [obj.id for obj in gt_sg_eval.objects]
    for sid in gt_ids:
        for oid in gt_ids:
            if sid == oid:
                continue
            gt_preds = gt_edge_map.get((sid, oid), [])
            if gt_preds:
                dense_gt_relations.extend((sid, oid, pred) for pred in gt_preds)
            else:
                dense_gt_relations.append((sid, oid, none_label))

    predicate_ranks: List[float] = []
    predicate_gt_labels: List[str] = []
    triplet_ranks: List[float] = []
    triplet_gt_labels: List[str] = []

    for gt_sid, gt_oid, gt_pred in dense_gt_relations:
        s_ok = gt_sid in gt_to_pred
        o_ok = gt_oid in gt_to_pred
        predicate_gt_labels.append(gt_pred)
        triplet_gt_labels.append(gt_pred)

        if not (s_ok and o_ok):
            predicate_ranks.append(float("inf"))
            triplet_ranks.append(float("inf"))
            continue

        ps = pred_obj_map[gt_to_pred[gt_sid]]
        po = pred_obj_map[gt_to_pred[gt_oid]]
        explicit_preds = pred_edge_map.get((ps.id, po.id), [])
        preds_for_pair = (
            explicit_preds
            if explicit_preds
            else [none_label] if implicit_none_for_missing_edges
            else []
        )

        if gt_pred == none_label:
            pair_dist = float(np.linalg.norm(gt_obj_map[gt_sid].position - gt_obj_map[gt_oid].position))
            if pair_dist > none_distance_threshold:
                predicate_rank = 1.0
            elif preds_for_pair:
                predicate_rank = float(min(
                    pred_matcher.rank_in_vocab(pp, gt_pred, parity_pred_vocab, parity_pred_vocab_embs)
                    for pp in preds_for_pair
                ))
            else:
                predicate_rank = float("inf")
        elif preds_for_pair:
            predicate_rank = float(min(
                pred_matcher.rank_in_vocab(pp, gt_pred, parity_pred_vocab, parity_pred_vocab_embs)
                for pp in preds_for_pair
            ))
        else:
            predicate_rank = float("inf")
        predicate_ranks.append(predicate_rank)

        gt_subj = gt_obj_map[gt_sid]
        gt_obj = gt_obj_map[gt_oid]
        if not preds_for_pair:
            triplet_ranks.append(float("inf"))
            continue

        sim_s = obj_vocab_embs @ obj_matcher.encode(ps.label)
        sim_o = obj_vocab_embs @ obj_matcher.encode(po.label)
        gt_s_idx = obj_vocab.index(gt_subj.label)
        gt_o_idx = obj_vocab.index(gt_obj.label)
        gt_p_idx = parity_pred_vocab.index(gt_pred)

        best_triplet_rank = float("inf")
        for pred in preds_for_pair:
            sim_p = parity_pred_vocab_embs @ pred_matcher.encode(pred)
            conf = np.einsum("i,j,k->ijk", sim_s, sim_o, sim_p)

            if gt_pred == none_label:
                sorted_conf = np.sort(conf.reshape(-1))[::-1][:rel_topk]
                gt_none_confs = conf[:, :, gt_p_idx].reshape(-1)
                hit_mask = np.isin(sorted_conf, gt_none_confs)
                rank = float(np.argmax(hit_mask) + 1) if np.any(hit_mask) else float(rel_topk + 1)
            else:
                gt_score = conf[gt_s_idx, gt_o_idx, gt_p_idx]
                rank = float(int(np.sum(conf > gt_score)) + 1)

            best_triplet_rank = min(best_triplet_rank, rank)
        triplet_ranks.append(best_triplet_rank)

    return {
        "parity_pred_ranks": np.array(predicate_ranks, dtype=float),
        "parity_pred_gt_labels": predicate_gt_labels,
        "parity_triplet_ranks": np.array(triplet_ranks, dtype=float),
        "parity_triplet_gt_labels": triplet_gt_labels,
    }
