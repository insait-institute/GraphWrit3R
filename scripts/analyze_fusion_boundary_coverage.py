#!/usr/bin/env python3
"""Summarize PCD/3DGS voxel coverage and actual fusion-boundary discarding.

The raw-bridge summaries distinguish three quantities that are easy to conflate:

1. exact same-cell overlap (diagnostic only);
2. support within the configured matching radius; and
3. voxels actually discarded after applying the missing-voxel policy.

Optionally, a trainer result/config JSON can be supplied to report the encoded-token
coverage and whether unmatched PCD tokens were retained by the evaluated model.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge-root",
        type=Path,
        required=True,
        help="Root containing <split>/<scene>/summary.json raw-bridge outputs.",
    )
    parser.add_argument(
        "--trainer-results",
        type=Path,
        help="Optional eval_results.json or all_results.json with chorus usage metrics.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Optional model config.json recording the unmatched-token policy.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def summarize_scenes(records: list[dict[str, Any]]) -> dict[str, Any]:
    sonata = sum(int(record["summary"]["sonata_voxels"]) for record in records)
    output = sum(int(record["summary"]["output_voxels"]) for record in records)
    missing = sum(int(record["summary"]["missing_voxels"]) for record in records)
    dropped = sum(int(record["summary"]["dropped_missing_voxels"]) for record in records)
    filled = sum(int(record["summary"]["filled_missing_voxels"]) for record in records)
    supported = sonata - missing
    voxel_sizes = {float(record["summary"]["voxel_size"]) for record in records}
    voxel_size = next(iter(voxel_sizes)) if len(voxel_sizes) == 1 else None
    voxel_volume = voxel_size**3 if voxel_size is not None else None

    missing_fractions = [
        int(record["summary"]["missing_voxels"])
        / max(int(record["summary"]["sonata_voxels"]), 1)
        for record in records
    ]
    exact_reference = sum(
        int(record["exact_overlap"]["reference_unique"])
        for record in records
        if record.get("exact_overlap")
    )
    exact_overlap = sum(
        int(record["exact_overlap"]["overlap_unique"])
        for record in records
        if record.get("exact_overlap")
    )
    exact_coverages = [
        float(record["exact_overlap"]["coverage"])
        for record in records
        if record.get("exact_overlap")
    ]

    return {
        "scenes": len(records),
        "voxel_size_m": voxel_size,
        "sonata_input_voxels": sonata,
        "supported_within_match_radius_voxels": supported,
        "unsupported_within_match_radius_voxels": missing,
        "nearest_filled_voxels": filled,
        "output_voxels": output,
        "actually_discarded_voxels": dropped,
        "support_coverage_micro": supported / max(sonata, 1),
        "unsupported_fraction_micro": missing / max(sonata, 1),
        "unsupported_fraction_per_scene": distribution(missing_fractions),
        "actual_discard_fraction_micro": dropped / max(sonata, 1),
        "exact_same_cell_coverage_micro_diagnostic": (
            exact_overlap / exact_reference if exact_reference else None
        ),
        "exact_same_cell_coverage_per_scene_diagnostic": distribution(exact_coverages),
        "actual_discarded_occupied_voxel_volume_m3": (
            dropped * voxel_volume if voxel_volume is not None else None
        ),
        "unsupported_occupied_voxel_volume_m3_if_not_filled": (
            missing * voxel_volume if voxel_volume is not None else None
        ),
        "volume_note": (
            "Volumes count occupied voxels times voxel_size^3; they are not a fraction "
            "of the scenes' axis-aligned bounding-box volume."
        ),
    }


def load_bridge_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        summary = read_json(summary_path)
        diagnostics_path = summary_path.with_name("match_diagnostics.json")
        exact_overlap = None
        if diagnostics_path.exists():
            diagnostics = read_json(diagnostics_path)
            exact_overlap = diagnostics.get("chorus", {}).get("raw_bridge_overlap")
        records.append(
            {
                "split": summary_path.parent.parent.name,
                "scene_id": summary_path.parent.name,
                "summary": summary,
                "exact_overlap": exact_overlap,
            }
        )
    if not records:
        raise FileNotFoundError(f"No <split>/<scene>/summary.json files under {root}")
    return records


def trainer_metrics(path: Path) -> dict[str, Any]:
    data = read_json(path)
    prefixes = ("eval_avg_", "eval_")

    def first(suffix: str) -> Any:
        for prefix in prefixes:
            key = prefix + suffix
            if key in data:
                return data[key]
        return None

    sonata = first("chorus_usage_sonata_tokens_avg")
    returned = first("chorus_usage_returned_tokens_avg")
    matched = first("chorus_usage_matched_tokens_avg")
    coverage = first("chorus_usage_matched_to_sonata_frac")
    return {
        "source": str(path),
        "matched_encoded_tokens_avg": matched,
        "sonata_encoded_tokens_avg": sonata,
        "returned_encoded_tokens_avg": returned,
        "matched_to_sonata_fraction": coverage,
        "unmatched_sonata_fraction": 1.0 - float(coverage) if coverage is not None else None,
        "returned_to_sonata_fraction": (
            float(returned) / float(sonata) if sonata not in (None, 0) and returned is not None else None
        ),
        "no_match_scene_count": first("chorus_usage_pcd_no_match_count"),
    }


def model_policy(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return {
        "source": str(path),
        "fusion_mode": data.get("chorus_fusion_mode"),
        "missing_policy": data.get("chorus_missing_policy"),
        "discard_unmatched_tokens": data.get("chorus_discard_unmatched_tokens"),
        "match_grid_radius": data.get("chorus_match_grid_radius"),
    }


def main() -> None:
    args = parse_args()
    records = load_bridge_records(args.bridge_root)
    split_names = sorted({record["split"] for record in records})
    result: dict[str, Any] = {
        "bridge_root": str(args.bridge_root),
        "all": summarize_scenes(records),
        "splits": {
            split: summarize_scenes(
                [record for record in records if record["split"] == split]
            )
            for split in split_names
        },
    }
    if args.trainer_results:
        result["encoded_token_evaluation"] = trainer_metrics(args.trainer_results)
    if args.model_config:
        result["model_policy"] = model_policy(args.model_config)

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
