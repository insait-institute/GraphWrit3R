"""Inference script for scene-graph-finetuned SpatialLM.

Reads a PLY point cloud, generates a scene graph (objects + relationships)
in the same JSON format as scene_graph.json.

Usage:
    python inference_scene_graph.py \
        -p pcd/scene0000_00.ply \
        -o scene0000_00_sg.json \
        --model_path saves/scene_graph_checkpoints \
        --code_template_file scene_graph_code_template.txt
"""

import os
import json
import glob
import argparse

import torch
import numpy as np
from tqdm import tqdm
from threading import Thread
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer, set_seed

from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose
from spatiallm.layout.scene_graph_layout import SceneGraphLayout
from spatiallm.relationships_only import (
    aggregate_relationship_diagnostics,
    build_relationships_only_prompt,
    extract_json_value,
    extract_objects_from_prompt,
    load_given_objects_map,
    normalize_relationships,
    renumber_given_objects,
    relationships_summary_path,
)


def _extract_json_blob(text: str):
    """Backward-compatible wrapper for JSON object/list extraction."""
    return extract_json_value(text)


def _extract_objects_json_from_prompt(prompt_text: str):
    """Backward-compatible wrapper for prompt object extraction."""
    return extract_objects_from_prompt(prompt_text)


def _load_given_objects_map(
    dataset_json_path: str,
    include_assistant_targets: bool = False,
):
    """Backward-compatible wrapper for loading immutable scene objects."""
    return load_given_objects_map(
        dataset_json_path,
        include_assistant_targets=include_assistant_targets,
    )


def _load_human_prompt_map(dataset_json_path: str):
    """Load scene_id -> exact human prompt used during training."""
    with open(dataset_json_path, "r") as f:
        samples = json.load(f)

    mapping = {}
    for sample in samples:
        point_clouds = sample.get("point_clouds", [])
        conversations = sample.get("conversations", [])
        if not point_clouds or not conversations:
            continue

        scene_id = os.path.basename(point_clouds[0]).replace(".ply", "")
        human_msg = None
        for msg in conversations:
            if msg.get("from") == "human":
                human_msg = msg.get("value", "")
                break
        if human_msg is None:
            continue

        mapping[scene_id] = human_msg

    return mapping


def preprocess_point_cloud(points, colors, grid_size, num_bins):
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {"name": "pcd", "coord": points.copy(), "color": colors.copy()}
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def make_dummy_chorus_point_cloud(device):
    return torch.zeros((1, 1, 9), dtype=torch.float32, device=device)


def load_scene_ids_arg(scene_ids_arg):
    if not scene_ids_arg:
        return []
    if os.path.isfile(scene_ids_arg):
        with open(scene_ids_arg, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
    return [item.strip() for item in scene_ids_arg.split(",") if item.strip()]


def find_chorus_cache_dir(aligned_root, scene_id):
    if not aligned_root or not scene_id:
        return None
    candidates = [
        os.path.join(aligned_root, scene_id),
        os.path.join(aligned_root, "train", scene_id),
        os.path.join(aligned_root, "val", scene_id),
        os.path.join(aligned_root, "test", scene_id),
    ]
    candidates.extend(glob.glob(os.path.join(aligned_root, "*", scene_id)))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def load_chorus_scene_origin(aligned_root, scene_id):
    cache_dir = find_chorus_cache_dir(aligned_root, scene_id)
    if cache_dir is None:
        return None
    origin_path = os.path.join(cache_dir, "sonata_origin.npy")
    if not os.path.isfile(origin_path):
        return None
    origin = np.load(origin_path).astype(np.float32, copy=False).reshape(-1)
    if origin.shape[0] < 3:
        return None
    return origin[:3]


def scene_id_from_pcd_path(path):
    return os.path.basename(path).replace(".ply", "")


def get_scene_output_path(output_arg, scene_id):
    if os.path.splitext(output_arg)[-1]:
        return output_arg
    return os.path.join(output_arg, f"{scene_id}_sg.json")


def build_scene_jobs(
    point_cloud_arg,
    val_json_path,
    allow_missing_point_cloud=False,
    scene_ids_arg=None,
):
    """Build (scene_id, pcd_path) jobs. pcd_path may be None for Chorus-only modes."""
    pcd_paths_by_scene = {}
    if point_cloud_arg:
        if os.path.isfile(point_cloud_arg):
            pcd_files = [point_cloud_arg]
        else:
            pcd_files = sorted(glob.glob(os.path.join(point_cloud_arg, "*.ply")))
        pcd_paths_by_scene = {scene_id_from_pcd_path(path): path for path in pcd_files}

    explicit_scene_ids = load_scene_ids_arg(scene_ids_arg)
    val_scene_ids = []
    if val_json_path:
        with open(val_json_path, "r") as f:
            val_data = json.load(f)
        seen = set()
        for sample in val_data:
            for pcd_path in sample.get("point_clouds", []):
                scene_id = scene_id_from_pcd_path(pcd_path)
                if scene_id not in seen:
                    val_scene_ids.append(scene_id)
                    seen.add(scene_id)

    if explicit_scene_ids:
        scene_ids = explicit_scene_ids
        print(f"Using {len(scene_ids)} explicitly provided scene ids")
    elif val_scene_ids:
        if pcd_paths_by_scene:
            scene_ids = [scene_id for scene_id in val_scene_ids if scene_id in pcd_paths_by_scene]
            print(
                f"Filtered to {len(scene_ids)} validation scenes "
                f"(from {len(val_scene_ids)} in val JSON)"
            )
        elif allow_missing_point_cloud:
            scene_ids = val_scene_ids
            print(f"Using {len(scene_ids)} validation scenes from val JSON without PCD files")
        else:
            raise ValueError("--point_cloud is required")
    elif pcd_paths_by_scene:
        scene_ids = list(pcd_paths_by_scene.keys())
    else:
        raise ValueError(
            "No inference scenes found. Provide --point_cloud, --scene_ids, or a val JSON "
            "with a Chorus-only checkpoint/config."
        )

    return [(scene_id, pcd_paths_by_scene.get(scene_id)) for scene_id in scene_ids]


def generate_scene_graph(
    model,
    point_cloud,
    tokenizer,
    code_template_file,
    min_extent,
    scene_id=None,
    scene_graph_mode="full",
    given_objects_json=None,
    prompt_override=None,
    top_k=1,
    top_p=0.95,
    temperature=0.0,
    num_beams=1,
    seed=-1,
    max_new_tokens=30000,
    relationship_diagnostics=None,
):
    if seed >= 0:
        set_seed(seed)

    if scene_graph_mode == "relationships_only":
        if given_objects_json is None:
            raise ValueError(
                "Relationship-only inference requires ground-truth objects from --val_json."
            )
        prompt = build_relationships_only_prompt(given_objects_json)
    elif prompt_override is not None:
        prompt = prompt_override.replace(
            "<point_cloud>",
            "<|point_start|><|point_pad|><|point_end|>",
            1,
        )
    else:
        with open(code_template_file, "r") as f:
            code_template = f.read()
        task_prompt = (
            f"Detect objects and their relationships. "
            f"The reference code is as followed: {code_template}"
        )
        prompt = f"<|point_start|><|point_pad|><|point_end|>{task_prompt}"

    if model.config.model_type == "spatiallm_qwen":
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        conversation = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    # Ensure pad_token_id is set
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    model_inputs = {"input_ids": input_ids}
    if point_cloud is not None:
        model_inputs["point_clouds"] = point_cloud
    if scene_id is not None:
        model_inputs["scene_ids"] = [scene_id]

    generate_kwargs = dict(
        model_inputs,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    streamer = TextIteratorStreamer(
        tokenizer, timeout=120.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs["streamer"] = streamer
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    print("Generating scene graph...\n")
    generated = []
    for text in streamer:
        generated.append(text)
        #print(text, end="", flush=True)
    print("\nDone!")
    layout_str = "".join(generated)

    if scene_graph_mode == "relationships_only" or (
        scene_graph_mode == "auto" and given_objects_json is not None
    ):
        if given_objects_json is None:
            return {"raw_output": layout_str}

        parsed = _extract_json_blob(layout_str)
        relationships, diagnostics = normalize_relationships(
            parsed,
            given_objects_json,
        )
        if relationship_diagnostics is not None:
            relationship_diagnostics.update(diagnostics)
        if parsed is None:
            print(
                "Warning: Failed to parse relationship-only output; "
                "saving the immutable objects with an empty relationship list."
            )
        elif diagnostics["malformed_relationships"]:
            print(
                "Warning: Could not serialize "
                f"{diagnostics['malformed_relationships']} structurally malformed "
                "relationship(s)."
            )

        sg_json = {
            "num_objects": int(given_objects_json.get("num_objects", len(given_objects_json.get("objects", [])))),
            "objects": given_objects_json.get("objects", []),
            "relationships": relationships,
        }
        return sg_json

    try:
        sg_layout = SceneGraphLayout(
            layout_str, distance_predicates_are_quantized=True
        )
        sg_layout.undiscretize_and_unnormalize(
            num_bins=model.config.point_config["num_bins"]
        )
        sg_layout.translate(min_extent)
        sg_json = sg_layout.to_json()
        return sg_json
    except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as e:
        print(f"\nWarning: Failed to parse scene graph ({type(e).__name__}: {e}). Saving raw output.")
        return {"raw_output": layout_str}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SpatialLM scene-graph inference")
    parser.add_argument("-p", "--point_cloud", type=str, default=None,
                        help="PLY file or folder of PLY files. Optional for Chorus-only modes.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output JSON file or folder")
    parser.add_argument("-m", "--model_path", type=str,
                        default="saves/scene_graph_checkpoints",
                        help="Path to finetuned model checkpoint")
    parser.add_argument("-t", "--code_template_file", type=str,
                        default="scene_graph_code_template.txt")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--inference_dtype", type=str, default="bfloat16")
    parser.add_argument("--no_cleanup", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=30000)
    parser.add_argument("--val_json", type=str, default=None,
                        help="Path to scene_graph_val.json to restrict inference to validation scenes only")
    parser.add_argument(
        "--scene_ids",
        type=str,
        default=None,
        help="Comma-separated scene ids or a text file with one scene id per line.",
    )
    parser.add_argument("--chorus_fusion", action="store_true",
                        help="Enable Chorus+PCD fusion at inference if the checkpoint config does not already enable it.")
    parser.add_argument("--disable_chorus_fusion", action="store_true",
                        help="Force the checkpoint to run as PCD/Sonata-only, ignoring saved Chorus fusion weights.")
    parser.add_argument("--chorus_repo_root", type=str, default="third_party/chorus")
    parser.add_argument("--chorus_config", type=str, default="chorus_3dgs")
    parser.add_argument("--chorus_checkpoint", type=str, default=None)
    parser.add_argument("--chorus_aligned_root", type=str, default=None)
    parser.add_argument("--chorus_input_mode", type=str, default=None, choices=["prepared", "native"])
    parser.add_argument(
        "--chorus_fusion_mode",
        type=str,
        default=None,
        choices=[
            "avg",
            "append",
            "pcd",
            "chorus",
            "chorus_matched",
            "chorus_coord_matched",
            "transformer",
        ],
        help=(
            "Override fusion token source at inference: avg, append, pcd, "
            "chorus, chorus_matched, chorus_coord_matched, or transformer."
        ),
    )
    parser.add_argument("--chorus_missing_policy", type=str, default="pcd", choices=["pcd", "error"])
    parser.add_argument(
        "--chorus_discard_unmatched_tokens",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Drop unmatched PCD/Sonata tokens when aligned Chorus fusion has matches.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip scenes whose output JSON already exists in the output directory.",
    )
    parser.add_argument(
        "--scene_graph_mode",
        type=str,
        default="full",
        choices=["full", "relationships_only", "auto"],
        help=(
            "Inference mode: 'full' predicts full scene graph, "
            "'relationships_only' uses given objects from --val_json and predicts only relationships, "
            "'auto' uses given objects when available and falls back to full otherwise."
        ),
    )
    parser.add_argument(
        "--relationships_only",
        "--relationships-only",
        action="store_true",
        help=(
            "Ask the model to predict only relationships between immutable "
            "ground-truth objects loaded from --val_json. The saved files remain "
            "full evaluator-compatible scene-graph JSON."
        ),
    )
    args = parser.parse_args()

    if args.relationships_only:
        args.scene_graph_mode = "relationships_only"
    if args.scene_graph_mode == "relationships_only" and not args.val_json:
        parser.error("relationship-only inference requires --val_json")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_3d_tokens = True
    if args.disable_chorus_fusion:
        config.chorus_fusion_enabled = False
    elif args.chorus_fusion or getattr(config, "chorus_fusion_enabled", False):
        config.chorus_fusion_enabled = True
        config.chorus_repo_root = args.chorus_repo_root or getattr(config, "chorus_repo_root", None)
        config.chorus_config = args.chorus_config or getattr(config, "chorus_config", "chorus_3dgs")
        config.chorus_checkpoint = args.chorus_checkpoint or getattr(config, "chorus_checkpoint", None)
        config.chorus_aligned_root = args.chorus_aligned_root or getattr(config, "chorus_aligned_root", None)
        config.chorus_input_mode = args.chorus_input_mode or getattr(config, "chorus_input_mode", "prepared")
        config.chorus_fusion_mode = args.chorus_fusion_mode or getattr(config, "chorus_fusion_mode", "avg")
        config.chorus_missing_policy = args.chorus_missing_policy or getattr(config, "chorus_missing_policy", "pcd")
        config.chorus_discard_unmatched_tokens = (
            args.chorus_discard_unmatched_tokens
            if args.chorus_discard_unmatched_tokens is not None
            else getattr(config, "chorus_discard_unmatched_tokens", False)
        )
        config.chorus_modality_dropout_rate = 0.0
        if config.chorus_checkpoint is None or config.chorus_aligned_root is None:
            raise ValueError(
                "Chorus fusion inference requires `chorus_checkpoint` and `chorus_aligned_root` "
                "either in the checkpoint config or as CLI arguments."
            )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=getattr(torch, args.inference_dtype),
        trust_remote_code=True,
    )
    model.to("cuda")
    if hasattr(model, "set_point_backbone_dtype"):
        model.set_point_backbone_dtype(torch.float32)
    model.eval()

    num_bins = model.config.point_config["num_bins"]

    # Filter to validation set if --val_json is provided
    given_objects_map = {}
    object_id_mappings = {}
    prompt_map = {}
    if args.val_json:
        if args.scene_graph_mode in ("relationships_only", "auto"):
            given_objects_map = _load_given_objects_map(
                args.val_json,
                include_assistant_targets=(
                    args.scene_graph_mode == "relationships_only"
                ),
            )
            if args.scene_graph_mode == "relationships_only":
                renumbered_objects_map = {}
                for scene_id, objects_payload in given_objects_map.items():
                    (
                        renumbered_objects_map[scene_id],
                        object_id_mappings[scene_id],
                    ) = renumber_given_objects(objects_payload)
                given_objects_map = renumbered_objects_map
            print(f"Loaded given objects for {len(given_objects_map)} scenes from val JSON")
        if args.scene_graph_mode != "relationships_only":
            prompt_map = _load_human_prompt_map(args.val_json)
            print(f"Loaded training prompts for {len(prompt_map)} scenes from val JSON")

    chorus_only_3d = bool(
        getattr(config, "chorus_fusion_enabled", False)
        and getattr(config, "chorus_fusion_mode", None) in {"chorus", "chorus_matched"}
    )
    allow_missing_point_cloud = chorus_only_3d
    jobs = build_scene_jobs(
        point_cloud_arg=args.point_cloud,
        val_json_path=args.val_json,
        allow_missing_point_cloud=allow_missing_point_cloud,
        scene_ids_arg=args.scene_ids,
    )
    print(f"Running inference for {len(jobs)} scenes")
    if args.scene_graph_mode == "relationships_only":
        missing_object_scenes = [
            scene_id for scene_id, _ in jobs if scene_id not in given_objects_map
        ]
        if missing_object_scenes:
            preview = ", ".join(missing_object_scenes[:5])
            suffix = "..." if len(missing_object_scenes) > 5 else ""
            raise ValueError(
                "Could not load ground-truth objects for "
                f"{len(missing_object_scenes)} inference scene(s): {preview}{suffix}"
            )

    skipped_existing = []
    relationships_only_scene_stats = {}
    for scene_id, pcd_file in tqdm(jobs):
        output_path = get_scene_output_path(args.output, scene_id)
        if args.skip_existing and os.path.isfile(output_path):
            print(f"Skipping {scene_id}: existing scene graph found at {output_path}")
            skipped_existing.append(scene_id)
            continue

        input_pcd = None
        min_extent = np.zeros(3, dtype=np.float32)
        if pcd_file is None:
            if not allow_missing_point_cloud:
                raise ValueError(f"Missing point cloud path for scene {scene_id}")
            if getattr(config, "chorus_missing_policy", "pcd") != "error":
                raise ValueError(
                    "Chorus-only inference without --point_cloud requires "
                    "--chorus_missing_policy error to avoid falling back to a dummy PCD."
                )
            origin = load_chorus_scene_origin(config.chorus_aligned_root, scene_id)
            if origin is None:
                print(
                    f"Warning: missing sonata_origin.npy for {scene_id}; "
                    "using zero output translation."
                )
            else:
                min_extent = origin
            input_pcd = make_dummy_chorus_point_cloud(model.device)
        else:
            pcd = load_o3d_pcd(pcd_file)
            grid_size = SceneGraphLayout.get_grid_size(num_bins)

            if not args.no_cleanup:
                pcd = cleanup_pcd(pcd, voxel_size=grid_size)

            points, colors = get_points_and_colors(pcd)
            min_extent = np.min(points, axis=0)

            input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)
            input_pcd = input_pcd.to("cuda")

        given_objects_json = None
        prompt_override = (
            None
            if args.scene_graph_mode == "relationships_only"
            else prompt_map.get(scene_id)
        )
        if (
            args.val_json
            and args.scene_graph_mode != "relationships_only"
            and prompt_override is None
        ):
            print(f"Warning: missing training prompt for {scene_id}; using fallback inference prompt.")

        if args.scene_graph_mode in ("relationships_only", "auto"):
            given_objects_json = given_objects_map.get(scene_id)

        scene_relationship_diagnostics = {}
        sg_json = generate_scene_graph(
            model, input_pcd, tokenizer, args.code_template_file,
            min_extent=min_extent,
            scene_id=scene_id,
            scene_graph_mode=args.scene_graph_mode,
            given_objects_json=given_objects_json,
            prompt_override=prompt_override,
            top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
            num_beams=args.num_beams, seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            relationship_diagnostics=scene_relationship_diagnostics,
        )
        if args.scene_graph_mode == "relationships_only":
            scene_relationship_diagnostics["object_id_mapping"] = (
                object_id_mappings[scene_id]
            )
            relationships_only_scene_stats[scene_id] = (
                scene_relationship_diagnostics
            )

        if os.path.splitext(args.output)[-1]:
            with open(output_path, "w") as f:
                json.dump(sg_json, f, indent=2)
        else:
            os.makedirs(args.output, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(sg_json, f, indent=2)

    if skipped_existing:
        print(
            "Skipped "
            f"{len(skipped_existing)} scene(s) with existing scene graphs: "
            + ", ".join(skipped_existing)
        )
    if args.scene_graph_mode == "relationships_only":
        summary = aggregate_relationship_diagnostics(
            args.output,
            relationships_only_scene_stats,
            skipped_existing,
        )
        totals = summary["totals"]
        summary_path = relationships_summary_path(args.output)
        summary_parent = os.path.dirname(summary_path)
        if summary_parent:
            os.makedirs(summary_parent, exist_ok=True)
        with open(summary_path, "w") as file:
            json.dump(summary, file, indent=2)
        print(
            "Relationship-only endpoint summary: "
            f"{totals['unknown_endpoint_relationships']}/"
            f"{totals['saved_relationships']} saved relationship(s) reference "
            f"at least one unknown object ID. Details: {summary_path}"
        )
    print("All done.")
