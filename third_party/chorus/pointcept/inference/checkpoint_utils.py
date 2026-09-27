"""Checkpoint resolution helpers for standalone inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from huggingface_hub import hf_hub_download


def _candidate_cache_roots() -> list[Path]:
    roots = []
    for env_name in ("CHORUS_CHECKPOINT_DIR", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return list(dict.fromkeys(roots))


def _find_cached_hf_checkpoint(filename: str, repo_id: str) -> Optional[Path]:
    repo_cache_name = "models--" + repo_id.replace("/", "--")
    for root in _candidate_cache_roots():
        for candidate in (
            root / filename,
            root / repo_cache_name / "snapshots",
        ):
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                matches = sorted(candidate.glob(f"*/{filename}"))
                if matches:
                    return matches[-1]
    return None


def resolve_checkpoint_reference(
    checkpoint_ref: str,
    hub_cfg: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    if not checkpoint_ref:
        raise ValueError("checkpoint_ref must be a non-empty path or checkpoint name")

    checkpoint_path = Path(checkpoint_ref).expanduser()
    if checkpoint_path.is_file():
        resolved = dict(
            source="local",
            requested=checkpoint_ref,
            resolved_name=checkpoint_path.name,
            local_path=str(checkpoint_path.resolve()),
        )
        if logger is not None:
            logger.info("Using local checkpoint: %s", resolved["local_path"])
        return resolved

    if hub_cfg is None:
        raise FileNotFoundError(
            f"Checkpoint not found locally and no Hugging Face repo configured: {checkpoint_ref}"
        )

    repo_id = hub_cfg.get("repo_id")
    if not repo_id:
        raise ValueError("checkpoint_hub.repo_id must be set to resolve non-local checkpoints")

    revision = hub_cfg.get("revision", "main")
    filename = checkpoint_ref
    if "/" not in checkpoint_ref and not filename.endswith((".pth", ".pt", ".ckpt")):
        filename = f"{checkpoint_ref}.pth"

    cached_path = _find_cached_hf_checkpoint(filename, repo_id)
    if cached_path is not None:
        resolved = dict(
            source="huggingface-cache",
            requested=checkpoint_ref,
            resolved_name=filename,
            local_path=str(cached_path.resolve()),
            repo_id=repo_id,
            revision=revision,
        )
        if logger is not None:
            logger.info("Using cached Hugging Face checkpoint: %s", resolved["local_path"])
        return resolved

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type=hub_cfg.get("repo_type", "model"),
    )
    resolved = dict(
        source="huggingface",
        requested=checkpoint_ref,
        resolved_name=filename,
        local_path=local_path,
        repo_id=repo_id,
        revision=revision,
    )
    if logger is not None:
        logger.info(
            "Resolved Hugging Face checkpoint %s from %s@%s",
            filename,
            repo_id,
            revision,
        )
    return resolved
