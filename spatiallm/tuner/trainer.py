import copy
import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.distributed as dist
from transformers import Seq2SeqTrainer
from typing_extensions import override

from spatiallm.model import mm_debug
from spatiallm.tuner.framework import logging
from spatiallm.tuner.framework.utils import is_transformers_version_greater_than
from spatiallm.tuner.framework.callbacks import (
    LogCallback,
    ReporterCallback,
    get_swanlab_callback,
)
from spatiallm.tuner.framework.loader import load_tokenizer, load_model
from spatiallm.tuner.hparams import get_train_args, read_args
from spatiallm.tuner.data import (
    IGNORE_INDEX,
    get_dataset,
    get_template_and_fix_tokenizer,
    register_spatiallm_templates,
    SFTDataCollatorWith4DAttentionMask,
)


if TYPE_CHECKING:
    from transformers import (
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    from spatiallm.tuner.hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
    )

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        eval_chorus_fusion_modes: Optional[list[str]] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)

        self.finetuning_args = finetuning_args
        self.eval_chorus_fusion_modes = self._normalize_chorus_fusion_modes(
            eval_chorus_fusion_modes
        )
        self._logged_optimizer_trainability = False
        self._reset_train_component_window()
        self._reset_eval_component_window()
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

    @staticmethod
    def _normalize_chorus_fusion_modes(modes: Any) -> list[str]:
        if not modes:
            return []
        if isinstance(modes, str):
            modes = [item.strip() for item in modes.split(",")]

        normalized_modes = []
        for mode in modes:
            if mode is None:
                continue
            mode = str(mode).strip().lower()
            if not mode:
                continue
            if mode == "3dgs":
                mode = "chorus"
            if mode not in {
                "avg",
                "append",
                "pcd",
                "chorus",
                "chorus_matched",
                "chorus_coord_matched",
                "transformer",
            }:
                raise ValueError(
                    "`eval_chorus_fusion_modes` must contain only avg, append, pcd, "
                    "chorus, chorus_matched, chorus_coord_matched, transformer, or 3dgs."
                )
            if mode not in normalized_modes:
                normalized_modes.append(mode)
        return normalized_modes

    def _unwrap_model(self, model: Optional["torch.nn.Module"] = None) -> "torch.nn.Module":
        model = self.model if model is None else model
        return (
            self.accelerator.unwrap_model(model)
            if hasattr(self, "accelerator") and self.accelerator is not None
            else model
        )

    def _current_chorus_fusion_mode(self) -> Optional[str]:
        model = self._unwrap_model()
        chorus_fusion = getattr(model, "chorus_fusion_encoder", None)
        if chorus_fusion is not None:
            return getattr(chorus_fusion, "fusion_mode", None)
        config = getattr(model, "config", None)
        return getattr(config, "chorus_fusion_mode", None)

    @contextmanager
    def _temporary_chorus_fusion_mode(self, mode: str):
        model = self._unwrap_model()
        config = getattr(model, "config", None)
        chorus_fusion = getattr(model, "chorus_fusion_encoder", None)
        old_config_mode = getattr(config, "chorus_fusion_mode", None)
        old_fusion_mode = getattr(chorus_fusion, "fusion_mode", None)

        if config is not None:
            config.chorus_fusion_mode = mode
        if chorus_fusion is not None:
            chorus_fusion.fusion_mode = mode

        try:
            yield
        finally:
            if config is not None:
                config.chorus_fusion_mode = old_config_mode
            if chorus_fusion is not None:
                chorus_fusion.fusion_mode = old_fusion_mode

    @staticmethod
    def _alias_metric_prefix(
        metrics: dict[str, float],
        source_prefix: str,
        target_prefix: str,
    ) -> dict[str, float]:
        aliases = {}
        for key, value in metrics.items():
            if key == source_prefix or key.startswith(f"{source_prefix}_"):
                aliases[f"{target_prefix}{key[len(source_prefix):]}"] = value
        return aliases

    def _log_optimizer_trainability_once(self, model: "torch.nn.Module") -> None:
        if self._logged_optimizer_trainability or self.optimizer is None:
            return
        self._logged_optimizer_trainability = True

        optimizer_param_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group.get("params", [])
        }
        unwrapped_model = (
            self.accelerator.unwrap_model(model)
            if hasattr(self, "accelerator") and self.accelerator is not None
            else model
        )
        buckets = {
            "sonata.point_backbone": "point_backbone.",
            "chorus.backbone": "chorus_fusion_encoder.chorus_model.backbone.",
            "chorus.fourier_head": "chorus_fusion_encoder.fourier_head.",
            "point_proj": "point_proj.",
            "language_model": "model.",
            "lm_head": "lm_head.",
        }
        lines = []
        for label, prefix in buckets.items():
            trainable = 0
            in_optimizer = 0
            missing = 0
            tensors = 0
            for name, param in unwrapped_model.named_parameters():
                if not name.startswith(prefix) or not param.requires_grad:
                    continue
                tensors += 1
                trainable += param.numel()
                if id(param) in optimizer_param_ids:
                    in_optimizer += param.numel()
                else:
                    missing += param.numel()
            lines.append(
                f"{label}: tensors={tensors} optimizer_params={in_optimizer:,}/{trainable:,} missing={missing:,}"
            )
        logger.info_rank0("Optimizer trainability coverage:\n%s", "\n".join(lines))

    @override
    def create_optimizer(self):
        optimizer = super().create_optimizer()
        self._log_optimizer_trainability_once(self.model)
        return optimizer

    @staticmethod
    def _clone_data_collator_for_split(data_collator: Any, split: str) -> Any:
        cloned_collator = copy.copy(data_collator)
        template = getattr(cloned_collator, "template", None)
        if template is None:
            return cloned_collator

        cloned_template = copy.copy(template)
        plugin = getattr(cloned_template, "mm_plugin", None)
        if plugin is not None:
            cloned_plugin = copy.copy(plugin)
            if hasattr(cloned_plugin, "set_runtime_split"):
                cloned_plugin.set_runtime_split(split)
            cloned_template.mm_plugin = cloned_plugin
        cloned_collator.template = cloned_template
        return cloned_collator

    @contextmanager
    def _data_collator_runtime_split(self, split: str):
        old_collator = self.data_collator
        self.data_collator = self._clone_data_collator_for_split(old_collator, split)
        try:
            yield
        finally:
            self.data_collator = old_collator

    @override
    def get_train_dataloader(self):
        with self._data_collator_runtime_split("train"):
            return super().get_train_dataloader()

    @override
    def get_eval_dataloader(self, eval_dataset=None):
        with self._data_collator_runtime_split("eval"):
            return super().get_eval_dataloader(eval_dataset)

    @override
    def get_test_dataloader(self, test_dataset):
        with self._data_collator_runtime_split("eval"):
            return super().get_test_dataloader(test_dataset)

    def _reset_train_component_window(self) -> None:
        self._train_window_steps = 0
        self._train_next_token_loss_sum = 0.0
        self._train_chorus_contrastive_loss_sum = 0.0
        self._train_chorus_contrastive_steps = 0
        self._train_chorus_usage_counts = {
            "avg": 0,
            "append": 0,
            "chorus": 0,
            "chorus_matched": 0,
            "chorus_coord_matched": 0,
            "pcd": 0,
            "missing": 0,
        }
        self._train_chorus_pcd_reason_counts = {
            "missing_scene_id": 0,
            "missing_chorus": 0,
            "no_match": 0,
            "drop_chorus": 0,
            "forced_pcd": 0,
        }
        self._train_chorus_usage_samples = 0
        self._train_chorus_match_stats = {
            "matched_tokens": 0,
            "returned_tokens": 0,
            "used_chorus_tokens": 0,
            "sonata_tokens": 0,
            "chorus_tokens": 0,
            "chorus_gate_weighted_sum": 0.0,
            "chorus_gate_count": 0,
            "sidecar_reports": 0,
            "sidecar_enabled_reports": 0,
            "sidecar_unmapped_chorus_tokens": 0,
            "coord_match_ratio_sum": 0.0,
            "coord_coverage_ratio_sum": 0.0,
            "coord_match_reports": 0,
        }

    def _reset_eval_component_window(self) -> None:
        self._eval_steps = 0
        self._eval_next_token_loss_sum = 0.0
        self._eval_chorus_contrastive_loss_sum = 0.0
        self._eval_chorus_contrastive_steps = 0
        self._eval_chorus_usage_counts = {
            "avg": 0,
            "append": 0,
            "chorus": 0,
            "chorus_matched": 0,
            "chorus_coord_matched": 0,
            "pcd": 0,
            "missing": 0,
        }
        self._eval_chorus_pcd_reason_counts = {
            "missing_scene_id": 0,
            "missing_chorus": 0,
            "no_match": 0,
            "drop_chorus": 0,
            "forced_pcd": 0,
        }
        self._eval_chorus_usage_samples = 0
        self._eval_chorus_match_stats = {
            "matched_tokens": 0,
            "returned_tokens": 0,
            "used_chorus_tokens": 0,
            "sonata_tokens": 0,
            "chorus_tokens": 0,
            "chorus_gate_weighted_sum": 0.0,
            "chorus_gate_count": 0,
            "sidecar_reports": 0,
            "sidecar_enabled_reports": 0,
            "sidecar_unmapped_chorus_tokens": 0,
            "coord_match_ratio_sum": 0.0,
            "coord_coverage_ratio_sum": 0.0,
            "coord_match_reports": 0,
        }

    def _maybe_log_mm_grads(self, model: "torch.nn.Module") -> None:
        if not (mm_debug.enabled() and mm_debug.grads_enabled()):
            return

        unwrapped_model = (
            self.accelerator.unwrap_model(model)
            if hasattr(self, "accelerator") and self.accelerator is not None
            else model
        )
        debug_state = getattr(unwrapped_model, "_last_mm_debug", None)
        if not isinstance(debug_state, dict):
            debug_state = {}

        point_summary = mm_debug.grad_summary(getattr(unwrapped_model, "point_backbone", None))
        chorus_fusion = getattr(unwrapped_model, "chorus_fusion_encoder", None)
        chorus_backbone = getattr(getattr(chorus_fusion, "chorus_model", None), "backbone", None)
        chorus_summary = mm_debug.grad_summary(chorus_backbone)
        chorus_head_summary = mm_debug.grad_summary(getattr(chorus_fusion, "fourier_head", None))

        warnings = []
        if point_summary["trainable_params"] > 0 and point_summary["grad_elements"] == 0:
            warnings.append("point_backbone has trainable params but no grads")
        if chorus_summary["trainable_params"] > 0 and chorus_summary["grad_elements"] == 0:
            warnings.append("chorus_backbone has trainable params but no grads")
        if chorus_head_summary["trainable_params"] > 0 and chorus_head_summary["grad_elements"] == 0:
            warnings.append("chorus_fourier_head has trainable params but no grads")

        mm_debug.log(
            "grads",
            (
                f"used_3d={debug_state.get('used_3d')} "
                f"samples={debug_state.get('samples')} "
                f"point_backbone_grad=({mm_debug.format_grad_summary(point_summary)}) "
                f"chorus_backbone_grad=({mm_debug.format_grad_summary(chorus_summary)}) "
                f"chorus_fourier_head_grad=({mm_debug.format_grad_summary(chorus_head_summary)}) "
                f"warnings={warnings or 'none'}"
            ),
        )

    @staticmethod
    def _tensor_to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _get_chorus_contrastive_loss_weight(self) -> float:
        model = (
            self.accelerator.unwrap_model(self.model)
            if hasattr(self, "accelerator") and self.accelerator is not None
            else self.model
        )
        config = getattr(model, "config", None)
        loss_weight = self._tensor_to_float(
            getattr(
                config,
                "chorus_contrastive_loss_current_weight",
                getattr(config, "chorus_contrastive_loss_weight", None),
            )
        )
        return float(loss_weight) if loss_weight is not None else 0.0

    def _scheduled_chorus_contrastive_loss_weight(self) -> float:
        model = self._unwrap_model()
        config = getattr(model, "config", None)
        if config is None:
            return 0.0

        peak = self._tensor_to_float(getattr(config, "chorus_contrastive_loss_weight", None))
        if peak is None:
            return 0.0
        peak = max(float(peak), 0.0)

        final = self._tensor_to_float(
            getattr(config, "chorus_contrastive_loss_final_weight", None)
        )
        if final is None:
            return peak
        final = max(float(final), 0.0)

        max_steps = int(getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0) or 0)
        if max_steps <= 0:
            return peak

        step = max(int(getattr(self.state, "global_step", 0) or 0), 0)
        progress = min(max(step / max_steps, 0.0), 1.0)
        warmup = min(max(float(getattr(config, "chorus_contrastive_warmup_ratio", 0.0)), 0.0), 1.0)
        decay_start = min(
            max(float(getattr(config, "chorus_contrastive_decay_start_ratio", warmup)), warmup),
            1.0,
        )

        if warmup > 0.0 and progress < warmup:
            return peak * progress / warmup
        if progress < decay_start:
            return peak
        if decay_start >= 1.0:
            return peak

        decay_progress = (progress - decay_start) / (1.0 - decay_start)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(decay_progress, 0.0), 1.0)))
        return final + (peak - final) * cosine

    def _update_chorus_contrastive_loss_weight(self) -> float:
        weight = self._scheduled_chorus_contrastive_loss_weight()
        model = self._unwrap_model()
        config = getattr(model, "config", None)
        if config is not None:
            config.chorus_contrastive_loss_current_weight = float(weight)
        return weight

    def _reduce_sum_count(self, value_sum: float, count: int) -> tuple[float, int]:
        if not dist.is_available() or not dist.is_initialized():
            return value_sum, count

        device = (
            self.accelerator.device
            if hasattr(self, "accelerator") and self.accelerator is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        packed = torch.tensor([value_sum, float(count)], device=device, dtype=torch.float64)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        return float(packed[0].item()), int(round(float(packed[1].item())))

    def _record_chorus_usage(self, usage_items: Any, split: str) -> None:
        if not usage_items:
            return
        if isinstance(usage_items, dict):
            usage_items = [usage_items]

        counts = (
            self._train_chorus_usage_counts
            if split == "train"
            else self._eval_chorus_usage_counts
        )
        reason_counts = (
            self._train_chorus_pcd_reason_counts
            if split == "train"
            else self._eval_chorus_pcd_reason_counts
        )
        match_stats = (
            self._train_chorus_match_stats
            if split == "train"
            else self._eval_chorus_match_stats
        )
        for usage in usage_items:
            if not isinstance(usage, dict):
                continue
            mode = usage.get("mode", "missing")
            if mode not in counts:
                mode = "missing"
            counts[mode] += 1
            if mode == "pcd":
                reason = usage.get("reason", "forced_pcd")
                if reason not in reason_counts:
                    reason = "forced_pcd"
                reason_counts[reason] += 1
            match_stats["matched_tokens"] += int(usage.get("matched_tokens", 0) or 0)
            match_stats["returned_tokens"] += int(
                usage.get("returned_tokens", usage.get("sonata_tokens", 0)) or 0
            )
            match_stats["used_chorus_tokens"] += int(
                usage.get("used_chorus_tokens", usage.get("matched_tokens", 0)) or 0
            )
            match_stats["sonata_tokens"] += int(usage.get("sonata_tokens", 0) or 0)
            match_stats["chorus_tokens"] += int(usage.get("chorus_tokens", 0) or 0)
            gate_count = int(usage.get("chorus_gate_count", 0) or 0)
            gate_avg = float(usage.get("chorus_gate_avg", 0.0) or 0.0)
            match_stats["chorus_gate_weighted_sum"] += gate_avg * gate_count
            match_stats["chorus_gate_count"] += gate_count
            sidecar = usage.get("sidecar_match")
            if isinstance(sidecar, dict):
                match_stats["sidecar_reports"] += 1
                if sidecar.get("enabled", False):
                    match_stats["sidecar_enabled_reports"] += 1
                match_stats["sidecar_unmapped_chorus_tokens"] += int(
                    sidecar.get("unmapped_chorus_tokens", 0) or 0
                )
            if "coord_match_ratio" in usage or "coord_coverage_ratio" in usage:
                match_stats["coord_match_reports"] += 1
                match_stats["coord_match_ratio_sum"] += float(
                    usage.get("coord_match_ratio", 0.0) or 0.0
                )
                match_stats["coord_coverage_ratio_sum"] += float(
                    usage.get("coord_coverage_ratio", 0.0) or 0.0
                )
            if split == "train":
                self._train_chorus_usage_samples += 1
            else:
                self._eval_chorus_usage_samples += 1

    def _reduced_counter(self, counts: dict[str, int]) -> dict[str, int]:
        reduced_counts = {}
        for key, value in counts.items():
            reduced_value, _ = self._reduce_sum_count(float(value), int(value))
            reduced_counts[key] = int(round(reduced_value))
        return reduced_counts

    def _reduced_chorus_usage(self, split: str) -> tuple[dict[str, int], dict[str, int], int]:
        counts = (
            self._train_chorus_usage_counts
            if split == "train"
            else self._eval_chorus_usage_counts
        )
        reason_counts = (
            self._train_chorus_pcd_reason_counts
            if split == "train"
            else self._eval_chorus_pcd_reason_counts
        )
        total = (
            self._train_chorus_usage_samples
            if split == "train"
            else self._eval_chorus_usage_samples
        )
        reduced_counts = self._reduced_counter(counts)
        reduced_reason_counts = self._reduced_counter(reason_counts)
        reduced_total, _ = self._reduce_sum_count(float(total), int(total))
        return reduced_counts, reduced_reason_counts, int(round(reduced_total))

    def _reduced_chorus_match_stats(self, split: str) -> dict[str, float]:
        stats = (
            self._train_chorus_match_stats
            if split == "train"
            else self._eval_chorus_match_stats
        )
        int_keys = {
            "matched_tokens",
            "returned_tokens",
            "used_chorus_tokens",
            "sonata_tokens",
            "chorus_tokens",
            "chorus_gate_count",
            "sidecar_reports",
            "sidecar_enabled_reports",
            "sidecar_unmapped_chorus_tokens",
            "coord_match_reports",
        }
        reduced_stats = {}
        for key, value in stats.items():
            reduced_value, _ = self._reduce_sum_count(float(value), int(round(float(value))))
            if key in int_keys:
                reduced_stats[key] = int(round(reduced_value))
            else:
                reduced_stats[key] = float(reduced_value)
        return reduced_stats

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        # Add component losses to training logs so report_to=wandb gets separate curves.
        if "loss" in logs:
            logs = dict(logs)
            reduced_steps_sum, reduced_steps = self._reduce_sum_count(
                float(self._train_window_steps),
                self._train_window_steps,
            )
            if reduced_steps > 0 and reduced_steps_sum > 0:
                next_sum, _ = self._reduce_sum_count(
                    self._train_next_token_loss_sum,
                    self._train_window_steps,
                )
                chorus_sum, chorus_steps = self._reduce_sum_count(
                    self._train_chorus_contrastive_loss_sum,
                    self._train_chorus_contrastive_steps,
                )
                next_avg = next_sum / reduced_steps_sum
                chorus_weight = self._get_chorus_contrastive_loss_weight()

                logs["next_token_loss"] = next_avg
                reconstructed_total_loss = next_avg
                chorus_avg = chorus_sum / chorus_steps if chorus_steps > 0 else 0.0
                logs["chorus_contrastive_loss"] = chorus_avg
                logs["chorus_contrastive_loss_weight"] = chorus_weight
                logs["chorus_contrastive_loss_weighted"] = chorus_avg * chorus_weight
                logs["chorus_contrastive_steps"] = chorus_steps
                if chorus_steps > 0:
                    reconstructed_total_loss += chorus_avg * chorus_weight
                usage_counts, reason_counts, usage_total = self._reduced_chorus_usage("train")
                if usage_total > 0:
                    logs["chorus_usage/avg_count"] = usage_counts["avg"]
                    logs["chorus_usage/append_count"] = usage_counts["append"]
                    logs["chorus_usage/chorus_count"] = usage_counts["chorus"]
                    logs["chorus_usage/chorus_matched_count"] = usage_counts["chorus_matched"]
                    logs["chorus_usage/chorus_coord_matched_count"] = (
                        usage_counts["chorus_coord_matched"]
                    )
                    logs["chorus_usage/pcd_count"] = usage_counts["pcd"]
                    logs["chorus_usage/pcd_missing_scene_id_count"] = reason_counts["missing_scene_id"]
                    logs["chorus_usage/pcd_missing_chorus_count"] = reason_counts["missing_chorus"]
                    logs["chorus_usage/pcd_no_match_count"] = reason_counts["no_match"]
                    logs["chorus_usage/pcd_drop_chorus_count"] = reason_counts["drop_chorus"]
                    logs["chorus_usage/pcd_forced_count"] = reason_counts["forced_pcd"]
                    logs["chorus_usage/avg_frac"] = usage_counts["avg"] / usage_total
                    logs["chorus_usage/append_frac"] = usage_counts["append"] / usage_total
                    logs["chorus_usage/chorus_frac"] = usage_counts["chorus"] / usage_total
                    logs["chorus_usage/chorus_matched_frac"] = (
                        usage_counts["chorus_matched"] / usage_total
                    )
                    logs["chorus_usage/chorus_coord_matched_frac"] = (
                        usage_counts["chorus_coord_matched"] / usage_total
                    )
                    logs["chorus_usage/pcd_frac"] = usage_counts["pcd"] / usage_total
                    match_stats = self._reduced_chorus_match_stats("train")
                    logs["chorus_usage/matched_tokens_avg"] = (
                        match_stats["matched_tokens"] / usage_total
                    )
                    logs["chorus_usage/returned_tokens_avg"] = (
                        match_stats["returned_tokens"] / usage_total
                    )
                    logs["chorus_usage/sonata_tokens_avg"] = (
                        match_stats["sonata_tokens"] / usage_total
                    )
                    logs["chorus_usage/chorus_tokens_avg"] = (
                        match_stats["chorus_tokens"] / usage_total
                    )
                    logs["chorus_usage/used_chorus_tokens_avg"] = (
                        match_stats["used_chorus_tokens"] / usage_total
                    )
                    logs["chorus_usage/sidecar_reports"] = match_stats["sidecar_reports"]
                    logs["chorus_usage/sidecar_enabled_reports"] = (
                        match_stats["sidecar_enabled_reports"]
                    )
                    if match_stats["sonata_tokens"] > 0:
                        logs["chorus_usage/matched_to_sonata_frac"] = (
                            match_stats["matched_tokens"] / match_stats["sonata_tokens"]
                        )
                    if match_stats["chorus_tokens"] > 0:
                        logs["chorus_usage/matched_to_chorus_frac"] = (
                            match_stats["used_chorus_tokens"] / match_stats["chorus_tokens"]
                        )
                        logs["chorus_usage/sidecar_unmapped_frac"] = (
                            match_stats["sidecar_unmapped_chorus_tokens"]
                            / match_stats["chorus_tokens"]
                        )
                    if match_stats["chorus_gate_count"] > 0:
                        logs["chorus_usage/gate_avg"] = (
                            match_stats["chorus_gate_weighted_sum"]
                            / match_stats["chorus_gate_count"]
                        )
                    if match_stats["coord_match_reports"] > 0:
                        logs["chorus_usage/coord_match_ratio_avg"] = (
                            match_stats["coord_match_ratio_sum"]
                            / match_stats["coord_match_reports"]
                        )
                        logs["chorus_usage/coord_coverage_ratio_avg"] = (
                            match_stats["coord_coverage_ratio_sum"]
                            / match_stats["coord_match_reports"]
                        )
                logs["reconstructed_total_loss"] = reconstructed_total_loss

            self._reset_train_component_window()

        super().log(logs, *args, **kwargs)

    @override
    def training_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        num_items_in_batch: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        self._update_chorus_contrastive_loss_weight()
        loss = super().training_step(model, inputs, num_items_in_batch)
        unwrapped_model = (
            self.accelerator.unwrap_model(model)
            if hasattr(self, "accelerator") and self.accelerator is not None
            else model
        )
        next_token_loss = self._tensor_to_float(
            getattr(unwrapped_model, "_last_next_token_loss", None)
        )
        chorus_contrastive_loss = self._tensor_to_float(
            getattr(unwrapped_model, "_last_chorus_contrastive_loss", None)
        )
        self._record_chorus_usage(
            getattr(unwrapped_model, "_last_chorus_modality_usage", None),
            "train",
        )
        if next_token_loss is not None:
            self._train_window_steps += 1
            self._train_next_token_loss_sum += float(next_token_loss)
            if chorus_contrastive_loss is not None:
                self._train_chorus_contrastive_steps += 1
                self._train_chorus_contrastive_loss_sum += float(chorus_contrastive_loss)

        self._maybe_log_mm_grads(model)
        return loss

    def _evaluate_once(self, *args, **kwargs) -> dict[str, float]:
        self._reset_eval_component_window()
        metrics = super().evaluate(*args, **kwargs)

        reduced_steps_sum, reduced_steps = self._reduce_sum_count(
            float(self._eval_steps),
            self._eval_steps,
        )
        if reduced_steps > 0 and reduced_steps_sum > 0:
            next_sum, _ = self._reduce_sum_count(
                self._eval_next_token_loss_sum,
                self._eval_steps,
            )
            chorus_sum, chorus_steps = self._reduce_sum_count(
                self._eval_chorus_contrastive_loss_sum,
                self._eval_chorus_contrastive_steps,
            )
            next_avg = next_sum / reduced_steps_sum
            chorus_weight = self._get_chorus_contrastive_loss_weight()

            metric_key_prefix = kwargs.get("metric_key_prefix", "eval")
            metrics[f"{metric_key_prefix}_next_token_loss"] = next_avg
            reconstructed_total_loss = next_avg
            chorus_avg = chorus_sum / chorus_steps if chorus_steps > 0 else 0.0
            metrics[f"{metric_key_prefix}_chorus_contrastive_loss"] = chorus_avg
            metrics[f"{metric_key_prefix}_chorus_contrastive_loss_weight"] = chorus_weight
            metrics[f"{metric_key_prefix}_chorus_contrastive_loss_weighted"] = (
                chorus_avg * chorus_weight
            )
            metrics[f"{metric_key_prefix}_chorus_contrastive_steps"] = chorus_steps
            if chorus_steps > 0:
                reconstructed_total_loss += chorus_avg * chorus_weight
            usage_counts, reason_counts, usage_total = self._reduced_chorus_usage("eval")
            if usage_total > 0:
                metrics[f"{metric_key_prefix}_chorus_usage_avg_count"] = usage_counts["avg"]
                metrics[f"{metric_key_prefix}_chorus_usage_append_count"] = usage_counts["append"]
                metrics[f"{metric_key_prefix}_chorus_usage_chorus_count"] = usage_counts["chorus"]
                metrics[f"{metric_key_prefix}_chorus_usage_chorus_matched_count"] = (
                    usage_counts["chorus_matched"]
                )
                metrics[f"{metric_key_prefix}_chorus_usage_chorus_coord_matched_count"] = (
                    usage_counts["chorus_coord_matched"]
                )
                metrics[f"{metric_key_prefix}_chorus_usage_pcd_count"] = usage_counts["pcd"]
                metrics[f"{metric_key_prefix}_chorus_usage_pcd_missing_scene_id_count"] = (
                    reason_counts["missing_scene_id"]
                )
                metrics[f"{metric_key_prefix}_chorus_usage_pcd_missing_chorus_count"] = (
                    reason_counts["missing_chorus"]
                )
                metrics[f"{metric_key_prefix}_chorus_usage_pcd_no_match_count"] = (
                    reason_counts["no_match"]
                )
                metrics[f"{metric_key_prefix}_chorus_usage_pcd_drop_chorus_count"] = (
                    reason_counts["drop_chorus"]
                )
                metrics[f"{metric_key_prefix}_chorus_usage_pcd_forced_count"] = (
                    reason_counts["forced_pcd"]
                )
                metrics[f"{metric_key_prefix}_chorus_usage_avg_frac"] = (
                    usage_counts["avg"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_append_frac"] = (
                    usage_counts["append"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_chorus_frac"] = (
                    usage_counts["chorus"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_chorus_matched_frac"] = (
                    usage_counts["chorus_matched"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_chorus_coord_matched_frac"] = (
                    usage_counts["chorus_coord_matched"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_pcd_frac"] = (
                    usage_counts["pcd"] / usage_total
                )
                match_stats = self._reduced_chorus_match_stats("eval")
                metrics[f"{metric_key_prefix}_chorus_usage_matched_tokens_avg"] = (
                    match_stats["matched_tokens"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_returned_tokens_avg"] = (
                    match_stats["returned_tokens"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_sonata_tokens_avg"] = (
                    match_stats["sonata_tokens"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_chorus_tokens_avg"] = (
                    match_stats["chorus_tokens"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_used_chorus_tokens_avg"] = (
                    match_stats["used_chorus_tokens"] / usage_total
                )
                metrics[f"{metric_key_prefix}_chorus_usage_sidecar_reports"] = (
                    match_stats["sidecar_reports"]
                )
                metrics[f"{metric_key_prefix}_chorus_usage_sidecar_enabled_reports"] = (
                    match_stats["sidecar_enabled_reports"]
                )
                if match_stats["sonata_tokens"] > 0:
                    metrics[f"{metric_key_prefix}_chorus_usage_matched_to_sonata_frac"] = (
                        match_stats["matched_tokens"] / match_stats["sonata_tokens"]
                    )
                if match_stats["chorus_tokens"] > 0:
                    metrics[f"{metric_key_prefix}_chorus_usage_matched_to_chorus_frac"] = (
                        match_stats["used_chorus_tokens"] / match_stats["chorus_tokens"]
                    )
                    metrics[f"{metric_key_prefix}_chorus_usage_sidecar_unmapped_frac"] = (
                        match_stats["sidecar_unmapped_chorus_tokens"]
                        / match_stats["chorus_tokens"]
                    )
                if match_stats["chorus_gate_count"] > 0:
                    metrics[f"{metric_key_prefix}_chorus_usage_gate_avg"] = (
                        match_stats["chorus_gate_weighted_sum"]
                        / match_stats["chorus_gate_count"]
                    )
                if match_stats["coord_match_reports"] > 0:
                    metrics[f"{metric_key_prefix}_chorus_usage_coord_match_ratio_avg"] = (
                        match_stats["coord_match_ratio_sum"]
                        / match_stats["coord_match_reports"]
                    )
                    metrics[f"{metric_key_prefix}_chorus_usage_coord_coverage_ratio_avg"] = (
                        match_stats["coord_coverage_ratio_sum"]
                        / match_stats["coord_match_reports"]
                    )
            metrics[f"{metric_key_prefix}_reconstructed_total_loss"] = reconstructed_total_loss

        # Hugging Face's internal evaluation loop logs metrics before this
        # override adds the custom eval component losses. Log once more here so
        # periodic step-based evals emit the augmented metrics to stdout/W&B and
        # persist them in trainer_state.json.
        if metrics:
            self.log(metrics)

        return metrics

    @override
    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        metric_key_prefix = kwargs.get("metric_key_prefix", "eval")
        metrics = self._evaluate_once(*args, **kwargs)

        if not self.eval_chorus_fusion_modes:
            return metrics

        current_mode = self._current_chorus_fusion_mode()
        extra_metrics = {}
        for mode in self.eval_chorus_fusion_modes:
            mode_prefix = f"{metric_key_prefix}_{mode}"
            if mode == current_mode:
                alias_metrics = self._alias_metric_prefix(
                    metrics,
                    metric_key_prefix,
                    mode_prefix,
                )
                if alias_metrics:
                    extra_metrics.update(alias_metrics)
                    self.log(alias_metrics)
                continue

            mode_kwargs = dict(kwargs)
            mode_kwargs["metric_key_prefix"] = mode_prefix
            with self._temporary_chorus_fusion_mode(mode):
                extra_metrics.update(self._evaluate_once(*args, **mode_kwargs))

        metrics.update(extra_metrics)
        return metrics

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        self._update_chorus_contrastive_loss_weight()
        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        unwrapped_model = (
            self.accelerator.unwrap_model(model)
            if hasattr(self, "accelerator") and self.accelerator is not None
            else model
        )
        next_token_loss = self._tensor_to_float(
            getattr(unwrapped_model, "_last_next_token_loss", None)
        )
        chorus_contrastive_loss = self._tensor_to_float(
            getattr(unwrapped_model, "_last_chorus_contrastive_loss", None)
        )
        self._record_chorus_usage(
            getattr(unwrapped_model, "_last_chorus_modality_usage", None),
            "eval",
        )
        if next_token_loss is not None:
            self._eval_steps += 1
            self._eval_next_token_loss_sum += float(next_token_loss)
            if chorus_contrastive_loss is not None:
                self._eval_chorus_contrastive_steps += 1
                self._eval_chorus_contrastive_loss_sum += float(chorus_contrastive_loss)

        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = (
                self.processing_class.pad_token_id
            )
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    register_spatiallm_templates(
        cutoff_len=data_args.cutoff_len,
        num_bins=data_args.num_bins,
        do_augmentation=data_args.do_augmentation,
        random_scaling=data_args.random_scaling,
        random_rotation=data_args.random_rotation,
        scene_graph_mode=data_args.scene_graph_mode,
        pcd_only_random_scaling=data_args.pcd_only_random_scaling,
        pcd_only_scaling_min=data_args.pcd_only_scaling_min,
        pcd_only_scaling_max=data_args.pcd_only_scaling_max,
        chorus_fusion_enabled=model_args.chorus_fusion_enabled,
        chorus_aligned_root=model_args.chorus_aligned_root,
        chorus_fusion_mode=model_args.chorus_fusion_mode,
        chorus_modality_dropout_rate=model_args.chorus_modality_dropout_rate,
        chorus_drop_pcd_probability=model_args.chorus_drop_pcd_probability,
    )

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(model_args, data_args, training_args)
    model = load_model(
        tokenizer, data_args, model_args, finetuning_args, training_args.do_train
    )

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=(
            8 if training_args.do_train else None
        ),  # for shift short attention
        label_pad_token_id=(
            IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        ),
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [
        tokenizer.eos_token_id
    ] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        eval_chorus_fusion_modes=data_args.eval_chorus_fusion_modes,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args)
    )

    callbacks: list[Any] = []
    callbacks.append(LogCallback())
    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    callbacks.append(
        ReporterCallback(model_args, data_args, finetuning_args, generating_args)
    )  # add to last

    run_sft(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
    )

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    _training_function(config={"args": args})


if __name__ == "__main__":
    run_exp()
