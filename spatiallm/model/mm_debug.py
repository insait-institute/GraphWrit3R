import math
import os
from typing import Iterable, Optional

import torch
from torch import nn


_TRUE_VALUES = {"1", "true", "yes", "on"}
_COUNTERS: dict[str, int] = {}


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def enabled() -> bool:
    return _env_true("SPATIALLM_DEBUG_MM")


def grads_enabled() -> bool:
    return _env_true("SPATIALLM_DEBUG_MM_GRADS")


def assert_2d_enabled() -> bool:
    return _env_true("SPATIALLM_DEBUG_MM_ASSERT_2D")


def rank0() -> bool:
    return int(os.getenv("LOCAL_RANK", "0")) == 0


def limit() -> int:
    try:
        return int(os.getenv("SPATIALLM_DEBUG_MM_LIMIT", "5"))
    except ValueError:
        return 5


def should_log(key: str) -> bool:
    if not enabled() or not rank0():
        return False

    max_count = limit()
    count = _COUNTERS.get(key, 0)
    if max_count >= 0 and count >= max_count:
        return False

    _COUNTERS[key] = count + 1
    return True


def log(key: str, message: str) -> None:
    if should_log(key):
        print(f"[SPATIALLM_MM_DEBUG:{key}:{_COUNTERS[key]}] {message}", flush=True)


def tensor_shape(value) -> Optional[list[int]]:
    if torch.is_tensor(value):
        return list(value.shape)
    return None


def mask_sum(value) -> Optional[int]:
    if torch.is_tensor(value):
        return int(value.bool().sum().item())
    return None


def _summarize_params(parameters: Iterable[nn.Parameter]) -> dict[str, int]:
    total = 0
    trainable = 0
    for param in parameters:
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return {"trainable": trainable, "total": total}


def module_param_summary(module: Optional[nn.Module]) -> dict[str, int]:
    if module is None:
        return {"trainable": 0, "total": 0}
    return _summarize_params(module.parameters())


def named_param_summary(parameters: Iterable[tuple[str, nn.Parameter]]) -> dict[str, int]:
    return _summarize_params(param for _, param in parameters)


def format_param_summary(summary: dict[str, int]) -> str:
    total = summary["total"]
    trainable = summary["trainable"]
    pct = 0.0 if total == 0 else 100.0 * trainable / total
    return f"{trainable:,}/{total:,} trainable ({pct:.2f}%)"


def grad_summary(module: Optional[nn.Module]) -> dict[str, float | int]:
    if module is None:
        return {
            "trainable_params": 0,
            "params_with_grad": 0,
            "grad_elements": 0,
            "grad_norm": 0.0,
        }

    trainable_params = 0
    params_with_grad = 0
    grad_elements = 0
    sq_norm = 0.0
    for param in module.parameters():
        if not param.requires_grad:
            continue
        trainable_params += param.numel()
        if param.grad is None:
            continue
        params_with_grad += 1
        grad = param.grad.detach().float()
        grad_elements += grad.numel()
        sq_norm += float(torch.sum(grad * grad).item())

    return {
        "trainable_params": trainable_params,
        "params_with_grad": params_with_grad,
        "grad_elements": grad_elements,
        "grad_norm": math.sqrt(sq_norm),
    }


def format_grad_summary(summary: dict[str, float | int]) -> str:
    return (
        f"norm={summary['grad_norm']:.6g}, "
        f"params_with_grad={summary['params_with_grad']}, "
        f"grad_elements={summary['grad_elements']}, "
        f"trainable_params={summary['trainable_params']}"
    )
