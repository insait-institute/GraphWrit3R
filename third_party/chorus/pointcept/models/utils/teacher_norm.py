import torch
import torch.nn as nn
import torch.distributed as dist


def _concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all ranks without gradient sync issues."""
    if (not dist.is_available()) or (not dist.is_initialized()):
        return tensor
    tensor = tensor.contiguous()
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


class TeacherNorm(nn.Module):
    """Normalize teacher targets with running mean/std statistics."""

    def __init__(
        self,
        feature_dim: int,
        agg_dims=(0,),
        momentum: float = 0.9,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("std", torch.ones(feature_dim))
        self.momentum = momentum
        self.agg_dims = tuple(agg_dims)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if (not self.training) or self.momentum == 0.0:
            mean = self.mean
            std = self.std
        else:
            x_all = _concat_all_gather(x)
            reduce_dims = tuple(self.agg_dims)
            mean = x_all.mean(dim=reduce_dims, keepdim=False)
            std = x_all.std(dim=reduce_dims, keepdim=False, unbiased=False)
            self.mean.mul_(1 - self.momentum).add_(mean * self.momentum)
            self.std.mul_(1 - self.momentum).add_(std * self.momentum)
        return (x - mean) / torch.clamp(std, min=self.eps)
