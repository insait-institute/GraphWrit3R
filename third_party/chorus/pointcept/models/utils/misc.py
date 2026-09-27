"""
General Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import time
import functools
from itertools import chain


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def timer_decorator(func):
    """
    A decorator that times a method call and accumulates the total time
    on the instance ('self').
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Initialize attributes on the instance if they don't exist
        if not hasattr(self, '_total_forward_time'):
            self._total_forward_time = 0.0
            self._forward_call_count = 0

        # Start timer
        start_time = time.perf_counter()
        
        # Call the original function (e.g., forward)
        result = func(self, *args, **kwargs)
        
        # Stop timer and update totals
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        self._total_forward_time += elapsed_time
        self._forward_call_count += 1
        
        # Optional: Print the time for this call and the total
        print(f"[Timer] Call {self._forward_call_count}: {elapsed_time:.6f}s | Total Time: {self._total_forward_time:.6f}s")
        
        return result
    return wrapper
