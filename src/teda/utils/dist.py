"""Thin torch.distributed helpers. Everything works unchanged in a single process."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List

import torch
import torch.distributed as dist


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed(timeout_minutes: int = 60) -> DistContext:
    """Initialise from torchrun env vars (RANK / WORLD_SIZE / LOCAL_RANK); falls back to 1 process."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, timeout=timedelta(minutes=timeout_minutes))
    return DistContext(rank, world, local, device)


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Average a python scalar across ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


def all_reduce_sum(values: List[float], device: torch.device) -> List[float]:
    if not (dist.is_available() and dist.is_initialized()):
        return [float(v) for v in values]
    t = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def gather_objects(obj: Any) -> List[Any]:
    """all_gather_object; returns a list with one entry per rank (rank order)."""
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]
