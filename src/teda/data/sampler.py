"""Duration-bucketed dynamic batching with DDP sharding.

* A batch holds as many utterances as fit in `max_frames` PADDED frames (n * longest). Sorting by length
  inside buckets keeps padding small.
* All ranks build the same global batch list from (seed, epoch); rank r takes batches[r::world].
  For training the list is padded to a multiple of world_size so every rank runs the same number of steps
  (a mismatch would hang DDP). Evaluation does not pad: no collective ops happen inside the eval loop.
* `skip` lets a resumed run continue in the middle of an epoch without loading the skipped data.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional, Sequence


class DynamicBatchSampler:
    def __init__(
        self,
        lengths: Sequence[int],
        max_frames: int,
        world_size: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        bucket_size: int = 512,
        max_batch_size: Optional[int] = None,
        pad_to_world: bool = True,
    ):
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        too_long = [i for i, n in enumerate(lengths) if n > max_frames]
        if too_long:
            raise ValueError(
                f"{len(too_long)} utterances are longer than max_frames={max_frames} frames "
                f"(e.g. index {too_long[0]}, {lengths[too_long[0]]} frames); raise max_frames or train.max_dur"
            )
        self.lengths = list(lengths)
        self.max_frames = max_frames
        self.world = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.bucket = bucket_size
        self.max_bs = max_batch_size
        self.pad_to_world = pad_to_world
        self.epoch = 0
        self.skip = 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch = epoch
        self.skip = skip

    def _global_batches(self, epoch: int) -> List[List[int]]:
        n = len(self.lengths)
        idx = list(range(n))
        rng = random.Random(self.seed * 1_000_003 + epoch)
        if self.shuffle:
            rng.shuffle(idx)
        batches: List[List[int]] = []
        for s in range(0, n, self.bucket):
            chunk = sorted(idx[s : s + self.bucket], key=lambda i: self.lengths[i])
            cur: List[int] = []
            cur_max = 0
            for i in chunk:
                new_max = max(cur_max, self.lengths[i])
                if cur and (new_max * (len(cur) + 1) > self.max_frames or (self.max_bs and len(cur) >= self.max_bs)):
                    batches.append(cur)
                    cur, cur_max = [], 0
                    new_max = self.lengths[i]
                cur.append(i)
                cur_max = new_max
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        if self.pad_to_world and self.world > 1 and batches:
            k = (-len(batches)) % self.world
            batches += batches[:k]
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        mine = self._global_batches(self.epoch)[self.rank :: self.world]
        for b in mine[self.skip :]:
            yield b

    def __len__(self) -> int:
        return len(self._global_batches(self.epoch)[self.rank :: self.world])
