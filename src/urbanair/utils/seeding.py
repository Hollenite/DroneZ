from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedBundle:
    seed: int
    rng: random.Random


def create_seed_bundle(seed: int | None) -> SeedBundle:
    value = 0 if seed is None else int(seed)
    return SeedBundle(seed=value, rng=random.Random(value))


def derive_seed(seed: int, offset: int) -> int:
    return (int(seed) * 9973 + int(offset) * 7919) % (2**31)
