"""Separated RNG domains: optimization >= one million; held-out seeds below it."""

import random

import numpy as np
import torch


def seed_everything(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def training_seed(seed: int, stream: int, index: int = 0) -> int:
    """Disjoint stream blocks for the supported experiment seeds and run lengths."""
    if not 0 <= index < 100000 or not 0 <= stream < 5:
        raise ValueError("Seed stream exhausted")
    return 1_000_000 + seed * 500_000 + stream * 100_000 + index


def episode_seeds(cfg: dict, split: str) -> list[int]:
    if split not in ("validation", "final"):
        raise ValueError("Split must be validation or final")
    n = cfg["experiment"]["validation_episodes" if split == "validation" else "final_eval_episodes"]
    start = cfg["evaluation"][f"{split}_seed_start"]
    return list(range(start, start + n))
