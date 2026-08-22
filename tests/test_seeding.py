"""Determinism (TASKS.md P0-9)."""

from __future__ import annotations

import random

import numpy as np
import torch

from cmreg.seeding import seed_everything, seed_worker


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_one_call_seeds_every_rng() -> None:
    seed_everything(11)
    first = _draw()
    seed_everything(11)
    assert _draw() == first
    seed_everything(12)
    assert _draw() != first


def test_seed_worker_is_module_level_and_picklable() -> None:
    """It is pickled to reach a worker on Windows, which spawns rather than forks."""
    import pickle

    assert pickle.loads(pickle.dumps(seed_worker)) is seed_worker


def test_seed_worker_derives_from_the_torch_seed() -> None:
    torch.manual_seed(5)
    seed_worker(0)
    first = (random.random(), float(np.random.rand()))
    torch.manual_seed(5)
    seed_worker(0)
    assert (random.random(), float(np.random.rand())) == first
