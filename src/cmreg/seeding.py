"""Every RNG this project touches, seeded from one place.

Ported from ``../Thermal-To-Optical-Translation/src/t2o/seeding.py``, which documents the
bug it fixed: ``numpy.random`` and Python's global ``random`` were never seeded, and the
train ``DataLoader`` had no ``worker_init_fn``, so at ``workers > 0`` each worker process
started from an unseeded ``random``/numpy state.

The stakes here are the same shape. TASKS.md X-3 requires >= 5 seeds with confidence
intervals on anything stochastic -- RANSAC and training both -- and unseeded RNG is variance
that cannot be attributed to anything.

**Deliberately absent: ``torch.use_deterministic_algorithms(True)`` and the cuDNN
determinism flags.** Seed-to-seed variance is a quantity this project reports, not one it
suppresses; forcing it away would hide the measurement and cost throughput.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed torch (CPU and CUDA), numpy, and Python's ``random``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Redundant on current torch -- `torch.manual_seed` forwards here itself. Repeated
    # anyway so the set of RNGs this function covers can be read off the function, not off
    # torch's source; a future torch that stops forwarding would otherwise silently drop CUDA.
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for a ``DataLoader``: PyTorch's own documented recipe.

    ``worker_id`` is unused -- the ``DataLoader`` passes it positionally, so it is part of
    the signature, but torch has already folded it into each worker's *torch* RNG seed. The
    per-worker seed is therefore read back out of ``torch.initial_seed()`` rather than
    derived again here, which keeps ``random``/numpy in step with torch inside the worker
    instead of inventing a second, unrelated stream. The ``% 2**32`` is numpy's range limit.

    **Must stay a module-level function.** The training machine is native Windows, which
    spawns rather than forks (PLAN.md §8), so ``worker_init_fn`` is pickled to reach the
    worker. A lambda or a closure would fail there and nowhere else -- exactly the
    silent-hang class of bug §8 warns about.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def cell_seed(seed: int, index: int, matcher: str) -> int:
    """The seed for one ``(pair, matcher)`` evaluation cell.

    Hashed over all three inputs rather than derived arithmetically like
    ``gt/synthetic.py::warp_seed``, because one of them is a string. The property that matters
    is the same one: **a cell's result must depend only on the cell.**
    """
    digest = hashlib.sha256(f"{seed}:{index}:{matcher}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def seed_cell(seed: int, index: int, matcher: str) -> None:
    """Reseed every RNG before one evaluation cell runs.

    Measured, not assumed: RoMa samples its correspondences with ``torch.multinomial``
    (``romatch``'s ``sample``), so with an unseeded ambient stream two runs of an identical
    config over identical pairs gave ``reg/mace`` 121.2 and 43.7 on the same eight MSRS pairs
    -- a factor of three, from nothing but RNG state. Recorded in TASKS.md P0-2.

    Seeding once per *run* would not be enough. The loop is pairs-outer/matchers-inner, so
    every matcher consumes from the stream the next one draws from: RoMa's number would then
    depend on which other matchers shared its config file, and the P3-7 grid would disagree
    with a single-matcher rerun of the same cell. Keying on ``(seed, index, matcher)`` makes
    the row reproducible in isolation, which is what the results store's one-row-per-cell
    model already claims.
    """
    seed_everything(cell_seed(seed, index, matcher))
