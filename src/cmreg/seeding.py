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
