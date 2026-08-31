"""The evaluation runner (TASKS.md P3-5): one config -> one W&B run per matcher -> per-pair
rows to Parquet plus the copy-pasteable console block."""

from __future__ import annotations

from cmreg.eval.runner import COMPARISON_KEYS, RunnerError, run_benchmark

__all__ = ["COMPARISON_KEYS", "RunnerError", "run_benchmark"]
