"""Summary aggregation and the copy-pasteable console block.

Two responsibilities, deliberately together: **nothing else in the package formats a result.**
Training happens on a Windows server that cannot hand files back, so the plain-text block this
module renders is the primary channel by which numbers reach a human -- W&B carries the same
values, and Parquet keeps the per-pair rows where they were produced.

The aggregation rules are stated here once and tested, because every one of them is a place
where two benchmarks silently mean different things by the same word:

* ``reg/epe_mean`` is the mean over pairs of each pair's mean per-pixel EPE; ``reg/epe_median``
  is the *median over pairs* of that same per-pair mean. Not the median of the pooled pixels:
  pooling weights a 4K pair 25x a 640x512 one, so a dataset-wise median would mostly describe
  whichever images happen to be largest.
* ``reg/mace`` is the mean per-pair four-corner error, over successes only.
* ``reg/auc_*`` and ``reg/success_rate_*`` run over per-pair corner errors **including
  failures**, which enter as ``inf`` and contribute zero to both.
* ``match/*`` and ``time/*`` are means over all pairs, failures included -- a matcher that
  finds nothing still spent the time and still found nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cmreg.metrics import RegistrationMetrics, auc, success_rate
from cmreg.metrics.schema import (
    MATCH_INLIER_RATIO,
    MATCH_INLIERS,
    MATCH_REPROJ_ERR,
    MATCH_TOTAL,
    TIME_ESTIMATE_MS,
    TIME_EXTRACT_MS,
    TIME_MATCH_MS,
    TIME_TOTAL_MS,
)
from cmreg.results.store import PairRow

_RULE_WIDTH = 68
_KEY_WIDTH = 26
# Wide enough for the longest metric stem (`success_rate_3px`, 16 chars) plus a separating
# space -- at exactly 16 the columns abut and the header becomes unreadable.
_COLUMN_WIDTH = 18


class ReportError(ValueError):
    """Raised when a summary is asked for over rows that cannot support one."""


@dataclass(frozen=True, slots=True)
class Summary:
    """One matcher's aggregated result: the frozen metric keys, plus what produced them."""

    metrics: dict[str, float]
    n_pairs: int
    n_failed: int
    # The identity columns, all of which are constant within a summarised group.
    context: dict[str, str]


def summarize(rows: Sequence[PairRow], thresholds_px: Sequence[float]) -> Summary:
    """Aggregate one group of rows -- conventionally all rows for a single matcher."""
    if not rows:
        raise ReportError("cannot summarise zero rows")

    successes = [row for row in rows if row.success]
    n_failed = len(rows) - len(successes)

    # Failures enter the threshold metrics as infinite error and are absent from the means.
    # See the module docstring; this is the single most consequential line in the file.
    corner_errors = np.array(
        [row.corner_err if row.success and row.corner_err is not None else np.inf for row in rows],
        dtype=np.float64,
    )

    registration = RegistrationMetrics(
        epe_mean=_mean([row.epe_mean for row in successes]),
        epe_median=_median([row.epe_mean for row in successes]),
        mace=_mean([row.corner_err for row in successes]),
        auc={float(t): auc(corner_errors, float(t)) for t in thresholds_px},
        success_rate={float(t): success_rate(corner_errors, float(t)) for t in thresholds_px},
        failure_rate=n_failed / len(rows),
        n_pairs=len(rows),
    )

    metrics = registration.to_dict()
    metrics[MATCH_TOTAL] = _mean([float(row.n_matches) for row in rows])
    metrics[MATCH_INLIERS] = _mean([float(row.n_inliers) for row in rows])
    metrics[MATCH_INLIER_RATIO] = _mean([row.inlier_ratio for row in rows])
    metrics[MATCH_REPROJ_ERR] = _mean([row.reproj_err for row in successes])
    metrics[TIME_EXTRACT_MS] = _mean([row.extract_ms for row in rows])
    metrics[TIME_MATCH_MS] = _mean([row.match_ms for row in rows])
    metrics[TIME_ESTIMATE_MS] = _mean([row.estimate_ms for row in rows])
    metrics[TIME_TOTAL_MS] = _mean([row.total_ms for row in rows])

    first = rows[0]
    context = {
        "run": first.run_name,
        "config_hash": first.config_hash,
        "git_sha": first.git_sha,
        "matcher": first.matcher,
        "estimator": f"{first.estimator} @ {first.threshold_px:g}px",
        "preprocess": (
            f"{first.preprocess_ref} / {first.preprocess_mov} "
            f"x{first.upsample} {first.interpolation}"
        ),
        "warp": first.warp,
        "moving": first.moving,
        "dataset": f"{first.dataset} [{first.split}]",
        "domain": first.domain,
        "platform": first.platform,
        "seed": str(first.seed),
    }
    return Summary(metrics=metrics, n_pairs=len(rows), n_failed=n_failed, context=context)


def _mean(values: Sequence[float | None]) -> float:
    """Mean over the non-null entries. NaN for an empty set -- a run in which every pair
    failed has no mean, and returning 0.0 there would read as a perfect score."""
    present = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(present)) if present else float("nan")


def _median(values: Sequence[float | None]) -> float:
    present = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(present)) if present else float("nan")


def render(summary: Summary) -> str:
    """The console block. Fixed width, fixed key order, one drag to select."""
    rule = "-" * _RULE_WIDTH
    lines = ["=== CMREG RESULT ===", rule]
    lines += [f"{key:<14}{value}" for key, value in summary.context.items()]
    lines.append(rule)
    for key, value in summary.metrics.items():
        lines.append(f"{key:<{_KEY_WIDTH}}{_format(key, value)}")
    lines.append(rule)
    lines.append(
        f"{'failed pairs':<{_KEY_WIDTH}}{summary.n_failed} / {summary.n_pairs}"
        + _failure_note(summary)
    )
    lines.append("=== END ===")
    return "\n".join(lines)


def _failure_note(summary: Summary) -> str:
    """Spell out the asymmetry rather than trusting the reader to remember it."""
    if summary.n_failed == 0:
        return ""
    return "   (excluded from epe/mace/reproj; counted as inf in auc/success_rate)"


def _format(key: str, value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    # Counts read as counts; everything else gets four decimals so two methods that differ in
    # the third are distinguishable in a pasted block.
    return f"{value:.0f}" if key.endswith("n_pairs") else f"{value:.4f}"


def render_comparison(summaries: Sequence[Summary], keys: Sequence[str]) -> str:
    """A matcher-per-row table across a chosen metric subset, for multi-matcher runs."""
    if not summaries:
        raise ReportError("cannot compare zero summaries")
    # Widened to the longest name present rather than fixed: `matchanything-eloftr` is 20
    # characters and a fixed 16 pushes its row one column out of alignment, which is exactly
    # the kind of damage a copy-pasted block cannot survive -- this table is how results reach
    # the Mac from the training server, so it has to stay readable as plain text.
    names = [str(summary.context["matcher"]) for summary in summaries]
    name_width = max(len("matcher"), *(len(name) for name in names)) + 2
    header = f"{'matcher':<{name_width}}" + "".join(
        f"{key.split('/')[-1]:>{_COLUMN_WIDTH}}" for key in keys
    )
    rule = "-" * len(header)
    lines = ["=== CMREG COMPARISON ===", header, rule]
    for name, summary in zip(names, summaries, strict=True):
        cells = "".join(
            f"{_format(key, summary.metrics.get(key, float('nan'))):>{_COLUMN_WIDTH}}"
            for key in keys
        )
        lines.append(f"{name:<{name_width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)
