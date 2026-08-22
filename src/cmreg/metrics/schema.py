"""The frozen metric key vocabulary (TASKS.md P0-6).

Every run logs the same key names or cross-run comparison breaks. The keys below are the
schema from TASKS.md §0, declared once here so no script can invent a name; TASKS.md X-5
("metrics schema unchanged since P0-6, or migration applied to all prior runs") audits
against this file.

Keys for metrics this project cannot compute yet (``uncert/*`` needs P6-1, ``gtfree/*`` needs
P4-1) are declared anyway. Reserving the name now costs nothing; discovering at month four
that half the runs logged ``uncert/AUSE`` and half ``uncert/ause`` costs a re-run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Registration accuracy (PLAN.md §6.1).
EPE_MEAN = "reg/epe_mean"
EPE_MEDIAN = "reg/epe_median"
MACE = "reg/mace"

# Failure accounting. An *addition* to the minimum set TASKS.md §0 lists, not a rename of any
# of it, so X-5 ("schema unchanged, or migration applied to all prior runs") is satisfied
# without migration. It is here because the alternative is worse: a failed pair has no
# homography and so cannot contribute an EPE or a corner error, meaning `reg/epe_*` and
# `reg/mace` are necessarily means over *successes only*. Reporting those without the failure
# rate beside them lets a method that solves 5% of pairs beautifully outrank one that solves
# all of them well, which is how a benchmark table stops being defensible.
FAILURE_RATE = "reg/failure_rate"
N_PAIRS = "reg/n_pairs"

# Matching and robust estimation. `match/reproj_err` is reported but never led with: PLAN.md
# §6.4 records that it is gameable -- a degenerate 4-point homography scores near zero, and
# shrinking the RANSAC threshold lowers it artificially.
MATCH_TOTAL = "match/total"
MATCH_INLIERS = "match/inliers"
MATCH_INLIER_RATIO = "match/inlier_ratio"
MATCH_REPROJ_ERR = "match/reproj_err"

# Registration-error prediction (PLAN.md §6.2) -- the headline axis. Lands with P6-1.
AUSE = "uncert/ause"
SPEARMAN = "uncert/spearman"
AURG = "uncert/aurg"

# GT-free similarity signals (PLAN.md §6.3). Land with P4-1.
GTFREE_MI = "gtfree/mi"
GTFREE_NMI = "gtfree/nmi"
GTFREE_NCC = "gtfree/ncc"
GTFREE_EDGE_SSIM = "gtfree/edge_ssim"
GTFREE_MIND = "gtfree/mind"

# Efficiency (PLAN.md §6.5), stage-separated so a slow matcher and a slow estimator are
# distinguishable.
TIME_EXTRACT_MS = "time/extract_ms"
TIME_MATCH_MS = "time/match_ms"
TIME_ESTIMATE_MS = "time/estimate_ms"
TIME_TOTAL_MS = "time/total_ms"
SYS_PEAK_GPU_MB = "sys/peak_gpu_mb"
SYS_PARAMS_M = "sys/params_m"
SYS_GFLOPS = "sys/gflops"


def auc_key(threshold_px: float) -> str:
    """``reg/auc_3px`` for 3.0. Formatted through one function so 3 and 3.0 cannot produce
    two different keys for the same threshold."""
    return f"reg/auc_{_threshold(threshold_px)}px"


def success_rate_key(threshold_px: float) -> str:
    """``reg/success_rate_3px`` for 3.0."""
    return f"reg/success_rate_{_threshold(threshold_px)}px"


def _threshold(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


@dataclass(frozen=True, slots=True)
class RegistrationMetrics:
    """The GT-based accuracy block for one evaluation, in the frozen key names."""

    epe_mean: float
    epe_median: float
    mace: float
    auc: dict[float, float]
    success_rate: dict[float, float]
    # Fraction of pairs with no fitted homography. Unlike the three means above, `auc` and
    # `success_rate` *do* include failures -- they enter as infinite corner error, which
    # contributes zero to both -- so the two families are not means over the same set. That
    # asymmetry is deliberate and is why this field is not optional.
    failure_rate: float
    n_pairs: int

    def to_dict(self) -> dict[str, float]:
        """Flatten to the logged key names. This is the only place metric keys are emitted."""
        flat = {
            EPE_MEAN: self.epe_mean,
            EPE_MEDIAN: self.epe_median,
            MACE: self.mace,
            FAILURE_RATE: self.failure_rate,
            N_PAIRS: float(self.n_pairs),
        }
        flat.update({auc_key(t): v for t, v in self.auc.items()})
        flat.update({success_rate_key(t): v for t, v in self.success_rate.items()})
        return flat

    def as_record(self) -> dict[str, object]:
        """The nested form, for the Parquet per-pair store rather than for W&B."""
        return asdict(self)
