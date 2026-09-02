"""Which correspondences reach the solver (TASKS.md P3-12c, PLAN.md §7 "number of sampled
matches").

This is the *downstream* half of stage F. `preprocess.input_scale` (P3-12a) changes what the
matcher sees and therefore costs one match pass per level; this axis changes only what is handed
to `estimate_warp`, so N caps are N `cv2.findHomography` calls off one `MatchResult` -- the same
economics as P3-10's estimator and P3-4a's warp model, and the reason the two halves of P3-12
were split into separate stages rather than one table with two cost models.

**It is deliberately not `MatchConfig.max_keypoints`.** That is the detector's budget, and the
detector-free backends do not honour it -- `minima-roma` returns 10 000 matches under a 2 048
budget because `vismatch/im_models/minima.py` calls `sample()` with no `num=`, and
`matchanything.py` has no budget at all (TASKS.md P0-2). An axis expressed through it would be
flat for half of `reduced-8` for a reason that has nothing to do with the number of matches.
Selecting here is defined for every backend, at the cost of measuring "fewer correspondences in
the fit" rather than "a cheaper matcher" -- which is the question PLAN.md §7 actually asks.

Two orderings, and the second is a control rather than a fallback
----------------------------------------------------------------
`CONFIDENCE` takes the best-scoring matches, which is what a practitioner does and what the axis
is nominally about. `RANDOM` takes an unbiased sample of the same size. At equal cap the two
differ only in *which* correspondences survive, so:

* if they separate, the matcher's score carries real information about which matches are
  geometrically good -- evidence for PLAN.md §6.2's certainty-map baseline;
* if they do not, the axis is measuring the count alone, and the certainty map is not a useful
  selector. That is a negative result about the baseline the dense error head has to beat, and it
  is recorded rather than dropped (X-4).

`RANDOM` is also the only ordering defined for a backend that scores no matches -- `xfeat`, alone
in `reduced-8` (TASKS.md P0-2) -- which is why the confidence arm has a hole there and the axis
still covers all eight matchers.

Nesting, and why it is a required property rather than a nicety
--------------------------------------------------------------
Both orderings are **prefixes of one total order**, so the cap-256 set is a subset of the cap-512
set. Without that the axis is not a sweep but seven unrelated draws, and a difference between two
caps could be the sample rather than the size. Nothing downstream would notice if it broke, so
`tests/test_select.py` pins it.

**At no cap the order is the identity.** `selected_indices` returns `arange(n)` untouched rather
than a sorted permutation of it, which is what makes an uncapped run byte-identical to a run made
before this axis existed -- the property `Config.config_hash`'s exemption claims and
`tests/test_runner.py` checks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cmreg.config import EstimateConfig, MatchSelection

# The token a row carries when this run's (matcher, selection) is the one hole in the axis: a
# confidence-ranked cap against a backend that scores no matches. Greppable, and distinct from
# `estimator_needs_confidence` even though the missing signal is the same one -- that hole is
# PROSAC's and this one is the selector's, and a table that could not tell them apart would
# attribute a stage-F absence to stage D.
NEEDS_CONFIDENCE_REASON = "selection_needs_confidence"


def needs_confidence(config: EstimateConfig, confidence: NDArray[np.floating] | None) -> bool:
    """Whether this variant wants a per-match score the matcher did not supply.

    False when there is no cap: with nothing to drop the selection is inert, so an uncapped cell
    runs against a scoreless matcher exactly as it always has.
    """
    return (
        confidence is None
        and config.max_matches > 0
        and config.match_selection is MatchSelection.CONFIDENCE
    )


def selected_indices(
    n_matches: int,
    confidence: NDArray[np.floating] | None,
    config: EstimateConfig,
    seed: int,
) -> NDArray[np.intp]:
    """Indices of the correspondences to fit, as a prefix of this variant's total order.

    ``seed`` keys the random ordering and is expected to be ``seeding.py::cell_seed(gt.seed,
    pair index, matcher)`` -- the same key every other per-cell stochastic decision uses, so a
    row stays reproducible in isolation.

    Raises nothing: the one unsatisfiable combination is caught by :func:`needs_confidence` in
    the runner, which records it as a row rather than aborting eleven variants that had already
    paid for their match (`eval/runner.py::_unsupported`).
    """
    cap = config.max_matches
    if cap <= 0 or cap >= n_matches:
        # Untouched, not `argsort`ed then truncated to everything: a reordering would change
        # PROSAC's sample order and `Estimate.inlier_mask`'s indexing for no gain, and would
        # break the byte-identity an uncapped run has with every run made before this axis.
        return np.arange(n_matches, dtype=np.intp)

    if config.match_selection is MatchSelection.RANDOM:
        # A **local** Generator, never `np.random`'s global stream. `seeding.py::seed_cell` seeds
        # that stream for the *matcher*, whose dense backends draw from it a data-dependent
        # number of times; taking the permutation from it would make this subsample depend on how
        # many draws the matcher happened to make -- the P0-2 failure one layer further down.
        return np.asarray(np.random.default_rng(seed).permutation(n_matches)[:cap], dtype=np.intp)

    if confidence is None:  # pragma: no cover - the runner gates this via `needs_confidence`
        raise ValueError("confidence-ranked selection needs per-match confidences")
    # Stable, so ties keep the matcher's own order and the prefix property survives them: an
    # unstable sort may return a different permutation of an equal-scored block at each cap, and
    # the cap-256 set would then not be a subset of the cap-512 one. Descending by negation
    # rather than by `[::-1]`, which reverses the tie blocks and loses the same property.
    order = np.argsort(-np.asarray(confidence, dtype=np.float64), kind="stable")
    return np.asarray(order[:cap], dtype=np.intp)
