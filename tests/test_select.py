"""Correspondence selection (TASKS.md P3-12c, `estimate/select.py`).

Four properties, and each is one the axis is meaningless without:

* **nesting** -- a smaller cap's set is a subset of a larger one's, so a difference between two
  columns of the sweep is the *size* and not the sample;
* **identity at no cap** -- an uncapped cell is byte-identical to a run made before this axis,
  which is what `Config.config_hash`'s exemption claims;
* **independence from the ambient RNG** -- the random arm must not move when the matcher happens
  to draw a different number of times, which is the P0-2 failure one layer down;
* **the two arms actually differ** -- otherwise the control column measures nothing and the whole
  design rests on a knob that is not connected (PLAN.md §15A's bug in this stage's shape).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cmreg.config import EstimateConfig, MatchSelection
from cmreg.estimate import needs_confidence, selected_indices

N_MATCHES = 200
CAPS = (8, 16, 32, 64, 128)
SEED = 12345


def variant(cap: int, selection: MatchSelection) -> EstimateConfig:
    return EstimateConfig(max_matches=cap, match_selection=selection)


@pytest.fixture
def confidence() -> np.ndarray:
    """Scores with a deliberate block of exact ties. Ties are where an unstable sort loses the
    nesting property, and a uniform random draw would essentially never produce one."""
    scores = np.random.default_rng(0).random(N_MATCHES)
    scores[50:70] = 0.5
    return scores


@pytest.mark.parametrize("selection", list(MatchSelection))
def test_caps_are_nested(selection: MatchSelection, confidence: np.ndarray) -> None:
    """Every cap's set is a subset of the next larger one's -- for both arms, ties included."""
    chosen = {
        cap: set(selected_indices(N_MATCHES, confidence, variant(cap, selection), SEED).tolist())
        for cap in CAPS
    }
    for smaller, larger in pairwise(CAPS):
        assert chosen[smaller] < chosen[larger], f"{selection.value}: {smaller} not in {larger}"


@pytest.mark.parametrize("selection", list(MatchSelection))
def test_a_cap_selects_exactly_that_many(selection: MatchSelection, confidence: np.ndarray) -> None:
    for cap in CAPS:
        chosen = selected_indices(N_MATCHES, confidence, variant(cap, selection), SEED)
        assert len(chosen) == cap
        assert len(set(chosen.tolist())) == cap


@pytest.mark.parametrize("selection", list(MatchSelection))
def test_no_cap_is_the_identity(selection: MatchSelection, confidence: np.ndarray) -> None:
    """`arange(n)` exactly, not a permutation of it.

    The load-bearing property of the whole axis: it is what makes an uncapped run reproduce a
    pre-P3-12c one byte for byte, and therefore what lets `config_hash` drop the field. A sorted
    order here would change PROSAC's sample sequence and `Estimate.inlier_mask`'s indexing while
    every aggregate still looked plausible.
    """
    chosen = selected_indices(N_MATCHES, confidence, variant(0, selection), SEED)
    assert np.array_equal(chosen, np.arange(N_MATCHES))


@pytest.mark.parametrize("selection", list(MatchSelection))
def test_a_cap_above_the_match_count_is_the_identity(
    selection: MatchSelection, confidence: np.ndarray
) -> None:
    """A cap nothing reaches must be inert too, or the axis's widest columns differ from the
    anchor by a reordering rather than by a count -- and a matcher whose yield sits between two
    caps would show a step that is pure artefact."""
    chosen = selected_indices(N_MATCHES, confidence, variant(N_MATCHES + 1, selection), SEED)
    assert np.array_equal(chosen, np.arange(N_MATCHES))


def test_confidence_takes_the_highest_scores(confidence: np.ndarray) -> None:
    """The arm does what it is named for: no unselected match outscores a selected one."""
    cap = 32
    chosen = selected_indices(N_MATCHES, confidence, variant(cap, MatchSelection.CONFIDENCE), SEED)
    rest = np.setdiff1d(np.arange(N_MATCHES), chosen)
    assert confidence[chosen].min() >= confidence[rest].max()


def test_the_two_arms_select_differently(confidence: np.ndarray) -> None:
    """The control column is a control. If the random arm happened to reproduce the ranked one,
    the stage's central comparison would be flat for a reason that is not a finding."""
    ranked = selected_indices(N_MATCHES, confidence, variant(32, MatchSelection.CONFIDENCE), SEED)
    drawn = selected_indices(N_MATCHES, confidence, variant(32, MatchSelection.RANDOM), SEED)
    assert set(ranked.tolist()) != set(drawn.tolist())


def test_the_random_arm_is_keyed_on_the_seed(confidence: np.ndarray) -> None:
    config = variant(32, MatchSelection.RANDOM)
    first = selected_indices(N_MATCHES, confidence, config, SEED)
    assert np.array_equal(first, selected_indices(N_MATCHES, confidence, config, SEED))
    assert not np.array_equal(first, selected_indices(N_MATCHES, confidence, config, SEED + 1))


def test_the_random_arm_ignores_the_ambient_rng(confidence: np.ndarray) -> None:
    """The pin for the local `Generator`.

    `seeding.py::seed_cell` seeds the global stream for the *matcher*, whose dense backends draw
    from it a data-dependent number of times. Drawing this permutation from that stream would
    make the subsample depend on how many correspondences the matcher happened to sample -- a
    dependence no aggregate would reveal, and the exact shape of the bug P0-2 recorded.
    """
    config = variant(32, MatchSelection.RANDOM)
    np.random.seed(0)
    first = selected_indices(N_MATCHES, confidence, config, SEED)
    np.random.seed(0)
    np.random.random(97)
    assert np.array_equal(first, selected_indices(N_MATCHES, confidence, config, SEED))


def test_the_random_arm_needs_no_confidence() -> None:
    """`xfeat`'s column. The arm exists partly so the axis covers a scoreless backend at all."""
    chosen = selected_indices(N_MATCHES, None, variant(32, MatchSelection.RANDOM), SEED)
    assert len(chosen) == 32


def test_needs_confidence_only_where_the_cap_bites() -> None:
    """Uncapped, the selection is inert, so a scoreless matcher must not be recorded as a hole:
    it runs exactly as it always has."""
    assert needs_confidence(variant(32, MatchSelection.CONFIDENCE), None)
    assert not needs_confidence(variant(0, MatchSelection.CONFIDENCE), None)
    assert not needs_confidence(variant(32, MatchSelection.RANDOM), None)
    assert not needs_confidence(variant(32, MatchSelection.CONFIDENCE), np.zeros(N_MATCHES))
