"""The evaluation cell, end to end.

The load-bearing test in this file is :func:`test_identical_modalities_register_to_subpixel`.
The cell composes a sampled homography, its inverse, a warp, a preprocessing scale and a
matcher's coordinate frame, and getting any one of them backwards produces plausible-looking
large errors on cross-modal data -- indistinguishable from the modality gap the benchmark
exists to measure. Registering an image against a known warp of *itself* is the only
configuration in which such an error has nowhere to hide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cmreg.config import Config, Variant
from cmreg.eval import RunnerError, run_benchmark
from cmreg.gt import generator, sample_homography
from cmreg.results import FILENAME, read_rows

SUBPIXEL_PX = 1.0


def base_config(manifest: Path, run_dir: Path, **overrides) -> Config:
    """A no-preprocessing, single-matcher cell on the given dataset."""
    declared = {
        "runtime": {"name": "test", "path": str(run_dir), "wandb": False},
        "data": {"manifest": str(manifest), "split": "val"},
        # Both modalities are the same image in the aligned fixture, so any recipe would be
        # applied to both. `none` keeps the test about geometry and nothing else.
        "preprocess": {"reference": Variant.NONE.value, "moving": Variant.NONE.value},
        "match": {"matchers": ["sift"]},
    }
    for section, values in overrides.items():
        declared.setdefault(section, {}).update(values)
    return Config.model_validate(declared)


def test_identical_modalities_register_to_subpixel(aligned_dataset: Path, tmp_path: Path) -> None:
    """The direction-convention pin. See the module docstring."""
    config = base_config(aligned_dataset / "data.yaml", tmp_path / "run")
    (summary,) = run_benchmark(config)
    assert summary.n_failed == 0
    assert summary.metrics["reg/mace"] < SUBPIXEL_PX
    assert summary.metrics["reg/epe_mean"] < SUBPIXEL_PX
    assert summary.metrics["reg/success_rate_3px"] == pytest.approx(1.0)


@pytest.mark.parametrize("factor", [2, 3])
def test_upsampling_does_not_inflate_the_reported_error(
    aligned_dataset: Path, tmp_path: Path, factor: int
) -> None:
    """The scale-plumbing pin. Keypoints found in an upsampled image are not in native pixels;
    if the runner failed to map them back, every metric would be inflated by ``factor`` and
    the P3-9 ablation would report interpolation as catastrophic."""
    config = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / f"run{factor}",
        preprocess={"moving_upsample": factor},
    )
    (summary,) = run_benchmark(config)
    assert summary.metrics["reg/mace"] < SUBPIXEL_PX


def test_swapping_which_modality_moves_gives_the_same_answer(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """With identical modalities the choice of moving side is a relabelling, so a difference
    here would mean the reference and moving paths are not doing the same thing."""
    forward = run_benchmark(base_config(aligned_dataset / "data.yaml", tmp_path / "fwd"))[0]
    reversed_ = run_benchmark(
        base_config(aligned_dataset / "data.yaml", tmp_path / "rev", gt={"moving": "optical"})
    )[0]
    assert reversed_.metrics["reg/mace"] == pytest.approx(forward.metrics["reg/mace"], rel=1e-9)


def test_the_runner_and_the_gt_command_sample_the_same_warps(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """Both derive H from ``(seed, index)`` rather than sharing a file, so they agree by
    construction -- but only as long as they agree on the *index*, which is what this pins."""
    from cmreg.cli import main

    run_dir = tmp_path / "gt"
    assert (
        main(
            [
                "gt",
                "--data",
                str(aligned_dataset / "data.yaml"),
                "--split",
                "val",
                "--run-dir",
                str(run_dir),
            ]
        )
        == 0
    )
    recorded = json.loads((run_dir / "gt_val.json").read_text())
    config = base_config(aligned_dataset / "data.yaml", tmp_path / "bench")
    for index, entry in enumerate(recorded["pairs"]):
        shape = (entry["height"], entry["width"])
        direct = sample_homography(config.gt, generator(config.gt.seed, index), shape)
        assert np.allclose(direct.ravel(), entry["homography"])


def test_every_pair_produces_a_row_even_when_matching_fails(
    dataset_root: Path, tmp_path: Path
) -> None:
    """TASKS.md X-4. The noise fixture is unmatchable by construction -- 64x80 of uniform
    random pixels -- so this exercises the failure path over a whole split."""
    from tests.conftest import VAL_STEMS

    config = base_config(dataset_root / "data.yaml", tmp_path / "noise")
    (summary,) = run_benchmark(config)
    rows = read_rows(tmp_path / "noise")
    assert len(rows) == len(VAL_STEMS)
    assert summary.metrics["reg/n_pairs"] == len(VAL_STEMS)
    assert all(row.failure_reason is not None for row in rows if not row.success)


def test_a_run_leaves_a_parquet_file_and_a_config_snapshot(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """The snapshot -- not the authored YAML -- is what the aggregator reads, so a result can
    always be traced back to the configuration that produced it."""
    run_dir = tmp_path / "artifacts"
    config = base_config(aligned_dataset / "data.yaml", run_dir)
    run_benchmark(config)
    assert (run_dir / FILENAME).is_file()
    assert (run_dir / "config.yaml").is_file()
    rows = read_rows(run_dir)
    assert {row.config_hash for row in rows} == {config.config_hash()}


def test_multiple_matchers_share_one_parquet_and_are_distinguished_by_column(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    config = base_config(
        aligned_dataset / "data.yaml", tmp_path / "multi", match={"matchers": ["sift", "orb"]}
    )
    summaries = run_benchmark(config)
    assert len(summaries) == 2
    assert {row.matcher for row in read_rows(tmp_path / "multi")} == {"sift", "orb"}


def test_a_limit_caps_the_pairs(aligned_dataset: Path, tmp_path: Path) -> None:
    config = base_config(aligned_dataset / "data.yaml", tmp_path / "capped", data={"limit": 1})
    (summary,) = run_benchmark(config)
    assert summary.n_pairs == 1


def test_an_empty_split_fails_loudly(tmp_path: Path) -> None:
    import yaml

    root = tmp_path / "empty"
    for split in ("train", "val"):
        (root / split / "optical" / "images").mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/optical/images",
                "val": "val/optical/images",
                "rgbt": {"optical_token": "optical", "thermal_token": "thermal"},
            }
        )
    )
    with pytest.raises(RunnerError, match="no images"):
        run_benchmark(base_config(root / "data.yaml", tmp_path / "run"))


def _scientific(metrics: dict[str, float]) -> dict[str, float]:
    """Everything but wall-clock. `time/*` varies run to run by definition and comparing it
    would make a determinism test a flaky benchmark of the machine it runs on."""
    return {key: value for key, value in metrics.items() if not key.startswith("time/")}


def _register_stochastic_matcher(name: str) -> None:
    """A matcher that draws its correspondences from torch's ambient RNG.

    Stands in for RoMa, whose ``sample`` uses ``torch.multinomial``. The real thing needs
    weights and a network; the property under test is the runner's seeding contract, not the
    matcher, so a two-line stand-in tests it in milliseconds instead of minutes.
    """
    import torch

    from cmreg.matchers import MatchResult, register

    class _Stochastic:
        def __init__(self, config, device) -> None:
            del config, device

        @property
        def name(self) -> str:
            return name

        def __call__(self, image0, image1) -> MatchResult:
            height, width = image0.shape
            picks = torch.rand(64, 2) * torch.tensor([width - 1.0, height - 1.0])
            kpts = picks.numpy().astype(np.float64)
            return MatchResult(
                kpts0=kpts,
                kpts1=kpts,
                confidence=None,
                n_detected0=None,
                n_detected1=None,
                extract_ms=0.0,
                match_ms=0.0,
            )

    register(name, _Stochastic)


def test_a_stochastic_matcher_gives_the_same_rows_on_every_run(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """Regression pin for the P0-2 finding: with nothing seeding torch, two runs of an
    identical config over identical pairs gave ``reg/mace`` 121.2 and 43.7 on the same eight
    MSRS pairs. Dense matchers sample their correspondences, so an ambient RNG makes the whole
    benchmark unreproducible for exactly the matchers the project is about."""
    _register_stochastic_matcher("_stochastic_repeat")
    manifest = aligned_dataset / "data.yaml"
    kwargs = {"match": {"matchers": ["_stochastic_repeat"]}}
    first = run_benchmark(base_config(manifest, tmp_path / "a", **kwargs))[0]
    second = run_benchmark(base_config(manifest, tmp_path / "b", **kwargs))[0]
    assert _scientific(first.metrics) == _scientific(second.metrics)


def test_a_cell_does_not_depend_on_which_matchers_share_its_config(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """The loop is pairs-outer/matchers-inner, so a single per-run seed would leave each
    matcher drawing from whatever the previous one left behind -- and RoMa's number in the
    P3-7 grid would disagree with a single-matcher rerun of the same cell. `seed_cell` keys on
    ``(seed, index, matcher)`` precisely so the results store's one-row-per-cell model holds."""
    _register_stochastic_matcher("_stochastic_alone")
    manifest = aligned_dataset / "data.yaml"
    alone = run_benchmark(
        base_config(manifest, tmp_path / "alone", match={"matchers": ["_stochastic_alone"]})
    )[0]
    _, accompanied = run_benchmark(
        base_config(manifest, tmp_path / "with", match={"matchers": ["sift", "_stochastic_alone"]})
    )
    assert _scientific(accompanied.metrics) == _scientific(alone.metrics)


# --- TASKS.md P1-1b: the alignment audit and the mono-modal control -----------------------

# Every `gt` range at its neutral element, so `sample_homography` returns exactly I and the
# recovered homography is the pair's own residual misalignment. This is what
# `experiments/p1_alignment_audit.yaml` encodes; duplicated here so the test fails if the
# meaning of "identity warp" ever drifts from what that config assumes.
IDENTITY_WARP = {
    "rotation_deg": 0.0,
    "scale_min": 1.0,
    "scale_max": 1.0,
    "perspective": 0.0,
    "translation": 0.0,
}


def test_the_identity_warp_audit_recovers_a_known_camera_offset(
    offset_dataset: Path, tmp_path: Path
) -> None:
    """The pin for P1-1a's *method*, which nothing previously asserted.

    P1-1a concluded that MSRS/FLIR/LLVIP carry a 4-6 px residual by running this cell with a
    zero-magnitude warp and reading `corner_error(estimate.h, I)` as the dataset's own
    misalignment. That reading is only sound if the cell actually reports a known offset as
    that offset -- so here the fixture's rig displacement is known to the pixel, and the run
    has to return it.
    """
    from tests.conftest import OFFSET_PX

    config = base_config(offset_dataset / "data.yaml", tmp_path / "audit", gt=IDENTITY_WARP)
    (summary,) = run_benchmark(config)
    assert summary.n_failed == 0
    assert summary.metrics["reg/mace"] == pytest.approx(float(np.hypot(*OFFSET_PX)), abs=0.1)


def test_the_monomodal_control_is_subpixel_where_the_cross_modal_cell_is_not(
    offset_dataset: Path, tmp_path: Path
) -> None:
    """TASKS.md P1-1b's control, and the pin that ``gt.reference`` changes which file is read.

    Same pairs, same warp, one field apart: cross-modal carries the fixture's rig offset,
    mono-modal cannot, because both sides come from the one camera. A `reference` field that
    was accepted but ignored would make these two numbers equal.
    """
    from tests.conftest import OFFSET_PX

    manifest = offset_dataset / "data.yaml"
    cross = run_benchmark(base_config(manifest, tmp_path / "cross", gt=IDENTITY_WARP))[0]
    mono = run_benchmark(
        base_config(
            manifest,
            tmp_path / "mono",
            gt={**IDENTITY_WARP, "reference": "optical", "moving": "optical"},
        )
    )[0]
    assert cross.metrics["reg/mace"] == pytest.approx(float(np.hypot(*OFFSET_PX)), abs=0.1)
    assert mono.metrics["reg/mace"] < SUBPIXEL_PX


def test_the_monomodal_control_survives_a_real_warp(offset_dataset: Path, tmp_path: Path) -> None:
    """The control as it is actually run: at the benchmark's own Tier-1 warp, not at identity.

    At identity a mono-modal pair is byte-identical and the cell measures nothing. The
    question P1-1b asks of it -- is the pipeline sub-pixel-capable at the operating point every
    benchmark row uses? -- only has an answer under a real warp.
    """
    config = base_config(
        offset_dataset / "data.yaml", tmp_path / "warped", gt={"reference": "thermal"}
    )
    assert config.gt.is_monomodal
    (summary,) = run_benchmark(config)
    assert summary.metrics["reg/mace"] < SUBPIXEL_PX


def test_rows_carry_the_shape_and_the_estimated_homography(
    offset_dataset: Path, tmp_path: Path
) -> None:
    """Both are additions the residual decomposition cannot work without, and both are null
    exactly when a row failed."""
    from cmreg.metrics import corner_error

    config = base_config(offset_dataset / "data.yaml", tmp_path / "cols", gt=IDENTITY_WARP)
    (summary,) = run_benchmark(config)
    rows = read_rows(tmp_path / "cols")
    assert summary.n_failed == 0
    for row in rows:
        assert (row.height, row.width) == (240, 320)
        assert row.h is not None and row.height is not None and row.width is not None
        # The stored matrix has to reproduce the row's own corner error, or the column is
        # recording something other than the fit that was scored.
        recovered = corner_error(
            np.asarray(row.h).reshape(3, 3), np.eye(3), (row.height, row.width)
        )
        assert row.corner_err == pytest.approx(recovered, abs=1e-6)


def test_a_subsampled_run_reproduces_the_full_run_row_for_row(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """A subsample must be a *sample* of the full run, not a different experiment.

    The whole risk of TASKS.md P3-1's random-subsample path is here: the pair's index seeds its
    synthetic warp, so re-numbering the drawn pairs 0..N-1 would hand the same image a
    different warp than the full-split run it is supposed to sample from, and the 300-pair
    Stage-A cell would not be comparable with anything. Scoring identical `corner_err` per stem
    across the two runs is the only check that catches that.
    """
    full = base_config(aligned_dataset / "data.yaml", tmp_path / "full")
    run_benchmark(full)
    sampled = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / "sampled",
        data={"limit": 2, "subsample_seed": 0},
    )
    (summary,) = run_benchmark(sampled)
    assert summary.n_pairs == 2

    rows = read_rows(tmp_path / "full")
    reference = {row.stem: row.corner_err for row in rows}
    sampled_rows = read_rows(tmp_path / "sampled")
    head = [row.stem for row in rows][:2]
    assert [row.stem for row in sampled_rows] != head, "a head slice would pass this vacuously"
    for row in sampled_rows:
        assert row.stem in reference
        assert row.corner_err == pytest.approx(reference[row.stem])


def _register_raising_matcher(name: str) -> None:
    """A matcher that raises on every pair, standing in for the many ways a backend can."""
    from cmreg.matchers import register

    class _Raises:
        def __init__(self, config, device) -> None:
            del config, device

        @property
        def name(self) -> str:
            return name

        def __call__(self, image0, image1):
            del image0, image1
            raise RuntimeError("backend exploded")

    register(name, _Raises)


def test_a_matcher_that_raises_costs_one_row_not_the_whole_run(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """The P3-7 server crash cost 50 scored pairs x 20 matchers, because rows are written
    only after the last pair. A backend blowing up is a property of that cell; X-4 says the
    hard cases are rows, and the matchers sharing the run must still produce theirs."""
    _register_raising_matcher("_raises")
    run_dir = tmp_path / "raised"
    config = base_config(
        aligned_dataset / "data.yaml", run_dir, match={"matchers": ["sift", "_raises"]}
    )
    summaries = run_benchmark(config)

    rows = read_rows(run_dir / FILENAME)
    raised = [row for row in rows if row.matcher == "_raises"]
    assert raised, "the failing matcher must still be represented"
    assert all(row.failure_reason == "matcher_raised" for row in raised)
    assert all(not row.success for row in raised)
    # The shape is known here -- the pair decoded, the matcher is what failed.
    assert all(row.height is not None and row.width is not None for row in raised)
    sift = next(summary for summary in summaries if summary.context["matcher"] == "sift")
    assert sift.metrics["reg/success_rate_5px"] > 0.0, "the healthy matcher kept its results"
