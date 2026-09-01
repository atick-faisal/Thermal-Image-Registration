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
from cmreg.eval.runner import _variant_label
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
@pytest.mark.parametrize("kernel", ["nearest", "bilinear", "bicubic", "lanczos"])
def test_upsampling_does_not_inflate_the_reported_error(
    aligned_dataset: Path, tmp_path: Path, factor: int, kernel: str
) -> None:
    """The scale-plumbing pin. Keypoints found in an upsampled image are not in native pixels;
    if the runner failed to map them back, every metric would be inflated by ``factor`` and
    the P3-9 ablation would report interpolation as catastrophic.

    Run across all four kernels because stage C varies them: the mapping is
    ``(j + 0.5) / scale - 0.5`` for every one of them, so a kernel that failed here would mean
    the resampler and the coordinate convention disagree rather than that the kernel is poor.
    """
    config = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / f"run{factor}{kernel}",
        preprocess={"moving_upsample": factor, "moving_interpolation": kernel},
    )
    (summary,) = run_benchmark(config)
    assert summary.metrics["reg/mace"] < SUBPIXEL_PX


@pytest.mark.parametrize("factor", [0.5, 0.25])
def test_a_symmetric_resize_does_not_inflate_the_reported_error(
    aligned_dataset: Path, tmp_path: Path, factor: float
) -> None:
    """P3-12a's load-bearing test, and the reason stage F could not be launched before it.

    The reference side carried ``scale=1.0`` unconditionally until this axis existed, so the
    runner mapped only the *moving* keypoints back to native pixels. Resize both sides and leave
    that half-mapping in place and the estimate is fitted from resized reference points to
    native moving points -- a ``1/factor`` scale error in the recovered warp, which on this
    240x320 fixture is tens of pixels and reads as the matcher having failed rather than as a
    plumbing bug. Asserted against the axis's own soft floor (~``1/factor`` native pixels, since
    a matcher localises to at best a scaled pixel), which is two orders of magnitude below what
    a one-sided mapping produces.
    """
    config = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / f"run{factor}",
        preprocess={"input_scale": factor},
    )
    (summary,) = run_benchmark(config)
    assert summary.n_failed == 0
    assert summary.metrics["reg/mace"] < SUBPIXEL_PX / factor


def test_the_resolution_axis_leaves_the_metrics_in_native_pixels(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """What makes the levels of stage F comparable at all: the ground truth, the truth matrix
    and the reporting ladder are the pair's *native* frame at every level, so `height`/`width`
    and the 3/5/10/20 px columns mean one thing across the axis. Only the images the matcher
    sees shrink."""
    config = base_config(
        aligned_dataset / "data.yaml", tmp_path / "native", preprocess={"input_scale": 0.5}
    )
    run_benchmark(config)
    rows = read_rows(tmp_path / "native" / FILENAME)
    assert {(row.height, row.width) for row in rows} == {(240, 320)}
    assert {row.input_scale for row in rows} == {0.5}


def test_the_default_resolution_level_reproduces_a_pre_axis_run_row_for_row(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """P3-12a is additive (X-5). `input_scale` 1.0 resizes nothing, so a run that sets it
    explicitly must be indistinguishable from one written before the field existed -- which is
    what lets every stage A-E directory on the server keep the numbers it was scored under."""
    implicit = run_benchmark(base_config(aligned_dataset / "data.yaml", tmp_path / "implicit"))[0]
    explicit = run_benchmark(
        base_config(
            aligned_dataset / "data.yaml", tmp_path / "explicit", preprocess={"input_scale": 1.0}
        )
    )[0]
    assert implicit.metrics["reg/mace"] == explicit.metrics["reg/mace"]
    assert implicit.metrics["reg/epe_mean"] == explicit.metrics["reg/epe_mean"]


@pytest.mark.parametrize(
    ("scale", "expected"),
    [(1.0, "none-none-x1-magsac"), (0.5, "none-none-x1-r0.5-magsac")],
)
def test_the_resolution_level_names_a_cell_only_where_it_acts(
    aligned_dataset: Path, tmp_path: Path, scale: float, expected: str
) -> None:
    """The rule the kernel, the P3-10 threshold and the P3-4a warp model already follow: an
    axis appears in a W&B run name only where it is varied. Naming it unconditionally would
    rename every stage A-E run, and a project whose run names drift between stages is one
    nobody can read across them."""
    config = base_config(
        aligned_dataset / "data.yaml", tmp_path / f"name{scale}", preprocess={"input_scale": scale}
    )
    assert _variant_label(config, config.estimate) == expected


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(1, "none-none-x1-magsac"), (2, "none-none-x2-lanczos-magsac")],
)
def test_the_interpolation_kernel_names_a_cell_only_where_it_acts(
    aligned_dataset: Path, tmp_path: Path, factor: int, expected: str
) -> None:
    """The W&B run name is derived, so two cells differing only in kernel must not collide
    under one name (X-2). At x1 they cannot differ at all -- ``upsample`` returns the input
    untouched -- so the kernel is omitted there, which is what keeps every stage-A and stage-B
    run name (all x1) unchanged by this."""
    config = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / f"run{factor}",
        preprocess={"moving_upsample": factor, "moving_interpolation": "lanczos"},
    )
    assert _variant_label(config, config.estimate) == expected


def test_the_threshold_names_a_cell_only_where_it_is_swept(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """The same rule as the kernel above, for P3-10's second axis.

    Stages A-C hold the threshold fixed, so naming it there would rename every run already on
    the server. Stage D sweeps it, and without it the three thresholds of one estimator would
    collide into a single W&B run name (X-2).
    """
    unswept = base_config(aligned_dataset / "data.yaml", tmp_path / "unswept")
    assert _variant_label(unswept, unswept.estimate) == "none-none-x1-magsac"

    swept = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / "swept",
        estimate={"sweep_methods": ["magsac", "lmeds"], "sweep_thresholds_px": [1.0, 3.0]},
    )
    assert [_variant_label(swept, v) for v in swept.estimate.variants()] == [
        "none-none-x1-magsac@1px",
        "none-none-x1-magsac@3px",
        "none-none-x1-lmeds@1px",
        "none-none-x1-lmeds@3px",
    ]


def test_sweeping_only_the_estimator_leaves_the_threshold_out_of_the_name(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """An axis appears in the label only where it is *varied*; the threshold is constant here."""
    config = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / "run",
        estimate={"sweep_methods": ["magsac", "lmeds"]},
    )
    assert [_variant_label(config, v) for v in config.estimate.variants()] == [
        "none-none-x1-magsac",
        "none-none-x1-lmeds",
    ]


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


def _pair_shape(manifest: Path) -> tuple[int, int]:
    """The fixture's image shape, read rather than restated so the two cannot drift."""
    from cmreg.imaging import read_gray

    image = next(iter(sorted((manifest.parent / "val" / "optical" / "images").iterdir())))
    return read_gray(image).shape


def _offset_calibration(manifest: Path, path: Path, scale: float = 1.0) -> Path:
    """The `offset_dataset` fixture's rig displacement, written as a calibration constant.

    The fixture's docstring fixes the direction: optical `(x, y)` and thermal `(x - dx, y - dy)`
    are the same scene point, so the map from thermal into optical -- the direction the cell
    estimates -- is a translation by `(+dx, +dy)`. That translation displaces all four corners
    equally, which is what a corner field of four identical shifts means.
    """
    from cmreg.gt import ResidualCalibration, write_calibration
    from tests.conftest import OFFSET_PX

    shape = _pair_shape(manifest)
    dx, dy = (v * scale for v in OFFSET_PX)
    return write_calibration(
        ResidualCalibration(
            dataset=manifest.parent.name,
            height=shape[0],
            width=shape[1],
            corner_shift=((dx, dy),) * 4,
            matchers=("fixture",),
            spread_px=0.0,
            worst_case_px=0.0,
            n_pairs=4,
            split="val",
            git_sha="test",
            note="the offset_dataset fixture's known rig displacement",
        ),
        path,
    )


def test_composing_a_known_calibration_removes_the_offset(
    offset_dataset: Path, tmp_path: Path
) -> None:
    """**The direction pin for TASKS.md P2-12.** The mirror of the identity-warp audit above.

    That test asserts an uncomposed cell *reports* the fixture's rig offset. This one asserts
    that composing the very same offset as a calibration constant *removes* it: the truth
    becomes `R . inv(H_gt)` and a matcher that recovers the pair exactly should now score near
    zero rather than near `hypot(5, 3)`.

    It is the pin because the arithmetic has a plausible wrong answer that no unit test of the
    pieces would catch. Composing `R` in the opposite direction -- `inv(H_gt . R)` -- is equally
    well-typed, runs without error, and *doubles* the misalignment instead of removing it. The
    assertion on the wrong-direction magnitude below is what makes that failure loud.
    """
    from tests.conftest import OFFSET_PX

    manifest = offset_dataset / "data.yaml"
    calibration = _offset_calibration(manifest, tmp_path / "cal.json")

    plain = run_benchmark(base_config(manifest, tmp_path / "plain", gt=IDENTITY_WARP))[0]
    composed = run_benchmark(
        base_config(
            manifest,
            tmp_path / "composed",
            gt={**IDENTITY_WARP, "residual_calibration": str(calibration)},
        )
    )[0]

    offset = float(np.hypot(*OFFSET_PX))
    assert plain.metrics["reg/mace"] == pytest.approx(offset, abs=0.1)
    assert composed.metrics["reg/mace"] < SUBPIXEL_PX
    # And not merely smaller: composed the wrong way round the error would be ~2x the offset,
    # which is on the far side of `plain` rather than below it.
    assert composed.metrics["reg/mace"] < plain.metrics["reg/mace"]


def test_composing_survives_a_real_warp(offset_dataset: Path, tmp_path: Path) -> None:
    """The composition must hold under a non-trivial `H_gt`, not only under the identity.

    Under the identity warp `truth` is just `R`, so an implementation that ignored `H_gt`
    entirely would pass the test above. A full Tier-1 warp separates them: only
    `R . inv(H_gt)`, in that order, scores near zero here.
    """
    manifest = offset_dataset / "data.yaml"
    calibration = _offset_calibration(manifest, tmp_path / "cal.json")

    plain = run_benchmark(base_config(manifest, tmp_path / "plain"))[0]
    composed = run_benchmark(
        base_config(
            manifest,
            tmp_path / "composed",
            gt={"residual_calibration": str(calibration)},
        )
    )[0]
    assert composed.metrics["reg/mace"] < plain.metrics["reg/mace"]
    assert composed.metrics["reg/mace"] < SUBPIXEL_PX


def test_a_zero_calibration_is_a_no_op(offset_dataset: Path, tmp_path: Path) -> None:
    """Composing nothing must change nothing -- row for row, not merely on average.

    Without this, a bug that quietly perturbed every composed run would be invisible: the
    headline metric would still move in the right direction on the test above.
    """
    from cmreg.results import read_rows

    manifest = offset_dataset / "data.yaml"
    calibration = _offset_calibration(manifest, tmp_path / "cal.json", scale=0.0)

    run_benchmark(base_config(manifest, tmp_path / "plain", gt=IDENTITY_WARP))
    run_benchmark(
        base_config(
            manifest,
            tmp_path / "zero",
            gt={**IDENTITY_WARP, "residual_calibration": str(calibration)},
        )
    )
    plain = {row.stem: row.corner_err for row in read_rows(tmp_path / "plain")}
    zero = {row.stem: row.corner_err for row in read_rows(tmp_path / "zero")}
    assert plain.keys() == zero.keys()
    for stem, error in plain.items():
        assert zero[stem] == pytest.approx(error, abs=1e-6)


def test_the_row_records_which_constant_was_composed(offset_dataset: Path, tmp_path: Path) -> None:
    """X-2: a pasted table has to be traceable to the exact matrix that produced it, and the
    configured *path* cannot do that across two machines."""
    from cmreg.gt import load_calibration
    from cmreg.results import read_rows

    manifest = offset_dataset / "data.yaml"
    calibration = _offset_calibration(manifest, tmp_path / "cal.json")
    run_benchmark(
        base_config(
            manifest,
            tmp_path / "composed",
            gt={**IDENTITY_WARP, "residual_calibration": str(calibration)},
        )
    )
    digest = load_calibration(calibration).digest()
    rows = read_rows(tmp_path / "composed")
    assert {row.residual_calibration for row in rows} == {digest}

    run_benchmark(base_config(manifest, tmp_path / "plain", gt=IDENTITY_WARP))
    assert {row.residual_calibration for row in read_rows(tmp_path / "plain")} == {None}


def test_a_calibration_for_another_dataset_is_refused(offset_dataset: Path, tmp_path: Path) -> None:
    """A startup error, not a per-pair one: it is identical on all 300 pairs and should say so
    once rather than write 300 rows blaming the data."""
    manifest = offset_dataset / "data.yaml"
    path = _offset_calibration(manifest, tmp_path / "cal.json")
    wrong = json.loads(path.read_text())
    wrong["dataset"] = "somewhere-else"
    path.write_text(json.dumps(wrong))

    with pytest.raises(RunnerError, match="somewhere-else"):
        run_benchmark(
            base_config(
                manifest,
                tmp_path / "run",
                gt={**IDENTITY_WARP, "residual_calibration": str(path)},
            )
        )


def test_a_calibration_on_a_monomodal_control_is_refused(
    offset_dataset: Path, tmp_path: Path
) -> None:
    """The control's residual is zero by construction (P1-1b), so composing a constant there
    would manufacture the very misalignment the control exists to exclude."""
    manifest = offset_dataset / "data.yaml"
    calibration = _offset_calibration(manifest, tmp_path / "cal.json")
    with pytest.raises(RunnerError, match="mono-modal"):
        run_benchmark(
            base_config(
                manifest,
                tmp_path / "run",
                gt={
                    **IDENTITY_WARP,
                    "reference": "optical",
                    "moving": "optical",
                    "residual_calibration": str(calibration),
                },
            )
        )


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


def _register_confidenceless_matcher(name: str) -> None:
    """A matcher that finds real matches but scores none of them.

    Stands in for ``xfeat``, ``sift-nn`` and ``orb-nn``, the three vismatch backends that
    return no per-match confidence (TASKS.md P0-2) -- ``xfeat`` being in `experiments/GRID.md`
    §6's `reduced-8`, so stage D meets this on the server. Delegating to a real matcher rather
    than fabricating correspondences keeps the other eleven variants' rows meaningful.
    """
    from cmreg.matchers import MatchResult, get_matcher, register

    class _NoConfidence:
        def __init__(self, config, device) -> None:
            self._inner = get_matcher("sift", config, device)

        @property
        def name(self) -> str:
            return name

        def __call__(self, image0, image1):
            result = self._inner(image0, image1)
            return MatchResult(
                kpts0=result.kpts0,
                kpts1=result.kpts1,
                confidence=None,
                n_detected0=result.n_detected0,
                n_detected1=result.n_detected1,
                extract_ms=result.extract_ms,
                match_ms=result.match_ms,
            )

    register(name, _NoConfidence)


SWEPT_METHODS = ("magsac", "ransac", "lmeds", "prosac")
SWEPT_THRESHOLDS = (1.0, 5.0)


def test_a_swept_run_reproduces_single_estimator_runs_row_for_row(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """The integrity check for the whole of P3-10, and the reason the stage costs 4-5 h not 38.

    Stage D sweeps the estimator *inside* the pair loop -- one match, many fits -- on the claim
    that this is the same experiment as running each estimator separately. That claim is only
    true if OpenCV's solvers carry no state between calls, so variant *k* cannot depend on
    variants 1..k-1 having run. Nothing else in the suite would notice if they did: every
    swept table would look entirely plausible and every number in it would be wrong.

    Bit-identical, not approximately equal. An order dependence would show up as a small
    difference, which is exactly what ``pytest.approx`` would swallow.
    """
    swept = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / "swept",
        estimate={
            "sweep_methods": list(SWEPT_METHODS),
            "sweep_thresholds_px": list(SWEPT_THRESHOLDS),
            "threshold_px": SWEPT_THRESHOLDS[0],
        },
    )
    run_benchmark(swept)
    swept_rows = read_rows(tmp_path / "swept")

    for method in SWEPT_METHODS:
        for threshold in SWEPT_THRESHOLDS:
            single = base_config(
                aligned_dataset / "data.yaml",
                tmp_path / f"single_{method}_{threshold:g}",
                estimate={"method": method, "threshold_px": threshold},
            )
            run_benchmark(single)
            alone = {row.stem: row for row in read_rows(single.runtime.path)}
            inside = {
                row.stem: row
                for row in swept_rows
                if row.estimator == method and row.threshold_px == threshold
            }
            assert inside.keys() == alone.keys(), f"{method}@{threshold:g}px lost pairs"
            for stem, row in inside.items():
                reference = alone[stem]
                assert row.corner_err == reference.corner_err
                assert row.h == reference.h
                assert row.n_inliers == reference.n_inliers
                assert row.reproj_err == reference.reproj_err
                assert row.epe_mean == reference.epe_mean


def test_a_swept_warp_run_reproduces_single_model_runs_row_for_row(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """The same integrity check as above, for P3-4a's axis -- and it needs its own.

    The estimator sweep's claim rests on OpenCV's solvers carrying no state between calls. The
    warp sweep adds a second way to be wrong that the estimator sweep cannot exhibit: three
    *different solvers* now run off one match, and a dispatch that leaked a model between
    variants -- or lifted the wrong matrix into the shared tail -- would produce a table that
    looks entirely plausible. Bit-identical, so an order dependence cannot hide inside a
    tolerance.

    RANSAC throughout, because it is the one estimator all three models admit
    (`estimate/robust.py::SUPPORTED_ESTIMATORS`); MAGSAC would make the similarity column a hole
    and test nothing.
    """
    models = ("homography", "affine", "similarity")
    swept = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / "swept_warp",
        estimate={"sweep_warp_models": list(models), "method": "ransac"},
    )
    run_benchmark(swept)
    swept_rows = read_rows(tmp_path / "swept_warp")

    for model in models:
        single = base_config(
            aligned_dataset / "data.yaml",
            tmp_path / f"single_{model}",
            estimate={"warp_model": model, "method": "ransac"},
        )
        run_benchmark(single)
        alone = {row.stem: row for row in read_rows(single.runtime.path)}
        inside = {row.stem: row for row in swept_rows if row.warp == model}
        assert inside.keys() == alone.keys(), f"{model} lost pairs"
        for stem, row in inside.items():
            reference = alone[stem]
            assert row.corner_err == reference.corner_err
            assert row.h == reference.h
            assert row.n_inliers == reference.n_inliers
            assert row.epe_mean == reference.epe_mean


def test_the_three_warp_models_produce_different_fits(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """The axis reaches the solver. Without this, a sweep whose three columns were identical
    would be PLAN.md §15A's bug -- a swept knob not connected to what it names -- in stage E's
    shape, and every other assertion here would still pass."""
    config = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / "run",
        estimate={
            "sweep_warp_models": ["homography", "affine", "similarity"],
            "method": "ransac",
        },
    )
    run_benchmark(config)
    rows = [row for row in read_rows(tmp_path / "run") if row.success and row.h is not None]
    assert rows, "no successful fit to compare"
    by_stem: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        assert row.h is not None
        by_stem.setdefault(row.stem, {})[row.warp] = row.h
    fits = next(models for models in by_stem.values() if len(models) == 3)
    assert len({tuple(h) for h in fits.values()}) == 3
    # An affine and a similarity have no perspective term whatever the correspondences say.
    for model in ("affine", "similarity"):
        assert fits[model][6:] == [0.0, 0.0, 1.0]


def test_every_variant_scores_the_same_pairs(aligned_dataset: Path, tmp_path: Path) -> None:
    """A swept directory holds N equally-sized populations, not one pooled one.

    `reg/n_pairs` has to mean the same thing in every column of a stage-D table, so a failure
    -- an unreadable pair, a matcher that raised, an unsupported estimator -- is a row *per
    variant* rather than one row for the run.
    """
    config = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / "run",
        estimate={"sweep_methods": ["magsac", "lmeds"], "sweep_thresholds_px": [1.0, 3.0]},
    )
    summaries = run_benchmark(config)
    assert len(summaries) == 4, "one summary per (matcher, estimation variant)"
    assert len({summary.n_pairs for summary in summaries}) == 1
    rows = read_rows(tmp_path / "run")
    cells = {(row.estimator, row.threshold_px) for row in rows}
    assert cells == {("magsac", 1.0), ("magsac", 3.0), ("lmeds", 1.0), ("lmeds", 3.0)}


def test_an_estimator_the_matcher_cannot_support_is_rows_not_an_abort(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """PROSAC needs per-match confidences and three backends supply none (TASKS.md P0-2).

    For a single-estimator run, raising is right -- it fails identically on all 300 pairs. In a
    sweep it is not: aborting there discards eleven variants that ran fine, *after* the matching
    they depend on has already been paid for. So it is recorded, with a reason naming the cause,
    and the match counts stay honest because the matching really did happen.
    """
    _register_confidenceless_matcher("_no_confidence")
    run_dir = tmp_path / "gap"
    config = base_config(
        aligned_dataset / "data.yaml",
        run_dir,
        match={"matchers": ["_no_confidence"]},
        estimate={"sweep_methods": ["magsac", "prosac"]},
    )
    run_benchmark(config)

    rows = read_rows(run_dir)
    prosac = [row for row in rows if row.estimator == "prosac"]
    magsac = [row for row in rows if row.estimator == "magsac"]
    assert prosac and len(prosac) == len(magsac), "the unsupported cell is still a population"
    assert all(row.failure_reason == "estimator_needs_confidence" for row in prosac)
    assert all(not row.success for row in prosac)
    # The matching happened and cost time; only the estimate is missing. Zeroing these would
    # read as a matcher that found nothing, which is a different failure entirely.
    assert all(row.n_matches > 0 for row in prosac)
    assert [row.n_matches for row in prosac] == [row.n_matches for row in magsac]
    assert any(row.success for row in magsac), "the supported variant is unaffected"


def test_a_warp_model_the_estimator_cannot_fit_is_rows_not_an_abort(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """The same policy for P3-4a's gap, whose cause is different and whose token says so.

    MAGSAC cannot fit a 4-DoF similarity in opencv 5.0.0 at all. That is a property of the
    (model, estimator) pair alone, not of the matcher, so stage E's grid genuinely has holes --
    and a hole recorded as `estimator_unsupported_for_warp` is one a table can name (X-4), where
    an abort would discard the two models that fitted fine off the same matching.
    """
    run_dir = tmp_path / "warp_gap"
    config = base_config(
        aligned_dataset / "data.yaml",
        run_dir,
        estimate={"sweep_warp_models": ["homography", "similarity"], "method": "magsac"},
    )
    run_benchmark(config)

    rows = read_rows(run_dir)
    similarity = [row for row in rows if row.warp == "similarity"]
    homography = [row for row in rows if row.warp == "homography"]
    assert similarity and len(similarity) == len(homography)
    assert all(row.failure_reason == "estimator_unsupported_for_warp" for row in similarity)
    assert all(not row.success for row in similarity)
    # The matching happened; only the fit is missing. Same distinction as the PROSAC gap above.
    assert [row.n_matches for row in similarity] == [row.n_matches for row in homography]
    assert any(row.success for row in homography), "the supported model is unaffected"


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
