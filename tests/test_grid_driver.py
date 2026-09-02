"""The shared stage machinery (`scripts/stages.py`) and each stage driver's grid.

Only the guard is tested here. The rest of the driver is a `cmreg bench` invocation and a
table renderer, both covered where they live -- but the guard is the one piece whose failure
is *silent*, and it has already cost one server trip: stage-A run 2 asked for `flir` composed,
found run 1's pre-composition rows on disk, skipped the cell, and printed the old floors under
the new banner. Nothing in the pasted console said so.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from pathlib import Path

import pytest

import p3_stageb_polarity  # on the path via `[tool.pytest.ini_options] pythonpath`
import p3c_upsample
import p3d_estimator
import p3e_warp
import p3f_resolution
import p3g_matchcount
import stages
from cmreg.metrics.schema import (
    EPE_MEDIAN,
    FAILURE_RATE,
    MATCH_INLIER_RATIO,
    TIME_ESTIMATE_MS,
)
from cmreg.results import PairRow, write_rows
from tests.test_results import make_row


def _run_dir(tmp_path: Path, digest: str | None) -> Path:
    write_rows([make_row("0", residual_calibration=digest)], tmp_path)
    return tmp_path


# The hash `tests.test_results.make_row` stamps on every row it builds.
ROW_HASH = "0123456789abcdef"


def test_resuming_onto_a_differently_composed_run_is_refused(tmp_path: Path) -> None:
    """The regression: rows carrying a constant, under a cell that no longer composes.

    Reported separately from the general `config_hash` mismatch below even though the
    constant's path is inside that hash, because this is the one whose fix an operator has to
    be told -- produce the constant, or stop asking for it.
    """
    with pytest.raises(SystemExit, match="was scored with residual_calibration"):
        stages.refuse_a_stale_run(_run_dir(tmp_path, "210142740cdba163"), ROW_HASH, None)


def test_resuming_onto_a_matching_run_is_permitted(tmp_path: Path) -> None:
    """The reason the skip exists at all: an interrupted run must still pick up where it
    stopped, or a crashed 20-matcher cell costs the whole grid."""
    stages.refuse_a_stale_run(_run_dir(tmp_path, None), ROW_HASH, None)


def test_resuming_onto_a_differently_configured_run_is_refused(tmp_path: Path) -> None:
    """The general case the calibration check was a special case of.

    Stage B (P3-8) varies the preprocess pair, which lives inside `config_hash` and nowhere in
    a printed table: rows scored under `none/percentile` tabulate under an `invert/percentile`
    banner exactly as run 2's uncomposed rows tabulated under a composed one.
    """
    with pytest.raises(SystemExit, match=f"config_hash {ROW_HASH}"):
        stages.refuse_a_stale_run(_run_dir(tmp_path, None), "ffffffffffffffff", None)


def test_the_intended_hash_is_the_one_the_run_would_stamp(tmp_path: Path) -> None:
    """`intended_hash` resolves the config through `cmreg`'s own parser, so it must agree with
    `Config.load` on the same overrides -- and must move when a scientific field moves."""
    from cmreg.config import Config

    argv = ["bench", "-c", stages.CONFIG, "--preprocess-ref", "none"]
    assert (
        stages.intended_hash(argv)
        == Config.load(Path(stages.CONFIG), {"preprocess": {"reference": "none"}}).config_hash()
    )
    assert stages.intended_hash(argv) != stages.intended_hash(
        ["bench", "-c", stages.CONFIG, "--preprocess-ref", "invert"]
    )
    # `runtime` is excluded from the hash, so a relabelled cell is the same experiment.
    assert stages.intended_hash(argv) == stages.intended_hash(
        [*argv, "--group", "p3b_polarity_msrs", "--name", "stageb_msrs_neither"]
    )


class TestStageB:
    """`scripts/p3_stageb_polarity.py` (P3-8). Two things are tested here and nothing else: the
    cell enumeration, because a collision would resume-skip half the stage in silence, and the
    two summary blocks, because their arithmetic *is* the finding and lives nowhere else."""

    def test_the_grid_is_sixteen_cells_with_distinct_directories(self) -> None:
        dirs = [
            p3_stageb_polarity.run_dir_for(cell, polarity)
            for cell in stages.CELLS
            for polarity in p3_stageb_polarity.POLARITIES
        ]
        assert len(dirs) == 16
        assert len(set(dirs)) == 16

    def test_each_relative_polarity_has_two_recipes(self) -> None:
        """The design (GRID.md §6): the 2x2 is two recipes per relative polarity, differing
        only in which side carries the inversion. A polarity list that lost that symmetry
        would leave the `within` column of the summary block meaningless."""
        flipped = [p for p in p3_stageb_polarity.POLARITIES if p.flipped]
        same = [p for p in p3_stageb_polarity.POLARITIES if not p.flipped]
        assert len(flipped) == len(same) == 2
        assert {p.label for p in flipped} == {"optical", "thermal"}

    def test_a_purely_relative_effect_is_reported_as_one(self) -> None:
        """Rows built so that only the relative polarity matters: the two flipped recipes agree
        to the digit, the two unflipped ones agree, and the pairs differ. The block must report
        ~0 within and a large ratio -- the reading that says 'invert the optical grayscale' is a
        statement about the pair, not about the optical image."""
        table = _polarity_table({"optical": 5.0, "thermal": 5.0, "neither": 25.0, "both": 25.0})
        block = p3_stageb_polarity.relative_polarity_block({"msrs": table})
        row = _row_for("msrs", block)
        assert row[1:4] == ["0.00", "0.00", "20.00"]
        assert row[4] == "inf"

    def test_a_side_specific_effect_is_not_reported_as_relative(self) -> None:
        """The alternative hypothesis, which the 2x2 exists to separate: inverting the optical
        side helps and inverting the thermal side does not, so the two flipped recipes disagree
        as much as the pairs do and the ratio lands near 1."""
        table = _polarity_table({"optical": 5.0, "thermal": 25.0, "neither": 25.0, "both": 25.0})
        row = _row_for("msrs", p3_stageb_polarity.relative_polarity_block({"msrs": table}))
        assert row[1:4] == ["20.00", "0.00", "20.00"]
        assert row[4] == "1.00"

    def test_generality_counts_the_matchers_the_anchor_helps(self) -> None:
        """PLAN.md Figure 6 is a count across matchers, so a matcher the inversion *hurts* must
        not be averaged away by one it helps a great deal."""
        table = {
            "optical": {"roma": _metrics(5.0), "sift": _metrics(30.0)},
            "neither": {"roma": _metrics(25.0), "sift": _metrics(20.0)},
        }
        row = _row_for("msrs", p3_stageb_polarity.generality_block({"msrs": table}))
        assert row[1] == "1/2"  # roma improved by 20 px, sift got 10 px worse
        assert row[2] == "5.00"  # median of [+20, -10]


def _metrics(mace: float, success: float = 0.5) -> dict[str, float]:
    return {
        p3_stageb_polarity.HEADLINE: mace,
        p3_stageb_polarity.SECONDARY: success,
        "reg/n_pairs": 300.0,
    }


def _polarity_table(mace: dict[str, float]) -> p3_stageb_polarity.Table:
    return {label: {"roma": _metrics(value)} for label, value in mace.items()}


def _row_for(dataset: str, block: str) -> list[str]:
    (line,) = [row for row in block.splitlines() if row.startswith(dataset)]
    return line.split()


class TestStageC:
    """`scripts/p3c_upsample.py` (P3-9). Same two things as stage B -- the cell enumeration and
    the arithmetic of the summary blocks -- plus the matcher policy, which is the one part of
    this stage a reader of the pasted table cannot check for themselves."""

    def test_the_grid_is_thirteen_cells_per_dataset_with_distinct_directories(self) -> None:
        settings = p3c_upsample.grid(p3c_upsample.FACTORS, p3c_upsample.KERNELS)
        assert len(settings) == 13
        dirs = [
            p3c_upsample.run_dir_for(cell, setting) for cell in stages.CELLS for setting in settings
        ]
        assert len(set(dirs)) == len(dirs)

    def test_the_x1_cell_is_shared_across_every_kernel(self) -> None:
        """The collapse is exact -- at x1 `preprocess.upsample` returns the input untouched, so
        four kernel cells would be four copies of one run. If it were ever enumerated per
        kernel, the stage would pay for three redundant cells and W&B would hold four runs
        claiming different recipes for identical images."""
        settings = p3c_upsample.grid(p3c_upsample.FACTORS, p3c_upsample.KERNELS)
        assert len([s for s in settings if s.factor == 1]) == 1
        for kernel in p3c_upsample.KERNELS:
            columns = p3c_upsample.columns_for(kernel, settings)
            assert columns[0].label == "x1"

    def test_only_the_anchor_kernel_carries_the_resolution_blind_matchers(self) -> None:
        """The design's load-bearing claim (module docstring): `roma`, `minima-roma`,
        `matchanything-roma` and `superpoint-lightglue` resize their inputs internally, so a
        kernel row for them measures a resample prefilter and costs ~93% of the stage."""
        anchor = p3c_upsample.Setting(2, p3c_upsample.ANCHOR_KERNEL)
        other = p3c_upsample.Setting(2, "nearest")
        assert set(anchor.matchers) == set(stages.REDUCED_8)
        assert set(other.matchers) == set(p3c_upsample.RESPONSIVE)
        assert not set(other.matchers) & set(p3c_upsample.RESOLUTION_BLIND)

    def test_xoftr_is_dropped_above_its_architectural_ceiling(self) -> None:
        """XoFTR's positional encoding caps at 2048 px, so x4 on a 640-wide set raises. Dropped
        rather than left to fail 300 times -- and named in `excluded`, because X-4 makes this a
        recorded exclusion rather than a silent one."""
        assert "xoftr" in p3c_upsample.Setting(3, "nearest").matchers
        assert "xoftr" not in p3c_upsample.Setting(4, "nearest").matchers
        assert p3c_upsample.Setting(4, "nearest").excluded == ("xoftr",)

    def test_the_recipe_block_counts_the_matchers_upsampling_helps(self) -> None:
        """PLAN.md §15B's claim is a count across matchers, exactly as Figure 6's was: one
        matcher helped a great deal must not average away one it hurts."""
        table = {
            "x1": {"roma": _metrics(25.0), "sift": _metrics(20.0)},
            "x3_bicubic": {"roma": _metrics(5.0), "sift": _metrics(30.0)},
        }
        row = _row_for("flir", p3c_upsample.recipe_block({"flir": table}))
        assert row[1] == "1/2"  # roma improved by 20 px, sift got 10 px worse
        assert row[2] == "5.00"  # median of [+20, -10]

    def test_a_factor_effect_with_no_kernel_effect_reports_a_large_ratio(self) -> None:
        """The reading the block exists for: if every kernel gives the same number and the
        factor moves it, the kernel is a free choice for stages D-G. Built so the kernels agree
        exactly at each factor and the factors differ."""
        block = p3c_upsample.axis_block({"flir": _upsample_table({2: 10.0, 3: 30.0})}, _SETTINGS)
        row = _row_for("flir", block)
        assert row[1] == "0.00"  # every kernel agrees at a fixed factor
        assert row[2] == "20.00"  # x1 20 -> x2 10 -> x3 30 spans 20 px
        assert row[3] == "inf"

    def test_a_kernel_effect_is_not_reported_as_a_factor_effect(self) -> None:
        """The alternative the block has to separate: the kernels disagree as much as the
        factors do, so the ratio lands near 1 and the paper has to report the kernel."""
        table = _upsample_table({2: 10.0, 3: 30.0}, kernel_offsets={"nearest": 20.0})
        row = _row_for("flir", p3c_upsample.axis_block({"flir": table}, _SETTINGS))
        assert row[1] == "20.00"
        assert row[3] == "1.00"


_SETTINGS = p3c_upsample.grid((1, 2, 3), p3c_upsample.KERNELS)


def _upsample_table(
    mace: dict[int, float], kernel_offsets: dict[str, float] | None = None
) -> p3c_upsample.Table:
    """One matcher, `reg/mace` keyed by factor, optionally shifted for named kernels. Only
    `eloftr` is populated: it is in `RESPONSIVE`, which is the population `axis_block` reads."""
    offsets = kernel_offsets or {}
    table: p3c_upsample.Table = {"x1": {"eloftr": _metrics(20.0)}}
    for setting in _SETTINGS:
        if setting.factor > 1:
            value = mace[setting.factor] + offsets.get(setting.kernel, 0.0)
            table[setting.label] = {"eloftr": _metrics(value)}
    return table


class TestStageD:
    """`scripts/p3d_estimator.py` (P3-10). The stage that does not have one cell per row.

    Its grid is 10 `cmreg bench` invocations carrying 12 estimation variants each, so the two
    things worth pinning are the arithmetic that makes that legitimate (the invocation really
    does resolve to twelve cells, over the reduced-8 list, with the printed anchor among them)
    and the integrity block, which is the only check that the swept knob reaches the solver.
    """

    def test_the_stage_is_ten_match_passes_carrying_twelve_variants_each(self) -> None:
        """The cost model: `experiments/GRID.md` §6 froze 120 invocations at ~38 h, and the
        estimator being downstream of the matcher makes it 10 at ~4-5 h. If this ever becomes
        one invocation per variant again, the stage silently costs eight times what it should.
        """
        passes = [
            p3d_estimator.run_dir_for(cell, seed)
            for cell in stages.CELLS
            if cell.dataset in p3d_estimator.DATASETS
            for seed in p3d_estimator.SEEDS
        ]
        assert len(passes) == 10
        assert len(set(passes)) == len(passes)
        assert len(p3d_estimator.ESTIMATORS) * len(p3d_estimator.THRESHOLDS) == 12

    def test_the_invocation_resolves_to_the_twelve_variants_over_reduced_eight(self) -> None:
        """Resolved through `cmreg`'s own parser, not rebuilt here: a second path to the answer
        is a second path that can disagree with the run it is supposed to describe. Also pins
        that the anchor -- the variant whose console block the runner prints -- is one of the
        swept cells, which is the condition `EstimateConfig` enforces at load."""
        from cmreg.cli import build_parser, overrides_from_args
        from cmreg.config import Config

        cell = next(c for c in stages.CELLS if c.dataset == "flir")
        argv = _argv_for(cell)
        config = Config.load(
            build_parser().parse_args(argv).config,
            overrides_from_args(build_parser().parse_args(argv)),
        )
        variants = config.estimate.variants()
        assert len(variants) == 12
        assert config.match.matchers == stages.REDUCED_8
        assert (config.estimate.method.value, config.estimate.threshold_px) == p3d_estimator.ANCHOR
        assert p3d_estimator.ANCHOR in [(v.method.value, v.threshold_px) for v in variants]

    def test_the_seed_axis_is_the_five_x3_asks_for(self) -> None:
        """GRID.md §5's *reason* for five seeds was wrong -- OpenCV's estimators are
        deterministic, so the estimator contributes no variance -- but the five stay, because
        the warp draw and the matcher's sampling do, and X-3 wants that interval before an
        "A beats B" claim enters the paper."""
        assert len(p3d_estimator.SEEDS) == 5

    def test_the_integrity_block_passes_on_a_well_behaved_sweep(self) -> None:
        """lmeds identical across thresholds, everything else moving: the expected shape."""
        block = p3d_estimator.integrity_block({"flir": _estimator_rows()})
        assert "**FAIL**" not in block
        for estimator in p3d_estimator.ESTIMATORS:
            assert _verdict_for(block, estimator) == "PASS", estimator

    def test_a_threshold_that_reaches_nothing_is_caught(self) -> None:
        """PLAN.md §15A's bug, in this stage's shape: the upstream harness put its confidence
        into `cv2.findHomography`'s `mask` slot, so the knob it reported sweeping was never
        applied. Here that would look like *every* estimator being flat across thresholds, and
        the block has to say so rather than print a plausible table of identical columns."""
        block = p3d_estimator.integrity_block({"flir": _estimator_rows(magsac_moves=False)})
        assert _verdict_for(block, "magsac") == "**FAIL**"

    def test_lmeds_losing_pairs_to_the_inlier_gate_is_not_a_failure(self) -> None:
        """The finding this check was rebuilt around: OpenCV thresholds lmeds's inlier *mask*
        even though it does not threshold its fit, so a tight column can lose pairs. The check
        is on the homographies of the pairs that survived, so that must still read PASS."""
        rows = _estimator_rows()
        rows["lmeds", 1.0, "sift"] = rows["lmeds", 1.0, "sift"][:1]
        block = p3d_estimator.integrity_block({"flir": rows})
        line = _line_for(block, "lmeds")
        assert line[-1] == "PASS", line
        # One pair solved at all three thresholds, not two: the 1 px column lost the other.
        assert line[2:4] == ["1", "1"], line

    def test_a_median_over_matchers_is_not_shifted_by_an_estimator_that_lacks_one(self) -> None:
        """F37, at fixture scale. `xfeat` has no PROSAC cell, so a median over "whatever this
        estimator has" is a median over seven matchers for PROSAC and eight for the rest --
        and on `flir` that reversed the ordering against MAGSAC. The fixture is built so the
        naive median reverses it too, and the assertion is that the block does not.
        """
        series = _estimator_series(
            {
                "magsac": {"m1": 10.0, "m2": 11.0, "m3": 20.0, "m4": 40.0},
                "prosac": {"m1": 12.0, "m2": 13.0, "m3": 21.0},
            }
        )
        # PROSAC is worse than MAGSAC on every matcher they share, and the naive median says
        # otherwise purely because dropping `m4` moves which value sits in the middle.
        assert statistics.median([10.0, 11.0, 20.0, 40.0]) > statistics.median([12.0, 13.0, 21.0])

        assert p3d_estimator._common_matchers(series) == ["m1", "m2", "m3"]
        block = p3d_estimator.seed_block({"flir": series}, (0,), p3d_estimator.HEADLINE)
        assert _level_for(block, "magsac@1px") < _level_for(block, "prosac@1px")
        assert "m4 excluded" in block

    def test_an_estimator_that_buys_mace_by_declining_pairs_ranks_last_on_the_success_block(
        self,
    ) -> None:
        """Why both metrics are printed (F33/F34). `reg/mace` is a mean over successes and
        `reg/success_rate_10px` counts a failure as infinite error, so an estimator that
        declines the pairs it cannot solve leads the first block and trails the second. Stage D
        was nearly recorded off the first alone.
        """
        series = _estimator_series(
            {
                "magsac": {"m1": 60.0, "m2": 62.0, "m3": 64.0},
                "lmeds": {"m1": 14.0, "m2": 15.0, "m3": 16.0},
            },
            success={"magsac": 0.70, "lmeds": 0.60},
        )
        mace = p3d_estimator.seed_block({"flir": series}, (0,), p3d_estimator.HEADLINE)
        rate = p3d_estimator.seed_block({"flir": series}, (0,), p3d_estimator.SECONDARY)
        assert _level_for(mace, "lmeds@1px") < _level_for(mace, "magsac@1px")
        assert _level_for(rate, "lmeds@1px") < _level_for(rate, "magsac@1px")


def _line_for(block: str, estimator: str) -> list[str]:
    """The data row for one estimator -- matched on the dataset column, because the block's
    header comments mention the estimators by name too."""
    (line,) = [
        row for row in block.splitlines() if row.startswith("flir") and row.split()[1] == estimator
    ]
    return line.split()


def _verdict_for(block: str, estimator: str) -> str:
    return _line_for(block, estimator)[-1]


def _argv_for(cell: stages.Cell) -> list[str]:
    """The argv `p3d_estimator.run` would build, captured without running anything."""
    captured: list[list[str]] = []
    original = p3d_estimator.run_cell
    p3d_estimator.run_cell = lambda _cell, _dir, argv, _banner: captured.append(argv) or True
    try:
        p3d_estimator.run(cell, 0, "cpu", stages.DryRun(wandb=False))
    finally:
        p3d_estimator.run_cell = original
    return captured[0]


def _estimator_series(
    mace: dict[str, dict[str, float]], success: dict[str, float] | None = None
) -> p3d_estimator.Series:
    """One dataset's summarised cells, flat across the threshold axis.

    Every estimator named in `mace` carries the matchers given for it and no others, which is
    how a cell absent for one estimator alone -- `xfeat` under PROSAC -- is expressed. Estimators
    not named carry the first entry's numbers, so a block that iterates all four still renders.
    """
    rates = success or {}
    default = next(iter(mace.values()))
    series: p3d_estimator.Series = {}
    for estimator in p3d_estimator.ESTIMATORS:
        values = mace.get(estimator, default)
        for threshold in p3d_estimator.THRESHOLDS:
            for matcher, value in values.items():
                series[estimator, threshold, matcher] = [
                    {
                        p3d_estimator.HEADLINE: value,
                        p3d_estimator.SECONDARY: rates.get(estimator, 0.5),
                        FAILURE_RATE: 0.0,
                        "reg/n_pairs": 300.0,
                    }
                ]
    return series


def _level_for(block: str, cell: str) -> float:
    """The level column of one `estimator@threshold` row of a `seed_block`."""
    (line,) = [row for row in block.splitlines() if row.startswith("flir") and cell in row]
    return float(line.split()[2])


def _estimator_rows(magsac_moves: bool = True) -> p3d_estimator.Rows:
    """Two pairs per cell: lmeds identical across thresholds, the others varying with it."""
    rows: p3d_estimator.Rows = {}
    for estimator in p3d_estimator.ESTIMATORS:
        flat = estimator == "lmeds" or (estimator == "magsac" and not magsac_moves)
        for threshold in p3d_estimator.THRESHOLDS:
            offset = 0.0 if flat else threshold
            rows[estimator, threshold, "sift"] = [
                make_row(stem, h=[1.0, 0.0, offset, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                for stem in ("a", "b")
            ]
    return rows


class TestStageE:
    """`scripts/p3e_warp.py` (P3-11). The stage whose conclusion is bounded before it runs.

    Three things are worth pinning here, and they are the three a reader of the pasted table
    cannot check for themselves: the arithmetic that makes two match passes legitimate, the
    `similarity/magsac` hole being a *named* row rather than a silent `--` (X-4), and the excess
    block's verdict -- which is the only place the stage distinguishes "the homography has more
    capacity" from "Tier-1's ground truth is projective and hands it the win".
    """

    def test_the_stage_is_two_match_passes_carrying_six_variants_each(self) -> None:
        """The cost model. GRID.md §6: the warp model is downstream of the matcher exactly as
        the estimator is, so one match pass per dataset feeds three fits per pair per estimator.
        If this ever becomes one invocation per variant, the stage re-runs RoMa six times to
        change which matrix `cv2` fits."""
        passes = [
            p3e_warp.run_dir_for(cell) for cell in stages.CELLS if cell.dataset in p3e_warp.DATASETS
        ]
        assert len(passes) == 2
        assert len(set(passes)) == len(passes)
        assert len(p3e_warp.MODELS) * len(p3e_warp.ESTIMATORS) == 6

    def test_the_invocation_resolves_to_the_six_variants_over_reduced_eight(self) -> None:
        """Resolved through `cmreg`'s own parser, not rebuilt here. Also pins that the anchor --
        the variant whose console block the runner prints -- is one of the swept cells, which is
        the condition `EstimateConfig` enforces at load."""
        from cmreg.cli import build_parser, overrides_from_args
        from cmreg.config import Config

        cell = next(c for c in stages.CELLS if c.dataset == "flir")
        args = build_parser().parse_args(p3e_warp.argv_for(cell))
        config = Config.load(args.config, overrides_from_args(args))
        variants = config.estimate.variants()

        assert len(variants) == 6
        assert config.match.matchers == stages.REDUCED_8
        anchor = (config.estimate.warp_model.value, config.estimate.method.value)
        assert anchor == p3e_warp.ANCHOR
        assert anchor in [(v.warp_model.value, v.method.value) for v in variants]
        # One axis at a time: the threshold is frozen at stage D's anchor across all six.
        assert {v.threshold_px for v in variants} == {p3e_warp.THRESHOLD}

    def test_the_run_argv_is_the_scientific_argv_plus_the_invocation(self) -> None:
        """`floors_for` and the test above resolve `argv_for`, and the server runs `run`. If the
        two could drift, every floor printed beside a row would belong to a different config
        than the row does -- which is invisible in the table and wrong in the third decimal."""
        cell = next(c for c in stages.CELLS if c.dataset == "flir")
        scientific = p3e_warp.argv_for(cell)
        assert _stage_e_argv(cell)[: len(scientific)] == scientific

    def test_similarity_under_magsac_is_a_named_row_and_not_a_silent_hole(self) -> None:
        """P3-4a F39: OpenCV fits a 4-DoF similarity by RANSAC and LMEDS only, so the anchor
        estimator is unavailable on exactly one of three columns. X-4 makes that a row carrying
        its reason, and this stage's table has to say so where it prints `--`."""
        from cmreg.config import Estimator, WarpModel
        from cmreg.estimate import SUPPORTED_ESTIMATORS

        assert Estimator.MAGSAC not in SUPPORTED_ESTIMATORS[WarpModel.SIMILARITY]
        assert Estimator.RANSAC in SUPPORTED_ESTIMATORS[WarpModel.SIMILARITY]
        assert Estimator.RANSAC.value == p3e_warp.CONTROL

        rows = _warp_rows({"homography": 6.0, "affine": 12.0})
        note = p3e_warp._unsupported_note(rows, "magsac")
        assert note is not None and "similarity" in note
        table = p3e_warp.model_table(
            next(c for c in stages.CELLS if c.dataset == "flir"),
            "magsac",
            _warp_series({"homography": 6.0, "affine": 12.0}),
            rows,
            _warp_floors(),
        )
        assert note in table
        # The column exists and is empty, rather than being absent from the table entirely.
        assert "similarity" in table
        assert table.splitlines()[-2].endswith("--")

    def test_the_excess_block_calls_a_floored_gap_the_ground_truth(self) -> None:
        """The block the stage exists for (GRID.md §6). Built so all three models sit exactly
        2 px above their own floors and the raw mace differs by the floor gap alone: the honest
        reading is that the axis measured the ground truth, and a table without the floor column
        would have reported a 15 px 'capacity' win."""
        mace = {"homography": 2.0, "affine": 12.0, "similarity": 17.0}
        block = p3e_warp.excess_block(
            {"flir": _warp_rows(mace)}, {"flir": _warp_series(mace)}, {"flir": _warp_floors()}
        )
        assert "GROUND TRUTH" in block
        assert "CAPACITY" not in block

    def test_the_excess_block_calls_an_unfloored_gap_capacity(self) -> None:
        """The converse, which is the reading that would licence the axis as a result: the
        restricted models are far above their own floors, so most of the gap survives it."""
        mace = {"homography": 2.0, "affine": 60.0, "similarity": 80.0}
        block = p3e_warp.excess_block(
            {"flir": _warp_rows(mace)}, {"flir": _warp_series(mace)}, {"flir": _warp_floors()}
        )
        assert "CAPACITY" in block
        assert "GROUND TRUTH" not in block

    def test_a_median_over_matchers_is_not_shifted_by_a_model_that_lacks_one(self) -> None:
        """F37 in this stage's shape. A matcher that failed every pair under one model has no
        mace there, and a median over 'whatever this model has' would compare a median over
        three matchers with one over four."""
        series = _warp_series({"homography": 5.0}, per_matcher={"affine": {"m1": 9.0, "m2": 10.0}})
        assert p3e_warp._common_matchers(series) == ["m1", "m2"]
        block = p3e_warp.control_block({"flir": series}, {"flir": _warp_rows()}, p3e_warp.HEADLINE)
        assert "m3 excluded" in block

    def test_the_control_block_does_not_score_an_estimator_that_never_ran(self) -> None:
        """F47, found in the stage's own server output. `reg/mace` is NaN for a cell with no
        successes, so filtering on the aggregated metric hid this on the headline table -- but
        `reg/success_rate_10px` reads a truthful 0.0 for `similarity/magsac`, and the block
        printed `magsac 0.0000 | ransac 0.0633 | delta -0.0633` about an estimator OpenCV
        cannot fit at all. The predicate has to be the capability gap, not the metric's value."""
        series, rows = _warp_series({"homography": 5.0}), _warp_rows()
        for metric in p3e_warp.AGGREGATE_METRICS:
            block = p3e_warp.control_block({"flir": series}, {"flir": rows}, metric)
            (row,) = [line for line in block.splitlines() if line.split()[1:2] == ["similarity"]]
            magsac, ransac, delta = row.split()[2:5]
            assert magsac == "--", f"{metric}: magsac scored on a model it cannot fit"
            assert delta == "--", f"{metric}: a delta against an estimator that never ran"
            assert ransac != "--"

    def test_one_raised_pair_does_not_disguise_a_cell_opencv_cannot_fit(self) -> None:
        """F47's second half, also from the server output. `run_benchmark` catches per *pair*,
        not per variant, so a pair whose fit raises discards every variant of that
        (pair, matcher) -- including the `estimator_unsupported_for_warp` rows already built for
        it. Three such pairs out of 300 made an `all`-quantified predicate report `similarity`
        as supported, which printed a 0.00 ms cost and dropped the note explaining the `--`."""
        rows = _warp_rows()
        key = ("similarity", "magsac", "m1")
        rows[key] = [
            make_row(
                "raised",
                matcher="m1",
                warp="similarity",
                estimator="magsac",
                success=False,
                failure_reason="estimator_failed",
            ),
            *rows[key][1:],
        ]
        assert p3e_warp._is_unsupported(rows, key)
        note = p3e_warp._unsupported_note(rows, "magsac")
        assert note is not None and "similarity" in note

    def test_a_cell_opencv_cannot_fit_is_not_reported_as_the_cheapest_one(self) -> None:
        """`estimator_unsupported_for_warp` rows carry `estimate_ms=0.0` truthfully -- nothing
        was estimated -- and a cost table that prints that as `0.00` ranks the one column that
        could not run as the fastest on the page."""
        block = p3e_warp.cost_block(
            {"flir": _warp_series({"homography": 5.0})}, {"flir": _warp_rows()}
        )
        (row,) = [line for line in block.splitlines() if line.startswith("m1")]
        assert row.split()[-2] == "--"  # simi/magsac
        assert row.split()[-1] != "--"  # simi/ransac ran


def _stage_e_argv(cell: stages.Cell) -> list[str]:
    """The argv `p3e_warp.run` would build, captured without running anything."""
    captured: list[list[str]] = []
    original = p3e_warp.run_cell
    p3e_warp.run_cell = lambda _cell, _dir, argv, _banner: captured.append(argv) or True
    try:
        p3e_warp.run(cell, "cpu", stages.DryRun(wandb=False))
    finally:
        p3e_warp.run_cell = original
    return captured[0]


# Two pairs' floors, chosen so the affine floor (10 px) is most of any plausible affine error and
# the homography floor is exactly zero -- which is what it is by construction (P3-4a F41).
_FLOOR = {"homography": 0.0, "affine": 10.0, "similarity": 15.0}


def _warp_floors() -> p3e_warp.Floors:
    return {stem: dict(_FLOOR) for stem in ("a", "b")}


def _warp_series(
    mace: dict[str, float],
    per_matcher: dict[str, dict[str, float]] | None = None,
    matchers: tuple[str, ...] = ("m1", "m2", "m3"),
) -> p3e_warp.Series:
    """One dataset's summarised cells, flat across the estimator axis.

    `per_matcher` names a model whose cells exist for *some* matchers only, which is how a model
    that solved nothing for one matcher is expressed.

    `similarity/magsac` is **present**, exactly as the server run produces it (F39/F47): the
    runner summarises the unsupported rows like any others, so the cell exists carrying a NaN
    mace and a truthful `success_rate_10px` of 0.0. A fixture that omitted the cell instead is
    what let a block read that 0.0 as a score without a test noticing.
    """
    overrides = per_matcher or {}
    series: p3e_warp.Series = {}
    for model in p3e_warp.MODELS:
        for estimator in p3e_warp.ESTIMATORS:
            unsupported = model == "similarity" and estimator == "magsac"
            values = overrides.get(model) or {
                matcher: mace.get(model, 20.0) for matcher in matchers
            }
            for matcher, value in values.items():
                series[model, estimator, matcher] = [
                    {
                        p3e_warp.HEADLINE: float("nan") if unsupported else value,
                        p3e_warp.SECONDARY: 0.0 if unsupported else 0.5,
                        FAILURE_RATE: 1.0 if unsupported else 0.0,
                        "reg/n_pairs": 300.0,
                        "time/estimate_ms": 0.0 if unsupported else 1.0,
                    }
                ]
    return series


def _warp_rows(
    mace: dict[str, float] | None = None, matchers: tuple[str, ...] = ("m1", "m2", "m3")
) -> p3e_warp.Rows:
    """Two pairs per (model, estimator, matcher), `similarity/magsac` recorded as unsupported.

    Takes the same `mace` dict `_warp_series` does, so the per-pair rows the excess block reads
    and the summarised cells the raw-mace column reads cannot disagree -- which is the whole
    thing that block is comparing.
    """
    errors = mace or {}
    rows: p3e_warp.Rows = {}
    for model in p3e_warp.MODELS:
        for estimator in p3e_warp.ESTIMATORS:
            unsupported = model == "similarity" and estimator == "magsac"
            for matcher in matchers:
                rows[model, estimator, matcher] = [
                    make_row(
                        stem,
                        matcher=matcher,
                        warp=model,
                        estimator=estimator,
                        success=not unsupported,
                        failure_reason="estimator_unsupported_for_warp" if unsupported else None,
                        corner_err=None if unsupported else errors.get(model, 20.0),
                    )
                    for stem in ("a", "b")
                ]
    return rows


class TestStageF:
    """`scripts/p3f_resolution.py` (P3-12b). The stage whose cheapest mistake is invisible.

    Four things are pinned here. The cost model, because this axis is the one that *cannot* be
    swept off a single `MatchResult` and a driver that tried would silently score three of its
    four levels at the wrong resolution. The `config_hash` exemption, because it is what makes
    `refuse_a_stale_run` refuse a stale *level* directory rather than tabulating x0.5's rows
    under the x1 banner. The floor cell's composition, which is this stage's silent failure --
    see the driver's module docstring. And the two blocks whose arithmetic *is* the finding.
    """

    def test_the_stage_is_one_match_pass_per_level(self) -> None:
        """GRID.md §6: the resolution axis sits **upstream** of the matcher
        (`eval/runner.py::_identity_columns`), so unlike stages D and E it costs one match pass
        per level rather than one fit off a shared match. Twelve distinct directories, and the
        floor arm's must not collide with the cross-modal arm's -- a collision would resume-skip
        the control onto the benchmark's own rows and print them as the floor."""
        dirs = [
            p3f_resolution.run_dir_for(cell, level)
            for cell in stages.CELLS
            if cell.dataset in p3f_resolution.DATASETS
            for level in p3f_resolution.LEVELS
        ]
        dirs += [p3f_resolution.floor_run_dir_for(level) for level in p3f_resolution.LEVELS]
        assert len(dirs) == 12
        assert len(set(dirs)) == len(dirs)

    def test_every_level_divides_the_benchmark_shape_exactly(self) -> None:
        """P3-12a F56: `rescale` refuses an anisotropic resize rather than rounding it, because
        `to_native` inverts a *single* scale and an x/y mismatch is a systematic sub-pixel bias
        that reads as a worse matcher. Both stage-F datasets are 640x512, so the level list has
        to be exact on both axes -- which is why these four levels and not, say, x0.6."""
        for level in p3f_resolution.LEVELS:
            for extent in (640, 512):
                scaled = level * extent
                assert scaled == int(scaled), f"x{level:g} is anisotropic on {extent}"

    def test_the_floor_cell_does_not_compose_the_rig_constant(self) -> None:
        """**The oracle for this stage's silent failure.**

        `stages.CELLS` marks `flir` `composes=True`. A control cell built from it would fold
        `calibration/flir.json`'s ~9.5 px rig constant into a pair matched against a warped copy
        of *itself* -- which has no cross-modal pairing and therefore no residual to remove. The
        floor would read ~9-10 px instead of ~0.2-1.0 px and would swamp the axis it exists to
        bound. Asserted on the *resolved* config, through `cmreg`'s own parser, so the pin is on
        what the run does rather than on which flags this file happens to pass.
        """
        from cmreg.cli import build_parser, overrides_from_args
        from cmreg.config import Config

        for level in p3f_resolution.LEVELS:
            args = build_parser().parse_args(p3f_resolution.floor_argv_for(level))
            config = Config.load(args.config, overrides_from_args(args))
            assert config.gt.residual_calibration is None, f"x{level:g} composed a rig it lacks"
            assert config.gt.is_monomodal
            assert config.gt.reference_modality.value == p3f_resolution.FLOOR_MODALITY
            # An asymmetric recipe would manufacture the polarity gap the control removes.
            assert config.preprocess.reference.value == "none"
            assert config.preprocess.moving.value == "none"
            assert config.preprocess.input_scale == level
        # The contrast, on the same dataset at the same level: the cross-modal arm *does*
        # compose, so this is a property of the control and not of the driver forgetting a flag.
        cell = next(c for c in stages.CELLS if c.dataset == p3f_resolution.FLOOR_DATASET)
        args = build_parser().parse_args(p3f_resolution.argv_for(cell, 0.5))
        composed = Config.load(args.config, overrides_from_args(args))
        assert composed.gt.residual_calibration is not None
        assert not composed.gt.is_monomodal

    def test_the_anchor_level_keeps_the_pre_axis_hash_and_the_others_move(self) -> None:
        """P3-12a F58, in the form `refuse_a_stale_run` consumes it. `input_scale` is dropped
        from the payload at 1.0, so the anchor level resolves to the hash every stage A-E
        directory on the server already carries -- and every other level hashes apart, which is
        what stops x0.5's rows being resumed into the x1 cell and printed under its banner."""
        cell = next(c for c in stages.CELLS if c.dataset == "dronevehicle")
        anchor = stages.intended_hash(p3f_resolution.argv_for(cell, 1.0))
        others = {
            level: stages.intended_hash(p3f_resolution.argv_for(cell, level))
            for level in p3f_resolution.LEVELS
            if level != 1.0
        }
        assert anchor not in others.values()
        assert len(set(others.values())) == len(others)
        # The scalar exemption itself: passing the default explicitly changes nothing.
        without = [flag for flag in p3f_resolution.argv_for(cell, 1.0) if flag != "--input-scale"]
        assert stages.intended_hash([f for f in without if f != "1"]) == anchor

    def test_the_run_argv_is_the_scientific_argv_plus_the_invocation(self) -> None:
        """Both arms. The tests above resolve `argv_for`/`floor_argv_for` and the server runs
        `run`/`run_floor`; if the two could drift, every floor printed beside a row would belong
        to a different config than the row does -- invisible in the table, wrong in the answer."""
        cell = next(c for c in stages.CELLS if c.dataset == "flir")
        scientific = p3f_resolution.argv_for(cell, 0.5)
        assert _stage_f_argv(cell, 0.5)[: len(scientific)] == scientific
        floor = p3f_resolution.floor_argv_for(0.5)
        assert _stage_f_floor_argv(0.5)[: len(floor)] == floor

    def test_the_floor_prediction_is_the_anchor_row_over_the_level(self) -> None:
        """F54 says the floor is "close to 1/s" and the fixture's numbers say what that means:
        0.45 px at x0.5 and 1.02 px at x0.25 against ~0.2 px at x1 is the x1 floor *scaled*, not
        1/s in absolute terms. Predicted off this run's own anchor row so it is calibrated on the
        data it is checked against."""
        table = _res_table({1.0: 0.2, 0.5: 0.45, 0.25: 1.02})
        levels = [1.0, 0.5, 0.25]
        assert p3f_resolution._predicted_floor(table, levels, 0.5) == pytest.approx(0.4)
        assert p3f_resolution._predicted_floor(table, levels, 0.25) == pytest.approx(0.8)

    def test_a_level_swamped_by_its_own_floor_is_flagged(self) -> None:
        """The block's whole job. Built so x0.25's error is barely above the control's at the
        same level: without the floor column the row reads as a working registration, and with
        it the row is the axis measuring its own localisation limit."""
        errors = _res_errors({1.0: 6.0, 0.25: 2.0})
        floor = _res_errors({1.0: 0.2, 0.25: 1.8})
        block = p3f_resolution.floor_limited_block(
            {"flir": _res_table({1.0: 6.0, 0.25: 2.0})},
            {"flir": errors},
            _res_table({1.0: 0.2, 0.25: 1.8}),
            floor,
            (1.0, 0.25),
        )
        rows = {line.split()[1]: line for line in block.splitlines() if line.startswith("flir")}
        assert "FLOOR-LIMITED" in rows["x0.25"]
        assert "FLOOR-LIMITED" not in rows["x1"]
        assert "measured" in rows["x0.25"]

    def test_a_dataset_without_a_control_says_so_rather_than_borrowing_one(self) -> None:
        """The floor is bought on `flir` only, so `dronevehicle` is read against the x1/s
        prediction. A floor whose provenance a reader cannot see is worse than no floor."""
        block = p3f_resolution.floor_limited_block(
            {"dronevehicle": _res_table({1.0: 6.0, 0.5: 9.0})},
            {"dronevehicle": _res_errors({1.0: 6.0, 0.5: 9.0})},
            _res_table({1.0: 0.2, 0.5: 0.45}),
            _res_errors({1.0: 0.2, 0.5: 0.45}),
            (1.0, 0.5),
        )
        rows = [line for line in block.splitlines() if line.startswith("dronevehicle")]
        assert rows and all(line.split()[-1] == "pred" for line in rows)
        assert not any("measured" in line for line in rows)

    def test_a_median_over_matchers_is_not_shifted_by_a_level_that_lost_one(self) -> None:
        """F37 in this stage's shape, and it bites harder here than it did in stage D: x0.25 is
        precisely the level at which a matcher is expected to stop solving, so a median over
        "whatever this level has" would compare a median over eight matchers with one over six
        and report the survivors' scores as the level's."""
        table = _res_table({1.0: 5.0, 0.25: 9.0})
        table["x0.25"]["m3"] = dict(table["x0.25"]["m3"], **{p3f_resolution.HEADLINE: float("nan")})
        assert p3f_resolution._common_matchers(table, [1.0, 0.25]) == ["m1", "m2"]
        note = p3f_resolution._excluded_note({"flir": table}, (1.0, 0.25))
        assert note is not None and "m3 excluded" in note

    def test_the_floor_and_the_error_are_medians_over_one_matcher_set(self) -> None:
        """F37 applied *across arms* rather than across levels, and it is the defect the first
        smoke run of this driver actually had: the benchmark's row excluded a matcher the control
        kept, so the printed ratio divided a median over two matchers by a median over four. The
        control is the easier problem -- a pair matched against a warped copy of itself -- so it
        goes on solving after the benchmark stops, and the two sets diverge at exactly the low
        levels this stage exists to characterise."""
        matchers = ("m1", "m2", "m3", "m4")
        table = _res_table({1.0: 6.0, 0.25: 6.0}, matchers=matchers)
        for name in ("m3", "m4"):
            table["x0.25"][name] = dict(
                table["x0.25"][name], **{p3f_resolution.HEADLINE: float("nan")}
            )
        floor = _res_table({1.0: 0.2, 0.25: 0.5}, matchers=matchers)
        errors: p3f_resolution.Errors = {
            label: dict.fromkeys(matchers, 6.0) for label in ("x1", "x0.25")
        }
        floor_errors: p3f_resolution.Errors = {
            "x1": dict.fromkeys(matchers, 0.2),
            # The two the benchmark lost score far worse in the control. Including them takes the
            # floor from 0.50 to 25.25 and the row from "registration" to "FLOOR-LIMITED".
            "x0.25": {"m1": 0.5, "m2": 0.5, "m3": 50.0, "m4": 50.0},
        }
        block = p3f_resolution.floor_limited_block(
            {"flir": table}, {"flir": errors}, floor, floor_errors, (1.0, 0.25)
        )
        (row,) = [line for line in block.splitlines() if line.split()[1:2] == ["x0.25"]]
        assert row.split()[3] == "0.50"
        assert "FLOOR-LIMITED" not in row

    def test_the_ranking_block_reports_a_saturated_level_rather_than_crashing(self) -> None:
        """`statistics.correlation` raises on a constant series, and a level where every matcher
        scores 0.000 legitimately produces one -- which is the *expected* shape of x0.25 on the
        failure-inclusive metric. Reported as `flat`; a crash here would cost the whole console
        block after four hours of GPU time."""
        table = _res_table(
            {1.0: 5.0, 0.25: 9.0}, per_matcher={1.0: {"m1": 4.0, "m2": 5.0, "m3": 6.0}}
        )
        block = p3f_resolution.ranking_block({"flir": table}, (1.0, 0.25), p3f_resolution.HEADLINE)
        assert "flat" in block


def _stage_f_argv(cell: stages.Cell, level: float) -> list[str]:
    """The argv `p3f_resolution.run` would build, captured without running anything."""
    return _capture(lambda: p3f_resolution.run(cell, level, "cpu", stages.DryRun(wandb=False)))


def _stage_f_floor_argv(level: float) -> list[str]:
    return _capture(lambda: p3f_resolution.run_floor(level, "cpu", stages.DryRun(wandb=False)))


def _capture(call: Callable[[], object]) -> list[str]:
    captured: list[list[str]] = []
    original = p3f_resolution.run_cell
    p3f_resolution.run_cell = lambda _cell, _dir, argv, _banner: captured.append(argv) or True
    try:
        call()
    finally:
        p3f_resolution.run_cell = original
    return captured[0]


def _res_table(
    mace: dict[float, float],
    per_matcher: dict[float, dict[str, float]] | None = None,
    matchers: tuple[str, ...] = ("m1", "m2", "m3"),
) -> p3f_resolution.Table:
    """One arm's summarised cells, keyed by level label.

    `per_matcher` names a level whose matchers carry different values, which is how an ordering
    change is expressed -- the ranking block is the one renderer that reads across matchers
    within a level rather than across levels within a matcher.
    """
    overrides = per_matcher or {}
    table: p3f_resolution.Table = {}
    for level, value in mace.items():
        values = overrides.get(level) or dict.fromkeys(matchers, value)
        table[p3f_resolution.label_for(level)] = {
            matcher: {
                p3f_resolution.HEADLINE: cell,
                p3f_resolution.SECONDARY: 0.5,
                FAILURE_RATE: 0.0,
                "reg/n_pairs": 300.0,
                "time/total_ms": 100.0 * level,
            }
            for matcher, cell in values.items()
        }
    return table


def _res_errors(
    median: dict[float, float], matchers: tuple[str, ...] = ("m1", "m2", "m3")
) -> p3f_resolution.Errors:
    """The per-pair median corner errors the floor block divides. Takes the same shape
    `_res_table` does, so the two cannot disagree about what a level scored."""
    return {
        p3f_resolution.label_for(level): dict.fromkeys(matchers, value)
        for level, value in median.items()
    }


class TestStageG:
    """`scripts/p3g_matchcount.py` (P3-12c-b). The stage whose widest columns must be the anchor.

    Four things are pinned here. The **cell shape**, because the uncapped column collapses to one
    cell and a driver expecting eighteen would render a permanent `--` that reads as a failed
    cell. The **ladder**, because it was chosen against stage A's measured yields rather than
    P0-2's single-pair probe, and a ladder that stops at 64 makes `sift` an all-anchor row. The
    **`config_hash` relation**, because P3-12c's two scalars are dropped at their anchor values
    and `refuse_a_stale_run` needs a swept directory to hash apart from an unswept one. And the
    **integrity block's arithmetic**, which is this stage's silent failure: if an inert cap
    reorders instead of reproducing, every wide column above it is a permutation rather than a
    count and each one still looks entirely plausible.
    """

    def test_the_stage_is_two_match_passes_carrying_seventeen_variants(self) -> None:
        """GRID.md §6: the axis sits **downstream** of the matcher, so seventeen cells cost one
        match pass per dataset -- the economics of stages D and E, not of stage F. Seventeen and
        not eighteen: with no cap the two orderings select identically, so `EstimateConfig.
        variants()` emits the uncapped cell once (F75) and `columns()` mirrors it."""
        dirs = [
            p3g_matchcount.run_dir_for(cell)
            for cell in stages.CELLS
            if cell.dataset in p3g_matchcount.DATASETS
        ]
        assert len(dirs) == 2
        assert len(set(dirs)) == len(dirs)
        columns = p3g_matchcount.columns()
        assert len(columns) == 17
        assert len(set(columns)) == len(columns)
        assert [cap for cap, _ in columns].count(0) == 1

    def test_the_invocation_resolves_to_the_seventeen_variants_over_reduced_eight(self) -> None:
        """Resolved through `cmreg`'s own parser and override table rather than by rebuilding
        the config here: a second path to the same answer is a second path that can disagree
        with the run it describes. The four settled axes are asserted *across every variant*,
        because the stage's whole claim is that only the match count moves."""
        from cmreg.cli import build_parser, overrides_from_args
        from cmreg.config import Config

        cell = next(c for c in stages.CELLS if c.dataset == "flir")
        args = build_parser().parse_args(p3g_matchcount.argv_for(cell))
        config = Config.load(args.config, overrides_from_args(args))
        variants = config.estimate.variants()

        assert len(variants) == len(p3g_matchcount.columns())
        assert {(v.max_matches, v.match_selection.value) for v in variants} == set(
            p3g_matchcount.columns()
        )
        assert {v.method.value for v in variants} == {p3g_matchcount.ESTIMATOR}
        assert {v.threshold_px for v in variants} == {p3g_matchcount.THRESHOLD}
        assert {v.warp_model.value for v in variants} == {p3g_matchcount.WARP_MODEL}
        assert config.preprocess.input_scale == p3g_matchcount.INPUT_SCALE
        assert list(config.match.matchers) == list(stages.REDUCED_8)

    def test_the_ladder_bites_every_matcher_and_none_of_it_is_below_a_minimal_sample(
        self,
    ) -> None:
        """**Why nine levels and not GRID.md's frozen six.** `sift`'s measured yield is 46.3 on
        `flir` and 32.1 on `dronevehicle`, so a ladder stopping at 64 is inert for it in every
        column and one of reduced-8 reproduces the anchor throughout. Three responsive caps per
        matcher is the bar; the floor is 4, below which a homography has no minimal sample."""
        capped = [cap for cap in p3g_matchcount.CAPS if cap != 0]
        assert min(capped) >= 4
        assert capped == sorted(capped, reverse=True)
        for dataset, yields in p3g_matchcount.MEASURED_YIELD.items():
            for matcher, supply in yields.items():
                biting = [cap for cap in capped if cap < supply]
                assert len(biting) >= 3, f"{dataset}/{matcher} responds at {biting}"
        # The frozen ladder, stated as the thing this one replaces rather than left implicit.
        sift = min(p3g_matchcount.MEASURED_YIELD[d]["sift"] for d in p3g_matchcount.DATASETS)
        assert sift < 64, "a six-level ladder stopping at 64 would leave sift at its anchor"

    def test_every_matcher_in_the_yield_table_is_one_the_stage_runs(self) -> None:
        """`MEASURED_YIELD` is divided by (block 2's `knee/yield`) and read as the reason the
        ladder is what it is, so a name that drifts out of `reduced-8` would put a fraction in
        the console against a matcher no column has."""
        for dataset, yields in p3g_matchcount.MEASURED_YIELD.items():
            assert dataset in p3g_matchcount.DATASETS
            assert set(yields) == set(stages.REDUCED_8)

    def test_the_sweep_hashes_apart_from_the_uncapped_anchor(self) -> None:
        """P3-12c F76, in the form `refuse_a_stale_run` consumes it. `max_matches` is dropped
        from the hash payload at `0` and `match_selection` at `confidence`, so this stage's
        anchor cell resolves to the digest every stage A-F directory already carries -- and the
        swept config hashes apart, which is what stops stage E's rows being resumed into this
        stage's directory and printed under its banner.

        The **relation** is asserted and never the values: the digest is platform-dependent
        (F64), so a number pinned here would fail on the server."""
        cell = next(c for c in stages.CELLS if c.dataset == "dronevehicle")
        swept = p3g_matchcount.argv_for(cell)
        anchor = _drop_flags(swept, ("--sweep-max-matches", "--sweep-selections"))
        pre_axis = _drop_flags(anchor, ("--max-matches", "--match-selection"))
        assert stages.intended_hash(anchor) == stages.intended_hash(pre_axis)
        assert stages.intended_hash(swept) != stages.intended_hash(anchor)

    def test_the_run_argv_is_the_scientific_argv_plus_the_invocation(self) -> None:
        """The split exists so the tests above resolve the very config the run used. If `run`
        ever adds a scientific flag of its own, they would be checking a config nothing ran."""
        cell = next(c for c in stages.CELLS if c.dataset == "flir")
        scientific = p3g_matchcount.argv_for(cell)
        assert _stage_g_argv(cell)[: len(scientific)] == scientific

    def test_a_cap_above_the_yield_is_marked_as_the_anchor_reproduced(self) -> None:
        """F74 made visible in the deliverable. Without the `=` marker the wide columns read as
        a suspiciously flat result instead of the correctness property they are."""
        series, rows = _g_cells("roma", n_matches=50)
        cell = next(c for c in stages.CELLS if c.dataset == "flir")
        table = p3g_matchcount.count_table(cell, series, rows, p3g_matchcount.HEADLINE)
        lines = {line.split()[0]: line for line in table.splitlines() if line[:1].isalnum()}
        # 50 matches: 64 and above cannot bite, 32 and below do.
        assert lines["64/conf"].rstrip().endswith("=")
        assert not lines["32/conf"].rstrip().endswith("=")

    def test_the_integrity_block_catches_an_inert_cap_that_moved(self) -> None:
        """**The oracle for this stage's silent failure.** An inert cap must return the identical
        homography, not a permutation of the same correspondences fitted again -- `h` is compared
        exactly because OpenCV's solvers carry no RNG state between calls."""
        _, rows = _g_cells("roma", n_matches=50)
        assert "PASS" in p3g_matchcount.integrity_block({"flir": rows})
        moved = dict(rows)
        moved[(64, "confidence", "roma")] = [
            _g_row("a", "roma", 64, "confidence", n_matches=50, corner_err=9.0)
        ]
        block = p3g_matchcount.integrity_block({"flir": moved})
        assert "FAIL" in block
        assert "does not reproduce the uncapped fit" in block

    def test_the_integrity_block_catches_a_ratio_taken_over_the_wrong_denominator(self) -> None:
        """F77: `inlier_ratio` is inliers over `n_selected` and `n_matches` deliberately did not
        move, so a capped row's ratio is only reproducible from the file if the right
        denominator was used. Nothing downstream would notice if it were not."""
        _, rows = _g_cells("roma", n_matches=50)
        broken = dict(rows)
        row = rows[(32, "random", "roma")][0]
        broken[(32, "random", "roma")] = [
            _g_row("a", "roma", 32, "random", n_matches=50, inlier_ratio=row.n_inliers / 50)
        ]
        assert "FAIL" in p3g_matchcount.integrity_block({"flir": broken})

    def test_the_confidence_hole_is_a_named_row_and_not_a_departure(self) -> None:
        """`xfeat` scores no matches (P0-2), so a confidence-ranked cap is undefined for it. The
        cells are recorded (X-4) and must not be read as "departs at the widest cap", which is
        what a knee computed off a NaN would say."""
        series, rows = _g_cells("xfeat", n_matches=50, hole=True)
        assert "xfeat" in p3g_matchcount.hole_block({"flir": rows})
        block = p3g_matchcount.departure_block(
            {"flir": series}, {"flir": rows}, p3g_matchcount.HEADLINE
        )
        confidence = next(
            line for line in block.splitlines() if "xfeat" in line and "confidence" in line
        )
        assert "hole" in confidence
        # The random arm is defined for it and is the reason the axis still covers all eight.
        random_arm = next(
            line for line in block.splitlines() if "xfeat" in line and "random" in line
        )
        assert "hole" not in random_arm

    def test_the_knee_is_the_narrowest_cap_that_holds_all_the_way_down(self) -> None:
        """A cap that scores well *below* a departure is not promoted. Stage F's F69 is one such
        cell -- its `mace` beat every neighbour by 40% while its median barely moved -- and a
        knee read off a single comparison would have believed it."""
        lucky = {0: 5.0, 1024: 5.0, 512: 5.0, 256: 5.0, 128: 5.0, 64: 9.0, 32: 5.0, 16: 20.0}
        series, rows = _g_cells("roma", n_matches=4096, mace=lucky)
        block = p3g_matchcount.departure_block(
            {"flir": series}, {"flir": rows}, p3g_matchcount.HEADLINE
        )
        row = next(line for line in block.splitlines() if "confidence" in line and "roma" in line)
        assert row.split()[-3] == "128"

    def test_a_knee_that_never_bit_is_named_rather_than_quoted_as_a_fraction(self) -> None:
        """A knee wider than the matcher's yield is not a decimation fraction: it says the
        matcher departs as soon as the cap does anything. `sift` on `flir` yields 46, so a knee
        of 64 would print 1.382 -- "safe to decimate to 138%" -- and the honest answer is that
        no real cap held. Decided per pair off the rows, not off the mean yield."""
        held = dict.fromkeys(p3g_matchcount.CAPS, 5.0) | {32: 50.0, 16: 60.0, 8: 70.0}
        series, rows = _g_cells("sift", n_matches=50, mace=held)
        block = p3g_matchcount.departure_block(
            {"flir": series}, {"flir": rows}, p3g_matchcount.HEADLINE
        )
        row = next(line for line in block.splitlines() if "confidence" in line and "sift" in line)
        assert row.split()[-3] == "64"
        assert row.split()[-1] == "inert"

    def test_a_median_over_arms_is_not_shifted_by_a_matcher_only_one_of_them_solved(self) -> None:
        """Stage D's F37, one axis over. A median over two matchers in one arm against one
        matcher in the other compares two populations, and dropping the second-worst matcher
        lifts a median enough to invert an ordering."""
        series, rows = _g_cells("roma", n_matches=4096, mace=5.0)
        lost, lost_rows = _g_cells("eloftr", n_matches=800, mace=40.0)
        lost[(64, "confidence", "eloftr")] = [_g_metrics(float("nan"))]
        series |= lost
        rows |= lost_rows
        block = p3g_matchcount.ordering_block(
            {"flir": series}, {"flir": rows}, p3g_matchcount.HEADLINE
        )
        at_64 = next(
            line for line in block.splitlines() if line.startswith("flir") and " 64" in line
        )
        # Only `roma` survives at that cap, so both arms read its value and the delta is zero.
        assert at_64.split()[2] == "1"
        assert at_64.split()[3] == "5.00"


def _drop_flags(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    """`argv` without those flags and the value each carries."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in flags:
            skip = True
            continue
        out.append(token)
    return out


def _stage_g_argv(cell: stages.Cell) -> list[str]:
    """The argv `p3g_matchcount.run` would build, captured without running anything."""
    captured: list[list[str]] = []
    original = p3g_matchcount.run_cell
    p3g_matchcount.run_cell = lambda _cell, _dir, argv, _banner: captured.append(argv) or True
    try:
        p3g_matchcount.run(cell, "cpu", stages.DryRun(wandb=False))
    finally:
        p3g_matchcount.run_cell = original
    return captured[0]


def _g_metrics(mace: float, *, ratio: float = 0.8) -> dict[str, float]:
    return {
        p3g_matchcount.HEADLINE: mace,
        EPE_MEDIAN: 1.0,
        p3g_matchcount.SECONDARY: 0.5,
        FAILURE_RATE: 0.0,
        MATCH_INLIER_RATIO: ratio,
        TIME_ESTIMATE_MS: 1.0,
        "reg/n_pairs": 300.0,
    }


def _g_row(
    stem: str,
    matcher: str,
    cap: int,
    selection: str,
    *,
    n_matches: int,
    n_inliers: int = 40,
    corner_err: float = 2.0,
    inlier_ratio: float | None = None,
) -> PairRow:
    """One row with `n_selected` and `inlier_ratio` kept consistent with the cap (F77)."""
    n_selected = n_matches if cap == 0 else min(cap, n_matches)
    inliers = min(n_inliers, n_selected)
    return make_row(
        stem,
        matcher=matcher,
        max_matches=cap,
        match_selection=selection,
        n_matches=n_matches,
        n_selected=n_selected,
        n_inliers=inliers,
        inlier_ratio=inliers / n_selected if inlier_ratio is None else inlier_ratio,
        corner_err=corner_err,
    )


def _g_cells(
    matcher: str,
    *,
    n_matches: int,
    mace: float | dict[int, float] = 5.0,
    hole: bool = False,
) -> tuple[p3g_matchcount.Series, p3g_matchcount.Rows]:
    """One matcher's seventeen cells. `hole=True` makes the confidence arm the `xfeat` case."""
    series: p3g_matchcount.Series = {}
    rows: p3g_matchcount.Rows = {}
    for cap, selection in p3g_matchcount.columns():
        key = (cap, selection, matcher)
        blocked = hole and cap != 0 and selection == "confidence"
        value = mace.get(cap, float("nan")) if isinstance(mace, dict) else mace
        series[key] = [_g_metrics(float("nan") if blocked else value)]
        if blocked:
            rows[key] = [
                make_row(
                    "a",
                    matcher=matcher,
                    success=False,
                    max_matches=cap,
                    match_selection=selection,
                    failure_reason=p3g_matchcount.NEEDS_CONFIDENCE,
                    n_matches=n_matches,
                    n_selected=0,
                    n_inliers=0,
                    inlier_ratio=0.0,
                )
            ]
        else:
            rows[key] = [_g_row("a", matcher, cap, selection, n_matches=n_matches)]
    return series, rows
