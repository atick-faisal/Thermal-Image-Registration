"""The shared stage machinery (`scripts/stages.py`) and each stage driver's grid.

Only the guard is tested here. The rest of the driver is a `cmreg bench` invocation and a
table renderer, both covered where they live -- but the guard is the one piece whose failure
is *silent*, and it has already cost one server trip: stage-A run 2 asked for `flir` composed,
found run 1's pre-composition rows on disk, skipped the cell, and printed the old floors under
the new banner. Nothing in the pasted console said so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import p3_stageb_polarity  # on the path via `[tool.pytest.ini_options] pythonpath`
import p3c_upsample
import p3d_estimator
import stages
from cmreg.results import write_rows
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
