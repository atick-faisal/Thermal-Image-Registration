"""P3-10 / Stage D: the estimator x threshold sweep, on the training server.

`experiments/GRID.md` §6 froze this stage as 4 estimators x threshold 1/3/5 px x
`driving+aerial` x 5 seeds = 24 cells x 5 = **120 `cmreg bench` invocations**, which at stage
C's measured ~19 min per reduced-8 cell is ~38 h. Almost all of it is waste.

**The estimator axis is downstream of the matcher.** `eval/runner.py::_evaluate` matches once
and then calls `estimate_warp`; `config.estimate` is the only thing this stage varies. The
frozen grid re-runs RoMa twelve times per pair to change a RANSAC threshold. Sweeping *inside*
the pair loop instead is **10 match passes** -- 2 datasets x 5 seeds, and the seeds genuinely do
need re-matching, since `gt.seed` draws the synthetic warp and `seed_cell` seeds the matcher's
own sampling -- feeding twelve cheap `cv2.findHomography` calls off each `MatchResult`.
**~4-5 h.**

    uv run python scripts/p3d_estimator.py
    uv run python scripts/p3d_estimator.py --datasets flir --seeds 0

That equivalence is a *measured* property, not an assumption: OpenCV's robust solvers are
deterministic and carry no RNG state between calls (opencv 5.0.0 -- repeated fits are
bit-identical and `cv2.setRNGSeed` does not move them), so variant *k* cannot depend on variants
1..k-1 having run. It is pinned in two places, because nothing else in the suite would notice if
it stopped holding and every table below would look entirely plausible while being wrong:
`tests/test_estimate.py` names the cause, `tests/test_runner.py` scores a swept run against
single-estimator runs row for row.

**Why five seeds, given that.** `experiments/GRID.md` §5 said stage D carries five because
"RANSAC's sampling ... is where seed-to-seed variance is the measurement rather than a
nuisance". That premise is **false**: per the paragraph above the estimator contributes exactly
zero variance. The five seeds are kept and the reason is rewritten -- they measure the variance
of the *synthetic warp draw and the matcher's own correspondence sampling*, which is the
interval an "estimator A beats B" claim needs before it enters the paper (X-3, P8-2's Wilcoxon).
Block 2 below is where that interval is read.

**`xfeat` cannot run PROSAC.** PROSAC draws its minimal samples in descending confidence order,
and `xfeat` is one of three vismatch backends that return no per-match confidence (TASKS.md
P0-2) -- and it is in `reduced-8`. Those cells are recorded as
`estimator_needs_confidence` rows rather than aborting the run that eleven other variants
depend on, and they print as `--` here. Stated in every table it touches (X-4).

**Two metrics, both aggregated.** Amended after the 2026-08-31 run, from its own output. The
per-estimator tables always carried `reg/mace` and `reg/success_rate_10px` side by side, but the
two aggregate blocks reduced the first only -- and the first is a mean over *successes*, so LMEDS
leaving a third of the pairs unsolved read as a fourfold accuracy win over MAGSAC on `dronevehicle`
while it was in fact losing to it on the success rate for seven of eight matchers (TASKS.md
F33/F34). Both blocks are now rendered once per metric, cells carry the failure rate that makes
the two disagree, and cross-estimator medians run over `_common_matchers` rather than over
whatever each estimator happened to have (F37). Re-running this script on a server whose ten run
directories are already complete re-renders every block from Parquet without matching anything.

Needs a GPU; `--device cpu` with the dry-run overrides is how the plumbing is proved before the
trip. Every block it prints is meant to be copied out of the console whole.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from cmreg.metrics.schema import FAILURE_RATE, MACE, N_PAIRS, TIME_ESTIMATE_MS, success_rate_key
from cmreg.results import PairRow
from stages import CELLS, CONFIG, REDUCED_8, Cell, DryRun, run_cell

# `reg/mace` leads, as it does in stages A-C: P3-7's F7/F13 established that a thresholded rate
# is a function of the dataset's residual floor while a mean over corner errors is not, and
# `dronevehicle` here is a floor (GRID.md §3).
HEADLINE = MACE
SECONDARY_THRESHOLD = 10.0
SECONDARY = success_rate_key(SECONDARY_THRESHOLD)

# **Both of them lead in the aggregate blocks, and stage D's own output is why.** `reg/mace`
# is a mean over *successes only* while `reg/success_rate_10px` counts a failed pair as
# infinite error (`results/report.py:75`), so an estimator that declines the pairs it cannot
# solve buys the first with the second. LMEDS does exactly that here -- 4x the median mace of
# MAGSAC on `dronevehicle` while losing to it on the success rate for seven of eight matchers
# (TASKS.md F34). Blocks 2 and 3 are therefore rendered once per metric: read on mace alone,
# this stage's axis reads backwards.
AGGREGATE_METRICS = (HEADLINE, SECONDARY)

# Two decimals for a pixel error, four for a rate: the estimator differences in the success
# rate live in the third decimal, and `.2f` would print several of the twelve cells identical.
_RATE_METRICS = (SECONDARY, FAILURE_RATE)

# `mace | success | failure` is 21 characters at this project's widest cell (`sift` on
# `dronevehicle`, 517.27 px); 24 keeps a gap between columns.
_CELL_WIDTH = 24

# GRID.md §6's `driving+aerial`: the best-characterised driving set and the only aerial one.
DATASETS = ("flir", "dronevehicle")
SEEDS = (0, 1, 2, 3, 4)

ESTIMATORS = ("magsac", "ransac", "lmeds", "prosac")
THRESHOLDS = (1.0, 3.0, 5.0)
# The anchor cell of GRID.md §1, and what `p3a_baseline_grid.yaml` already declares. It is the
# variant whose console block the runner prints, so it must be one of the swept cells --
# `EstimateConfig` enforces that rather than leaving it to this file.
ANCHOR = ("magsac", 3.0)

# LMEDS minimises the *median* residual, so `threshold_px` never reaches its solve and its
# **homography** is identical across the threshold axis. Its **inlier mask** is not: OpenCV
# thresholds the mask anyway, so `n_inliers` moves and a tight threshold can trip
# `estimate/robust.py`'s four-inlier gate on a fit that was fine (measured while authoring this
# stage; the project previously believed the whole LMEDS row was flat, and `config/schema.py`
# said so). That makes it the stage's free falsification, on `h` rather than on any aggregate:
# block 4 checks it, and a mismatch means the knob being swept is not the knob reaching the
# solver -- exactly the failure PLAN.md §15A records in the upstream harness.
FIT_THRESHOLD_BLIND = "lmeds"

# Student-t 97.5th percentile at 4 degrees of freedom, for the n=5 seed interval of block 2.
# Named rather than inlined because it is the one number there that is an assumption.
_T_CRITICAL_N5 = 2.776


def run_dir_for(cell: Cell, seed: int) -> Path:
    return Path("runs") / f"staged_{cell.dataset}_s{seed}"


def run(cell: Cell, seed: int, device: str, dry: DryRun) -> bool:
    """One (dataset, seed) match pass, carrying all twelve estimation variants."""
    run_dir = run_dir_for(cell, seed)
    argv = [
        "bench",
        "-c",
        CONFIG,
        "--data",
        str(cell.manifest),
        "--domain",
        cell.domain,
        "--platform",
        cell.platform,
        "--matchers",
        ",".join(REDUCED_8),
        "--seed",
        str(seed),
        # The anchor, restated so this stage does not silently inherit a different one if the
        # anchor config is ever edited: it is the variant the runner prints, and a printed block
        # belonging to no column of the tables below would be unreadable.
        "--estimator",
        ANCHOR[0],
        "--threshold",
        f"{ANCHOR[1]:g}",
        "--sweep-estimators",
        ",".join(ESTIMATORS),
        "--sweep-thresholds",
        ",".join(f"{t:g}" for t in THRESHOLDS),
        "--device",
        device,
        "--wandb" if dry.wandb else "--no-wandb",
        "--name",
        run_dir.name,
        # One W&B group per dataset; `eval/runner.py::_group` extends it per (matcher, variant)
        # so the five seeds of one cell aggregate and the twelve variants stay apart.
        "--group",
        f"p3d_estimator_{cell.dataset}",
        "--run-dir",
        str(run_dir),
    ]
    # The `R` policy is per dataset and carried from `stages.CELLS` rather than restated:
    # `flir` composes, `dronevehicle` does not (GRID.md §3). An estimator axis does not touch it.
    if cell.calibration is not None:
        argv += ["--residual-calibration", str(cell.calibration)]
    if dry.limit is not None:
        argv += ["--limit", str(dry.limit)]
    # Appended after the stage's own list so a laptop smoke run overrides it -- and a row
    # produced that way is not a stage-D row, as `DryRun` says.
    if dry.matchers is not None:
        argv += ["--matchers", ",".join(dry.matchers)]
    banner = (
        f"########## {run_dir.name} -- {len(ESTIMATORS) * len(THRESHOLDS)} variants off one "
        f"match pass ({cell.why}) ##########"
    )
    return run_cell(cell, run_dir, argv, banner)


Series = dict[tuple[str, float, str], list[dict[str, float]]]
"""`{(estimator, threshold_px, matcher): one metric dict per seed}` for one dataset."""


Rows = dict[tuple[str, float, str], list[PairRow]]
"""`{(estimator, threshold_px, matcher): every row across every seed}` for one dataset."""


def read_dataset(cell: Cell, seeds: tuple[int, ...]) -> tuple[Series, Rows]:
    """One dataset's seeds, both summarised and raw.

    Read once and passed to every renderer below -- six of them want these numbers, and
    re-reading is six chances to read a different directory than the table above it. The raw
    rows come back too because the integrity check (block 4) is an assertion about individual
    homographies, and no aggregate can express it.
    """
    from cmreg.results import read_rows, summarize

    series: Series = defaultdict(list)
    rows_by_cell: Rows = defaultdict(list)
    for seed in seeds:
        rows = read_rows(run_dir_for(cell, seed))
        keys = list(dict.fromkeys((r.estimator, r.threshold_px, r.matcher) for r in rows))
        for key in keys:
            group = [r for r in rows if (r.estimator, r.threshold_px, r.matcher) == key]
            series[key].append(summarize(group, (SECONDARY_THRESHOLD,)).metrics)
            rows_by_cell[key].extend(group)
    return dict(series), dict(rows_by_cell)


def _matchers_in(series: Series) -> list[str]:
    """Every matcher present, in the order the cells ran them."""
    return list(dict.fromkeys(matcher for _, _, matcher in series))


def _mean_over_seeds(series: Series, key: tuple[str, float, str], metric: str) -> float:
    """The metric averaged across seeds, or NaN where no seed produced one.

    NaN is the honest answer for a cell that never succeeded -- `xfeat` under PROSAC, or a
    matcher that failed every pair -- and it renders as `--` rather than as a number.
    """
    values = [m[metric] for m in series.get(key, []) if not math.isnan(m[metric])]
    return statistics.fmean(values) if values else float("nan")


def _cell_text(series: Series, key: tuple[str, float, str]) -> str:
    """`mace | success@10px | failure rate`, for one cell.

    The failure rate rides along rather than living in a block of its own because the first two
    numbers only disagree when it moves, and reading that off two separate tables is how this
    stage was nearly recorded backwards (TASKS.md F34).
    """
    if key not in series:
        return "--"
    mace = _mean_over_seeds(series, key, HEADLINE)
    if math.isnan(mace):
        return "--"
    return (
        f"{mace:.2f} | {_mean_over_seeds(series, key, SECONDARY):.3f}"
        f" | {_mean_over_seeds(series, key, FAILURE_RATE):.2f}"
    )


def _short(metric: str) -> str:
    """`reg/success_rate_10px` -> `success_rate_10px`, for a column header that has to fit."""
    return metric.split("/")[-1]


def _fmt(metric: str, value: float) -> str:
    return f"{value:.4f}" if metric in _RATE_METRICS else f"{value:.2f}"


def _common_matchers(series: Series) -> list[str]:
    """The matchers *every* estimator produced geometry for.

    Every cross-estimator aggregate below is taken over this set rather than over each
    estimator's own matchers, and stage D is the reason. `xfeat` returns no per-match confidence
    and so has no PROSAC cell (P0-2), which made the printed median a median over seven matchers
    for PROSAC and over eight for the other three -- and dropping the second-worst matcher lifts
    it. On `flir` that inverted the result: `magsac@3px` 13.13 px against `prosac@3px` 11.01 as
    printed, 10.49 against 11.01 over the common seven (TASKS.md F37).

    Applicability is read off `reg/mace` whichever metric is being aggregated. A cell that solved
    no pair at all has no mace, while `reg/success_rate_10px` still reads 0.0 there -- a real
    number meaning "never ran", and averaging it in would charge an estimator for a matcher it
    could not be run against.
    """
    return [
        matcher
        for matcher in _matchers_in(series)
        if all(
            any(
                not math.isnan(_mean_over_seeds(series, (estimator, threshold, matcher), HEADLINE))
                for threshold in THRESHOLDS
            )
            for estimator in ESTIMATORS
        )
    ]


def _excluded_note(all_series: dict[str, Series]) -> str | None:
    """Name the matchers `_common_matchers` dropped, out of the run's own numbers.

    Over every dataset at once: the blocks that carry this note render all of them, and a note
    naming only the first dataset's exclusions would be a footnote that does not cover its table.
    """
    absent = list(
        dict.fromkeys(
            matcher
            for series in all_series.values()
            for matcher in _matchers_in(series)
            if matcher not in _common_matchers(series)
        )
    )
    if not absent:
        return None
    return (
        f"# medians are over the matchers every estimator solved; {', '.join(absent)} excluded, "
        "since a median\n#   over a different matcher set per estimator compares two "
        "populations (F37)."
    )


def _scored(series: Series) -> int:
    for metrics in series.values():
        return int(metrics[0][N_PAIRS])
    return 0


def _unsupported_note(rows: Rows, estimator: str) -> str | None:
    """Name the matchers this estimator could not run against, out of the run's own rows.

    Read from `failure_reason` rather than from a list kept here, so the note cannot drift from
    what actually happened. `xfeat` is the member of reduced-8 that returns no per-match
    confidence today (TASKS.md P0-2) and so the one PROSAC cannot order its samples for -- but a
    hardcoded name would go quietly wrong the moment vismatch changes, and a `--` with no
    explanation beside it is exactly the kind of hole X-4 exists to prevent.
    """
    absent = []
    for matcher in dict.fromkeys(m for _, _, m in rows):
        cells = [
            row
            for (method, _, name), group in rows.items()
            if method == estimator and name == matcher
            for row in group
        ]
        if cells and all(row.failure_reason == "estimator_needs_confidence" for row in cells):
            absent.append(matcher)
    if not absent:
        return None
    return (
        f"# `--` for {', '.join(absent)}: no per-match confidence, so {estimator} cannot order "
        "its samples (P0-2). Recorded as rows, not dropped (X-4)."
    )


def estimator_table(cell: Cell, estimator: str, series: Series, rows: Rows, seeds: int) -> str:
    """Matchers down, inlier threshold across, for one estimator.

    One table per estimator rather than one twelve-column table: this block reaches a human by
    copy-paste out of a server console, and twelve columns of `mace | rate` do not survive the
    trip.
    """
    matchers = _matchers_in(series)
    if not matchers:
        return ""
    width = max(len("matcher"), *(len(name) for name in matchers)) + 2
    composed = "composed" if cell.composes else "a floor, not an accuracy"
    header = f"{'matcher':<{width}}" + "".join(f"{f'{t:g}px':>{_CELL_WIDTH}}" for t in THRESHOLDS)
    lines = [
        f"=== CMREG STAGE D: {cell.dataset} / {estimator} ({composed}) -- {HEADLINE} px | "
        f"{SECONDARY} | {FAILURE_RATE} ===",
        f"# pairs scored: {_scored(series)}, mean over {seeds} seeds. columns are the "
        "estimator's INLIER threshold,",
        "#   which is unrelated to the reporting ladder the success rate is read at.",
        "# mace is a mean over SUCCESSES; the success rate counts a failure as infinite error.",
        "#   The third number is the failure rate, and it is what makes the first two disagree.",
    ]
    if note := _unsupported_note(rows, estimator):
        lines.append(note)
    if estimator == FIT_THRESHOLD_BLIND:
        lines.append(
            "# lmeds never reads the threshold when it FITS, so these columns differ only in "
            "which pairs survived:"
        )
        lines.append(
            "#   OpenCV still thresholds its inlier mask, so a tight column can lose pairs to "
            "the four-inlier gate. Block 4 checks the fit itself."
        )
    lines += [header, "-" * len(header)]
    for matcher in matchers:
        cells = "".join(
            f"{_cell_text(series, (estimator, t, matcher)):>{_CELL_WIDTH}}" for t in THRESHOLDS
        )
        lines.append(f"{matcher:<{width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def seed_block(all_series: dict[str, Series], seeds: tuple[int, ...], metric: str) -> str:
    """Does any estimator difference clear its own seed-to-seed noise, on one metric?

    The block the five seeds exist for, and the one that decides whether stage D has a finding
    or a flat row. Rendered once per entry in `AGGREGATE_METRICS`, because on this stage the two
    entries answer it differently: an estimator that declines the pairs it cannot solve shrinks
    a mean over successes and shrinks the success rate with it, so `reg/mace` alone reads the
    axis backwards (TASKS.md F33/F34).

    Medians over matchers throughout: `mace` spans 7-900 px across this project's matcher list,
    and one classical failure would otherwise set the statistic (P3-9's practice). Over
    `_common_matchers`, so that the twelve cells are medians of the same population.

    The interval is a Student-t one at n=5 and is the only assumption in this file; the seed
    *spread* beside it is assumption-free, and where the two disagree the spread is the one to
    believe.
    """
    label = f"median {_short(metric)}"
    column = max(len(label), 12) + 2
    header = f"{'dataset':<16}{'cell':>16}{label:>{column}}{'+-95% CI':>12}{'seed spread':>14}"
    lines = [
        f"=== CMREG STAGE D: is the estimator axis bigger than the seed noise? [{metric}] ===",
        f"# over {len(seeds)} seeds ({', '.join(str(s) for s in seeds)}), median over the "
        "matchers of each statistic.",
        "# CI is Student-t at n=5 on the per-seed means -- an assumption; `seed spread`",
        "#   (max - min across seeds) is not. Read the spread first.",
        "# FOOTER is the finding: an axis whose best-to-worst range is inside the seed spread",
        "#   has not measured anything, and X-4 says that is a row to report, not to hide.",
    ]
    if note := _excluded_note(all_series):
        lines.append(note)
    lines += [header, "-" * len(header)]
    for dataset, series in all_series.items():
        matchers = _common_matchers(series)
        levels: dict[tuple[str, float], float] = {}
        for estimator in ESTIMATORS:
            for threshold in THRESHOLDS:
                means, spreads, intervals = [], [], []
                for matcher in matchers:
                    values = [
                        m[metric]
                        for m in series.get((estimator, threshold, matcher), [])
                        if not math.isnan(m[metric])
                    ]
                    if not values:
                        continue
                    means.append(statistics.fmean(values))
                    spreads.append(max(values) - min(values))
                    intervals.append(
                        _T_CRITICAL_N5 * statistics.stdev(values) / math.sqrt(len(values))
                        if len(values) > 1
                        else float("nan")
                    )
                if not means:
                    continue
                level = statistics.median(means)
                levels[estimator, threshold] = level
                interval = [v for v in intervals if not math.isnan(v)]
                spread = statistics.median(interval) if interval else float("nan")
                lines.append(
                    f"{dataset:<16}{f'{estimator}@{threshold:g}px':>16}"
                    f"{_fmt(metric, level):>{column}}{_fmt(metric, spread):>12}"
                    f"{_fmt(metric, statistics.median(spreads)):>14}"
                )
        if len(levels) > 1:
            lines.append(
                f"{dataset:<16}{'-> best-worst':>16}"
                f"{_fmt(metric, max(levels.values()) - min(levels.values())):>{column}}"
                f"{'  (compare against the seed spread column above)':>26}"
            )
    lines.append("=== END ===")
    return "\n".join(lines)


def _spread(values: list[float]) -> float | None:
    present = [v for v in values if not math.isnan(v)]
    return max(present) - min(present) if len(present) > 1 else None


def axis_block(all_series: dict[str, Series], metric: str) -> str:
    """Which of the two axes carries the effect -- the estimator, or its threshold?

    The direct analogue of stage C's `axis_block`, and what settles the estimator and threshold
    for stages E-G. A threshold spread that is small against the estimator spread means the
    threshold is a free choice; ~1 means the paper has to report both.

    Rendered once per entry in `AGGREGATE_METRICS` for the reason `seed_block` gives, and the
    ratio is the place it showed most: stage D's aerial ratio is 5.59 on `reg/mace` and 2.12 on
    `reg/success_rate_10px`, because most of what the mace ratio measures is LMEDS declining
    pairs rather than registering them better (TASKS.md F33/F34).

    LMEDS is excluded from the *threshold* spread rather than left in: it ignores the threshold
    by construction, so including it would dilute that axis with three copies of one number and
    make the ratio a statement about the matcher list instead of about the axis.
    """
    responsive = [e for e in ESTIMATORS if e != FIT_THRESHOLD_BLIND]
    header = f"{'dataset':<16}{'across estimators':>20}{'across thresholds':>20}{'ratio':>10}"
    lines = [
        f"=== CMREG STAGE D: does the estimator matter, or only its threshold? [{metric}] ===",
        f"# medians of the {_short(metric)} spread, over the matchers, on the seed-averaged "
        "values.",
        "# across estimators = spread over the four estimators at a fixed threshold.",
        f"# across thresholds = spread over 1/3/5 px at a fixed estimator, "
        f"{FIT_THRESHOLD_BLIND} excluded (it ignores the threshold).",
        "# ratio = estimators / thresholds. Large means the estimator is the axis and the",
        "#   threshold is a free choice for stages E-G; ~1 means both have to be reported.",
    ]
    if note := _excluded_note(all_series):
        lines.append(note)
    lines += [header, "-" * len(header)]
    for dataset, series in all_series.items():
        over_estimators: list[float] = []
        over_thresholds: list[float] = []
        # The estimator spread is a cross-estimator statistic and so runs over the common
        # matchers; the threshold spread is taken inside one estimator and does not have to.
        common = _common_matchers(series)
        for matcher in _matchers_in(series):
            if matcher in common:
                for threshold in THRESHOLDS:
                    values = [
                        _mean_over_seeds(series, (e, threshold, matcher), metric)
                        for e in ESTIMATORS
                        if (e, threshold, matcher) in series
                    ]
                    if (spread := _spread(values)) is not None:
                        over_estimators.append(spread)
            for estimator in responsive:
                values = [
                    _mean_over_seeds(series, (estimator, t, matcher), metric)
                    for t in THRESHOLDS
                    if (estimator, t, matcher) in series
                ]
                if (spread := _spread(values)) is not None:
                    over_thresholds.append(spread)
        if not over_estimators or not over_thresholds:
            lines.append(f"{dataset:<16}{'needs both axes':>50}")
            continue
        estimator_spread = statistics.median(over_estimators)
        threshold_spread = statistics.median(over_thresholds)
        ratio = estimator_spread / threshold_spread if threshold_spread > 0.0 else float("inf")
        lines.append(
            f"{dataset:<16}{_fmt(metric, estimator_spread):>20}"
            f"{_fmt(metric, threshold_spread):>20}{ratio:>10.2f}"
        )
    lines.append("=== END ===")
    return "\n".join(lines)


def integrity_block(all_rows: dict[str, Rows]) -> str:
    """The sweep's own falsification: is the knob being swept the knob reaching the solver?

    Asserted on the **homographies**, per pair, because that is where the invariant actually
    lives and no aggregate can express it. LMEDS fits by minimising the median residual and
    never reads `threshold_px`, so on any pair it solves at all three thresholds it must return
    the *same matrix* three times. Every other estimator must move that matrix on at least some
    pairs -- if none of them do, the threshold is reaching nothing, which is PLAN.md §15A's bug,
    where the upstream harness's confidence landed in `cv2.findHomography`'s `mask` slot and
    silently never applied.

    The `solved` column is the second half of the finding, and it is why this is not a check on
    `reg/mace`. OpenCV thresholds LMEDS's inlier *mask* even though it does not threshold its
    fit, so a tight threshold moves `n_inliers` and can trip `estimate/robust.py`'s four-inlier
    gate on a fit that was perfectly good -- an LMEDS column can go empty at 1 px while being
    geometrically identical to the 5 px one wherever it survives. A check on any success-weighted
    aggregate would read that as a violation and be wrong.

    Free, because the sweep already produced both halves. Printed as a verdict rather than a
    table: a stage that has to be audited by eye is a stage that will not be.
    """
    header = (
        f"{'dataset':<16}{'estimator':>12}{'solved at all 3':>18}{'h identical':>14}"
        f"{'expected':>12}{'verdict':>10}"
    )
    lines = [
        "=== CMREG STAGE D: sweep integrity ===",
        f"# {FIT_THRESHOLD_BLIND} never reads the threshold, so on every pair it solves at all",
        "#   three its homography must be IDENTICAL. Every other estimator must move it on at",
        "#   least one pair, or the threshold is reaching nothing (PLAN.md §15A's bug).",
        "# `solved at all 3` is itself a result: OpenCV thresholds lmeds's inlier MASK even",
        "#   though it does not threshold its fit, so a tight threshold can fail a cell whose",
        "#   geometry was fine. That is why this check is on h and not on mace.",
        header,
        "-" * len(header),
    ]
    for dataset, rows in all_rows.items():
        matchers = list(dict.fromkeys(matcher for _, _, matcher in rows))
        for estimator in ESTIMATORS:
            solved = identical = 0
            for matcher in matchers:
                columns = [
                    {
                        (row.seed, row.stem): row.h
                        for row in rows.get((estimator, threshold, matcher), [])
                        if row.success
                    }
                    for threshold in THRESHOLDS
                ]
                if not all(columns):
                    continue
                shared = set(columns[0]).intersection(*(set(c) for c in columns[1:]))
                solved += len(shared)
                identical += sum(
                    1 for key in shared if all(c[key] == columns[0][key] for c in columns)
                )
            if not solved:
                lines.append(
                    f"{dataset:<16}{estimator:>12}{0:>18}{'--':>14}{'--':>12}{'NO DATA':>10}"
                )
                continue
            blind = estimator == FIT_THRESHOLD_BLIND
            # "all identical" for the blind one; "not all identical" for every other.
            verdict = "PASS" if (identical == solved) == blind else "**FAIL**"
            lines.append(
                f"{dataset:<16}{estimator:>12}{solved:>18}{identical:>14}"
                f"{('all' if blind else 'not all'):>12}{verdict:>10}"
            )
    lines.append("=== END ===")
    return "\n".join(lines)


def cost_block(all_series: dict[str, Series]) -> str:
    """Mean `time/estimate_ms` per (matcher, estimator), averaged over thresholds and seeds.

    **`time/total_ms` is deliberately not this block's column.** Every one of a swept run's
    twelve rows carries the same match cost, so summing or averaging `total_ms` over the sweep
    reports twelve times a bill the stage never paid. `estimate_ms` is the part that actually
    varies with the axis, and it is the only per-estimator cost this project measures -- P3-14
    and Figure 11 read it from here.
    """
    lines = [
        f"=== CMREG STAGE D: {TIME_ESTIMATE_MS} by estimator (mean over thresholds, seeds) ===",
        "# NOT time/total_ms: all twelve variants of a pair share one match, so a total over",
        "#   the sweep would report twelve times a cost the stage never paid.",
    ]
    names = [name for series in all_series.values() for name in _matchers_in(series)]
    width = max(len("dataset/matcher"), *(len(n) for n in names)) + 2 if names else 0
    for dataset, series in all_series.items():
        matchers = _matchers_in(series)
        if not matchers:
            continue
        header = f"{dataset + '/matcher':<{width}}" + "".join(f"{e:>14}" for e in ESTIMATORS)
        lines += ["", header, "-" * len(header)]
        for matcher in matchers:
            cells = ""
            for estimator in ESTIMATORS:
                values = [
                    _mean_over_seeds(series, (estimator, t, matcher), TIME_ESTIMATE_MS)
                    for t in THRESHOLDS
                    if (estimator, t, matcher) in series
                ]
                present = [v for v in values if not math.isnan(v)]
                cells += f"{statistics.fmean(present):>14.2f}" if present else f"{'--':>14}"
            lines.append(f"{matcher:<{width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def main_script(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device; 'auto' picks the best available (default: cuda)",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=list(SEEDS),
        help="restrict the seed axis; the interval in block 2 wants all five",
    )
    dry = parser.add_argument_group(
        "dry-run overrides",
        "For proving the plumbing on a laptop. A row produced under any of these is not a "
        "stage-D row and is not comparable with the rest of the table.",
    )
    dry.add_argument("--limit", type=int, help="override the config's 300-pair budget")
    dry.add_argument("--matchers", nargs="+", help="override the stage's reduced-8 matcher list")
    dry.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="skip W&B; X-1 forbids this for a real run, but the Mac has no login",
    )
    args = parser.parse_args(argv)
    overrides = DryRun(
        limit=args.limit,
        matchers=tuple(args.matchers) if args.matchers else None,
        wandb=args.wandb,
    )

    cells = [cell for cell in CELLS if cell.dataset in args.datasets]
    seeds = tuple(sorted(set(args.seeds)))
    print(
        f"########## {len(cells) * len(seeds)} match passes x "
        f"{len(ESTIMATORS) * len(THRESHOLDS)} estimation variants: "
        f"{', '.join(c.dataset for c in cells)} x seeds {', '.join(str(s) for s in seeds)} "
        "##########"
    )
    # Dataset-outer so an interrupted run leaves whole datasets finished rather than two
    # half-finished ones, and the tables below can be printed for what completed.
    completed: list[Cell] = []
    for cell in cells:
        # Every seed is attempted even when one is skipped, rather than short-circuiting: a
        # dataset is skipped for want of a manifest, and printing that once per pass is what
        # tells a reader of the pasted console which of the ten are missing.
        ran = [run(cell, seed, args.device, overrides) for seed in seeds]
        if all(ran):
            completed.append(cell)

    read = {cell.dataset: read_dataset(cell, seeds) for cell in completed}
    all_series = {dataset: series for dataset, (series, _) in read.items()}
    all_rows = {dataset: rows for dataset, (_, rows) in read.items()}
    for cell in completed:
        for estimator in ESTIMATORS:
            block = estimator_table(
                cell, estimator, all_series[cell.dataset], all_rows[cell.dataset], len(seeds)
            )
            if block:
                print()
                print(block, flush=True)
    if all_series:
        blocks = [
            *(seed_block(all_series, seeds, metric) for metric in AGGREGATE_METRICS),
            *(axis_block(all_series, metric) for metric in AGGREGATE_METRICS),
            integrity_block(all_rows),
            cost_block(all_series),
        ]
        for block in blocks:
            print()
            print(block, flush=True)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
