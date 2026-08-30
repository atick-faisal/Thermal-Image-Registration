"""P3-10 / Stage D: the estimator x threshold sweep, on the training server.

`experiments/GRID.md` §6 froze this stage as 4 estimators x threshold 1/3/5 px x
`driving+aerial` x 5 seeds = 24 cells x 5 = **120 `cmreg bench` invocations**, which at stage
C's measured ~19 min per reduced-8 cell is ~38 h. Almost all of it is waste.

**The estimator axis is downstream of the matcher.** `eval/runner.py::_evaluate` matches once
and then calls `estimate_homography`; `config.estimate` is the only thing this stage varies. The
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

from cmreg.metrics.schema import MACE, N_PAIRS, TIME_ESTIMATE_MS, success_rate_key
from cmreg.results import PairRow
from stages import CELLS, CONFIG, REDUCED_8, Cell, DryRun, run_cell

# `reg/mace` leads, as it does in stages A-C: P3-7's F7/F13 established that a thresholded rate
# is a function of the dataset's residual floor while a mean over corner errors is not, and
# `dronevehicle` here is a floor (GRID.md §3).
HEADLINE = MACE
SECONDARY_THRESHOLD = 10.0
SECONDARY = success_rate_key(SECONDARY_THRESHOLD)

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
    if key not in series:
        return "--"
    mace = _mean_over_seeds(series, key, HEADLINE)
    if math.isnan(mace):
        return "--"
    return f"{mace:.2f} | {_mean_over_seeds(series, key, SECONDARY):.3f}"


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
    header = f"{'matcher':<{width}}" + "".join(f"{f'{t:g}px':>22}" for t in THRESHOLDS)
    lines = [
        f"=== CMREG STAGE D: {cell.dataset} / {estimator} ({composed}) -- {HEADLINE} px | "
        f"{SECONDARY} ===",
        f"# pairs scored: {_scored(series)}, mean over {seeds} seeds. columns are the "
        "estimator's INLIER threshold,",
        "#   which is unrelated to the reporting ladder the success rate is read at.",
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
        cells = "".join(f"{_cell_text(series, (estimator, t, matcher)):>22}" for t in THRESHOLDS)
        lines.append(f"{matcher:<{width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def seed_block(all_series: dict[str, Series], seeds: tuple[int, ...]) -> str:
    """Does any estimator difference clear its own seed-to-seed noise?

    The block the five seeds exist for, and the one that decides whether stage D has a finding
    or a flat row. Medians over matchers throughout: `mace` spans 7-900 px across this project's
    matcher list, and one classical failure would otherwise set the statistic (P3-9's practice).

    The interval is a Student-t one at n=5 and is the only assumption in this file; the seed
    *spread* beside it is assumption-free, and where the two disagree the spread is the one to
    believe.
    """
    header = f"{'dataset':<16}{'cell':>16}{'median mace':>14}{'+-95% CI':>12}{'seed spread':>14}"
    lines = [
        "=== CMREG STAGE D: is the estimator axis bigger than the seed noise? ===",
        f"# over {len(seeds)} seeds ({', '.join(str(s) for s in seeds)}), median over the "
        "matchers of each statistic.",
        "# CI is Student-t at n=5 on the per-seed means -- an assumption; `seed spread`",
        "#   (max - min across seeds) is not. Read the spread first.",
        "# FOOTER is the finding: an axis whose best-to-worst range is inside the seed spread",
        "#   has not measured anything, and X-4 says that is a row to report, not to hide.",
        header,
        "-" * len(header),
    ]
    for dataset, series in all_series.items():
        levels: dict[tuple[str, float], float] = {}
        for estimator in ESTIMATORS:
            for threshold in THRESHOLDS:
                means, spreads, intervals = [], [], []
                for matcher in _matchers_in(series):
                    values = [
                        m[HEADLINE]
                        for m in series.get((estimator, threshold, matcher), [])
                        if not math.isnan(m[HEADLINE])
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
                lines.append(
                    f"{dataset:<16}{f'{estimator}@{threshold:g}px':>16}{level:>14.2f}"
                    f"{statistics.median(interval) if interval else float('nan'):>12.2f}"
                    f"{statistics.median(spreads):>14.2f}"
                )
        if len(levels) > 1:
            lines.append(
                f"{dataset:<16}{'-> best-worst':>16}"
                f"{max(levels.values()) - min(levels.values()):>14.2f}"
                f"{'  (compare against the seed spread column above)':>26}"
            )
    lines.append("=== END ===")
    return "\n".join(lines)


def _spread(values: list[float]) -> float | None:
    present = [v for v in values if not math.isnan(v)]
    return max(present) - min(present) if len(present) > 1 else None


def axis_block(all_series: dict[str, Series]) -> str:
    """Which of the two axes carries the effect -- the estimator, or its threshold?

    The direct analogue of stage C's `axis_block`, and what settles the estimator and threshold
    for stages E-G. A threshold spread that is small against the estimator spread means the
    threshold is a free choice; ~1 means the paper has to report both.

    LMEDS is excluded from the *threshold* spread rather than left in: it ignores the threshold
    by construction, so including it would dilute that axis with three copies of one number and
    make the ratio a statement about the matcher list instead of about the axis.
    """
    responsive = [e for e in ESTIMATORS if e != FIT_THRESHOLD_BLIND]
    header = f"{'dataset':<16}{'across estimators':>20}{'across thresholds':>20}{'ratio':>10}"
    lines = [
        "=== CMREG STAGE D: does the estimator matter, or only its threshold? ===",
        "# medians of the mace spread, over the matchers, on the seed-averaged values.",
        "# across estimators = spread over the four estimators at a fixed threshold.",
        f"# across thresholds = spread over 1/3/5 px at a fixed estimator, "
        f"{FIT_THRESHOLD_BLIND} excluded (it ignores the threshold).",
        "# ratio = estimators / thresholds. Large means the estimator is the axis and the",
        "#   threshold is a free choice for stages E-G; ~1 means both have to be reported.",
        header,
        "-" * len(header),
    ]
    for dataset, series in all_series.items():
        over_estimators: list[float] = []
        over_thresholds: list[float] = []
        for matcher in _matchers_in(series):
            for threshold in THRESHOLDS:
                values = [
                    _mean_over_seeds(series, (e, threshold, matcher), HEADLINE)
                    for e in ESTIMATORS
                    if (e, threshold, matcher) in series
                ]
                if (spread := _spread(values)) is not None:
                    over_estimators.append(spread)
            for estimator in responsive:
                values = [
                    _mean_over_seeds(series, (estimator, t, matcher), HEADLINE)
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
            f"{dataset:<16}{estimator_spread:>20.2f}{threshold_spread:>20.2f}{ratio:>10.2f}"
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
        for block in (
            seed_block(all_series, seeds),
            axis_block(all_series),
            integrity_block(all_rows),
            cost_block(all_series),
        ):
            print()
            print(block, flush=True)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
