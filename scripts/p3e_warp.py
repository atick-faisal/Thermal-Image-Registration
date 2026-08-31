"""P3-11 / Stage E: the warp-model axis, on the training server.

`experiments/GRID.md` §6 froze this stage as homography / affine / similarity x
`driving+aerial` x reduced-8 x 1 seed. The naive reading is 6 `cmreg bench` invocations; the
warp model is **downstream of the matcher** exactly as the estimator is, so it is a third axis
of the same `EstimateConfig` sweep and the stage is **2 match passes carrying 6 estimation
variants each**.

    uv run python scripts/p3e_warp.py
    uv run python scripts/p3e_warp.py --datasets flir

**Sweeping is not the cheaper option here and is not taken for cost.** At three models the six
naive cells would run ~1.3 h and two swept passes ~1.5-1.8 h (stage D measured a swept pass at
~3.8 single-variant cells -- a variant is an estimate *and* a score, F38). It is taken because
one match pass per dataset removes the matcher as a source of between-column difference, which
is the entire point of an axis whose columns differ by four degrees of freedom.

**The estimator cannot be held fixed across the axis** (P3-4a F39). OpenCV fits a 4-DoF
similarity by RANSAC and LMEDS only; every USAC method, MAGSAC included, raises. Holding the
anchor's MAGSAC fixed would therefore confound the model axis with the estimator axis on exactly
one of three columns. So the stage carries a **RANSAC control column** across all three models --
the one estimator all three admit -- and reports MAGSAC for the two that admit it.
`similarity/magsac` is recorded as `estimator_unsupported_for_warp` rows and prints as `--`,
because aborting there would discard five variants after the matching they depend on was paid
for (X-4).

**The stage's conclusion is bounded before it runs** (P3-4a F40/F45). Tier-1 samples a
*projective* warp, so a restricted model carries a floor: on a 640-wide pair the best possible
affine is ~11.2 px and the best possible similarity ~15.6 px, against a 5 px headline. A
restricted model therefore *cannot* win here, and the question the stage actually answers is
narrower: **how much of the homography's advantage is the extra capacity delivering, and how
much is the ground truth simply handing it over?** Block 2 is that question, and it is asked per
pair -- a dataset-mean floor beside a dataset-mean MACE cannot separate a model that sits 2 px
above its floor everywhere from one that is at its floor on half the pairs and 4 px above it on
the rest.

**Two corrections carried from stage D's own output** (GRID.md §6): aggregate blocks are
rendered once per metric and never on a success-conditioned mean alone -- `reg/mace` is a mean
over successes, so a column that declines the pairs it cannot solve buys it with its failure
rate (F33/F34) -- and every cross-column median runs over `_common_matchers`, the matchers each
arm actually has, because dropping the second-worst matcher from one arm inverted an ordering in
stage D (F37).

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

# `reg/mace` leads, as it does in stages A-D: P3-7's F7/F13 established that a thresholded rate
# is a function of the dataset's residual floor while a mean over corner errors is not.
HEADLINE = MACE
SECONDARY_THRESHOLD = 10.0
SECONDARY = success_rate_key(SECONDARY_THRESHOLD)

# Both lead in the aggregate blocks, for the reason stage D's output forced (F33/F34): `reg/mace`
# is a mean over *successes* while `reg/success_rate_10px` charges a failure as infinite error,
# so a column that declines its hard pairs improves the first and loses the second. The
# restricted models here have every incentive to decline -- their fits are worse by construction.
AGGREGATE_METRICS = (HEADLINE, SECONDARY)

# Two decimals for a pixel error, four for a rate.
_RATE_METRICS = (SECONDARY, FAILURE_RATE)

# `mace | success | failure` at this project's widest cell; 24 keeps a gap between columns.
_CELL_WIDTH = 24

# GRID.md §6's `driving+aerial`: the best-characterised driving set and the only aerial one.
DATASETS = ("flir", "dronevehicle")
# The whole axis (TASKS.md P3-4). TPS and homography + residual flow are P3-4b -- they are
# dense-field-valued and break `PairRow.h`, `corner_error` and P2-12's composition, so a
# five-column stage E waits for them and a three-column one does not.
MODELS = ("homography", "affine", "similarity")
# MAGSAC is the anchor (GRID.md §1, frozen by stage D). RANSAC is the **control column** and not
# a second result: it is the only estimator all three models admit, so it is the one column in
# which the model axis is unconfounded (F39).
ESTIMATORS = ("magsac", "ransac")
CONTROL = "ransac"
# Frozen by stage D: MAGSAC's threshold row is flat inside the seed spread, so 3 px is a free
# choice rather than a tuned one (F35). Held fixed here so the stage varies one axis.
THRESHOLD = 3.0
# The variant whose console block the runner prints. `EstimateConfig` enforces that it is one of
# the swept cells rather than leaving it to this file.
ANCHOR = ("homography", "magsac")
# One seed. GRID.md §5: an "A beats B" claim gets five, a scoping row does not -- and the model
# axis here is separated by floors of 11-16 px against a 2.45-3.67 px seed spread (F35), so it
# clears the noise by an order of magnitude without them. The RANSAC control column does *not*
# clear it and is read as a scoping row, which is what stage D's F33 already established for
# every estimator difference.
SEED = 0


def run_dir_for(cell: Cell) -> Path:
    return Path("runs") / f"stagee_{cell.dataset}"


def argv_for(cell: Cell) -> list[str]:
    """The cell's **scientific** flags -- everything that enters `config_hash`.

    Separated from the invocation flags below so that `floors_for` and the tests can resolve the
    very config the run used, through `cmreg`'s own parser, rather than rebuilding it. A second
    path to the same answer is a second path that can disagree with the run it describes -- the
    argument `stages.intended_hash` makes, and this is the third consumer of it.
    """
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
        str(SEED),
        # The anchor, restated so this stage does not silently inherit a different one if the
        # anchor config is ever edited: it is the variant the runner prints, and a printed block
        # belonging to no column of the tables below would be unreadable.
        "--warp-model",
        ANCHOR[0],
        "--estimator",
        ANCHOR[1],
        "--threshold",
        f"{THRESHOLD:g}",
        "--sweep-warp-models",
        ",".join(MODELS),
        "--sweep-estimators",
        ",".join(ESTIMATORS),
    ]
    # The `R` policy is per dataset and carried from `stages.CELLS` rather than restated:
    # `flir` composes, `dronevehicle` does not (GRID.md §3). A warp-model axis does not touch it
    # -- but it *does* change the floor those rows are read against, which is why `floors_for`
    # composes whatever this resolves to rather than a constant of its own.
    if cell.calibration is not None:
        argv += ["--residual-calibration", str(cell.calibration)]
    return argv


def run(cell: Cell, device: str, dry: DryRun) -> bool:
    """One dataset's match pass, carrying all six estimation variants."""
    run_dir = run_dir_for(cell)
    argv = [
        *argv_for(cell),
        "--device",
        device,
        "--wandb" if dry.wandb else "--no-wandb",
        "--name",
        run_dir.name,
        # One W&B group per dataset; `eval/runner.py::_group` extends it per (matcher, variant)
        # so the six variants stay apart.
        "--group",
        f"p3e_warp_{cell.dataset}",
        "--run-dir",
        str(run_dir),
    ]
    if dry.limit is not None:
        argv += ["--limit", str(dry.limit)]
    # Appended after the stage's own list so a laptop smoke run overrides it -- and a row
    # produced that way is not a stage-E row, as `DryRun` says.
    if dry.matchers is not None:
        argv += ["--matchers", ",".join(dry.matchers)]
    banner = (
        f"########## {run_dir.name} -- {len(MODELS) * len(ESTIMATORS)} variants off one "
        f"match pass ({cell.why}) ##########"
    )
    return run_cell(cell, run_dir, argv, banner)


Series = dict[tuple[str, str, str], list[dict[str, float]]]
"""`{(warp, estimator, matcher): [one metric dict]}` for one dataset. A list of one, so the
renderers below are the same shape as stage D's, which averages over five seeds."""


Rows = dict[tuple[str, str, str], list[PairRow]]
"""`{(warp, estimator, matcher): every row}` for one dataset."""


Floors = dict[str, dict[str, float]]
"""`{stem: {warp model: floor px}}` -- the per-pair model floor of the scored truths."""


def floors_for(cell: Cell, rows: Rows) -> Floors:
    """Each scored pair's own model floor, against the truth its row was scored on.

    The stage is unreadable without this (P3-4a F40) and it is only sharp per pair: an affine
    that sits 2 px above its floor on every pair and one that is *at* its floor on half of them
    and 4 px above on the rest have the same dataset mean and are not the same result.

    Resolved through `cmreg`'s own parser and `data/splits.py::select_pairs`, not rebuilt here --
    a second path to "which pairs did this run score, under which warps" is a second path that
    can disagree with the run it is describing (the argument `stages.intended_hash` makes). The
    truth is built by `p3_warp_floor.py::scored_truths`, which reproduces
    `eval/runner.py::_load_pair`: `inv(H_gt)`, composed with `R` where the config names one. F45
    is the record of what floors against `H_gt` instead cost -- nothing in aggregate, and a
    per-pair correlation of 0.70/0.55, which is exactly what this block would have consumed.

    Costs seconds: no matcher, no GPU, no image decoded. The shape comes from the rows
    themselves, so a dataset whose pairs differ in resolution cannot silently be floored at one.
    """
    from cmreg.cli import build_parser, overrides_from_args
    from cmreg.config import Config, WarpModel
    from cmreg.data import DatasetManifest, select_pairs
    from cmreg.gt import load_calibration
    from cmreg.metrics import model_floor
    from p3_warp_floor import scored_truths

    args = build_parser().parse_args(argv_for(cell))
    config = Config.load(args.config, overrides_from_args(args))
    manifest = DatasetManifest.load(config.data.manifest)
    selected = select_pairs(
        manifest.images(config.data.split), config.data.limit, config.data.subsample_seed
    )
    shapes = _shapes_by_stem(rows)
    calibration = (
        load_calibration(config.gt.residual_calibration)
        if config.gt.residual_calibration is not None
        else None
    )

    floors: Floors = {}
    for index, path in selected:
        shape = shapes.get(path.stem)
        if shape is None:
            # The pair was never scored -- unreadable, or dropped for a shape mismatch. It has
            # no row to be floored beside, so it has no floor either.
            continue
        truth = scored_truths(shape, config.gt, (index,), calibration)[0]
        floors[path.stem] = {model.value: model_floor(truth, shape, model) for model in WarpModel}
    return floors


def _shapes_by_stem(rows: Rows) -> dict[str, tuple[int, int]]:
    """Each scored pair's reference shape, read off its own row rather than assumed."""
    shapes: dict[str, tuple[int, int]] = {}
    for group in rows.values():
        for row in group:
            if row.height is not None and row.width is not None:
                shapes[row.stem] = (row.height, row.width)
    return shapes


def read_dataset(cell: Cell) -> tuple[Series, Rows]:
    """One dataset's run directory, both summarised and raw.

    Read once and passed to every renderer below -- four of them want these numbers, and
    re-reading is four chances to read a different directory than the table above it. The raw
    rows come back too because block 2 is a per-pair statement and no aggregate can express it.
    """
    from cmreg.results import read_rows, summarize

    series: Series = defaultdict(list)
    rows_by_cell: Rows = defaultdict(list)
    rows = read_rows(run_dir_for(cell))
    for key in dict.fromkeys((r.warp, r.estimator, r.matcher) for r in rows):
        group = [r for r in rows if (r.warp, r.estimator, r.matcher) == key]
        series[key].append(summarize(group, (SECONDARY_THRESHOLD,)).metrics)
        rows_by_cell[key].extend(group)
    return dict(series), dict(rows_by_cell)


def _matchers_in(series: Series) -> list[str]:
    """Every matcher present, in the order the cells ran them."""
    return list(dict.fromkeys(matcher for _, _, matcher in series))


def _metric(series: Series, key: tuple[str, str, str], metric: str) -> float:
    """One cell's metric, or NaN where it produced none.

    NaN is the honest answer for a cell that never succeeded -- `similarity/magsac`, which cannot
    run at all -- and it renders as `--` rather than as a number.
    """
    values = [m[metric] for m in series.get(key, []) if not math.isnan(m[metric])]
    return statistics.fmean(values) if values else float("nan")


def _cell_text(series: Series, key: tuple[str, str, str]) -> str:
    """`mace | success@10px | failure rate`, for one cell.

    The failure rate rides along rather than living in a block of its own because the first two
    numbers only disagree when it moves -- stage D was nearly recorded backwards for want of it
    (F34), and a restricted model has more reason to decline a pair than any estimator does.
    """
    if key not in series:
        return "--"
    mace = _metric(series, key, HEADLINE)
    if math.isnan(mace):
        return "--"
    return (
        f"{mace:.2f} | {_metric(series, key, SECONDARY):.3f}"
        f" | {_metric(series, key, FAILURE_RATE):.2f}"
    )


def _short(metric: str) -> str:
    """`reg/success_rate_10px` -> `success_rate_10px`, for a column header that has to fit."""
    return metric.split("/")[-1]


def _fmt(metric: str, value: float) -> str:
    return f"{value:.4f}" if metric in _RATE_METRICS else f"{value:.2f}"


def _common_matchers(series: Series) -> list[str]:
    """The matchers *every* model produced geometry for, under at least one estimator.

    Every cross-model aggregate below is taken over this set rather than over each model's own
    matchers. Stage D is the reason: `xfeat` has no PROSAC cell, which made one arm's median a
    median over seven matchers and the others' over eight -- and dropping the second-worst
    matcher lifts it enough to invert the ordering (F37).

    Applicability is read off `reg/mace` whichever metric is being aggregated. A cell that solved
    no pair has no mace, while `reg/success_rate_10px` still reads 0.0 there -- a real number
    meaning "never ran", and averaging it in would charge a model for a matcher it never had.
    """
    return [
        matcher
        for matcher in _matchers_in(series)
        if all(
            any(
                not math.isnan(_metric(series, (model, estimator, matcher), HEADLINE))
                for estimator in ESTIMATORS
            )
            for model in MODELS
        )
    ]


def _excluded_note(all_series: dict[str, Series]) -> str | None:
    """Name the matchers `_common_matchers` dropped, out of the run's own numbers."""
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
        f"# medians are over the matchers every model solved; {', '.join(absent)} excluded, "
        "since a median\n#   over a different matcher set per model compares two populations "
        "(F37)."
    )


def _scored(series: Series) -> int:
    for metrics in series.values():
        return int(metrics[0][N_PAIRS])
    return 0


def _is_unsupported(rows: Rows, key: tuple[str, str, str]) -> bool:
    """True when every row of this cell is a capability gap rather than a measurement.

    `estimator_unsupported_for_warp` rows carry `estimate_ms=0.0` (`eval/runner.py::
    _unsupported_row`), which is correct -- nothing was estimated -- and reads in a cost table as
    "free". A cell that could not run has no cost, so it prints `--`.
    """
    group = rows.get(key, [])
    return bool(group) and all(
        row.failure_reason == "estimator_unsupported_for_warp" for row in group
    )


def _unsupported_note(rows: Rows, estimator: str) -> str | None:
    """Name the (model, matcher) cells this estimator could not fit, out of the run's own rows.

    Read from `failure_reason` rather than from a list kept here, so the note cannot drift from
    what happened. Today that is `similarity` under every USAC method (P3-4a F39), and
    `estimate/robust.py::SUPPORTED_ESTIMATORS` asserts the gap against the installed OpenCV --
    so a build that closes it fails the suite instead of leaving this note standing. A `--` with
    no explanation beside it is exactly the hole X-4 exists to prevent.
    """
    absent = []
    for model in MODELS:
        cells = [
            row
            for (warp, method, _), group in rows.items()
            if warp == model and method == estimator
            for row in group
        ]
        if cells and all(row.failure_reason == "estimator_unsupported_for_warp" for row in cells):
            absent.append(model)
    if not absent:
        return None
    return (
        f"# `--` for {', '.join(absent)}: OpenCV cannot fit that model by {estimator} "
        "(P3-4a F39). Recorded as rows, not dropped (X-4)."
    )


def _floor_text(floors: Floors, model: str) -> str:
    values = [pair[model] for pair in floors.values() if model in pair]
    return f"{statistics.fmean(values):.2f}" if values else "--"


def model_table(cell: Cell, estimator: str, series: Series, rows: Rows, floors: Floors) -> str:
    """Matchers down, warp models across, for one estimator -- **each column over its floor**.

    One table per estimator rather than one six-column table: this block reaches a human by
    copy-paste out of a server console, and six columns of `mace | rate | rate` do not survive
    the trip.

    The floor row is not decoration. GRID.md §3 reports every composed row beside `R` for the
    same reason: a restricted model's number is a statement about registration only once the
    part of it the ground truth put there is on the page beside it.
    """
    matchers = _matchers_in(series)
    if not matchers:
        return ""
    width = max(len("matcher"), len("FLOOR (mean)"), *(len(name) for name in matchers)) + 2
    composed = "composed" if cell.composes else "a floor, not an accuracy"
    header = f"{'matcher':<{width}}" + "".join(f"{m:>{_CELL_WIDTH}}" for m in MODELS)
    lines = [
        f"=== CMREG STAGE E: {cell.dataset} / {estimator} ({composed}) -- {HEADLINE} px | "
        f"{SECONDARY} | {FAILURE_RATE} ===",
        f"# pairs scored: {_scored(series)}, seed {SEED}. Columns are the fitted WARP MODEL;"
        f" estimator held at {estimator} @ {THRESHOLD:g}px.",
        "# mace is a mean over SUCCESSES; the success rate counts a failure as infinite error.",
        "#   The third number is the failure rate, and it is what makes the first two disagree.",
        "# FLOOR is the smallest mace this model could have on these pairs, before any matcher",
        "#   (P3-4a F40/F45). A restricted column cannot beat it, so read the gap, not the value.",
    ]
    if note := _unsupported_note(rows, estimator):
        lines.append(note)
    if estimator == CONTROL:
        lines.append(
            "# ransac is the CONTROL column: the only estimator all three models admit, so it is"
        )
        lines.append(
            "#   the one table in which the model axis is not confounded with the estimator (F39)."
        )
    lines += [header, "-" * len(header)]
    lines.append(
        f"{'FLOOR (mean)':<{width}}"
        + "".join(f"{_floor_text(floors, m):>{_CELL_WIDTH}}" for m in MODELS)
    )
    for matcher in matchers:
        cells = "".join(
            f"{_cell_text(series, (m, estimator, matcher)):>{_CELL_WIDTH}}" for m in MODELS
        )
        lines.append(f"{matcher:<{width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def _paired(rows: Rows, floors: Floors, key: tuple[str, str, str]) -> list[tuple[float, float]]:
    """`(corner_err, that pair's floor)` for every pair this cell solved and can be floored."""
    model = key[0]
    return [
        (row.corner_err, floors[row.stem][model])
        for row in rows.get(key, [])
        if row.success and row.corner_err is not None and row.stem in floors
    ]


def _medians(rows: Rows, floors: Floors, key: tuple[str, str, str]) -> tuple[float, float, float]:
    """`(median error, median excess, median floor)` for one cell, over **one** population.

    All three medians and not a mix of means: `reg/mace` is a mean over successes, and one
    classical failure sets it (P3-9's practice, and stage D's F34 is the record of what reading
    a success-conditioned mean beside anything else costs). More to the point here, the excess
    is a *paired* quantity -- each pair's error minus that pair's own floor -- so pairing it with
    a mean taken over a different summary would compare two statistics of location and call the
    difference a finding.

    Per pair and not per dataset, because that is the only form in which the excess means
    anything: the floor varies pair to pair by more than the models differ from each other
    (p90 19 px against an 11 px mean on a 640-wide set), so a mean-minus-mean would be dominated
    by which pairs each column happened to solve.
    """
    paired = _paired(rows, floors, key)
    if not paired:
        return (float("nan"), float("nan"), float("nan"))
    return (
        statistics.median([error for error, _ in paired]),
        statistics.median([error - floor for error, floor in paired]),
        statistics.median([floor for _, floor in paired]),
    )


def excess_block(
    all_rows: dict[str, Rows], all_series: dict[str, Series], all_floors: dict[str, Floors]
) -> str:
    """Is the model axis measuring capacity, or the ground truth's perspective content?

    **The block this stage exists for**, and GRID.md §6 states its rule before the run: *"If the
    measured gap between a model and its own floor is smaller than the gap between models, the
    axis is reporting the ground truth's perspective content and the paper should say so"* (X-4).

    Both quantities on one line so the reader does not have to divide two tables:

    * `excess` -- median over pairs of `corner_err - floor`, then median over matchers. What the
      model got *wrong on its own account*, with the part the ground truth imposed removed.
    * `corner_err` -- median over the same pairs, i.e. what a table without a floor column
      would report as the finding. **Not `reg/mace`**: that is a mean over successes, and a
      paired per-pair quantity cannot be subtracted from it.

    The footer divides the spread of the first by the second: **the fraction of the between-model
    gap that survives the floor correction.** Below half, most of what the axis appears to measure
    was already in the ground truth before any matcher ran. It cannot exceed 1 whenever the model
    with the larger floor also scores worse -- the expected case -- so it is read as a fraction and
    not as a two-sided test.

    A **negative** excess is free falsification: the floor is a strict lower bound, so no fit of
    that model can score below it, and a column that does means the floor was measured against a
    different truth than the rows were scored on -- which is exactly what P3-4a F45 was. Flagged
    rather than left to be noticed.

    Rendered under the control estimator, where all three models exist.
    """
    header = f"{'dataset':<16}{'model':>14}{'excess over floor':>20}{'corner_err':>12}{'floor':>10}"
    lines = [
        "=== CMREG STAGE E: capacity, or the ground truth? ===",
        f"# under the control estimator ({CONTROL}), the one all three models admit (F39).",
        "# excess = median over pairs of (corner_err - that pair's own floor), median over",
        "#   matchers. The part of the error the MODEL is responsible for.",
        "# All three columns are medians over the SAME pairs -- not reg/mace, which is a mean",
        "#   over successes and cannot be subtracted from a paired per-pair quantity.",
        "# FOOTER is the finding: the ratio is the fraction of the between-model gap that",
        "#   SURVIVES the floor. Below 0.5, the axis is mostly reporting Tier-1's perspective",
        "#   jitter and the paper must say so (X-4). A negative excess falsifies the floor.",
    ]
    if note := _excluded_note(all_series):
        lines.append(note)
    lines += [header, "-" * len(header)]
    for dataset, rows in all_rows.items():
        series, floors = all_series[dataset], all_floors[dataset]
        matchers = _common_matchers(series)
        excesses: dict[str, float] = {}
        errors: dict[str, float] = {}
        for model in MODELS:
            cells = [
                _medians(rows, floors, (model, CONTROL, matcher))
                for matcher in matchers
                if not math.isnan(_medians(rows, floors, (model, CONTROL, matcher))[0])
            ]
            if not cells:
                lines.append(f"{dataset:<16}{model:>14}{'--':>20}{'--':>12}{'--':>10}")
                continue
            errors[model] = statistics.median([error for error, _, _ in cells])
            excesses[model] = statistics.median([excess for _, excess, _ in cells])
            floor = statistics.median([value for _, _, value in cells])
            lines.append(
                f"{dataset:<16}{model:>14}{excesses[model]:>20.2f}"
                f"{errors[model]:>12.2f}{floor:>10.2f}"
            )
        if len(errors) > 1 and len(excesses) > 1:
            model_gap = max(errors.values()) - min(errors.values())
            excess_gap = max(excesses.values()) - min(excesses.values())
            if model_gap <= 0.0:
                # No axis to attribute. Reported rather than divided by zero.
                lines.append(
                    f"{dataset:<16}{'-> gap':>14}{excess_gap:>20.2f}{0.0:>12.2f}{'--':>10}"
                )
                lines.append(f"{'':<16}{'':>14}   FLAT: the three models scored the same")
            else:
                ratio = excess_gap / model_gap
                verdict = (
                    "CAPACITY: most of the between-model gap survives the floor"
                    if ratio >= 0.5
                    else "GROUND TRUTH: most of the between-model gap is floor, not registration"
                )
                lines.append(
                    f"{dataset:<16}{'-> gap':>14}{excess_gap:>20.2f}"
                    f"{model_gap:>12.2f}{ratio:>10.2f}"
                )
                lines.append(f"{'':<16}{'':>14}   {verdict}")
            if any(value < 0.0 for value in excesses.values()):
                below = ", ".join(m for m, v in excesses.items() if v < 0.0)
                lines.append(
                    f"{'':<16}{'':>14}   **CHECK**: {below} scored BELOW a strict lower bound; "
                    "the floor is not the truth these rows were scored on (F45)"
                )
    lines.append("=== END ===")
    return "\n".join(lines)


def control_block(all_series: dict[str, Series], metric: str) -> str:
    """What the control column cost: MAGSAC against RANSAC, on the models that admit both.

    The stage has to run RANSAC because `similarity` admits nothing else, and a reader is
    entitled to ask what changing the estimator did to the two columns that did not have to.
    Rendered once per entry in `AGGREGATE_METRICS` and over `_common_matchers` -- the two
    corrections stage D's own output forced (F34, F37).

    Read as a scoping row, not a result: P3-10's F33 already measured the four estimators as
    sitting inside the seed spread on the failure-inclusive metric, and this stage carries one
    seed, so a difference here is bounded by that finding rather than establishing a new one.
    """
    label = f"median {_short(metric)}"
    column = max(len(label), 12) + 2
    header = f"{'dataset':<16}{'model':>14}{'magsac':>{column}}{'ransac':>{column}}{'delta':>12}"
    lines = [
        f"=== CMREG STAGE E: what the control column cost [{metric}] ===",
        f"# medians over the matchers, at {THRESHOLD:g}px. `similarity` has no magsac cell at "
        "all (F39),",
        "#   which is why ransac is the column the model axis is actually read in.",
        "# One seed: read this as scoping. Stage D's F33 put every estimator difference inside",
        "#   the seed spread on the failure-inclusive metric, and that bound still holds here.",
    ]
    if note := _excluded_note(all_series):
        lines.append(note)
    lines += [header, "-" * len(header)]
    for dataset, series in all_series.items():
        matchers = _common_matchers(series)
        for model in MODELS:
            cells: dict[str, float] = {}
            for estimator in ESTIMATORS:
                values = [
                    _metric(series, (model, estimator, matcher), metric)
                    for matcher in matchers
                    if not math.isnan(_metric(series, (model, estimator, matcher), metric))
                ]
                if values:
                    cells[estimator] = statistics.median(values)
            if not cells:
                continue
            texts = [_fmt(metric, cells[e]) if e in cells else "--" for e in ("magsac", CONTROL)]
            delta = (
                _fmt(metric, cells["magsac"] - cells[CONTROL])
                if "magsac" in cells and CONTROL in cells
                else "--"
            )
            lines.append(
                f"{dataset:<16}{model:>14}{texts[0]:>{column}}{texts[1]:>{column}}{delta:>12}"
            )
    lines.append("=== END ===")
    return "\n".join(lines)


def cost_block(all_series: dict[str, Series], all_rows: dict[str, Rows]) -> str:
    """Mean `time/estimate_ms` per (matcher, model, estimator).

    **`time/total_ms` is deliberately not this block's column.** All six of a swept run's rows
    for a pair carry the same match cost, so summing or averaging `total_ms` over the sweep
    reports six times a bill the stage never paid. `estimate_ms` is the part that varies with the
    axis, and stage D measured vanilla RANSAC running to its iteration cap at cross-modal inlier
    ratios (5-20x MAGSAC). This stage runs RANSAC on three models, so this block is where the
    price of the control column is measured rather than extrapolated (Figure 11, P3-14).

    A cell OpenCV cannot fit prints `--` and not `0.00`: its rows carry `estimate_ms=0.0`
    truthfully, and in a cost table that reads as the cheapest column on the page.
    """
    lines = [
        f"=== CMREG STAGE E: {TIME_ESTIMATE_MS} by model and estimator ===",
        "# NOT time/total_ms: all six variants of a pair share one match, so a total over the",
        "#   sweep would report six times a cost the stage never paid.",
    ]
    names = [name for series in all_series.values() for name in _matchers_in(series)]
    width = max(len("dataset/matcher"), *(len(n) for n in names)) + 2 if names else 0
    columns = [(model, estimator) for model in MODELS for estimator in ESTIMATORS]
    for dataset, series in all_series.items():
        matchers = _matchers_in(series)
        if not matchers:
            continue
        rows = all_rows[dataset]
        header = f"{dataset + '/matcher':<{width}}" + "".join(
            f"{f'{m[:4]}/{e[:6]}':>14}" for m, e in columns
        )
        lines += ["", header, "-" * len(header)]
        for matcher in matchers:
            cells = ""
            for model, estimator in columns:
                key = (model, estimator, matcher)
                value = _metric(series, key, TIME_ESTIMATE_MS)
                if math.isnan(value) or _is_unsupported(rows, key):
                    cells += f"{'--':>14}"
                else:
                    cells += f"{value:>14.2f}"
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
    dry = parser.add_argument_group(
        "dry-run overrides",
        "For proving the plumbing on a laptop. A row produced under any of these is not a "
        "stage-E row and is not comparable with the rest of the table.",
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
    print(
        f"########## {len(cells)} match passes x {len(MODELS) * len(ESTIMATORS)} estimation "
        f"variants: {', '.join(c.dataset for c in cells)} ##########"
    )
    completed = [cell for cell in cells if run(cell, args.device, overrides)]

    read = {cell.dataset: read_dataset(cell) for cell in completed}
    all_series = {dataset: series for dataset, (series, _) in read.items()}
    all_rows = {dataset: rows for dataset, (_, rows) in read.items()}
    all_floors = {
        cell.dataset: floors_for(cell, all_rows[cell.dataset])
        for cell in completed
        if cell.dataset in all_rows
    }
    for cell in completed:
        for estimator in ESTIMATORS:
            block = model_table(
                cell,
                estimator,
                all_series[cell.dataset],
                all_rows[cell.dataset],
                all_floors[cell.dataset],
            )
            if block:
                print()
                print(block, flush=True)
    if all_series:
        blocks = [
            excess_block(all_rows, all_series, all_floors),
            *(control_block(all_series, metric) for metric in AGGREGATE_METRICS),
            cost_block(all_series, all_rows),
        ]
        for block in blocks:
            print()
            print(block, flush=True)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
