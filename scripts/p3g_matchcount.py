"""P3-12c / Stage F': the match-count axis, on the training server.

The downstream half of P3-12 and PLAN.md §7's "number of sampled matches". Stage F
(`scripts/p3f_resolution.py`) resizes the images and therefore pays one match pass per level;
this axis subsamples the *correspondences* between matching and estimation, so seventeen cells
cost **two match passes** -- one per dataset -- exactly as stages D and E do.

    uv run python scripts/p3g_matchcount.py
    uv run python scripts/p3g_matchcount.py --datasets flir

**"Match count" is the correspondences fed to the fit, not the matcher's own budget**, and that
reading is forced rather than chosen (P3-12c F70). `MatchConfig.max_keypoints` is not honoured by
the detector-free backends -- `minima-roma` returns 10 000 matches under a 2 048 budget,
`matchanything-roma` has no budget at all -- so an axis expressed through it is flat for half of
reduced-8 for a reason that is not the number of matches, and the half that does respond answers
a different question (a cheaper matcher) from the half that does not.

Absolute caps, not per-matcher fractions
----------------------------------------
The open question TASKS.md P3-12c-b left this driver, settled against stage A's own `match/total`
column (`MEASURED_YIELD` below) rather than against P0-2's single-pair probe. The two are
different measurements -- an absolute cap is a fixed correspondence budget, a fraction is
decimation -- and absolute wins on four counts:

* it is what P3-12c-a implements. `EstimateConfig.sweep_max_matches` is a descending tuple of
  ints with a validator to match, so fractions would reopen the axis for a new field, a new
  validator and a sixth `config_hash` exemption before anything could run;
* **the fraction is recoverable from the table and the reverse is not.** `match/total` is a
  column on every row, so `cap / yield` reads off the file; recovering an absolute budget from a
  fraction needs a second file;
* a cap above a matcher's yield is the anchor *by construction* (F74), so an "inert" column is a
  correctness property rather than a defect -- and the cap at which each matcher **departs** from
  its own anchor is the finding (block 2);
* only at equal absolute budget does the confidence-vs-random control compare across matchers. At
  equal fraction each column is a different number of correspondences per matcher, and the
  comparison is readable only within a row.

Nine levels, and why six would have been too few
------------------------------------------------
`GRID.md` §6 froze this stage at thirteen cells (six caps stopping at 64). `sift`'s **measured**
yield is 46.3 on `flir` and 32.1 on `dronevehicle` -- not P0-2's 113, which was one pair -- so a
ladder stopping at 64 leaves `sift` at its anchor in every column, and one of reduced-8 would be
an all-anchor row. At 1024...8 every matcher has at least three responsive columns, and the 16/8
end also walks the fit down to near its 4-point minimal sample, which is the regime P4-6 needs a
figure of. Seventeen cells, +~15 min against thirteen.

Two orderings, and the second is a control rather than a fallback
-----------------------------------------------------------------
`confidence` takes the top-N by the matcher's own per-match score; `random` takes a seeded prefix
of a permutation (`estimate/select.py`). At equal cap the two differ only in *which* matches
survive, so block 3 measures whether the certainty map identifies geometrically good
correspondences -- PLAN.md §6.2's baseline probed off an ablation that had to run anyway, and a
negative row under X-4 if the two agree. `random` is also the only arm defined for `xfeat`, alone
in reduced-8 in scoring no matches (P0-2); its confidence cells are recorded as
`selection_needs_confidence` rows (block 5) rather than aborting the sixteen variants that had
already paid for their match.

Reporting rules inherited, both non-optional here
-------------------------------------------------
`reg/epe_median` prints beside `reg/mace`, because a cap is precisely the kind of change that
shifts the whole distribution by an amount a mean over successes cannot resolve -- stage F's F63
found exactly that on the resolution axis, and F69 found a single cell swinging `mace` 10.27 ->
6.10 -> 9.60 while its median barely moved. And every cross-column median runs over the matchers
*every* column solved (F37), never over whatever each column happened to have.

**The tables are transposed relative to stages D-F**: caps read down and matchers across. Nine
levels x two orderings against stage F's four columns is the reason -- a stage-E-shaped table
would be ~240 characters wide and reach this Mac by copy-paste through a wrapped terminal. Down
is also where a sweep belongs: monotonicity in the axis reads as monotonicity in the column.

Needs a GPU; `--device cpu` with the dry-run overrides is how the plumbing is proved before the
trip. Every block it prints is meant to be copied out of the console whole.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from cmreg.metrics.schema import (
    EPE_MEDIAN,
    FAILURE_RATE,
    MACE,
    MATCH_INLIER_RATIO,
    N_PAIRS,
    TIME_ESTIMATE_MS,
    success_rate_key,
)
from cmreg.results import PairRow
from stages import CELLS, CONFIG, REDUCED_8, Cell, DryRun, run_cell

# `reg/mace` leads, as it does in stages A-F: P3-7's F7/F13 established that a thresholded rate
# is a function of the dataset's residual floor while a mean over corner errors is not.
HEADLINE = MACE
SECONDARY_THRESHOLD = 10.0
SECONDARY = success_rate_key(SECONDARY_THRESHOLD)

# The three the deliverable tables and the aggregate blocks are rendered once per. `EPE_MEDIAN`
# is **required** rather than a nicety (stage F's F63/F69): `reg/mace` is a mean over successes,
# so a cap that shifts the whole distribution a little shows up in the median and can hide in the
# mean, and a cap that blows up a handful of successes does the reverse.
AGGREGATE_METRICS = (HEADLINE, EPE_MEDIAN, SECONDARY)
# The deliverable table adds the failure rate, which is what makes the first three disagree.
TABLE_METRICS = (*AGGREGATE_METRICS, FAILURE_RATE)
# The ordering control adds the inlier ratio, which is the most direct read of the question it
# asks: a selector that picks geometrically good matches raises the fraction of them the solver
# calls inliers, whether or not the extra accuracy survives to `reg/mace` (F71 saw the two arms
# separate on this column first, on 20 Mac pairs).
ORDERING_METRICS = (*AGGREGATE_METRICS, MATCH_INLIER_RATIO)

# Two decimals for a pixel error, four for a rate.
_RATE_METRICS = (SECONDARY, FAILURE_RATE, MATCH_INLIER_RATIO)
# Metrics a *larger* value of which is better. Read by `_departed`, which cannot otherwise tell
# a cap that lost 5% of the success rate from one that gained it.
_HIGHER_IS_BETTER = (SECONDARY, MATCH_INLIER_RATIO)

# One matcher column. 11 fits `_abbr`'s 9 characters plus a gap; 8 matchers keeps the table
# inside ~102 characters, which a wrapped terminal survives.
_CELL_WIDTH = 11
_LABEL_WIDTH = 14

# GRID.md §6's `driving+aerial`: the best-characterised driving set and the only aerial one.
DATASETS = ("flir", "dronevehicle")

# The axis. Descending with 0 (no cap) first, which `EstimateConfig._descending_caps` requires,
# and every entry >= 4 because a homography has no smaller minimal sample.
CAPS = (0, 1024, 512, 256, 128, 64, 32, 16, 8)
# Which correspondences survive the cap. `confidence` is the axis as stated; `random` is the
# control, not a fallback -- see the module docstring.
SELECTIONS = ("confidence", "random")
# The variant whose console block the runner prints, and the one the seventeen are read against.
# `EstimateConfig` enforces that it is one of the swept cells rather than leaving it to this file.
ANCHOR = (0, "confidence")

# Held fixed at the axes this project has already settled, and **restated in `argv`** so this
# stage cannot inherit a different one if the anchor config is ever edited: MAGSAC @ 3 px is
# stage D's answer (P3-10 F35, its threshold row flat inside the seed spread), `homography` is
# stage E's (P3-11), and x1 is stage F's (P3-12b F65 -- the benchmark is quoted at native
# resolution). All three are the config's defaults, so restating them moves no hash.
ESTIMATOR = "magsac"
THRESHOLD = 3.0
WARP_MODEL = "homography"
INPUT_SCALE = 1.0

# One seed. GRID.md §5: an "A beats B" claim gets five, a scoping row does not. This axis is
# read as a curve against each matcher's *own* anchor rather than as a ranking between matchers,
# and the anchor column is shared by all seventeen cells, so the synthetic-warp draw that the
# five seeds of stage D measure is common to every column here and cannot separate them.
SEED = 0

# Each matcher's mean `match/total`, off **stage A's own runs** -- `logs/2026-08-28-p3a-stage-a-
# flir-composed-msrs.txt` and `logs/2026-08-27-p3a-stage-a-llvip-dronevehicle.txt`, 300 val pairs
# each. This is the table the cap ladder was chosen against, and it is here rather than in a
# comment because block 2 divides by it: a knee of 64 means something very different for `sift`
# (yield 46) than for `minima-roma` (yield 10 000), and that ratio is the fraction reading the
# per-matcher-fraction alternative would have measured directly.
#
# **A mean, so it bounds nothing per pair.** Whether a cap actually bit a given cell is read from
# the rows themselves (`_inert`, block 4), never from this table.
MEASURED_YIELD: dict[str, dict[str, float]] = {
    "flir": {
        "roma": 4096.0,
        "minima-roma": 10000.0,
        "matchanything-roma": 4990.22,
        "eloftr": 789.52,
        "xoftr": 1613.38,
        "superpoint-lightglue": 351.06,
        "xfeat": 797.78,
        "sift": 46.31,
    },
    "dronevehicle": {
        "roma": 4096.0,
        "minima-roma": 10000.0,
        "matchanything-roma": 4990.82,
        "eloftr": 594.62,
        "xoftr": 1478.27,
        "superpoint-lightglue": 317.52,
        "xfeat": 782.72,
        "sift": 32.13,
    },
}

# Block 2's band: the smallest cap whose metric is still within 5% of the same matcher's uncapped
# value, *and* that holds at every wider cap too. The second half is what stops one lucky cell
# setting the knee -- stage F's F69 is a single cell whose `mace` beat every neighbour by 40%
# while its median barely moved, and a knee read off a single comparison would have believed it.
DEPARTURE_TOLERANCE = 0.05

# The token `eval/runner.py::_unsupported` writes for this stage's one capability hole.
NEEDS_CONFIDENCE = "selection_needs_confidence"

# How many individual violations block 4 names before it stops. A count is the finding; three
# examples are enough to debug from and a per-row dump is 40 800 lines into a copy-paste console.
_MAX_EXAMPLES = 3


def columns() -> tuple[tuple[int, str], ...]:
    """The seventeen cells, anchor first, then each cap with both orderings.

    **Seventeen, not eighteen.** With no cap nothing is dropped, so the two orderings select the
    identical correspondences in the identical order and `EstimateConfig.variants()` emits the
    uncapped cell once (F75). Mirrored here rather than re-derived, and
    `tests/test_grid_driver.py` pins the two against each other -- a driver that expected an
    eighteenth column would render a permanent `--` and read as a failed cell.
    """
    return (ANCHOR, *((cap, sel) for cap in CAPS if cap != 0 for sel in SELECTIONS))


def label_for(cap: int, selection: str) -> str:
    """`all` / `1024/conf` / `1024/rand` -- one table row's name."""
    return "all" if cap == 0 else f"{cap}/{selection[:4]}"


def run_dir_for(cell: Cell) -> Path:
    return Path("runs") / f"stageg_{cell.dataset}"


def argv_for(cell: Cell) -> list[str]:
    """The cell's **scientific** flags -- everything that enters `config_hash`.

    Separated from the invocation flags below so the tests can resolve the very config the run
    used, through `cmreg`'s own parser, rather than rebuilding it. A second path to the same
    answer is a second path that can disagree with the run it describes (`stages.intended_hash`).
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
        # The four settled axes, restated. See their constants for why.
        "--warp-model",
        WARP_MODEL,
        "--estimator",
        ESTIMATOR,
        "--threshold",
        f"{THRESHOLD:g}",
        "--input-scale",
        f"{INPUT_SCALE:g}",
        # The anchor, which `EstimateConfig` requires to appear in both sweep lists.
        "--max-matches",
        str(ANCHOR[0]),
        "--match-selection",
        ANCHOR[1],
        "--sweep-max-matches",
        ",".join(str(cap) for cap in CAPS),
        "--sweep-selections",
        ",".join(SELECTIONS),
    ]
    # The `R` policy is per dataset and carried from `stages.CELLS` rather than restated:
    # `flir` composes, `dronevehicle` does not (GRID.md §3). A match-count axis does not touch it.
    if cell.calibration is not None:
        argv += ["--residual-calibration", str(cell.calibration)]
    return argv


def run(cell: Cell, device: str, dry: DryRun) -> bool:
    """One dataset's match pass, carrying all seventeen estimation variants."""
    run_dir = run_dir_for(cell)
    argv = [
        *argv_for(cell),
        "--device",
        device,
        "--wandb" if dry.wandb else "--no-wandb",
        "--name",
        run_dir.name,
        # One W&B group per dataset; `eval/runner.py::_group` extends it per (matcher, variant)
        # so the seventeen variants stay apart.
        "--group",
        f"p3g_matchcount_{cell.dataset}",
        "--run-dir",
        str(run_dir),
    ]
    if dry.limit is not None:
        argv += ["--limit", str(dry.limit)]
    # Appended after the stage's own list so a laptop smoke run overrides it -- and a row
    # produced that way is not a stage-F' row, as `DryRun` says.
    if dry.matchers is not None:
        argv += ["--matchers", ",".join(dry.matchers)]
    banner = (
        f"########## {run_dir.name} -- {len(columns())} variants off one match pass "
        f"({cell.why}) ##########"
    )
    return run_cell(cell, run_dir, argv, banner)


Key = tuple[int, str, str]
"""`(max_matches, match_selection, matcher)` -- one cell of one dataset."""


Series = dict[Key, list[dict[str, float]]]
"""`{key: [one metric dict]}` for one dataset. A list of one, so the renderers below are the
same shape as stage D's, which averages over five seeds."""


Rows = dict[Key, list[PairRow]]
"""`{key: every row}` for one dataset."""


def read_dataset(cell: Cell) -> tuple[Series, Rows]:
    """One dataset's run directory, both summarised and raw.

    Read once and passed to every renderer below -- five of them want these numbers, and
    re-reading is five chances to read a different directory than the table above it. The raw
    rows come back too because block 4 is a per-pair assertion and no aggregate can express it.
    """
    from cmreg.results import read_rows, summarize

    series: Series = defaultdict(list)
    rows_by_cell: Rows = defaultdict(list)
    rows = read_rows(run_dir_for(cell))
    for key in dict.fromkeys(_key(row) for row in rows):
        group = [row for row in rows if _key(row) == key]
        series[key].append(summarize(group, (SECONDARY_THRESHOLD,)).metrics)
        rows_by_cell[key].extend(group)
    return dict(series), dict(rows_by_cell)


def _key(row: PairRow) -> Key:
    """This row's cell.

    `max_matches` and `match_selection` are nullable because every stage A-F file on the server
    predates the axis (`results/store.py`), and a null there is "uncapped, confidence" -- which is
    what those runs did. Normalised here so a pre-axis directory read by this driver lands in the
    anchor column instead of a column of its own.
    """
    cap = ANCHOR[0] if row.max_matches is None else row.max_matches
    selection = ANCHOR[1] if row.match_selection is None else row.match_selection
    return (cap, selection, row.matcher)


def _matchers_in(series: Series) -> list[str]:
    """Every matcher present, in the order the cells ran them."""
    return list(dict.fromkeys(matcher for _, _, matcher in series))


def _metric(series: Series, key: Key, metric: str) -> float:
    """One cell's metric, or NaN where it produced none.

    NaN is the honest answer for a cell that never succeeded -- `xfeat` under a confidence-ranked
    cap, which cannot run at all -- and it renders as `--` rather than as a number.
    """
    values = [m[metric] for m in series.get(key, []) if not math.isnan(m[metric])]
    return statistics.fmean(values) if values else float("nan")


def _short(metric: str) -> str:
    """`reg/success_rate_10px` -> `success_rate_10px`, for a header that has to fit."""
    return metric.split("/")[-1]


def _fmt(metric: str, value: float) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:.4f}" if metric in _RATE_METRICS else f"{value:.2f}"


def _abbr(matchers: list[str]) -> dict[str, str]:
    """Short column headers, or the full names when shortening would collide.

    `superpoint-lightglue` at four characters per hyphen-part is `supe-ligh`, which fits an
    11-character column; `matchanything-roma` and `matchanything-eloftr` both start `matc-` and
    would not be distinguishable. Only one of that pair is in reduced-8, but a driver that
    mislabels a column when the matcher list changes is a driver that has to be re-read to be
    trusted, so the collision is detected rather than assumed away.
    """
    short = {name: "-".join(part[:4] for part in name.split("-")) for name in matchers}
    if len(set(short.values())) != len(matchers):
        return {name: name for name in matchers}
    return short


def _scored(series: Series) -> int:
    for metrics in series.values():
        return int(metrics[0][N_PAIRS])
    return 0


def _is_hole(rows: Rows, key: Key) -> bool:
    """True when this cell is the capability gap rather than a measurement.

    Read from `failure_reason` and never from a metric's value (stage D's F51): a
    `selection_needs_confidence` row reads a truthful `reg/success_rate_10px` of 0.0000, which in
    an aggregate is a real number meaning "never ran", and averaging it in charges the confidence
    arm for a matcher it never had.

    **`any`, not `all`**, for the reason stage E's F47 records: the gap is a property of the
    matcher (`estimate/select.py::needs_confidence`), so one row carrying the token settles the
    cell -- and it has to, because a pair whose fit raises takes every variant of that
    (pair, matcher) down with it.
    """
    return any(row.failure_reason == NEEDS_CONFIDENCE for row in rows.get(key, []))


def _inert(rows: Rows, key: Key) -> bool:
    """True when this cap dropped nothing on any scored pair, so the cell **is** the anchor.

    A cap above the matcher's yield is the anchor by construction (F74): `selected_indices`
    returns `arange(n)` untouched rather than a sorted permutation of it. Computed from the rows'
    own `n_matches` rather than from `MEASURED_YIELD`, which is a mean over pairs and settles
    nothing about any individual one.
    """
    cap = key[0]
    group = rows.get(key, [])
    return bool(group) and cap != 0 and all(row.n_matches <= cap for row in group)


def _common_matchers(series: Series, rows: Rows, cap: int) -> list[str]:
    """The matchers *both* orderings produced geometry for at this cap.

    Every cross-arm median in block 3 is taken over this set rather than over each arm's own
    matchers. Stage D is the reason (F37): `xfeat` has no PROSAC cell, which made one arm's
    median a median over seven matchers and the others' over eight -- and dropping the
    second-worst matcher lifts it enough to invert the ordering. The identical hole is here, one
    axis over: `xfeat` has no confidence cell at any cap.

    Applicability is read off `reg/mace` whichever metric is being aggregated, and off
    `_is_hole` before that. A cell that solved no pair has no mace; a cell that could not run has
    no mace *and* a truthful 0.0000 success rate.
    """
    return [
        matcher
        for matcher in _matchers_in(series)
        if all(
            not _is_hole(rows, (cap, selection, matcher))
            and not math.isnan(_metric(series, (cap, selection, matcher), HEADLINE))
            for selection in SELECTIONS
        )
    ]


# --- block 1: the deliverable ---------------------------------------------------------------


def _grid(
    title: str,
    notes: list[str],
    matchers: list[str],
    cell_text: Callable[[int, str, str], str],
) -> str:
    """Caps down, matchers across. The shape every table in this stage shares.

    Transposed relative to stages D-F because nine levels x two orderings is seventeen columns
    the other way round; see the module docstring. `cell_text` is `(cap, selection, matcher) ->
    str`.
    """
    abbr = _abbr(matchers)
    header = f"{'cap/sel':<{_LABEL_WIDTH}}" + "".join(
        f"{abbr[name]:>{_CELL_WIDTH}}" for name in matchers
    )
    lines = [title, *notes, header, "-" * len(header)]
    for cap, selection in columns():
        cells = "".join(f"{cell_text(cap, selection, name):>{_CELL_WIDTH}}" for name in matchers)
        lines.append(f"{label_for(cap, selection):<{_LABEL_WIDTH}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def count_table(cell: Cell, series: Series, rows: Rows, metric: str) -> str:
    """One dataset, one metric: **the stage's deliverable.**

    A trailing `=` marks a cell in which the cap dropped nothing on any scored pair, so the row
    is the uncapped anchor reproduced rather than a measurement of that cap (F74). It is the one
    piece of information a reader cannot recover from the number itself, and without it the wide
    columns look like a suspiciously flat result instead of the correctness property they are.
    """
    matchers = _matchers_in(series)
    if not matchers:
        return ""
    yields = MEASURED_YIELD.get(cell.dataset, {})
    yield_row = "".join(
        f"{(f'{yields[name]:.0f}' if name in yields else '?'):>{_CELL_WIDTH}}" for name in matchers
    )

    def text(cap: int, selection: str, matcher: str) -> str:
        key = (cap, selection, matcher)
        if _is_hole(rows, key):
            return "hole"
        value = _fmt(metric, _metric(series, key, metric))
        return f"{value}=" if value != "--" and _inert(rows, key) else value

    composed = "composed" if cell.composes else "uncomposed"
    return _grid(
        f"=== CMREG STAGE G: {cell.dataset} / match count -- {metric} ({composed}) ===",
        [
            f"# pairs scored: {_scored(series)}, seed {SEED}. {ESTIMATOR}@{THRESHOLD:g}px, "
            f"{WARP_MODEL}, input_scale x{INPUT_SCALE:g}.",
            "# Rows cap the correspondences fed to the FIT (estimate.max_matches), NOT the "
            "matcher's own",
            "#   budget -- match.max_keypoints is the detector's and half of reduced-8 ignores "
            "it (F70).",
            "# `all` is the uncapped anchor and is ONE row, not two: with nothing to drop the "
            "two orderings",
            "#   select identically (F75). `conf` = top-N by the matcher's score, `rand` = a "
            "seeded prefix.",
            "# A trailing `=` means the cap dropped nothing on any pair, so that cell IS the "
            "anchor (F74).",
            "# `hole` = selection_needs_confidence: the matcher scores no matches, so a "
            "confidence-ranked",
            "#   cap is undefined for it (P0-2). Recorded as rows, not dropped (X-4).",
            "# yield row = mean match/total from stage A, so cap/yield is the fraction this "
            "cap decimates to.",
            f"{'yield':<{_LABEL_WIDTH}}{yield_row}",
        ],
        matchers,
        text,
    )


# --- block 2: the headline ------------------------------------------------------------------


def _departed(metric: str, value: float, anchor: float) -> bool:
    """True when `value` is outside the tolerance band around this matcher's own anchor.

    Direction-aware: `reg/mace` and `reg/epe_median` are worse when larger and
    `reg/success_rate_10px` when smaller, and a band applied blind would call a cap that *gained*
    5% of the success rate a departure.
    """
    if math.isnan(value) or math.isnan(anchor):
        return True
    if metric in _HIGHER_IS_BETTER:
        return value < anchor * (1.0 - DEPARTURE_TOLERANCE)
    return value > anchor * (1.0 + DEPARTURE_TOLERANCE)


def _knee(series: Series, rows: Rows, selection: str, matcher: str, metric: str) -> int | None:
    """The smallest cap this matcher holds its anchor at, or None if it departs immediately.

    Walked from the widest cap down and stopped at the first departure, so the answer is
    "the narrowest budget that is still free *all the way down from uncapped*" rather than
    "the narrowest budget that happens to score well". Stage F's F69 is why: a single cell there
    beat every neighbour by 40% on `reg/mace` while its median barely moved, and a knee read off
    one comparison would have believed it.
    """
    anchor = _metric(series, (ANCHOR[0], ANCHOR[1], matcher), metric)
    if math.isnan(anchor):
        return None
    smallest: int | None = None
    for cap in (c for c in CAPS if c != 0):
        key = (cap, selection, matcher)
        if key not in series or _is_hole(rows, key):
            return smallest
        if _departed(metric, _metric(series, key, metric), anchor):
            return smallest
        smallest = cap
    return smallest


def _ratio_text(
    rows: Rows, yields: dict[str, float], knee: int | None, selection: str, matcher: str
) -> str:
    """`knee / yield`, or `inert` when the knee is a cap that never bit.

    A knee wider than the matcher's yield is not a decimation fraction at all -- it is the
    statement that the matcher departs as soon as the cap does anything, and printing 1.382 for
    it would read as "safe to decimate to 138%". Decided on `_inert`, which is per pair, rather
    than on `knee > yield`, which is a mean over pairs and settles nothing about any of them.
    """
    if knee is None:
        return "--"
    if _inert(rows, (knee, selection, matcher)):
        return "inert"
    supply = yields.get(matcher)
    return f"{knee / supply:.3f}" if supply else "?"


def departure_block(all_series: dict[str, Series], all_rows: dict[str, Rows], metric: str) -> str:
    """**The stage's answer**: how few correspondences a robust homography fit actually needs.

    Per (dataset, ordering, matcher), the narrowest cap still within `DEPARTURE_TOLERANCE` of
    that matcher's *own* uncapped value, and every wider cap too. Quoted beside the matcher's
    stage-A yield and the ratio between them, which is the fraction the per-matcher-fraction
    alternative would have measured directly -- the recoverability argument in the module
    docstring, made arithmetic rather than asserted.

    Rendered once per entry in `AGGREGATE_METRICS`. A knee on `reg/mace` alone would be a knee on
    a mean over successes, which is the failure stage F's F63 records.
    """
    label = _short(metric)
    lines = [
        f"=== CMREG STAGE G: smallest cap holding within {DEPARTURE_TOLERANCE:.0%} of the "
        f"uncapped anchor [{metric}] ===",
        "# Read down from `all`: the knee is the narrowest cap at which THIS matcher, and every",
        "#   wider cap, stays inside the band. A single cap that scores well below its knee is",
        "#   not promoted -- stage F's F69 is one such cell and it was a tail artefact.",
        "# `--` = departs at the widest cap already; `hole` = no confidence arm (P0-2).",
        "# knee/yield is cap over the matcher's mean stage-A match/total: the decimation fraction,",
        "#   recovered from an absolute ladder (which the reverse could not do). `?` where this",
        "#   run's matcher has no stage-A yield (a dry-run override).",
        "# `inert` there means the knee is a cap that dropped nothing on any pair, i.e. THIS "
        "matcher",
        "#   departs as soon as the cap actually bites. Read off the rows, not off the mean "
        "yield --",
        "#   a ratio above 1.0 would otherwise read as 'safe to decimate to 138%'.",
    ]
    header = (
        f"{'dataset':<14}{'ordering':<12}{'matcher':<22}"
        f"{'anchor ' + label:>22}{'knee':>8}{'yield':>10}{'knee/yield':>12}"
    )
    lines += [header, "-" * len(header)]
    for dataset, series in all_series.items():
        rows = all_rows[dataset]
        yields = MEASURED_YIELD.get(dataset, {})
        for selection in SELECTIONS:
            for matcher in _matchers_in(series):
                anchor = _metric(series, (ANCHOR[0], ANCHOR[1], matcher), metric)
                if any(_is_hole(rows, (cap, selection, matcher)) for cap in CAPS if cap != 0):
                    knee_text, ratio_text = "hole", "--"
                else:
                    knee = _knee(series, rows, selection, matcher, metric)
                    knee_text = "--" if knee is None else str(knee)
                    ratio_text = _ratio_text(rows, yields, knee, selection, matcher)
                supply = f"{yields[matcher]:.0f}" if matcher in yields else "?"
                lines.append(
                    f"{dataset:<14}{selection:<12}{matcher:<22}"
                    f"{_fmt(metric, anchor):>22}{knee_text:>8}{supply:>10}{ratio_text:>12}"
                )
    lines.append("=== END ===")
    return "\n".join(lines)


# --- block 3: the certainty-map control -----------------------------------------------------


def ordering_block(all_series: dict[str, Series], all_rows: dict[str, Rows], metric: str) -> str:
    """`confidence` against `random` at equal cap -- PLAN.md §6.2's baseline, probed for free.

    At equal cap the two arms differ only in *which* correspondences survive, so a separation is
    evidence that the matcher's own score identifies geometrically good matches and a flat block
    is evidence that it does not. Either way it is a measurement of the certainty map that the
    dense error head has to beat, taken off an ablation that had to run anyway -- and a flat one
    is recorded, not dropped (X-4).

    Medians over `_common_matchers` at that cap, never over each arm's own (F37). `xfeat` is
    absent from every row by construction, which is stated rather than left to be noticed.
    """
    label = f"median {_short(metric)}"
    column = max(len(label), 12) + 2
    better = "higher" if metric in _HIGHER_IS_BETTER else "lower"
    lines = [
        f"=== CMREG STAGE G: confidence vs random at equal cap [{metric}] ===",
        "# The two arms fit the SAME NUMBER of correspondences and differ only in which ones,",
        "#   so this is whether the matcher's certainty map picks geometrically good matches",
        "#   (PLAN.md §6.2's baseline). A flat column is the negative result, not a missing one.",
        f"# delta = confidence - random; {better} is better for this metric.",
        "# Medians over the matchers BOTH arms solved at that cap (F37); xfeat is in none of",
        "#   them, having no confidence arm at all (P0-2).",
    ]
    header = (
        f"{'dataset':<14}{'cap':>8}{'matchers':>10}"
        f"{'confidence':>{column}}{'random':>{column}}{'delta':>{column}}"
    )
    lines += [header, "-" * len(header)]
    for dataset, series in all_series.items():
        rows = all_rows[dataset]
        for cap in (c for c in CAPS if c != 0):
            shared = _common_matchers(series, rows, cap)
            if not shared:
                continue
            arms: dict[str, float] = {}
            for selection in SELECTIONS:
                values = [
                    _metric(series, (cap, selection, matcher), metric)
                    for matcher in shared
                    if not math.isnan(_metric(series, (cap, selection, matcher), metric))
                ]
                if values:
                    arms[selection] = statistics.median(values)
            if len(arms) != len(SELECTIONS):
                continue
            delta = arms["confidence"] - arms["random"]
            lines.append(
                f"{dataset:<14}{cap:>8}{len(shared):>10}"
                f"{_fmt(metric, arms['confidence']):>{column}}"
                f"{_fmt(metric, arms['random']):>{column}}"
                f"{delta:>{column}.4f}"
            )
    lines.append("=== END ===")
    return "\n".join(lines)


# --- block 4: the integrity check -----------------------------------------------------------


def _anchor_rows(rows: Rows, matcher: str) -> dict[str, PairRow]:
    return {row.stem: row for row in rows.get((ANCHOR[0], ANCHOR[1], matcher), [])}


def integrity_block(all_rows: dict[str, Rows]) -> str:
    """**A cap above a pair's yield must BE the anchor, not a reordering of it** (F74).

    `selected_indices` returns `arange(n)` untouched when the cap cannot bite, rather than a
    sorted permutation of the same set. That is what makes an uncapped run byte-identical to one
    made before this axis and therefore what lets `config_hash` drop the field. If it stopped
    holding, every wide column of every table above would differ from the anchor by a reordering
    -- PROSAC's sample order and `inlier_mask`'s indexing both move -- and each of them would
    still look entirely plausible. Nothing else in the suite would notice.

    Two further reproductions, both from F77: `n_selected` is what the solver saw and must be
    `min(cap, n_matches)`, and `inlier_ratio` is over `n_selected` rather than over `n_matches`,
    so a capped row's ratio is recomputable from the file alone.

    **Nesting is not checkable here** -- the cap-64 set being a subset of the cap-128 set (F72)
    is a statement about the selected indices, which are not persisted. It is pinned in
    `tests/test_select.py` against a fixture with a deliberate block of exact ties, because a
    uniform random draw essentially never produces one.
    """
    lines = [
        "=== CMREG STAGE G: integrity -- an inert cap reproduces the anchor exactly ===",
        "# Every (pair, matcher) whose n_matches <= cap must reproduce its uncapped row on",
        "#   corner_err, h, n_inliers and success. A mismatch voids every wide column above.",
        "# n_selected must be min(cap, n_matches); inlier_ratio must be n_inliers/n_selected",
        "#   (F77 -- the denominator moved and n_matches deliberately did not).",
        "# NESTING (cap-64 subset of cap-128, F72) is NOT checkable from Parquet: the selected",
        "#   indices are not persisted. It is pinned in tests/test_select.py.",
    ]
    header = (
        f"{'dataset':<14}{'inert rows':>14}{'reproduced':>14}"
        f"{'n_selected ok':>16}{'ratio ok':>12}{'verdict':>10}"
    )
    lines += [header, "-" * len(header)]
    examples: list[str] = []
    for dataset, rows in all_rows.items():
        inert = reproduced = selected_ok = ratio_ok = total = 0
        for (cap, selection, matcher), group in rows.items():
            anchors = _anchor_rows(rows, matcher)
            for row in group:
                if row.failure_reason == NEEDS_CONFIDENCE:
                    continue
                total += 1
                expected = row.n_matches if cap == 0 else min(cap, row.n_matches)
                if row.n_selected == expected:
                    selected_ok += 1
                elif len(examples) < _MAX_EXAMPLES:
                    examples.append(
                        f"#   n_selected: {dataset}/{matcher}/{row.stem} cap {cap} -> "
                        f"{row.n_selected}, expected {expected}"
                    )
                if _ratio_reproduces(row):
                    ratio_ok += 1
                elif len(examples) < _MAX_EXAMPLES:
                    examples.append(
                        f"#   inlier_ratio: {dataset}/{matcher}/{row.stem} cap {cap} -> "
                        f"{row.inlier_ratio:.6f}, expected {row.n_inliers}/{row.n_selected}"
                    )
                if cap == 0 or row.n_matches > cap:
                    continue
                inert += 1
                anchor = anchors.get(row.stem)
                if anchor is not None and _same_fit(row, anchor):
                    reproduced += 1
                elif len(examples) < _MAX_EXAMPLES:
                    examples.append(
                        f"#   anchor: {dataset}/{matcher}/{row.stem} cap {cap}/{selection} "
                        f"({row.n_matches} matches) does not reproduce the uncapped fit"
                    )
        passed = inert == reproduced and selected_ok == total and ratio_ok == total
        lines.append(
            f"{dataset:<14}{inert:>14,}{reproduced:>14,}"
            f"{selected_ok:>16,}{ratio_ok:>12,}{'PASS' if passed else 'FAIL':>10}"
        )
    if examples:
        lines += ["# first mismatches:", *examples]
    lines.append("=== END ===")
    return "\n".join(lines)


def _same_fit(row: PairRow, anchor: PairRow) -> bool:
    """Whether these two rows are the same fit of the same correspondences.

    `h` is compared exactly rather than within a tolerance: OpenCV's solvers are deterministic
    and carry no RNG state between calls (the property stage D's sweep rests on, pinned in
    `tests/test_estimate.py`), so an inert cap that reordered nothing must return the identical
    matrix. A tolerance here would accept exactly the drift this block exists to catch.
    """
    return (
        row.success == anchor.success
        and row.h == anchor.h
        and row.n_inliers == anchor.n_inliers
        and row.corner_err == anchor.corner_err
    )


def _ratio_reproduces(row: PairRow) -> bool:
    """`inlier_ratio == n_inliers / n_selected`, to float tolerance.

    Unlike `_same_fit` this one is a division rather than a copy, so it is compared within
    `1e-9` -- the ratio is stored as a float and recomputing it is not bit-exact.
    """
    if row.n_selected is None:
        return False
    if row.n_selected == 0:
        return row.inlier_ratio == 0.0
    return abs(row.inlier_ratio - row.n_inliers / row.n_selected) < 1e-9


# --- block 5: the hole ----------------------------------------------------------------------


def hole_block(all_rows: dict[str, Rows]) -> str:
    """Name the `selection_needs_confidence` cells out of the run's own rows.

    Read from `failure_reason` rather than from a list kept here, so the note cannot drift from
    what happened -- and named distinctly from stage D's `estimator_needs_confidence` even
    though the missing signal is the same one. That hole is PROSAC's and this one is the
    selector's; a table that could not tell them apart would attribute a stage-F' absence to
    stage D (F51's lesson, applied before it could bite).
    """
    found: dict[tuple[str, str], int] = defaultdict(int)
    for dataset, rows in all_rows.items():
        for (_, _, matcher), group in rows.items():
            hits = sum(1 for row in group if row.failure_reason == NEEDS_CONFIDENCE)
            if hits:
                found[(dataset, matcher)] += hits
    lines = [
        "=== CMREG STAGE G: the confidence arm's hole ===",
        "# A confidence-ranked cap needs a per-match score; three vismatch backends return none",
        "#   and one of them, xfeat, is in reduced-8 (P0-2). Those cells are RECORDED as rows",
        "#   carrying selection_needs_confidence rather than aborting the sixteen variants that",
        "#   had already paid for their match (X-4).",
        "# The random arm is defined for every backend, which is the second reason it is swept.",
    ]
    if not found:
        lines += ["", "# none: every matcher in this run scored its matches."]
    else:
        header = f"{'dataset':<14}{'matcher':<22}{'rows':>10}"
        lines += [header, "-" * len(header)]
        for (dataset, matcher), hits in sorted(found.items()):
            lines.append(f"{dataset:<14}{matcher:<22}{hits:>10,}")
    lines.append("=== END ===")
    return "\n".join(lines)


# --- block 6: what the axis costs -----------------------------------------------------------


def cost_block(cell: Cell, series: Series, rows: Rows) -> str:
    """Mean `time/estimate_ms` per (cap, ordering, matcher).

    **`time/total_ms` is deliberately not this block's column.** All seventeen of a swept run's
    rows for a pair carry the same match cost, so a total over the sweep reports seventeen times
    a bill the stage never paid (stage D's cost lesson, stage E's block 4).

    MAGSAC's iteration count is bounded by its inlier ratio and its per-iteration cost by the
    correspondence count, so this is the one axis in the project so far where the *estimator*
    should get cheaper as the science gets worse -- a free, honest row for PLAN.md §6.5 and
    Figure 11's Pareto, and the only place a "fit fewer matches" recommendation could be priced.
    """
    matchers = _matchers_in(series)
    if not matchers:
        return ""

    def text(cap: int, selection: str, matcher: str) -> str:
        key = (cap, selection, matcher)
        if _is_hole(rows, key):
            return "--"
        value = _metric(series, key, TIME_ESTIMATE_MS)
        return "--" if math.isnan(value) else f"{value:.2f}"

    return _grid(
        f"=== CMREG STAGE G: {cell.dataset} / {TIME_ESTIMATE_MS} by cap ===",
        [
            "# NOT time/total_ms: all seventeen variants of a pair share one match, so a total",
            "#   over the sweep would report seventeen times a cost the stage never paid.",
            "# A cell that could not run prints `--` and not 0.00 -- its rows carry",
            "#   estimate_ms=0.0 truthfully, which in a cost table reads as the cheapest column.",
        ],
        matchers,
        text,
    )


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
        "stage-F' row and is not comparable with the rest of the table.",
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
        f"########## {len(cells)} match passes x {len(columns())} estimation variants: "
        f"{', '.join(c.dataset for c in cells)} ##########"
    )
    completed = [cell for cell in cells if run(cell, args.device, overrides)]

    read = {cell.dataset: read_dataset(cell) for cell in completed}
    all_series = {dataset: series for dataset, (series, _) in read.items()}
    all_rows = {dataset: rows for dataset, (_, rows) in read.items()}
    for cell in completed:
        for metric in TABLE_METRICS:
            block = count_table(cell, all_series[cell.dataset], all_rows[cell.dataset], metric)
            if block:
                print()
                print(block, flush=True)
    if all_series:
        blocks = [
            *(departure_block(all_series, all_rows, metric) for metric in AGGREGATE_METRICS),
            *(ordering_block(all_series, all_rows, metric) for metric in ORDERING_METRICS),
            integrity_block(all_rows),
            hole_block(all_rows),
            *(
                cost_block(cell, all_series[cell.dataset], all_rows[cell.dataset])
                for cell in completed
            ),
        ]
        for block in blocks:
            if block:
                print()
                print(block, flush=True)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
