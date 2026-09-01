"""P3-12b / Stage F: the input-resolution axis, on the training server.

`experiments/GRID.md` §6 froze this stage as input resolution x `driving+aerial` x reduced-8 x
1 seed, and §6's "Stage F's resolution half, and the frame it is scored in" is its design. This
file is that design and nothing more; the **match-count** half is P3-12c and is deliberately not
here (`max_keypoints` is not honoured by the detector-free backends -- `minima-roma` returns
10 000 under any budget, `TODO(P3-12)` in `matchers/vismatch_backend.py`).

    uv run python scripts/p3f_resolution.py --device cuda
    uv run python scripts/p3f_resolution.py --datasets flir --levels 1 0.5

**Both sides are resized, by one factor** (`preprocess.input_scale`, P3-12a). Stage C measured
the asymmetric direction and it is dominated by an N-fold *scale mismatch* between the views
rather than by resolution (P3-9 F25), so a one-sided sweep here would have re-measured F25 and
called it resolution.

**The frame is native, and that is the whole of the design** (P3-12a F52). `H_gt`, the scored
truth, the dense field, `corner_error` and `PairRow.height`/`width` all stay at the pair's own
shape; only the two images handed to the matcher shrink, and both sides' keypoints come back
through `Preprocessed.to_native` before estimation. So every level is scored against
*identically the same ground truth* on *the same reporting ladder*, GRID.md §3's `R` composition
is untouched, and a stage-F row is directly comparable with stage A's. A decode-time resize --
shrinking the pair before `H_gt` is sampled -- would have put every metric in *scaled* pixels,
where 5 px at x0.25 is 20 native px, and would have broken composition outright.

**One match pass per cell, unlike stages D and E.** This axis sits *upstream* of the matcher
(`eval/runner.py::_identity_columns`), so it cannot be swept off one `MatchResult` the way the
estimator and the warp model were. Twelve cells, twelve passes.

**Stage C's `responsive-4` split does not apply, and saying why is part of the record.** Four of
reduced-8 resize their inputs to a fixed internal resolution -- `roma`/`minima-roma` to 560x560,
SuperPoint to a 1024 px long side, `matchanything-roma` inside its model config -- which makes
them blind to pixels *added*, which was stage C's direction, but **not** to pixels *removed*.
`roma` upsampling a 160-wide crop back to 560x560 cannot recover what `cv2.INTER_AREA` discarded.
All eight matchers therefore run at all four levels.

**The floor is measured, not quoted** (F54, and GRID.md §6 asks for exactly this). At scale `s` a
matcher localises to at best one *scaled* pixel, so the axis carries a soft floor of its own that
grows as the level falls: on the mono-modal fixture `reg/mace` reads 0.45 px at x0.5 and 1.02 px
at x0.25 against ~0.2 px at x1, i.e. `floor(x1) / s`. The measured version on real data is
P1-1b's mono-modal control (`gt.reference == gt.moving`) re-run per level, which this stage
carries on `flir`.

**The floor cells must not compose `R`, and getting that wrong is this stage's silent failure.**
`stages.CELLS` marks `flir` `composes=True`, so a driver that reused `argv_for` for the control
would fold `calibration/flir.json`'s ~9.5 px rig constant into a cell that has no rig: a pair
matched against a warped copy of *itself* has no cross-modal pairing and therefore no residual.
The floor would read ~9-10 px instead of ~0.2-1.0 px, would then swamp the very axis it exists to
bound, and would be reported as "every level is floor-limited" -- which is backwards. `FLOOR_CELL`
below is a separate `Cell` with `composes=False` so that decision lives in one visible place, and
`tests/test_grid_driver.py::TestStageF` pins the resolved config rather than the argument.

Needs a GPU; `--device cpu` with the dry-run overrides is how the plumbing is proved before the
trip. Every block it prints is meant to be copied out of the console whole.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

from cmreg.metrics.schema import (
    FAILURE_RATE,
    MACE,
    N_PAIRS,
    TIME_TOTAL_MS,
    success_rate_key,
)
from stages import CELLS, CONFIG, REDUCED_8, Cell, DryRun, run_cell

# `reg/mace` leads, as it does in stages A-E: P3-7's F7/F13 established that a thresholded rate
# is a function of the dataset's residual floor while a mean over corner errors is not.
HEADLINE = MACE
SECONDARY_THRESHOLD = 10.0
SECONDARY = success_rate_key(SECONDARY_THRESHOLD)

# Both lead in the aggregate blocks, for the reason stage D's output forced (F33/F34) and
# GRID.md §6 requires every later driver to carry: `reg/mace` is a mean over *successes* while
# `reg/success_rate_10px` charges a failure as infinite error, so a level that declines the pairs
# it can no longer solve improves the first and loses the second. Shrinking an image is precisely
# the change that produces that asymmetry.
AGGREGATE_METRICS = (HEADLINE, SECONDARY)

# Two decimals for a pixel error, four for a rate.
_RATE_METRICS = (SECONDARY, FAILURE_RATE)

# `mace | success | failure` at this project's widest cell.
_CELL_WIDTH = 24

# GRID.md §6's `driving+aerial`: the best-characterised driving set and the only aerial one.
DATASETS = ("flir", "dronevehicle")

# The four factors that divide 640x512 exactly -> 640/480/320/160 wide. Not a round-number
# choice: `preprocess/variants.py::rescale` **refuses** an anisotropic resize rather than
# rounding it, because `Preprocessed.to_native` inverts a *single* scale and an x/y mismatch
# leaves a systematic sub-pixel bias that reads as a worse matcher (P3-12a F56).
LEVELS = (1.0, 0.75, 0.5, 0.25)
# The level every other one is read against, and the one that must reproduce stage E's anchor
# column: at 1.0 `input_scale` resizes nothing and is dropped from `config_hash` (F58).
ANCHOR_LEVEL = 1.0

# The measured floor is bought on one dataset. The floor is a property of (matcher, scale) and
# both sets are 640x512, so one establishes it; `dronevehicle` is quoted against the `floor(x1)/s`
# prediction instead, and the block says which of the two it is using per row.
FLOOR_DATASET = "flir"
# Thermal, not optical: it is the lower-texture side and therefore the harder floor, and P1-1b
# measured the two modalities within 3% of each other (0.1949 vs 0.1965 px on `msrs`), so the
# second cell would buy a third decimal at twice the price.
FLOOR_MODALITY = "thermal"
# **`composes=False`, and that is the whole point of this object** -- see the module docstring.
# A mono-modal pair has no cross-modal pairing and therefore no residual `R` to remove; folding
# `flir`'s rig constant in here would report a ~9.5 px rig error as this axis's floor.
FLOOR_CELL = Cell(
    FLOOR_DATASET,
    "driving",
    "public",
    "the mono-modal control: the axis's own localisation floor, uncomposed",
    composes=False,
)

# One seed. GRID.md §5: an "A beats B" claim gets five, a scoping row does not. The levels here
# are separated by factors of two in the pixel count, well outside the 2.45-3.67 px seed spread
# stage D measured (F35).
SEED = 0


def label_for(level: float) -> str:
    """`0.75` -> `x0.75`. The same `:g` the runner's own W&B label uses
    (`eval/runner.py::_variant_label`), so a directory name and its run name cannot disagree."""
    return f"x{level:g}"


def run_dir_for(cell: Cell, level: float) -> Path:
    return Path("runs") / f"stagef_{cell.dataset}_{label_for(level)}"


def floor_run_dir_for(level: float) -> Path:
    return Path("runs") / f"stagef_floor_{FLOOR_CELL.dataset}_{label_for(level)}"


def argv_for(cell: Cell, level: float) -> list[str]:
    """One cross-modal cell's **scientific** flags -- everything that enters `config_hash`.

    Separated from the invocation flags in `run` below so the tests can resolve the very config
    the run used, through `cmreg`'s own parser, rather than rebuilding it. A second path to the
    same answer is a second path that can disagree with the run it describes -- the argument
    `stages.intended_hash` makes, and this is its fourth consumer.
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
        "--input-scale",
        f"{level:g}",
    ]
    # The `R` policy is per dataset and carried from `stages.CELLS` rather than restated:
    # `flir` composes, `dronevehicle` does not (GRID.md §3). The resolution axis does not touch
    # that decision -- the images shrink, the ground truth and the constant stay native.
    if cell.calibration is not None:
        argv += ["--residual-calibration", str(cell.calibration)]
    return argv


def floor_argv_for(level: float) -> list[str]:
    """One control cell's scientific flags. Three departures from `argv_for`, each load-bearing.

    * `--reference` equal to `--moving` is the control itself (P1-1b): both sides of every pair
      are read from one modality, so the moving side is a *known warp of the reference side* and
      nothing is left to fail except the machinery.
    * `--preprocess-ref none --preprocess-mov none`, because the anchor's `invert`/`percentile`
      is an asymmetric recipe and would manufacture the very polarity gap the control exists to
      remove.
    * **no `--residual-calibration`**, which is why `FLOOR_CELL` exists at all. See the module
      docstring: composing here would report the rig as the floor.
    """
    return [
        "bench",
        "-c",
        CONFIG,
        "--data",
        str(FLOOR_CELL.manifest),
        "--domain",
        FLOOR_CELL.domain,
        "--platform",
        FLOOR_CELL.platform,
        "--matchers",
        ",".join(REDUCED_8),
        "--seed",
        str(SEED),
        "--input-scale",
        f"{level:g}",
        "--moving",
        FLOOR_MODALITY,
        "--reference",
        FLOOR_MODALITY,
        "--preprocess-ref",
        "none",
        "--preprocess-mov",
        "none",
    ]


def _invocation(argv: list[str], run_dir: Path, group: str, device: str, dry: DryRun) -> list[str]:
    """The non-scientific tail every cell shares: device, W&B, and where the rows land.

    `runtime` is excluded from `config_hash`, so nothing here changes the experiment -- which is
    exactly why it is separated from the two builders above rather than interleaved with them.
    """
    argv = [
        *argv,
        "--device",
        device,
        "--wandb" if dry.wandb else "--no-wandb",
        "--name",
        run_dir.name,
        # One W&B group per (dataset, arm); `eval/runner.py::_variant_label` already appends
        # `-r{input_scale:g}` to the run name, so the four levels stay apart inside it.
        "--group",
        group,
        "--run-dir",
        str(run_dir),
    ]
    if dry.limit is not None:
        argv += ["--limit", str(dry.limit)]
    # Appended after the stage's own list so a laptop smoke run overrides it -- and a row
    # produced that way is not a stage-F row, as `DryRun` says.
    if dry.matchers is not None:
        argv += ["--matchers", ",".join(dry.matchers)]
    return argv


def run(cell: Cell, level: float, device: str, dry: DryRun) -> bool:
    """One (dataset, level) match pass. False when it was skipped for want of a manifest."""
    run_dir = run_dir_for(cell, level)
    argv = _invocation(
        argv_for(cell, level), run_dir, f"p3f_resolution_{cell.dataset}", device, dry
    )
    banner = f"########## {run_dir.name} -- both sides at {level:g}x ({cell.why}) ##########"
    return run_cell(cell, run_dir, argv, banner)


def run_floor(level: float, device: str, dry: DryRun) -> bool:
    """One control cell. Runs through `FLOOR_CELL`, never through `stages.CELLS`'s `flir`.

    Using the composing cell here would not merely mislabel the row: `run_cell` requires the
    constant to exist for a composing cell and `refuse_a_stale_run` compares its digest, so the
    control would silently be scored against the rig it does not have.
    """
    run_dir = floor_run_dir_for(level)
    argv = _invocation(
        floor_argv_for(level),
        run_dir,
        f"p3f_resolution_floor_{FLOOR_CELL.dataset}",
        device,
        dry,
    )
    banner = (
        f"########## {run_dir.name} -- MONO-MODAL {FLOOR_MODALITY} control at {level:g}x, "
        f"uncomposed ##########"
    )
    return run_cell(FLOOR_CELL, run_dir, argv, banner)


Table = dict[str, dict[str, dict[str, float]]]
"""`{level label: {matcher: {metric key: value}}}` for one arm of one dataset."""

Errors = dict[str, dict[str, float]]
"""`{level label: {matcher: median corner_err over the pairs it solved}}`.

Carried beside `Table` because the floor comparison is a statement about the typical pair and
`reg/mace` is a *mean over successes*: one projective blow-up sets it, and stage D's F34 is the
record of what reading a success-conditioned mean beside anything else costs.
"""


def _read_level(run_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """One run directory, summarised per matcher and reduced to a median corner error per matcher.

    Both come off one read. Two reads of the same directory are two chances for the floor printed
    beside a row to have come from a different file than the row did.
    """
    from cmreg.results import read_rows, summarize

    rows = read_rows(run_dir)
    names = list(dict.fromkeys(row.matcher for row in rows))
    metrics: dict[str, dict[str, float]] = {}
    medians: dict[str, float] = {}
    for name in names:
        group = [row for row in rows if row.matcher == name]
        metrics[name] = summarize(group, (SECONDARY_THRESHOLD,)).metrics
        solved = [row.corner_err for row in group if row.success and row.corner_err is not None]
        if solved:
            medians[name] = statistics.median(solved)
    return metrics, medians


def read_arm(run_dirs: dict[str, Path]) -> tuple[Table, Errors]:
    """Summarise one arm's completed levels. Read once and passed to every renderer below --
    five of them want the same numbers, and re-reading is five chances to read a different
    directory than the table above it."""
    table: Table = {}
    errors: Errors = {}
    for label, run_dir in run_dirs.items():
        table[label], errors[label] = _read_level(run_dir)
    return table, errors


def _levels_in(table: Table, levels: tuple[float, ...]) -> list[float]:
    """The levels this arm actually has rows for, in the stage's own order."""
    return [level for level in levels if label_for(level) in table]


def _matchers_in(table: Table) -> list[str]:
    """Every matcher present, in the order the cells ran them."""
    return list(dict.fromkeys(name for column in table.values() for name in column))


def _metric(table: Table, label: str, matcher: str, metric: str) -> float:
    """One cell's metric, or NaN where it produced none. NaN renders as `--`, never as 0."""
    metrics = table.get(label, {}).get(matcher)
    return float("nan") if metrics is None else metrics[metric]


def _common_matchers(table: Table, levels: list[float]) -> list[str]:
    """The matchers *every* level produced geometry for.

    Every cross-level aggregate below is taken over this set rather than over each level's own
    matchers. Stage D is the reason (F37): a median over seven matchers against one over eight
    inverted an ordering, because dropping the second-worst matcher lifts the median.

    Applicability is read off `reg/mace` whichever metric is being aggregated. A cell that solved
    no pair has no mace, while `reg/success_rate_10px` still reads a truthful 0.0 there -- a real
    number meaning "never solved anything", and averaging it in charges a level for a matcher it
    never had. This is exactly the F47 predicate rule in this stage's shape, and it matters more
    here than it did there: x0.25 is the level at which a matcher is expected to stop solving.
    """
    return [
        matcher
        for matcher in _matchers_in(table)
        if all(
            not math.isnan(_metric(table, label_for(level), matcher, HEADLINE)) for level in levels
        )
    ]


def _excluded_note(all_tables: dict[str, Table], levels: tuple[float, ...]) -> str | None:
    """Name the matchers `_common_matchers` dropped, over **every** arm at once.

    Across all datasets rather than the first one that has an exclusion: the note is printed once
    above a block that renders every dataset, and a per-dataset note would leave the second
    table's exclusions unnamed -- a `--` with no explanation beside it being exactly the hole X-4
    exists to prevent.
    """
    absent = list(
        dict.fromkeys(
            name
            for table in all_tables.values()
            for name in _matchers_in(table)
            if name not in _common_matchers(table, _levels_in(table, levels))
        )
    )
    if not absent:
        return None
    return (
        f"# medians are over the matchers every level solved; {', '.join(absent)} excluded, "
        "since a median\n#   over a different matcher set per level compares two populations "
        "(F37)."
    )


def _scored(table: Table) -> int:
    for column in table.values():
        for metrics in column.values():
            return int(metrics[N_PAIRS])
    return 0


def _short(metric: str) -> str:
    """`reg/success_rate_10px` -> `success_rate_10px`, for a column header that has to fit."""
    return metric.split("/")[-1]


def _fmt(metric: str, value: float) -> str:
    return f"{value:.4f}" if metric in _RATE_METRICS else f"{value:.2f}"


def _cell_text(table: Table, label: str, matcher: str) -> str:
    """`mace | success@10px | failure rate`, for one cell.

    The failure rate rides along rather than living in a block of its own because the first two
    numbers only disagree when it moves -- and shrinking an image is the change most likely to
    move it, since a matcher that can no longer find correspondences declines the pair rather
    than fitting it badly (stage D's F34, stage E's F48).
    """
    mace = _metric(table, label, matcher, HEADLINE)
    if math.isnan(mace):
        return "--"
    return (
        f"{mace:.2f} | {_metric(table, label, matcher, SECONDARY):.3f}"
        f" | {_metric(table, label, matcher, FAILURE_RATE):.2f}"
    )


def resolution_table(cell: Cell, levels: list[float], table: Table) -> str:
    """Matchers down, resolution levels across, for one dataset. **The stage's deliverable.**

    Every column is in *native* pixels and against *identically the same ground truth* (F52), so
    this table is directly comparable with stage A's and with every other table in the paper --
    which a decode-time resize would have destroyed while looking exactly the same.
    """
    matchers = _matchers_in(table)
    if not matchers:
        return ""
    width = max(len("matcher"), *(len(name) for name in matchers)) + 2
    composed = "composed" if cell.composes else "uncomposed"
    header = f"{'matcher':<{width}}" + "".join(
        f"{label_for(level):>{_CELL_WIDTH}}" for level in levels
    )
    lines = [
        f"=== CMREG STAGE F: {cell.dataset} / input resolution ({composed}) -- {HEADLINE} px | "
        f"{SECONDARY} | {FAILURE_RATE} ===",
        f"# pairs scored: {_scored(table)}, seed {SEED}. Columns resize BOTH sides by that "
        "factor before matching.",
        "# Every number is in NATIVE pixels: the ground truth, the truth matrix and the "
        "reporting ladder",
        "#   do not move with the level, only the images the matcher sees (P3-12a F52).",
        "# mace is a mean over SUCCESSES; the success rate counts a failure as infinite error.",
        "#   The third number is the failure rate, and it is what makes the first two disagree.",
        header,
        "-" * len(header),
    ]
    for matcher in matchers:
        cells = "".join(
            f"{_cell_text(table, label_for(level), matcher):>{_CELL_WIDTH}}" for level in levels
        )
        lines.append(f"{matcher:<{width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def _predicted_floor(table: Table, levels: list[float], level: float) -> float:
    """`floor(x1) / s` -- the axis's own floor, predicted from its measured anchor row.

    F54 reads "close to `1/s`" and the fixture's numbers say what that means: 0.45 px at x0.5 and
    1.02 px at x0.25 against ~0.2 px at x1 is not `1/s` in absolute terms (that would be 2 and 4)
    but the x1 floor *scaled* by `1/s`, which is what one scaled pixel of localisation error
    becomes when it is measured in native pixels. Predicted from this run's own x1 row rather
    than from the fixture's 0.2 px, so the prediction is calibrated on the data it is checked
    against.
    """
    anchor = label_for(ANCHOR_LEVEL)
    if anchor not in table or level <= 0.0:
        return float("nan")
    values = [
        _metric(table, anchor, matcher, HEADLINE)
        for matcher in _common_matchers(table, levels)
        if not math.isnan(_metric(table, anchor, matcher, HEADLINE))
    ]
    return statistics.median(values) / level if values else float("nan")


def floor_table(levels: list[float], table: Table) -> str:
    """The mono-modal control, per matcher and level: what this axis costs before any modality gap.

    P1-1b measured this pipeline's floor at ~0.2 px on `msrs` and `dronevehicle` but never on
    `flir`, so the x1 row here is itself new. The rest of the table is the number every row of
    the block above has to be read against.
    """
    matchers = _matchers_in(table)
    if not matchers:
        return ""
    width = max(len("matcher"), len("PREDICTED x1/s"), *(len(n) for n in matchers)) + 2
    header = f"{'matcher':<{width}}" + "".join(f"{label_for(v):>12}" for v in levels)
    lines = [
        f"=== CMREG STAGE F: the axis's own floor -- {FLOOR_DATASET} mono-modal "
        f"{FLOOR_MODALITY}, {HEADLINE} px ===",
        "# P1-1b's control (gt.reference == gt.moving), UNCOMPOSED, preprocess none/none, "
        "re-run per level.",
        "# Uncomposed deliberately: a pair matched against a warped copy of itself has no",
        "#   cross-modal pairing and therefore no residual R to remove. Composing flir's rig",
        "#   constant here would report ~9.5 px of rig as this axis's floor.",
        "# PREDICTED is this run's own x1 row divided by s -- one *scaled* pixel of localisation",
        "#   error, expressed in native pixels (F54). Agreement is the prediction holding.",
        header,
        "-" * len(header),
    ]
    predicted = "".join(
        f"{_predicted_floor(table, levels, level):>12.2f}"
        if not math.isnan(_predicted_floor(table, levels, level))
        else f"{'--':>12}"
        for level in levels
    )
    lines.append(f"{'PREDICTED x1/s':<{width}}{predicted}")
    for matcher in matchers:
        cells = ""
        for level in levels:
            value = _metric(table, label_for(level), matcher, HEADLINE)
            cells += f"{'--':>12}" if math.isnan(value) else f"{value:>12.2f}"
        lines.append(f"{matcher:<{width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def _median_error(errors: Errors, label: str, matchers: list[str]) -> float:
    """Median over matchers of each matcher's median `corner_err`.

    Medians throughout and not a mix with `reg/mace`: the floor comparison below divides one of
    these by another, and dividing a median by a mean over successes would compare two statistics
    of location and call the ratio a finding.
    """
    values = [errors.get(label, {})[m] for m in matchers if m in errors.get(label, {})]
    return statistics.median(values) if values else float("nan")


def floor_limited_block(
    all_tables: dict[str, Table],
    all_errors: dict[str, Errors],
    floor_table_: Table,
    floor_errors: Errors,
    levels: tuple[float, ...],
) -> str:
    """How much of each level's error is the axis's own floor rather than registration?

    **Reported as a ratio, and explicitly not as stage E's `excess`.** There the floor was an
    exact algebraic lower bound -- the best a restricted warp model could possibly do -- so a
    column scoring below it falsified the floor and was flagged `**CHECK**` (P3-4a F45). This
    floor is an *empirical* estimate of a localisation limit measured in a different cell, so a
    row landing under it is noise and means nothing; subtracting them would invent a signed
    quantity neither measurement supports.

    `dronevehicle` has no measured control (the floor is a property of (matcher, scale) and both
    sets are 640x512), so its rows are read against the `floor(x1)/s` prediction and the `source`
    column says so. A floor a reader cannot tell the provenance of is worse than no floor.
    """
    floor_levels = _levels_in(floor_table_, levels)
    header = (
        f"{'dataset':<16}{'level':>8}{'median corner_err':>20}{'floor':>10}"
        f"{'floor/err':>11}{'source':>12}"
    )
    lines = [
        "=== CMREG STAGE F: is a level measuring registration, or its own floor? ===",
        "# median over matchers of each matcher's median corner_err, against the mono-modal",
        "#   control at the same level. Medians throughout: reg/mace is a mean over successes",
        "#   and cannot be divided by a median (F34).",
        "# NOT stage E's `excess`. That floor was a strict algebraic lower bound and a row",
        "#   beneath it falsified the measurement (F45); this one is an empirical estimate from",
        "#   a different cell, so a row beneath it is noise. Read the ratio, never a difference.",
        "# FLOOR-LIMITED at >= 0.50: at that point most of what the level reports is the axis's",
        "#   own localisation limit. Below 0.20 the row is a statement about registration.",
        "# A `measured` row takes both columns over the matchers the benchmark AND the control",
        "#   both solved at every level; a `x1/s pred` row has no control to intersect with.",
    ]
    if note := _excluded_note(all_tables, levels):
        lines.append(note)
    lines += [header, "-" * len(header)]
    floor_matchers = _common_matchers(floor_table_, floor_levels)
    for dataset, table in all_tables.items():
        levels_here = _levels_in(table, levels)
        matchers = _common_matchers(table, levels_here)
        errors = all_errors[dataset]
        for level in levels_here:
            label = label_for(level)
            measured = dataset == FLOOR_DATASET and label in floor_errors
            if measured:
                # **Both columns over one matcher set**, and the intersection rather than either
                # arm's own. The two arms lose matchers at different levels -- the control is the
                # easier problem, so it keeps solving after the benchmark stops -- and a ratio of
                # a median over one population to a median over another is not a ratio of
                # anything (F37, here applied *across* arms rather than across levels).
                shared = [name for name in matchers if name in floor_matchers]
                error = _median_error(errors, label, shared)
                floor = _median_error(floor_errors, label, shared)
                source = "measured"
            else:
                error = _median_error(errors, label, matchers)
                floor = _predicted_floor(floor_table_, floor_levels, level)
                source = "x1/s pred"
            if math.isnan(error) or math.isnan(floor) or error <= 0.0:
                lines.append(f"{dataset:<16}{label:>8}{'--':>20}{'--':>10}{'--':>11}{source:>12}")
                continue
            ratio = floor / error
            verdict = "  FLOOR-LIMITED" if ratio >= 0.5 else ""
            lines.append(
                f"{dataset:<16}{label:>8}{error:>20.2f}{floor:>10.2f}"
                f"{ratio:>11.2f}{source:>12}{verdict}"
            )
    lines.append("=== END ===")
    return "\n".join(lines)


def cost_block(all_tables: dict[str, Table], levels: tuple[float, ...]) -> str:
    """Mean `time/total_ms` per (matcher, level), and each level's ratio to x1.

    **`time/total_ms` is the right column here and was the wrong one in stage E.** There, six
    estimation variants shared one match pass and a total over the sweep would have reported six
    times a bill the stage never paid. Here every level *is* its own match pass, so the total is
    exactly what the level cost.

    The prior, written down before the run (X-4): stage C found GPU cost dominated by the fixed
    per-pair pipeline rather than by matching -- every backend rose only x1.23-x3.62 across x1-x4
    when the pixel count implied x16 (P3-9 F28) -- so the expectation is that shrinking buys far
    less than the pixel count suggests. This block is where PLAN.md §6.5's "resolution
    sensitivity" and Figure 11's Pareto get their measured numbers.
    """
    lines = [
        f"=== CMREG STAGE F: {TIME_TOTAL_MS} by level, and the ratio to {label_for(ANCHOR_LEVEL)} "
        "===",
        "# time/total_ms IS the right column here: unlike stages D and E, this axis is upstream",
        "#   of the matcher, so every level is its own match pass and pays its own bill.",
        "# PRIOR (P3-9 F28): GPU cost is dominated by the fixed per-pair pipeline, so expect the",
        "#   saving to fall far short of the pixel count. A flat row is a finding (X-4).",
    ]
    for dataset, table in all_tables.items():
        matchers = _matchers_in(table)
        levels_here = _levels_in(table, levels)
        if not matchers:
            continue
        width = max(len(f"{dataset}/matcher"), *(len(n) for n in matchers)) + 2
        header = f"{dataset + '/matcher':<{width}}" + "".join(
            f"{label_for(v):>10}{'ratio':>8}" for v in levels_here
        )
        lines += ["", header, "-" * len(header)]
        for matcher in matchers:
            anchor = _metric(table, label_for(ANCHOR_LEVEL), matcher, TIME_TOTAL_MS)
            cells = ""
            for level in levels_here:
                value = _metric(table, label_for(level), matcher, TIME_TOTAL_MS)
                if math.isnan(value):
                    cells += f"{'--':>10}{'--':>8}"
                    continue
                ratio = (
                    f"{value / anchor:>8.2f}"
                    if not math.isnan(anchor) and anchor > 0.0
                    else f"{'--':>8}"
                )
                cells += f"{value:>10.1f}{ratio}"
            lines.append(f"{matcher:<{width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def ranking_block(all_tables: dict[str, Table], levels: tuple[float, ...], metric: str) -> str:
    """Is the benchmark's matcher *ordering* invariant to input resolution?

    The actionable block. Stage B asked the identical question of polarity and the answer -- the
    ordering is polarity-invariant even where the level is not (P3-8 F19/F20) -- is what kept the
    anchor's recipe in place. If the ordering survives x0.5 here, every later stage is licensed
    to run at half resolution for whatever the cost block says that saves; if it does not, a
    benchmark quoted at one resolution does not transfer to another and the paper has to say so.

    Rendered once per entry in `AGGREGATE_METRICS`: a level that shortens the tail without moving
    the mode reorders a success-conditioned mean and a failure-inclusive rate differently, which
    is what stage E's F48 found for the warp models.
    """
    header = f"{'dataset':<16}{'level':>8}{'n':>5}{'rho vs ' + label_for(ANCHOR_LEVEL):>12}"
    lines = [
        f"=== CMREG STAGE F: is the matcher ranking invariant to resolution? [{metric}] ===",
        f"# Spearman of {metric} against the {label_for(ANCHOR_LEVEL)} column, over the matchers "
        "both levels solved.",
        "# ~1 means resolution is a LEVEL effect, not an ordering effect -- and a later stage may",
        "#   then be run cheap. Read `n` first: rho over three points takes a handful of values.",
        header,
        "-" * len(header),
    ]
    anchor = label_for(ANCHOR_LEVEL)
    for dataset, table in all_tables.items():
        if anchor not in table:
            continue
        for level in _levels_in(table, levels):
            if level == ANCHOR_LEVEL:
                continue
            label = label_for(level)
            names = [
                name
                for name in _matchers_in(table)
                if not math.isnan(_metric(table, anchor, name, metric))
                and not math.isnan(_metric(table, label, name, metric))
            ]
            if len(names) < 3:
                # A rank correlation over two points is +-1 whatever the numbers say.
                continue
            left = [_metric(table, anchor, name, metric) for name in names]
            right = [_metric(table, label, name, metric) for name in names]
            if len(set(left)) < 2 or len(set(right)) < 2:
                # `statistics.correlation` raises on a constant series, which a saturated level
                # (every matcher at 0.000) legitimately produces. Reported, not crashed on.
                lines.append(f"{dataset:<16}{label:>8}{len(names):>5}{'flat':>12}")
                continue
            # stdlib rather than `scipy.stats.spearmanr`: `method="ranked"` is Spearman exactly
            # (3.12+), it ties-average like scipy, and it is typed. Stage C's practice.
            rho = statistics.correlation(left, right, method="ranked")
            lines.append(f"{dataset:<16}{label:>8}{len(names):>5}{rho:>12.3f}")
    if len(lines) == 6:
        # Only reachable under a dry run's matcher override.
        lines.append("(no level shares three or more matchers with the anchor)")
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
        "--levels",
        nargs="+",
        type=float,
        choices=LEVELS,
        default=list(LEVELS),
        help="input-resolution levels; both sides are resized by each in turn",
    )
    parser.add_argument(
        "--skip-floor",
        action="store_true",
        help=f"skip the {FLOOR_DATASET} mono-modal control; every row then quotes the x1/s "
        "prediction instead of a measured floor",
    )
    dry = parser.add_argument_group(
        "dry-run overrides",
        "For proving the plumbing on a laptop. A row produced under any of these is not a "
        "stage-F row and is not comparable with the rest of the table.",
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
    levels = tuple(level for level in LEVELS if level in args.levels)

    cells = [cell for cell in CELLS if cell.dataset in args.datasets]
    floor_levels = () if args.skip_floor else levels
    print(
        f"########## {len(cells) * len(levels) + len(floor_levels)} match passes "
        f"({len(levels)} levels x {', '.join(c.dataset for c in cells)}"
        f"{'' if args.skip_floor else f', + {len(floor_levels)} {FLOOR_DATASET} floor cells'}) "
        "##########",
        flush=True,
    )

    # Cross-modal first, control second: if the trip is interrupted, the stage's deliverable
    # survives and only its floor column is missing.
    completed: dict[str, dict[str, Path]] = {}
    for cell in cells:
        done = {
            label_for(level): run_dir_for(cell, level)
            for level in levels
            if run(cell, level, args.device, overrides)
        }
        if done:
            completed[cell.dataset] = done
    floor_done = {
        label_for(level): floor_run_dir_for(level)
        for level in floor_levels
        if run_floor(level, args.device, overrides)
    }

    read = {dataset: read_arm(dirs) for dataset, dirs in completed.items()}
    all_tables = {dataset: table for dataset, (table, _) in read.items()}
    all_errors = {dataset: errors for dataset, (_, errors) in read.items()}
    floor_table_, floor_errors = read_arm(floor_done)

    for cell in cells:
        table = all_tables.get(cell.dataset)
        if table is None:
            continue
        block = resolution_table(cell, _levels_in(table, levels), table)
        if block:
            print()
            print(block, flush=True)
    if floor_table_:
        print()
        print(floor_table(_levels_in(floor_table_, levels), floor_table_), flush=True)
    if all_tables:
        blocks = [
            floor_limited_block(all_tables, all_errors, floor_table_, floor_errors, levels),
            cost_block(all_tables, levels),
            *(ranking_block(all_tables, levels, metric) for metric in AGGREGATE_METRICS),
        ]
        for block in blocks:
            print()
            print(block, flush=True)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
