"""P3-9 / Stage C: the thermal upsampling ablation, on the training server.

PLAN.md §15B records the recipe all three sibling implementations converged on: *invert the
optical grayscale, upsample the thermal x3 bicubic, RoMa, MAGSAC*. Stage B (P3-8) measured the
first half and returned a negative result -- inversion is the best of four polarities in 13 of
80 (matcher, dataset) cells. This stage measures the second half: **does upsampling the thermal
side help, and does the interpolation kernel matter?**

**It ran on 2026-08-29 and the answer is no, twice over** (TASKS.md P3-9, F23-F30). x3 bicubic
improves 3 of 8 matchers on flir and 2 of 8 on dronevehicle, never by over 7.4%, while costing
up to 27x. It is not a resolution gain: the runner only scores pairs whose modalities already
share a shape and the reference side is never resized, so xN is an Nx *scale mismatch* between
the views -- inert for a backend that resizes internally, absorbed by sift's scale-space
pyramid, and fatal for a fixed-stride learned matcher (success@10px 0.72/0.84/0.43 -> 0.16/
0.13/0.00 for eloftr/xoftr/xfeat). The factor beats the kernel 14-20x. Kept as it ran.

    uv run python scripts/p3c_upsample.py
    uv run python scripts/p3c_upsample.py --datasets flir --factors 1 3

Runs off `experiments/p3a_baseline_grid.yaml` itself with `--upsample` / `--interpolation`
overridden per cell, for the reason `scripts/p3_stageb_polarity.py` states: a stage's
scientific diff from the anchor is exactly the fields it varies, and a second YAML restating
the other twenty is how a stage quietly redefines its own defaults (P3-1).

**Half of reduced-8 cannot see a resolution change at all**, which is what shapes the grid.
Four backends resize their inputs to a fixed internal resolution, and the source says so:
`roma`/`minima-roma` fix 560x560 (`romatch/models/matcher.py:617`), SuperPoint fixes a 1024 px
long side (`LightGlue/lightglue/superpoint.py:115`), and `matchanything-roma` was flat by
measurement. For those four the upsampling axis is a **resample prefilter**, not a resolution
change, so they sit out the kernel axis; the anchor-kernel factor column keeps all eight, which
is where the prefilter effect gets its 300-pair number.

That premise was chosen on a Mac-CPU probe (`--limit 1`, flir val, x1 -> x4 bicubic, ms) and
**the accuracy half of it held while the cost half did not** (P3-9 F28). The 300-pair GPU run:

    matcher               CPU x1 -> x4        CPU     GPU x1 -> x4       GPU
    matchanything-roma  19219 -> 19452       flat      387 ->  537      x1.39
    superpoint-lightglue 3139 ->  3161       flat      101 ->  127      x1.26
    roma                20460 -> 20252 (x3)  flat      979 -> 1993      x2.04
    minima-roma         24692 -> 24705 (x3)  flat      799 ->  985      x1.23
    eloftr                943 -> 12282      x13.0      102 ->  238      x2.33
    xoftr                1148 -> raises       x3.5     109 ->  266 (x3) x2.44
    xfeat                 101 ->   603       x6.0       68 ->  106      x1.56
    sift                   35 ->   225       x6.4      134 ->  485      x3.62

Nothing is cost-flat on the GPU. CPU cost is dominated by matching FLOPs; GPU cost is dominated
by the fixed per-pair pipeline -- `cv2.resize`, the host->device transfer, and the backend's own
internal resize -- all of which scale with the input pixel count while the matching does not.
`sift` is CPU-bound even under `--device cuda`, which is the control that says so: it is the
steepest GPU row.

So **read this grid's design off accuracy invariance, not off runtime.** That is what held: the
four resize-internally backends move `reg/mace` by <=2.8% on flir and <=14.8% on dronevehicle
across the whole factor axis, with no consistent direction -- a prefilter, exactly as intended,
and four kernel columns of it would have bought four indistinguishable rows.

**`xoftr` cannot run above x3.** Its positional encoding is a fixed 256 cells at 1/8 stride, so
2048 px is the ceiling and 640x4 raises `RuntimeError: The size of tensor a (320) must match
the size of tensor b (256)` (`XoFTR/src/xoftr/utils/position_encoding.py:36`). Both datasets
here are 640 wide. It is dropped from the x4 cells rather than left to fail 300 times: the
runner would turn each into a `matcher_raised` row with a logged traceback, and 300 tracebacks
in a console that reaches the Mac by copy-paste is a log nobody can read. The exclusion is
stated in every table it affects (X-4 -- recorded, not silently dropped).

Needs a GPU: **4 h 34 min for 26 cells, measured** (~3.5 h was projected from matcher time
alone). `--device cpu` with the dry-run overrides is how the plumbing is proved before the trip.

Every block it prints is meant to be copied out of the console whole.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from cmreg.metrics.schema import MACE, N_PAIRS, TIME_TOTAL_MS, success_rate_key
from p3a_grid import CELLS, CONFIG, REDUCED_8, Cell, DryRun, run_cell

# `reg/mace` leads, as it does in stages A and B: P3-7's F7/F13 established that a thresholded
# rate is a function of the dataset's residual floor while a mean over corner errors is not,
# and `dronevehicle` here is a floor (GRID.md §3).
HEADLINE = MACE
SECONDARY_THRESHOLD = 10.0
SECONDARY = success_rate_key(SECONDARY_THRESHOLD)

# GRID.md §6's `driving+aerial`: the best-characterised driving set and the only aerial one.
DATASETS = ("flir", "dronevehicle")

FACTORS = (1, 2, 3, 4)
KERNELS = ("bicubic", "nearest", "bilinear", "lanczos")
# The anchor's kernel (GRID.md §1), so the factor column is a diff against stage A in one axis.
ANCHOR_KERNEL = "bicubic"
# The factor PLAN.md §15B's inherited recipe prescribes; block 2 is the test of that claim.
RECIPE_FACTOR = 3

# Backends that resize their inputs to a fixed internal resolution, verified by source and by
# the flat runtimes in the module docstring. Upsampling reaches them only as a prefilter, so
# they sit out the kernel axis -- and stay in the anchor-kernel factor column, which is what
# measures that prefilter at 300 pairs rather than at the 4 the design was chosen on.
RESOLUTION_BLIND = ("roma", "minima-roma", "matchanything-roma", "superpoint-lightglue")
RESPONSIVE = tuple(name for name in REDUCED_8 if name not in RESOLUTION_BLIND)

# A hard architectural ceiling, not a budget: see the module docstring. Keyed by matcher so a
# second one costs a line rather than a special case.
MAX_FACTOR = {"xoftr": 3}


@dataclass(frozen=True, slots=True)
class Setting:
    """One cell of the stage: an upsampling factor, a kernel, and the matchers it carries."""

    factor: int
    kernel: str

    @property
    def label(self) -> str:
        # x1 carries no kernel because at x1 there is none: `preprocess.upsample` returns the
        # input untouched, so the four kernels are one cell. The derived W&B run name drops it
        # at x1 for the same reason (`eval/runner.py::_variant_label`).
        return "x1" if self.factor == 1 else f"x{self.factor}_{self.kernel}"

    @property
    def matchers(self) -> tuple[str, ...]:
        pool = REDUCED_8 if self.kernel == ANCHOR_KERNEL else RESPONSIVE
        return tuple(name for name in pool if self.factor <= MAX_FACTOR.get(name, max(FACTORS)))

    @property
    def excluded(self) -> tuple[str, ...]:
        """Matchers this cell would carry but for their factor ceiling -- named in the table."""
        pool = REDUCED_8 if self.kernel == ANCHOR_KERNEL else RESPONSIVE
        return tuple(name for name in pool if name not in self.matchers)


def grid(factors: tuple[int, ...], kernels: tuple[str, ...]) -> tuple[Setting, ...]:
    """The stage's cells: the shared x1 cell, the anchor-kernel factor column, and the kernel
    axis above x1. 13 per dataset rather than 16 -- collapsing x1's four kernels into one cell
    is exact, not an approximation, since they produce a bit-identical image."""
    settings: list[Setting] = []
    if 1 in factors and ANCHOR_KERNEL in kernels:
        settings.append(Setting(1, ANCHOR_KERNEL))
    for kernel in kernels:
        settings += [Setting(f, kernel) for f in factors if f > 1]
    return tuple(settings)


def columns_for(kernel: str, settings: tuple[Setting, ...]) -> list[Setting]:
    """One kernel's table reads left to right x1, x2, x3, x4 -- and its x1 column is the shared
    cell, which is why this is a lookup over the whole grid rather than a filter on `kernel`."""
    shared = [s for s in settings if s.factor == 1]
    return shared + [s for s in settings if s.factor > 1 and s.kernel == kernel]


def run_dir_for(cell: Cell, setting: Setting) -> Path:
    return Path("runs") / f"stagec_{cell.dataset}_{setting.label}"


def run(cell: Cell, setting: Setting, device: str, dry: DryRun) -> bool:
    """One (dataset, factor, kernel) cell. False when it was skipped for want of a manifest."""
    run_dir = run_dir_for(cell, setting)
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
        ",".join(setting.matchers),
        "--upsample",
        str(setting.factor),
        "--interpolation",
        setting.kernel,
        "--device",
        device,
        "--wandb" if dry.wandb else "--no-wandb",
        "--name",
        run_dir.name,
        # One W&B group per dataset, so its thirteen cells aggregate together rather than
        # landing under stage A's group with nothing distinguishing them (TASKS.md §0).
        "--group",
        f"p3c_upsample_{cell.dataset}",
        "--run-dir",
        str(run_dir),
    ]
    # The `R` policy is per dataset and carried from `p3a_grid.CELLS` rather than restated:
    # `flir` composes, `dronevehicle` does not (GRID.md §3). An upsampling axis does not touch
    # that decision.
    if cell.calibration is not None:
        argv += ["--residual-calibration", str(cell.calibration)]
    if dry.limit is not None:
        argv += ["--limit", str(dry.limit)]
    # Appended after the stage's own list so a laptop smoke run overrides it -- and a row
    # produced that way is not a stage-C row, as `DryRun` says.
    if dry.matchers is not None:
        argv += ["--matchers", ",".join(dry.matchers)]
    banner = (
        f"########## {run_dir.name} -- {len(setting.matchers)} matchers ({cell.why}) ##########"
    )
    return run_cell(cell, run_dir, argv, banner)


Table = dict[str, dict[str, dict[str, float]]]
"""`{setting label: {matcher: {metric key: value}}}` for one dataset."""


def read_table(cell: Cell, settings: tuple[Setting, ...]) -> Table:
    """Summarise one dataset's cells. Read once and passed to every renderer below -- five of
    them want the same numbers, and re-reading is five chances to read a different directory
    than the table above it."""
    from cmreg.results import read_rows, summarize

    table: Table = {}
    for setting in settings:
        rows = read_rows(run_dir_for(cell, setting))
        names = list(dict.fromkeys(row.matcher for row in rows))
        table[setting.label] = {
            name: summarize(
                [row for row in rows if row.matcher == name], (SECONDARY_THRESHOLD,)
            ).metrics
            for name in names
        }
    return table


def _cell_text(metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return "--"
    return f"{metrics[HEADLINE]:.2f} | {metrics[SECONDARY]:.3f}"


def _scored(table: Table) -> int:
    for column in table.values():
        for metrics in column.values():
            return int(metrics[N_PAIRS])
    return 0


def _matchers_in(table: Table, labels: list[str]) -> list[str]:
    """Every matcher appearing in these columns, in the order the cells ran them."""
    return list(dict.fromkeys(name for label in labels for name in table.get(label, {})))


def factor_table(cell: Cell, kernel: str, settings: tuple[Setting, ...], table: Table) -> str:
    """Matchers down, upsampling factor across, for one kernel."""
    columns = columns_for(kernel, settings)
    labels = [s.label for s in columns]
    matchers = _matchers_in(table, labels)
    if not matchers:
        return ""
    width = max(len("matcher"), *(len(name) for name in matchers)) + 2
    composed = "composed" if cell.composes else "a floor, not an accuracy"
    header = f"{'matcher':<{width}}" + "".join(f"{f'x{s.factor}':>22}" for s in columns)
    lines = [
        f"=== CMREG STAGE C: {cell.dataset} / {kernel} ({composed}) -- {HEADLINE} px | "
        f"{SECONDARY} ===",
        f"# pairs scored: {_scored(table)}",
        "# the x1 column is one shared cell: at x1 every kernel produces the same image.",
    ]
    dropped = sorted({name for s in columns for name in s.excluded})
    if dropped:
        lines.append(f"# absent above its factor ceiling: {', '.join(dropped)} (see the header)")
    lines += [header, "-" * len(header)]
    for matcher in matchers:
        cells = "".join(f"{_cell_text(table[label].get(matcher)):>22}" for label in labels)
        lines.append(f"{matcher:<{width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def recipe_block(tables: dict[str, Table]) -> str:
    """PLAN.md §15B's claim, as a table: does upsampling x3 bicubic help *every* matcher?

    The direct analogue of stage B's Figure 6 block, and the question this stage exists to
    answer. Counted per dataset rather than pooled -- a pooled count is dominated by whichever
    dataset has the widest spread, and P3-7's F1 is the record of what happens when one
    dataset's floor speaks for the grid.
    """
    anchor = Setting(RECIPE_FACTOR, ANCHOR_KERNEL).label
    header = f"{'dataset':<16}{'improved':>12}{'median gain px':>18}{'median gain %':>16}"
    lines = [
        f"=== CMREG STAGE C: the inherited x{RECIPE_FACTOR} {ANCHOR_KERNEL} upsample "
        "(PLAN.md §15B) ===",
        f"# `{anchor}` against `x1`. improved = matchers whose mace is lower upsampled.",
        "# median gain in px, and as a fraction of the x1 mace so the two datasets compare.",
        header,
        "-" * len(header),
    ]
    for dataset, table in tables.items():
        names = [n for n in table.get(anchor, {}) if n in table.get("x1", {})]
        if not names:
            lines.append(f"{dataset:<16}{'needs the x1 and ' + anchor + ' cells':>46}")
            continue
        gains = [table["x1"][n][HEADLINE] - table[anchor][n][HEADLINE] for n in names]
        relative = [
            gain / table["x1"][n][HEADLINE] * 100.0
            for gain, n in zip(gains, names, strict=True)
            if table["x1"][n][HEADLINE] > 0.0
        ]
        improved = sum(1 for gain in gains if gain > 0.0)
        lines.append(
            f"{dataset:<16}{f'{improved}/{len(gains)}':>12}"
            f"{statistics.median(gains):>18.2f}{statistics.median(relative):>16.1f}"
        )
    lines.append("=== END ===")
    return "\n".join(lines)


def _spread(values: list[float]) -> float | None:
    return max(values) - min(values) if len(values) > 1 else None


def axis_block(tables: dict[str, Table], settings: tuple[Setting, ...]) -> str:
    """Which of the two axes carries the effect -- the factor, or the kernel?

    Both spreads are computed over the resolution-responsive matchers alone, because they are
    the only ones present in every kernel column, and a spread taken over a different matcher
    set per axis compares two different populations rather than two axes. Medians throughout:
    mace spans 7-900 px across this project's matcher list, and one classical failure would
    otherwise set the statistic.

    A kernel spread that is small against the factor spread settles the kernel for stages D-G;
    one that is not means the kernel is a knob the paper has to report.
    """
    factors = sorted({s.factor for s in settings if s.factor > 1})
    kernels = sorted({s.kernel for s in settings if s.factor > 1})
    header = f"{'dataset':<16}{'across kernels':>18}{'across factors':>18}{'ratio':>10}"
    lines = [
        "=== CMREG STAGE C: does the kernel matter, or only the factor? ===",
        f"# over {', '.join(RESPONSIVE)} -- the matchers present in every kernel column.",
        "# across kernels = median spread of mace over the kernels at a fixed factor.",
        "# across factors = median spread of mace over the factors at a fixed kernel (x1 in).",
        "# ratio = factors / kernels. Large means the factor is the axis and the kernel is a",
        "#   free choice for stages D-G; ~1 means the kernel has to be reported too.",
        header,
        "-" * len(header),
    ]
    for dataset, table in tables.items():
        over_kernels: list[float] = []
        over_factors: list[float] = []
        for matcher in RESPONSIVE:
            for factor in factors:
                values = [
                    table[label][matcher][HEADLINE]
                    for kernel in kernels
                    if matcher in table.get((label := Setting(factor, kernel).label), {})
                ]
                if (spread := _spread(values)) is not None:
                    over_kernels.append(spread)
            for kernel in kernels:
                labels = ["x1"] + [Setting(f, kernel).label for f in factors]
                values = [
                    table[label][matcher][HEADLINE]
                    for label in labels
                    if matcher in table.get(label, {})
                ]
                if (spread := _spread(values)) is not None:
                    over_factors.append(spread)
        if not over_kernels or not over_factors:
            lines.append(f"{dataset:<16}{'needs the full kernel axis':>46}")
            continue
        kernel_spread = statistics.median(over_kernels)
        factor_spread = statistics.median(over_factors)
        ratio = factor_spread / kernel_spread if kernel_spread > 0.0 else float("inf")
        lines.append(f"{dataset:<16}{kernel_spread:>18.2f}{factor_spread:>18.2f}{ratio:>10.2f}")
    lines.append("=== END ===")
    return "\n".join(lines)


def cost_block(tables: dict[str, Table], settings: tuple[Setting, ...]) -> str:
    """Mean `time/total_ms` per (matcher, factor) at the anchor kernel.

    Upsampling is the one preprocessing axis that could buy accuracy with runtime, so the
    trade-off is the finding rather than an aside, and it feeds P3-14 and Figure 11.

    It also refuted the cost half of this stage's own design premise (P3-9 F28). The premise was
    "a backend that resizes internally costs the same at x4 as at x1", measured on the Mac CPU;
    on the GPU **every** row rises, by x1.23 to x3.62, because what scales there is the fixed
    per-pair pipeline rather than the matching. Do not read internal-resize behaviour off this
    block -- read it off the accuracy tables above, where it is what the design actually rests on.
    """
    columns = columns_for(ANCHOR_KERNEL, settings)
    labels = [s.label for s in columns]
    lines = [
        f"=== CMREG STAGE C: {TIME_TOTAL_MS} by upsampling factor ({ANCHOR_KERNEL}) ===",
        "# every row rises with the factor, including the backends that resize internally: on a",
        "#   GPU the cost that scales is the fixed per-pair pipeline (resize, transfer, the",
        "#   backend's own resize), not the matching. `sift` runs on the CPU whatever --device",
        "#   says, and is the steepest row here -- which is the control for that reading.",
    ]
    # One width for every dataset in the block: a per-dataset width misaligns the second
    # header against the first, and this block is read by eye out of a pasted console.
    names = [name for table in tables.values() for name in _matchers_in(table, labels)]
    width = max(len("dataset/matcher"), *(len(name) for name in names)) + 2 if names else 0
    for dataset, table in tables.items():
        matchers = _matchers_in(table, labels)
        if not matchers:
            continue
        header = f"{dataset + '/matcher':<{width}}" + "".join(
            f"{f'x{s.factor}':>14}" for s in columns
        )
        lines += ["", header, "-" * len(header)]
        for matcher in matchers:
            cells = "".join(
                f"{table[label][matcher][TIME_TOTAL_MS]:>14.0f}"
                if matcher in table.get(label, {})
                else f"{'--':>14}"
                for label in labels
            )
            lines.append(f"{matcher:<{width}}{cells}")
    lines.append("=== END ===")
    return "\n".join(lines)


def ranking_block(tables: dict[str, Table], settings: tuple[Setting, ...]) -> str:
    """Spearman of `reg/mace` against the x1 column, per (dataset, factor, kernel).

    Stage B's F19 asked the same question of polarity and answered it: preprocessing moved the
    *level* and not the *ordering*, which is what let stage A's conclusions survive it. If
    upsampling reorders the field instead, every stage authored as a diff against a x1 anchor
    inherits that, and GRID.md §1 has to say so.
    """
    header = f"{'dataset':<16}{'cell':>16}{'n':>5}{'rho vs x1':>12}"
    lines = [
        "=== CMREG STAGE C: is the matcher ranking invariant to upsampling? ===",
        "# Spearman of reg/mace against the x1 column, over the matchers shared with it.",
        "# ~1 means upsampling is a level effect, not an ordering effect (cf. P3-8 F19).",
        "# READ `n` FIRST. Only the anchor-kernel rows carry all eight matchers; a kernel-axis",
        "#   row has the responsive four or three, where rho is quantised to a handful of values",
        "#   (+-1, +-0.5 at n=3) and a matcher set that upsampling has driven into saturation.",
        "#   A low rho there says the ordering is meaningless, not that it was rearranged.",
        header,
        "-" * len(header),
    ]
    for dataset, table in tables.items():
        for setting in settings:
            if setting.factor == 1:
                continue
            label = setting.label
            names = [n for n in table.get(label, {}) if n in table.get("x1", {})]
            if len(names) < 3:
                # Spearman over two points is +1 or -1 whatever the numbers say.
                continue
            left = [table["x1"][n][HEADLINE] for n in names]
            right = [table[label][n][HEADLINE] for n in names]
            # stdlib rather than `scipy.stats.spearmanr`: `method="ranked"` is Spearman
            # exactly (3.12+), it ties-average like scipy, and it is typed.
            rho = statistics.correlation(left, right, method="ranked")
            lines.append(f"{dataset:<16}{label:>16}{len(names):>5}{rho:>12.3f}")
    if len(lines) == 5:
        # Only reachable under a dry run's matcher override: a rank correlation over fewer
        # than three matchers is +1 or -1 whatever the numbers are, so nothing is printed.
        lines.append("(no cell shares three or more matchers with x1)")
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
        "--factors",
        nargs="+",
        type=int,
        choices=FACTORS,
        default=list(FACTORS),
        help="restrict the factor axis; the summary blocks want all four",
    )
    parser.add_argument(
        "--kernels", nargs="+", choices=KERNELS, default=list(KERNELS), help="restrict the kernels"
    )
    dry = parser.add_argument_group(
        "dry-run overrides",
        "For proving the plumbing on a laptop. A row produced under any of these is not a "
        "stage-C row and is not comparable with the rest of the table.",
    )
    dry.add_argument("--limit", type=int, help="override the config's 300-pair budget")
    dry.add_argument("--matchers", nargs="+", help="override the stage's per-cell matcher list")
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
    settings = grid(tuple(sorted(set(args.factors))), tuple(args.kernels))
    print(
        f"########## {len(cells) * len(settings)} cells: "
        f"{', '.join(c.dataset for c in cells)} x {', '.join(s.label for s in settings)} "
        "##########"
    )
    # Dataset-outer so an interrupted run leaves whole datasets finished rather than two
    # half-finished ones, and the tables below can be printed for what completed.
    completed: list[Cell] = []
    for cell in cells:
        # Every setting is attempted even when one is skipped, rather than short-circuiting: a
        # dataset is skipped for want of a manifest, and printing that once per cell is what
        # tells a reader of the pasted console which of the twenty-six are missing.
        ran = [run(cell, setting, args.device, overrides) for setting in settings]
        if all(ran):
            completed.append(cell)

    tables = {cell.dataset: read_table(cell, settings) for cell in completed}
    for cell in completed:
        for kernel in args.kernels:
            if block := factor_table(cell, kernel, settings, tables[cell.dataset]):
                print()
                print(block, flush=True)
    if tables:
        for block in (
            recipe_block(tables),
            axis_block(tables, settings),
            cost_block(tables, settings),
            ranking_block(tables, settings),
        ):
            print()
            print(block, flush=True)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
