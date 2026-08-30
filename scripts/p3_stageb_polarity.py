"""P3-8 / Stage B: the inverted-grayscale generality cell, on the training server.

PLAN.md §15B records that all three sibling implementations invert the *optical* grayscale
before matching thermal against it. Nothing has ever measured whether that generalises past
RoMa, and PLAN.md §7 / Figure 6 make the claim across *every* matcher -- which is why
`experiments/GRID.md` §6 keeps this stage as wide as stage A rather than dropping to reduced-8.

**The 2x2, and why it is not an optical-only on/off.** What a matcher sees is *relative*
polarity, so inverting both sides restores the relation inverting neither had. The four cells
therefore fall into two pairs:

    reference / moving          label       relative polarity
    invert    / percentile        optical     flipped   <- the anchor recipe (GRID.md §1)
    none      / percentile_invert thermal     flipped
    none      / percentile        neither     as captured
    invert    / percentile_invert both        as captured

If the effect is relative polarity, `optical` and `thermal` agree and `neither` and `both`
agree. If instead inverting *the optical image specifically* is what helps -- which is what the
sibling recipe asserts -- they will not, because the two members of a pair differ only in which
side was inverted. An optical-only on/off cannot tell those apart, and only the second reading
licenses "invert the optical side" as a recipe rather than as a coincidence of this data.

Runs off `experiments/p3a_baseline_grid.yaml` itself, with the preprocess pair overridden per
cell. Stage B's scientific diff from the anchor is exactly those two fields, and a second YAML
restating the other twenty is how a stage quietly redefines its own defaults (P3-1).

    uv run python scripts/p3_stageb_polarity.py
    uv run python scripts/p3_stageb_polarity.py --datasets flir --polarities optical neither

Naming: `scripts/p3b_calibrate.py` is P2-12's *calibration* driver, not stage B. This stage's
run directories are `runs/stageb_<dataset>_<label>` for that reason.

Needs a GPU: 16 cells x 300 pairs x 20 matchers is ~7.5 h there (GRID.md §7's measured
5.65 s/pair) and would be ~8 days on a Mac CPU. `--device cpu` with the dry-run overrides is
how the plumbing is proved before the trip.

Every block it prints is meant to be copied out of the console whole -- results reach the Mac
as console text and nothing else.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from cmreg.metrics.schema import MACE, N_PAIRS, success_rate_key
from stages import CELLS, CONFIG, Cell, DryRun, run_cell

# Read beside `reg/mace`, never instead of it: P3-7's F7/F13 established that a thresholded
# rate is a function of the dataset's residual floor and a mean over corner errors is not, and
# three of the four columns here are floors (GRID.md §3).
SECONDARY_THRESHOLD = 10.0
SECONDARY = success_rate_key(SECONDARY_THRESHOLD)

# `reg/mace`, as stage A leads with (P3-7 F7/F13).
HEADLINE = MACE


@dataclass(frozen=True, slots=True)
class Polarity:
    """One cell of the 2x2. `flipped` is the relative polarity presented to the matcher."""

    label: str
    reference: str
    moving: str
    flipped: bool

    @property
    def recipe(self) -> str:
        return f"{self.reference}/{self.moving}"


POLARITIES = (
    Polarity("neither", "none", "percentile", flipped=False),
    Polarity("optical", "invert", "percentile", flipped=True),
    Polarity("thermal", "none", "percentile_invert", flipped=True),
    Polarity("both", "invert", "percentile_invert", flipped=False),
)

# The recipe every other stage runs (GRID.md §1), so every delta in this stage is quoted
# against it and stage A's table is the row it belongs to.
ANCHOR = "optical"


def run_dir_for(cell: Cell, polarity: Polarity) -> Path:
    return Path("runs") / f"stageb_{cell.dataset}_{polarity.label}"


def run(cell: Cell, polarity: Polarity, device: str, dry: DryRun) -> bool:
    """One (dataset, polarity) cell. False when it was skipped for want of a manifest."""
    run_dir = run_dir_for(cell, polarity)
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
        "--preprocess-ref",
        polarity.reference,
        "--preprocess-mov",
        polarity.moving,
        "--device",
        device,
        "--wandb" if dry.wandb else "--no-wandb",
        "--name",
        run_dir.name,
        # One W&B group per dataset, so its four polarity cells aggregate together and stage B
        # does not land ungrouped beside stage A's runs (TASKS.md §0).
        "--group",
        f"p3b_polarity_{cell.dataset}",
        "--run-dir",
        str(run_dir),
    ]
    # The `R` policy is per dataset and is carried from `stages.CELLS` rather than restated:
    # `flir` composes and the other three do not (GRID.md §3), and a polarity axis does not
    # touch that decision.
    if cell.calibration is not None:
        argv += ["--residual-calibration", str(cell.calibration)]
    if dry.limit is not None:
        argv += ["--limit", str(dry.limit)]
    if dry.matchers is not None:
        argv += ["--matchers", ",".join(dry.matchers)]
    banner = f"########## {run_dir.name} -- {polarity.recipe} ({cell.why}) ##########"
    return run_cell(cell, run_dir, argv, banner)


Table = dict[str, dict[str, dict[str, float]]]
"""`{polarity label: {matcher: {metric key: value}}}` for one dataset."""


def read_table(cell: Cell, polarities: list[Polarity]) -> Table:
    """Summarise one dataset's polarity cells. Read once and passed to every renderer below --
    three of them want the same numbers, and re-reading is three chances to read a different
    run directory than the table above it."""
    from cmreg.results import read_rows, summarize

    table: Table = {}
    for polarity in polarities:
        rows = read_rows(run_dir_for(cell, polarity))
        names = list(dict.fromkeys(row.matcher for row in rows))
        table[polarity.label] = {
            name: summarize(
                [row for row in rows if row.matcher == name], (SECONDARY_THRESHOLD,)
            ).metrics
            for name in names
        }
    return table


def dataset_table(cell: Cell, table: Table) -> str:
    """Matchers down, the polarity cells across, `reg/mace` and `success_rate_10px` in each,
    plus the delta of the anchor recipe against no inversion at all.

    The delta column is the finding rather than a convenience: PLAN.md Figure 6 is a grouped
    bar chart of exactly this quantity, and a reader of a pasted table cannot subtract twenty
    rows by eye.
    """
    labels = list(table)
    matchers = list(dict.fromkeys(name for column in table.values() for name in column))
    width = max(len("matcher"), *(len(name) for name in matchers)) + 2
    composed = "composed" if cell.composes else "a floor, not an accuracy"
    header = (
        f"{'matcher':<{width}}"
        + "".join(f"{label:>22}" for label in labels)
        + f"{'d vs neither':>14}"
    )
    lines = [
        f"=== CMREG STAGE B: {cell.dataset} ({composed}) -- {HEADLINE} px | {SECONDARY} ===",
        f"# pairs scored: {_scored(table)}",
        "# d = mace(optical anchor) - mace(neither); negative means inverting the optical side",
        "#     helped this matcher on this dataset.",
        header,
        "-" * len(header),
    ]
    for matcher in matchers:
        cells = "".join(f"{_cell_text(table[label].get(matcher)):>22}" for label in labels)
        lines.append(f"{matcher:<{width}}{cells}{_delta(table, matcher):>14}")
    lines.append("=== END ===")
    return "\n".join(lines)


def _scored(table: Table) -> int:
    for column in table.values():
        for metrics in column.values():
            return int(metrics[N_PAIRS])
    return 0


def _cell_text(metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return "--"
    return f"{metrics[HEADLINE]:.2f} | {metrics[SECONDARY]:.3f}"


def _delta(table: Table, matcher: str) -> str:
    """`mace(anchor) - mace(neither)`, as text, or `--` when either cell is absent."""
    anchor = table.get(ANCHOR, {}).get(matcher)
    other = table.get("neither", {}).get(matcher)
    if anchor is None or other is None:
        return "--"
    return f"{anchor[HEADLINE] - other[HEADLINE]:+.2f}"


def _shared(table: Table, left: str, right: str) -> list[str]:
    """The matchers scored in both columns, in the order they were run."""
    if left not in table or right not in table:
        return []
    return [name for name in table[left] if name in table[right]]


def generality_block(tables: dict[str, Table]) -> str:
    """PLAN.md Figure 6, as a table: does inverting the optical side help *every* matcher?

    Counted per dataset rather than pooled. A pooled count would be dominated by whichever
    dataset has the widest spread, and P3-7's F1/F11 is the record of what happens when one
    dataset's floor is allowed to speak for the grid.
    """
    header = f"{'dataset':<16}{'improved':>12}{'median gain px':>18}{'median gain %':>16}"
    lines = [
        "=== CMREG STAGE B: generality of the optical inversion (PLAN.md Figure 6) ===",
        "# anchor `optical` (invert/percentile) against `neither` (none/percentile).",
        "# improved = matchers whose mace is lower under the anchor. Median gain in px, and as",
        "# a fraction of the `neither` mace so datasets with different floors are comparable.",
        header,
        "-" * len(header),
    ]
    for dataset, table in tables.items():
        names = _shared(table, ANCHOR, "neither")
        if not names:
            lines.append(f"{dataset:<16}{'needs the optical and neither cells':>46}")
            continue
        gains = [table["neither"][n][HEADLINE] - table[ANCHOR][n][HEADLINE] for n in names]
        relative = [
            gain / table["neither"][n][HEADLINE] * 100.0
            for gain, n in zip(gains, names, strict=True)
            if table["neither"][n][HEADLINE] > 0.0
        ]
        improved = sum(1 for gain in gains if gain > 0.0)
        lines.append(
            f"{dataset:<16}{f'{improved}/{len(gains)}':>12}"
            f"{statistics.median(gains):>18.2f}{statistics.median(relative):>16.1f}"
        )
    lines.append("=== END ===")
    return "\n".join(lines)


def relative_polarity_block(tables: dict[str, Table]) -> str:
    """Is the effect *relative* polarity, or inverting the optical side specifically?

    `optical` and `thermal` present the same relative polarity to the matcher and differ only
    in which side carries the inversion; so do `neither` and `both`. If relative polarity is
    the whole story, the disagreement *within* each pair is small against the disagreement
    *across* them, and "invert the optical grayscale" is then a statement about the pair rather
    than about the optical image. Medians rather than means throughout: mace spans 7-900 px
    across this matcher list (P3-7's master table), and one classical failure would otherwise
    set the statistic for all twenty.
    """
    header = f"{'dataset':<16}{'within flip':>14}{'within same':>14}{'across':>12}{'ratio':>10}"
    lines = [
        "=== CMREG STAGE B: relative polarity vs. which side was inverted ===",
        "# within = median |mace difference| between two recipes of the SAME relative polarity",
        "#   (optical vs thermal, and neither vs both) -- ~0 if only the relation matters.",
        "# across = median |mace difference| between the two relative polarities.",
        "# ratio = across / max(within). Large means the matchers see the relation; ~1 means",
        "#   inverting the optical side specifically is what does the work.",
        header,
        "-" * len(header),
    ]
    for dataset, table in tables.items():
        if not {"optical", "thermal", "neither", "both"} <= set(table):
            lines.append(f"{dataset:<16}{'needs all four polarity cells':>50}")
            continue
        within_flip = _median_gap(table, "optical", "thermal")
        within_same = _median_gap(table, "neither", "both")
        across = _median_gap(table, ANCHOR, "neither")
        worst = max(within_flip, within_same)
        ratio = across / worst if worst > 0.0 else float("inf")
        lines.append(
            f"{dataset:<16}{within_flip:>14.2f}{within_same:>14.2f}{across:>12.2f}{ratio:>10.2f}"
        )
    lines.append("=== END ===")
    return "\n".join(lines)


def _median_gap(table: Table, left: str, right: str) -> float:
    gaps = [
        abs(table[left][name][HEADLINE] - table[right][name][HEADLINE])
        for name in _shared(table, left, right)
    ]
    return statistics.median(gaps) if gaps else float("nan")


def main_script(argv: list[str] | None = None) -> int:
    datasets = [cell.dataset for cell in CELLS]
    labels = [p.label for p in POLARITIES]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device; 'auto' picks the best available (default: cuda)",
    )
    parser.add_argument("--datasets", nargs="+", choices=datasets, default=datasets)
    parser.add_argument(
        "--polarities",
        nargs="+",
        choices=labels,
        default=labels,
        help="restrict the 2x2; the summary blocks need all four",
    )
    dry = parser.add_argument_group(
        "dry-run overrides",
        "For proving the plumbing on a laptop. A row produced under any of these is not a "
        "stage-B row and is not comparable with the rest of the table.",
    )
    dry.add_argument("--limit", type=int, help="override the config's 300-pair budget")
    dry.add_argument("--matchers", nargs="+", help="override the config's 20-matcher list")
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
    polarities = [p for p in POLARITIES if p.label in args.polarities]
    print(
        f"########## {len(cells) * len(polarities)} cells: "
        f"{', '.join(c.dataset for c in cells)} x {', '.join(p.label for p in polarities)} "
        "##########"
    )
    # Dataset-outer so an interrupted run leaves whole datasets finished rather than four
    # quarter-finished ones, and the tables below can be printed for what completed.
    completed: list[Cell] = []
    for cell in cells:
        # Every polarity is attempted even when one is skipped, rather than short-circuiting:
        # a dataset is skipped for want of a manifest, and printing that once per cell is what
        # tells a reader of the pasted console which of the sixteen are missing.
        ran = [run(cell, polarity, args.device, overrides) for polarity in polarities]
        if all(ran):
            completed.append(cell)

    tables = {cell.dataset: read_table(cell, polarities) for cell in completed}
    for cell in completed:
        print()
        print(dataset_table(cell, tables[cell.dataset]), flush=True)
    if tables:
        print()
        print(generality_block(tables), flush=True)
        print()
        print(relative_polarity_block(tables), flush=True)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
