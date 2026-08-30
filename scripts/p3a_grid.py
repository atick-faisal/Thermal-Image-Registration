"""P3-7 / Stage A: the matcher x dataset baseline grid, on the training server.

`experiments/GRID.md` is the design this discharges; `experiments/p3a_baseline_grid.yaml` is
the anchor cell it runs. This driver exists for the one thing neither can express: the four
dataset cells must differ **only** in their manifest, their run directory and their
domain/platform labels. Launching them by hand is how a cell ends up with a different limit or
a different device than the three it will be tabulated beside.

    uv run python scripts/p3a_grid.py
    uv run python scripts/p3a_grid.py --datasets flir dronevehicle

Needs a GPU. `--device` defaults to `cuda` and `resolve_device` raises rather than falling
back, deliberately: all 20 matchers cost ~145 s/pair on a Mac CPU (summing the P0-2 timings),
so the four cells would take ~48 h there against an estimated 1-2 h on an A100. Pass
`--device cpu` only to accept that knowingly.

Resumable, Windows-safe, and skipping a dataset with no pointer `data.yaml` rather than
failing the other three -- all of that is `scripts/stages.py`, which every stage driver shares.

Every block it prints is meant to be copied out of the console whole -- results reach the Mac
as console text and nothing else. The last block is the cross-dataset table, which is the one
that does not exist in any single run directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cmreg.cli import main
from stages import CELLS, CONFIG, Cell, DryRun, run_cell

# The headline column of the cross-dataset table. MACE rather than EPE because it is the metric
# PLAN.md §6.1 leads with, and rather than `auc_5px` because a ranking has to survive the
# datasets where every matcher scores zero.
HEADLINE = "reg/mace"


def run(cell: Cell, device: str, dry: DryRun) -> bool:
    """One dataset cell of stage A: the anchor config, relabelled for this dataset."""
    run_dir = Path("runs") / cell.name
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
        "--device",
        device,
        "--wandb" if dry.wandb else "--no-wandb",
        "--name",
        cell.name,
        "--run-dir",
        str(run_dir),
    ]
    # Both overrides are deliberately absent unless asked for: the config's 300 pairs and its
    # 20-matcher list are scientific choices (GRID.md §1, §4), and a driver that restated
    # either could drift from it.
    if cell.calibration is not None:
        argv += ["--residual-calibration", str(cell.calibration)]
    if dry.limit is not None:
        argv += ["--limit", str(dry.limit)]
    if dry.matchers is not None:
        argv += ["--matchers", ",".join(dry.matchers)]
    return run_cell(cell, run_dir, argv, f"########## {cell.name} -- {cell.why} ##########")


def cross_dataset_table(cells: list[Cell]) -> str:
    """One matcher per row, one dataset per column, `reg/mace` in the cells.

    The four run directories each hold their own matcher table; what none of them holds is the
    comparison *across* datasets, which is the master table of PLAN.md Figure 5 and the reason
    the stage exists. Built here rather than in `results/report.py` because it is the only
    place in the project that reads several runs at once -- P8-1's aggregation layer is where
    it belongs once there is more than one caller.
    """
    from cmreg.results import read_rows, summarize

    columns: dict[str, dict[str, float]] = {}
    scored: dict[str, int] = {}
    for cell in cells:
        rows = read_rows(Path("runs") / cell.name)
        names = list(dict.fromkeys(row.matcher for row in rows))
        summaries = {
            name: summarize([row for row in rows if row.matcher == name], (5.0,)) for name in names
        }
        columns[cell.dataset] = {name: s.metrics[HEADLINE] for name, s in summaries.items()}
        # Read back rather than assumed to be the config's 300: a dry run overrides the budget,
        # and a header that states the intended number over a table built from another one is
        # the kind of error a pasted block carries forever.
        scored[cell.dataset] = next(iter(summaries.values())).n_pairs

    # Named rather than counted: a reader of a pasted table cannot otherwise tell a composed
    # column from an uncomposed one, and the two are not comparable across a row.
    composed = ", ".join(c.dataset for c in cells if c.composes)
    matchers = list(dict.fromkeys(name for column in columns.values() for name in column))
    width = max(len("matcher"), *(len(name) for name in matchers)) + 2
    header = f"{'matcher':<{width}}" + "".join(f"{name:>16}" for name in columns)
    counts = ", ".join(f"{name} {scored[name]}" for name in columns)
    lines = [
        f"=== CMREG STAGE A: {HEADLINE} (px), val split, seed 0 ===",
        f"# pairs scored: {counts}",
        f"# R composed into the GT for: {composed or 'none'} (GRID.md \u00a73). A column with no "
        "composition is a floor, not an accuracy.",
        header,
        "-" * len(header),
    ]
    for matcher in matchers:
        cells_text = "".join(
            f"{column.get(matcher, float('nan')):>16.4f}" for column in columns.values()
        )
        lines.append(f"{matcher:<{width}}{cells_text}")
    lines.append("=== END ===")
    return "\n".join(lines)


def main_script(argv: list[str] | None = None) -> int:
    names = [cell.dataset for cell in CELLS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device; 'auto' picks the best available (default: cuda)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=names,
        default=names,
        help="restrict the run; default is all of them",
    )
    dry = parser.add_argument_group(
        "dry-run overrides",
        "For proving the plumbing on a laptop. A row produced under any of these is not a "
        "stage-A row and is not comparable with the rest of the table.",
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

    selected = [cell for cell in CELLS if cell.dataset in args.datasets]
    print(f"########## {len(selected)} cells: {', '.join(c.name for c in selected)} ##########")
    completed = [cell for cell in selected if run(cell, args.device, overrides)]

    for cell in completed:
        print(f"########## {cell.name} ##########", flush=True)
        main(["report", str(Path("runs") / cell.name)])
    if completed:
        print()
        print(cross_dataset_table(completed), flush=True)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
