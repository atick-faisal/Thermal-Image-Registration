"""P1-1c: re-measure the alignment residual on full splits, on the training server.

P1-1b decomposed the residual on 50 val pairs per dataset and found FLIR-aligned carries a
fixed -1.18 deg camera roll (83% systematic). P3-1 intends to compose that consensus into the
Tier-1 GT as a published calibration constant, and 50 pairs is too thin a basis for a constant
that every subsequent number inherits.

Two questions this answers, in order of importance:

1. **Is the FLIR roll split-invariant?** A rig property must be identical on train and val. If
   the two splits disagree, it is not a rig property, and composing it into the GT is invalid --
   which retires the whole plan for FLIR. This is the decisive cell and the reason for the run.
2. Do the other datasets' residuals hold at full scale, and is DroneVehicle's 28% gross-failure
   rate (14 of 50) a stable property rather than a small-sample artefact?

Resumable: a cell whose `pairs.parquet` already exists is skipped, so an interrupted run picks
up where it stopped. Windows-safe -- the CLI is called in-process, not through a shell.

    uv run python scripts/p1c_full_residual.py
    uv run python scripts/p1c_full_residual.py --datasets flir msrs

Needs a GPU. `--device` defaults to `cuda` and `resolve_device` raises rather than falling back,
which is the intended behaviour: RoMa measured 23.9 s/pair on a Mac CPU, so the two FLIR cells
alone would take ~34 h there against roughly an hour on a GPU. Pass `--device cpu` only to
deliberately accept that.

A dataset with no pointer `data.yaml` is **skipped with the command that would produce it**,
not treated as an error. The four sets do not arrive by one route: `llvip` and `dronevehicle`
have adapters in this repo, while `msrs` and `flir` are converted by the sibling project and
only *verified* here (`data/adapters/sibling.py`). A run that had to be all-or-nothing would
block the decisive FLIR cell behind a 13 GB DroneVehicle download it does not need.

Every block it prints is meant to be copied out of the console whole.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from cmreg.cli import main

# RoMa alone: P1-1a already established that three matchers spanning three decades of cost agree
# on FLIR to within 4%, so the cross-paradigm check is done and this run needs the accurate one.
MATCHER = "roma"
CONFIG = "experiments/p1_alignment_audit.yaml"


@dataclass(frozen=True, slots=True)
class Cell:
    dataset: str
    split: str
    # 0 = the whole split. Capped only where the split is large enough that the extra pairs buy
    # no precision -- 1,000 pairs already pins a mean to a few percent.
    limit: int
    why: str

    @property
    def name(self) -> str:
        return f"p1c_residual_{self.dataset}_{self.split}"

    @property
    def manifest(self) -> Path:
        return Path("dataset/processed") / self.dataset / "data.yaml"


CELLS = (
    Cell("flir", "val", 0, "the constant P3-1 wants to compose (1,013 pairs)"),
    Cell("flir", "train", 0, "THE DECISIVE CELL: a rig roll must match val (4,129 pairs)"),
    Cell("msrs", "val", 0, "full split, 361 pairs"),
    Cell("llvip", "val", 1000, "capped; 1280x1024 and the slowest per pair"),
    Cell("dronevehicle", "val", 1000, "capped; is the 28% gross-failure rate stable?"),
)


# How each dataset's processed tree comes into being. `msrs` and `flir` are the sibling
# project's output -- this repo converts neither -- so pointing a reader at `cmreg ingest` alone
# would send them in a circle.
_HOW_TO_INGEST = {
    "msrs": (
        "in the sibling repo: uv run python scripts/adapt_datasets.py --dataset msrs"
        ", then back here: cmreg ingest msrs"
    ),
    "flir": (
        "in the sibling repo: uv run python scripts/adapt_datasets.py --dataset flir"
        ", then back here: cmreg ingest flir"
    ),
    "llvip": "cmreg ingest llvip --dataset-root <tree holding raw/ and processed/>",
    "dronevehicle": "cmreg ingest dronevehicle --dataset-root <tree holding raw/ and processed/>",
}


def run(cell: Cell, device: str) -> None:
    run_dir = Path("runs") / cell.name
    banner = f"########## {cell.name} -- {cell.why} ##########"
    if not cell.manifest.exists():
        print(
            f"########## SKIP {cell.name}: no {cell.manifest} ##########\n"
            f"  to produce it -> {_HOW_TO_INGEST[cell.dataset]}",
            flush=True,
        )
        return
    if (run_dir / "pairs.parquet").exists():
        print(f"########## SKIP {cell.name} (already complete) ##########", flush=True)
    else:
        print(banner, flush=True)
        code = main(
            [
                "bench",
                "-c",
                CONFIG,
                "--data",
                f"dataset/processed/{cell.dataset}/data.yaml",
                "--split",
                cell.split,
                "--limit",
                str(cell.limit),
                "--matchers",
                MATCHER,
                "--device",
                device,
                "--wandb",
                "--name",
                cell.name,
                "--run-dir",
                str(run_dir),
            ]
        )
        if code != 0:
            raise SystemExit(f"{cell.name} failed with exit code {code}")
    print(banner, flush=True)
    main(["residual", str(run_dir)])


def main_script(argv: list[str] | None = None) -> int:
    names = sorted({cell.dataset for cell in CELLS})
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
    args = parser.parse_args(argv)

    selected = [cell for cell in CELLS if cell.dataset in args.datasets]
    print(f"########## {len(selected)} cells: {', '.join(c.name for c in selected)} ##########")
    for cell in selected:
        run(cell, args.device)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
