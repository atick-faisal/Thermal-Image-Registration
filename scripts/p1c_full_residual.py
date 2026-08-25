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

Every block it prints is meant to be copied out of the console whole.
"""

from __future__ import annotations

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


CELLS = (
    Cell("flir", "val", 0, "the constant P3-1 wants to compose (1,013 pairs)"),
    Cell("flir", "train", 0, "THE DECISIVE CELL: a rig roll must match val (4,129 pairs)"),
    Cell("msrs", "val", 0, "full split, 361 pairs"),
    Cell("llvip", "val", 1000, "capped; 1280x1024 and the slowest per pair"),
    Cell("dronevehicle", "val", 1000, "capped; is the 28% gross-failure rate stable?"),
)


def run(cell: Cell) -> None:
    run_dir = Path("runs") / cell.name
    banner = f"########## {cell.name} -- {cell.why} ##########"
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
                "cuda",
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


def main_script() -> int:
    for cell in CELLS:
        run(cell)
    print("\n########## DONE -- copy every block above ##########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
