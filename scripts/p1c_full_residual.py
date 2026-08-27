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
3. **P1-1d, added after the first run answered 1 and 2.** The FLIR roll is a rig constant
   (-1.2823 deg on val, -1.2841 on train), but composing it still leaves 6.56 px of per-pair
   scatter. That remainder is either real per-pair misalignment or roma's cross-modal
   localisation limit, and a matcher swap on the same cell is the discriminator.

Note both defaults this script exists to get right, because passing the config by hand gets them
wrong: `p1_alignment_audit.yaml` carries `device: cpu` and `limit: 50` -- it was written for Mac
CPU work -- so a hand-written `cmreg bench -c` against it silently measures 50 *consecutive*
frames on the CPU. P1-1c is the record of what that costs: on a driving set 50 consecutive frames
are one scene, and the scatter came back 3.6x too low.

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

# RoMa is the default: P1-1a established that three matchers spanning three decades of cost agree
# on FLIR's residual *magnitude* to within 4%, so for magnitude the cross-paradigm check is done
# and this run needs the accurate one. P1-1d re-opens it for the *scatter*, which P1-1a never
# measured -- hence `Cell.matcher` rather than one module-level constant.
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
    matcher: str = MATCHER

    @property
    def name(self) -> str:
        # The matcher is in the name only when it is not the default, so every P1-1c run
        # directory keeps the path it was already written to and stays skippable.
        suffix = "" if self.matcher == MATCHER else f"_{self.matcher}"
        return f"p1c_residual_{self.dataset}_{self.split}{suffix}"

    @property
    def manifest(self) -> Path:
        return Path("dataset/processed") / self.dataset / "data.yaml"


CELLS = (
    Cell("flir", "val", 0, "the constant P3-1 wants to compose (1,013 pairs)"),
    Cell("flir", "train", 0, "THE DECISIVE CELL: a rig roll must match val (4,129 pairs)"),
    Cell("msrs", "val", 0, "full split, 361 pairs"),
    Cell("llvip", "val", 1000, "capped; 1280x1024 and the slowest per pair"),
    Cell("dronevehicle", "val", 1000, "capped; is the 28% gross-failure rate stable?"),
    # P1-1d. The consensus is settled; what is left is whether the 6.56 px of *scatter* on this
    # exact cell is the data's (parallax -- a homography is exact for one depth plane, and a
    # two-camera rig sees depth-dependent disparity) or roma's cross-modal localisation limit.
    # A matcher swap separates them: parallax survives it, a localisation limit does not. Same
    # dataset, split and limit as the roma cell above so the two scatters are directly
    # comparable -- that comparability is the whole measurement.
    Cell(
        "flir",
        "val",
        0,
        "P1-1d: is the 6.56 px scatter the data's or roma's? (1,013 pairs)",
        matcher="superpoint-lightglue",
    ),
    # The third leg, added after P1-1d. roma and splg agree on how *much* consensus there is
    # (13%) and not on what it is made of -- 38% apart on rotation, opposite signs on the first
    # principal scale. Their two consensus fields differ by 3.44 px, which is 84% of the scatter
    # that difference has to be small against, so P3-1 cannot publish either one as the
    # calibration. With a third matcher the across-matcher spread becomes a measured uncertainty
    # to report beside the constant instead of a known-non-zero unknown.
    Cell(
        "flir",
        "val",
        0,
        "P1-1d: third leg -- how far apart do three matchers put the consensus?",
        matcher="eloftr",
    ),
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
                cell.matcher,
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
