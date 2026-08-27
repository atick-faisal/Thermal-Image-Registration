"""P2-12: publish each composing dataset's residual `R` as a calibration constant.

P3-7's stage-A run 1 made this the critical path. Ten of twenty matchers scored
`success_rate_5px` of exactly 0.0000 on `flir` over 300 pairs, and twelve structurally
unrelated matchers piled into a 4.5-5.8 px band that is the dataset's own residual rather than
anything about them (F1). Until `R` is composed into the Tier-1 ground truth, the headline
threshold measures the rig, the per-dataset floors differ (5.9 vs 4.7 px), and no cross-dataset
row of stages B-G is comparable. **Stage A must be re-run once after this script.**

What it does per dataset: three identity-warp cells -- one per matcher -- then
`cmreg calibrate` over all three, which takes the element-wise median of their consensus corner
fields and prints the constant with its across-matcher spread. Reading a stored `h` column is
all `calibrate` does, so combining the legs costs seconds; the cells themselves are the run.

Which datasets, and why only two: `experiments/GRID.md` §3 holds the policy. `flir` and `llvip`
compose; `msrs` (13% systematic) and `dronevehicle` (1%) must not, because there the
composition would be fitting noise. They are deliberately absent rather than commented out --
a cell that exists is a cell somebody eventually runs.

**`flir` is a check, not a measurement.** Its constant is already published (TASKS.md P1-1d,
1,013 pairs, three matchers) and checked in at `calibration/flir.json`. This script re-derives
it from 300 random pairs and prints both, so a disagreement beyond P1-1d's stated 1.23 px mean
/ 2.33 px worst case is visible. A disagreement is a **finding to record, not a file to
overwrite** -- `--out` writes `flir` to a scratch path for exactly that reason.

**Why 300 random pairs is enough for a constant.** P1-1c measured the asymmetry: a contiguous
50-pair slice of `flir` val estimated the *systematic* term at x0.98 of its full-split value and
the *random* term 3.6x too low. `R_bar` is an average over pairs, so it centres correctly even
under a bad sample; `scatter` is a spread over pairs and does not. Composition consumes the
systematic term alone, and 300 *random* pairs is a strictly better sample than the 50 correlated
ones that already got it right -- it is also stage A's own budget (GRID.md §4), so a constant and
the run that consumes it rest on the same evidence.

    uv run python scripts/p3b_calibrate.py --device cuda
    uv run python scripts/p3b_calibrate.py --device cuda --datasets llvip

Needs a GPU: `--device` defaults to `cuda` and `resolve_device` raises rather than falling back.
RoMa measured 23.9 s/pair on Mac CPU, so one 300-pair leg is ~2 h there against minutes on an
A100.

Resumable per cell, and a dataset with no pointer `data.yaml` is skipped with the command that
would produce it. Every block is meant to be copied out of the console whole -- the training
server returns text and not files, so a constant that does not survive a console copy never
reaches the repo.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from cmreg.cli import main

CONFIG = "experiments/p1_alignment_audit.yaml"
# The three legs of P1-1d, unchanged. Not a wider list: what a fourth matcher buys is a smaller
# standard error on an uncertainty that is already 30% of the scatter it must be small against,
# and what it costs is another 300-pair cell per dataset.
MATCHERS = ("roma", "superpoint-lightglue", "eloftr")
# GRID.md §4's anchor budget, and `subsample_seed` makes it a *random* draw rather than
# `images[:300]` -- see the module docstring on what the head slice cost P1-1b.
PAIRS = 300
SUBSAMPLE_SEED = 0


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    why: str
    # Where the constant lands, and it is never `calibration/<name>.json` for either dataset
    # here -- that path is what `eval/runner.py` composes from, and neither of these should be
    # picked up automatically. `flir` already has a better constant (1,013 pairs vs 300) and
    # this run only checks it; `llvip` was measured and **rejected** (see `why`), so its file
    # is evidence rather than an input.
    out: Path
    # Appended to the note stored in the file, so the JSON explains itself wherever it ends up.
    caveat: str

    @property
    def manifest(self) -> Path:
        return Path("dataset/processed") / self.name / "data.yaml"


DATASETS = (
    Dataset(
        "flir",
        "CHECK: does 300 random pairs reproduce P1-1d's 1,013-pair constant? "
        "ANSWERED 2026-08-27 -- yes, to 0.382 px mean corner distance.",
        out=Path("runs") / "p3b_check" / "flir.json",
        caveat=(
            "CHECK RUN against the published 1,013-pair constant in calibration/flir.json; "
            "not the published constant itself."
        ),
    ),
    Dataset(
        "llvip",
        "MEASURE: GRID.md §3 marked llvip 'compose' on 50 head-sliced pairs P1-1c flagged as "
        "a provisional lower bound. ANSWERED 2026-08-27 -- it does NOT compose: the constant "
        "is 2.21 px and the across-matcher worst case is 2.87 px, larger than the constant.",
        out=Path("calibration") / "rejected" / "llvip.json",
        caveat=(
            "MEASURED AND REJECTED -- llvip does NOT compose; see TASKS.md P2-12. Kept as the "
            "evidence for that decision, deliberately not at calibration/llvip.json where the "
            "runner would find it."
        ),
    ),
)

# `msrs` and `flir` are the sibling project's output -- this repo converts neither -- so
# pointing a reader at `cmreg ingest` alone would send them in a circle.
_HOW_TO_INGEST = {
    "flir": (
        "in the sibling repo: uv run python scripts/adapt_datasets.py --dataset flir"
        ", then back here: cmreg ingest flir"
    ),
    "llvip": "cmreg ingest llvip --dataset-root <tree holding raw/ and processed/>",
}


def cell_name(dataset: str, matcher: str) -> str:
    return f"p3b_audit_{dataset}_{matcher}"


def run_leg(dataset: Dataset, matcher: str, device: str) -> Path:
    """One matcher's identity-warp cell. Returns its run directory."""
    run_dir = Path("runs") / cell_name(dataset.name, matcher)
    if (run_dir / "pairs.parquet").exists():
        print(f"########## SKIP {run_dir.name} (already complete) ##########", flush=True)
        return run_dir
    print(f"########## {run_dir.name} ##########", flush=True)
    code = main(
        [
            "bench",
            "-c",
            CONFIG,
            "--data",
            str(dataset.manifest),
            "--split",
            "val",
            "--limit",
            str(PAIRS),
            "--subsample-seed",
            str(SUBSAMPLE_SEED),
            "--matchers",
            matcher,
            "--device",
            device,
            "--wandb",
            "--name",
            run_dir.name,
            "--run-dir",
            str(run_dir),
        ]
    )
    if code != 0:
        raise SystemExit(f"{run_dir.name} failed with exit code {code}")
    return run_dir


def run(dataset: Dataset, device: str) -> None:
    if not dataset.manifest.exists():
        print(
            f"########## SKIP {dataset.name}: no {dataset.manifest} ##########\n"
            f"  to produce it -> {_HOW_TO_INGEST[dataset.name]}",
            flush=True,
        )
        return

    print(f"\n########## {dataset.name} -- {dataset.why} ##########", flush=True)
    run_dirs = [run_leg(dataset, matcher, device) for matcher in MATCHERS]

    # Per leg first: the spread is a summary, and P1-1d's finding was *which* matcher stood
    # apart -- roma, on the dense/sparse boundary -- which only the individual blocks show.
    for run_dir in run_dirs:
        main(["residual", str(run_dir)])

    note = (
        f"P2-12, {PAIRS} random val pairs (subsample_seed {SUBSAMPLE_SEED}), identity warp, "
        f"element-wise median of {len(MATCHERS)} matchers' consensus corner fields. "
        f"{dataset.caveat}"
    )
    print(f"########## {dataset.name} calibration ##########", flush=True)
    code = main(
        ["calibrate", *[str(d) for d in run_dirs], "--out", str(dataset.out), "--note", note]
    )
    if code != 0:
        raise SystemExit(f"{dataset.name} calibration failed with exit code {code}")
    print(
        f"########## {dataset.name}: {dataset.out} -- NOT composed automatically ##########\n"
        "  Neither dataset here writes to calibration/<name>.json. Read the spread against the\n"
        "  magnitude before composing anything: a constant whose across-matcher worst case\n"
        "  approaches its own magnitude is not worth composing (P2-12, llvip).",
        flush=True,
    )


def main_script(argv: list[str] | None = None) -> int:
    names = [d.name for d in DATASETS]
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

    selected = [d for d in DATASETS if d.name in args.datasets]
    print(
        f"########## {len(selected)} datasets x {len(MATCHERS)} matchers x {PAIRS} pairs ##########"
    )
    for dataset in selected:
        run(dataset, args.device)
    print(
        "\n########## DONE -- copy every block above ##########\n"
        "  Next: commit calibration/llvip.json, then re-run stage A on all four datasets\n"
        "  (scripts/p3a_grid.py). TASKS.md P3-7 F1 -- run 1's rows are pre-composition floors.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
