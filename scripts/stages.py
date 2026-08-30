"""What every Phase-3 stage driver shares: the cells, the guards, and the invocation.

`scripts/p3a_grid.py` carried all of this while stages A and B were the only consumers, with a
TODO to extract it "when stage D becomes the third consumer; two is not yet worth the churn".
Stage D (P3-10, `scripts/p3d_estimator.py`) is that third consumer, so here it is.

Nothing here is stage-specific. A stage supplies the flags in its `argv` and the directory it
puts the rows in; the preconditions -- a manifest, a calibration constant where
`experiments/GRID.md` §3 requires one, and rows that were scored under *this* experiment -- are
identical for all of them, and a second copy of any of them is a second place for one to be
dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cmreg.cli import main

CONFIG = "experiments/p3a_baseline_grid.yaml"
REDUCED_8 = (
    "roma",
    "minima-roma",
    "matchanything-roma",
    "eloftr",
    "xoftr",
    "superpoint-lightglue",
    "xfeat",
    "sift",
)


@dataclass(frozen=True, slots=True)
class Cell:
    dataset: str
    # Reported per-domain and never pooled (config/schema.py::Domain): `dronevehicle` is the
    # only aerial set here, and averaging it into the driving rows would hide that it is the
    # hard case the paper is about.
    domain: str
    platform: str
    why: str
    # Whether this dataset's residual `R` is composed into the Tier-1 GT (TASKS.md P2-12).
    # Per dataset and carried from `experiments/GRID.md` §3, which is the one place that
    # decision lives: `msrs` (13% systematic) and `dronevehicle` (1%) must not compose,
    # because there the constant would be fitting noise. A dataset marked True whose
    # `calibration/<name>.json` is missing **fails the cell** rather than quietly running
    # uncomposed -- an uncomposed row looks exactly like a composed one in the table, and
    # P3-7's F1 is the record of what a floor nobody noticed costs.
    composes: bool

    @property
    def name(self) -> str:
        return f"p3a_{self.dataset}"

    @property
    def manifest(self) -> Path:
        return Path("dataset/processed") / self.dataset / "data.yaml"

    @property
    def calibration(self) -> Path | None:
        return Path("calibration") / f"{self.dataset}.json" if self.composes else None


@dataclass(frozen=True, slots=True)
class DryRun:
    """The three ways to make this driver cheap enough to test on a laptop.

    Grouped into one object rather than passed as loose arguments so that the docstring below
    can say the thing that matters once: **a stage-A row produced under any of these is not
    comparable with the rest of the table.** They exist to prove the plumbing runs before it
    reaches the server, and for nothing else.
    """

    limit: int | None = None
    matchers: tuple[str, ...] | None = None
    wandb: bool = True


CELLS = (
    Cell(
        "flir",
        "driving",
        "public",
        "the best-characterised set (P1-1c/d): 5.9 px residual, composed",
        composes=True,
    ),
    Cell("msrs", "driving", "public", "4.7 px residual, 13% systematic", composes=False),
    Cell(
        "llvip",
        "driving",
        "public",
        "night-time; where sparse detectors fail outright. Does NOT compose (P2-12)",
        composes=False,
    ),
    Cell(
        "dronevehicle",
        "aerial",
        "drone",
        "THE AERIAL CELL: 4.7 px median with a 28% gross-failure rate",
        composes=False,
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


def intended_hash(argv: list[str]) -> str:
    """The `config_hash` the cell described by `argv` would stamp on its rows.

    Resolved through `cmreg`'s own parser and override table rather than by rebuilding the
    config here: a second path to the same answer is a second path that can disagree with the
    run it is supposed to be checking.
    """
    from cmreg.cli import build_parser, overrides_from_args
    from cmreg.config import Config

    args = build_parser().parse_args(argv)
    return Config.load(args.config, overrides_from_args(args)).config_hash()


def refuse_a_stale_run(run_dir: Path, wanted_hash: str, wanted_calibration: str | None) -> None:
    """Refuse to resume onto a run directory that was scored under a *different experiment*.

    The skip this guards exists so an interrupted run picks up where it stopped. It cannot tell
    that apart from a *stale* run, and P2-12 created exactly that case: stage-A run 2 asked for
    `flir` composed, found run 1's pre-composition `pairs.parquet` sitting there, skipped the
    cell, and printed run 1's floors into the composed table under the composed banner. Nothing
    in the pasted output said so -- an uncomposed row looks exactly like a composed one.

    The check is on `config_hash`, which covers **every scientific field at once**, so a stage
    that varies a new axis inherits it: stage B (P3-8) varies the preprocess pair and stage D
    (P3-10) varies the estimation sweep, and a stale directory differing in either is exactly as
    invisible in a printed table as the composition was. A swept directory needs nothing special
    here -- its twelve variants share one config and so one hash, which is why
    `Config.config_hash` keeps an *empty* sweep out of its payload: without that, adding those
    two fields would have made this guard refuse every stage A-C directory already on the
    server, for a code change that altered no science.
    `residual_calibration` is inside that hash and is reported separately anyway, because it is
    the mismatch whose fix an operator has to be told (produce or remove a constant), where the
    general one only needs the directory named.

    Raising rather than re-running: the stale directory is 25-35 min of GPU time and deleting it
    is the operator's call, not a driver's side effect.
    """
    from cmreg.results import read_rows

    row = read_rows(run_dir)[0]
    if row.config_hash == wanted_hash and row.residual_calibration == wanted_calibration:
        return
    if row.residual_calibration != wanted_calibration:
        detail = (
            f"was scored with residual_calibration={row.residual_calibration!r} but this cell "
            f"now wants {wanted_calibration!r} (GRID.md \u00a73)"
        )
    else:
        detail = (
            f"was scored under config_hash {row.config_hash} but this cell resolves to "
            f"{wanted_hash}"
        )
    raise SystemExit(
        f"{run_dir} {detail}. Resuming would tabulate the old rows under the new banner. "
        f"Delete the directory to re-run it:\n"
        f"  rm -rf {run_dir}     (PowerShell: Remove-Item -Recurse -Force {run_dir})"
    )


def run_cell(cell: Cell, run_dir: Path, argv: list[str], banner: str) -> bool:
    """Run one `cmreg bench` invocation. False when it was skipped for want of a manifest.

    Every stage differs only in the flags it puts in `argv` and the directory it puts the rows
    in. See this module's docstring for why the preconditions live here rather than in each of
    them.
    """
    if not cell.manifest.exists():
        print(
            f"########## SKIP {run_dir.name}: no {cell.manifest} ##########\n"
            f"  to produce it -> {_HOW_TO_INGEST[cell.dataset]}",
            flush=True,
        )
        return False
    if cell.calibration is not None and not cell.calibration.exists():
        # Fatal, not a skip. GRID.md §3 marks this dataset "compose", and running it without
        # the constant produces a table that looks identical and measures the rig.
        raise SystemExit(
            f"{run_dir.name}: {cell.calibration} is missing. GRID.md §3 marks {cell.dataset} as "
            "composing its residual; produce the constant with "
            f"`uv run python scripts/p3b_calibrate.py --datasets {cell.dataset}` first."
        )
    if (run_dir / "pairs.parquet").exists():
        from cmreg.gt import load_calibration

        wanted = None if cell.calibration is None else load_calibration(cell.calibration).digest()
        refuse_a_stale_run(run_dir, intended_hash(argv), wanted)
        print(f"########## SKIP {run_dir.name} (already complete) ##########", flush=True)
        return True

    print(banner, flush=True)
    code = main(argv)
    if code != 0:
        raise SystemExit(f"{run_dir.name} failed with exit code {code}")
    return True
