"""``cmreg`` command line entry point.

Two conventions carried over from ``../Thermal-To-Optical-Translation/src/t2o/cli.py``:

1. **Heavy imports live inside each subcommand**, not at module level, so ``cmreg --version``
   and ``cmreg --help`` stay fast and importable without a GPU-capable torch build.
2. **An explicit flag -> config-path override table.** Deliberately partial: rarely-tweaked
   fields stay config-file-only, because a flag for every field is a flag nobody reads and an
   experiment whose identity lives in shell history rather than in a tracked file.

``basicConfig`` is called here and nowhere else in the package.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from cmreg import __version__
from cmreg.config.schema import Estimator, Modality, Variant

logger = logging.getLogger("cmreg")

# Where dataset bytes live: the sibling tree that already holds MSRS and FLIR-aligned, so that
# every optical-thermal set on this machine sits in one place rather than one per project.
# This repo keeps only the pointer manifests under `dataset/processed/<name>/data.yaml`.
DATASET_ROOT = Path(
    "/Users/ai/.GoogleDrive/Python/Iberdrola/Thermal-To-Optical-Translation/dataset"
)

_VARIANTS = [v.value for v in Variant]
_MODALITIES = [m.value for m in Modality]

# flag name -> dotted path into the config. See convention 2 above.
_OVERRIDES: dict[str, tuple[str, ...]] = {
    "device": ("runtime", "device"),
    "name": ("runtime", "name"),
    "run_dir": ("runtime", "path"),
    "wandb": ("runtime", "wandb"),
    "data": ("data", "manifest"),
    "split": ("data", "split"),
    "limit": ("data", "limit"),
    "seed": ("gt", "seed"),
    "moving": ("gt", "moving"),
    "reference": ("gt", "reference"),
    "matchers": ("match", "matchers"),
    "estimator": ("estimate", "method"),
    "threshold": ("estimate", "threshold_px"),
    "preprocess_ref": ("preprocess", "reference"),
    "preprocess_mov": ("preprocess", "moving"),
    "upsample": ("preprocess", "moving_upsample"),
}


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build a nested override mapping from whichever flags were actually given.

    ``None`` means "not supplied" and is skipped, so an omitted flag never shadows the value
    in the config file with a default that happens to look the same.
    """
    overrides: dict[str, Any] = {}
    for flag, path in _OVERRIDES.items():
        value = getattr(args, flag, None)
        if value is None:
            continue
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
    return overrides


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", type=Path, help="experiment YAML under experiments/")
    parser.add_argument("--device")
    parser.add_argument("--name")
    parser.add_argument("--run-dir", dest="run_dir", type=Path)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--data", type=Path, help="path to the dataset's data.yaml")
    parser.add_argument("--split", choices=("train", "val"))
    parser.add_argument("--limit", type=int, help="cap the number of pairs (0 = all)")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--moving", choices=_MODALITIES, help="which modality receives the synthetic warp"
    )
    parser.add_argument(
        "--reference",
        choices=_MODALITIES,
        help="reference modality (default: the other one; equal to --moving is the P1-1b "
        "mono-modal control)",
    )
    parser.add_argument(
        "--matchers",
        type=_comma_separated,
        help="comma-separated matcher names, e.g. 'sift,orb' (see `cmreg matchers`)",
    )
    parser.add_argument("--estimator", choices=[e.value for e in Estimator])
    parser.add_argument("--threshold", type=float, help="estimator inlier threshold in pixels")
    parser.add_argument("--preprocess-ref", dest="preprocess_ref", choices=_VARIANTS)
    parser.add_argument("--preprocess-mov", dest="preprocess_mov", choices=_VARIANTS)
    parser.add_argument("--upsample", type=int, help="thermal upsampling factor (1-8)")


def _comma_separated(value: str) -> tuple[str, ...]:
    """``--matchers sift,orb`` -> ``("sift", "orb")``.

    A comma list rather than ``action="append"``: sweeps are launched from shell loops, and a
    single token is far easier to build there than a repeated flag.
    """
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cmreg",
        description="Cross-modal optical-thermal image registration.",
    )
    parser.add_argument("--version", action="version", version=f"cmreg {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    gt = subparsers.add_parser("gt", help="generate Tier-1 synthetic-warp ground truth for a split")
    _add_config_args(gt)
    gt.set_defaults(handler=_run_gt)

    bench = subparsers.add_parser("bench", help="run one benchmark cell over a split")
    _add_config_args(bench)
    bench.set_defaults(handler=_run_bench)

    report = subparsers.add_parser(
        "report", help="re-render the console block from an existing run directory"
    )
    report.add_argument("run_dir", type=Path, help="a run directory, or a .parquet file")
    report.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(3.0, 5.0, 10.0),
        help="corner-error thresholds in pixels for AUC and success rate",
    )
    report.set_defaults(handler=_run_report)

    residual = subparsers.add_parser(
        "residual",
        help="decompose an identity-warp run's residual into its systematic and random parts",
    )
    residual.add_argument("run_dir", type=Path, help="a run directory, or a .parquet file")
    residual.set_defaults(handler=_run_residual)

    ingest = subparsers.add_parser(
        "ingest", help="adapt a raw public dataset and write its pointer manifest"
    )
    ingest.add_argument("dataset", nargs="?", help="dataset name (see --list)")
    ingest.add_argument(
        "--dataset-root",
        dest="dataset_root",
        type=Path,
        default=DATASET_ROOT,
        help=f"tree holding raw/ and processed/ (default: {DATASET_ROOT})",
    )
    ingest.add_argument(
        "--manifest-dir",
        dest="manifest_dir",
        type=Path,
        default=Path("dataset/processed"),
        help="where to write this repo's pointer data.yaml (default: dataset/processed)",
    )
    ingest.add_argument("--list", action="store_true", help="show the inventory table and exit")
    ingest.set_defaults(handler=_run_ingest)

    matchers = subparsers.add_parser("matchers", help="list the registered matchers")
    matchers.set_defaults(handler=_run_matchers)

    return parser


def _run_gt(args: argparse.Namespace) -> int:
    import json

    from cmreg.config import Config
    from cmreg.data import DatasetManifest
    from cmreg.gt import dense_displacement, generator, overlap_ratio, sample_homography
    from cmreg.imaging import read_shape

    config = Config.load(args.config, overrides_from_args(args))
    manifest = DatasetManifest.load(config.data.manifest)
    images = manifest.images(config.data.split)
    if config.data.limit:
        images = images[: config.data.limit]
    if not images:
        logger.error("no images in the '%s' split of %s", config.data.split, manifest.path)
        return 1

    manifest.pairing.validate_pairs(images)

    records = []
    for index, image in enumerate(images):
        shape = read_shape(image)
        homography = sample_homography(config.gt, generator(config.gt.seed, index), shape)
        records.append(
            {
                "stem": image.stem,
                "height": shape[0],
                "width": shape[1],
                # Row-major 9 floats. The dense field is *not* stored -- it is reconstructed
                # exactly from H and the shape (gt/synthetic.py explains the arithmetic).
                "homography": homography.ravel().tolist(),
                "overlap": overlap_ratio(dense_displacement(homography, shape)),
            }
        )

    run_dir = Path(config.runtime.path)
    config.snapshot(run_dir)
    target = run_dir / f"gt_{config.data.split}.json"
    target.write_text(
        json.dumps(
            {
                "config_hash": config.config_hash(),
                "manifest": str(manifest.path),
                "split": config.data.split,
                "pairs": records,
            },
            indent=2,
        )
        + "\n"
    )
    mean_overlap = sum(r["overlap"] for r in records) / len(records)
    logger.info(
        "wrote %d Tier-1 GT records to %s (mean overlap %.3f)", len(records), target, mean_overlap
    )
    return 0


def _run_bench(args: argparse.Namespace) -> int:
    from cmreg.config import Config
    from cmreg.eval import run_benchmark

    config = Config.load(args.config, overrides_from_args(args))
    run_benchmark(config)
    return 0


def _run_report(args: argparse.Namespace) -> int:
    from cmreg.eval import COMPARISON_KEYS
    from cmreg.results import read_rows, render, render_comparison, summarize

    rows = read_rows(args.run_dir)
    # Preserve first-appearance order rather than sorting: it is the order the run produced
    # them in, which is the order the original block was printed in, so a re-render is
    # diffable against a pasted one.
    names = list(dict.fromkeys(row.matcher for row in rows))
    summaries = [
        summarize([row for row in rows if row.matcher == name], tuple(args.thresholds))
        for name in names
    ]
    for summary in summaries:
        print(render(summary))
    if len(summaries) > 1:
        print(render_comparison(summaries, COMPARISON_KEYS))
    return 0


def _run_residual(args: argparse.Namespace) -> int:
    """TASKS.md P1-1b. Only meaningful on an identity-warp run -- see `analysis/residual.py`,
    which states that precondition and why the rows alone cannot enforce it."""
    from cmreg.analysis.residual import AnalysisError, by_matcher, render
    from cmreg.results import read_rows

    rows = read_rows(args.run_dir)
    try:
        structures = by_matcher(rows)
    except AnalysisError as exc:
        logger.error("%s: %s", args.run_dir, exc)
        return 1
    for structure in structures:
        print(render(structure))
    return 0


def _run_ingest(args: argparse.Namespace) -> int:
    """Adapt one raw dataset, or list what is already adapted.

    The dataset bytes live in ``--dataset-root`` (the sibling tree that already holds MSRS and
    FLIR-aligned); this repo keeps only a pointer ``data.yaml``. That root is a flag rather
    than a config field for the reason ``cmreg/device.py`` records for the device: it differs
    between the Mac and the Windows training box, and an experiment's ``config_hash()`` must
    not depend on where its files happen to sit.
    """
    from cmreg.data import DatasetManifest
    from cmreg.data.adapters import adapt, available, write_pointer_manifest

    processed = args.dataset_root / "processed"
    if args.list:
        rows = []
        for name in sorted({*available(), *(p.name for p in _adapted_dirs(processed))}):
            manifest = args.manifest_dir / name / "data.yaml"
            if not manifest.is_file():
                rows.append((name, "-", "-", "not ingested"))
                continue
            loaded = DatasetManifest.load(manifest)
            rows.append(
                (
                    name,
                    str(len(loaded.images("train"))),
                    str(len(loaded.images("val"))),
                    str(loaded.root),
                )
            )
        width = max(len(row[0]) for row in rows)
        print(f"{'dataset':<{width}}  {'train':>7}  {'val':>7}  location")
        for name, train, val, where in rows:
            print(f"{name:<{width}}  {train:>7}  {val:>7}  {where}")
        return 0

    if not args.dataset:
        logger.error("ingest needs a dataset name, or --list; known: %s", ", ".join(available()))
        return 1

    inventory = adapt(
        args.dataset, args.dataset_root / "raw" / args.dataset, processed / args.dataset
    )
    manifest = write_pointer_manifest(
        args.manifest_dir / inventory.name / "data.yaml", processed / inventory.name
    )
    # Load it back rather than trusting the write: the pointer is the only thing standing
    # between an adapted tree and every downstream run, and a manifest that does not resolve
    # would otherwise surface as "no images in the 'val' split" much later.
    loaded = DatasetManifest.load(manifest)
    loaded.pairing.validate_pairs(loaded.images("val"))

    logger.info(
        "%s: %d train + %d val pairs at %s [%s/%s]%s",
        inventory.name,
        inventory.train_pairs,
        inventory.val_pairs,
        inventory.resolution,
        inventory.domain,
        inventory.platform,
        f" -- {inventory.note}" if inventory.note else "",
    )
    logger.info("manifest: %s -> %s", manifest, loaded.root)
    return 0


def _adapted_dirs(processed: Path) -> list[Path]:
    return sorted(p for p in processed.iterdir() if p.is_dir()) if processed.is_dir() else []


def _run_matchers(args: argparse.Namespace) -> int:
    del args
    from cmreg.matchers import available

    for name in available():
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
