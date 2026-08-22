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

logger = logging.getLogger("cmreg")

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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
