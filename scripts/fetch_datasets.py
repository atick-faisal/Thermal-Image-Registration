"""Fetch the public optical-thermal datasets PLAN.md §4.1 benchmarks against.

Ported from ``../Thermal-To-Optical-Translation/scripts/fetch_datasets.py`` and trimmed to the
registration-relevant sets: that project needed detection datasets (CPLID, HIT-UAV, TTPLA)
this one has no use for, and needs the aerial set it never fetched.

Downloads land in the **sibling tree**, next to the MSRS and FLIR-aligned copies already
there, not under this repo::

    <root>/raw/<name>/        as downloaded
    <root>/processed/<name>/  written later by `cmreg ingest`

``--root`` defaults to that sibling and is a flag rather than a config field for the same
reason ``cmreg/device.py`` keeps the device out of ``config_hash()``: it differs between this
Mac and the Windows training box, and an experiment's hash must not.

Standalone script, not part of the ``cmreg`` package -- it owns its own ``logging.basicConfig``
the way ``cmreg/cli.py`` does for the package proper.

Verified 2026-08-24 on macOS:

- ``dronevehicle`` downloads cleanly from Hugging Face (three zips, 14 GB).
- ``llvip`` downloads cleanly from Google Drive (LLVIP.zip, 4.0 GB).
- ``m3fd`` **fails**: ``FileURLRetrievalError: Too many users have viewed or downloaded this
  file recently``. Google Drive's per-file quota, not a bug here and not fixable from this
  side; the message says it can take 24 hours to clear. Retry later. The IDs below are the
  *individual* files rather than the shared folder the sibling script used, which is a strict
  improvement: the folder also carries ``M3FD_Fusion.zip``, ``roadscene.zip`` and ``tno.zip``,
  none of which this project benchmarks, and fetching the folder downloads all four.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("fetch_datasets")

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# The sibling tree that already holds raw/{cplid,flir,hituav,msrs} and processed/{flir,msrs}.
DEFAULT_ROOT = Path(
    "/Users/ai/.GoogleDrive/Python/Iberdrola/Thermal-To-Optical-Translation/dataset"
)


@dataclass(frozen=True, slots=True)
class DatasetSource:
    name: str
    note: str
    # Exactly one of these is set; it selects the fetch strategy.
    hf_repo_id: str | None = None
    gdown_file_id: str | None = None


SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        name="dronevehicle",
        # 28,439 drone RGB-IR pairs (VisDrone/DroneVehicle, T-CSVT 2022). The aerial domain
        # family of PLAN.md §1, and the only one of these three that is not driving footage.
        # This HF mirror ships train/val/test as three zips; the GitHub original is a set of
        # Baidu/Drive links that gdown cannot follow.
        hf_repo_id="McCheng/DroneVehicle",
        note="aerial, 14 GB, train/val/test zips",
    ),
    DatasetSource(
        name="llvip",
        # bupt-ai-cz/LLVIP download_dataset.md -- the registered/aligned set, not the separate
        # "raw" download of unregistered pairs and video, which this project has no use for.
        gdown_file_id="1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv",
        note="driving/street, 4.0 GB, 15,488 pairs at 1280x1024",
    ),
    DatasetSource(
        name="m3fd",
        # JinyuanLiu-CV/TarDAL. The *detection* archive alone; see the module docstring for
        # why this is the file id and not the folder id the sibling script used.
        gdown_file_id="1C8kkYkj1Xls6UtvJ4h6UajiPcvaQ7eeI",
        note="driving/street, 4,200 pairs at 1024x768 -- Drive quota blocked 2026-08-24",
    ),
)

_BY_NAME = {source.name: source for source in SOURCES}

# On disk already, adapted by the sibling project. Named here so `--list` shows the whole
# inventory rather than only the part this script can fetch.
PREFETCHED = {
    "msrs": "driving/street, 1,524 pairs, adapted by the sibling project",
    "flir": "driving/street, FLIR-aligned, adapted by the sibling project",
}


class FetchError(RuntimeError):
    """Raised when a dataset name matches no known source."""


def fetch_huggingface(source: DatasetSource, dest: Path) -> None:
    assert source.hf_repo_id is not None
    from huggingface_hub import snapshot_download

    logger.info("downloading hf dataset %s -> %s", source.hf_repo_id, dest)
    snapshot_download(
        repo_id=source.hf_repo_id, repo_type="dataset", local_dir=str(dest), max_workers=8
    )


def fetch_gdown_file(source: DatasetSource, dest: Path) -> None:
    assert source.gdown_file_id is not None
    import gdown

    logger.info("downloading gdown file %s -> %s", source.gdown_file_id, dest)
    # gdown ships no py.typed marker, so pyright treats its re-exports as private.
    gdown.download(  # pyright: ignore[reportPrivateImportUsage]
        id=source.gdown_file_id, output=f"{dest}/", quiet=False
    )


def fetch_dataset(name: str, root: Path, force: bool = False) -> Path:
    """Fetch one dataset into ``root/raw/<name>``, skipping a populated destination."""
    source = _BY_NAME.get(name)
    if source is None:
        if name in PREFETCHED:
            raise FetchError(f"{name} is already on disk ({PREFETCHED[name]}); nothing to fetch")
        raise FetchError(f"unknown dataset {name!r}; known: {', '.join(sorted(_BY_NAME))}")

    dest = root / "raw" / name
    if dest.exists() and any(dest.iterdir()) and not force:
        logger.info("%s already populated at %s; skipping (use --force to refetch)", name, dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    if source.hf_repo_id is not None:
        fetch_huggingface(source, dest)
    else:
        fetch_gdown_file(source, dest)
    return dest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=[source.name for source in SOURCES],
        help="which datasets to fetch (default: all fetchable ones)",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="dataset tree root")
    parser.add_argument("--force", action="store_true", help="refetch even if already present")
    parser.add_argument("--list", action="store_true", help="list known datasets and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if args.list:
        for source in SOURCES:
            logger.info("%-14s fetchable  %s", source.name, source.note)
        for name, note in sorted(PREFETCHED.items()):
            logger.info("%-14s on disk    %s", name, note)
        return 0

    # One dataset's failure must not abandon the others: a Google Drive quota error on M3FD
    # is a routine outcome (see the module docstring) and says nothing about LLVIP.
    failed: list[str] = []
    for name in args.dataset:
        try:
            fetch_dataset(name, args.root, force=args.force)
        except FetchError as exc:
            logger.error("%s", exc)
            failed.append(name)
        except Exception as exc:  # network and quota failures are expected here
            logger.error("%s: fetch failed: %s", name, exc)
            failed.append(name)

    if failed:
        logger.error("failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
