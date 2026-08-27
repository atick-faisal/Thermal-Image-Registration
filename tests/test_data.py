"""The data contract: pairing by substitution, the manifest as source of truth, frozen splits."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from cmreg.data import (
    DatasetManifest,
    ManifestError,
    Pairing,
    PairingError,
    SplitDriftError,
    freeze_split,
    load_split_manifest,
    select_pairs,
    verify_split,
    write_split_manifest,
)
from tests.conftest import TRAIN_STEMS, VAL_STEMS


def test_thermal_counterpart_is_derived_not_indexed() -> None:
    pairing = Pairing()
    optical = Path("dataset/val/optical/images/07_val_003.png")
    assert pairing.thermal_path(optical) == Path("dataset/val/thermal/images/07_val_003.png")
    assert pairing.optical_path(pairing.thermal_path(optical)) == optical


def test_ambiguous_segment_is_refused() -> None:
    """Two 'optical' segments means the substitution has no single right answer."""
    with pytest.raises(PairingError, match="found 2"):
        Pairing().thermal_path(Path("optical/val/optical/images/a.png"))
    with pytest.raises(PairingError, match="found 0"):
        Pairing().thermal_path(Path("val/rgb/images/a.png"))


def test_validate_pairs_reports_every_miss(tmp_path: Path) -> None:
    (tmp_path / "optical").mkdir()
    missing = [tmp_path / "optical" / f"{i}.png" for i in range(3)]
    with pytest.raises(FileNotFoundError, match="3/3 thermal counterparts missing"):
        Pairing().validate_pairs(missing)


def test_manifest_reads_splits_and_tokens(manifest_path: Path) -> None:
    manifest = DatasetManifest.load(manifest_path)
    assert manifest.train_images.name == "images"
    assert manifest.pairing == Pairing(optical_token="optical", thermal_token="thermal")
    assert len(manifest.images("train")) == len(TRAIN_STEMS)
    assert len(manifest.images("val")) == len(VAL_STEMS)


def test_manifest_accepts_a_directory(manifest_path: Path) -> None:
    assert DatasetManifest.load(manifest_path.parent).path == manifest_path.resolve()


def test_every_pair_in_the_fixture_resolves(manifest_path: Path) -> None:
    manifest = DatasetManifest.load(manifest_path)
    manifest.pairing.validate_pairs(manifest.images("val"))


def test_identical_tokens_are_refused(tmp_path: Path, dataset_root: Path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(dataset_root, root)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/optical/images",
                "val": "val/optical/images",
                "rgbt": {"optical_token": "images", "thermal_token": "images"},
            }
        )
    )
    with pytest.raises(ManifestError, match="must differ"):
        DatasetManifest.load(root / "data.yaml")


def test_missing_split_names_what_was_missing(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "."}))
    with pytest.raises(ManifestError, match="missing required 'val'"):
        DatasetManifest.load(tmp_path / "data.yaml")


def test_unknown_split_is_refused(manifest_path: Path) -> None:
    with pytest.raises(ManifestError, match="unknown split"):
        DatasetManifest.load(manifest_path).images("test")


def test_frozen_split_round_trips(manifest_path: Path, tmp_path: Path) -> None:
    manifest = DatasetManifest.load(manifest_path)
    record = freeze_split("fixture", manifest)
    assert record.train_stems == tuple(sorted(TRAIN_STEMS))
    assert record.val_stems == tuple(sorted(VAL_STEMS))

    path = write_split_manifest(record, tmp_path / "splits" / "fixture.json")
    assert load_split_manifest(path) == record
    verify_split(record, manifest)


def test_split_drift_is_loud(manifest_path: Path, tmp_path: Path) -> None:
    """A re-fetch that reshuffles files must fail, not silently change every number."""
    original = DatasetManifest.load(manifest_path)
    record = freeze_split("fixture", original)

    root = tmp_path / "drifted"
    shutil.copytree(original.root, root)
    victim = next(iter(sorted((root / "val" / "optical" / "images").iterdir())))
    victim.unlink()

    with pytest.raises(SplitDriftError, match=r"val: \+0 -1"):
        verify_split(record, DatasetManifest.load(root / "data.yaml"))


# --- subsample selection (TASKS.md P3-1) ------------------------------------------------
#
# `select_pairs` exists because P1-1c measured what the head slice costs: 50 consecutive
# frames of a driving set are one scene, so the systematic term came back right (x0.98) and
# the random one 3.6x too low. The tests below pin the three properties that finding needs --
# it is a real sample, it is reproducible, and it carries each pair's *split* index so a
# subsample and the full run it samples from agree about which warp pair i receives.

_IMAGES = tuple(Path(f"images/{i:03d}.png") for i in range(20))


def test_no_limit_selects_the_whole_split() -> None:
    assert select_pairs(_IMAGES, 0) == tuple(enumerate(_IMAGES))
    assert select_pairs(_IMAGES, len(_IMAGES) + 5, subsample_seed=1) == tuple(enumerate(_IMAGES))


def test_without_a_seed_the_selection_is_the_head_slice() -> None:
    """The pre-P1-1c behaviour, kept as the default so old configs still mean what they meant."""
    assert select_pairs(_IMAGES, 4) == tuple(enumerate(_IMAGES[:4]))


def test_a_seeded_subsample_is_not_the_head_slice() -> None:
    """The P1-1c regression itself. A sample that happened to return the first N pairs would
    reproduce the one-scene artefact while looking like a fix."""
    selected = select_pairs(_IMAGES, 5, subsample_seed=0)
    assert [path for _, path in selected] != list(_IMAGES[:5])


def test_a_seeded_subsample_is_reproducible_and_seed_dependent() -> None:
    assert select_pairs(_IMAGES, 6, subsample_seed=0) == select_pairs(_IMAGES, 6, subsample_seed=0)
    assert select_pairs(_IMAGES, 6, subsample_seed=0) != select_pairs(_IMAGES, 6, subsample_seed=1)


def test_a_subsample_is_a_distinct_ordered_subset_carrying_split_indices() -> None:
    selected = select_pairs(_IMAGES, 7, subsample_seed=3)
    indices = [index for index, _ in selected]
    assert len(selected) == 7
    assert len(set(indices)) == 7, "drawn without replacement"
    assert indices == sorted(indices), "split order is restored after sampling"
    # The index is the pair's position in the *split*, not in the subsample: it seeds the
    # synthetic warp (`gt/synthetic.py::warp_seed`), so re-numbering 0..N-1 would give the same
    # image a different warp here than in the full-split run.
    assert all(_IMAGES[index] == path for index, path in selected)


def test_the_subsample_does_not_consume_the_ambient_rng() -> None:
    """`seeding.py::seed_cell` reseeds `random` and numpy's global state per evaluation cell,
    so a selection drawn from either would depend on whatever ran before it in the process."""
    import random

    random.seed(1234)
    before = random.random()
    random.seed(1234)
    select_pairs(_IMAGES, 5, subsample_seed=7)
    assert random.random() == before
