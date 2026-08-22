"""The evaluation cell, end to end.

The load-bearing test in this file is :func:`test_identical_modalities_register_to_subpixel`.
The cell composes a sampled homography, its inverse, a warp, a preprocessing scale and a
matcher's coordinate frame, and getting any one of them backwards produces plausible-looking
large errors on cross-modal data -- indistinguishable from the modality gap the benchmark
exists to measure. Registering an image against a known warp of *itself* is the only
configuration in which such an error has nowhere to hide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cmreg.config import Config, Variant
from cmreg.eval import RunnerError, run_benchmark
from cmreg.gt import generator, sample_homography
from cmreg.results import FILENAME, read_rows

SUBPIXEL_PX = 1.0


def base_config(manifest: Path, run_dir: Path, **overrides) -> Config:
    """A no-preprocessing, single-matcher cell on the given dataset."""
    declared = {
        "runtime": {"name": "test", "path": str(run_dir), "wandb": False},
        "data": {"manifest": str(manifest), "split": "val"},
        # Both modalities are the same image in the aligned fixture, so any recipe would be
        # applied to both. `none` keeps the test about geometry and nothing else.
        "preprocess": {"reference": Variant.NONE.value, "moving": Variant.NONE.value},
        "match": {"matchers": ["sift"]},
    }
    for section, values in overrides.items():
        declared.setdefault(section, {}).update(values)
    return Config.model_validate(declared)


def test_identical_modalities_register_to_subpixel(aligned_dataset: Path, tmp_path: Path) -> None:
    """The direction-convention pin. See the module docstring."""
    config = base_config(aligned_dataset / "data.yaml", tmp_path / "run")
    (summary,) = run_benchmark(config)
    assert summary.n_failed == 0
    assert summary.metrics["reg/mace"] < SUBPIXEL_PX
    assert summary.metrics["reg/epe_mean"] < SUBPIXEL_PX
    assert summary.metrics["reg/success_rate_3px"] == pytest.approx(1.0)


@pytest.mark.parametrize("factor", [2, 3])
def test_upsampling_does_not_inflate_the_reported_error(
    aligned_dataset: Path, tmp_path: Path, factor: int
) -> None:
    """The scale-plumbing pin. Keypoints found in an upsampled image are not in native pixels;
    if the runner failed to map them back, every metric would be inflated by ``factor`` and
    the P3-9 ablation would report interpolation as catastrophic."""
    config = base_config(
        aligned_dataset / "data.yaml",
        tmp_path / f"run{factor}",
        preprocess={"moving_upsample": factor},
    )
    (summary,) = run_benchmark(config)
    assert summary.metrics["reg/mace"] < SUBPIXEL_PX


def test_swapping_which_modality_moves_gives_the_same_answer(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """With identical modalities the choice of moving side is a relabelling, so a difference
    here would mean the reference and moving paths are not doing the same thing."""
    forward = run_benchmark(base_config(aligned_dataset / "data.yaml", tmp_path / "fwd"))[0]
    reversed_ = run_benchmark(
        base_config(aligned_dataset / "data.yaml", tmp_path / "rev", gt={"moving": "optical"})
    )[0]
    assert reversed_.metrics["reg/mace"] == pytest.approx(forward.metrics["reg/mace"], rel=1e-9)


def test_the_runner_and_the_gt_command_sample_the_same_warps(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """Both derive H from ``(seed, index)`` rather than sharing a file, so they agree by
    construction -- but only as long as they agree on the *index*, which is what this pins."""
    from cmreg.cli import main

    run_dir = tmp_path / "gt"
    assert (
        main(
            [
                "gt",
                "--data",
                str(aligned_dataset / "data.yaml"),
                "--split",
                "val",
                "--run-dir",
                str(run_dir),
            ]
        )
        == 0
    )
    recorded = json.loads((run_dir / "gt_val.json").read_text())
    config = base_config(aligned_dataset / "data.yaml", tmp_path / "bench")
    for index, entry in enumerate(recorded["pairs"]):
        shape = (entry["height"], entry["width"])
        direct = sample_homography(config.gt, generator(config.gt.seed, index), shape)
        assert np.allclose(direct.ravel(), entry["homography"])


def test_every_pair_produces_a_row_even_when_matching_fails(
    dataset_root: Path, tmp_path: Path
) -> None:
    """TASKS.md X-4. The noise fixture is unmatchable by construction -- 64x80 of uniform
    random pixels -- so this exercises the failure path over a whole split."""
    from tests.conftest import VAL_STEMS

    config = base_config(dataset_root / "data.yaml", tmp_path / "noise")
    (summary,) = run_benchmark(config)
    rows = read_rows(tmp_path / "noise")
    assert len(rows) == len(VAL_STEMS)
    assert summary.metrics["reg/n_pairs"] == len(VAL_STEMS)
    assert all(row.failure_reason is not None for row in rows if not row.success)


def test_a_run_leaves_a_parquet_file_and_a_config_snapshot(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    """The snapshot -- not the authored YAML -- is what the aggregator reads, so a result can
    always be traced back to the configuration that produced it."""
    run_dir = tmp_path / "artifacts"
    config = base_config(aligned_dataset / "data.yaml", run_dir)
    run_benchmark(config)
    assert (run_dir / FILENAME).is_file()
    assert (run_dir / "config.yaml").is_file()
    rows = read_rows(run_dir)
    assert {row.config_hash for row in rows} == {config.config_hash()}


def test_multiple_matchers_share_one_parquet_and_are_distinguished_by_column(
    aligned_dataset: Path, tmp_path: Path
) -> None:
    config = base_config(
        aligned_dataset / "data.yaml", tmp_path / "multi", match={"matchers": ["sift", "orb"]}
    )
    summaries = run_benchmark(config)
    assert len(summaries) == 2
    assert {row.matcher for row in read_rows(tmp_path / "multi")} == {"sift", "orb"}


def test_a_limit_caps_the_pairs(aligned_dataset: Path, tmp_path: Path) -> None:
    config = base_config(aligned_dataset / "data.yaml", tmp_path / "capped", data={"limit": 1})
    (summary,) = run_benchmark(config)
    assert summary.n_pairs == 1


def test_an_empty_split_fails_loudly(tmp_path: Path) -> None:
    import yaml

    root = tmp_path / "empty"
    for split in ("train", "val"):
        (root / split / "optical" / "images").mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/optical/images",
                "val": "val/optical/images",
                "rgbt": {"optical_token": "optical", "thermal_token": "thermal"},
            }
        )
    )
    with pytest.raises(RunnerError, match="no images"):
        run_benchmark(base_config(root / "data.yaml", tmp_path / "run"))
