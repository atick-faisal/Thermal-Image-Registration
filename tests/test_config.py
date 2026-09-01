"""The config layer's two load-bearing behaviours: unknown-key rejection and hash stability."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cmreg.config import (
    Config,
    ConfigError,
    Domain,
    EstimateConfig,
    Estimator,
    Platform,
    PreprocessConfig,
    WarpModel,
    deep_merge,
)


def _write(tmp_path: Path, data: dict) -> Path:
    target = tmp_path / "experiment.yaml"
    target.write_text(yaml.safe_dump(data))
    return target


def test_defaults_load_without_a_file() -> None:
    config = Config.load()
    assert config.data.split == "val"
    assert config.eval.domain is Domain.DRIVING
    assert config.eval.platform is Platform.PUBLIC


def test_unknown_keys_are_rejected_not_ignored(tmp_path: Path) -> None:
    """A silently-dropped typo costs a full sweep to discover."""
    path = _write(tmp_path, {"gt": {"rotaton_deg": 10.0}})
    with pytest.raises(ConfigError, match=r"gt\.rotaton_deg"):
        Config.load(path)


def test_every_error_is_reported_not_just_the_first(tmp_path: Path) -> None:
    path = _write(tmp_path, {"gt": {"typo_one": 1, "typo_two": 2}, "data": {"typo_three": 3}})
    with pytest.raises(ConfigError) as excinfo:
        Config.load(path)
    assert "3 errors" in str(excinfo.value)


def test_missing_config_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Config.load(tmp_path / "absent.yaml")


def test_overrides_merge_nestedly_without_wiping_the_section(tmp_path: Path) -> None:
    path = _write(tmp_path, {"gt": {"seed": 3, "rotation_deg": 12.0}})
    config = Config.load(path, {"gt": {"seed": 9}})
    assert config.gt.seed == 9
    assert config.gt.rotation_deg == 12.0


def test_config_hash_ignores_runtime() -> None:
    """The same experiment on two machines under two names must carry one hash."""
    base = Config.load()
    renamed = Config.load(overrides={"runtime": {"name": "other", "device": "cuda"}})
    assert renamed.config_hash() == base.config_hash()


def test_config_hash_tracks_the_science() -> None:
    assert Config.load(overrides={"gt": {"seed": 1}}).config_hash() != Config.load().config_hash()
    assert (
        Config.load(overrides={"data": {"split": "train"}}).config_hash()
        != Config.load().config_hash()
    )


def test_snapshot_round_trips(tmp_path: Path) -> None:
    config = Config.load(overrides={"runtime": {"name": "snap"}, "gt": {"seed": 5}})
    written = config.snapshot(tmp_path / "run")
    restored = Config.model_validate(yaml.safe_load(written.read_text()))
    assert restored.config_hash() == config.config_hash()
    assert restored.runtime.name == "snap"


def test_sections_are_frozen() -> None:
    config = Config.load()
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError on frozen
        config.gt.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("section", "payload", "message"),
    [
        ("gt", {"scale_min": 1.5, "scale_max": 1.2}, "scale_min <= scale_max"),
        ("gt", {"rotation_deg": -1.0}, ">= 0"),
        ("data", {"split": "test"}, "train"),
        ("data", {"limit": -1}, ">= 0"),
        ("eval", {"thresholds_px": [5.0, 3.0]}, "ascending"),
        ("eval", {"thresholds_px": []}, "non-empty"),
    ],
)
def test_impossible_values_are_rejected(section: str, payload: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Config.load(overrides={section: payload})


def test_deep_merge_leaves_untouched_keys_alone() -> None:
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3}


def test_the_reference_modality_defaults_to_the_other_side() -> None:
    """`None` means "whichever one `moving` is not". Two callers deriving that separately is
    how a benchmark quietly starts registering an image against itself."""
    from cmreg.config import Modality

    assert Config().gt.reference_modality is Modality.OPTICAL
    assert not Config().gt.is_monomodal
    flipped = Config.model_validate({"gt": {"moving": "optical"}})
    assert flipped.gt.reference_modality is Modality.THERMAL
    assert not flipped.gt.is_monomodal


def test_a_monomodal_config_is_a_different_experiment() -> None:
    """`reference` is scientific, so it is inside `config_hash()`: a control cell and the
    benchmark cell it controls for must not share a fingerprint (TASKS.md P1-1b)."""
    control = Config.model_validate({"gt": {"reference": "thermal", "moving": "thermal"}})
    assert control.gt.is_monomodal
    assert control.config_hash() != Config().config_hash()


def test_composing_a_residual_is_a_different_experiment(tmp_path: Path) -> None:
    """`gt.residual_calibration` is scientific, so it must move the hash.

    A run that composes a rig constant into its ground truth is not the same measurement as one
    that does not -- P3-7's F1 is the whole argument -- and two rows sharing a `config_hash`
    across that boundary would be pooled as if they were replicates.
    """
    plain = Config.model_validate({})
    composed = Config.model_validate({"gt": {"residual_calibration": str(tmp_path / "flir.json")}})
    assert composed.config_hash() != plain.config_hash()


def test_the_residual_calibration_defaults_to_none() -> None:
    """Tier-1 as originally specified. Composition is opt-in per dataset (GRID.md §3): `msrs`
    and `dronevehicle` must never compose, so the default cannot be "whatever file is there"."""
    assert Config.model_validate({}).gt.residual_calibration is None


def test_an_unswept_config_resolves_to_itself() -> None:
    """Every config written before P3-10 still describes exactly one estimation cell, and the
    runner's variant loop is therefore uniform rather than conditional."""
    config = EstimateConfig()
    assert not config.is_sweeping
    assert config.variants() == (config,)


def test_the_sweep_is_the_cross_product_of_the_two_axes() -> None:
    config = EstimateConfig(
        sweep_methods=(Estimator.MAGSAC, Estimator.LMEDS), sweep_thresholds_px=(1.0, 3.0, 5.0)
    )
    assert [variant.label for variant in config.variants()] == [
        "magsac@1px",
        "magsac@3px",
        "magsac@5px",
        "lmeds@1px",
        "lmeds@3px",
        "lmeds@5px",
    ]


def test_the_warp_model_is_the_third_axis_of_the_same_sweep() -> None:
    """P3-4a. Model-outer, so a swept directory groups by warp model first."""
    config = EstimateConfig(
        sweep_warp_models=(WarpModel.HOMOGRAPHY, WarpModel.AFFINE),
        sweep_methods=(Estimator.MAGSAC, Estimator.LMEDS),
    )
    assert [(v.warp_model.value, v.method.value) for v in config.variants()] == [
        ("homography", "magsac"),
        ("homography", "lmeds"),
        ("affine", "magsac"),
        ("affine", "lmeds"),
    ]


def test_the_default_config_hash_is_unchanged_by_the_warp_model_field() -> None:
    """The P3-4a exemption in `Config.config_hash`, pinned rather than trusted.

    `warp_model` is a *scalar* with a default, so unlike P3-10's empty sweep lists it would move
    every hash in the project -- and `scripts/stages.py::refuse_a_stale_run` would then refuse
    every completed stage A-D directory on the server for a change that altered no science. The
    literal below is the hash those directories were scored under; it must not move again.

    The exemption is keyed on `homography` specifically, so this test also fails if the default
    is ever changed, which is the point at which the exemption would silently re-point.
    """
    assert EstimateConfig().warp_model is WarpModel.HOMOGRAPHY
    assert Config().config_hash() == "04f02efbd8b566ed"


def test_the_default_config_hash_is_unchanged_by_the_resolution_field() -> None:
    """The P3-12a exemption, the third instance of the scalar rule and pinned the same way.

    `input_scale` always has a value, so without the exemption adding stage F's axis would move
    every hash in the project and `scripts/stages.py::refuse_a_stale_run` would refuse every
    completed stage A-E directory on the server for a change that altered no science. The
    literal is the hash those directories were scored under, and it must not move again.

    Asserting the default alongside it is the point at which the exemption would otherwise
    silently re-point, exactly as the `warp_model` test above does.
    """
    assert PreprocessConfig().input_scale == 1.0
    assert Config().config_hash() == "04f02efbd8b566ed"


def test_a_non_default_resolution_level_hashes_apart() -> None:
    """The property the resume guard needs: a run that matched at half resolution is a
    different experiment, and resuming one onto the other's directory is refused."""
    default = Config()
    halved = default.model_copy(
        update={"preprocess": default.preprocess.model_copy(update={"input_scale": 0.5})}
    )
    assert halved.config_hash() != default.config_hash()


def test_the_resolution_level_is_bounded_at_both_ends() -> None:
    """Below ~0.05 a 640x512 pair is 32x26 and no matcher's stride reaches it; above 4x a dense
    backend no longer fits the card."""
    for value in (0.0, 0.01, 4.5):
        with pytest.raises(ValidationError):
            PreprocessConfig(input_scale=value)


def test_a_non_default_warp_model_hashes_apart() -> None:
    """The property the resume guard actually needs: a run fitting a different model is a
    different experiment, and resuming one onto the other's directory is refused."""
    default = Config()
    affine = default.model_copy(
        update={"estimate": default.estimate.model_copy(update={"warp_model": WarpModel.AFFINE})}
    )
    assert affine.config_hash() != default.config_hash()


def test_the_axes_sweep_independently() -> None:
    """`sweep_methods` alone is four cells, not twelve: the unswept axis falls back to its
    scalar rather than being treated as empty."""
    methods_only = EstimateConfig(sweep_methods=(Estimator.MAGSAC, Estimator.RANSAC))
    assert [v.threshold_px for v in methods_only.variants()] == [3.0, 3.0]
    thresholds_only = EstimateConfig(sweep_thresholds_px=(3.0, 5.0))
    assert [v.method for v in thresholds_only.variants()] == [Estimator.MAGSAC] * 2


def test_a_variant_is_the_config_a_single_cell_run_would_have_used() -> None:
    """Variants carry empty sweep lists, which is what makes "a swept row equals a single-run
    row" a statement about the config layer and not only about the runner."""
    swept = EstimateConfig(
        sweep_methods=(Estimator.MAGSAC, Estimator.LMEDS), sweep_thresholds_px=(1.0, 3.0)
    )
    for variant in swept.variants():
        assert not variant.is_sweeping
        assert variant == EstimateConfig(method=variant.method, threshold_px=variant.threshold_px)


def test_the_anchor_must_be_one_of_the_swept_cells() -> None:
    """Otherwise the scalars are dead weight that still enter `config_hash`, and the block the
    runner prints would belong to no table (`eval/runner.py::_publish`)."""
    with pytest.raises(ValidationError):
        EstimateConfig(method=Estimator.RANSAC, sweep_methods=(Estimator.MAGSAC, Estimator.LMEDS))
    with pytest.raises(ValidationError):
        EstimateConfig(threshold_px=7.0, sweep_thresholds_px=(1.0, 3.0))
    with pytest.raises(ValidationError):
        EstimateConfig(
            warp_model=WarpModel.SIMILARITY,
            sweep_warp_models=(WarpModel.HOMOGRAPHY, WarpModel.AFFINE),
        )


def test_swept_thresholds_must_be_ascending_and_unique() -> None:
    """The swept values become the columns of a table that reaches a human by copy-paste."""
    with pytest.raises(ValidationError):
        EstimateConfig(threshold_px=1.0, sweep_thresholds_px=(5.0, 3.0, 1.0))
    with pytest.raises(ValidationError):
        EstimateConfig(sweep_thresholds_px=(3.0, 3.0))


def test_an_empty_sweep_leaves_config_hash_where_it_was() -> None:
    """X-5, and the resume guard in particular.

    Two defaulted fields would otherwise move every hash in the project, and
    `scripts/stages.py::refuse_a_stale_run` -- whose job is to refuse a directory scored under
    a *different experiment* -- would refuse every completed stage A/B/C directory for a code
    change that altered no science. The literal is the pre-P3-10 hash of the default config.
    """
    assert Config().config_hash() == "04f02efbd8b566ed"


def test_a_sweep_is_a_different_experiment() -> None:
    swept = Config(estimate=EstimateConfig(sweep_methods=(Estimator.MAGSAC, Estimator.LMEDS)))
    assert swept.config_hash() != Config().config_hash()
