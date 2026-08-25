"""The experiment config: pydantic sections, YAML loading, hashing, and snapshotting.

An experiment *is* a config file. ``experiments/*.yaml`` is tracked in git; ``runs/`` is
not. House rule (AGENTS.md): **every field carries an inline comment naming its failure
mode.** A knob whose failure mode nobody can state is a knob nobody should be turning.

Paths are deliberately never checked for existence here -- configs are authored on the Mac
and resolved on the Windows training box. Existence is a startup check in the layer that
consumes the path, where the error message can say what it was looking for.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from cmreg.config.base import ConfigBase, ConfigError, as_config_error, deep_merge

# Matches SplitManifest's sha256-hex-truncated convention (data/splits.py).
_HASH_LENGTH = 16


class Domain(StrEnum):
    """The three domain families of PLAN.md §1. Reported separately, never pooled: a single
    pooled number hides that the aerial domain is the hard case."""

    DRIVING = "driving"
    AERIAL = "aerial"
    INDUSTRIAL = "industrial"


class Platform(StrEnum):
    """Capture platform. Drives the per-platform breakdown (TASKS.md P3-15) and is a
    required W&B tag."""

    HANDHELD = "handheld"
    SMARTPHONE = "smartphone"
    DRONE = "drone"
    PUBLIC = "public"


class Modality(StrEnum):
    """Which of the two images something refers to.

    Named ``optical``/``thermal`` rather than ``moving``/``fixed`` because the choice of which
    one moves is per-experiment (``GTConfig.moving``): the production path warps optical into
    thermal geometry, the public benchmarks are indexed the other way (``data/pairing.py``)."""

    OPTICAL = "optical"
    THERMAL = "thermal"


class Variant(StrEnum):
    """A named preprocessing recipe (TASKS.md P3-2).

    Recipes rather than a composable op list: the benchmark reports one string per axis, and a
    free-form pipeline spelling would make two runs that did the same thing unjoinable in the
    results store. Adding a recipe is one registry entry (``preprocess/variants.py``)."""

    NONE = "none"
    INVERT = "invert"
    CLAHE = "clahe"
    CLAHE_INVERT = "clahe_invert"
    PERCENTILE = "percentile"
    PERCENTILE_INVERT = "percentile_invert"
    GRADIENT = "gradient"


class Interpolation(StrEnum):
    """Resampling kernel for thermal upsampling (TASKS.md P3-9)."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


class Estimator(StrEnum):
    """Robust homography estimators (TASKS.md P3-3, PLAN.md §4.1)."""

    MAGSAC = "magsac"
    RANSAC = "ransac"
    LMEDS = "lmeds"
    PROSAC = "prosac"


class RuntimeConfig(ConfigBase):
    """Where and how a run executes. **Excluded wholesale from `config_hash()`** -- the same
    experiment on two machines under two names must carry one hash, or the hash means
    nothing."""

    # "auto" resolves at startup. A hardcoded "cuda" fails the Mac dev box; a hardcoded
    # "cpu" silently wastes the A100s.
    device: str = "auto"
    # Names the run *directory* and the `run_name` column. Two runs sharing a name overwrite
    # each other's outputs and become indistinguishable in the results store. The W&B run name
    # is not this -- it is derived per matcher into TASKS.md §0's frozen
    # `{phase}_{method}_{dataset}_{variant}_s{seed}` format (`eval/runner.py::_publish`), so a
    # campaign that overrides the matcher on the command line cannot file every cell under one
    # W&B name.
    name: str = "unnamed"
    # Run output root. Under `runs/`, which is git-ignored: results must not travel back to
    # the server through git.
    path: Path = Path("runs/unnamed")
    # W&B off by default so tests and smoke runs never touch the network.
    wandb: bool = False
    # One of the four projects frozen in TASKS.md §0. A typo here scatters a campaign across
    # two projects and breaks cross-run comparison.
    wandb_project: str = "cmreg-bench"
    # W&B `group=` for a factorial cell, so seeds aggregate automatically. Empty means
    # ungrouped, which loses that aggregation silently.
    group: str = ""


class DataConfig(ConfigBase):
    """Which pairs this run sees."""

    # Path to the dataset's data.yaml. The manifest is the single source of truth; anything
    # re-declared here could disagree with what is on disk.
    manifest: Path = Path("dataset/processed/msrs/data.yaml")
    # "train" or "val". Evaluating on train is the leakage failure X-6 exists to catch, so
    # it is spelled out in the config rather than defaulted implicitly.
    split: str = "val"
    # Cap on pairs, for smoke runs. 0 means all; a nonzero value left in a real config
    # silently reports a subset as if it were the whole benchmark.
    limit: int = 0

    @field_validator("split")
    @classmethod
    def _known_split(cls, value: str) -> str:
        if value not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {value!r}")
        return value

    @field_validator("limit")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"limit must be >= 0 (0 means all), got {value}")
        return value


class GTConfig(ConfigBase):
    """Tier-1 synthetic-warp ground truth (PLAN.md §5, TASKS.md P2-1).

    These bounds define what "misaligned" means for every downstream number, so changing one
    changes the benchmark. They live in the hashed part of the config for exactly that reason.
    """

    # Scientific, not invocational -- so it belongs here and not in `runtime`. A silently
    # changed seed is drift that no amount of averaging will reveal.
    seed: int = 0
    # Which modality receives the synthetic warp; the other is the reference frame every
    # metric is expressed in. Flipping it silently changes what the benchmark measures --
    # thermal is typically the lower-resolution, harder side, so warping optical instead
    # makes every number look better for no methodological reason.
    moving: Modality = Modality.THERMAL
    # Which modality is the reference frame. `None` means "whichever one `moving` is not" --
    # the cross-modal benchmark, and what every run before TASKS.md P1-1b did. Setting it
    # *equal* to `moving` is the mono-modal control: both sides are read from one modality, so
    # the pair's own cross-modal offset is gone and what is left is the floor of the pipeline
    # itself. Failure mode: left equal to `moving` by accident, and a control cell is filed as
    # a benchmark row roughly an order of magnitude too good.
    reference: Modality | None = None
    # Rotation half-range in degrees (PLAN.md §5 Tier 1: +/-30 deg). Widening it past what
    # any matcher tolerates turns the whole benchmark into a floor of failures.
    rotation_deg: float = 30.0
    # Scale sampled log-uniformly in [min, max] so 0.8 and 1.25 are equally likely; sampling
    # uniformly would bias every run toward magnification.
    scale_min: float = 0.8
    scale_max: float = 1.25
    # Corner displacement as a fraction of image size, before the homography is fitted to the
    # displaced corners. Too large and the warp folds the image over itself.
    perspective: float = 0.05
    # Translation as a fraction of image size. Combined with `perspective` this is what
    # controls how much of the source leaves the frame.
    translation: float = 0.05

    @model_validator(mode="after")
    def _scale_range_ordered(self) -> Self:
        if not 0.0 < self.scale_min <= self.scale_max:
            raise ValueError(
                f"require 0 < scale_min <= scale_max, got {self.scale_min} / {self.scale_max}"
            )
        return self

    @field_validator("rotation_deg", "perspective", "translation")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError(f"must be >= 0, got {value}")
        return value

    @property
    def reference_modality(self) -> Modality:
        """The resolved reference side. A property so no caller re-derives the default and
        two of them end up disagreeing about what `None` meant."""
        if self.reference is not None:
            return self.reference
        return Modality.OPTICAL if self.moving is Modality.THERMAL else Modality.THERMAL

    @property
    def is_monomodal(self) -> bool:
        """True when both sides come from the same modality (the P1-1b control)."""
        return self.reference_modality is self.moving


class EvalConfig(ConfigBase):
    """How results are scored and reported."""

    # Corner-error thresholds in pixels for AUC and success rate. Frozen alongside the
    # metrics schema: changing them makes every prior number incomparable (X-5).
    thresholds_px: tuple[float, ...] = (3.0, 5.0, 10.0)
    # Which domain family these pairs belong to. Reported per-domain, never pooled.
    domain: Domain = Domain.DRIVING
    # Which capture platform. A wrong value here silently mislabels a per-platform row.
    platform: Platform = Platform.PUBLIC

    @field_validator("thresholds_px")
    @classmethod
    def _sorted_positive(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(t <= 0.0 for t in value):
            raise ValueError(f"thresholds_px must be non-empty and positive, got {value}")
        if list(value) != sorted(value):
            raise ValueError(f"thresholds_px must be ascending, got {value}")
        return value


class PreprocessConfig(ConfigBase):
    """The preprocessing front-end (TASKS.md P3-2, PLAN.md §4.1).

    ``reference``/``moving`` default to the recipe the three sibling implementations converged
    on (PLAN.md §15B): invert the optical grayscale, percentile-normalise the thermal.
    """

    # Recipe for the reference (unwarped) side. `invert` is the hand-crafted polarity fix the
    # learned front-end of P5-5 has to beat; turning it off is the P3-8 generality ablation.
    reference: Variant = Variant.INVERT
    # Recipe for the moving (warped) side.
    moving: Variant = Variant.PERCENTILE
    # Percentile clip bounds. PLAN.md §15D records this discrepancy rather than inheriting it:
    # production uses 0.3/99.75, the batch pipeline and display code use 2/98. Making it a
    # field means a run states which it used instead of two codebases disagreeing silently.
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    # CLAHE contrast ceiling. Too high amplifies sensor noise into false corners; too low is
    # indistinguishable from no CLAHE at all and the ablation row goes flat for the wrong reason.
    clahe_clip: float = 2.0
    # CLAHE tile grid (square). Tiles larger than the structures of interest wash out exactly
    # the local contrast the operator exists to recover.
    clahe_tile: int = 8
    # Thermal upsampling factor (PLAN.md §4.1: x1-4). Keypoints are mapped back to native
    # pixels before estimation, so this never changes the units a metric is reported in --
    # if that plumbing were dropped, every upsampled run would report errors inflated by the
    # factor and look catastrophically worse.
    moving_upsample: int = 1
    # Kernel used for that upsampling. `nearest` is included only as the ablation's floor;
    # it introduces staircase edges that gradient-based detectors happily fire on.
    moving_interpolation: Interpolation = Interpolation.BICUBIC

    @model_validator(mode="after")
    def _percentiles_ordered(self) -> Self:
        if not 0.0 <= self.percentile_low < self.percentile_high <= 100.0:
            raise ValueError(
                "require 0 <= percentile_low < percentile_high <= 100, got "
                f"{self.percentile_low} / {self.percentile_high}"
            )
        return self

    @field_validator("moving_upsample")
    @classmethod
    def _upsample_range(cls, value: int) -> int:
        if not 1 <= value <= 8:
            raise ValueError(f"moving_upsample must be in [1, 8], got {value}")
        return value

    @field_validator("clahe_tile")
    @classmethod
    def _positive_tile(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"clahe_tile must be >= 1, got {value}")
        return value

    @field_validator("clahe_clip")
    @classmethod
    def _positive_clip(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError(f"clahe_clip must be > 0, got {value}")
        return value


class MatchConfig(ConfigBase):
    """Which matchers run, and their shared budget."""

    # Registry names (`cmreg.matchers.available()`). Validated there rather than here so the
    # config layer stays importable without cv2/torch; an unknown name fails at startup with
    # the available list, not mid-sweep.
    matchers: tuple[str, ...] = ("sift",)
    # Detector budget. Uncapped, a textured 640x512 pair yields tens of thousands of SIFT
    # keypoints and the quadratic brute-force match dominates the runtime table (PLAN.md §6.5).
    max_keypoints: int = 4096
    # Lowe ratio. Raising it trades precision for recall -- more matches, lower inlier ratio,
    # and a robust estimator that has to work harder; 0.8 is Lowe's own value.
    ratio_test: float = 0.8

    @field_validator("matchers")
    @classmethod
    def _non_empty_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("matchers must name at least one matcher")
        if len(set(value)) != len(value):
            raise ValueError(f"matchers contains duplicates: {value}")
        return value

    @field_validator("ratio_test")
    @classmethod
    def _ratio_range(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"ratio_test must be in (0, 1], got {value}")
        return value

    @field_validator("max_keypoints")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 4:
            raise ValueError(f"max_keypoints must be >= 4 (a homography needs 4), got {value}")
        return value


class EstimateConfig(ConfigBase):
    """Robust estimation (TASKS.md P3-3). Swept by P3-10."""

    # Which estimator. LMEDS ignores `threshold_px` entirely (it minimises the median residual)
    # and assumes under 50% outliers -- a threshold sweep row for LMEDS is therefore flat by
    # construction, which is a property of the method and not a bug in the sweep.
    method: Estimator = Estimator.MAGSAC
    # Inlier threshold in pixels. PLAN.md §6.4: shrinking this lowers `match/reproj_err`
    # artificially, which is why that metric is reported but never led with.
    threshold_px: float = 3.0
    # Iteration cap. Too low and the estimator silently returns whatever it found first, which
    # reads as a hard matcher failure rather than an exhausted search.
    max_iters: int = 10_000
    # Target confidence. Passed by keyword: `cv2.findHomography`'s fifth positional slot is
    # `mask`, and PLAN.md §15A records the upstream harness losing this value to exactly that
    # off-by-one -- its confidence silently stayed at OpenCV's 0.995 default.
    confidence: float = 0.9999

    @field_validator("threshold_px")
    @classmethod
    def _positive_threshold(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError(f"threshold_px must be > 0, got {value}")
        return value

    @field_validator("max_iters")
    @classmethod
    def _positive_iters(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"max_iters must be >= 1, got {value}")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {value}")
        return value


class Config(ConfigBase):
    """The fully-resolved experiment."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    gt: GTConfig = Field(default_factory=GTConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    estimate: EstimateConfig = Field(default_factory=EstimateConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def load(
        cls, path: Path | str | None = None, overrides: dict[str, Any] | None = None
    ) -> Config:
        """Defaults, then the YAML file, then ``overrides`` -- each layer merged nestedly."""
        declared: dict[str, Any] = {}
        if path is not None:
            source = Path(path)
            if not source.is_file():
                raise ConfigError(f"config file not found: {source}")
            try:
                loaded = yaml.safe_load(source.read_text()) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"{source}: could not be parsed: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ConfigError(f"{source}: expected a mapping at the top level")
            declared = loaded
        if overrides:
            declared = deep_merge(declared, overrides)

        try:
            return cls.model_validate(declared)
        except ValidationError as exc:
            raise as_config_error(exc) from exc

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """A fingerprint of the *experiment*, not the invocation.

        ``runtime`` is excluded wholly, so the same experiment run on two GPUs under two
        names carries one hash -- which is what lets that hash mean anything when it is
        stamped onto a results row.
        """
        payload = json.dumps(self.model_dump(mode="json", exclude={"runtime"}), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:_HASH_LENGTH]

    def snapshot(self, run_dir: Path | str) -> Path:
        """Write the fully-resolved config (``runtime`` included) into the run directory.

        This snapshot -- not the authored YAML -- is what the aggregator later reads, so a
        result can always be traced to the exact configuration that produced it.
        """
        directory = Path(run_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "config.yaml"
        target.write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))
        return target
