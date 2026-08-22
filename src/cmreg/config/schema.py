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


class RuntimeConfig(ConfigBase):
    """Where and how a run executes. **Excluded wholesale from `config_hash()`** -- the same
    experiment on two machines under two names must carry one hash, or the hash means
    nothing."""

    # "auto" resolves at startup. A hardcoded "cuda" fails the Mac dev box; a hardcoded
    # "cpu" silently wastes the A100s.
    device: str = "auto"
    # The run's identity in W&B and on disk. Two runs sharing a name overwrite each other's
    # outputs and become indistinguishable in the results store.
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


class Config(ConfigBase):
    """The fully-resolved experiment."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    gt: GTConfig = Field(default_factory=GTConfig)
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
