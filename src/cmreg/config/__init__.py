"""Experiment configuration: strict pydantic sections loaded from tracked YAML."""

from __future__ import annotations

from cmreg.config.base import ConfigBase, ConfigError, as_config_error, deep_merge
from cmreg.config.schema import (
    Config,
    DataConfig,
    Domain,
    EstimateConfig,
    Estimator,
    EvalConfig,
    GTConfig,
    Interpolation,
    MatchConfig,
    MatchSelection,
    Modality,
    Platform,
    PreprocessConfig,
    RuntimeConfig,
    Variant,
    WarpModel,
)

__all__ = [
    "Config",
    "ConfigBase",
    "ConfigError",
    "DataConfig",
    "Domain",
    "EstimateConfig",
    "Estimator",
    "EvalConfig",
    "GTConfig",
    "Interpolation",
    "MatchConfig",
    "MatchSelection",
    "Modality",
    "Platform",
    "PreprocessConfig",
    "RuntimeConfig",
    "Variant",
    "WarpModel",
    "as_config_error",
    "deep_merge",
]
