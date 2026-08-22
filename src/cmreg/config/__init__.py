"""Experiment configuration: strict pydantic sections loaded from tracked YAML."""

from __future__ import annotations

from cmreg.config.base import ConfigBase, ConfigError, as_config_error, deep_merge
from cmreg.config.schema import (
    Config,
    DataConfig,
    Domain,
    EvalConfig,
    GTConfig,
    Platform,
    RuntimeConfig,
)

__all__ = [
    "Config",
    "ConfigBase",
    "ConfigError",
    "DataConfig",
    "Domain",
    "EvalConfig",
    "GTConfig",
    "Platform",
    "RuntimeConfig",
    "as_config_error",
    "deep_merge",
]
