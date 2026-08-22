"""Shared config machinery: the strict base model, the error type, and the YAML merge.

Ported near-verbatim from ``../Thermal-To-Optical-Translation/src/t2o/config/base.py``,
which records the reasoning: pydantic v2 replaces a hand-rolled YAML->dataclass loader,
keeping its two best behaviours -- unknown-key rejection and ``snapshot()``.

Unknown keys are rejected rather than ignored. A silently-dropped typo in a benchmark config
costs a full sweep to discover, so the trade is heavily in favour of failing fast.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class ConfigError(ValueError):
    """Raised for an unknown key, a malformed value, or an impossible combination."""


class ConfigBase(BaseModel):
    """Base for every config section.

    ``extra="forbid"`` is the load-bearing setting; ``frozen=True`` makes sections value
    objects, so nothing downstream can mutate the config a run was hashed and snapshotted
    from. ``validate_default=True`` means a default that violates its own validator fails
    at import time rather than only when someone happens to override it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` over ``base``.

    Nested rather than top-level so a YAML file that sets one key in a section does not
    wipe that section's other defaults.
    """
    merged = dict(base)
    for key, value in update.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def as_config_error(exc: ValidationError) -> ConfigError:
    """Render a pydantic ``ValidationError`` as a ``ConfigError`` with dotted locations.

    Every error is reported, not just the first, so a config with three typos fails once
    instead of three runs in a row.
    """
    lines = [f"{_location(error['loc'])}: {error['msg']}" for error in exc.errors()]
    plural = "s" if len(lines) > 1 else ""
    return ConfigError(f"invalid config ({len(lines)} error{plural}):\n  " + "\n  ".join(lines))


def _location(loc: tuple[int | str, ...]) -> str:
    """Format a pydantic error location as a dotted config path."""
    return ".".join(str(part) for part in loc) or "<root>"
