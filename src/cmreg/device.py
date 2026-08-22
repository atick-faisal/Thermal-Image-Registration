"""Turning ``runtime.device`` into a concrete torch device string.

Device is an *invocational* property, not a scientific one: the same experiment must carry the
same ``config_hash()`` on the Mac and on the A100 box (``config/schema.py``'s ``RuntimeConfig``
is excluded from the hash wholesale). That is why it is resolved here and passed to
``get_matcher`` as an argument, rather than being a field of the hashed ``MatchConfig``.

Mirrors ``vismatch/utils.py:get_default_device`` rather than importing it, so the core package
resolves a device on a machine that never installed the ``matchers`` extra.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


class DeviceError(RuntimeError):
    """Raised when a configured device cannot be used on this machine."""


def resolve_device(spec: str) -> str:
    """``"auto"`` -> the best available device; anything else validated and passed through.

    An explicit ``"cuda"`` on a machine without CUDA **raises** rather than falling back. A
    silent fallback is how a benchmark cell budgeted for the A100s runs overnight on CPU and
    reports a runtime table nobody can interpret (PLAN.md §6.5).
    """
    import torch  # heavy; imported here so `cmreg --version` stays fast (AGENTS.md)

    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    try:
        torch.device(spec)
    except (RuntimeError, ValueError) as exc:
        raise DeviceError(f"{spec!r} is not a torch device string: {exc}") from exc

    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise DeviceError(f"config asks for {spec!r} but torch reports no CUDA on this machine")
    if spec.startswith("mps") and not torch.backends.mps.is_available():
        raise DeviceError(f"config asks for {spec!r} but torch reports no MPS on this machine")
    return spec
