"""The package imports without torch-heavy work and exposes its version."""

from __future__ import annotations

import cmreg


def test_version_is_exposed() -> None:
    assert cmreg.__version__.count(".") == 2
