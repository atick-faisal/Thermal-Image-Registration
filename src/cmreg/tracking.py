"""Optional Weights & Biases logging. A no-op unless ``runtime.wandb`` is on.

Ported from ``../Thermal-To-Optical-Translation/src/t2o/tracking.py``. Two hard-won lessons
carried over from there:

1. **Never pass ``step=`` for staged metrics.** W&B's step counter only increases, so an
   explicit ``step=`` that restarts per stage logs into the past and W&B silently drops it.
   Log the stage/epoch as a *value*.
2. **Telemetry must not take down a run.** ``log()`` disables itself and warns rather than
   raising.

``run_tags`` and ``run_name`` implement TASKS.md §0's tag vocabulary and naming convention.
Both are *derived from the resolved config* rather than authored in YAML: a campaign
overrides ``--seed`` and ``--name`` on every launch, so a tag written into a config file
would be wrong on most of them.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from cmreg.config import Config

logger = logging.getLogger(__name__)

_DIRTY_SUFFIX = "-dirty"
_UNTRACKED_SUFFIX = "-untracked"


def git_sha() -> str:
    """The current commit, suffixed with what the working tree adds to it.

    TASKS.md X-2 requires every claim to be traceable to a run and an artifact version; a
    result produced from an uncommitted tree is traceable to nothing, so the flag is part of
    the identifier rather than a separate field that can be dropped.

    **Two suffixes, not one.** ``-dirty`` meant "anything at all in `git status`", which
    included untracked files, and P3-7's stage-A runs 2 and 3 both carried it for two
    untracked artefacts the run never read (a log file and a stray calibration constant) --
    costing two rounds of asking the server what had changed, for an answer of "nothing".
    Untracked files are still reported, because an untracked ``.py`` can absolutely change a
    result, but as ``-untracked``: it says the code that ran is the commit plus something the
    puller does not have, where ``-dirty`` says the commit itself is not what ran. Both are
    still a warning; only one of them means a re-pull cannot reproduce the tracked code.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        tracked = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if tracked:
        return f"{sha}{_DIRTY_SUFFIX}"
    return f"{sha}{_UNTRACKED_SUFFIX}" if untracked else sha


def run_name(phase: str, method: str, dataset: str, variant: str, seed: int) -> str:
    """TASKS.md §0's run-name format: ``{phase}_{method}_{dataset}_{variant}_s{seed}``."""
    return f"{phase}_{method}_{dataset}_{variant}_s{seed}"


def run_tags(config: Config, matcher: str, preprocess: str, estimator: str, warp: str) -> list[str]:
    """TASKS.md §0's required tag set, derived so a tag cannot disagree with its run.

    The axes a run varies (matcher, preprocess, estimator, warp) are arguments rather than
    config fields because they are not yet modelled in the schema -- they land with the
    benchmark runner in P3-5. Everything already in the config is read from it.
    """
    return [
        f"matcher:{matcher}",
        f"preprocess:{preprocess}",
        f"dataset:{config.data.manifest.parent.name}",
        f"domain:{config.eval.domain.value}",
        f"estimator:{estimator}",
        f"warp:{warp}",
        f"seed:{config.gt.seed}",
        f"git_sha:{git_sha()}",
        f"platform:{config.eval.platform.value}",
    ]


class RunTracker:
    """Logs to wandb; silently does nothing if ``runtime.wandb`` is off."""

    def __init__(self, config: Config, tags: list[str] | None = None) -> None:
        self.enabled = config.runtime.wandb
        self._run: Any = None
        if not self.enabled:
            return

        try:
            import wandb
        except ImportError:  # pragma: no cover - wandb is a declared dependency
            logger.warning("runtime.wandb is set but wandb is not installed; skipping")
            self.enabled = False
            return

        # wandb writes into `dir`, which the caller has not necessarily created yet.
        run_dir = Path(config.runtime.path)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._run = wandb.init(
            project=config.runtime.wandb_project,
            name=config.runtime.name,
            group=config.runtime.group or None,
            tags=tags,
            config=config.to_dict() | {"config_hash": config.config_hash()},
            dir=str(run_dir),
        )
        logger.info(
            "wandb run '%s' initialised under project '%s' (group %s)",
            config.runtime.name,
            config.runtime.wandb_project,
            config.runtime.group or "none",
        )

    def log(self, metrics: dict[str, float]) -> None:
        """Log a metric dict. Never raises: telemetry must not take down a run.

        No ``step`` parameter, deliberately -- see the module docstring.
        """
        if self._run is None:
            return
        try:
            self._run.log(metrics)
        except Exception:
            logger.warning(
                "wandb logging failed; disabling it for the rest of this run", exc_info=True
            )
            self._run = None
            self.enabled = False

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                logger.warning("wandb finish failed", exc_info=True)
            self._run = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.finish()
