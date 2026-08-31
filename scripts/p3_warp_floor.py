"""P3-4a: the model floor -- what a restricted warp model can achieve before any matcher runs.

Tier-1 ground truth is a full projective homography (`gt/synthetic.py::sample_homography`), so a
6-DoF affine cannot represent its perspective term and a 4-DoF similarity cannot represent that
or its shear. Neither can reach zero error however good the matcher is, and what they are left
with is a **floor** in exactly the sense `experiments/GRID.md` §3's dataset residual `R` is one.

Stage E (TASKS.md P3-11) is unreadable without this number. "Affine scores 9 px and homography
6 px" says something about the models only once the affine floor is known; if that floor is
8.5 px the row is reporting the ground truth's own perspective content and nothing else. The
same argument P3-7's F1 had to make after the fact about `flir`'s 5.9 px -- made here *before*
the stage is paid for, because unlike `R` this floor costs nothing to measure.

It needs no matcher, no GPU and no dataset: the floor is a property of the sampled warp and the
image shape alone, so this runs on the Mac in seconds and is the one part of P3-4a that does not
wait for the server. Shapes are the four benchmark datasets' native resolutions (GRID.md §7).

    uv run python scripts/p3_warp_floor.py
    uv run python scripts/p3_warp_floor.py --pairs 1000 --perspective 0.0

`--perspective` exists to show the mechanism rather than to propose a change: at 0.0 the sampled
warp is a similarity and both floors collapse, which is what identifies the perspective jitter
as the term the restricted models cannot follow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

import numpy as np

from cmreg.config import GTConfig, WarpModel
from cmreg.gt import generator, sample_homography
from cmreg.metrics import model_floor

logger = logging.getLogger(__name__)

# Native resolutions, from `experiments/GRID.md` §7's cost table. `llvip` is the one 1280-wide
# set, and the floor scales with the image, so a single shape would misreport three of the four.
SHAPES: dict[str, tuple[int, int]] = {
    "flir": (512, 640),
    "msrs": (480, 640),
    "dronevehicle": (512, 640),
    "llvip": (1024, 1280),
}

# The ladder every benchmark row is read against (GRID.md §2), so the floor can be placed on it.
THRESHOLDS_PX = (3.0, 5.0, 10.0, 20.0)


@dataclass(frozen=True, slots=True)
class Floor:
    """One (dataset, model) cell: the distribution of the floor over sampled warps."""

    dataset: str
    model: WarpModel
    values: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def median(self) -> float:
        return float(np.median(self.values))

    @property
    def p90(self) -> float:
        return float(np.percentile(self.values, 90))

    def below(self, threshold: float) -> float:
        """Fraction of warps whose floor leaves the threshold reachable at all.

        This is the column that matters: a pair whose floor exceeds 10 px cannot be scored a
        success at 10 px by *any* fit of that model, so `reg/success_rate_10px` for that row is
        bounded above by this number before a matcher is chosen.
        """
        return float((self.values < threshold).mean())


def measure(dataset: str, shape: tuple[int, int], config: GTConfig, pairs: int) -> list[Floor]:
    """Sample `pairs` ground-truth warps for one dataset and floor each model against them.

    The warps come from `sample_homography` under the real `GTConfig`, keyed on `(seed, index)`
    exactly as the runner keys them -- so these are the warps a bench run on this dataset would
    actually score against, not a re-derivation that could drift from them.
    """
    warps = [sample_homography(config, generator(config.seed, i), shape) for i in range(pairs)]
    return [
        Floor(dataset, model, np.array([model_floor(h, shape, model) for h in warps]))
        for model in WarpModel
    ]


def render(floors: list[Floor]) -> str:
    """The copy-pasteable block. Server runs reach the Mac as console text and nothing else."""
    lines = [
        "########## P3-4a: model floor, mean corner error in px ##########",
        "",
        "The smallest error a fit of this model can have against Tier-1's projective truth.",
        "A strict lower bound: fitted to the four corners, which minimises the very quantity",
        "`reg/mace` reports, so no matcher on any correspondence set can score below it.",
        "",
        f"{'dataset':<14}{'model':<13}{'mean':>8}{'median':>9}{'p90':>9}"
        + "".join(f"{f'<{t:g}px':>10}" for t in THRESHOLDS_PX),
    ]
    lines.append("-" * len(lines[-1]))
    for floor in floors:
        lines.append(
            f"{floor.dataset:<14}{floor.model.value:<13}"
            f"{floor.mean:>8.2f}{floor.median:>9.2f}{floor.p90:>9.2f}"
            + "".join(f"{floor.below(t):>10.3f}" for t in THRESHOLDS_PX)
        )
    return "\n".join(lines)


def main_script(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=300, help="warps per cell (stage budget)")
    parser.add_argument("--seed", type=int, default=0, help="`gt.seed`, as the runner uses it")
    parser.add_argument(
        "--perspective",
        type=float,
        default=None,
        help="override GTConfig.perspective; 0.0 collapses the truth to a similarity",
    )
    args = parser.parse_args(argv)

    overrides = {"seed": args.seed}
    if args.perspective is not None:
        overrides["perspective"] = args.perspective
    config = GTConfig(**overrides)
    logger.info(
        "flooring %d warps per dataset at rotation +/-%.0f deg, scale %.2f-%.2f, "
        "perspective %.3f, translation %.3f",
        args.pairs,
        config.rotation_deg,
        config.scale_min,
        config.scale_max,
        config.perspective,
        config.translation,
    )

    floors = [
        floor
        for dataset, shape in SHAPES.items()
        for floor in measure(dataset, shape, config, args.pairs)
    ]
    print(render(floors))
    return 0


if __name__ == "__main__":
    sys.exit(main_script())
