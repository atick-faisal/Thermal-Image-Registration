"""The evaluation cell (TASKS.md P3-5): one config, one Parquet file, one console block.

AGENTS.md's non-negotiable is that there is **one evaluation path** -- every method computes
every metric through this code, because that is what makes the comparison table defensible.
This module is that path.

Protocol
--------
Tier-1 ground truth assumes the two modalities of a pair *start* aligned, so that the true
cross-modal homography between them is the identity, and manufactures a known misalignment by
breaking it with a sampled warp (PLAN.md §5):

1. Sample ``H_gt`` for pair *i* from ``(seed, i)`` -- the same derivation ``cmreg gt`` uses,
   so a bench run and its GT file agree by construction rather than by coincidence.
2. Warp the moving modality by ``H_gt``. The other modality is the reference frame, and every
   metric below is in *its* pixels.
3. The matcher's job is to undo that: recover the map from the warped moving image back to the
   reference. The truth for that direction is therefore ``inv(H_gt)``, and getting this inverse
   backwards is the single most likely error in the whole pipeline -- which is why
   ``tests/test_runner.py`` pins it end to end with a same-modality pair rather than only
   unit-testing the pieces.

**That assumption is false on the public benchmarks, by enough to matter** (TASKS.md P1-1a).
MSRS, FLIR-aligned and LLVIP each carry a 4-6 px residual `R` of their own and DroneVehicle
~59 px, so the true correspondence is `R . inv(H_gt)` rather than `inv(H_gt)`. Left alone, `R`
is a systematic floor under every number this module produces, and P3-7's stage-A run 1
measured the cost: ten of twenty matchers scored `success_rate_5px` of exactly 0.0000 on
`flir`, twelve of them piled into a 4.5-5.8 px band that is the dataset's residual rather than
anything about them, and no cross-dataset row was comparable because the floors differ (5.9 vs
4.7 px).

So step 3 above composes a **dataset-level** calibration constant when `gt.residual_calibration`
names one (TASKS.md P2-12, `gt/calibration.py`): the moving image is still warped by `H_gt`
alone, and only the *truth* changes. Which datasets may compose is `experiments/GRID.md` §3's
decision, not this module's, and a per-pair `R_i` is never admissible -- it comes from a
matcher, so folding it in would score that matcher against its own output (P1-1b).

`R` itself is measured by running this same cell with a zero-magnitude warp
(`experiments/p1_alignment_audit.yaml`), which is why `h` is persisted per row:
`analysis/residual.py` decomposes those residuals and `cmreg calibrate` publishes the constant.

The two modalities are also chosen independently (`gt.reference` / `gt.moving`). Setting them
to the *same* modality is the P1-1b mono-modal control: the pair's own residual is gone by
construction, and what the cell then measures is its own floor rather than the data's.

Loop order is pairs-outer, matchers-inner: decoding, warping and preprocessing depend only on
the config, so doing them once per pair rather than once per (pair, matcher) is the difference
between a benchmark that runs overnight and one that does not.

Failures are rows. A pair the matcher cannot solve is recorded with ``success=False`` and a
reason, never dropped (TASKS.md X-4) -- a benchmark that silently discards its hard cases
reports the score of an easier dataset than the one it claims.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmreg.config import Config, Modality
from cmreg.data import DatasetManifest, select_pairs
from cmreg.device import resolve_device
from cmreg.estimate import Estimate, EstimateError, estimate_homography
from cmreg.gt import (
    DenseGT,
    ResidualCalibration,
    dense_displacement,
    generator,
    load_calibration,
    overlap_ratio,
    sample_homography,
)
from cmreg.imaging import ImagingError, read_gray
from cmreg.matchers import MatchResult, get_matcher
from cmreg.metrics import corner_error, diagonal, endpoint_error
from cmreg.preprocess import GrayImage, preprocess_moving, preprocess_reference
from cmreg.results import PairRow, Summary, render, render_comparison, summarize, write_rows
from cmreg.seeding import seed_cell
from cmreg.tracking import RunTracker, git_sha, run_name, run_tags
from cmreg.warp import WarpError, apply_warp

logger = logging.getLogger(__name__)

# The only warp model implemented; a column rather than a constant so P3-4's affine, similarity,
# TPS and residual-flow entries slot in without a schema migration.
WARP_MODEL = "homography"

# Metrics shown in the multi-matcher comparison table. The headline three plus one threshold:
# enough to rank methods at a glance, short enough to stay one line per matcher.
#
# That threshold is **5 px, not the tightest one**. TASKS.md P1-1d measured that no dataset in
# the benchmark supports a 3 px Tier-1 threshold on its native alignment -- the typical FLIR
# pair retains 4-5 px after every reproducible rig error is removed -- so a 3 px column reads
# 0.0000 for every matcher and ranks nothing. `experiments/GRID.md` §2 reports the full
# 3/5/10/20 ladder; this is the one column that has to earn its width.
COMPARISON_KEYS = (
    "reg/mace",
    "reg/epe_mean",
    "reg/auc_5px",
    "reg/success_rate_5px",
    "reg/failure_rate",
    "match/inliers",
    "time/total_ms",
)


class RunnerError(RuntimeError):
    """Raised for a run that cannot start -- an empty split, an unknown matcher."""


@dataclass(frozen=True, slots=True)
class _Pair:
    """One decoded pair, warped and preprocessed once for all matchers."""

    stem: str
    shape: tuple[int, int]
    reference: GrayImage
    moving_warped: GrayImage
    # `H_gt` maps the moving image into the reference canvas; the matcher has to recover its
    # inverse, composed with the dataset's residual `R` when one is configured. Both are
    # carried so nothing downstream has to remember which is which.
    truth: np.ndarray
    gt_field: DenseGT
    overlap: float


def run_benchmark(config: Config) -> tuple[Summary, ...]:
    """Run every configured matcher over a split and return one summary each."""
    manifest = DatasetManifest.load(config.data.manifest)
    images = manifest.images(config.data.split)
    if not images:
        raise RunnerError(f"no images in the '{config.data.split}' split of {manifest.path}")
    # `(split index, path)`, because the index keys the pair's synthetic warp and must not
    # depend on how the split was sampled (`data/splits.py::select_pairs`).
    selected = select_pairs(images, config.data.limit, config.data.subsample_seed)
    manifest.pairing.validate_pairs([path for _, path in selected])

    # Resolved once, and logged: PLAN.md §15B records RoMa on Windows leaving DINOv2 on the CPU
    # even with CUDA available, so a run that quietly landed on the wrong device has to be
    # visible in its own log rather than only in a runtime column six hours later.
    device = resolve_device(config.runtime.device)
    matchers = {name: get_matcher(name, config.match, device) for name in config.match.matchers}
    # Read once and validated here rather than per pair: a calibration naming the wrong dataset
    # is a configuration error, identical on all 300 pairs, and should abort before the first
    # matcher loads instead of writing 6000 rows that blame the data (the same reasoning that
    # keeps `EstimateError` fatal below).
    calibration = _load_calibration(config, manifest.path.parent.name)
    logger.info(
        "benchmarking %s over %d pairs of %s [%s] on %s",
        ", ".join(matchers),
        len(selected),
        manifest.path.parent.name,
        config.data.split,
        device,
    )
    if config.gt.is_monomodal:
        # Said out loud, because the numbers a control produces look like an implausibly good
        # benchmark row and the only thing distinguishing them is this configuration.
        logger.warning(
            "MONO-MODAL CONTROL: both sides are %s; this is a pipeline floor (TASKS.md P1-1b), "
            "not a cross-modal benchmark result",
            config.gt.moving.value,
        )

    rows: list[PairRow] = []
    identity = _identity_columns(config, manifest, calibration)
    for position, (index, optical_path) in enumerate(selected):
        pair = _load_pair(optical_path, manifest, config, index, calibration)
        if pair is None:
            rows.extend(
                _failed_row(optical_path.stem, name, identity, "unreadable_pair")
                for name in matchers
            )
            continue
        for name, matcher in matchers.items():
            try:
                rows.append(_evaluate(pair, index, name, matcher, config, identity))
            except EstimateError:
                # The one class of error that is about the *config*, not the pair -- PROSAC
                # without confidences fails identically on all 300 pairs, so aborting says so
                # once instead of writing 6000 rows that all blame the data.
                raise
            except Exception:
                # A matcher raising is a property of that (pair, matcher), and the run's rows
                # are only written after the last pair: letting it escape discards every pair
                # already scored. X-4 says the hard cases are rows, and this is the hardest.
                logger.exception(
                    "%s on %s raised; recording the cell as a failure", name, pair.stem
                )
                rows.append(_failed_row(pair.stem, name, identity, "matcher_raised", pair.shape))
        if (position + 1) % 50 == 0:
            logger.info("  %d / %d pairs", position + 1, len(selected))

    run_dir = Path(config.runtime.path)
    config.snapshot(run_dir)
    write_rows(rows, run_dir)

    summaries = tuple(
        summarize([row for row in rows if row.matcher == name], config.eval.thresholds_px)
        for name in matchers
    )
    _publish(config, summaries)
    return summaries


def _load_calibration(config: Config, dataset: str) -> ResidualCalibration | None:
    """Read the configured residual constant and check it describes this run.

    The shape half of the check waits until the first pair is decoded (`_load_pair`); the
    dataset half is answerable now, and answering it now is what turns "silently registered
    against the wrong rig for six hours" into a startup error.
    """
    if config.gt.residual_calibration is None:
        return None
    if config.gt.is_monomodal:
        # Both sides are read from one modality, so the pair's cross-modal offset is gone by
        # construction (P1-1b) and there is nothing left for `R` to remove. Composing one here
        # would *inject* the rig's misalignment into a control whose entire purpose is to
        # measure the pipeline's own floor.
        raise RunnerError(
            "gt.residual_calibration is set on a mono-modal control; the pair's residual is "
            "zero by construction there and composing a constant would manufacture one"
        )
    calibration = load_calibration(config.gt.residual_calibration)
    if calibration.dataset != dataset:
        raise RunnerError(
            f"calibration {config.gt.residual_calibration} is for '{calibration.dataset}' but "
            f"this run is on '{dataset}'"
        )
    logger.info(
        "composing %s's residual into the ground truth: %.2f px over %d matchers (%s), "
        "stated uncertainty %.2f px mean / %.2f px worst case [%s]",
        calibration.dataset,
        calibration.magnitude_px(),
        len(calibration.matchers),
        ", ".join(calibration.matchers),
        calibration.spread_px,
        calibration.worst_case_px,
        calibration.digest(),
    )
    return calibration


def _identity_columns(
    config: Config, manifest: DatasetManifest, calibration: ResidualCalibration | None
) -> dict[str, object]:
    """The columns constant across every row of this run."""
    return {
        "dataset": manifest.path.parent.name,
        "split": config.data.split,
        "domain": config.eval.domain.value,
        "platform": config.eval.platform.value,
        "preprocess_ref": config.preprocess.reference.value,
        "preprocess_mov": config.preprocess.moving.value,
        "upsample": config.preprocess.moving_upsample,
        "interpolation": config.preprocess.moving_interpolation.value,
        "estimator": config.estimate.method.value,
        "threshold_px": config.estimate.threshold_px,
        "warp": WARP_MODEL,
        "moving": config.gt.moving.value,
        "reference": config.gt.reference_modality.value,
        "seed": config.gt.seed,
        "config_hash": config.config_hash(),
        # The constant's digest, not its path: `config_hash` already covers the configured
        # path, and two different constants can share a filename across two machines. This is
        # what makes a pasted table traceable to the exact matrix that produced it.
        "residual_calibration": None if calibration is None else calibration.digest(),
        "git_sha": git_sha(),
        "run_name": config.runtime.name,
    }


def _load_pair(
    optical_path: Path,
    manifest: DatasetManifest,
    config: Config,
    index: int,
    calibration: ResidualCalibration | None,
) -> _Pair | None:
    """Decode, warp and preprocess one pair. ``None`` when the pair cannot be used at all."""
    thermal_path = manifest.pairing.thermal_path(optical_path)
    try:
        optical = read_gray(optical_path)
        thermal = read_gray(thermal_path)
    except ImagingError:
        logger.warning("could not read pair %s; recording it as a failure", optical_path.stem)
        return None

    if optical.shape != thermal.shape:
        # A "pre-registered" dataset whose modalities differ in size is not pre-registered.
        # TASKS.md P1-3 says to spot-check rather than trust the README; this is that check,
        # enforced per pair so one bad file does not invalidate a six-hour run.
        logger.warning(
            "%s: optical %s and thermal %s differ in shape; pair is not pre-registered",
            optical_path.stem,
            optical.shape,
            thermal.shape,
        )
        return None

    # A lookup rather than a two-way conditional, because reference and moving are chosen
    # independently: setting them to the same modality is the P1-1b mono-modal control.
    by_modality = {Modality.OPTICAL: optical, Modality.THERMAL: thermal}
    reference = by_modality[config.gt.reference_modality]
    moving = by_modality[config.gt.moving]
    shape = reference.shape

    homography = sample_homography(config.gt, generator(config.gt.seed, index), shape)
    try:
        moving_warped = np.asarray(apply_warp(moving, homography, out_shape=shape), np.uint8)
        truth = np.linalg.inv(homography)
        if calibration is not None:
            calibration.validate_for(manifest.path.parent.name, shape)
            # `R . inv(H_gt)`, and the order is the whole of it. `R` maps moving-native pixels
            # to reference pixels (that is what the identity-warp audit estimates, since there
            # `H_gt = I` and the estimator runs src=moving, dst=reference). A warped-moving
            # pixel `x'` is `inv(H_gt) x'` in moving-native, which lands at `R inv(H_gt) x'` in
            # the reference. Writing `inv(H_gt . R)` instead would compose `R` in the opposite
            # direction and *double* the misalignment rather than removing it -- which is why
            # this is pinned by an oracle rather than by this comment
            # (`tests/test_runner.py::test_composing_a_known_calibration_removes_the_offset`).
            truth = calibration.homography() @ truth
    except (WarpError, np.linalg.LinAlgError):  # pragma: no cover - sample_homography validates
        logger.warning("%s: sampled homography is not usable", optical_path.stem)
        return None

    return _Pair(
        stem=optical_path.stem,
        shape=shape,
        reference=preprocess_reference(reference, config.preprocess).image,
        moving_warped=moving_warped,
        truth=truth,
        gt_field=dense_displacement(truth, shape),
        # On `H_gt`, deliberately, and not on `inv(truth)`: overlap is "how much of the moving
        # image the synthetic warp pushed off the canvas", a property of the warp this run
        # applied. `R` is a handful of pixels of rig offset and folding it in here would make
        # a composed run's overlap column incomparable with an uncomposed one for no gain.
        overlap=overlap_ratio(dense_displacement(homography, shape)),
    )


def _evaluate(
    pair: _Pair, index: int, name: str, matcher, config: Config, identity: dict[str, object]
) -> PairRow:
    """Match, estimate and score one (pair, matcher) cell."""
    # Before the matcher, not before the run: see `seeding.py::seed_cell`. Dense matchers
    # sample their correspondences stochastically, and an ambient RNG makes a cell's result
    # depend on which other matchers happened to share its config.
    seed_cell(config.gt.seed, index, name)
    moving = preprocess_moving(pair.moving_warped, config.preprocess)
    result: MatchResult = matcher(pair.reference, moving.image)

    # Back to native pixels before estimation, so every metric is in reference-image units
    # whatever the upsampling factor was (preprocess/variants.py explains the half-pixel term).
    kpts_reference = result.kpts0
    kpts_moving = moving.to_native(result.kpts1)

    start = time.perf_counter()
    # src is the warped moving image, dst is the reference: the estimate must map the moving
    # image back into the reference frame, which is the direction `truth` describes.
    estimate = estimate_homography(kpts_moving, kpts_reference, config.estimate, result.confidence)
    estimate_ms = (time.perf_counter() - start) * 1e3

    return _row(pair, name, identity, result, estimate, estimate_ms)


def _row(
    pair: _Pair,
    matcher: str,
    identity: dict[str, object],
    result: MatchResult,
    estimate: Estimate,
    estimate_ms: float,
) -> PairRow:
    corner_err: float | None = None
    epe_mean: float | None = None
    epe_median: float | None = None
    failure_reason = estimate.failure_reason

    if estimate.h is not None:
        corner_err = corner_error(estimate.h, pair.truth, pair.shape, saturate=True)
        predicted = dense_displacement(estimate.h, pair.shape)
        # Scored on the GT's valid mask, not the prediction's: the pixels with ground truth
        # are a property of the pair, and letting a method choose its own scored region would
        # reward one that maps most of the image off-canvas.
        error = endpoint_error(
            predicted.flow,
            pair.gt_field.flow,
            pair.gt_field.valid,
            saturate_at=diagonal(pair.shape),
        )
        epe_mean, epe_median = error.mean, error.median

    return PairRow(
        stem=pair.stem,
        height=pair.shape[0],
        width=pair.shape[1],
        matcher=matcher,
        success=estimate.h is not None,
        failure_reason=failure_reason,
        overlap=pair.overlap,
        corner_err=corner_err,
        epe_mean=epe_mean,
        epe_median=epe_median,
        # Row-major, matching `cmreg gt`'s JSON convention. `None` on failure keeps the
        # nullable columns null exactly when `success` is False (results/store.py).
        h=None if estimate.h is None else [float(v) for v in estimate.h.ravel()],
        n_detected_ref=result.n_detected0,
        n_detected_mov=result.n_detected1,
        n_matches=len(result),
        n_inliers=estimate.n_inliers,
        inlier_ratio=estimate.inlier_ratio,
        reproj_err=None if estimate.h is None else estimate.reproj_err,
        extract_ms=result.extract_ms,
        match_ms=result.match_ms,
        estimate_ms=estimate_ms,
        # The method's cost, excluding decode and warp: those are dataset properties shared by
        # every matcher, and folding them in would flatten the runtime table (PLAN.md §6.5).
        total_ms=result.extract_ms + result.match_ms + estimate_ms,
        **identity,  # type: ignore[arg-type]
    )


def _failed_row(
    stem: str,
    matcher: str,
    identity: dict[str, object],
    reason: str,
    shape: tuple[int, int] | None = None,
) -> PairRow:
    """A row for a cell that produced no estimate, because the pair or the matcher failed."""
    return PairRow(
        stem=stem,
        # Null when the pair could not be decoded, so there is no shape to report.
        height=None if shape is None else shape[0],
        width=None if shape is None else shape[1],
        matcher=matcher,
        success=False,
        failure_reason=reason,
        overlap=float("nan"),
        corner_err=None,
        epe_mean=None,
        epe_median=None,
        h=None,
        n_detected_ref=None,
        n_detected_mov=None,
        n_matches=0,
        n_inliers=0,
        inlier_ratio=0.0,
        reproj_err=None,
        extract_ms=0.0,
        match_ms=0.0,
        estimate_ms=0.0,
        total_ms=0.0,
        **identity,  # type: ignore[arg-type]
    )


def _publish(config: Config, summaries: Sequence[Summary]) -> None:
    """Send each summary to W&B and print its console block.

    One W&B run per matcher, named to TASKS.md §0's frozen format. ``runtime.name`` names the
    local run directory; the W&B name is *derived*, because a campaign that overrides the
    matcher on the command line would otherwise file every cell under one name.
    """
    for summary in summaries:
        matcher = summary.context["matcher"]
        variant = (
            f"{config.preprocess.reference.value}-{config.preprocess.moving.value}"
            f"-x{config.preprocess.moving_upsample}-{config.estimate.method.value}"
        )
        cell = config.model_copy(
            update={
                "runtime": config.runtime.model_copy(
                    update={
                        "name": run_name(
                            "p3",
                            matcher,
                            config.data.manifest.parent.name,
                            variant,
                            config.gt.seed,
                        )
                    }
                )
            }
        )
        tags = run_tags(
            cell,
            matcher=matcher,
            preprocess=variant,
            estimator=config.estimate.method.value,
            warp=WARP_MODEL,
        )
        with RunTracker(cell, tags) as tracker:
            tracker.log(summary.metrics)
        print(render(summary))

    if len(summaries) > 1:
        print(render_comparison(summaries, COMPARISON_KEYS))
