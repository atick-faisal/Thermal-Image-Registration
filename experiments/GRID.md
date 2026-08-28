# CM-Reg-Bench — the experiment grid (TASKS.md P3-1)

P3-1 says it plainly: **do not run the naive cross-product.** 20 matchers x 7 preprocess x
5 estimators x 5 warps x 4 datasets x 5 seeds is ~10^5 cells, most of which answer nothing.
This file is the staged design that replaces it, and it is written *before* anything launches
so that a stage cannot quietly redefine its own defaults.

Every stage is "the anchor cell, varying one axis". Reading a row therefore needs only this
file and the axis named in it.

---

## 1. The anchor cell

| axis | value | why this one |
|---|---|---|
| split | `val` | X-6: train is reserved. `flir` train is used once, in P1-1c, as a disjoint check on a constant — never as a benchmark row |
| pairs | **300, `subsample_seed: 0`** | see §4 |
| moving / reference | thermal / optical | the production direction (PLAN.md §15B); thermal is the harder, lower-resolution side and warping optical instead flatters every number |
| Tier-1 warp | ±30°, scale 0.8–1.25 log-uniform, perspective 0.05, translation 0.05 | PLAN.md §5's full range. P3-7 once suspected this range of causing `auc_3px = 0`; P1-1a settled that it was not (§3) |
| preprocess | `invert` / `percentile` (2/98), ×1 bicubic | the recipe all three sibling implementations converged on (PLAN.md §15B) |
| estimator | MAGSAC @ 3 px, 10 000 iters, conf 0.9999 | P3-3's default; the 3 px here is the *estimator inlier* threshold and is unrelated to §2's reporting ladder |
| warp model | homography | the only one implemented (P3-4 adds the rest, and is stage E's precondition) |
| `max_keypoints` | 4096 | not honoured by the detector-free backends — `minima-roma` returns 10 000 under any budget (`TODO(P3-12)`) |
| seed | 0 | one seed outside stage D; see §5 |

`experiments/p3a_baseline_grid.yaml` **is** this cell, in the config schema, and every later
stage is authored as a diff against it.

## 2. The reporting ladder — 3 / 5 / 10 / 20 px, read from 5 up

`EvalConfig.thresholds_px` moves from `(3, 5, 10)` to `(3, 5, 10, 20)`.

- **Additive, so X-5 stays clean.** `metrics/schema.py::auc_key` derives a key per threshold,
  so 20 px costs one dict entry and no prior run needs migrating. Removing 3 px would have
  cost exactly that, which is why it stays even though nothing can reach it.
- **The headline is quoted at 5 px and above.** P1-1d: after every reproducible rig error is
  removed, the typical FLIR-aligned pair still departs from its own consensus by 4.09 / 4.85 /
  5.48 px under three matchers spanning three decades of cost — against a 0.2 px pipeline
  floor (P1-1b sweep B). **No dataset in the benchmark supports a 3 px Tier-1 threshold on its
  native alignment.** A 3 px column is therefore reporting the dataset's residual
  misalignment, not the matcher, and is kept only for continuity with P1-1a/b/c.
- 20 px exists because `dronevehicle`'s useful signal lives above 10 px (28% of its pairs are
  gross failures at *any* threshold, and the rest are ~4.7 px median).

## 3. The `R` policy, per dataset

Carried from P1-1d unchanged so that no stage re-litigates it. `R` is a dataset's own
residual cross-modal misalignment; Tier-1 assumes it is zero, so it is a systematic floor
under every Tier-1 number.

| dataset | policy | basis |
|---|---|---|
| `msrs` | **do not compose.** Report Tier-1 error as relative to the dataset's own alignment, thresholds above the floor | 13% systematic; composition would fit noise |
| `flir` | **compose the three-matcher median corner field**, published with its ±1.23 px (2.33 px worst-case) across-matcher spread as a stated uncertainty — *and* report the floor anyway | reproduces across disjoint splits to 0.14%; removes ~9.5 px, injects ~1.2 px |
| `llvip` | **do not compose** (revised 2026-08-27, P2-12). Report the floor | measured at 300 random pairs × 3 matchers: the constant is 2.21 px and the across-matcher worst case 2.87 px — *larger than the constant*. 0.3–4.7% systematic, below `msrs`'s 13% |
| `dronevehicle` | **do not compose, and treat Tier-1 as invalid on the 28% of pairs that fail catastrophically** whatever the threshold. The aerial domain needs Tier-2 (P2-10) or Tier-3 (P2-8) | 1% systematic; 4.73 px median under a 77.97 px mean |

**Never compose a per-pair `R_i`** (P1-1b): it comes from a matcher, so folding it into the GT
scores that matcher against its own output. Only a dataset-level constant is admissible, and
only where the scatter it leaves is small.

### Composition, as implemented (P2-12)

`gt.residual_calibration` names a constant under `calibration/`; the truth becomes
`R . inv(H_gt)` while the moving image is still warped by `H_gt` alone, so only the *ground
truth* changes and no pixel is resampled twice. The direction is pinned end to end by
`tests/test_runner.py::test_composing_a_known_calibration_removes_the_offset` against a fixture
whose rig displacement is known to the pixel — composed the other way round the error doubles
rather than vanishing, and that is not a difference an eye catches in a table.

The constant is produced by `cmreg calibrate` over identity-warp runs (one per matcher), which
takes the element-wise median of their consensus corner fields — P1-1d's manual construction,
now the command. It reads the stored `h` column and never a matcher, so a third leg costs
seconds. `scripts/p3b_calibrate.py` is the server driver.

| | status |
|---|---|
| `calibration/flir.json` | **published and independently checked.** P1-1d's 1,013-pair three-matcher median; a disjoint 300-pair random sample reproduces it to **0.382 px** mean corner distance, well inside its stated ±1.23 px |
| `calibration/rejected/llvip.json` | **measured and rejected.** Kept as the evidence, deliberately not at the path the runner reads |

**The test a constant has to pass, stated once so no future dataset re-argues it:** the
across-matcher spread must be small *against the constant's own magnitude*, not merely small in
absolute terms. Both constants have a ~1.2 px spread; on `flir` that is 13% of a 9.46 px
constant and on `llvip` 57% of a 2.21 px one, with a worst case of 130%. Composing `llvip`
would remove 2.2 px and inject up to 2.9 px.

`scripts/p3a_grid.py` carries the policy per cell and **fails** a composing cell whose constant
is absent rather than running it uncomposed: the two produce tables that look identical, and
P3-7's F1 is the record of what an unnoticed floor costs.

Measured on `flir` val, 50 random pairs, `eloftr`, identity warp — composition removes what it
claims to and no more:

| | uncomposed | composed |
|---|---|---|
| `reg/mace` | 13.10 | **8.62** |
| `reg/epe_median` | 5.48 | **2.25** |
| `reg/success_rate_3px` | 0.00 | **0.30** |

The 3 px column moving off zero for the first time on this dataset is the point: what remains is
the ~4-5 px of per-pair scatter P1-1d showed no calibration can remove.

## 4. Pair budget — 300 random, not a head slice

P1-1c measured what `images[:N]` costs: 50 consecutive frames of a driving set are one scene,
so the systematic term came back right (x0.98 on `flir` val) and the random term 3.6x too low.
Stage A would have inherited that on every cell.

`DataConfig.subsample_seed` draws `limit` pairs uniformly instead, **carrying each pair's split
index** so its synthetic warp is the one the full-split run would have given it
(`data/splits.py::select_pairs`). 300 of `msrs`'s 361 val pairs is nearly the whole split; of
`llvip`'s 3,463 it is 8.7%. Uniform across datasets is deliberate — an unequal budget makes a
per-dataset row's confidence interval depend on the dataset for a reason no reader can see.

## 5. Seeds

One seed everywhere except stage D. X-3 asks for >=5 seeds with CIs on anything stochastic;
the cost of honouring that on every stage is 5x the whole grid. What is actually stochastic
here is **RANSAC's sampling and the dense matchers' correspondence sampling**, and `seed_cell`
already pins both per `(seed, pair, matcher)` (P0-9). Stage D varies the estimator, so it is
where seed-to-seed variance is the measurement rather than a nuisance, and it carries 5 seeds.
Any *claim* of the form "A beats B" gets its 5 seeds before it enters the paper (P8-2's
Wilcoxon needs them); a scoping row does not.

## 6. The stages

`reduced-8` = `roma`, `minima-roma`, `matchanything-roma`, `eloftr`, `xoftr`,
`superpoint-lightglue`, `xfeat`, `sift` — the two best cross-modal families, the two
semi-dense entries, the strongest sparse learned arm, the runtime outlier, and the classical
floor. Named here so no stage picks its own.

`driving+aerial` = `flir` + `dronevehicle`: the best-characterised driving set (P1-1c/d) and
the only aerial one.

| stage | axis varied | matchers | datasets | cells | seeds | task |
|---|---|---|---|---|---|---|
| **A** | — (the anchor itself) | all 20 | all 4 | 4 | 1 | P3-7 |
| B | `invert` on/off, both sides | all 20 | all 4 | 16 | 1 | P3-8 |
| C | upsample x1/2/3/4 x 4 kernels | reduced-8 | driving+aerial | 32 | 1 | P3-9 |
| D | 4 estimators x threshold 1/3/5 px | reduced-8 | driving+aerial | 24 | **5** | P3-10 |
| E | homography / affine / similarity / TPS / H+flow | reduced-8 | driving+aerial | 10 | 1 | P3-11 |
| F | input resolution x match count | reduced-8 | driving+aerial | ~16 | 1 | P3-12 |
| G | blur / noise / JPEG / FOV overlap x severity | reduced-8 | driving+aerial | ~40 | 1 | P3-13 |

A "cell" is one `cmreg bench` invocation over 300 pairs with its full matcher list.

**Stage B is deliberately as wide as A.** The inverted-grayscale generality claim (PLAN.md
§7, Figure 6) is only worth making across *every* matcher; a hole in that cell is the whole
finding.

Stages E, F and G have unmet preconditions (P3-4's warp models, P2-2's overlap generator,
P2-3's degradations) and are listed to fix their shape, not to be launched.

## 7. Cost

**Rewritten from stage A's own `time/total_ms`** (runs 1 and 2, 2026-08-27), as the previous
version of this section said it should be. The projection it replaces summed the P0-2 macOS-CPU
timings to 144.6 s/pair for all 20 matchers and assumed a 30-50x A100 speedup; the speedup is
**20-29x**, so every row below is ~1.45x its predecessor.

| dataset | resolution | s/pair, 20 matchers | 300 pairs |
|---|---|---|---|
| `flir` | 640x512 | 5.0 | 25 min |
| `msrs` | 640x480 | 5.1 | 25 min |
| `dronevehicle` | 640x512 | 5.2 | 26 min |
| `llvip` | **1280x1024** | **7.3** | 37 min |

| stage | pairs scored | cost |
|---|---|---|
| A | 1,200 | **~1.9 h**, measured |
| B | 4,800 | ~7.5 h, scaled from A at 5.65 s/pair |
| C-D | reduced-8, ~16,800 | still roughly a day each; the reduced-8 subset's own rate is not measured yet |

Resolution is the only variable that moves the per-pair cost materially -- three 640-wide sets
sit within 4% of each other and the one 1280-wide set costs 45% more. A fifth dataset's budget
should be read off its resolution, not off the mean.

## 8. What this grid does not cover

- **CM-RoMa** (P5-12) re-enters as one more matcher name in stage A's list, under this exact
  protocol and with no special-casing. Nothing here needs changing for it.
- **`m3fd`** is blocked upstream (P1-3) and is a fifth dataset column when it lands.
- **METU-VisTIR** is outside the Tier-1 protocol entirely (X-7, P1-3) — pose GT, not pixel
  alignment.
