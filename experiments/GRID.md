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
| preprocess | `invert` / `percentile` (2/98), ×1 bicubic | the recipe all three sibling implementations converged on (PLAN.md §15B). **Stage B measured it as non-optimal on 3 of 4 datasets** and it stays anyway — see below |
| estimator | MAGSAC @ 3 px, 10 000 iters, conf 0.9999 | P3-3's default; the 3 px here is the *estimator inlier* threshold and is unrelated to §2's reporting ladder |
| warp model | homography | the only one implemented (P3-4 adds the rest, and is stage E's precondition) |
| `max_keypoints` | 4096 | not honoured by the detector-free backends — `minima-roma` returns 10 000 under any budget (`TODO(P3-12)`) |
| seed | 0 | one seed outside stage D; see §5 |

`experiments/p3a_baseline_grid.yaml` **is** this cell, in the config schema, and every later
stage is authored as a diff against it.

**The anchor's polarity is a recorded bias, not a neutral choice** (P3-8 F15/F19/F20). It is
the best of stage B's four polarities in only 13 of 80 (matcher, dataset) cells, and it is the
*worst* of the four for 19 of 20 matchers on `dronevehicle`. It stays the anchor because the
matcher *ordering* it produces is polarity-invariant (Spearman 0.87-0.99 against all three
alternatives, same top-ranked matcher in 4/4 datasets) and re-anchoring would invalidate stage
A and every stage authored against it, to buy at most 11% on one cell. Quote an absolute
stage-A number with this caveat, and never as evidence that inverting the optical side helps.

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
- **Amended by stage A run 3 (2026-08-28), on the one composed column only.** The claim above
  holds for a dataset *on its native alignment*; once `flir`'s constant is composed the 3 px
  column reads 0.17–0.23 across the top twelve rather than 0.00, and it does discriminate. It
  is still not the headline — the ranking it produces there agrees with the 5 px column, so it
  adds no information and it exists on exactly one of four datasets.
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

**Measured effect, stage A run 3** (300 random val pairs × 20 matchers): composing `flir` drops
`reg/mace` on every matcher (best 13.13 → 7.21), pulls `epe_median` from a 4.5–5.8 px band to
1.96–2.98 px, and takes `success_rate_5px` from ten exact zeros to 0.36–0.54. It does *not* go
to zero — the ~4 px across-pair scatter survives, which is the systematic/random split behaving
as P1-1d described it.

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

### Stage B's 2x2, and why it is not an optical-only on/off

`scripts/p3_stageb_polarity.py`, driving `p3a_baseline_grid.yaml` itself with
`--preprocess-ref` / `--preprocess-mov` per cell — stage B's scientific diff from the anchor is
exactly those two fields, and a second YAML restating the other twenty is how a stage quietly
redefines its own defaults. (`scripts/p3b_calibrate.py` is P2-12's *calibration* driver, not
stage B; the run directories here are `runs/stageb_<dataset>_<label>`.)

| label | reference / moving | relative polarity |
|---|---|---|
| `neither` | `none` / `percentile` | as captured |
| **`optical`** | **`invert` / `percentile`** | flipped — **the anchor recipe (§1)** |
| `thermal` | `none` / `percentile_invert` | flipped |
| `both` | `invert` / `percentile_invert` | as captured |

A matcher sees *relative* polarity, so inverting both sides restores the relation inverting
neither had, and the four cells fall into two pairs. If relative polarity is the whole story,
`optical` and `thermal` agree and `neither` and `both` agree; if inverting the *optical image
specifically* is what helps — which is what PLAN.md §15B's recipe asserts — they do not,
because the two members of a pair differ only in which side carries the inversion. An
optical-only on/off (8 cells) cannot separate those two readings, and only the second licenses
"invert the optical grayscale" as a recipe rather than as a coincidence of this data. The
driver prints that comparison as a `within` / `across` / `ratio` block beside Figure 6's
per-matcher improvement count.

The percentile normalisation stays on the thermal side in all four, so the axis varied is
polarity alone; upsampling stays at ×1 (that is stage C).

Stage A's anchor cells are **not** reused as this stage's `optical` column, though their
`config_hash` is identical and the resume guard would verify it: run 2's `llvip` and
`dronevehicle` rows carry `e60e196-dirty`, and re-running costs ~1.9 h against sixteen cells
that then share one code state. (That decision paid for itself: all 80 re-run anchor cells
reproduced stage A to the printed digit, which is the determinism check of P3-8 F22.)

**Stage B ran on 2026-08-28 and the answer is a negative one.** Neither reading survives as a
recipe. `optical` and `thermal` do land ~3x closer together than either does to `neither` /
`both` on three of four datasets, so what a matcher responds to is the *relation* between the
sides rather than which side was inverted — but the relation that wins is a property of the
dataset (`msrs` wants them to disagree, `flir` and `dronevehicle` want them to agree) and, on
`flir`, of the matcher family (the classical arm wants the opposite of every learned one). The
matchers a reduced-8 would pick barely respond at all: `matchanything-roma` spans 1.02-1.41x
across the four cells where `disk-lightglue` spans 8.04x. Full record and F15-F22 in TASKS.md
P3-8; the consequences that land here are the §1 anchor note above and the §7 rewrite below.

Stages E, F and G have unmet preconditions (P3-4's warp models, P2-2's overlap generator,
P2-3's degradations) and are listed to fix their shape, not to be launched.

## 7. Cost

**Rewritten from stage A's own `time/total_ms`** (runs 1 and 2, 2026-08-27), as the previous
version of this section said it should be. The projection it replaces summed the P0-2 macOS-CPU
timings to 144.6 s/pair for all 20 matchers and assumed a 30-50x A100 speedup; the speedup is
**20-29x**, so every row below is ~1.45x its predecessor.

**Budget against the wall-clock column.** The `time/total_ms` column is matcher time only;
stage B's sixteen cells measured 1.38-1.58x it, because loading 20 backends, reading the images
and finalising the W&B run are charged **per cell rather than per pair** (P3-8's cost revision).

| dataset | resolution | s/pair, 20 matchers | 300 pairs, matcher time | **300 pairs, wall clock** |
|---|---|---|---|---|
| `flir` | 640x512 | 5.0 | 25 min | **34.5 min** |
| `msrs` | 640x480 | 5.1 | 25 min | **35.2 min** |
| `dronevehicle` | 640x512 | 5.2 | 26 min | **38.2 min** |
| `llvip` | **1280x1024** | **7.3** | 37 min | **58.4 min** |

| stage | pairs scored | cost |
|---|---|---|
| A | 1,200 | **~1.9 h** matcher time / **~2.8 h** wall clock, measured |
| B | 4,800 | **11 h 4 min, measured** (7.5 h was projected from matcher time alone) |
| C-D | reduced-8, ~16,800 | still roughly a day each; the reduced-8 subset's own rate is not measured yet, and its **per-cell** overhead is the larger unknown -- stage C is 32 cells against stage B's 16 |

Resolution is the only variable that moves the per-*pair* cost materially -- three 640-wide sets
sit within 4% of each other and the one 1280-wide set costs 45% more. A fifth dataset's budget
should be read off its resolution, not off the mean. Cell *count* is the other half of the
estimate and does not scale with pairs at all.

## 8. What this grid does not cover

- **CM-RoMa** (P5-12) re-enters as one more matcher name in stage A's list, under this exact
  protocol and with no special-casing. Nothing here needs changing for it.
- **`m3fd`** is blocked upstream (P1-3) and is a fifth dataset column when it lands.
- **METU-VisTIR** is outside the Tier-1 protocol entirely (X-7, P1-3) — pose GT, not pixel
  alignment.
