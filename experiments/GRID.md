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
| preprocess | `invert` / `percentile` (2/98), ×1 bicubic | the recipe all three sibling implementations converged on (PLAN.md §15B). **Stage B measured the polarity as non-optimal on 3 of 4 datasets** and it stays anyway — see below. **Stage C measured the ×1** and it is the right factor, so the two halves of this row carry opposite caveats |
| estimator | MAGSAC @ 3 px, 10 000 iters, conf 0.9999 | P3-3's default, **and the one field stage D measured and left where it was** (P3-10 F33/F35): MAGSAC is best or tied on the failure-inclusive metric in 15 of 16 (matcher, dataset) cells, and its own threshold row is flat inside the seed noise, so the 3 px is a free choice rather than a tuned one. Frozen for stages E-G on that basis. The 3 px here is the *estimator inlier* threshold and is unrelated to §2's reporting ladder |
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

**The ×1 is the opposite case: measured, and resolved in the anchor's favour** (P3-9 F24/F25).
Upsampling the thermal side ×3 bicubic — the other half of §15B's recipe — improves 3 of 8
matchers on `flir` and 2 of 8 on `dronevehicle`, by at most 7.4%, while costing up to 27× on the
matchers it reaches. It is not a resolution gain at all: the runner only scores pairs whose
modalities already share a shape (`eval/runner.py:294`) and the reference side is never resized
(`preprocess/variants.py:185`), so ×N is an N× *scale mismatch* between the two views. The
recipe survived in the production path it came from because RoMa resizes to 560×560 and threw it
away. Do not carry the polarity caveat above onto this field.

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

One seed everywhere except stage D, which carries five. X-3 asks for >=5 seeds with CIs on
anything stochastic; the cost of honouring that on every stage is 5x the whole grid.

**The reason this section gave until 2026-08-30 was wrong, and is recorded rather than quietly
replaced.** It read: *"what is actually stochastic here is RANSAC's sampling and the dense
matchers' correspondence sampling ... stage D varies the estimator, so it is where seed-to-seed
variance is the measurement rather than a nuisance."* Measured while authoring P3-10:
**OpenCV's robust estimators are deterministic.** Repeated `cv2.findHomography` fits on
identical input are bit-identical under MAGSAC, RANSAC, LMEDS and PROSAC, and `cv2.setRNGSeed`
does not move them (opencv 5.0.0). RANSAC's sampling contributes exactly zero variance to
anything in this grid, and the one axis whose seeds were justified by it is the estimator axis.

The two terms that *are* stochastic are the **synthetic warp draw** (`gt.seed`) and the
**matcher's own correspondence sampling** (`seed_cell`, pinned per `(seed, pair, matcher)`,
P0-9). Stage D's five seeds measure those, which is still the interval X-3 asks for and still
what P8-2's Wilcoxon needs, so the budget does not change -- only the sentence justifying it.
Any *claim* of the form "A beats B" gets its five seeds before it enters the paper; a scoping
row does not.

**What the five bought, now that they have run.** The seed *spread* is the column stage D's
finding is read against, and it is what turned MAGSAC's threshold row from "small" into "flat":
2.45-3.67 px on `flir`, against a 0.05-0.17 px range across 1/3/5 px for the three RoMa
variants (P3-10 F35). An axis inside that spread has not measured anything, which is a row to report
under X-4 rather than a disappointment. The same column on the failure-inclusive metric is what
sizes the estimator axis itself: 0.0233-0.0533 on `flir` and 0.0233-0.0400 on `dronevehicle`,
against a 0.0233 / 0.0374 range over the four estimators at their best thresholds (F33). The four
are inside the noise; only the 1 px column is outside it. The interval is also what a "MAGSAC beats RANSAC" claim
needs before P8-2's Wilcoxon can be run on it.

That same determinism is what makes stage D affordable at all (§6): twelve estimator variants
taken off one `MatchResult` are order-independent, and therefore the same experiment as twelve
separate runs. `tests/test_estimate.py` pins the cause and
`tests/test_runner.py::test_a_swept_run_reproduces_single_estimator_runs_row_for_row` pins the
consequence, because nothing else in the suite would notice if it stopped holding: every swept
table would look entirely plausible and every number in it would be wrong.

## 6. The stages

`reduced-8` = `roma`, `minima-roma`, `matchanything-roma`, `eloftr`, `xoftr`,
`superpoint-lightglue`, `xfeat`, `sift` — the two best cross-modal families, the two
semi-dense entries, the strongest sparse learned arm, the runtime outlier, and the classical
floor. Named here so no stage picks its own, and defined once in code as
`scripts/p3a_grid.py::REDUCED_8` so no stage can restate it differently.

`responsive-4` = `eloftr`, `xoftr`, `xfeat`, `sift` — the members of reduced-8 whose input
resolution actually reaches the model. Stage C's subsection below is where that is measured.

`driving+aerial` = `flir` + `dronevehicle`: the best-characterised driving set (P1-1c/d) and
the only aerial one.

| stage | axis varied | matchers | datasets | cells | seeds | task |
|---|---|---|---|---|---|---|
| **A** | — (the anchor itself) | all 20 | all 4 | 4 | 1 | P3-7 |
| B | `invert` on/off, both sides | all 20 | all 4 | 16 | 1 | P3-8 |
| **C** | upsample ×1/2/3/4 × 4 kernels | reduced-8 / responsive-4 | driving+aerial | **26** | 1 | P3-9 |
| D | 4 estimators x threshold 1/3/5 px | reduced-8 | driving+aerial | 24 (**10 match passes**) | **5** | P3-10 |
| E | homography / affine / similarity / TPS / H+flow | reduced-8 | driving+aerial | 10 | 1 | P3-11 |
| F | input resolution (**both sides**) x match count | reduced-8 | driving+aerial | ~16 | 1 | P3-12 |
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

### Stage C's grid, and why it is 26 cells rather than 32

`scripts/p3c_upsample.py`, driving `p3a_baseline_grid.yaml` with `--upsample` /
`--interpolation` per cell. Two departures from the naive 4×4, both measured rather than
assumed (TASKS.md P3-9):

**The ×1 column is one cell, not four.** `preprocess.upsample` returns the input untouched at
×1, so the four kernels there produce a bit-identical image; the collapse is exact. It is run
under the anchor kernel and shared across all four kernel tables, and the derived W&B run name
drops the kernel at ×1 for the same reason (`eval/runner.py::_variant_label`). 16 → 13 per
dataset.

**The kernel axis runs on the resolution-responsive four.** Half of reduced-8 resizes its
inputs to a fixed internal resolution and therefore cannot see a resolution change at all:
`roma` / `minima-roma` fix 560×560 (`romatch/models/matcher.py:617`), SuperPoint fixes a
1024 px long side (`LightGlue/lightglue/superpoint.py:115`), and `matchanything-roma` is flat
by measurement (its wrapper only pads to a multiple of 32, so the resize is inside the model
config). For those four the axis is a *resample prefilter*, so they stay in the
anchor-kernel factor column — which is what measures that prefilter at 300 pairs — and sit out
the three other kernels, where they would have bought four indistinguishable rows.

**Read that split off accuracy, not off runtime** (P3-9 F28). The design was chosen on a
Mac-CPU probe where those four were *cost*-flat — `matchanything-roma` read 19,219 / 19,135 /
19,387 / 19,452 ms across ×1–×4 while `eloftr` read 943 / 2,360 / 4,238 / 12,282 — and on the
GPU that is simply false: every backend rises with the factor, by ×1.23 (`minima-roma`) to
×3.62 (`sift`), because what scales on a GPU is the fixed per-pair pipeline (`cv2.resize`, the
host→device transfer, the backend's own resize) rather than the matching. `eloftr` was the
steepest CPU row and is mid-table on GPU; `sift`, CPU-bound whatever `--device` says, is the
steepest GPU row. **A dev-machine cost profile does not transfer to the server and does not
even preserve the ordering** — state a design premise in the quantity you measured on the device
that will run it.

What did hold is the accuracy invariance the split actually rests on: across the whole factor
axis those four move `reg/mace` by ≤2.8% on `flir` and ≤14.8% on `dronevehicle`, with no
consistent direction.

**`xoftr` is capped at ×3.** Its positional encoding is a fixed 256 cells at 1/8 stride, so
2048 px is the ceiling; ×4 on either 640-wide dataset raises
(`XoFTR/src/xoftr/utils/position_encoding.py:36`). It is dropped from the ×4 cells rather than
left to produce 300 `matcher_raised` rows and 300 logged tracebacks in a console that reaches
the Mac by copy-paste, and every table it affects names the exclusion (X-4).

The consequence for reading the stage: a kernel column and the anchor column carry different
matcher sets, so compare *within* a column, and take the four-kernel comparison only over the
responsive four. The driver's `axis_block` does exactly that, and says so in its header.

**Stage C ran on 2026-08-29 and the answer is negative, with a mechanism.** ×3 bicubic improves
3 of 8 matchers on `flir` and 2 of 8 on `dronevehicle`, never by more than 7.4%, while costing
up to 27×. Upsampling the moving side alone is not a resolution gain but an N× **scale
mismatch**: pairs are only scored when their modalities already share a shape
(`eval/runner.py:294`) and the reference is never resized (`preprocess/variants.py:185`). That
sorts the eight matchers into exactly three groups — resized-internally (unaffected),
scale-invariant by construction (`sift`, unaffected), and fixed-stride learned (`eloftr`,
`xoftr`, `xfeat`: destroyed, success@10px 0.72→0.16, 0.84→0.13, 0.43→0.00). The factor dominates
the kernel 14–20×, and the entire kernel effect is `nearest` costing the two semi-dense entries
~2×. **Stages D–G therefore run at ×1, where no kernel exists**; a later stage that does resample
must keep `nearest` away from a semi-dense matcher. Full record and F23–F30 in TASKS.md P3-9;
what lands here is the §1 note above, the premise correction above that, the stage-F row below
and the §7 rewrite.

**Stage D is 24 cells but only 10 match passes.** The axis it varies —
`estimate_homography(..., config.estimate, ...)` at `eval/runner.py:366` — is downstream of the
matcher, so re-running a matcher per estimator is re-running RoMa twelve times to change a
RANSAC threshold (~38 h). One match pass per (dataset, seed) feeding twelve estimator calls is
~4–5 h — **measured at ~11 h**, still a 3.5× saving but not the 8× projected, because a variant
is an estimate *and a score* (§7). The seeds do need re-matching: `config.gt.seed` draws the synthetic warp and `seed_cell`
seeds the matcher's own sampling. `PairRow` already carries `estimator` and `threshold_px`, so
the store needs nothing; the `config_hash`/resume semantics of a directory holding twelve
variants is the open question, and P3-10 records it.

**Stage F must resize *both* sides.** Stage C measured the asymmetric direction and it is
dominated by scale mismatch (P3-9 F25), so an asymmetric resolution sweep would re-measure that
rather than resolution. The precondition: `PreprocessConfig` has no reference-side resize field,
because `preprocess_reference` deliberately never resizes — P3-12 has to add one, or express the
sweep as a decode-time resize of the pair before the frame is fixed.

### Stage D's twelve variants, and why they cost ten match passes

`scripts/p3d_estimator.py`, driving `p3a_baseline_grid.yaml` with `--sweep-estimators` /
`--sweep-thresholds` per cell. The naive reading of the table row above is 24 cells x 5 seeds =
**120 `cmreg bench` invocations**, which at stage C's measured ~19 min per reduced-8 cell is
~38 h. Almost all of that is waste.

**The estimator axis is downstream of the matcher.** `eval/runner.py::_evaluate` matches once
and then calls `estimate_homography`; `config.estimate` is the only thing this stage varies. The
frozen grid re-runs RoMa twelve times per pair to change a RANSAC threshold. So the runner
sweeps *inside* the pair loop instead: `EstimateConfig.variants()` resolves the cross-product,
and one match pass feeds twelve `cv2.findHomography` calls. The stage is **10 match passes** --
2 datasets x 5 seeds, and the seeds genuinely do need re-matching, since `gt.seed` draws the
warp and `seed_cell` seeds the matcher's sampling -- at **~4-5 h**.

`PairRow` already carried `estimator` and `threshold_px` as columns (P3-2), so the results store
needed nothing: a swept directory holds twelve equally-sized populations, distinguished
natively, and `cmreg report` groups on those columns rather than on the matcher alone.

Three decisions it settles, each of which could have gone the other way:

- **One config, one hash, one snapshot per run directory.** The twelve variants share the
  sweep's `config_hash`. Stamping each row with the hash of the single-variant config that would
  have produced it was rejected: `config.snapshot()` writes exactly one `config.yaml`, and a
  hash matching no file on disk destroys the correspondence that makes a run traceable (X-2).
  `stages.py::refuse_a_stale_run` therefore needs no change and gets *stronger* -- changing a
  swept value changes the hash, so resuming onto a directory scored under a different sweep is
  refused. `Config.config_hash` keeps an **empty** sweep out of its payload for the converse
  reason: without that, adding two defaulted fields would have moved every hash in the project
  and made the guard refuse every completed stage A-C directory for a change that altered no
  science.
- **The sweep is two additive lists beside the anchor, not a list of whole configs** -- P3-1's
  rule, that a stage's diff from the anchor is exactly the fields it varies. The anchor
  `(method, threshold_px)` must be one of the swept cells, enforced in `EstimateConfig`: it is
  the variant whose console block the runner prints, and a printed block belonging to no column
  of the stage's tables would be unreadable.
- **`xfeat` cannot run PROSAC**, and that is a row rather than an abort. PROSAC orders its
  minimal samples by confidence, and `xfeat` is one of three vismatch backends returning none
  (P0-2) -- and it is in reduced-8. `estimate/robust.py` still raises for a single-estimator
  run, where failing identically on all 300 pairs is worth saying once; inside a sweep the
  runner records `estimator_needs_confidence` instead, because aborting there discards eleven
  variants *after* the matching they depend on has been paid for. Named in every table it
  touches, out of the rows themselves rather than a hardcoded list (X-4).

**LMEDS is the stage's free falsification, and authoring it corrected an inherited belief.**
This repo asserted in two places that LMEDS "ignores `threshold_px` entirely" and that its
threshold row is "flat by construction". Half true: LMEDS minimises the *median* residual, so
the threshold never reaches its **fit** and its homography really is identical across 1/3/5 px
-- but OpenCV thresholds the returned **inlier mask** anyway, so `n_inliers`, `inlier_ratio`,
`reproj_err` and `estimate/robust.py`'s four-inlier gate all move, and a tight threshold can
fail a cell whose geometry was fine. The driver's integrity block is therefore asserted on the
homographies of pairs solved at all three thresholds, not on any success-weighted aggregate: an
aggregate check would have read the lost pairs as a violation and been wrong. It reads PASS when
LMEDS is identical and every other estimator is not; every estimator flat would be PLAN.md
§15A's bug -- a swept knob that never reaches the solver -- in this stage's shape.

**What it settled, 2026-08-31** (TASKS.md P3-10 F33-F38). MAGSAC @ 3 px stays the anchor, now
on measured grounds. The estimator axis is real on `reg/mace` and nearly absent on the metric
that counts failures, and the gap between those two readings is the finding: LMEDS trades a
fifth of its pairs for a mean over the ones it kept. Vanilla RANSAC is dominated on accuracy and
costs more than the whole matching bill. Two corrections came out of the stage's own output and
are worth carrying into stages E-G's drivers: **aggregate blocks are rendered once per metric**,
never on a success-conditioned mean alone, and **cross-estimator medians run over the matchers
every arm has** -- `xfeat` having no PROSAC cell shifted the printed median enough to invert the
MAGSAC/PROSAC ordering on `flir` (F37). Both now live in `scripts/p3d_estimator.py` and are
pinned in `tests/test_grid_driver.py::TestStageD`.

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
| C | reduced-8 / responsive-4, 7,800 | **4 h 34 min wall clock, measured** (26 cells, 2026-08-29) against ~3.5 h projected from matcher time — ratio **1.31**, inside stage B's 1.38–1.58 band, so the per-cell overhead correction now holds on a second stage. Upsampling does raise the bill, but far less than the pixel count suggests: on GPU every backend rises only ×1.23–×3.62 across ×1–×4, because what scales is the fixed per-pair pipeline and not the matching (P3-9 F28) |
| D | reduced-8, 288,000 rows off 24,000 matches | **~11 h 10 min, measured** (2026-08-31; 60 min 38 s per `flir` pass, 71–74 min per `dronevehicle` one) against **~4–5 h projected**. The collapse to **10 match passes** was still right — the frozen 120 invocations were ~38 h — but it won ~3.5×, not ~8×: **a swept pass costs ~3.8 single-variant cells**, because the twelve variants are twelve estimate-*and-score* cycles and not twelve `cv2.findHomography` calls. One `dronevehicle` pass is ~13 min matching, ~30 min estimation (~24 of it vanilla RANSAC alone), ~25 min scoring 28,800 rows, ~2 min Parquet and 96 W&B runs. Budget any future stage that sweeps a *scoring* axis at that ratio. Read the estimator's own bill off `time/estimate_ms`, never off `time/total_ms`, which every one of a pair's twelve rows charges the full match to |

Resolution is the only variable that moves the per-*pair* cost materially -- three 640-wide sets
sit within 4% of each other and the one 1280-wide set costs 45% more. A fifth dataset's budget
should be read off its resolution, not off the mean. That rate is for *dataset* resolution, i.e.
a bigger file decoded and matched; stage C's upsampling enlarges only the preprocessed moving
image and costs less (`roma` +26% at ×2 against the +45% a 2×-wide dataset would imply), so do
not budget stage F off this line. Cell *count* is the other half of the
estimate and does not scale with pairs at all.

## 8. What this grid does not cover

- **CM-RoMa** (P5-12) re-enters as one more matcher name in stage A's list, under this exact
  protocol and with no special-casing. Nothing here needs changing for it.
- **`m3fd`** is blocked upstream (P1-3) and is a fifth dataset column when it lands.
- **METU-VisTIR** is outside the Tier-1 protocol entirely (X-7, P1-3) — pose GT, not pixel
  alignment.
