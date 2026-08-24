# Cross-Modal Optical–Thermal Image Registration

`cmreg` is the research harness behind **CM-Reg-Bench** (a multi-domain optical–thermal
registration benchmark), **CM-RoMa** (asymmetric modality adapters, learned polarity
normalization, a calibrated dense error head, uncertainty-gated refinement), and the
**dense registration-error prediction** task that is the project's headline contribution.

The research design (`PLAN.md`) and the task board built on it (`TASKS.md`) are kept as
local working documents and are not tracked here. `PLAN.md` §15 also indexes the existing
implementations worth porting rather than rewriting — read it, on a machine that has it,
before writing anything that looks like it might already exist.

## Install

One `uv.lock` serves both machines. Pick exactly one extra:

```sh
uv sync --extra cpu --extra matchers    # Mac dev box
uv sync --extra gpu --extra matchers    # CUDA 13 training server
```

`cpu` and `gpu` are declared as `conflicts` so uv resolves them from the same lockfile
against different PyTorch indices. Syncing without one of them leaves torch unresolved.

`matchers` is orthogonal to both and pulls in [`vismatch`][vismatch], which is the learned
arm of the benchmark — RoMa, MINIMA, MatchAnything, LoFTR, LightGlue and the rest. It is
optional because its transitive tree is large and the registry is built to tolerate its
absence: sync without it and `cmreg matchers` reports the OpenCV arm alone, with a warning,
rather than failing to import.

[vismatch]: https://github.com/gmberton/vismatch

## Use

```sh
uv run cmreg --version
uv run cmreg matchers                              # registered matcher names
uv run cmreg gt     -c experiments/smoke.yaml      # Tier-1 dense synthetic-warp GT
uv run cmreg bench  -c experiments/p3_msrs_classical.yaml   # one benchmark cell
uv run cmreg report runs/p3_msrs_classical         # re-render the result block
```

`cmreg bench` resolves `runtime.device` (`auto` → `cuda` / `mps` / `cpu`) once and logs it.
An explicit `cuda` on a machine without CUDA raises rather than falling back — a benchmark
cell budgeted for the A100s that quietly ran overnight on CPU produces a runtime table
nobody can interpret.

### Getting results off the training server

The Windows training box cannot hand files back, so results travel two ways and neither
needs a file transfer:

* **W&B** carries summary metrics, the resolved config, and the tag set. One run per
  matcher, named to `TASKS.md` §0's `{phase}_{method}_{dataset}_{variant}_s{seed}` format.
* **The console block** printed at the end of every `bench` carries the full frozen metric
  schema, the config hash and the git SHA in one copy-pasteable selection. `cmreg report`
  re-renders it from an existing run directory without re-running anything.

The per-pair Parquet store stays on the machine that produced it. It is the substrate the
Phase-8 aggregator reads, not a thing anyone has to move.

### Getting a dataset

Fetch, then adapt. Both are one command each:

```sh
uv run python scripts/fetch_datasets.py --dataset dronevehicle llvip   # into <root>/raw/
uv run cmreg ingest dronevehicle                                       # into <root>/processed/
uv run cmreg ingest --list                                             # the inventory table
```

Dataset bytes live in one tree shared with the sibling `Thermal-To-Optical-Translation`
project (`--dataset-root`), so every optical-thermal set on a machine sits in one place. This
repo keeps only a pointer manifest per dataset, which `cmreg ingest` generates:

```yaml
# dataset/processed/msrs/data.yaml
path: /abs/path/to/Thermal-To-Optical-Translation/dataset/processed/msrs
train: train/visible/images
val: val/visible/images
rgbt:
  optical_token: visible
  thermal_token: infrared
```

`dataset/` is git-ignored wholesale, so those pointers never travel through git -- which is
why they are generated rather than committed. The `rgbt` tokens name the *path segment* that
distinguishes the two modalities; pairing substitutes one for the other rather than zipping
two sorted directory listings.

## Workflow

An **experiment is a config file**. `experiments/*.yaml` is tracked; `runs/` is not.
Every run snapshots its fully-resolved config into its run directory, and carries a
`config_hash()` computed over everything *except* the `runtime` section — so the same
experiment launched on two machines under two names is one hash.

## Data is never committed

`dataset/` is git-ignored wholesale. The optical/thermal pairs are unpublished research
data and this remote is intended to be public. What *is* committed is the split
membership: sorted filename stems plus a sha256 per split, in `splits/*.json`. A dataset
that drifts under a frozen split fails loudly rather than silently changing every number
downstream.

## Smoke-fixture discipline

Because no data and no matcher weights exist on the dev machine, **every component must
run end-to-end on synthetic data, on CPU, in seconds, as a pytest.** Tests build their own
paired dataset with `tmp_path_factory`; nothing in the suite touches the network.

The pre-push gate, also wired as a `prek` hook:

```sh
uv run ruff check && uv run pyright && uv run pytest -m "not slow"
```
