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
uv sync --extra cpu    # Mac dev box
uv sync --extra gpu    # CUDA 13 training server
```

The two extras are declared as `conflicts` so uv resolves them from the same lockfile
against different PyTorch indices. Syncing without an extra leaves torch unresolved.

## Use

```sh
uv run cmreg --version
uv run cmreg gt --config experiments/smoke.yaml    # Tier-1 dense synthetic-warp GT
```

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
