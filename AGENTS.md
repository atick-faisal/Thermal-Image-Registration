# AGENTS.md

## Working order

1. Read `PLAN.md` and `TASKS.md`. They always track the current state of the
   implementation and may be updated as it proceeds. Both are untracked working
   documents — a fresh clone will not have them, and without them steps 2-4 are guesswork,
   so ask rather than infer.
2. Check `PLAN.md` §15 before writing anything. Four sibling projects on this machine
   already contain working RoMa thermal↔optical registration, a 70-matcher harness, a
   preprocessing catalogue, and the config/seeding/tracking layer. Port, don't reinvent.
3. Check the current state of the codebase.
4. Determine the immediate next step(s). Not everything at once.
5. Implement that step. Verify and test with the available tools.
6. Mark the task done in `TASKS.md`, commit with a gitmoji + conventional message, stop.
7. Wait for verification before starting the next step.

## House style

Adopted wholesale from `../Thermal-To-Optical-Translation/PLAN.md` §13.

- `src/` layout; Python ≥3.12; `from __future__ import annotations` in every module.
- Fully annotated. `@dataclass(frozen=True, slots=True)` for value objects; `TypedDict`
  for batch dicts; `Protocol` for pluggable hooks; `StrEnum` for choice-typed config.
  **pydantic only in `config/`.**
- pyright `standard`; ruff line-length 100.
- Module docstrings explain *why*, citing upstream `file:line` for every deviation.
- **An inline comment on every config field naming its failure mode.** A knob whose failure
  mode nobody can state is a knob nobody should be turning.
- `logging.getLogger(__name__)` with **%-style lazy formatting — never f-strings, never
  `print`**. `basicConfig` only in `cli.py` and in each standalone `scripts/*.py`.
- A custom exception subclass per layer (`ConfigError`, `ManifestError`, `PairingError`,
  `SplitDriftError`, `WarpError`). Fail fast at startup, with messages saying what was tried.
- argparse CLI with an explicit flag→config-path override table. Heavy imports live inside
  each subcommand so `cmreg --version` stays fast and importable without a GPU torch build.
- pytest; synthetic datasets via `tmp_path_factory`; a `slow` marker for anything that
  isn't seconds-on-CPU.
- gitmoji + conventional commits. **Never push** — ask.

## Non-negotiables

- **`gt/warp.py` is validated analytically before anything consumes it** (`TASKS.md` P2-4).
  It sits on the critical path of Phases 4, 5 and 6; if it is subtly wrong, all three train
  and evaluate against corrupted ground truth and the failure surfaces months later.
- **One evaluation path.** Every method computes every metric through the same code. That
  is what makes the comparison table defensible.
- **The metrics schema is frozen** (`TASKS.md` P0-6, `src/cmreg/metrics/schema.py`). Runs do
  not invent key names. Changing it means migrating every prior run.
- **Negative results are recorded, not dropped.** A flat ablation row is a finding.
