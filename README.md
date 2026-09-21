# agile-sim

A simulation app for exploring how teams work, plan, and interact. The user plays the role of a coach: they design or pick a scenario, watch it run turn-by-turn, adjust parameters, and chat with the simulated people to understand and influence outcomes.

The simulation is driven by AI (via OpenRouter) representing **characters** in an organization; structured state (work, vitals, channels) lives in a ledger per run.

## Status

**Harness + web UI** — The CLI loop lives under [`harness/`](harness/); a FastAPI app under [`harness/web/`](harness/web/) browses finished runs, supports interactive read mode (inspector, roster, timeline scrub), and can start **live** step-by-step runs from `/new` when `secrets.yaml` is present. Sprite pipeline: [`docs/assets.md`](docs/assets.md). Build CSS once (`./scripts/build_css.sh`), then `agile-harness serve` and open the printed URLs (or `agile-harness view runs/<id>`). See [`docs/web-conventions.md`](docs/web-conventions.md).

See [`docs/concept.md`](docs/concept.md) for the problem definition, and [`docs/requirements.md`](docs/requirements.md) and [`docs/architecture.md`](docs/architecture.md) for scope and design.

## Harness (CLI)

Requirements: Python 3.11+, and a repo-root **`secrets.yaml`** with `openrouter_api_key` (OpenRouter).

Copy the example and edit:

```bash
cp secrets.example.yaml secrets.yaml
# Edit secrets.yaml — see comments inside secrets.example.yaml
```

```bash
cd agile-sim
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
# Single run
agile-harness run scenarios/two-devs-and-a-pm --model google/gemini-3-flash-preview
# Or let OpenRouter pick a model per call (the model that actually served each
# call is recorded in summary.json totals.served_models and shown next to the
# cost badge in the UI):
agile-harness run scenarios/two-devs-and-a-pm --model openrouter/auto
# Conflict + leadership outcomes (coach DMs `dm/<id>` + team channel `#eng-pilot`)
agile-harness run scenarios/priority-conflict-coaching -v
# Two teams, shared staging (Falcon vs Raven) + coach A/B matrix example
agile-harness run scenarios/two-teams-shared-staging --model google/gemini-3-flash-preview
agile-harness run scenarios/two-teams-shared-staging --coach-mode none ...
agile-harness run scenarios/two-teams-shared-staging --coach-mode preset \
  --coach-preset scenarios/two-teams-shared-staging/coach_presets/agile_light.yaml ...
agile-harness matrix experiments/coaching_ab_matrix.yaml
# Batch (same scenario, N seeds)
agile-harness batch scenarios/two-devs-and-a-pm --runs 5 --model google/gemini-3-flash-preview --concurrency 2
# Matrix (see experiments/ for an example file)
agile-harness matrix experiments/variants.example.yaml
# One shot: matrix → results.json → optional judge → report.md → comparison.md (LLM)
# Judge: --judge auto|on|off (auto: on when batch manifest has ≤ 12 runs, override with EXPERIMENT_AUTO_JUDGE_LIMIT); --no-judge skips.
agile-harness experiment experiments/priority_coach_matrix.yaml -v
# Read-only UI (after building CSS)
./scripts/build_css.sh
agile-harness serve --runs-dir runs   # http://127.0.0.1:8765
agile-harness view runs/<run_or_batch_child>
agile-harness analyse runs/<run_folder_or_batch_folder>  # writes report.md inside that folder
agile-harness analyse runs/<batch_folder> --judge-report runs/<batch_folder>/judge_report.md
agile-harness judge runs/<batch_folder> --model google/gemini-3-flash-preview
```

Add `-v` / `--verbose` on `run`, `batch`, or `matrix` to print each turn, agent call, and coach step to **stderr** (flushed as they happen) so long runs show progress; stdout is unchanged so you can still pipe JSON from `run` if needed.

Spike script (hardcoded scenario, useful to compare with harness behavior): `python experiments/spike.py --model ...`

## Docs

- [`docs/concept.md`](docs/concept.md) — what this is, who it's for, what it's trying to answer.
- [`docs/simulation-model.md`](docs/simulation-model.md) — how the simulation produces realistic behavior without modeling every ticket and meeting.
- [`docs/requirements.md`](docs/requirements.md) — functional and non-functional requirements (draft).
- [`docs/architecture.md`](docs/architecture.md) — early architecture sketch.
- [`docs/agentic-design.md`](docs/agentic-design.md) — agent loops, prompts, memory, tools, and library choice.
- [`docs/ui-design.md`](docs/ui-design.md) — UI shape, design language, interaction patterns, frontend stack.

## Contributing

Not open for contributions yet — the design is still being shaped. PRs welcome once the initial scope is locked in.
