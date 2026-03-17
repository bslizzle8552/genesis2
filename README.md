# Genesis2

SWARM Intelligence Evolution simulator where agents evolve workflows, strategy, and artifacts in an energy economy.

## Features

- Python simulation engine with structured genomes
- Mutation + asexual reproduction
- World board with task tiers/domains and economy scoring
- Per-generation JSON logging in `runs/`
- Local web UI with presets sourced from `config/*.json`, custom params, dashboard, board/report inspector
- Windows launcher (`START_GENESIS2.bat`)

## Quick start

```bash
python -m src.main --config config/default.json
python -m src.backends.server --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

## First experiment

The default config is already set to:
- 10 agents
- 50 generations
- local-only execution (no external APIs)
- full generation logging


## Presets

The UI automatically loads every JSON file in `config/` as a run preset (for example `experiment_fast.json`, `experiment_default.json`, `experiment_stress.json`).
You can still override seed/agent/generation values directly before pressing run.

## Healthy swarm scoring harness

Run a scored sweep:

```bash
python -m src.main --experiment-config config/experiment_healthy_swarm_matrix_v1.json
```

Run anti-dominance family matrix (baseline + 5 policy families):

```bash
python -m src.main --anti-dominance-config config/experiment_anti_dominance_matrix_v1.json
```

Each harness run now writes:
- `comparison_summary.json` (legacy run list + scores)
- `runs_detailed.json` (per-trial diagnostics + component score breakdown)
- `config_aggregates.json` (mean/std/min/max score by config)
- `leaderboard.json` (ranked configs)
- `healthy_swarm_score.json` inside each run directory


## Dist. Intelligence Ready - Stable tuning preset

Run the built-in ecological readiness tuner (growth to ~100 by gen ~80, then stable band persistence):

```bash
python -m src.main --tuning-config config/tuning_dist_intelligence_ready_stable.json
```

Tune controls in that JSON file:
- `search.timeout_seconds`
- `search.search_budget`
- `search.target_qualifying_configs`

The run stops when timeout is reached or enough qualifying configs are harvested. Outputs include:
- `runs_detailed.json`, `config_aggregates.json`, `leaderboard.json`
- `harvested_stable_swarms_registry.json` (qualifying config registry + goal conditions)


## Adaptive tuning rig human-readable reports

Start the local control server:

```bash
python -m src.backends.server --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` and use **Adaptive Tuning Rig**.

Each tuning session now writes a unique folder under `runs/tuning_sessions/<session_id>/` with:
- `final_session_summary.json` (machine + human-readable report sections)
- `final_session_report.md` (plain-English session notebook)
- `runs.jsonl` (full per-run records)

Each run directory still keeps existing artifacts and now includes additional observability exports (`reproduction_events.json`, `lineage_summary.json`, etc.).

### Advisory API configuration

You can configure advisory mode in three places:
1. UI fields (endpoint, model, API key env var name, enabled toggle)
2. Environment variables:
   - `GENESIS2_ADVISORY_ENDPOINT`
   - `GENESIS2_ADVISORY_MODEL`
   - `GENESIS2_ADVISORY_API_KEY_ENV` (defaults to `GENESIS2_ADVISORY_API_KEY`)
3. Optional config file: `config/advisory.json` (or `config/tuner_advisory.json`)

Fallback behavior is deterministic-only tuning when advisory is disabled, endpoint is missing, or API key is absent.
