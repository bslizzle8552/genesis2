# Genesis2

SWARM Intelligence Evolution simulator where agents evolve workflows, strategy, and artifacts in an energy economy.

## Features

- Python simulation engine with structured genomes
- Mutation + asexual reproduction
- World board with task tiers/domains and economy scoring
- Per-generation JSON logging in `runs/`
- Local web UI with presets, custom params, dashboard, board/report inspector
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
