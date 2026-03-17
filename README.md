# Genesis2

Genesis2 is now a **clean simulator + external controller** architecture.

## Simulator contract

`src.main` does only four things:
1. Load config
2. Run simulation
3. Write structured JSON output
4. Exit

Run it with:

```bash
python -m src.main --config config/default.json
```

Optional explicit output path:

```bash
python -m src.main --config config/default.json --output-json runs/my_run/simulator_output.json
```

The simulator output JSON always includes:
- `config_used`
- `start_population`
- `max_population`
- `final_population`
- `generations_in_target_band`
- `viable_lineages`
- `late_births`
- `dominance_share`
- `stability_flag`
- `failure_mode` (`overshoot|collapse|dominance|low_diversity|weak_growth|unstable|success`)

## External tuning loop

Use `tuning_runner.py` as the controller:

```bash
python tuning_runner.py --runner-config config/tuning_runner_default.json --api-key <your_key>
```

What it does:
- Loads objective and bounded lever definitions
- Requests initial recommendation from Anthropic (if key provided)
- Runs simulator as a subprocess each iteration
- Reads simulator JSON output
- Produces per-run human-readable summaries
- Calls Anthropic again with compact run history + latest summary
- Applies strict bounds and per-run max step limits
- Falls back to deterministic adjustments when API fails/unavailable

Outputs are written to unique session folders under `runs/tuning_runner/`.

## Local UI

The local UI/server is optional and not required for simulator + tuning loop workflows.
