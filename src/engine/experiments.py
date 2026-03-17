from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

from src.engine.simulation import SimulationConfig, SimulationEngine


SWEEP_KEYS = [
    "seed",
    "agents",
    "generations",
    "initial_energy",
    "reproduction_threshold",
    "mutation_rate",
    "upkeep_cost",
    "tasks_per_generation",
    "tier_mix",
    "reward_policy_id",
]


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value]


def run_experiment_batch(config_path: str | Path) -> Dict[str, Any]:
    spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
    base = dict(spec.get("base", {}))
    sweep = dict(spec.get("sweep", {}))
    output_root = Path(spec.get("output_root", "runs/experiments"))
    experiment_id = spec.get("experiment_id", "exp_batch")

    sweep_items = [(k, _as_list(v)) for k, v in sweep.items() if k in SWEEP_KEYS]
    if not sweep_items:
        sweep_items = [("seed", _as_list(base.get("seed", 42)))]

    keys = [k for k, _ in sweep_items]
    combinations = list(itertools.product(*[vals for _, vals in sweep_items]))
    comparisons: List[Dict[str, Any]] = []

    for idx, combo in enumerate(combinations, start=1):
        params = dict(base)
        params.update(dict(zip(keys, combo)))
        params.setdefault("log_dir", str(output_root))
        params["experiment_id"] = experiment_id
        params["run_label"] = f"{experiment_id}_{idx}"

        cfg = SimulationConfig(**params)
        result = SimulationEngine(cfg).run()
        timeline = result.get("timeline", [])
        peak = max(timeline, key=lambda item: item.get("population", 0)) if timeline else {}
        final_gen = timeline[-1] if timeline else {}
        phase = result.get("phase_diagnostics", {})

        comparisons.append(
            {
                "run_label": cfg.run_label,
                "seed": cfg.seed,
                "summary_path": result.get("summary_path"),
                "final_population": result.get("final_population", 0),
                "peak_population": peak.get("population", 0),
                "generation_of_peak_population": peak.get("generation", 0),
                "stabilization_generation": phase.get("stabilization_start_generation"),
                "total_solved": result.get("totals", {}).get("solved", 0),
                "solve_rate": round(result.get("totals", {}).get("solved", 0) / max(1, cfg.generations * cfg.tasks_per_generation), 6),
                "collaboration_share": final_gen.get("collaboration_share", 0.0),
                "artifact_assisted_share": round(result.get("totals", {}).get("artifact_reuse", 0) / max(1, result.get("totals", {}).get("solved", 0)), 6),
                "diversity_index": final_gen.get("diversity_score", 0.0),
                "energy_gini": _read_last_metric(result.get("summary_path"), "generation_metrics", "energy_gini"),
                "top_10pct_energy_share": _read_last_metric(result.get("summary_path"), "generation_metrics", "top_10pct_energy_share"),
                "unique_reproducers": _read_last_metric(result.get("summary_path"), "generation_metrics", "cumulative_unique_reproducers"),
                "repeat_reproducers": _read_last_metric(result.get("summary_path"), "generation_metrics", "cumulative_repeat_reproducers"),
                "births_concentration_share_top_5": _read_last_metric(result.get("summary_path"), "generation_metrics", "birth_share_top_5"),
            }
        )

    out = output_root / experiment_id / "comparison_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment_id": experiment_id, "runs": comparisons}, indent=2), encoding="utf-8")
    return {"experiment_id": experiment_id, "comparison_summary": str(out), "runs": comparisons}


def _read_last_metric(summary_path: str | None, stream_name: str, field: str) -> Any:
    if not summary_path:
        return None
    stream_path = Path(summary_path).with_name(f"{stream_name}.jsonl")
    if not stream_path.exists():
        return None
    lines = [line for line in stream_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    row = json.loads(lines[-1])
    return row.get(field)
