from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict

from src.engine.simulation import SimulationEngine, load_config

TARGET_MIN = 90
TARGET_MAX = 110


def _build_simulator_output(config_path: str, run_result: Dict[str, Any]) -> Dict[str, Any]:
    timeline = list(run_result.get("timeline", []))
    populations = [int(step.get("population", 0)) for step in timeline]
    start_population = int(run_result.get("config", {}).get("agents", 0))
    max_population = max(populations, default=start_population)
    final_population = int(run_result.get("final_population", 0))
    generations_in_target_band = sum(1 for p in populations if TARGET_MIN <= p <= TARGET_MAX)
    viable_lineages = int((timeline[-1] if timeline else {}).get("lineage_count", 0))
    dominance_share = float((timeline[-1] if timeline else {}).get("dominance_metrics", {}).get("top_lineage_energy_share", 0.0))

    late_count = max(1, len(timeline) // 4)
    late_window = timeline[-late_count:] if timeline else []
    late_births = int(sum(int(step.get("births", 0)) for step in late_window))
    late_pop_std = float(pstdev([int(step.get("population", 0)) for step in late_window])) if len(late_window) > 1 else 0.0

    if max_population > 130:
        failure_mode = "overshoot"
    elif final_population < 45:
        failure_mode = "collapse"
    elif dominance_share > 0.70:
        failure_mode = "dominance"
    elif viable_lineages < 3:
        failure_mode = "low_diversity"
    elif max_population < 80 or final_population < 80:
        failure_mode = "weak_growth"
    elif late_pop_std > 15.0:
        failure_mode = "unstable"
    else:
        failure_mode = "success"

    return {
        "config_used": run_result.get("full_config") or run_result.get("config") or {},
        "config_path": str(config_path),
        "run_id": run_result.get("run_id"),
        "run_dir": run_result.get("run_dir"),
        "summary_path": run_result.get("summary_path"),
        "start_population": start_population,
        "max_population": max_population,
        "final_population": final_population,
        "generations_in_target_band": generations_in_target_band,
        "viable_lineages": viable_lineages,
        "late_births": late_births,
        "dominance_share": round(dominance_share, 6),
        "stability_flag": failure_mode == "success",
        "failure_mode": failure_mode,
        "timeline_generations": len(timeline),
        "late_population_std": round(late_pop_std, 6),
        "totals": run_result.get("totals", {}),
        "finished_at": run_result.get("finished_at"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis2 simulator")
    parser.add_argument("--config", default="config/default.json", help="Path to simulator configuration JSON")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path for structured simulator metrics JSON. Defaults to <run_dir>/simulator_output.json.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_result = SimulationEngine(cfg).run()
    simulator_output = _build_simulator_output(args.config, run_result)

    output_path = Path(args.output_json) if args.output_json else Path(run_result["run_dir"]) / "simulator_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(simulator_output, indent=2), encoding="utf-8")

    print(json.dumps({"output_json": str(output_path), **simulator_output}, indent=2))


if __name__ == "__main__":
    main()
