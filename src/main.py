from __future__ import annotations

import argparse
import json

from src.engine.experiments import run_experiment_batch
from src.engine.simulation import SimulationEngine, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis2 simulation runner")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--experiment-config", default=None)
    args = parser.parse_args()

    if args.experiment_config:
        result = run_experiment_batch(args.experiment_config)
        print(json.dumps(result, indent=2))
        return

    cfg = load_config(args.config)
    result = SimulationEngine(cfg).run()
    print(json.dumps({
        "summary_path": result["summary_path"],
        "final_population": result["final_population"],
        "generations": len(result["timeline"]),
    }, indent=2))


if __name__ == "__main__":
    main()
