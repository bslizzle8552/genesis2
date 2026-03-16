from __future__ import annotations

import argparse
import json

from src.engine.simulation import SimulationEngine, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis2 simulation runner")
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = SimulationEngine(cfg).run()
    print(json.dumps({
        "summary_path": result["summary_path"],
        "final_population": result["final_population"],
        "generations": len(result["timeline"]),
    }, indent=2))


if __name__ == "__main__":
    main()
