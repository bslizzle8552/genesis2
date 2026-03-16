from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


class SimulationLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.generations: List[Dict] = []

    def log_generation(self, payload: Dict) -> None:
        self.generations.append(payload)

    def finalize(self) -> Path:
        out = self.run_dir / "summary.json"
        out.write_text(json.dumps({"generations": self.generations}, indent=2), encoding="utf-8")
        return out


def summarize_energy(energies: List[float]) -> Dict[str, float]:
    if not energies:
        return {"min": 0, "max": 0, "mean": 0}
    return {"min": min(energies), "max": max(energies), "mean": mean(energies)}
