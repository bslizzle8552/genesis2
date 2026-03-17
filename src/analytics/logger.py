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
        self.stream_paths: Dict[str, Path] = {}

    def log_generation(self, payload: Dict) -> None:
        self.generations.append(payload)

    def log_stream(self, stream_name: str, payload: Dict) -> None:
        stream_path = self.stream_paths.setdefault(stream_name, self.run_dir / f"{stream_name}.jsonl")
        with stream_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def finalize(self, extra_payload: Dict | None = None) -> Path:
        out = self.run_dir / "summary.json"
        payload = {"generations": self.generations}
        if extra_payload:
            payload.update(extra_payload)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out


def summarize_energy(energies: List[float]) -> Dict[str, float]:
    if not energies:
        return {"min": 0, "max": 0, "mean": 0}
    return {"min": min(energies), "max": max(energies), "mean": mean(energies)}
