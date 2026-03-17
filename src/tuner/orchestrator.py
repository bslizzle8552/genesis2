from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List
import uuid

from src.engine.simulation import SimulationConfig, SimulationEngine


@dataclass
class TunerRunRecord:
    run_id: str
    sequence_index: int
    simulation_run_id: str
    run_dir: str
    final_population: int
    lineage_count: int
    diversity_score: float
    avg_energy: float
    extinction_events: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence_index": self.sequence_index,
            "simulation_run_id": self.simulation_run_id,
            "run_dir": self.run_dir,
            "final_population": self.final_population,
            "lineage_count": self.lineage_count,
            "diversity_score": self.diversity_score,
            "avg_energy": self.avg_energy,
            "extinction_events": self.extinction_events,
        }


class TunerResultsStore:
    def __init__(self, output_dir: Path, batch_id: str) -> None:
        self.output_dir = output_dir
        self.batch_id = batch_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_jsonl_path = self.output_dir / "runs.jsonl"
        self.summary_path = self.output_dir / "batch_summary.json"

    def append_run(self, record: TunerRunRecord) -> None:
        run_path = self.output_dir / f"run_{record.run_id}.json"
        run_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        with self.runs_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")

    def write_summary(self, payload: Dict[str, Any]) -> Path:
        self.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.summary_path


class TuningOrchestrator:
    def __init__(self, *, run_count: int, base_config: Dict[str, Any], output_root: str | Path, batch_id: str = "tuner_batch") -> None:
        self.run_count = max(1, int(run_count))
        self.base_config = dict(base_config)
        self.output_root = Path(output_root)
        self.batch_id = batch_id
        self.batch_dir = self.output_root / self.batch_id
        self.store = TunerResultsStore(self.batch_dir, self.batch_id)

    def run(self) -> Dict[str, Any]:
        records: List[TunerRunRecord] = []

        for index in range(1, self.run_count + 1):
            record = self._run_single(index)
            records.append(record)
            self.store.append_run(record)

        summary = {
            "batch_id": self.batch_id,
            "run_count_requested": self.run_count,
            "run_count_completed": len(records),
            "runs": [record.to_dict() for record in records],
            "paths": {
                "runs_jsonl": str(self.store.runs_jsonl_path),
                "summary": str(self.store.summary_path),
            },
        }
        self.store.write_summary(summary)
        return summary

    def _run_single(self, sequence_index: int) -> TunerRunRecord:
        params = dict(self.base_config)
        params.setdefault("log_dir", str(self.batch_dir / "runs"))
        params.setdefault("experiment_id", self.batch_id)
        params["run_label"] = f"{self.batch_id}_r{sequence_index}"

        cfg = SimulationConfig(**params)
        result = SimulationEngine(cfg).run()

        timeline = result.get("timeline", [])
        final_snapshot = timeline[-1] if timeline else {}

        lineage_count = int(final_snapshot.get("lineage_count", len(result.get("lineages", {}))))
        diversity_score = float(final_snapshot.get("diversity_score", 0.0))
        avg_energy = float(final_snapshot.get("energy_distribution", {}).get("mean", 0.0))
        extinction_events = int(
            sum(
                int(snapshot.get("dominance_metrics", {}).get("lineage_extinction_count", 0))
                for snapshot in timeline
            )
        )

        run_id = uuid.uuid4().hex
        return TunerRunRecord(
            run_id=run_id,
            sequence_index=sequence_index,
            simulation_run_id=str(result.get("run_id", "")),
            run_dir=str(result.get("run_dir", "")),
            final_population=int(result.get("final_population", 0)),
            lineage_count=lineage_count,
            diversity_score=round(diversity_score, 6),
            avg_energy=round(avg_energy, 6),
            extinction_events=extinction_events,
        )


def run_tuning_orchestrator(config_path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    orchestrator = TuningOrchestrator(
        run_count=int(payload.get("run_count", 1)),
        base_config=dict(payload.get("base_config", {})),
        output_root=payload.get("output_root", "runs/tuner"),
        batch_id=str(payload.get("batch_id", "tuner_batch")),
    )
    return orchestrator.run()
