from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List


class EventDataLayer:
    def __init__(self, run_dir: Path, snapshot_interval: int = 1, snapshots_enabled: bool = True) -> None:
        self.run_dir = run_dir
        self.events_dir = run_dir / "events"
        self.visualization_dir = run_dir / "visualization"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.events_dir / "events.jsonl"
        self.energy_metrics_path = self.events_dir / "energy_metrics.jsonl"
        self.dominance_metrics_path = self.events_dir / "dominance_metrics.jsonl"
        self.reproduction_metrics_path = self.events_dir / "reproduction_metrics.jsonl"
        self.snapshots_path = self.visualization_dir / "agent_states_per_generation.jsonl"
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.snapshots_enabled = snapshots_enabled
        self._events_by_generation: dict[int, list[dict]] = defaultdict(list)

    def emit_event(self, event: Dict) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        generation = int(event.get("generation", 0))
        self._events_by_generation[generation].append(event)

    def write_generation_snapshot(self, generation: int, agents: Iterable[Dict]) -> None:
        if not self.snapshots_enabled or generation % self.snapshot_interval != 0:
            return
        with self.snapshots_path.open("a", encoding="utf-8") as handle:
            for agent in agents:
                row = {
                    "generation": generation,
                    "agent_id": agent.get("agent_id"),
                    "lineage_id": agent.get("lineage_id"),
                    "energy": round(float(agent.get("energy", 0.0)), 4),
                }
                if "position" in agent:
                    row["position"] = agent.get("position")
                handle.write(json.dumps(row) + "\n")

    def write_generation_projections(self, generation: int, agents: Iterable[Dict]) -> None:
        events = self._events_by_generation.get(generation, [])
        energy_by_lineage: Counter[str] = Counter()
        pop_by_lineage: Counter[str] = Counter()
        total_energy = 0.0
        for agent in agents:
            lineage_id = str(agent.get("lineage_id"))
            energy = float(agent.get("energy", 0.0))
            total_energy += energy
            energy_by_lineage[lineage_id] += energy
            pop_by_lineage[lineage_id] += 1

        rewards = [e for e in events if e.get("event_type") == "reward_distributed"]
        reward_energy_delta = round(sum(float(e.get("energy_delta", 0.0)) for e in rewards), 6)
        rewards_count = len(rewards)
        energy_metrics = {
            "schema_version": "1.0",
            "generation": generation,
            "population": sum(pop_by_lineage.values()),
            "total_energy": round(total_energy, 6),
            "reward_distributions": rewards_count,
            "reward_energy_delta": reward_energy_delta,
            "avg_energy": round(total_energy / max(1, sum(pop_by_lineage.values())), 6),
        }

        dominance_metrics = {
            "schema_version": "1.0",
            "generation": generation,
            "lineage_count": len(pop_by_lineage),
            "top_lineage_population_share": round(max(pop_by_lineage.values(), default=0) / max(1, sum(pop_by_lineage.values())), 6),
            "top_lineage_energy_share": round(max(energy_by_lineage.values(), default=0.0) / max(1e-9, total_energy), 6),
            "lineage_population": dict(pop_by_lineage),
            "lineage_energy": {k: round(v, 6) for k, v in energy_by_lineage.items()},
        }

        births = [e for e in events if e.get("event_type") == "agent_reproduced"]
        reproduction_metrics = {
            "schema_version": "1.0",
            "generation": generation,
            "births": len(births),
            "unique_parents": len({e.get("agent_ids", [None])[0] for e in births if e.get("agent_ids")}),
            "lineage_births": dict(Counter(str(e.get("lineage_ids", ["unknown"])[0]) for e in births if e.get("lineage_ids"))),
        }

        self._append_jsonl(self.energy_metrics_path, energy_metrics)
        self._append_jsonl(self.dominance_metrics_path, dominance_metrics)
        self._append_jsonl(self.reproduction_metrics_path, reproduction_metrics)
        self._events_by_generation.pop(generation, None)

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


class RunDataReader:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def iter_agents_per_generation(self, generation: int | None = None) -> Dict[int, List[Dict]]:
        grouped: dict[int, list[dict]] = defaultdict(list)
        path = self.run_dir / "visualization" / "agent_states_per_generation.jsonl"
        if not path.exists():
            return {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gen = int(row.get("generation", 0))
            if generation is None or gen == generation:
                grouped[gen].append(row)
        return dict(grouped)

    def lineage_groups(self, generation: int | None = None) -> Dict[int, Dict[str, List[Dict]]]:
        grouped = self.iter_agents_per_generation(generation)
        result: dict[int, dict[str, list[dict]]] = {}
        for gen, agents in grouped.items():
            by_lineage: dict[str, list[dict]] = defaultdict(list)
            for agent in agents:
                by_lineage[str(agent.get("lineage_id"))].append(agent)
            result[gen] = dict(by_lineage)
        return result

    def interactions(self, generation: int | None = None) -> Dict[int, List[Dict]]:
        path = self.run_dir / "events" / "events.jsonl"
        grouped: dict[int, list[dict]] = defaultdict(list)
        if not path.exists():
            return {}
        interaction_events = {"collaboration_event", "problem_solved", "reward_distributed", "agent_reproduced"}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") not in interaction_events:
                continue
            gen = int(event.get("generation", 0))
            if generation is None or generation == gen:
                grouped[gen].append(event)
        return dict(grouped)


def reconstruct_generation(run_dir: Path, generation: int) -> List[Dict]:
    reader = RunDataReader(run_dir)
    snapshots = reader.iter_agents_per_generation(generation)
    if generation in snapshots:
        return snapshots[generation]

    state: dict[str, dict] = {}
    events_path = run_dir / "events" / "events.jsonl"
    if not events_path.exists():
        return []

    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event_generation = int(event.get("generation", 0))
        if event_generation > generation:
            break
        event_type = event.get("event_type")
        agent_ids = event.get("agent_ids", [])
        lineage_ids = event.get("lineage_ids", [])
        if event_type == "agent_created" and agent_ids:
            state[agent_ids[0]] = {
                "generation": event_generation,
                "agent_id": agent_ids[0],
                "lineage_id": lineage_ids[0] if lineage_ids else None,
                "energy": float(event.get("metadata", {}).get("energy", 0.0)),
            }
        elif event_type == "agent_died" and agent_ids:
            state.pop(agent_ids[0], None)
        elif event_type == "agent_reproduced" and len(agent_ids) >= 2:
            child_id = agent_ids[1]
            state[child_id] = {
                "generation": event_generation,
                "agent_id": child_id,
                "lineage_id": lineage_ids[1] if len(lineage_ids) > 1 else None,
                "energy": float(event.get("metadata", {}).get("child_start_energy", 0.0)),
            }
        elif event_type == "reward_distributed" and agent_ids:
            aid = agent_ids[0]
            if aid in state:
                state[aid]["energy"] = float(state[aid].get("energy", 0.0)) + float(event.get("energy_delta", 0.0))
    return list(state.values())
