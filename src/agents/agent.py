from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional

from src.agents.genome import Genome


ROLES = ["solver", "verifier", "decomposer", "critic", "coordinator"]


@dataclass
class Agent:
    agent_id: str
    parent_id: Optional[str]
    lineage_id: str
    generation_born: int
    genome: Genome
    energy: float = 100.0
    current_tasks: List[str] = field(default_factory=list)
    generation_age: int = 0
    artifact_store: List[str] = field(default_factory=list)

    def choose_role(self) -> str:
        strategy = self.genome.strategy
        if strategy["artifact_reuse_bias"] > 0.75:
            return "coordinator"
        if strategy["aggression"] > 0.6:
            return "solver"
        if self.genome.thresholds["verify_tier_gte"] <= 2:
            return "verifier"
        if self.genome.specialization["decomposition"] > 0.6:
            return "decomposer"
        return "critic"

    def bid_score(self, domain: str, tier: int) -> float:
        skill = self.genome.specialization.get(domain, 0.3)
        risk = self.genome.strategy["risk_tolerance"]
        return (skill * 0.7 + risk * 0.3) * (1 + tier * 0.1)

    def maybe_create_artifact(self, task_domain: str) -> Optional[str]:
        if random.random() < 0.15:
            artifact = f"{task_domain}_pattern_{random.randint(1000, 9999)}"
            self.artifact_store.append(artifact)
            return artifact
        return None

    def should_reproduce(self) -> bool:
        return self.energy >= self.genome.thresholds["reproduce_energy"] and self.generation_age >= 2

    def snapshot(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "lineage_id": self.lineage_id,
            "generation_born": self.generation_born,
            "energy": round(self.energy, 2),
            "generation_age": self.generation_age,
            "role": self.choose_role(),
            "artifacts": list(self.artifact_store),
            "genome": self.genome.to_dict(),
        }
