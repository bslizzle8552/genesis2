from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, List


DOMAINS = ["math", "logic", "code", "decomposition"]


@dataclass
class Genome:
    specialization: Dict[str, float]
    workflows: Dict[str, List[str]]
    thresholds: Dict[str, float]
    strategy: Dict[str, float]

    @staticmethod
    def random() -> "Genome":
        workflows = {
            "math": ["classify", "plan", "solve", "verify", "submit"],
            "logic": ["classify", "plan", "solve", "verify", "submit"],
            "code": ["classify", "plan", "solve", "self_test", "verify", "submit"],
            "decomposition": ["classify", "plan", "decompose", "solve", "verify", "submit"],
        }
        return Genome(
            specialization={d: random.uniform(0.3, 0.8) for d in DOMAINS},
            workflows=workflows,
            thresholds={"reproduce_energy": random.uniform(130, 155), "verify_tier_gte": random.choice([2, 3])},
            strategy={
                "aggression": random.uniform(0.2, 0.8),
                "risk_tolerance": random.uniform(0.2, 0.8),
                "artifact_reuse_bias": random.uniform(0.3, 0.9),
            },
        )

    def mutate(self, mutation_rate: float = 0.15) -> "Genome":
        def bounded(value: float, delta: float = 0.2) -> float:
            return max(0.0, min(1.0, value + random.uniform(-delta, delta)))

        specialization = {
            k: bounded(v) if random.random() < mutation_rate else v
            for k, v in self.specialization.items()
        }

        strategy = {
            k: bounded(v) if random.random() < mutation_rate else v
            for k, v in self.strategy.items()
        }

        thresholds = dict(self.thresholds)
        if random.random() < mutation_rate:
            thresholds["reproduce_energy"] = max(110.0, min(180.0, thresholds["reproduce_energy"] + random.uniform(-15, 15)))
        if random.random() < mutation_rate:
            thresholds["verify_tier_gte"] = random.choice([1, 2, 3, 4])

        workflows = {k: list(v) for k, v in self.workflows.items()}
        if random.random() < mutation_rate:
            domain = random.choice(list(workflows.keys()))
            flow = workflows[domain]
            candidates = ["classify", "plan", "solve", "verify", "submit", "self_test", "decompose", "critique"]
            if random.random() < 0.5 and len(flow) > 3:
                flow.pop(random.randrange(len(flow) - 1))
            else:
                flow.insert(random.randrange(len(flow)), random.choice(candidates))
            workflows[domain] = flow

        return Genome(specialization=specialization, workflows=workflows, thresholds=thresholds, strategy=strategy)

    def to_dict(self) -> dict:
        return {
            "specialization": self.specialization,
            "workflows": self.workflows,
            "thresholds": self.thresholds,
            "strategy": self.strategy,
        }
