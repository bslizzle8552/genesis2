from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List

DOMAINS = ["math", "logic", "code", "decomposition"]


@dataclass
class Problem:
    problem_id: str
    domain: str
    tier: int
    solved: bool = False
    verified: bool = False
    owner_id: str | None = None


def spawn_problems(generation: int, count: int, tier_mix: Dict[str, float] | None = None) -> List[Problem]:
    problems: List[Problem] = []
    mix = tier_mix if tier_mix else {"1": 0.35, "2": 0.30, "3": 0.20, "4": 0.15}
    tiers = [1, 2, 3, 4]
    weights = [float(mix.get(str(t), 0.0)) for t in tiers]
    if sum(weights) <= 0:
        weights = [0.35, 0.30, 0.20, 0.15]

    for i in range(count):
        tier = random.choices(tiers, weights=weights, k=1)[0]
        domain = random.choice(DOMAINS)
        problems.append(Problem(problem_id=f"G{generation}_P{i}", domain=domain, tier=tier))
    return problems
