from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List

DOMAINS = ["math", "logic", "code", "decomposition"]


@dataclass
class Problem:
    problem_id: str
    domain: str
    tier: int
    solved: bool = False
    verified: bool = False
    owner_id: str | None = None


def spawn_problems(generation: int, count: int) -> List[Problem]:
    problems: List[Problem] = []
    for i in range(count):
        tier = random.choices([1, 2, 3, 4], weights=[0.35, 0.30, 0.20, 0.15], k=1)[0]
        domain = random.choice(DOMAINS)
        problems.append(Problem(problem_id=f"G{generation}_P{i}", domain=domain, tier=tier))
    return problems
