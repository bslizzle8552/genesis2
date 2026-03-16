from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.world.problems import Problem


@dataclass
class WorldBoard:
    problems: List[Problem] = field(default_factory=list)
    claimed: Dict[str, str] = field(default_factory=dict)

    def open_problems(self) -> List[Problem]:
        return [p for p in self.problems if not p.solved]

    def claim(self, problem_id: str, agent_id: str) -> bool:
        if problem_id in self.claimed:
            return False
        self.claimed[problem_id] = agent_id
        return True

    def owner(self, problem_id: str) -> str | None:
        return self.claimed.get(problem_id)
