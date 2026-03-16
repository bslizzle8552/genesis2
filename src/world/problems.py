from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, List

DOMAINS = ["math", "logic", "code", "decomposition"]


PROMPT_BANK = {
    "math": {
        1: "Compute a simple arithmetic expression and explain each step.",
        2: "Solve a multi-step algebra problem and verify the result.",
        3: "Derive a compact formula for a recurring numeric pattern.",
        4: "Design and justify an approach for a constrained optimization puzzle.",
    },
    "logic": {
        1: "Evaluate a short set of boolean constraints and return the valid assignment.",
        2: "Resolve a small deduction puzzle with explicit justification.",
        3: "Identify contradictions across several rules and produce a consistent set.",
        4: "Build a robust argument for an adversarial reasoning scenario.",
    },
    "code": {
        1: "Write pseudo-code for a tiny transformation task and validate sample I/O.",
        2: "Implement an algorithmic routine with complexity notes.",
        3: "Refactor a buggy workflow into a reliable pipeline and include tests.",
        4: "Architect a fault-tolerant multi-stage solution under strict constraints.",
    },
    "decomposition": {
        1: "Break a small objective into ordered subtasks.",
        2: "Create a dependency-aware plan for a medium objective.",
        3: "Coordinate parallelizable subtasks and identify bottlenecks.",
        4: "Design an end-to-end execution strategy for a complex objective.",
    },
}


@dataclass
class Problem:
    problem_id: str
    generation: int
    domain: str
    tier: int
    prompt_text: str
    solved: bool = False
    verified: bool = False
    owner_id: str | None = None
    solved_generation: int | None = None
    agents_involved: List[str] = field(default_factory=list)
    contribution_chain: List[Dict[str, str]] = field(default_factory=list)
    reward_split: Dict[str, float] = field(default_factory=dict)
    resolution_mode: str = "solo"


def _problem_prompt(domain: str, tier: int) -> str:
    return PROMPT_BANK.get(domain, PROMPT_BANK["logic"]).get(tier, "Solve the task accurately and explain your process.")


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
        problems.append(
            Problem(
                problem_id=f"G{generation}_P{i}",
                generation=generation,
                domain=domain,
                tier=tier,
                prompt_text=_problem_prompt(domain, tier),
            )
        )
    return problems
