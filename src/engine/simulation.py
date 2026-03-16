from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Dict, List

from src.agents.agent import Agent
from src.agents.genome import Genome
from src.agents.reproduction import reproduce
from src.analytics.logger import SimulationLogger, summarize_energy
from src.world.board import WorldBoard
from src.world.economy import COSTS, REWARDS
from src.world.problems import spawn_problems


@dataclass
class SimulationConfig:
    seed: int = 42
    agents: int = 10
    generations: int = 50
    initial_energy: int = 100
    upkeep_cost: int = 6
    tasks_per_generation: int = 8
    log_dir: str = "runs"


class SimulationEngine:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        random.seed(config.seed)
        self.agents: List[Agent] = []
        self._next_agent_id = 1
        for _ in range(config.agents):
            agent_id = self._claim_agent_id()
            self.agents.append(
                Agent(
                    agent_id=agent_id,
                    parent_id=None,
                    lineage_id=agent_id,
                    generation_born=0,
                    genome=Genome.random(),
                    energy=float(config.initial_energy),
                )
            )

    def _claim_agent_id(self) -> str:
        agent_id = f"A{self._next_agent_id}"
        self._next_agent_id += 1
        return agent_id

    def run(self) -> Dict:
        run_root = Path(self.config.log_dir)
        run_name = f"run_seed{self.config.seed}_g{self.config.generations}"
        logger = SimulationLogger(run_root / run_name)
        snapshots: List[Dict] = []

        for generation in range(1, self.config.generations + 1):
            births = 0
            deaths = 0
            artifact_reuse = 0
            artifact_created = 0
            solved = 0
            verified = 0

            board = WorldBoard(problems=spawn_problems(generation, self.config.tasks_per_generation))
            for agent in sorted(self.agents, key=lambda a: a.energy, reverse=True):
                open_tasks = board.open_problems()
                if not open_tasks:
                    break

                best = max(open_tasks, key=lambda p: agent.bid_score(p.domain, p.tier))
                if not board.claim(best.problem_id, agent.agent_id):
                    continue
                best.owner_id = agent.agent_id

                agent.energy -= COSTS["solve_attempt"]
                solve_prob = agent.genome.specialization[best.domain] + (agent.genome.strategy["aggression"] * 0.2)
                if random.random() < solve_prob:
                    best.solved = True
                    solved += 1
                    agent.energy += REWARDS["correct_solution"]

                    if best.tier >= agent.genome.thresholds["verify_tier_gte"]:
                        agent.energy -= COSTS["verify_attempt"]
                        if random.random() < agent.genome.specialization.get("logic", 0.4):
                            best.verified = True
                            verified += 1
                            agent.energy += REWARDS["successful_verification"]
                else:
                    if random.random() < 0.25:
                        agent.energy += REWARDS["catch_incorrect"]

                if agent.artifact_store and random.random() < agent.genome.strategy["artifact_reuse_bias"]:
                    agent.energy += REWARDS["artifact_reuse"]
                    artifact_reuse += 1

                if created := agent.maybe_create_artifact(best.domain):
                    artifact_created += 1

            offspring: List[Agent] = []
            for agent in self.agents:
                agent.energy -= self.config.upkeep_cost
                agent.generation_age += 1
                if agent.should_reproduce():
                    offspring.append(reproduce(agent, self._next_agent_id, generation))
                    self._next_agent_id += 1
                    births += 1

            survivors = []
            for agent in self.agents:
                if agent.energy > 0:
                    survivors.append(agent)
                else:
                    deaths += 1

            self.agents = survivors + offspring
            lineages = Counter(a.lineage_id for a in self.agents)
            energies = [a.energy for a in self.agents]
            roles = Counter(a.choose_role() for a in self.agents)

            generation_log = {
                "generation": generation,
                "population": len(self.agents),
                "births": births,
                "deaths": deaths,
                "energy_distribution": summarize_energy(energies),
                "problem_outcomes": {"solved": solved, "verified": verified},
                "artifacts": {"created": artifact_created, "reused": artifact_reuse},
                "lineages": dict(lineages),
                "roles": dict(roles),
            }
            logger.log_generation(generation_log)
            snapshots.append(generation_log)

            if not self.agents:
                break

        summary_path = logger.finalize()
        return {
            "summary_path": str(summary_path),
            "final_population": len(self.agents),
            "agents": [a.snapshot() for a in self.agents],
            "timeline": snapshots,
        }


def load_config(path: str | Path) -> SimulationConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SimulationConfig(**data)
