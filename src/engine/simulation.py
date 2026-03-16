from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import mean
from typing import Dict, List, Optional

from src.agents.agent import Agent
from src.agents.genome import Genome
from src.agents.reproduction import reproduce
from src.analytics.logger import SimulationLogger, summarize_energy
from src.world.board import WorldBoard
from src.world.economy import COSTS, REWARDS
from src.world.problems import DOMAINS, Problem, spawn_problems


@dataclass
class SimulationConfig:
    seed: int = 42
    agents: int = 10
    generations: int = 50
    initial_energy: int = 100
    upkeep_cost: int = 6
    tasks_per_generation: int = 8
    log_dir: str = "runs"
    reproduction_threshold: float = 140.0
    mutation_rate: float = 0.15
    tier_mix: Dict[str, float] | None = None


class SimulationEngine:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        random.seed(config.seed)
        self.agents: List[Agent] = []
        self._next_agent_id = 1
        self.agent_energy_history: Dict[str, List[Dict[str, float]]] = {}
        self.agent_contributions: Dict[str, Dict[str, float]] = {}

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
            self._register_agent(agent_id)

    def _claim_agent_id(self) -> str:
        agent_id = f"A{self._next_agent_id}"
        self._next_agent_id += 1
        return agent_id

    def _register_agent(self, agent_id: str) -> None:
        self.agent_energy_history.setdefault(agent_id, [])
        self.agent_contributions.setdefault(
            agent_id,
            {
                "solves": 0,
                "verifications": 0,
                "subtasks": 0,
                "critiques": 0,
                "decompositions": 0,
                "integrations": 0,
                "artifacts_created": 0,
                "artifacts_reused": 0,
                "reward_final_solving": 0.0,
                "reward_verification": 0.0,
                "reward_subtasks": 0.0,
                "reward_critique": 0.0,
                "reward_decomposition": 0.0,
                "reward_integration": 0.0,
                "meaningful_points": 0.0,
            },
        )

    def _collect_lineages(self) -> Dict[str, List[str]]:
        lineages: Dict[str, List[str]] = {}
        for agent in self.agents:
            lineages.setdefault(agent.lineage_id, []).append(agent.agent_id)
        return lineages

    def _diversity_score(self) -> float:
        if not self.agents:
            return 0.0
        domains = DOMAINS
        values = []
        for domain in domains:
            values.append(mean(a.genome.specialization.get(domain, 0.0) for a in self.agents))

        max_dist = (len(domains) ** 0.5)
        dist = sum((v - mean(values)) ** 2 for v in values) ** 0.5
        return round(min(1.0, dist / max_dist + len({a.lineage_id for a in self.agents}) / max(1, len(self.agents))), 4)

    def _award(self, agent_id: str, amount: float, reward_source: str, event_counter: Optional[str] = None) -> None:
        if amount <= 0:
            return
        agent = next((a for a in self.agents if a.agent_id == agent_id), None)
        if agent is None:
            return
        agent.energy += amount
        contrib = self.agent_contributions[agent_id]
        contrib[f"reward_{reward_source}"] += amount
        contrib["meaningful_points"] += amount
        if event_counter:
            contrib[event_counter] += 1

    def _pick_support_agent(self, role: str, exclude: set[str]) -> Optional[Agent]:
        candidates = [a for a in self.agents if a.agent_id not in exclude and a.choose_role() == role]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.energy)

    def _compute_solver_dominance_risk(self, role_totals: Dict[str, Dict[str, float]], role_fitness: Dict[str, float]) -> Dict[str, float | str]:
        solver = role_totals.get("solver", {"population": 0, "reward_final_solving": 0.0})
        total_pop = sum(v.get("population", 0) for v in role_totals.values())
        total_final_reward = sum(v.get("reward_final_solving", 0.0) for v in role_totals.values())
        solver_pop_share = solver.get("population", 0) / max(1, total_pop)
        solver_final_reward_share = solver.get("reward_final_solving", 0.0) / max(1e-6, total_final_reward)

        solver_fitness = role_fitness.get("solver", 0.0)
        non_solver_fitness = mean([v for k, v in role_fitness.items() if k != "solver"]) if len(role_fitness) > 1 else 0.0
        fitness_pressure = 1.0 if solver_fitness > (non_solver_fitness * 1.7 + 1e-6) else 0.0

        risk_score = round(min(1.0, solver_pop_share * 0.45 + solver_final_reward_share * 0.45 + fitness_pressure * 0.10), 3)
        if risk_score >= 0.75:
            band = "high"
        elif risk_score >= 0.5:
            band = "medium"
        else:
            band = "low"
        return {
            "score": risk_score,
            "band": band,
            "solver_population_share": round(solver_pop_share, 3),
            "solver_final_reward_share": round(solver_final_reward_share, 3),
        }

    def _build_report(self, timeline: List[Dict], all_problems: List[Problem], totals: Dict[str, float]) -> Dict:
        solved = sum(1 for p in all_problems if p.solved)
        unsolved = len(all_problems) - solved

        by_agent = []
        for agent in self.agents:
            contrib = self.agent_contributions.get(agent.agent_id, {})
            by_agent.append(
                {
                    "agent_id": agent.agent_id,
                    "energy": round(agent.energy, 2),
                    "lineage_id": agent.lineage_id,
                    "score": round(contrib.get("meaningful_points", 0.0), 3),
                    "contributions": contrib,
                }
            )
        top_agents = sorted(by_agent, key=lambda a: (a["score"], a["energy"]), reverse=True)[:10]

        lineage_scores: Dict[str, Dict[str, float]] = {}
        for agent_id, contrib in self.agent_contributions.items():
            lineage_id = next((a.lineage_id for a in self.agents if a.agent_id == agent_id), None)
            if lineage_id is None:
                lineage_id = agent_id
            bucket = lineage_scores.setdefault(lineage_id, {"solves": 0, "verifications": 0, "meaningful_points": 0.0})
            bucket["solves"] += contrib.get("solves", 0)
            bucket["verifications"] += contrib.get("verifications", 0)
            bucket["meaningful_points"] += contrib.get("meaningful_points", 0.0)

        current_lineages = self._collect_lineages()
        top_lineages = []
        for lineage_id, members in current_lineages.items():
            metric = lineage_scores.get(lineage_id, {"solves": 0, "verifications": 0, "meaningful_points": 0.0})
            top_lineages.append(
                {
                    "lineage_id": lineage_id,
                    "population": len(members),
                    "solves": metric.get("solves", 0),
                    "verifications": metric.get("verifications", 0),
                    "meaningful_points": round(metric.get("meaningful_points", 0.0), 3),
                }
            )
        top_lineages.sort(key=lambda l: (l["meaningful_points"], l["population"]), reverse=True)

        role_contribution_totals: Dict[str, Dict[str, float]] = {}
        for agent in self.agents:
            role = agent.choose_role()
            contrib = self.agent_contributions.get(agent.agent_id, {})
            bucket = role_contribution_totals.setdefault(
                role,
                {
                    "population": 0,
                    "meaningful_points": 0.0,
                    "solves": 0,
                    "verifications": 0,
                    "subtasks": 0,
                    "critiques": 0,
                    "decompositions": 0,
                    "integrations": 0,
                    "reward_final_solving": 0.0,
                    "reward_verification": 0.0,
                    "reward_subtasks": 0.0,
                    "reward_critique": 0.0,
                    "reward_decomposition": 0.0,
                    "reward_integration": 0.0,
                },
            )
            bucket["population"] += 1
            bucket["meaningful_points"] += contrib.get("meaningful_points", 0.0)
            for key in [
                "solves",
                "verifications",
                "subtasks",
                "critiques",
                "decompositions",
                "integrations",
                "reward_final_solving",
                "reward_verification",
                "reward_subtasks",
                "reward_critique",
                "reward_decomposition",
                "reward_integration",
            ]:
                bucket[key] += contrib.get(key, 0.0)

        role_based_fitness = {
            role: round(metrics["meaningful_points"] / max(1, metrics["population"]), 3)
            for role, metrics in role_contribution_totals.items()
        }

        avg_energy = mean([a.energy for a in self.agents]) if self.agents else 0.0
        last_pop = timeline[-1]["population"] if timeline else 0

        warnings = []
        if solved == 0:
            warnings.append("No problems solved during run")
        if last_pop <= 1:
            warnings.append("Population collapse risk")
        if avg_energy < 15:
            warnings.append("Average energy critically low")

        solver_risk = self._compute_solver_dominance_risk(role_contribution_totals, role_based_fitness)

        return {
            "problems": {"solved": solved, "unsolved": unsolved},
            "top_agents": top_agents,
            "top_lineages": top_lineages[:10],
            "artifact_reuse_summary": {
                "created": totals["artifact_created"],
                "reused": totals["artifact_reuse"],
                "reuse_rate": round(totals["artifact_reuse"] / max(1, totals["artifact_created"]), 3),
            },
            "verification_summary": {
                "total_verifications": totals["verified"],
                "total_solves": totals["solved"],
                "verification_rate": round(totals["verified"] / max(1, totals["solved"]), 3),
            },
            "reward_sources": {
                "final_solving": round(totals["reward_final_solving"], 3),
                "verification": round(totals["reward_verification"], 3),
                "subtasks": round(totals["reward_subtasks"], 3),
                "critique": round(totals["reward_critique"], 3),
                "decomposition": round(totals["reward_decomposition"], 3),
                "integration": round(totals["reward_integration"], 3),
            },
            "role_contribution_totals": {k: {mk: round(mv, 3) for mk, mv in v.items()} for k, v in role_contribution_totals.items()},
            "role_based_fitness": role_based_fitness,
            "solver_dominance_risk": solver_risk,
            "warning_flags": warnings,
        }

    def _generate_markdown_summary(self, result: Dict, report: Dict) -> str:
        lines = [
            "# Genesis2 Run Summary",
            "",
            "## Configuration",
            f"- Seed: {self.config.seed}",
            f"- Starting agents: {self.config.agents}",
            f"- Generations: {self.config.generations}",
            f"- Initial energy: {self.config.initial_energy}",
            f"- Reproduction threshold: {self.config.reproduction_threshold}",
            f"- Mutation rate: {self.config.mutation_rate}",
            f"- Upkeep: {self.config.upkeep_cost}",
            "",
            "## Outcomes",
            f"- Final population: {result['final_population']}",
            f"- Problems solved: {report['problems']['solved']}",
            f"- Problems unsolved: {report['problems']['unsolved']}",
            f"- Total verifications: {report['verification_summary']['total_verifications']}",
            f"- Solver dominance risk: {report['solver_dominance_risk']['band']} ({report['solver_dominance_risk']['score']})",
            "",
            "## Reward Sources",
            f"- Final solving: {report['reward_sources']['final_solving']}",
            f"- Verification: {report['reward_sources']['verification']}",
            f"- Subtasks: {report['reward_sources']['subtasks']}",
            f"- Critique: {report['reward_sources']['critique']}",
            f"- Decomposition: {report['reward_sources']['decomposition']}",
            f"- Integration: {report['reward_sources']['integration']}",
            "",
            "## Warnings",
        ]
        if report["warning_flags"]:
            lines.extend(f"- ⚠️ {w}" for w in report["warning_flags"])
        else:
            lines.append("- None")

        lines.extend(["", "## Top Agents"])
        for row in report["top_agents"][:5]:
            lines.append(
                f"- {row['agent_id']} (lineage {row['lineage_id']}): score {row['score']}, energy {row['energy']}"
            )

        return "\n".join(lines) + "\n"

    def run(self, progress_callback=None) -> Dict:
        run_root = Path(self.config.log_dir)
        run_name = f"run_seed{self.config.seed}_g{self.config.generations}"
        logger = SimulationLogger(run_root / run_name)
        snapshots: List[Dict] = []
        board_events: List[Dict] = []
        all_problems: List[Problem] = []
        totals = {
            "solved": 0,
            "verified": 0,
            "subtasks": 0,
            "critiques": 0,
            "decompositions": 0,
            "integrations": 0,
            "artifact_reuse": 0,
            "artifact_created": 0,
            "reward_final_solving": 0.0,
            "reward_verification": 0.0,
            "reward_subtasks": 0.0,
            "reward_critique": 0.0,
            "reward_decomposition": 0.0,
            "reward_integration": 0.0,
        }

        tier_mix = self.config.tier_mix if self.config.tier_mix else {"1": 0.35, "2": 0.30, "3": 0.20, "4": 0.15}

        for generation in range(1, self.config.generations + 1):
            births = 0
            deaths = 0
            artifact_reuse = 0
            artifact_created = 0
            solved = 0
            verified = 0
            subtasks = 0
            critiques = 0
            decompositions = 0
            integrations = 0
            tier_solved = Counter()
            tier_total = Counter()
            generation_points = Counter()

            board = WorldBoard(problems=spawn_problems(generation, self.config.tasks_per_generation, tier_mix=tier_mix))
            all_problems.extend(board.problems)
            for problem in board.problems:
                tier_total[str(problem.tier)] += 1

            for agent in sorted(self.agents, key=lambda a: a.energy, reverse=True):
                open_tasks = board.open_problems()
                if not open_tasks:
                    break

                best = max(open_tasks, key=lambda p: agent.bid_score(p.domain, p.tier))
                if not board.claim(best.problem_id, agent.agent_id):
                    continue
                best.owner_id = agent.agent_id
                board_events.append(
                    {
                        "generation": generation,
                        "message_type": "claim",
                        "agent_id": agent.agent_id,
                        "problem_id": best.problem_id,
                        "tier": best.tier,
                        "domain": best.domain,
                        "detail": f"{agent.agent_id} claimed {best.problem_id}",
                    }
                )

                agent.energy -= COSTS["solve_attempt"]
                contributors = []
                exclude = {agent.agent_id}

                if best.tier >= 2:
                    decomposer = self._pick_support_agent("decomposer", exclude)
                    if decomposer:
                        exclude.add(decomposer.agent_id)
                        contributors.append({"role": "decomposer", "agent": decomposer, "type": "decomposition"})
                        decompositions += 1
                        totals["decompositions"] += 1
                        self._award(decomposer.agent_id, REWARDS["used_decomposition"], "decomposition", "decompositions")
                        generation_points[decomposer.agent_id] += REWARDS["used_decomposition"]
                        totals["reward_decomposition"] += REWARDS["used_decomposition"]

                if best.tier >= 2:
                    subtask_count = max(1, best.tier - 1)
                    for _ in range(subtask_count):
                        subtask_agent = self._pick_support_agent("coordinator", exclude) or self._pick_support_agent("critic", exclude)
                        if subtask_agent:
                            exclude.add(subtask_agent.agent_id)
                            contributors.append({"role": subtask_agent.choose_role(), "agent": subtask_agent, "type": "subtask"})
                            subtasks += 1
                            totals["subtasks"] += 1
                            self._award(subtask_agent.agent_id, REWARDS["useful_subtask"], "subtasks", "subtasks")
                            generation_points[subtask_agent.agent_id] += REWARDS["useful_subtask"]
                            totals["reward_subtasks"] += REWARDS["useful_subtask"]

                if best.tier >= 3:
                    critic = self._pick_support_agent("critic", exclude)
                    if critic:
                        exclude.add(critic.agent_id)
                        contributors.append({"role": "critic", "agent": critic, "type": "critique"})
                        critiques += 1
                        totals["critiques"] += 1
                        self._award(critic.agent_id, REWARDS["useful_critique"], "critique", "critiques")
                        generation_points[critic.agent_id] += REWARDS["useful_critique"]
                        totals["reward_critique"] += REWARDS["useful_critique"]

                support_bonus = min(0.35, len(contributors) * 0.06)
                monolithic_penalty = 0.0
                if best.tier >= 3 and not contributors:
                    monolithic_penalty = 0.2 + 0.05 * (best.tier - 3)

                solve_prob = agent.genome.specialization[best.domain] + (agent.genome.strategy["aggression"] * 0.2)
                solve_prob = max(0.05, min(0.95, solve_prob + support_bonus - monolithic_penalty))
                if random.random() < solve_prob:
                    best.solved = True
                    solved += 1
                    totals["solved"] += 1
                    tier_solved[str(best.tier)] += 1

                    base_final_reward = REWARDS["correct_solution"]
                    if best.tier >= 3 and not contributors:
                        base_final_reward *= 0.35

                    if contributors:
                        integrations += 1
                        totals["integrations"] += 1

                        solver_reward = base_final_reward * 0.45 + REWARDS["integration_bonus"]
                        self._award(agent.agent_id, solver_reward, "final_solving", "solves")
                        generation_points[agent.agent_id] += solver_reward
                        totals["reward_final_solving"] += solver_reward

                        type_weights = {
                            "decomposition": 1.1,
                            "subtask": 1.0,
                            "critique": 1.2,
                            "verification": 1.1,
                        }
                        total_weight = sum(type_weights[c["type"]] for c in contributors)
                        pool = base_final_reward * 0.55
                        for c in contributors:
                            share = pool * (type_weights[c["type"]] / max(1e-6, total_weight))
                            source_map = {
                                "decomposition": "decomposition",
                                "subtask": "subtasks",
                                "critique": "critique",
                                "verification": "verification",
                            }
                            counter_map = {
                                "decomposition": "decompositions",
                                "subtask": "subtasks",
                                "critique": "critiques",
                                "verification": "verifications",
                            }
                            self._award(c["agent"].agent_id, share, source_map[c["type"]], counter_map[c["type"]])
                            generation_points[c["agent"].agent_id] += share
                            totals[f"reward_{source_map[c['type']]}"] += share

                        self.agent_contributions[agent.agent_id]["integrations"] += 1
                        self.agent_contributions[agent.agent_id]["reward_integration"] += REWARDS["integration_bonus"]
                        totals["reward_integration"] += REWARDS["integration_bonus"]
                    else:
                        self._award(agent.agent_id, base_final_reward, "final_solving", "solves")
                        generation_points[agent.agent_id] += base_final_reward
                        totals["reward_final_solving"] += base_final_reward

                    board_events.append(
                        {
                            "generation": generation,
                            "message_type": "solve",
                            "agent_id": agent.agent_id,
                            "problem_id": best.problem_id,
                            "tier": best.tier,
                            "domain": best.domain,
                            "contributors": [
                                {"agent_id": c["agent"].agent_id, "role": c["role"], "contribution": c["type"]}
                                for c in contributors
                            ],
                            "detail": f"{agent.agent_id} solved {best.problem_id}",
                        }
                    )

                    verifier = self._pick_support_agent("verifier", {agent.agent_id})
                    if verifier and best.tier >= verifier.genome.thresholds["verify_tier_gte"]:
                        verifier.energy -= COSTS["verify_attempt"]
                        if random.random() < verifier.genome.specialization.get("logic", 0.4):
                            best.verified = True
                            verified += 1
                            totals["verified"] += 1
                            self._award(
                                verifier.agent_id,
                                REWARDS["successful_verification"],
                                "verification",
                                "verifications",
                            )
                            generation_points[verifier.agent_id] += REWARDS["successful_verification"]
                            totals["reward_verification"] += REWARDS["successful_verification"]
                            board_events.append(
                                {
                                    "generation": generation,
                                    "message_type": "verification",
                                    "agent_id": verifier.agent_id,
                                    "problem_id": best.problem_id,
                                    "tier": best.tier,
                                    "domain": best.domain,
                                    "detail": f"{verifier.agent_id} verified {best.problem_id}",
                                }
                            )
                else:
                    critic = self._pick_support_agent("critic", {agent.agent_id})
                    if critic and random.random() < 0.35:
                        self._award(critic.agent_id, REWARDS["catch_incorrect"], "critique", "critiques")
                        generation_points[critic.agent_id] += REWARDS["catch_incorrect"]
                        totals["reward_critique"] += REWARDS["catch_incorrect"]
                        critiques += 1
                        totals["critiques"] += 1
                        board_events.append(
                            {
                                "generation": generation,
                                "message_type": "catch_incorrect",
                                "agent_id": critic.agent_id,
                                "problem_id": best.problem_id,
                                "tier": best.tier,
                                "domain": best.domain,
                                "detail": f"{critic.agent_id} caught incorrect attempt on {best.problem_id}",
                            }
                        )

                if agent.artifact_store and random.random() < agent.genome.strategy["artifact_reuse_bias"]:
                    agent.energy += REWARDS["artifact_reuse"]
                    artifact_reuse += 1
                    totals["artifact_reuse"] += 1
                    self.agent_contributions[agent.agent_id]["artifacts_reused"] += 1

                if agent.maybe_create_artifact(best.domain):
                    artifact_created += 1
                    totals["artifact_created"] += 1
                    self.agent_contributions[agent.agent_id]["artifacts_created"] += 1

            offspring: List[Agent] = []
            for agent in self.agents:
                agent.energy -= self.config.upkeep_cost
                agent.generation_age += 1
                if (
                    agent.energy >= self.config.reproduction_threshold
                    and agent.generation_age >= 2
                    and generation_points.get(agent.agent_id, 0.0) >= REWARDS["useful_subtask"]
                ):
                    child = reproduce(agent, self._next_agent_id, generation, mutation_rate=self.config.mutation_rate)
                    offspring.append(child)
                    self._next_agent_id += 1
                    self._register_agent(child.agent_id)
                    births += 1
                    board_events.append(
                        {
                            "generation": generation,
                            "message_type": "birth",
                            "agent_id": agent.agent_id,
                            "problem_id": None,
                            "tier": None,
                            "domain": None,
                            "detail": f"{agent.agent_id} reproduced -> {child.agent_id}",
                        }
                    )

            survivors = []
            for agent in self.agents:
                if agent.energy > 0:
                    survivors.append(agent)
                else:
                    deaths += 1
                    board_events.append(
                        {
                            "generation": generation,
                            "message_type": "death",
                            "agent_id": agent.agent_id,
                            "problem_id": None,
                            "tier": None,
                            "domain": None,
                            "detail": f"{agent.agent_id} died",
                        }
                    )

            self.agents = survivors + offspring
            lineages = Counter(a.lineage_id for a in self.agents)
            energies = [a.energy for a in self.agents]
            roles = Counter(a.choose_role() for a in self.agents)

            for agent in self.agents:
                self.agent_energy_history.setdefault(agent.agent_id, []).append({"generation": generation, "energy": round(agent.energy, 3)})

            generation_log = {
                "generation": generation,
                "population": len(self.agents),
                "births": births,
                "deaths": deaths,
                "energy_distribution": summarize_energy(energies),
                "problem_outcomes": {
                    "solved": solved,
                    "verified": verified,
                    "subtasks": subtasks,
                    "critiques": critiques,
                    "decompositions": decompositions,
                    "integrations": integrations,
                },
                "reward_sources": {
                    "final_solving": round(totals["reward_final_solving"], 3),
                    "verification": round(totals["reward_verification"], 3),
                    "subtasks": round(totals["reward_subtasks"], 3),
                    "critique": round(totals["reward_critique"], 3),
                    "decomposition": round(totals["reward_decomposition"], 3),
                    "integration": round(totals["reward_integration"], 3),
                },
                "artifacts": {"created": artifact_created, "reused": artifact_reuse},
                "lineages": dict(lineages),
                "roles": dict(roles),
                "diversity_score": self._diversity_score(),
                "problem_success_by_tier": {
                    tier: {
                        "solved": tier_solved.get(tier, 0),
                        "total": tier_total.get(tier, 0),
                    }
                    for tier in ["1", "2", "3", "4"]
                },
            }
            logger.log_generation(generation_log)
            snapshots.append(generation_log)

            if progress_callback:
                progress_callback(generation_log)

            if not self.agents:
                break

        summary_path = logger.finalize()
        lineage_members = self._collect_lineages()

        agents_snapshot = [a.snapshot() for a in self.agents]
        for agent in agents_snapshot:
            agent_id = agent["agent_id"]
            agent["energy_history"] = self.agent_energy_history.get(agent_id, [])
            agent["contributions"] = self.agent_contributions.get(agent_id, {})
            agent["offspring"] = [a["agent_id"] for a in agents_snapshot if a.get("parent_id") == agent_id]
            agent["workflow"] = agent["genome"].get("workflows", {})

        report = self._build_report(snapshots, all_problems, totals)
        result = {
            "summary_path": str(summary_path),
            "final_population": len(self.agents),
            "agents": agents_snapshot,
            "timeline": snapshots,
            "board_messages": board_events,
            "totals": totals,
            "report": report,
            "lineages": lineage_members,
            "config": asdict(self.config),
            "problems": [
                {
                    "problem_id": p.problem_id,
                    "domain": p.domain,
                    "tier": p.tier,
                    "solved": p.solved,
                    "verified": p.verified,
                    "owner_id": p.owner_id,
                }
                for p in all_problems
            ],
        }

        markdown_path = Path(summary_path).with_name("run_summary.md")
        markdown_path.write_text(self._generate_markdown_summary(result, report), encoding="utf-8")
        result["markdown_summary_path"] = str(markdown_path)
        return result


def load_config(path: str | Path) -> SimulationConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SimulationConfig(**data)
