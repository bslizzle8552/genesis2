from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import mean
from typing import Dict, List

from src.agents.agent import Agent
from src.agents.genome import Genome
from src.agents.reproduction import reproduce
from src.analytics.logger import SimulationLogger, summarize_energy
from src.world.board import WorldBoard
from src.world.economy import COSTS, REWARDS
from src.world.problems import DOMAINS, Problem, spawn_problems


ROLE_BUCKETS = ["solver", "verifier", "decomposer", "critic", "coordinator"]
TIER_BOUNTIES = {1: 18.0, 2: 28.0, 3: 40.0, 4: 60.0}


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
    reproduction_contribution_threshold: float = 6.0
    solver_dominance_threshold: float = 0.65
    solver_dominance_window: int = 8


class SimulationEngine:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        random.seed(config.seed)
        self.agents: List[Agent] = []
        self._next_agent_id = 1
        self.agent_energy_history: Dict[str, List[Dict[str, float]]] = {}
        self.agent_contribution_history: Dict[str, List[float]] = {}
        self.agent_contributions: Dict[str, Dict[str, int]] = {}
        self.reproduced_roles: set[str] = set()

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
                "verification_catches": 0,
                "subtasks": 0,
                "plans": 0,
                "integrations": 0,
                "critiques": 0,
                "critique_changes": 0,
                "artifact_contributions": 0,
                "artifacts_created": 0,
                "artifacts_reused": 0,
                "meaningful_score": 0,
                "reward_earned": 0,
                "offspring": 0,
            },
        )
        self.agent_contribution_history.setdefault(agent_id, [])

    def _agent_lookup(self) -> Dict[str, Agent]:
        return {agent.agent_id: agent for agent in self.agents}

    def _pick_agent(self, role: str, problem: Problem, excluded: set[str] | None = None) -> Agent | None:
        banned = excluded if excluded else set()
        candidates = [a for a in self.agents if a.agent_id not in banned and a.energy > 1]
        if not candidates:
            return None

        role_weight = {
            "solver": lambda a: a.genome.strategy["aggression"] + a.bid_score(problem.domain, problem.tier),
            "verifier": lambda a: (1.1 - a.genome.thresholds["verify_tier_gte"] / 4.0) + a.genome.specialization.get("logic", 0.4),
            "decomposer": lambda a: a.genome.specialization.get("decomposition", 0.3) + (1.0 - a.genome.strategy["aggression"]),
            "critic": lambda a: a.genome.specialization.get("logic", 0.3) + a.genome.strategy["risk_tolerance"],
            "coordinator": lambda a: a.genome.strategy["artifact_reuse_bias"] + a.genome.specialization.get(problem.domain, 0.3),
        }
        scored = sorted(candidates, key=role_weight.get(role, lambda a: a.bid_score(problem.domain, problem.tier)), reverse=True)
        top = scored[: max(1, min(5, len(scored)))]
        return random.choice(top)

    def _contribution_points(self, contribution_type: str) -> int:
        return {
            "solve": 6,
            "verify": 5,
            "verify_catch": 6,
            "plan": 5,
            "subtask": 4,
            "critique": 4,
            "critique_change": 6,
            "integrate": 5,
            "artifact": 3,
        }.get(contribution_type, 0)

    def _record_contribution(self, agent_id: str, contribution_type: str, reward: float = 0.0) -> None:
        bucket = self.agent_contributions[agent_id]
        field_map = {
            "solve": "solves",
            "verify": "verifications",
            "verify_catch": "verification_catches",
            "plan": "plans",
            "subtask": "subtasks",
            "critique": "critiques",
            "critique_change": "critique_changes",
            "integrate": "integrations",
            "artifact": "artifact_contributions",
        }
        field = field_map.get(contribution_type)
        if field:
            bucket[field] += 1
        points = self._contribution_points(contribution_type)
        bucket["meaningful_score"] += points
        bucket["reward_earned"] += round(reward, 3)

    def _add_reward(self, agent_id: str, reward: float) -> None:
        self.agent_contributions[agent_id]["reward_earned"] += round(reward, 3)

    def _recent_contribution_score(self, agent_id: str, lookback: int = 5) -> float:
        history = self.agent_contribution_history.get(agent_id, [])
        if not history:
            return 0.0
        return mean(history[-lookback:])

    def _run_problem_ecology(self, problem: Problem, generation: int) -> Dict:
        chain: Dict[str, object] = {
            "problem_id": problem.problem_id,
            "tier": problem.tier,
            "domain": problem.domain,
            "planner_id": None,
            "subtask_ids": [],
            "verifier_id": None,
            "critic_id": None,
            "integrator_id": None,
            "artifact_ids": [],
            "plan_used": False,
            "subtasks_used": 0,
            "verification_catch": False,
            "critique_changed_result": False,
            "artifact_improved_outcome": False,
            "payouts": {},
            "tier_multiplier": 1.0,
            "solved": False,
            "attribution_log": [],
        }

        occupied: set[str] = set()
        planner = self._pick_agent("decomposer", problem)
        if planner and random.random() < planner.genome.specialization.get("decomposition", 0.4) + 0.2:
            planner.energy -= COSTS["solve_attempt"] * 0.4
            chain["planner_id"] = planner.agent_id
            chain["plan_used"] = True
            occupied.add(planner.agent_id)
            chain["attribution_log"].append(f"plan:{planner.agent_id}")
            self._record_contribution(planner.agent_id, "plan")

        subtask_goal = 1 if problem.tier <= 2 else (2 if problem.tier == 3 else 3)
        for _ in range(subtask_goal):
            contributor = self._pick_agent("solver", problem, excluded=occupied)
            if not contributor:
                break
            contributor.energy -= COSTS["solve_attempt"] * 0.35
            occupied.add(contributor.agent_id)
            chain["subtask_ids"].append(contributor.agent_id)
            chain["subtasks_used"] += 1
            chain["attribution_log"].append(f"subtask:{contributor.agent_id}")
            self._record_contribution(contributor.agent_id, "subtask")

        critic = self._pick_agent("critic", problem, excluded=occupied)
        critique_boost = 0.0
        if critic:
            critic.energy -= COSTS["solve_attempt"] * 0.25
            chain["critic_id"] = critic.agent_id
            occupied.add(critic.agent_id)
            self._record_contribution(critic.agent_id, "critique")
            if random.random() < critic.genome.specialization.get("logic", 0.4):
                chain["critique_changed_result"] = True
                critique_boost = 0.18
                self._record_contribution(critic.agent_id, "critique_change")
            chain["attribution_log"].append(f"critic:{critic.agent_id}")

        coordinator = self._pick_agent("coordinator", problem, excluded=occupied)
        artifact_boost = 0.0
        if coordinator and coordinator.artifact_store and random.random() < coordinator.genome.strategy["artifact_reuse_bias"]:
            coordinator.energy += REWARDS["artifact_reuse"]
            chain["artifact_ids"].append(coordinator.agent_id)
            chain["artifact_improved_outcome"] = True
            artifact_boost = 0.1
            self._record_contribution(coordinator.agent_id, "artifact")
            self.agent_contributions[coordinator.agent_id]["artifacts_reused"] += 1
            chain["attribution_log"].append(f"artifact:{coordinator.agent_id}")

        integrator = self._pick_agent("solver", problem)
        if not integrator:
            return chain
        chain["integrator_id"] = integrator.agent_id
        integrator.energy -= COSTS["solve_attempt"]
        problem.owner_id = integrator.agent_id
        solve_prob = (
            integrator.genome.specialization.get(problem.domain, 0.3)
            + (0.05 * chain["subtasks_used"])
            + (0.08 if chain["plan_used"] else 0.0)
            + critique_boost
            + artifact_boost
        )

        verifier = self._pick_agent("verifier", problem, excluded={integrator.agent_id})
        verification_success = False
        if verifier and problem.tier >= 2:
            verifier.energy -= COSTS["verify_attempt"]
            chain["verifier_id"] = verifier.agent_id
            verify_skill = verifier.genome.specialization.get("logic", 0.4)
            verification_success = random.random() < verify_skill
            if not verification_success and random.random() < verify_skill * 0.6:
                chain["verification_catch"] = True
                self._record_contribution(verifier.agent_id, "verify_catch")
            else:
                self._record_contribution(verifier.agent_id, "verify")
            chain["attribution_log"].append(f"verify:{verifier.agent_id}")

        solved = random.random() < min(0.95, solve_prob)
        if chain["verification_catch"] and random.random() < 0.7:
            solved = False
        chain["solved"] = solved
        problem.solved = solved
        problem.verified = verification_success

        if not solved:
            return chain

        has_intermediate = chain["plan_used"] or chain["subtasks_used"] > 0 or chain["artifact_improved_outcome"]
        tier_multiplier = 1.0
        if problem.tier == 1:
            tier_multiplier = 1.0 if has_intermediate else 0.9
        elif problem.tier == 2:
            tier_multiplier = 1.0 if chain["verifier_id"] else 0.72
        elif problem.tier == 3:
            tier_multiplier = 1.0 if (chain["verifier_id"] or has_intermediate) else 0.58
        else:
            full_chain = chain["plan_used"] and chain["subtasks_used"] > 0 and (chain["verifier_id"] or chain["critique_changed_result"])
            tier_multiplier = 1.0 if full_chain else 0.42
        chain["tier_multiplier"] = round(tier_multiplier, 3)

        bounty = TIER_BOUNTIES.get(problem.tier, 25.0) * tier_multiplier
        weights = {
            "planner": 0.16 if chain["planner_id"] else 0.0,
            "subtasks": 0.22 if chain["subtask_ids"] else 0.0,
            "verifier": 0.2 if chain["verifier_id"] else 0.0,
            "critic": 0.1 if chain["critique_changed_result"] else 0.0,
            "artifact": 0.07 if chain["artifact_ids"] else 0.0,
            "integrator": 0.31,
        }
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return chain
        weights = {k: v / total_weight for k, v in weights.items()}

        payouts: Dict[str, float] = {}
        if chain["planner_id"]:
            payouts[chain["planner_id"]] = payouts.get(chain["planner_id"], 0.0) + bounty * weights["planner"]
        if chain["subtask_ids"]:
            subshare = bounty * weights["subtasks"] / len(chain["subtask_ids"])
            for aid in chain["subtask_ids"]:
                payouts[aid] = payouts.get(aid, 0.0) + subshare
        if chain["verifier_id"]:
            payouts[chain["verifier_id"]] = payouts.get(chain["verifier_id"], 0.0) + bounty * weights["verifier"]
        if chain["critic_id"] and chain["critique_changed_result"]:
            payouts[chain["critic_id"]] = payouts.get(chain["critic_id"], 0.0) + bounty * weights["critic"]
        if chain["artifact_ids"]:
            ashare = bounty * weights["artifact"] / len(chain["artifact_ids"])
            for aid in chain["artifact_ids"]:
                payouts[aid] = payouts.get(aid, 0.0) + ashare
        payouts[integrator.agent_id] = payouts.get(integrator.agent_id, 0.0) + bounty * weights["integrator"]

        agent_lookup = self._agent_lookup()
        for aid, reward in payouts.items():
            if aid in agent_lookup:
                agent_lookup[aid].energy += reward
                if aid == integrator.agent_id:
                    self._record_contribution(aid, "solve", reward=reward)
                    self._record_contribution(aid, "integrate")
                else:
                    self._add_reward(aid, reward)
        chain["payouts"] = {aid: round(amount, 3) for aid, amount in payouts.items()}
        chain["attribution_log"].append(f"integrate:{integrator.agent_id}")
        return chain

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

    def _build_report(self, timeline: List[Dict], all_problems: List[Problem], totals: Dict[str, int], contribution_chains: List[Dict]) -> Dict:
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
                    "score": contrib.get("meaningful_score", 0),
                    "contributions": contrib,
                }
            )
        top_agents = sorted(by_agent, key=lambda a: (a["score"], a["energy"]), reverse=True)[:10]

        lineage_scores: Dict[str, Dict[str, int]] = {}
        for agent_id, contrib in self.agent_contributions.items():
            lineage_id = next((a.lineage_id for a in self.agents if a.agent_id == agent_id), None)
            if lineage_id is None:
                lineage_id = agent_id
            bucket = lineage_scores.setdefault(lineage_id, {"solves": 0, "verifications": 0, "members": 0})
            bucket["solves"] += contrib.get("solves", 0)
            bucket["verifications"] += contrib.get("verifications", 0)

        current_lineages = self._collect_lineages()
        top_lineages = []
        for lineage_id, members in current_lineages.items():
            metric = lineage_scores.get(lineage_id, {"solves": 0, "verifications": 0})
            top_lineages.append(
                {
                    "lineage_id": lineage_id,
                    "population": len(members),
                    "solves": metric.get("solves", 0),
                    "verifications": metric.get("verifications", 0),
                }
            )
        top_lineages.sort(key=lambda l: (l["solves"], l["population"]), reverse=True)

        avg_energy = mean([a.energy for a in self.agents]) if self.agents else 0.0
        last_pop = timeline[-1]["population"] if timeline else 0

        solver_shares = [snap.get("role_distribution", {}).get("solver", 0.0) for snap in timeline]
        excessive_windows = 0
        if len(solver_shares) >= self.config.solver_dominance_window:
            for idx in range(self.config.solver_dominance_window - 1, len(solver_shares)):
                window = solver_shares[idx - self.config.solver_dominance_window + 1 : idx + 1]
                if mean(window) >= self.config.solver_dominance_threshold:
                    excessive_windows += 1

        warnings = []
        if solved == 0:
            warnings.append("No problems solved during run")
        if last_pop <= 1:
            warnings.append("Population collapse risk")
        if avg_energy < 15:
            warnings.append("Average energy critically low")
        if excessive_windows >= 2:
            warnings.append("Solver dominance persisted above threshold")

        role_reward_totals: Dict[str, float] = {role: 0.0 for role in ROLE_BUCKETS}
        for agent in self.agents:
            role_reward_totals[agent.choose_role()] += self.agent_contributions.get(agent.agent_id, {}).get("reward_earned", 0)

        reproduced_roles = sorted(self.reproduced_roles)

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
            "role_reward_totals": {k: round(v, 3) for k, v in role_reward_totals.items()},
            "solver_dominance_diagnostic": {
                "threshold": self.config.solver_dominance_threshold,
                "window": self.config.solver_dominance_window,
                "windows_above_threshold": excessive_windows,
                "status": "warning" if excessive_windows >= 2 else "ok",
            },
            "reproduced_roles": reproduced_roles,
            "contribution_chain_summaries": contribution_chains[:20],
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
            f"- Solver dominance diagnostic: {report['solver_dominance_diagnostic']['status']}",
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
            "artifact_reuse": 0,
            "artifact_created": 0,
            "plan_used": 0,
            "subtasks_used": 0,
            "verification_catches": 0,
            "critique_caused_changes": 0,
            "artifacts_improved": 0,
        }
        contribution_chains: List[Dict] = []

        tier_mix = self.config.tier_mix if self.config.tier_mix else {"1": 0.35, "2": 0.30, "3": 0.20, "4": 0.15}

        for generation in range(1, self.config.generations + 1):
            births = 0
            deaths = 0
            artifact_reuse = 0
            artifact_created = 0
            solved = 0
            verified = 0
            subtasks = 0
            verification_catches = 0
            critique_changes = 0
            plan_used = 0
            artifact_improved = 0
            tier_solved = Counter()
            tier_total = Counter()
            generation_contribution_points: Dict[str, float] = {
                a.agent_id: self.agent_contributions[a.agent_id]["meaningful_score"] for a in self.agents
            }

            board = WorldBoard(problems=spawn_problems(generation, self.config.tasks_per_generation, tier_mix=tier_mix))
            all_problems.extend(board.problems)
            for problem in board.problems:
                tier_total[str(problem.tier)] += 1

            for problem in board.problems:
                chain = self._run_problem_ecology(problem, generation)
                contribution_chains.append(chain)
                board_events.append(
                    {
                        "generation": generation,
                        "message_type": "contribution_chain",
                        "agent_id": chain.get("integrator_id"),
                        "problem_id": problem.problem_id,
                        "tier": problem.tier,
                        "domain": problem.domain,
                        "detail": f"chain {problem.problem_id}: {', '.join(chain['attribution_log'])}",
                        "attribution": chain,
                    }
                )
                if chain["solved"]:
                    solved += 1
                    totals["solved"] += 1
                    tier_solved[str(problem.tier)] += 1
                if chain["verifier_id"] and not chain["verification_catch"]:
                    verified += 1
                    totals["verified"] += 1
                subtasks += chain["subtasks_used"]
                totals["subtasks"] += chain["subtasks_used"]
                totals["subtasks_used"] += chain["subtasks_used"]
                if chain["plan_used"]:
                    plan_used += 1
                    totals["plan_used"] += 1
                if chain["verification_catch"]:
                    verification_catches += 1
                    totals["verification_catches"] += 1
                if chain["critique_changed_result"]:
                    critique_changes += 1
                    totals["critique_caused_changes"] += 1
                if chain["artifact_improved_outcome"]:
                    artifact_improved += 1
                    totals["artifacts_improved"] += 1
                    artifact_reuse += 1
                    totals["artifact_reuse"] += 1

                if chain.get("integrator_id"):
                    integ = next((a for a in self.agents if a.agent_id == chain["integrator_id"]), None)
                    if integ and integ.maybe_create_artifact(problem.domain):
                        artifact_created += 1
                        totals["artifact_created"] += 1
                        self.agent_contributions[integ.agent_id]["artifacts_created"] += 1

            offspring: List[Agent] = []
            for agent in self.agents:
                agent.energy -= self.config.upkeep_cost
                agent.generation_age += 1
                recent_score = self._recent_contribution_score(agent.agent_id)
                if (
                    agent.energy >= self.config.reproduction_threshold
                    and agent.generation_age >= 2
                    and recent_score >= self.config.reproduction_contribution_threshold
                ):
                    child = reproduce(agent, self._next_agent_id, generation, mutation_rate=self.config.mutation_rate)
                    offspring.append(child)
                    self._next_agent_id += 1
                    self._register_agent(child.agent_id)
                    self.agent_contributions[agent.agent_id]["offspring"] += 1
                    self.reproduced_roles.add(agent.choose_role())
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
            role_distribution = {role: round(roles.get(role, 0) / max(1, len(self.agents)), 3) for role in ROLE_BUCKETS}

            for aid, starting_total in generation_contribution_points.items():
                prev = self.agent_contribution_history.setdefault(aid, [])
                delta = self.agent_contributions[aid]["meaningful_score"] - starting_total
                prev.append(delta)

            for agent in self.agents:
                self.agent_energy_history.setdefault(agent.agent_id, []).append({"generation": generation, "energy": round(agent.energy, 3)})

            generation_log = {
                "generation": generation,
                "population": len(self.agents),
                "births": births,
                "deaths": deaths,
                "energy_distribution": summarize_energy(energies),
                "problem_outcomes": {"solved": solved, "verified": verified, "subtasks": subtasks},
                "support_accounting": {
                    "plan_used": plan_used,
                    "subtasks_used": subtasks,
                    "verification_catches": verification_catches,
                    "critique_caused_changes": critique_changes,
                    "artifacts_improved": artifact_improved,
                },
                "artifacts": {"created": artifact_created, "reused": artifact_reuse},
                "lineages": dict(lineages),
                "roles": dict(roles),
                "role_distribution": role_distribution,
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

        report = self._build_report(snapshots, all_problems, totals, [c for c in contribution_chains if c["solved"]])
        result = {
            "summary_path": str(summary_path),
            "final_population": len(self.agents),
            "agents": agents_snapshot,
            "timeline": snapshots,
            "board_messages": board_events,
            "totals": totals,
            "contribution_chains": contribution_chains,
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
