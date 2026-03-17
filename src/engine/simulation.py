from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from statistics import mean
from typing import Dict, List, Optional
import uuid

from src.agents.agent import Agent
from src.agents.genome import Genome
from src.agents.reproduction import reproduce
from src.analytics.logger import SimulationLogger, summarize_energy
from src.analytics.metrics import detect_phase, gini, lineage_energy, percentile, role_energy, top_share
from src.world.board import WorldBoard
from src.world.economy import COSTS, REWARDS
from src.world.problems import DOMAINS, Problem, spawn_problems


ROLE_BUCKETS = ["solver", "verifier", "decomposer", "critic", "coordinator"]
TIER_BOUNTIES = {1: 22.0, 2: 30.0, 3: 46.0, 4: 70.0}


@dataclass
class SimulationConfig:
    seed: int = 42
    agents: int = 10
    generations: int = 50
    initial_energy: int = 100
    upkeep_cost: int = 6
    tasks_per_generation: int = 15
    log_dir: str = "runs"
    reproduction_threshold: float = 130.0
    mutation_rate: float = 0.15
    diversity_bonus: float = 1.0
    diversity_min_lineages: int = 4
    immigrant_injection_count: int = 2
    tier_mix: Dict[str, float] | None = None
    api_access: bool = False
    run_label: str = "default"
    experiment_id: str = "adhoc"
    reward_policy_id: str = "baseline"
    diagnostics_window: int = 12
    overwrite: bool = False


class SimulationEngine:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        random.seed(config.seed)
        self.agents: List[Agent] = []
        self._next_agent_id = 1
        self.agent_energy_history: Dict[str, List[Dict[str, float]]] = {}
        self.agent_contributions: Dict[str, Dict[str, float]] = {}
        self.agent_contribution_history: Dict[str, List[float]] = {}
        self.reproduced_roles: set[str] = set()
        self.role_reproduction_counts: Counter[str] = Counter()
        self.agent_lifecycle: Dict[str, Dict[str, float | int | None]] = {}
        self.artifact_registry: Dict[str, Dict[str, object]] = {}

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
                "critiques": 0,
                "plans": 0,
                "critique_changes": 0,
                "artifact_contributions": 0,
                "decompositions": 0,
                "integrations": 0,
                "artifacts_created": 0,
                "artifacts_reused": 0,
                "offspring": 0,
                "reward_final_solving": 0.0,
                "reward_verification": 0.0,
                "reward_subtasks": 0.0,
                "reward_critique": 0.0,
                "reward_decomposition": 0.0,
                "reward_integration": 0.0,
                "reward_earned": 0.0,
                "meaningful_score": 0.0,
                "meaningful_points": 0.0,
            },
        )
        self.agent_contribution_history.setdefault(agent_id, [])
        self.agent_lifecycle.setdefault(
            agent_id,
            {
                "lifetime_births": 0,
                "first_reproduction_generation": None,
                "last_reproduction_generation": None,
                "lifetime_energy_earned": 0.0,
                "lifetime_energy_spent": 0.0,
                "lifetime_contribution_score": 0.0,
                "lifespan_generations": 0,
            },
        )

    def _agent_lookup(self) -> Dict[str, Agent]:
        return {agent.agent_id: agent for agent in self.agents}

    def _pick_support_agent(self, role: str, excluded: set[str] | None = None) -> Agent | None:
        problem = Problem(problem_id="support", generation=0, domain=random.choice(DOMAINS), tier=1, prompt_text="")
        return self._pick_agent(role, problem, excluded=excluded)

    def _award(self, agent_id: str, reward: float, _reward_kind: str, contribution_key: str) -> None:
        if contribution_key in self.agent_contributions[agent_id]:
            self.agent_contributions[agent_id][contribution_key] += 1
        self._add_reward(agent_id, reward)

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
            "verify": 6,
            "verify_catch": 7,
            "plan": 6,
            "subtask": 5,
            "critique": 5,
            "critique_change": 7,
            "integrate": 6,
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
            bucket[field] = bucket.get(field, 0) + 1
        points = self._contribution_points(contribution_type)
        bucket["meaningful_score"] += points
        bucket["meaningful_points"] += points
        bucket["reward_earned"] += round(reward, 3)

    def _add_reward(self, agent_id: str, reward: float) -> None:
        self.agent_contributions[agent_id]["reward_earned"] += round(reward, 3)
        self.agent_lifecycle.setdefault(agent_id, {}).setdefault("lifetime_energy_earned", 0.0)
        self.agent_lifecycle[agent_id]["lifetime_energy_earned"] = round(
            float(self.agent_lifecycle[agent_id].get("lifetime_energy_earned", 0.0)) + float(reward), 4
        )

    def _spend_energy(self, agent_id: str, amount: float) -> None:
        self.agent_lifecycle.setdefault(agent_id, {}).setdefault("lifetime_energy_spent", 0.0)
        self.agent_lifecycle[agent_id]["lifetime_energy_spent"] = round(
            float(self.agent_lifecycle[agent_id].get("lifetime_energy_spent", 0.0)) + float(amount), 4
        )

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
        has_verification = bool(chain["verifier_id"])
        full_chain = chain["plan_used"] and chain["subtasks_used"] > 0 and (has_verification or chain["critique_changed_result"])

        tier_multiplier = 1.0
        if problem.tier == 1:
            tier_multiplier = 1.0 if not has_intermediate else 0.96
        elif problem.tier == 2:
            tier_multiplier = 1.0 if not has_verification else 1.06
        elif problem.tier == 3:
            if has_verification and has_intermediate:
                tier_multiplier = 1.12
            elif has_verification or has_intermediate:
                tier_multiplier = 0.92
            else:
                tier_multiplier = 0.72
        else:
            if full_chain:
                tier_multiplier = 1.22
            elif has_verification and has_intermediate:
                tier_multiplier = 0.88
            else:
                tier_multiplier = 0.55
        chain["tier_multiplier"] = round(tier_multiplier, 3)

        bounty = TIER_BOUNTIES.get(problem.tier, 25.0) * tier_multiplier
        weights = {
            "planner": 0.14 if chain["planner_id"] else 0.0,
            "subtasks": 0.2 if chain["subtask_ids"] else 0.0,
            "verifier": 0.24 if chain["verifier_id"] else 0.0,
            "critic": 0.12 if chain["critique_changed_result"] else 0.0,
            "artifact": 0.08 if chain["artifact_ids"] else 0.0,
            "integrator": 0.36,
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

    def _inject_immigrants(self, generation: int, reason: str) -> int:
        injected = max(0, int(self.config.immigrant_injection_count))
        if injected <= 0:
            return 0
        for _ in range(injected):
            agent_id = self._claim_agent_id()
            immigrant = Agent(
                agent_id=agent_id,
                parent_id=None,
                lineage_id=agent_id,
                generation_born=generation,
                genome=Genome.random(),
                energy=float(self.config.initial_energy * 0.85),
            )
            self.agents.append(immigrant)
            self._register_agent(agent_id)
        return injected

    def _calculate_ecosystem_diagnostics(self, timeline: List[Dict], role_contribution_totals: Dict[str, Dict[str, float]]) -> Dict:
        if not timeline:
            return {
                "carrying_capacity": {"estimated_population": 0.0, "risk": "high"},
                "reproduction_bottleneck_risk": "high",
                "diversity_collapse_risk": "high",
            }

        recent = timeline[-min(20, len(timeline)):]
        avg_pop = mean(s["population"] for s in recent)
        avg_births = mean(s["births"] for s in recent)
        avg_deaths = mean(s["deaths"] for s in recent)
        avg_energy = mean((s.get("energy_distribution") or {}).get("mean", 0.0) for s in recent)

        carrying_est = round(max(0.0, avg_pop + max(0.0, avg_births - avg_deaths) * 5), 2)
        carrying_risk = "low" if carrying_est >= 20 else ("medium" if carrying_est >= 14 else "high")

        repro_ratio = avg_births / max(0.1, avg_deaths)
        repro_risk = "low" if repro_ratio >= 0.9 else ("medium" if repro_ratio >= 0.6 else "high")
        if avg_energy < self.config.reproduction_threshold * 0.6:
            repro_risk = "high"

        recent_div = [s.get("diversity_score", 0.0) for s in recent]
        diversity_risk = "low"
        if recent_div and mean(recent_div) < 0.18:
            diversity_risk = "high"
        elif recent_div and mean(recent_div) < 0.3:
            diversity_risk = "medium"

        role_viability = {
            role: {
                "population": metrics.get("population", 0),
                "reproductions": int(self.role_reproduction_counts.get(role, 0)),
                "mean_points": round(metrics.get("meaningful_points", 0.0) / max(1, metrics.get("population", 0)), 3),
            }
            for role, metrics in role_contribution_totals.items()
        }

        return {
            "carrying_capacity": {
                "estimated_population": carrying_est,
                "avg_population_recent": round(avg_pop, 2),
                "avg_births_recent": round(avg_births, 3),
                "avg_deaths_recent": round(avg_deaths, 3),
                "avg_energy_recent": round(avg_energy, 3),
                "risk": carrying_risk,
            },
            "reproduction_bottleneck_risk": repro_risk,
            "diversity_collapse_risk": diversity_risk,
            "role_viability_by_reproduction": role_viability,
        }

    def _collect_lineages(self) -> Dict[str, List[str]]:
        lineages: Dict[str, List[str]] = {}
        for agent in self.agents:
            lineages.setdefault(agent.lineage_id, []).append(agent.agent_id)
        return lineages

    def _diversity_score(self) -> float:
        if not self.agents:
            return 0.0
        values = [mean(a.genome.specialization.get(domain, 0.0) for a in self.agents) for domain in DOMAINS]
        max_dist = len(DOMAINS) ** 0.5
        dist = sum((v - mean(values)) ** 2 for v in values) ** 0.5
        return round(min(1.0, dist / max_dist + len({a.lineage_id for a in self.agents}) / max(1, len(self.agents))), 4)

    def _plateau(self, series: List[float], window: int = 10, epsilon: float = 0.01) -> bool:
        if len(series) < window * 2:
            return False
        first = mean(series[-(window * 2):-window])
        last = mean(series[-window:])
        return abs(last - first) <= epsilon

    def _no_api_capability_report(self, timeline: List[Dict], all_problems: List[Problem], agents_snapshot: List[Dict]) -> Dict:
        tier_history = defaultdict(list)
        domain_history = defaultdict(list)
        for snap in timeline:
            for tier, vals in (snap.get("problem_success_by_tier") or {}).items():
                rate = vals["solved"] / vals["total"] if vals["total"] else 0.0
                tier_history[tier].append(round(rate, 4))
            for domain, vals in (snap.get("problem_success_by_domain") or {}).items():
                rate = vals["solved"] / vals["total"] if vals["total"] else 0.0
                domain_history[domain].append(round(rate, 4))

        solved = [p for p in all_problems if p.solved]
        collaborative = [p for p in solved if p.resolution_mode == "collaborative"]
        solo = [p for p in solved if p.resolution_mode == "solo"]

        first_block = timeline[:50]
        last_block = timeline[-50:]

        def block_rate(block: List[Dict]) -> float:
            total = sum(sum(v["total"] for v in (g.get("problem_success_by_tier") or {}).values()) for g in block)
            good = sum(sum(v["solved"] for v in (g.get("problem_success_by_tier") or {}).values()) for g in block)
            return round(good / total, 4) if total else 0.0

        founders = {a["agent_id"] for a in agents_snapshot if a.get("generation_born") == 0}
        founder_owned = [p for p in all_problems if p.owner_id in founders]
        evolved_owned = [p for p in all_problems if p.owner_id and p.owner_id not in founders]

        founder_rate = round(sum(1 for p in founder_owned if p.solved) / max(1, len(founder_owned)), 4)
        evolved_rate = round(sum(1 for p in evolved_owned if p.solved) / max(1, len(evolved_owned)), 4)

        unsolved_tiers = sorted({str(p.tier) for p in all_problems if not p.solved})
        unsolved_domains = sorted({p.domain for p in all_problems if not p.solved})

        improved = []
        plateaued = []
        for tier, rates in tier_history.items():
            if not rates:
                continue
            if self._plateau(rates):
                plateaued.append(f"tier_{tier}")
            elif rates[-1] - rates[0] > 0.03:
                improved.append(f"tier_{tier}")

        for domain, rates in domain_history.items():
            if not rates:
                continue
            if self._plateau(rates):
                plateaued.append(f"domain_{domain}")
            elif rates[-1] - rates[0] > 0.03:
                improved.append(f"domain_{domain}")

        artifact_solved = [p for p in solved if any(c.get("type") == "artifact_reuse" for c in p.contribution_chain)]

        return {
            "mode": "no_api" if not self.config.api_access else "with_api",
            "tier_success_over_time": dict(tier_history),
            "domain_success_over_time": dict(domain_history),
            "plateau_detected": bool(plateaued),
            "plateau_dimensions": plateaued,
            "first_50_vs_last_50": {
                "first_50_success_rate": block_rate(first_block),
                "last_50_success_rate": block_rate(last_block),
            },
            "founder_vs_evolved": {
                "random_founder_baseline": founder_rate,
                "evolved_population": evolved_rate,
                "delta": round(evolved_rate - founder_rate, 4),
            },
            "collaboration_effect": {
                "solo_successes": len(solo),
                "collaborative_successes": len(collaborative),
                "collaborative_share": round(len(collaborative) / max(1, len(solved)), 4),
            },
            "artifact_effect": {
                "artifact_assisted_successes": len(artifact_solved),
                "artifact_assisted_share": round(len(artifact_solved) / max(1, len(solved)), 4),
            },
            "unsolved_tiers": unsolved_tiers,
            "unsolved_domains": unsolved_domains,
            "improved_dimensions": improved,
            "plateaued_dimensions": plateaued,
        }

    def _observability_report(self, timeline: List[Dict], all_problems: List[Problem], agents_snapshot: List[Dict], totals: Dict[str, int]) -> Dict:
        energies = sorted(float(a.get("energy", 0.0)) for a in agents_snapshot)

        def percentile(values: List[float], pct: float) -> float:
            if not values:
                return 0.0
            index = int(round((len(values) - 1) * pct))
            return values[max(0, min(index, len(values) - 1))]

        def histogram(values: List[float], bins: int = 10) -> List[Dict[str, float]]:
            if not values:
                return []
            low, high = min(values), max(values)
            width = max(1.0, (high - low) / bins)
            result = []
            for i in range(bins):
                start = low + i * width
                end = start + width
                count = sum(1 for v in values if (start <= v < end) or (i == bins - 1 and v <= end))
                result.append({"start": round(start, 2), "end": round(end, 2), "count": count})
            return result

        births_per_agent = {aid: int(c.get("offspring", 0)) for aid, c in self.agent_contributions.items() if c.get("offspring", 0)}
        unique_reproducers = len(births_per_agent)
        repeat_reproducers = sum(1 for count in births_per_agent.values() if count > 1)

        lineage_counts = Counter()
        lineage_by_agent = {a.get("agent_id"): a.get("lineage_id") for a in agents_snapshot}
        for aid, births in births_per_agent.items():
            lineage_counts[str(lineage_by_agent.get(aid, aid))] += births

        contribution_type_map = {
            "solve": "final_solve",
            "verification": "verification",
            "subtask": "subtask",
            "plan": "decomposition",
            "decomposition": "decomposition",
            "critique": "critique",
            "artifact": "artifact",
            "artifact_reuse": "artifact",
            "integrate": "final_solve",
        }
        by_tier: Dict[str, Counter] = defaultdict(Counter)
        by_domain: Dict[str, Counter] = defaultdict(Counter)
        for problem in all_problems:
            for item in problem.contribution_chain:
                ctype = contribution_type_map.get(item.get("type"))
                if ctype:
                    by_tier[str(problem.tier)][ctype] += 1
                    by_domain[problem.domain][ctype] += 1

        solved = [p for p in all_problems if p.solved]
        all_solo = [p for p in all_problems if p.resolution_mode == "solo"]
        all_collab = [p for p in all_problems if p.resolution_mode == "collaborative"]

        collab_share_over_time = []
        for snap in timeline:
            gen = snap.get("generation")
            gen_solved = [p for p in solved if p.generation == gen]
            collab_share_over_time.append(
                {
                    "generation": gen,
                    "share": round(sum(1 for p in gen_solved if p.resolution_mode == "collaborative") / max(1, len(gen_solved)), 4),
                }
            )

        def solve_rate_by_dim(block: List[Dict], field: str, keys: List[str]) -> Dict[str, float]:
            out = {}
            for key in keys:
                solved_count = sum((snap.get(field, {}).get(key) or {}).get("solved", 0) for snap in block)
                total_count = sum((snap.get(field, {}).get(key) or {}).get("total", 0) for snap in block)
                out[key] = round(solved_count / max(1, total_count), 4)
            return out

        def contribution_share(block: List[Dict]) -> Dict[str, float]:
            totals = Counter()
            for snap in block:
                outcomes = snap.get("problem_outcomes") or {}
                totals["final_solve"] += outcomes.get("solved", 0)
                totals["verification"] += outcomes.get("verified", 0)
                totals["subtask"] += outcomes.get("subtasks", 0)
                totals["decomposition"] += outcomes.get("decompositions", 0)
                totals["critique"] += outcomes.get("critiques", 0)
            denom = sum(totals.values())
            return {k: round(v / max(1, denom), 4) for k, v in totals.items()}

        first_50 = timeline[:50]
        last_50 = timeline[-50:]
        first_tier = solve_rate_by_dim(first_50, "problem_success_by_tier", ["1", "2", "3", "4"])
        last_tier = solve_rate_by_dim(last_50, "problem_success_by_tier", ["1", "2", "3", "4"])
        first_domain = solve_rate_by_dim(first_50, "problem_success_by_domain", list(DOMAINS))
        last_domain = solve_rate_by_dim(last_50, "problem_success_by_domain", list(DOMAINS))

        first_collab = round(mean(d["share"] for d in collab_share_over_time[:50]), 4) if collab_share_over_time else 0.0
        last_collab = round(mean(d["share"] for d in collab_share_over_time[-50:]), 4) if collab_share_over_time else 0.0
        first_contrib = contribution_share(first_50)
        last_contrib = contribution_share(last_50)

        plateau_dimensions = []
        for key, val in last_tier.items():
            if abs(val - first_tier.get(key, 0.0)) < 0.02:
                plateau_dimensions.append(f"tier_{key}")
        for key, val in last_domain.items():
            if abs(val - first_domain.get(key, 0.0)) < 0.02:
                plateau_dimensions.append(f"domain_{key}")
        if abs(last_collab - first_collab) < 0.02:
            plateau_dimensions.append("collaboration_share")
        for key, val in last_contrib.items():
            if abs(val - first_contrib.get(key, 0.0)) < 0.02:
                plateau_dimensions.append(f"contribution_{key}")

        return {
            "energy": {
                "histogram": histogram(energies),
                "stats": {
                    "min": round(min(energies), 3) if energies else 0.0,
                    "median": round(percentile(energies, 0.5), 3),
                    "mean": round(mean(energies), 3) if energies else 0.0,
                    "p90": round(percentile(energies, 0.9), 3),
                    "p99": round(percentile(energies, 0.99), 3),
                },
                "top_10_richest": sorted(
                    [{"agent_id": a.get("agent_id"), "energy": round(float(a.get("energy", 0.0)), 3)} for a in agents_snapshot],
                    key=lambda x: x["energy"],
                    reverse=True,
                )[:10],
            },
            "reproduction": {
                "births_by_generation": {str(s.get("generation")): s.get("births", 0) for s in timeline},
                "births_by_role": dict(self.role_reproduction_counts),
                "births_by_lineage": dict(lineage_counts),
                "births_per_agent": births_per_agent,
                "unique_reproducers": unique_reproducers,
                "repeat_reproducers": repeat_reproducers,
            },
            "contribution": {
                "reward_totals_by_type": {
                    "final_solve": round(totals.get("reward_final_solving", 0.0), 3),
                    "verification": round(totals.get("reward_verification", 0.0), 3),
                    "subtask": round(totals.get("reward_subtasks", 0.0), 3),
                    "decomposition": round(totals.get("reward_decomposition", 0.0), 3),
                    "critique": round(totals.get("reward_critique", 0.0), 3),
                    "artifact": round(totals.get("artifact_reuse", 0.0), 3),
                },
                "types_by_tier": {k: dict(v) for k, v in by_tier.items()},
                "types_by_domain": {k: dict(v) for k, v in by_domain.items()},
            },
            "collaboration": {
                "share_by_tier": {
                    str(t): round(
                        sum(1 for p in solved if p.tier == t and p.resolution_mode == "collaborative")
                        / max(1, sum(1 for p in solved if p.tier == t)),
                        4,
                    )
                    for t in [1, 2, 3, 4]
                },
                "share_by_domain": {
                    d: round(
                        sum(1 for p in solved if p.domain == d and p.resolution_mode == "collaborative")
                        / max(1, sum(1 for p in solved if p.domain == d)),
                        4,
                    )
                    for d in DOMAINS
                },
                "share_over_time": collab_share_over_time,
                "success_rate_solo": round(sum(1 for p in all_solo if p.solved) / max(1, len(all_solo)), 4),
                "success_rate_collaborative": round(sum(1 for p in all_collab if p.solved) / max(1, len(all_collab)), 4),
            },
            "diversity": {
                "diversity_over_time": [{"generation": s.get("generation"), "value": round(s.get("diversity_score", 0.0), 4)} for s in timeline],
                "lineage_count_over_time": [{"generation": s.get("generation"), "value": s.get("lineage_count", 0)} for s in timeline],
                "top_lineage_share_over_time": [
                    {
                        "generation": s.get("generation"),
                        "value": round((max((s.get("lineages") or {"_": 0}).values()) if s.get("lineages") else 0) / max(1, s.get("population", 0)), 4),
                    }
                    for s in timeline
                ],
            },
            "plateau": {
                "first_50": {
                    "solve_rate_by_tier": first_tier,
                    "solve_rate_by_domain": first_domain,
                    "collaboration_share": first_collab,
                    "contribution_type_share": first_contrib,
                },
                "last_50": {
                    "solve_rate_by_tier": last_tier,
                    "solve_rate_by_domain": last_domain,
                    "collaboration_share": last_collab,
                    "contribution_type_share": last_contrib,
                },
                "plateau_detected": bool(plateau_dimensions),
                "plateau_dimensions": sorted(set(plateau_dimensions)),
            },
        }

    def _build_report(self, timeline: List[Dict], all_problems: List[Problem], totals: Dict[str, int], agents_snapshot: List[Dict]) -> Dict:
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
            lineage_id = next((a.lineage_id for a in self.agents if a.agent_id == agent_id), None) or agent_id
            bucket = lineage_scores.setdefault(lineage_id, {"solves": 0, "verifications": 0, "meaningful_points": 0.0})
            bucket["solves"] += contrib.get("solves", 0)
            bucket["verifications"] += contrib.get("verifications", 0)
            bucket["meaningful_points"] += contrib.get("meaningful_points", 0.0)

        top_lineages = []
        for lineage_id, members in self._collect_lineages().items():
            metric = lineage_scores.get(lineage_id, {"solves": 0, "verifications": 0})
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
        warnings = []
        if solved == 0:
            warnings.append("No problems solved during run")
        if timeline and timeline[-1]["population"] <= 1:
            warnings.append("Population collapse risk")
        if avg_energy < 15:
            warnings.append("Average energy critically low")
        if timeline and mean(s.get("diversity_score", 0.0) for s in timeline[: min(50, len(timeline))]) < 0.3:
            warnings.append("Early diversity collapse risk")
        excessive_windows = 0
        if excessive_windows >= 2:
            warnings.append("Solver dominance persisted above threshold")

        role_reward_totals: Dict[str, float] = {role: 0.0 for role in ROLE_BUCKETS}
        for agent in self.agents:
            role_reward_totals[agent.choose_role()] += self.agent_contributions.get(agent.agent_id, {}).get("reward_earned", 0)

        reproduced_roles = sorted(self.reproduced_roles)

        solver_risk = self._compute_solver_dominance_risk(role_contribution_totals, role_based_fitness)
        ecosystem_diagnostics = self._calculate_ecosystem_diagnostics(timeline, role_contribution_totals)

        no_api = self._no_api_capability_report(timeline, all_problems, agents_snapshot)
        observability = self._observability_report(timeline, all_problems, agents_snapshot, totals)

        kind_counts = Counter((p.domain, p.tier) for p in all_problems if p.solved)
        problem_kinds = [
            {"domain": domain, "tier": tier, "solved": count}
            for (domain, tier), count in sorted(kind_counts.items(), key=lambda x: x[1], reverse=True)
        ][:8]

        unresolved = [
            {
                "problem_id": p.problem_id,
                "tier": p.tier,
                "domain": p.domain,
                "prompt_text": p.prompt_text,
                "status": "partial" if p.owner_id else "unsolved",
            }
            for p in all_problems
            if not p.solved
        ][:25]

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
            "ecosystem_diagnostics": ecosystem_diagnostics,
            "role_reproduction": {"roles_reproduced": reproduced_roles, "counts": dict(self.role_reproduction_counts)},
            "diversity_support": {
                "diversity_bonus": self.config.diversity_bonus,
                "diversity_min_lineages": self.config.diversity_min_lineages,
                "immigrant_injection_count": self.config.immigrant_injection_count,
                "immigrants_injected": totals.get("diversity_injected", 0),
            },
            "no_api_capability_ceiling": no_api,
            "observability": observability,
            "readable_summary": {
                "what_they_are_solving": problem_kinds,
                "which_remain_unsolved": unresolved,
                "are_harder_tiers_improving": no_api["improved_dimensions"],
                "collaboration_vs_solo": no_api["collaboration_effect"],
                "ecosystem": ecosystem_diagnostics,
                "role_reproduction": {"roles_reproduced": reproduced_roles, "counts": dict(self.role_reproduction_counts)},
                "evolution_before_plateau": {
                    "plateau_detected": no_api["plateau_detected"],
                    "plateau_dimensions": no_api["plateau_dimensions"],
                    "first_last_delta": round(
                        no_api["first_50_vs_last_50"]["last_50_success_rate"]
                        - no_api["first_50_vs_last_50"]["first_50_success_rate"],
                        4,
                    ),
                },
                "observability": {
                    "energy": observability.get("energy", {}).get("stats", {}),
                    "reproduction": {
                        "unique_reproducers": observability.get("reproduction", {}).get("unique_reproducers", 0),
                        "repeat_reproducers": observability.get("reproduction", {}).get("repeat_reproducers", 0),
                    },
                    "collaboration": {
                        "solo_success_rate": observability.get("collaboration", {}).get("success_rate_solo", 0.0),
                        "collaborative_success_rate": observability.get("collaboration", {}).get("success_rate_collaborative", 0.0),
                    },
                    "plateau_dimensions": observability.get("plateau", {}).get("plateau_dimensions", []),
                },
            },
            "warning_flags": warnings,
        }

    def _compute_solver_dominance_risk(self, role_contribution_totals: Dict[str, Dict[str, float]], role_based_fitness: Dict[str, float]) -> Dict:
        solver_points = role_contribution_totals.get("solver", {}).get("meaningful_points", 0.0)
        total_points = sum(v.get("meaningful_points", 0.0) for v in role_contribution_totals.values())
        return {
            "solver_share": round(solver_points / max(1.0, total_points), 4),
            "solver_fitness": round(role_based_fitness.get("solver", 0.0), 4),
        }

    def _generate_markdown_summary(self, result: Dict, report: Dict) -> str:
        ceiling = report.get("no_api_capability_ceiling", {})
        obs = report.get("observability", {})
        lines = [
            "# Genesis2 Run Summary",
            "",
            "## Configuration",
            f"- Seed: {self.config.seed}",
            f"- Starting agents: {self.config.agents}",
            f"- Generations: {self.config.generations}",
            f"- API Access: {'enabled' if self.config.api_access else 'disabled (no-API)'}",
            "",
            "## Outcomes",
            f"- Final population: {result['final_population']}",
            f"- Problems solved: {report['problems']['solved']}",
            f"- Problems unsolved: {report['problems']['unsolved']}",
            "",
            "## No-API Capability Ceiling",
            f"- Improved dimensions: {', '.join(ceiling.get('improved_dimensions', [])) or 'None'}",
            f"- Plateaued dimensions: {', '.join(ceiling.get('plateaued_dimensions', [])) or 'None'}",
            f"- Unsolved tiers: {', '.join(ceiling.get('unsolved_tiers', [])) or 'None'}",
            f"- Unsolved domains: {', '.join(ceiling.get('unsolved_domains', [])) or 'None'}",
            f"- Collaboration share: {ceiling.get('collaboration_effect', {}).get('collaborative_share', 0):.2%}",
            f"- Artifact assisted share: {ceiling.get('artifact_effect', {}).get('artifact_assisted_share', 0):.2%}",
            "",
            "## Observability Diagnostics",
            f"- Energy min/median/mean/p90/p99: {obs.get('energy', {}).get('stats', {}).get('min', 0):.2f} / {obs.get('energy', {}).get('stats', {}).get('median', 0):.2f} / {obs.get('energy', {}).get('stats', {}).get('mean', 0):.2f} / {obs.get('energy', {}).get('stats', {}).get('p90', 0):.2f} / {obs.get('energy', {}).get('stats', {}).get('p99', 0):.2f}",
            f"- Reproducers unique/repeat: {obs.get('reproduction', {}).get('unique_reproducers', 0)} / {obs.get('reproduction', {}).get('repeat_reproducers', 0)}",
            f"- Solo vs collaborative success: {obs.get('collaboration', {}).get('success_rate_solo', 0):.2%} vs {obs.get('collaboration', {}).get('success_rate_collaborative', 0):.2%}",
            f"- Plateau dimensions: {', '.join(obs.get('plateau', {}).get('plateau_dimensions', [])) or 'None'}",
            "",
            "## Sample Problem Prompts",
        ]
        for p in result.get("problems", [])[:5]:
            lines.append(f"- {p['problem_id']} ({p['domain']}/T{p['tier']}): {p['prompt_text']}")
        return "\n".join(lines) + "\n"

    def _sanitize_label(self, label: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", label.strip())
        cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
        return cleaned or "run"

    def _build_run_identity(self) -> Dict[str, str]:
        label = self._sanitize_label(self.config.run_label) if self.config.run_label else ""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short_id = uuid.uuid4().hex[:8]
        run_id = f"{timestamp}__{short_id}"
        run_folder_name = f"{label}__{run_id}" if label else run_id
        return {
            "label": label,
            "timestamp": timestamp,
            "short_id": short_id,
            "run_id": run_id,
            "run_folder_name": run_folder_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def _create_run_dir(self) -> Dict[str, object]:
        run_root = Path(self.config.log_dir)
        attempts = 0
        while attempts < 10:
            run_identity = self._build_run_identity()
            run_dir = run_root / run_identity["run_folder_name"]
            if run_dir.exists() and not self.config.overwrite:
                attempts += 1
                continue
            run_dir.mkdir(parents=True, exist_ok=self.config.overwrite)
            return {
                "run_dir": run_dir,
                "run_identity": run_identity,
            }
        raise FileExistsError("Unable to allocate unique run directory without overwrite")

    def run(self, progress_callback=None) -> Dict:
        run_setup = self._create_run_dir()
        run_dir = run_setup["run_dir"]
        run_identity = run_setup["run_identity"]
        logger = SimulationLogger(run_dir)
        config_snapshot = asdict(self.config)
        (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")
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
            "diversity_injected": 0,
        }

        tier_mix = self.config.tier_mix if self.config.tier_mix else {"1": 0.35, "2": 0.30, "3": 0.20, "4": 0.15}
        cumulative_unique_reproducers: set[str] = set()
        cumulative_repeat_reproducers: set[str] = set()
        phase_first_generation: Dict[str, int] = {}
        detected_phases: List[Dict[str, object]] = []

        for generation in range(1, self.config.generations + 1):
            births = deaths = artifact_reuse = artifact_created = solved = verified = subtasks = 0
            critiques = decompositions = integrations = 0
            generation_points = defaultdict(float)
            births_by_agent: Counter[str] = Counter()
            generation_contribution_points = {
                a.agent_id: self.agent_contributions.get(a.agent_id, {}).get("meaningful_points", 0.0)
                for a in self.agents
            }
            tier_solved = Counter()
            tier_total = Counter()
            domain_solved = Counter()
            domain_total = Counter()

            board = WorldBoard(problems=spawn_problems(generation, self.config.tasks_per_generation, tier_mix=tier_mix))
            all_problems.extend(board.problems)
            for problem in board.problems:
                tier_total[str(problem.tier)] += 1
                domain_total[problem.domain] += 1

            current_lineages = len({a.lineage_id for a in self.agents})
            if current_lineages >= self.config.diversity_min_lineages and self.config.diversity_bonus > 0:
                for agent in self.agents:
                    agent.energy += self.config.diversity_bonus

            for agent in sorted(self.agents, key=lambda a: a.energy, reverse=True):
                open_tasks = board.open_problems()
                if not open_tasks:
                    break

                best = max(open_tasks, key=lambda p: agent.bid_score(p.domain, p.tier))
                if not board.claim(best.problem_id, agent.agent_id):
                    continue
                best.owner_id = agent.agent_id
                if agent.agent_id not in best.agents_involved:
                    best.agents_involved.append(agent.agent_id)
                best.contribution_chain.append({"generation": str(generation), "agent_id": agent.agent_id, "type": "claim", "detail": f"{agent.agent_id} claimed problem"})

                chain = self._run_problem_ecology(best, generation)

                board_events.append(
                    {
                        "generation": generation,
                        "message_type": "contribution_chain",
                        "agent_id": chain.get("integrator_id"),
                        "problem_id": best.problem_id,
                        "tier": best.tier,
                        "domain": best.domain,
                        "detail": f"chain {best.problem_id}: {', '.join(chain['attribution_log'])}",
                        "attribution": chain,
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

                critic = None
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
                    best.solved_generation = generation
                    solved += 1
                    totals["solved"] += 1
                    tier_solved[str(best.tier)] += 1
                    domain_solved[best.domain] += 1
                    agent.energy += REWARDS["correct_solution"]
                    self.agent_contributions[agent.agent_id]["solves"] += 1
                    best.contribution_chain.append({"generation": str(generation), "agent_id": agent.agent_id, "type": "solve", "detail": f"{agent.agent_id} provided accepted solve"})
                    best.reward_split[agent.agent_id] = round(REWARDS["correct_solution"], 2)

                    # Optional collaboration: a second contributor can verify/assist on harder tiers.
                    collaborators = [a for a in self.agents if a.agent_id != agent.agent_id]
                    if collaborators and best.tier >= 3 and random.random() < 0.35:
                        helper = max(collaborators, key=lambda c: c.bid_score(best.domain, best.tier))
                        helper_share = round(REWARDS["correct_solution"] * 0.35, 2)
                        agent_share = round(REWARDS["correct_solution"] - helper_share, 2)
                        best.reward_split[agent.agent_id] = agent_share
                        best.reward_split[helper.agent_id] = helper_share
                        best.resolution_mode = "collaborative"
                        if helper.agent_id not in best.agents_involved:
                            best.agents_involved.append(helper.agent_id)
                        best.contribution_chain.append(
                            {
                                "generation": str(generation),
                                "agent_id": helper.agent_id,
                                "type": "collaboration",
                                "detail": f"{helper.agent_id} contributed validating steps used in final solve",
                            }
                        )
                        helper.energy += helper_share

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

                    if best.tier >= agent.genome.thresholds["verify_tier_gte"]:
                        verifier = max(self.agents, key=lambda a: a.genome.specialization.get("logic", 0.0)) if self.agents else agent
                        verifier.energy -= COSTS["verify_attempt"]
                        if random.random() < verifier.genome.specialization.get("logic", 0.4):
                            best.verified = True
                            verified += 1
                            totals["verified"] += 1
                            verifier.energy += REWARDS["successful_verification"]
                            self.agent_contributions[verifier.agent_id]["verifications"] += 1
                            if verifier.agent_id not in best.agents_involved:
                                best.agents_involved.append(verifier.agent_id)
                            best.contribution_chain.append(
                                {
                                    "generation": str(generation),
                                    "agent_id": verifier.agent_id,
                                    "type": "verification",
                                    "detail": f"{verifier.agent_id} verified the accepted solve",
                                }
                            )
                            best.reward_split[verifier.agent_id] = round(
                                best.reward_split.get(verifier.agent_id, 0.0) + REWARDS["successful_verification"], 2
                            )
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
                    if random.random() < 0.25:
                        catcher = critic if critic else agent
                        agent.energy += REWARDS["catch_incorrect"]
                        best.contribution_chain.append({"generation": str(generation), "agent_id": agent.agent_id, "type": "catch_incorrect", "detail": f"{agent.agent_id} flagged incorrect work"})
                        board_events.append(
                            {
                                "generation": generation,
                                "message_type": "catch_incorrect",
                                "agent_id": catcher.agent_id,
                                "problem_id": best.problem_id,
                                "tier": best.tier,
                                "domain": best.domain,
                                "detail": f"{catcher.agent_id} caught incorrect attempt on {best.problem_id}",
                            }
                        )

                if agent.artifact_store and random.random() < agent.genome.strategy["artifact_reuse_bias"]:
                    agent.energy += REWARDS["artifact_reuse"]
                    artifact_reuse += 1
                    totals["artifact_reuse"] += 1
                    self.agent_contributions[agent.agent_id]["artifacts_reused"] += 1
                    reused_artifact_id = agent.artifact_store[0]
                    if reused_artifact_id in self.artifact_registry:
                        meta = self.artifact_registry[reused_artifact_id]
                        meta["times_reused"] = int(meta.get("times_reused", 0)) + 1
                        meta.setdefault("reuse_generations", []).append(generation)
                        meta.setdefault("reuser_agent_ids", []).append(agent.agent_id)
                        meta.setdefault("reuser_lineage_ids", []).append(agent.lineage_id)
                        meta["reused_in_successful_solve"] = bool(meta.get("reused_in_successful_solve")) or bool(best.solved)
                        meta["reused_in_collaborative_solve"] = bool(meta.get("reused_in_collaborative_solve")) or bool(best.resolution_mode == "collaborative")
                    best.contribution_chain.append({"generation": str(generation), "agent_id": agent.agent_id, "type": "artifact_reuse", "detail": f"{agent.agent_id} reused artifact on this problem"})

                if chain.get("integrator_id"):
                    integ = next((a for a in self.agents if a.agent_id == chain["integrator_id"]), None)
                    created_artifact = integ.maybe_create_artifact(problem.domain) if integ else None
                    if created_artifact:
                        artifact_created += 1
                        totals["artifact_created"] += 1
                        self.agent_contributions[integ.agent_id]["artifacts_created"] += 1
                        self.artifact_registry[created_artifact] = {
                            "artifact_id": created_artifact,
                            "creation_generation": generation,
                            "creator_agent_id": integ.agent_id,
                            "creator_lineage_id": integ.lineage_id,
                            "creator_role": integ.choose_role(),
                            "artifact_type": problem.domain,
                            "times_reused": 0,
                            "reuse_generations": [],
                            "reuser_agent_ids": [],
                            "reuser_lineage_ids": [],
                            "reused_in_successful_solve": False,
                            "reused_in_collaborative_solve": False,
                        }

            offspring: List[Agent] = []
            for agent in self.agents:
                agent.energy -= self.config.upkeep_cost
                agent.generation_age += 1
                if (
                    agent.energy >= self.config.reproduction_threshold
                    and agent.generation_age >= 2
                    and generation_points.get(agent.agent_id, 0.0) >= REWARDS["useful_subtask"] * 0.8
                ):
                    child = reproduce(agent, self._next_agent_id, generation, mutation_rate=self.config.mutation_rate)
                    offspring.append(child)
                    self._next_agent_id += 1
                    self._register_agent(child.agent_id)
                    self.agent_contributions[agent.agent_id]["offspring"] += 1
                    life = self.agent_lifecycle.setdefault(agent.agent_id, {})
                    life["lifetime_births"] = int(life.get("lifetime_births", 0)) + 1
                    if life.get("first_reproduction_generation") is None:
                        life["first_reproduction_generation"] = generation
                    life["last_reproduction_generation"] = generation
                    role = agent.choose_role()
                    self.reproduced_roles.add(role)
                    self.role_reproduction_counts[role] += 1
                    births += 1
                    births_by_agent[agent.agent_id] += 1
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
            lineage_count = len({a.lineage_id for a in self.agents})
            if lineage_count < self.config.diversity_min_lineages:
                injected = self._inject_immigrants(generation, reason="lineage_collapse")
                if injected:
                    births += injected
                    totals["diversity_injected"] += injected
            lineages = Counter(a.lineage_id for a in self.agents)
            energies = [a.energy for a in self.agents]
            roles = Counter(a.choose_role() for a in self.agents)
            role_distribution = {role: round(roles.get(role, 0) / max(1, len(self.agents)), 3) for role in ROLE_BUCKETS}

            current_agents = [a.snapshot() for a in self.agents]
            energies = [float(a.get("energy", 0.0)) for a in current_agents]
            energies_sorted = sorted(energies)
            lineage_totals = lineage_energy(current_agents)
            role_totals = role_energy(current_agents)
            total_energy = sum(energies_sorted)
            top_agents = sorted(current_agents, key=lambda a: float(a.get("energy", 0.0)), reverse=True)[:10]

            births_total_generation = sum(births_by_agent.values())
            unique_reproducers_generation = len(births_by_agent)
            repeat_reproducers_generation = sum(1 for c in births_by_agent.values() if c > 1)
            cumulative_unique_reproducers.update(births_by_agent.keys())
            cumulative_repeat_reproducers.update(aid for aid, count in births_by_agent.items() if count > 1)

            births_by_role = Counter()
            births_by_lineage = Counter()
            for aid, count in births_by_agent.items():
                parent = next((agent for agent in self.agents if agent.agent_id == aid), None)
                if parent:
                    births_by_role[parent.choose_role()] += count
                    births_by_lineage[parent.lineage_id] += count

            births_shares = sorted(births_by_agent.values(), reverse=True)
            share_top1 = round((births_shares[0] / births_total_generation), 6) if births_shares and births_total_generation else 0.0
            share_top5 = round((sum(births_shares[:5]) / births_total_generation), 6) if births_shares and births_total_generation else 0.0
            share_top10 = round((sum(births_shares[:10]) / births_total_generation), 6) if births_shares and births_total_generation else 0.0

            logger.log_stream("generation_metrics", {
                "schema_version": "1.0",
                "generation": generation,
                "population": len(self.agents),
                "total_energy": round(total_energy, 4),
                "energy_min": round(min(energies_sorted), 4) if energies_sorted else 0.0,
                "energy_median": round(percentile(energies_sorted, 0.5), 4),
                "energy_mean": round(sum(energies_sorted) / max(1, len(energies_sorted)), 4),
                "energy_p90": round(percentile(energies_sorted, 0.9), 4),
                "energy_p95": round(percentile(energies_sorted, 0.95), 4),
                "energy_p99": round(percentile(energies_sorted, 0.99), 4),
                "energy_max": round(max(energies_sorted), 4) if energies_sorted else 0.0,
                "top_1pct_energy_share": top_share(energies_sorted, 0.01),
                "top_10pct_energy_share": top_share(energies_sorted, 0.10),
                "energy_gini": gini(energies_sorted),
                "top_agents": [{"agent_id": a["agent_id"], "lineage_id": a["lineage_id"], "role": a["role"], "energy": a["energy"]} for a in top_agents],
                "births": births,
                "deaths": deaths,
                "unique_reproducers_generation": unique_reproducers_generation,
                "repeat_reproducers_generation": repeat_reproducers_generation,
                "cumulative_unique_reproducers": len(cumulative_unique_reproducers),
                "cumulative_repeat_reproducers": len(cumulative_repeat_reproducers),
                "birth_share_top_1": share_top1,
                "birth_share_top_5": share_top5,
                "birth_share_top_10": share_top10,
                "avg_births_per_reproducer": round(births_total_generation / max(1, unique_reproducers_generation), 6),
                "artifact_created": artifact_created,
                "artifact_reused": artifact_reuse,
                "collaboration_share_generation": round(sum(1 for p in board.problems if p.resolution_mode == "collaborative" and p.solved) / max(1, sum(1 for p in board.problems if p.solved)), 6),
                "lineage_energy": lineage_totals,
                "role_energy": role_totals,
            })

            for lineage_id, total in lineage_totals.items():
                logger.log_stream("lineage_metrics", {
                    "schema_version": "1.0",
                    "generation": generation,
                    "lineage_id": lineage_id,
                    "total_energy": round(total, 4),
                    "energy_share": round(total / max(1e-9, total_energy), 6),
                    "births": births_by_lineage.get(lineage_id, 0),
                    "lineage_size": int(lineages.get(lineage_id, 0)),
                })

            for role_name, role_bucket in role_totals.items():
                logger.log_stream("role_metrics", {
                    "schema_version": "1.0",
                    "generation": generation,
                    "role": role_name,
                    "total_energy": role_bucket["total_energy"],
                    "median_energy": role_bucket["median_energy"],
                    "mean_energy": role_bucket["mean_energy"],
                    "births": births_by_role.get(role_name, 0),
                })

            for problem in board.problems:
                participants = list(dict.fromkeys(problem.agents_involved))
                logger.log_stream("problem_metrics", {
                    "schema_version": "1.0",
                    "problem_id": problem.problem_id,
                    "generation": problem.generation,
                    "tier": problem.tier,
                    "domain": problem.domain,
                    "solved": problem.solved,
                    "collaboration_mode": problem.resolution_mode,
                    "participant_count": len(participants),
                    "participating_agent_ids": participants,
                    "participating_lineage_ids": sorted({next((a.lineage_id for a in self.agents if a.agent_id == aid), aid) for aid in participants}),
                    "artifact_assisted": any(step.get("type") == "artifact_reuse" for step in problem.contribution_chain),
                    "solve_duration": None if not problem.solved_generation else problem.solved_generation - problem.generation,
                    "final_reward_amount": round(sum(problem.reward_split.values()), 4),
                    "reward_split": problem.reward_split,
                    "participant_roles": {aid: next((a.choose_role() for a in self.agents if a.agent_id == aid), "unknown") for aid in participants},
                    "contribution_chain_sequence": [step.get("type") for step in problem.contribution_chain],
                })

            window = snapshots[-self.config.diagnostics_window + 1 :] + [{
                "population": len(self.agents),
                "births": births,
                "deaths": deaths,
                "diversity_score": self._diversity_score(),
                "collaboration_share": round(sum(1 for p in board.problems if p.resolution_mode == "collaborative" and p.solved) / max(1, sum(1 for p in board.problems if p.solved)), 6),
                "median_energy": round(percentile(energies_sorted, 0.5), 4),
            }]
            phase = detect_phase(
                [w.get("population", 0) for w in window],
                [w.get("births", 0) for w in window],
                [w.get("deaths", 0) for w in window],
                [w.get("median_energy", 0.0) for w in window],
                [w.get("diversity_score", 0.0) for w in window],
                [w.get("collaboration_share", 0.0) for w in window],
            )
            detected_phases.append({"generation": generation, **phase})
            if phase["phase"] not in phase_first_generation:
                phase_first_generation[phase["phase"]] = generation

            for aid, starting_total in generation_contribution_points.items():
                prev = self.agent_contribution_history.setdefault(aid, [])
                delta = self.agent_contributions[aid]["meaningful_points"] - starting_total
                prev.append(delta)

            for agent in self.agents:
                self.agent_energy_history.setdefault(agent.agent_id, []).append({"generation": generation, "energy": round(agent.energy, 3)})

            generation_log = {
                "generation": generation,
                "population": len(self.agents),
                "births": births,
                "deaths": deaths,
                "energy_distribution": {**summarize_energy(energies), "median": round(percentile(energies, 0.5), 4), "p90": round(percentile(energies, 0.9), 4), "p99": round(percentile(energies, 0.99), 4)},
                "collaboration_share": round(sum(1 for p in board.problems if p.resolution_mode == "collaborative" and p.solved) / max(1, sum(1 for p in board.problems if p.solved)), 6),
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
                "role_distribution": role_distribution,
                "diversity_score": self._diversity_score(),
                "lineage_count": len(lineages),
                "problem_success_by_tier": {
                    tier: {"solved": tier_solved.get(tier, 0), "total": tier_total.get(tier, 0)}
                    for tier in ["1", "2", "3", "4"]
                },
                "problem_success_by_domain": {
                    domain: {"solved": domain_solved.get(domain, 0), "total": domain_total.get(domain, 0)}
                    for domain in DOMAINS
                },
            }
            logger.log_generation(generation_log)
            snapshots.append(generation_log)

            if progress_callback:
                progress_callback(generation_log)
            if not self.agents:
                break

        for artifact in self.artifact_registry.values():
            logger.log_stream("artifact_metrics", {"schema_version": "1.0", **artifact})

        summary_path = logger.finalize()
        lineage_members = self._collect_lineages()

        agents_snapshot = [a.snapshot() for a in self.agents]
        for agent in agents_snapshot:
            agent_id = agent["agent_id"]
            agent["energy_history"] = self.agent_energy_history.get(agent_id, [])
            agent["contributions"] = self.agent_contributions.get(agent_id, {})
            life = dict(self.agent_lifecycle.get(agent_id, {}))
            life["lifetime_contribution_score"] = round(float(agent["contributions"].get("meaningful_score", 0.0)), 4)
            life["lifespan_generations"] = self.config.generations - int(agent.get("generation_born", 0))
            agent["lifecycle"] = life
            agent["offspring"] = [a["agent_id"] for a in agents_snapshot if a.get("parent_id") == agent_id]
            agent["workflow"] = agent["genome"].get("workflows", {})

        report = self._build_report(snapshots, all_problems, totals, agents_snapshot)
        contribution_chains = [
            {
                "problem_id": p.problem_id,
                "chain": p.contribution_chain,
            }
            for p in all_problems
        ]
        result = {
            "summary_path": str(summary_path),
            "run_id": run_identity["run_id"],
            "run_dir": str(run_dir),
            "run_label": run_identity["label"],
            "started_at": run_identity["started_at"],
            "final_population": len(self.agents),
            "agents": agents_snapshot,
            "timeline": snapshots,
            "board_messages": board_events,
            "totals": totals,
            "contribution_chains": contribution_chains,
            "report": report,
            "lineages": lineage_members,
            "config": config_snapshot,
            "phase_diagnostics": {
                "by_generation": detected_phases,
                "first_generation": phase_first_generation,
                "peak_population_generation": max(snapshots, key=lambda s: s.get("population", 0)).get("generation", 0) if snapshots else 0,
                "stabilization_start_generation": phase_first_generation.get("stabilization"),
            },
            "problems": [
                {
                    "problem_id": p.problem_id,
                    "generation": p.generation,
                    "domain": p.domain,
                    "tier": p.tier,
                    "prompt_text": p.prompt_text,
                    "solved": p.solved,
                    "verified": p.verified,
                    "owner_id": p.owner_id,
                    "status": "solved" if p.solved else ("partial" if p.owner_id else "unsolved"),
                    "resolution_mode": p.resolution_mode,
                    "time_to_solve": None if not p.solved_generation else p.solved_generation - p.generation,
                    "agents_involved": p.agents_involved,
                    "contribution_chain": p.contribution_chain,
                    "reward_split": p.reward_split,
                    "payout_attribution": p.reward_split,
                    "final_outcome": "verified_solution" if p.solved and p.verified else ("solved" if p.solved else "open"),
                }
                for p in all_problems
            ],
        }

        markdown_path = Path(summary_path).with_name("run_summary.md")
        markdown_path.write_text(self._generate_markdown_summary(result, report), encoding="utf-8")
        result["markdown_summary_path"] = str(markdown_path)

        finished_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "run_id": run_identity["run_id"],
            "label": run_identity["label"],
            "started_at": run_identity["started_at"],
            "finished_at": finished_at,
            "config_snapshot": config_snapshot,
            "files": sorted([path.name for path in run_dir.iterdir() if path.is_file()] + ["run_manifest.json"]),
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result["finished_at"] = finished_at
        result["manifest_path"] = str(run_dir / "run_manifest.json")
        return result


def load_config(path: str | Path) -> SimulationConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SimulationConfig(**data)
