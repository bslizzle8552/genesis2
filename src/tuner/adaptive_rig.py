from __future__ import annotations

from dataclasses import asdict
from src.engine.simulation import load_config
from pathlib import Path
from typing import Any, Dict, List

TARGET_GENERATIONS = 100
TARGET_POPULATION = 100
FIND_STABLE_SWARM_GOAL_PROFILE: Dict[str, Any] = {
    "starting_agents": 25,
    "generations": TARGET_GENERATIONS,
    "target_population": TARGET_POPULATION,
    "target_reach_generation": 80,
    "stability_band": (90, 110),
}


def build_find_stable_swarm_baseline(default_config_path: Path) -> Dict[str, Any]:
    cfg = load_config(default_config_path)
    params = asdict(cfg)
    params.update(
        {
            "preset_name": "adaptive_tuning_rig_find_stable_swarm",
            "agents": FIND_STABLE_SWARM_GOAL_PROFILE["starting_agents"],
            "generations": FIND_STABLE_SWARM_GOAL_PROFILE["generations"],
            "initial_energy": 100,
            "reproduction_threshold": 130,
            "mutation_rate": 0.28,
            "upkeep_cost": 6,
            "tasks_per_generation": 60,
            "diversity_bonus": 0.175,
            "diversity_min_lineages": 14,
            "anti_dominance_enabled": False,
            "diminishing_reward_enabled": False,
            "lineage_size_penalty_enabled": False,
            "lineage_energy_share_penalty_enabled": False,
            "reproduction_cooldown_enabled": False,
            "overwrite": False,
        }
    )
    return params


def score_and_label_run(run_result: Dict[str, Any]) -> Dict[str, Any]:
    goal = FIND_STABLE_SWARM_GOAL_PROFILE
    timeline = list(run_result.get("timeline", []))
    latest = timeline[-1] if timeline else {}
    final_population = int(run_result.get("final_population", 0))
    lineage_count = int(latest.get("lineage_count", 0))
    diversity_score = float(latest.get("diversity_score", 0.0))
    top_lineage_share = float(
        latest.get("dominance_metrics", {}).get(
            "top_lineage_energy_share", latest.get("top_lineage_energy_share", 1.0)
        )
    )
    top_3_lineage_share = float(
        latest.get("dominance_metrics", {}).get(
            "top_3_lineage_energy_share", latest.get("top_3_lineage_energy_share", 1.0)
        )
    )
    populations = [int(s.get("population", 0)) for s in timeline]
    late = populations[-20:] if populations else []
    late_std = 0.0
    if len(late) > 1:
        avg = sum(late) / len(late)
        late_std = (sum((p - avg) ** 2 for p in late) / len(late)) ** 0.5

    survives_full = len(timeline) >= int(goal["generations"])
    reaches_target_by_g80 = any(goal["stability_band"][0] <= p <= goal["stability_band"][1] for p in populations[: int(goal["target_reach_generation"])]) if populations else False
    first_population = populations[0] if populations else 0
    starts_near_target = first_population >= int(goal["stability_band"][0])
    starts_from_expected_range = 20 <= first_population <= 30
    late_range = (max(late) - min(late)) if late else 0
    severe_instability = late_std > 18.0 or late_range > 40
    overshoot = max(populations, default=0) > 130
    collapse = min(populations, default=0) < 45

    growth_end = min(len(populations), int(goal["target_reach_generation"]))
    growth_slice = populations[:growth_end]
    expected_growth = [
        goal["starting_agents"]
        + ((goal["target_population"] - goal["starting_agents"]) * ((idx + 1) / float(goal["target_reach_generation"])))
        for idx in range(growth_end)
    ]
    growth_error = (
        sum(abs(float(actual) - float(expected)) for actual, expected in zip(growth_slice, expected_growth)) / len(growth_slice)
        if growth_slice
        else float(goal["target_population"])
    )
    growth_component = max(0.0, 30.0 - (growth_error * 0.6))
    reach_component = 0.0
    if populations:
        reach_gen = next(
            (
                idx + 1
                for idx, pop in enumerate(populations)
                if goal["stability_band"][0] <= pop <= goal["stability_band"][1]
            ),
            None,
        )
        if reach_gen is not None:
            reach_component = max(0.0, 25.0 - abs(reach_gen - int(goal["target_reach_generation"])) * 0.8)

    gate_results = {
        "final_population_max_ok": final_population <= 130,
        "final_population_min_ok": final_population >= 60,
        "lineage_count_ok": lineage_count >= 10,
        "top_lineage_share_ok": top_lineage_share <= 0.6,
        "top_3_lineage_share_ok": top_3_lineage_share <= 0.8,
        "reaches_target_by_generation_80_ok": reaches_target_by_g80,
        "late_stability_ok": not severe_instability,
        "starts_near_25_ok": starts_from_expected_range,
    }
    passed_gates = all(gate_results.values()) and survives_full

    population_component = max(0.0, 15.0 - (abs(final_population - TARGET_POPULATION) * 0.75))
    lineage_component = min(10.0, max(0.0, (lineage_count - 10) * 1.2))
    diversity_component = min(10.0, max(0.0, (diversity_score - 0.12) * 90.0))
    stability_component = max(0.0, 10.0 - (late_std * 0.4))
    score = growth_component + reach_component + population_component + lineage_component + diversity_component + stability_component

    if starts_near_target:
        score -= 20.0
    if overshoot:
        score -= min(22.0, (max(populations) - 130) * 0.4)
    if collapse:
        score -= min(22.0, (45 - min(populations)) * 0.8)
    if severe_instability:
        score -= 12.0
    if not passed_gates:
        score -= 10.0
    score = max(0.0, min(100.0, score))

    healthy = (
        passed_gates
        and 80 <= final_population <= 120
        and lineage_count >= 14
        and diversity_score >= 0.15
        and late_std <= 12.0
    )
    near_healthy = (
        passed_gates
        and 90 <= final_population <= 110
        and lineage_count >= 12
        and diversity_score >= 0.15
        and late_std <= 18.0
    )

    if healthy:
        label = "healthy"
    elif near_healthy:
        label = "near_healthy"
    elif final_population > 130 or overshoot:
        label = "failed_overshoot"
    elif final_population < 50 or not survives_full:
        label = "failed_collapse"
    elif starts_near_target:
        label = "failed_wrong_start"
    elif top_lineage_share > 0.6 or top_3_lineage_share > 0.8:
        label = "failed_dominance"
    elif lineage_count < 10 or diversity_score < 0.15:
        label = "failed_low_diversity"
    elif not reaches_target_by_g80 or severe_instability:
        label = "failed_instability"
    else:
        label = "failed_low_diversity"

    return {
        "score": round(score, 4),
        "label": label,
        "survives_full": survives_full,
        "final_population": final_population,
        "lineage_count": lineage_count,
        "diversity_score": round(diversity_score, 6),
        "top_lineage_share": round(top_lineage_share, 6),
        "top_3_lineage_share": round(top_3_lineage_share, 6),
        "late_population_std": round(late_std, 6),
        "passed_hard_gates": passed_gates,
        "hard_gates": gate_results,
        "healthy": healthy,
        "near_healthy": near_healthy,
        "goal_profile": goal,
        "diagnosis": classify_failure_modes(run_result, {
            "final_population": final_population,
            "lineage_count": lineage_count,
            "diversity_score": diversity_score,
            "top_lineage_share": top_lineage_share,
            "late_population_std": late_std,
            "starts_near_target": starts_near_target,
        }),
    }


def classify_failure_modes(run_result: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
    timeline = list(run_result.get("timeline", []))
    populations = [int(s.get("population", 0)) for s in timeline]
    diagnoses: List[str] = []
    if max(populations, default=0) > 130:
        diagnoses.append("overshoot")
    if min(populations, default=999) < 45:
        diagnoses.append("collapse")
    if float(metrics.get("top_lineage_share", 0.0)) > 0.70:
        generation_count = len(populations)
        first_dominance = next(
            (
                idx
                for idx, step in enumerate(timeline, start=1)
                if float(step.get("dominance_metrics", {}).get("top_lineage_energy_share", step.get("top_lineage_energy_share", 0.0))) > 0.70
            ),
            generation_count,
        )
        if first_dominance <= max(10, generation_count // 3):
            diagnoses.append("early_dominance")
        else:
            diagnoses.append("late_dominance")
    if int(metrics.get("lineage_count", 0)) < 10 or float(metrics.get("diversity_score", 0.0)) < 0.15:
        diagnoses.append("low_diversity")
    if float(metrics.get("late_population_std", 0.0)) > 15.0:
        diagnoses.append("instability")
    if not timeline or float(run_result.get("tasks_solved", 0.0)) <= 0.0:
        diagnoses.append("weak_throughput")
    if metrics.get("starts_near_target"):
        diagnoses.append("wrong_start")
    return diagnoses


def adjust_parameters(params: Dict[str, Any], metrics: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    updated = dict(params)
    label = metrics.get("label")
    diagnoses = set(metrics.get("diagnosis", []))
    reason = "small mutation sweep"
    is_failed = isinstance(label, str) and label.startswith("failed_")

    if "wrong_start" in diagnoses:
        updated["agents"] = FIND_STABLE_SWARM_GOAL_PROFILE["starting_agents"]
        reason = "critical correction: force stable-swarm start population to 25"
    if label == "failed_collapse" or "collapse" in diagnoses:
        updated["initial_energy"] = min(200, int(updated.get("initial_energy", 100)) + 12)
        updated["reproduction_threshold"] = max(90, int(updated.get("reproduction_threshold", 130)) - 8)
        updated["upkeep_cost"] = max(2, int(updated.get("upkeep_cost", 6)) - 2)
        updated["tasks_per_generation"] = max(20, int(updated.get("tasks_per_generation", 60)) - 8)
        reason = "strong negative signal (collapse): sharply raise survivability and reduce pressure"
    elif label == "failed_overshoot" or "overshoot" in diagnoses:
        updated["upkeep_cost"] = min(14, int(updated.get("upkeep_cost", 6)) + 2)
        updated["reproduction_threshold"] = min(220, int(updated.get("reproduction_threshold", 130)) + 10)
        updated["reproduction_cooldown_enabled"] = True
        updated["reproduction_cooldown_generations"] = min(8, int(updated.get("reproduction_cooldown_generations", 2)) + 2)
        reason = "strong negative signal (overshoot): tighten reproduction aggressively"
    elif label in {"failed_low_diversity", "failed_dominance"} or {"low_diversity", "early_dominance", "late_dominance"}.intersection(diagnoses):
        updated["diversity_bonus"] = min(0.7, float(updated.get("diversity_bonus", 0.175)) + 0.05)
        updated["diversity_min_lineages"] = min(28, int(updated.get("diversity_min_lineages", 14)) + 2)
        updated["mutation_rate"] = min(0.5, float(updated.get("mutation_rate", 0.28)) + 0.03)
        updated["anti_dominance_enabled"] = True
        updated["lineage_size_penalty_enabled"] = True
        updated["lineage_energy_share_penalty_enabled"] = True
        reason = "strong negative signal (diversity/dominance): push anti-dominance and lineage spread"
    elif label == "failed_instability" or "instability" in diagnoses:
        updated["mutation_rate"] = max(0.1, float(updated.get("mutation_rate", 0.28)) - 0.05)
        updated["tasks_per_generation"] = max(20, int(updated.get("tasks_per_generation", 60)) - 8)
        updated["reproduction_cooldown_enabled"] = True
        reason = "strong negative signal (instability): lower volatility and smooth growth"
    elif is_failed:
        updated["mutation_rate"] = max(0.1, float(updated.get("mutation_rate", 0.28)) - 0.02)
        reason = "strong negative failed signal: move away from current region"

    if "weak_throughput" in diagnoses:
        updated["tasks_per_generation"] = min(120, int(updated.get("tasks_per_generation", 60)) + 6)
        updated["reproduction_threshold"] = max(90, int(updated.get("reproduction_threshold", 130)) - 4)
        reason = f"{reason}; throughput recovery tuning"

    return updated, reason


def canonical_config_signature(params: Dict[str, Any]) -> tuple:
    keys = [
        "agents",
        "initial_energy",
        "reproduction_threshold",
        "upkeep_cost",
        "reproduction_cooldown_enabled",
        "reproduction_cooldown_generations",
        "mutation_rate",
        "tasks_per_generation",
        "diversity_bonus",
        "diversity_min_lineages",
        "anti_dominance_enabled",
        "diminishing_reward_enabled",
        "lineage_size_penalty_enabled",
        "lineage_energy_share_penalty_enabled",
    ]
    return tuple((k, params.get(k)) for k in keys)


def generate_local_variants(base: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for idx in range(count):
        direction = -1 if idx % 2 == 0 else 1
        step = (idx // 2) + 1
        candidate = dict(base)
        candidate["mutation_rate"] = min(0.5, max(0.05, float(candidate.get("mutation_rate", 0.28)) + (0.01 * step * direction)))
        candidate["tasks_per_generation"] = max(20, min(120, int(candidate.get("tasks_per_generation", 60)) + (3 * step * direction)))
        candidate["diversity_bonus"] = max(0.05, min(0.8, float(candidate.get("diversity_bonus", 0.175)) + (0.02 * step * direction)))
        candidate["diversity_min_lineages"] = max(8, min(28, int(candidate.get("diversity_min_lineages", 14)) + (step * direction)))
        candidate["upkeep_cost"] = max(2, min(14, int(candidate.get("upkeep_cost", 6)) - direction))
        candidate["reproduction_threshold"] = max(90, min(220, int(candidate.get("reproduction_threshold", 130)) + (4 * step * direction)))
        variants.append(candidate)
    return variants
