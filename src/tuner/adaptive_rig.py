from __future__ import annotations

from dataclasses import asdict
from src.engine.simulation import load_config
from pathlib import Path
from typing import Any, Dict

TARGET_GENERATIONS = 100
TARGET_POPULATION = 100


def build_find_stable_swarm_baseline(default_config_path: Path) -> Dict[str, Any]:
    cfg = load_config(default_config_path)
    params = asdict(cfg)
    params.update(
        {
            "preset_name": "adaptive_tuning_rig_find_stable_swarm",
            "agents": 200,
            "generations": TARGET_GENERATIONS,
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
    timeline = list(run_result.get("timeline", []))
    latest = timeline[-1] if timeline else {}
    final_population = int(run_result.get("final_population", 0))
    lineage_count = int(latest.get("lineage_count", 0))
    diversity_score = float(latest.get("diversity_score", 0.0))
    populations = [int(s.get("population", 0)) for s in timeline]
    late = populations[-20:] if populations else []
    late_std = 0.0
    if len(late) > 1:
        avg = sum(late) / len(late)
        late_std = (sum((p - avg) ** 2 for p in late) / len(late)) ** 0.5

    survives_full = len(timeline) >= TARGET_GENERATIONS
    collapsed = (min(populations) if populations else 0) <= 0 or final_population < 40
    explosion = final_population > 300

    score = 0.0
    score += max(0.0, 45.0 - (abs(final_population - TARGET_POPULATION) * 0.75))
    score += min(20.0, lineage_count * 1.4)
    score += min(20.0, diversity_score * 100.0)
    score += max(0.0, 15.0 - (late_std * 0.5))

    if not survives_full:
        score -= 20.0
    if collapsed:
        score -= 60.0
    if explosion:
        score -= 50.0
    if lineage_count < 8:
        score -= 15.0
    if diversity_score < 0.10:
        score -= 20.0

    score = max(0.0, min(100.0, score))

    healthy = (
        survives_full
        and 80 <= final_population <= 120
        and lineage_count >= 12
        and diversity_score >= 0.15
        and late_std <= 20.0
        and not collapsed
        and not explosion
    )
    near_healthy = (
        survives_full
        and 65 <= final_population <= 140
        and lineage_count >= 10
        and diversity_score >= 0.12
        and late_std <= 35.0
        and not collapsed
        and not explosion
    )

    if healthy:
        label = "healthy"
    elif near_healthy:
        label = "near_healthy"
    elif collapsed:
        label = "collapsed"
    elif explosion:
        label = "population_explosion"
    elif diversity_score < 0.12:
        label = "low_diversity"
    elif lineage_count < 10:
        label = "low_lineage"
    elif late_std > 45.0:
        label = "chaotic_instability"
    else:
        label = "stalled_underpowered"

    return {
        "score": round(score, 4),
        "label": label,
        "survives_full": survives_full,
        "final_population": final_population,
        "lineage_count": lineage_count,
        "diversity_score": round(diversity_score, 6),
        "late_population_std": round(late_std, 6),
        "healthy": healthy,
        "near_healthy": near_healthy,
    }


def adjust_parameters(params: Dict[str, Any], metrics: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    updated = dict(params)
    label = metrics.get("label")
    reason = "small mutation sweep"

    if label == "collapsed" or metrics.get("final_population", 0) < 80:
        updated["initial_energy"] = min(180, int(updated.get("initial_energy", 100)) + 8)
        updated["reproduction_threshold"] = max(95, int(updated.get("reproduction_threshold", 130)) - 5)
        updated["upkeep_cost"] = max(2, int(updated.get("upkeep_cost", 6)) - 1)
        updated["tasks_per_generation"] = max(20, int(updated.get("tasks_per_generation", 60)) - 6)
        reason = "counter collapse: raise energy and ease reproduction pressure"
    elif label == "population_explosion" or metrics.get("final_population", 0) > 120:
        updated["upkeep_cost"] = min(12, int(updated.get("upkeep_cost", 6)) + 1)
        updated["reproduction_threshold"] = min(200, int(updated.get("reproduction_threshold", 130)) + 6)
        updated["reproduction_cooldown_enabled"] = True
        updated["reproduction_cooldown_generations"] = min(6, int(updated.get("reproduction_cooldown_generations", 2)) + 1)
        reason = "counter explosion: tighten reproduction and add cooldown"
    elif label in {"low_diversity", "low_lineage"}:
        updated["diversity_bonus"] = min(0.6, float(updated.get("diversity_bonus", 0.175)) + 0.03)
        updated["diversity_min_lineages"] = min(24, int(updated.get("diversity_min_lineages", 14)) + 1)
        updated["mutation_rate"] = min(0.45, float(updated.get("mutation_rate", 0.28)) + 0.02)
        updated["anti_dominance_enabled"] = True
        updated["lineage_size_penalty_enabled"] = True
        updated["lineage_energy_share_penalty_enabled"] = True
        reason = "improve diversity/lineage resilience with anti-dominance pressure"
    elif label == "chaotic_instability":
        updated["mutation_rate"] = max(0.12, float(updated.get("mutation_rate", 0.28)) - 0.03)
        updated["tasks_per_generation"] = max(25, int(updated.get("tasks_per_generation", 60)) - 5)
        reason = "reduce instability by lowering mutation and task load"

    return updated, reason
