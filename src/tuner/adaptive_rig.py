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

    survives_full = len(timeline) >= TARGET_GENERATIONS
    reaches_target_by_g80 = any(90 <= p <= 110 for p in populations[:80]) if populations else False
    late_range = (max(late) - min(late)) if late else 0
    severe_instability = late_std > 18.0 or late_range > 40

    gate_results = {
        "final_population_max_ok": final_population <= 150,
        "final_population_min_ok": final_population >= 50,
        "lineage_count_ok": lineage_count >= 10,
        "top_lineage_share_ok": top_lineage_share <= 0.6,
        "top_3_lineage_share_ok": top_3_lineage_share <= 0.8,
        "reaches_target_by_generation_80_ok": reaches_target_by_g80,
        "late_stability_ok": not severe_instability,
    }
    passed_gates = all(gate_results.values()) and survives_full

    if not passed_gates:
        score = 5.0
    else:
        population_component = max(0.0, 60.0 - (abs(final_population - TARGET_POPULATION) * 3.0))
        lineage_component = min(20.0, max(0.0, (lineage_count - 12) * 2.5 + 12.0))
        diversity_component = min(15.0, max(0.0, (diversity_score - 0.15) * 150.0 + 10.0))
        stability_component = max(0.0, 5.0 - (late_std * 0.25))
        score = max(0.0, min(100.0, population_component + lineage_component + diversity_component + stability_component))

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
    elif final_population > 150:
        label = "failed_overshoot"
    elif final_population < 50 or not survives_full:
        label = "failed_collapse"
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
    }


def adjust_parameters(params: Dict[str, Any], metrics: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    updated = dict(params)
    label = metrics.get("label")
    reason = "small mutation sweep"
    is_failed = isinstance(label, str) and label.startswith("failed_")

    if label == "failed_collapse":
        updated["initial_energy"] = min(200, int(updated.get("initial_energy", 100)) + 12)
        updated["reproduction_threshold"] = max(90, int(updated.get("reproduction_threshold", 130)) - 8)
        updated["upkeep_cost"] = max(2, int(updated.get("upkeep_cost", 6)) - 2)
        updated["tasks_per_generation"] = max(20, int(updated.get("tasks_per_generation", 60)) - 8)
        reason = "strong negative signal (collapse): sharply raise survivability and reduce pressure"
    elif label == "failed_overshoot":
        updated["upkeep_cost"] = min(14, int(updated.get("upkeep_cost", 6)) + 2)
        updated["reproduction_threshold"] = min(220, int(updated.get("reproduction_threshold", 130)) + 10)
        updated["reproduction_cooldown_enabled"] = True
        updated["reproduction_cooldown_generations"] = min(8, int(updated.get("reproduction_cooldown_generations", 2)) + 2)
        reason = "strong negative signal (overshoot): tighten reproduction aggressively"
    elif label in {"failed_low_diversity", "failed_dominance"}:
        updated["diversity_bonus"] = min(0.7, float(updated.get("diversity_bonus", 0.175)) + 0.05)
        updated["diversity_min_lineages"] = min(28, int(updated.get("diversity_min_lineages", 14)) + 2)
        updated["mutation_rate"] = min(0.5, float(updated.get("mutation_rate", 0.28)) + 0.03)
        updated["anti_dominance_enabled"] = True
        updated["lineage_size_penalty_enabled"] = True
        updated["lineage_energy_share_penalty_enabled"] = True
        reason = "strong negative signal (diversity/dominance): push anti-dominance and lineage spread"
    elif label == "failed_instability":
        updated["mutation_rate"] = max(0.1, float(updated.get("mutation_rate", 0.28)) - 0.05)
        updated["tasks_per_generation"] = max(20, int(updated.get("tasks_per_generation", 60)) - 8)
        updated["reproduction_cooldown_enabled"] = True
        reason = "strong negative signal (instability): lower volatility and smooth growth"
    elif is_failed:
        updated["mutation_rate"] = max(0.1, float(updated.get("mutation_rate", 0.28)) - 0.02)
        reason = "strong negative failed signal: move away from current region"

    return updated, reason
