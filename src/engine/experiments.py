from __future__ import annotations

import itertools
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

from src.analytics.swarm_scoring import scorer_from_dict
from src.engine.simulation import SimulationConfig, SimulationEngine


SWEEP_KEYS = [
    "seed",
    "agents",
    "generations",
    "initial_energy",
    "reproduction_threshold",
    "mutation_rate",
    "upkeep_cost",
    "tasks_per_generation",
    "tier_mix",
    "reward_policy_id",
    "anti_dominance_enabled",
    "diminishing_reward_enabled",
    "diminishing_reward_k",
    "lineage_size_penalty_enabled",
    "lineage_size_penalty_threshold",
    "lineage_size_penalty_multiplier",
    "lineage_energy_share_penalty_enabled",
    "lineage_energy_share_penalty_threshold",
    "lineage_energy_share_penalty_multiplier",
    "reproduction_cooldown_enabled",
    "reproduction_cooldown_generations",
    "reproduction_cost",
    "child_energy_fraction",
]


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _read_last_metric(summary_path: str | None, stream_name: str, field: str) -> Any:
    if not summary_path:
        return None
    stream_path = Path(summary_path).with_name(f"{stream_name}.jsonl")
    if not stream_path.exists():
        return None
    lines = [line for line in stream_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    row = json.loads(lines[-1])
    return row.get(field)


def _build_comparison_row(cfg: SimulationConfig, result: Dict[str, Any], run_score: Dict[str, Any]) -> Dict[str, Any]:
    timeline = result.get("timeline", [])
    peak = max(timeline, key=lambda item: item.get("population", 0)) if timeline else {}
    final_gen = timeline[-1] if timeline else {}
    late = timeline[-max(1, len(timeline) // 4):] if timeline else []
    phase = result.get("phase_diagnostics", {})
    solved = int(result.get("totals", {}).get("solved", 0))
    unsolved = max(0, int(cfg.generations * cfg.tasks_per_generation) - solved)
    summary_path = result.get("summary_path")

    return {
        "run_label": cfg.run_label,
        "seed": cfg.seed,
        "summary_path": summary_path,
        "run_dir": result.get("run_dir"),
        "final_population": result.get("final_population", 0),
        "late_avg_population": round(mean([g.get("population", 0) for g in late]), 4) if late else 0.0,
        "late_population_std": round(pstdev([g.get("population", 0) for g in late]), 4) if len(late) > 1 else 0.0,
        "peak_population": peak.get("population", 0),
        "generation_of_peak_population": peak.get("generation", 0),
        "stabilization_generation": phase.get("stabilization_start_generation"),
        "total_solved": solved,
        "problems_unsolved": unsolved,
        "solve_rate": round(solved / max(1, cfg.generations * cfg.tasks_per_generation), 6),
        "collaboration_share": final_gen.get("collaboration_share", 0.0),
        "artifact_assisted_share": round(result.get("totals", {}).get("artifact_reuse", 0) / max(1, solved), 6),
        "surviving_lineage_count": _read_last_metric(summary_path, "generation_metrics", "surviving_lineage_count"),
        "top_lineage_energy_share": _read_last_metric(summary_path, "generation_metrics", "top_lineage_energy_share"),
        "top_3_lineage_energy_share": _read_last_metric(summary_path, "generation_metrics", "top_3_lineage_energy_share"),
        "energy_median": _read_last_metric(summary_path, "generation_metrics", "energy_median"),
        "energy_mean": _read_last_metric(summary_path, "generation_metrics", "energy_mean"),
        "energy_p90": _read_last_metric(summary_path, "generation_metrics", "energy_p90"),
        "energy_p99": _read_last_metric(summary_path, "generation_metrics", "energy_p99"),
        "energy_min": _read_last_metric(summary_path, "generation_metrics", "energy_min"),
        "energy_gini": _read_last_metric(summary_path, "generation_metrics", "energy_gini"),
        "top_10pct_energy_share": _read_last_metric(summary_path, "generation_metrics", "top_10pct_energy_share"),
        "reward_share_top_lineage": _read_last_metric(summary_path, "generation_metrics", "reward_share_top_lineage"),
        "reward_share_top_10pct_agents": _read_last_metric(summary_path, "generation_metrics", "reward_share_top_10pct_agents"),
        "unique_reproducers": _read_last_metric(summary_path, "generation_metrics", "cumulative_unique_reproducers"),
        "repeat_reproducers": _read_last_metric(summary_path, "generation_metrics", "cumulative_repeat_reproducers"),
        "births_concentration_share_top_5": _read_last_metric(summary_path, "generation_metrics", "birth_share_top_5"),
        "scoring": run_score,
    }


def _normalize_config_for_grouping(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k not in {"seed", "run_label", "experiment_id", "log_dir"}}


def _persist_scoring_summary(result: Dict[str, Any], row: Dict[str, Any]) -> str | None:
    run_dir = result.get("run_dir")
    if not run_dir:
        return None
    path = Path(run_dir) / "healthy_swarm_score.json"
    path.write_text(json.dumps(row.get("scoring", {}), indent=2), encoding="utf-8")
    return str(path)


def _aggregate_trial_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [float(r.get("scoring", {}).get("composite_score", 0.0)) for r in rows]
    pass_rate = sum(1 for r in rows if r.get("scoring", {}).get("pass")) / max(1, len(rows))
    solved = [float(r.get("total_solved", 0)) for r in rows]
    top1 = [float(r.get("top_lineage_energy_share") or 1.0) for r in rows]
    unstable = (pstdev(scores) if len(scores) > 1 else 0.0) > 8.0 or pass_rate < 0.5
    return {
        "trial_count": len(rows),
        "score_mean": round(mean(scores), 4) if scores else 0.0,
        "score_std": round(pstdev(scores), 4) if len(scores) > 1 else 0.0,
        "score_min": round(min(scores), 4) if scores else 0.0,
        "score_max": round(max(scores), 4) if scores else 0.0,
        "solve_mean": round(mean(solved), 4) if solved else 0.0,
        "top_lineage_share_mean": round(mean(top1), 6) if top1 else 1.0,
        "pass_rate": round(pass_rate, 4),
        "unstable": unstable,
    }


def _write_experiment_outputs(output_dir: Path, experiment_id: str, detailed_rows: List[Dict[str, Any]], aggregate_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = sorted(aggregate_rows, key=lambda r: r.get("aggregate", {}).get("score_mean", 0.0), reverse=True)

    detailed_path = output_dir / "runs_detailed.json"
    aggregate_path = output_dir / "config_aggregates.json"
    leaderboard_path = output_dir / "leaderboard.json"

    detailed_path.write_text(json.dumps({"experiment_id": experiment_id, "runs": detailed_rows}, indent=2), encoding="utf-8")
    aggregate_path.write_text(json.dumps({"experiment_id": experiment_id, "configs": aggregate_rows}, indent=2), encoding="utf-8")
    leaderboard_path.write_text(json.dumps({"experiment_id": experiment_id, "leaderboard": leaderboard}, indent=2), encoding="utf-8")
    return {
        "runs_detailed": str(detailed_path),
        "config_aggregates": str(aggregate_path),
        "leaderboard": str(leaderboard_path),
    }


def run_experiment_batch(config_path: str | Path) -> Dict[str, Any]:
    spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
    base = dict(spec.get("base", {}))
    sweep = dict(spec.get("sweep", {}))
    output_root = Path(spec.get("output_root", "runs/experiments"))
    experiment_id = spec.get("experiment_id", "exp_batch")
    trials_per_config = max(1, int(spec.get("trials_per_config", 1)))

    sweep_items = [(k, _as_list(v)) for k, v in sweep.items() if k in SWEEP_KEYS]
    if not sweep_items:
        sweep_items = [("seed", _as_list(base.get("seed", 42)))]

    keys = [k for k, _ in sweep_items]
    combinations = list(itertools.product(*[vals for _, vals in sweep_items]))
    comparisons: List[Dict[str, Any]] = []
    scorer = scorer_from_dict(spec.get("scoring"))

    for idx, combo in enumerate(combinations, start=1):
        params = dict(base)
        params.update(dict(zip(keys, combo)))
        params.setdefault("log_dir", str(output_root))
        params["experiment_id"] = experiment_id
        base_seed = int(params.get("seed", 42))

        for trial in range(1, trials_per_config + 1):
            params["seed"] = base_seed + trial - 1
            params["run_label"] = f"{experiment_id}_{idx}_t{trial}"
            cfg = SimulationConfig(**params)
            result = SimulationEngine(cfg).run()
            run_score = scorer.score_run(result)
            row = _build_comparison_row(cfg, result, run_score)
            row["trial"] = trial
            row["config_index"] = idx
            row["config_group"] = _normalize_config_for_grouping(params)
            row["healthy_swarm_score_path"] = _persist_scoring_summary(result, row)
            comparisons.append(row)

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in comparisons:
        key = json.dumps(row["config_group"], sort_keys=True)
        grouped.setdefault(key, {"config": row["config_group"], "trials": []})["trials"].append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for item in grouped.values():
        aggregate_rows.append({"config": item["config"], "aggregate": _aggregate_trial_rows(item["trials"]), "trials": item["trials"]})

    output_dir = output_root / experiment_id
    out_files = _write_experiment_outputs(output_dir, experiment_id, comparisons, aggregate_rows)
    comparison_summary = output_dir / "comparison_summary.json"
    comparison_summary.write_text(json.dumps({"experiment_id": experiment_id, "runs": comparisons}, indent=2), encoding="utf-8")
    return {
        "experiment_id": experiment_id,
        "comparison_summary": str(comparison_summary),
        "runs": comparisons,
        "aggregates": aggregate_rows,
        "reports": out_files,
    }


def run_anti_dominance_experiments(config_path: str | Path) -> Dict[str, Any]:
    spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
    experiment_id = spec.get("experiment_id", "anti_dominance_matrix")
    output_root = Path(spec.get("output_root", "runs/experiments"))
    base = dict(spec.get("baseline", {}))
    variants = list(spec.get("experiments", []))
    trials_per_config = max(1, int(spec.get("trials_per_config", 1)))
    scorer = scorer_from_dict(spec.get("scoring"))

    runs: List[Dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        params = dict(base)
        params.update(variant.get("overrides", {}))
        params.setdefault("log_dir", str(output_root / experiment_id))
        params["experiment_id"] = experiment_id
        base_seed = int(params.get("seed", 42))

        for trial in range(1, trials_per_config + 1):
            params["seed"] = base_seed + trial - 1
            params["run_label"] = f"{variant.get('id', f'variant_{idx}')}_t{trial}"
            cfg = SimulationConfig(**params)
            result = SimulationEngine(cfg).run()
            run_score = scorer.score_run(result)
            row = _build_comparison_row(cfg, result, run_score)
            row["experiment_variant"] = variant.get("id", f"variant_{idx}")
            row["description"] = variant.get("description", "")
            row["trial"] = trial
            row["healthy_swarm_score_path"] = _persist_scoring_summary(result, row)
            runs.append(row)

    baseline_runs = [r for r in runs if r.get("experiment_variant") == "baseline_optimal_control"] or runs[:1]
    baseline_dom = mean([float(r.get("top_lineage_energy_share") or 0.0) for r in baseline_runs]) if baseline_runs else 0.0
    baseline_throughput = mean([float(r.get("total_solved") or 0.0) for r in baseline_runs]) if baseline_runs else 0.0

    for row in runs:
        current_dom = float(row.get("top_lineage_energy_share") or 0.0)
        current_throughput = float(row.get("total_solved") or 0.0)
        row["dominance_improved_vs_baseline"] = current_dom < baseline_dom
        row["throughput_degraded_vs_baseline"] = current_throughput < baseline_throughput

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(row.get("experiment_variant", "variant"), []).append(row)

    aggregates = []
    for variant, rows in grouped.items():
        aggregate = _aggregate_trial_rows(rows)
        aggregate["variant"] = variant
        aggregates.append({"variant": variant, "description": rows[0].get("description", ""), "aggregate": aggregate, "trials": rows})

    out = output_root / experiment_id / "anti_dominance_comparison_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment_id": experiment_id, "runs": runs}, indent=2), encoding="utf-8")
    out_files = _write_experiment_outputs(out.parent, experiment_id, runs, aggregates)
    return {"experiment_id": experiment_id, "comparison_summary": str(out), "runs": runs, "aggregates": aggregates, "reports": out_files}
