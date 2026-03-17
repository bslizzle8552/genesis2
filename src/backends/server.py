from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from collections import Counter, defaultdict, deque
from statistics import mean
from typing import Deque, Dict, List
from urllib import error, request
import uuid

from src.analytics.bvl import events_from_problem_metrics
from src.engine.simulation import SimulationEngine, load_config
from src.tuner.adaptive_rig import (
    adjust_parameters,
    build_find_stable_swarm_baseline,
    canonical_config_signature,
    generate_local_variants,
    score_and_label_run,
)


ROOT = Path(__file__).resolve().parents[2]
UI_PATH = ROOT / "src" / "ui" / "index.html"
CONFIG_PATH = ROOT / "config"
CONFIGS_PATH = ROOT / "configs"
DEFAULT_CONFIG = CONFIGS_PATH / "ecosystem_optimal_v1.json"
LEGACY_DEFAULT_CONFIG = CONFIG_PATH / "default.json"

LAST_RESULT = {
    "status": "idle",
    "result": None,
    "progress": None,
    "error": None,
    "run_dir": None,
    "problem_board_events": [],
}
RUN_LOCK = threading.Lock()
TUNING_STATE = {
    "status": "idle",
    "mode": None,
    "tuning_session_id": None,
    "message": None,
    "elapsed_seconds": 0,
    "batch_number": 0,
    "run_in_batch": 0,
    "current_run": 0,
    "max_runs": 0,
    "current_parameters": None,
    "best_score": 0.0,
    "score_progression": [],
    "best_run": None,
    "winning_config": None,
    "latest_outcome_label": None,
    "latest_adjustment_reason": None,
    "early_stop_reason": None,
    "repeatability": None,
    "candidate_configs": [],
    "final_outcome": None,
    "final_summary_path": None,
    "error": None,
    "session_diagnostics": None,
    "goal_profile": None,
    "advisory_settings": None,
    "advisory_usage": None,
    "human_readable_summary": None,
    "human_readable_report_path": None,
}

GOAL_PROFILE = {
    "mode": "find_stable_swarm",
    "starting_agents_target": 25,
    "target_population": 100,
    "target_reach_generation": 80,
    "stability_band": [90, 110],
    "horizon_generations": 100,
}

PLAIN_LABELS = {
    "healthy": "Stable and healthy swarm behavior.",
    "near_healthy": "Near healthy behavior with minor stability gaps.",
    "failed_overshoot": "Population grew too fast and exceeded the healthy range.",
    "failed_collapse": "Population fell too low to recover.",
    "failed_low_diversity": "Too few lineages survived.",
    "failed_dominance": "Too much of the swarm was controlled by one lineage.",
    "failed_instability": "Population became too volatile to stabilize.",
    "failed_wrong_start": "Run started from the wrong population instead of 25 agents.",
}


def _stable_swarm_baseline() -> Dict[str, object]:
    source = DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else LEGACY_DEFAULT_CONFIG
    params = build_find_stable_swarm_baseline(source)
    params.update({"seed": 42, "run_label": "adaptive_tuning_rig_baseline", "experiment_id": "adaptive_tuning_rig", "log_dir": "runs/tuning_sessions"})
    return params




def _load_advisory_file_defaults() -> Dict[str, object]:
    candidates = [
        ROOT / "config" / "advisory.json",
        ROOT / "config" / "tuner_advisory.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, OSError):
            continue
    return {}
def _advisory_defaults() -> Dict[str, object]:
    file_defaults = _load_advisory_file_defaults()
    defaults = {
        "advisory_api_enabled": False,
        "advisory_mode": "post_run",
        "advisory_timeout_seconds": 6,
        "advisory_max_calls_per_session": 20,
        "advisory_model_name": os.getenv("GENESIS2_ADVISORY_MODEL", "local-advisor"),
        "advisory_temperature": 0.1,
        "advisory_endpoint": os.getenv("GENESIS2_ADVISORY_ENDPOINT", ""),
        "advisory_api_key_env": os.getenv("GENESIS2_ADVISORY_API_KEY_ENV", "GENESIS2_ADVISORY_API_KEY"),
    }
    defaults.update({k: v for k, v in file_defaults.items() if k in defaults})
    return defaults


def _metric_from_timeline(timeline: List[Dict[str, object]], key: str, default: float = 0.0) -> float:
    if not timeline:
        return default
    return float(timeline[-1].get(key, default) or default)


def _build_advisory_payload(*, session_id: str, run_number: int, elapsed_seconds: int, current: Dict[str, object], result: Dict[str, object], metrics: Dict[str, object], run_records: List[Dict[str, object]]) -> Dict[str, object]:
    timeline = list(result.get("timeline", []))
    populations = [int(step.get("population", 0)) for step in timeline]
    late = populations[-20:] if populations else []
    peak = max(timeline, key=lambda item: int(item.get("population", 0))) if timeline else {}
    dominance = timeline[-1].get("dominance_metrics", {}) if timeline else {}
    solves = int(result.get("totals", {}).get("solved", 0))
    possible = max(1, int(current.get("generations", 100)) * int(current.get("tasks_per_generation", 60)))
    previous = []
    for rec in run_records[-5:]:
        prev = {
            "run": rec.get("run_in_batch"),
            "label": rec.get("label"),
            "score": rec.get("score"),
            "final_population": rec.get("metrics", {}).get("final_population"),
            "changed_parameters": rec.get("changed_parameters", {}),
            "score_delta": rec.get("score_delta", 0.0),
        }
        previous.append(prev)
    return {
        "session_info": {
            "tuning_session_id": session_id,
            "run_id": str(result.get("run_id", "")),
            "attempt_number": run_number,
            "elapsed_session_time_seconds": elapsed_seconds,
        },
        "goal_profile": GOAL_PROFILE,
        "parameters_used": {
            k: current.get(k)
            for k in [
                "agents", "generations", "initial_energy", "reproduction_threshold", "mutation_rate", "upkeep_cost",
                "tasks_per_generation", "diversity_bonus", "diversity_min_lineages", "anti_dominance_enabled",
                "lineage_size_penalty_enabled", "lineage_energy_share_penalty_enabled", "reproduction_cooldown_enabled",
                "reproduction_cooldown_generations",
            ]
        },
        "observed_results": {
            "final_population": int(result.get("final_population", 0)),
            "peak_population": int(peak.get("population", 0) or 0),
            "generation_of_peak_population": int(peak.get("generation", 0) or 0),
            "lineage_count": int(metrics.get("lineage_count", 0)),
            "diversity_score": float(metrics.get("diversity_score", 0.0)),
            "solve_rate": round(solves / possible, 6),
            "births_total": int(sum(int(step.get("births", 0)) for step in timeline)),
            "births_late_run": int(sum(int(step.get("births", 0)) for step in timeline[-20:])),
            "extinction_events": int(sum(int(step.get("dominance_metrics", {}).get("lineage_extinction_count", 0)) for step in timeline)),
            "top_lineage_share": float(metrics.get("top_lineage_share", 1.0)),
            "top_3_lineage_share": float(metrics.get("top_3_lineage_share", 1.0)),
            "inequality_metrics": {
                "energy_gini": float(dominance.get("energy_gini", 0.0) or 0.0),
                "reward_share_top_lineage": float(dominance.get("reward_share_top_lineage", 0.0) or 0.0),
            },
            "late_run_mean_population": round(mean(late), 4) if late else 0.0,
            "late_run_population_volatility": float(metrics.get("late_population_std", 0.0)),
            "target_band_reached": bool(metrics.get("hard_gates", {}).get("reaches_target_by_generation_80_ok", False)),
            "target_band_sustained": bool(metrics.get("passed_hard_gates", False)),
            "outcome_label": metrics.get("label"),
            "score": float(metrics.get("score", 0.0)),
            "failure_reasons": metrics.get("diagnosis", []),
            "early_stop_reason": TUNING_STATE.get("early_stop_reason"),
        },
        "recent_tuning_history": previous,
    }


def _default_advisory_response() -> Dict[str, object]:
    return {
        "diagnosis": {
            "primary_failure_mode": "mixed",
            "secondary_failure_modes": [],
            "reasoning_summary": "No API recommendation available.",
        },
        "parameter_recommendations": {},
        "suggested_next_config": {},
        "confidence": 0.0,
        "operator_summary": "Fallback to deterministic tuner.",
    }


def _call_advisory_api(payload: Dict[str, object], settings: Dict[str, object]) -> Dict[str, object]:
    endpoint = str(settings.get("advisory_endpoint") or "").strip()
    if not endpoint:
        return {"error": "endpoint_missing", "raw": None, "parsed": _default_advisory_response()}
    body = json.dumps({
        "model": settings.get("advisory_model_name"),
        "temperature": float(settings.get("advisory_temperature", 0.1)),
        "mode": settings.get("advisory_mode", "post_run"),
        "payload": payload,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(str(settings.get("advisory_api_key_env", "GENESIS2_ADVISORY_API_KEY")), "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        timeout = float(settings.get("advisory_timeout_seconds", 6))
        with request.urlopen(req, timeout=timeout) as resp:
            raw_text = resp.read().decode("utf-8")
        parsed = json.loads(raw_text)
        return {"error": None, "raw": raw_text, "parsed": parsed}
    except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return {"error": str(exc), "raw": None, "parsed": _default_advisory_response()}


def _safe_change(current: Dict[str, object], key: str, action: str, delta: float, min_v: float, max_v: float, max_step: float) -> float:
    value = float(current.get(key, 0.0))
    bounded = max(-max_step, min(max_step, float(delta)))
    if action == "increase":
        value += abs(bounded)
    elif action == "decrease":
        value -= abs(bounded)
    return max(min_v, min(max_v, value))


def _merge_advisory_with_deterministic(current: Dict[str, object], deterministic: Dict[str, object], advisory: Dict[str, object]) -> tuple[Dict[str, object], str]:
    merged = dict(deterministic)
    recs = advisory.get("parameter_recommendations", {}) if isinstance(advisory, dict) else {}
    safety = {
        "reproduction_threshold": (90.0, 220.0, 10.0),
        "mutation_rate": (0.05, 0.5, 0.04),
        "upkeep_cost": (2.0, 14.0, 2.0),
        "tasks_per_generation": (20.0, 120.0, 10.0),
        "diversity_bonus": (0.05, 0.8, 0.06),
        "diversity_min_lineages": (8.0, 28.0, 2.0),
    }
    applied = []
    for key, (mn, mx, max_step) in safety.items():
        rule = recs.get(key)
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action", "keep"))
        if action not in {"increase", "decrease", "keep"}:
            continue
        if action == "keep":
            continue
        delta = float(rule.get("delta", 0.0) or 0.0)
        adv_value = _safe_change(current, key, action, delta, mn, mx, max_step)
        det_delta = float(deterministic.get(key, current.get(key, 0.0))) - float(current.get(key, 0.0))
        if det_delta == 0.0 or (det_delta > 0 and adv_value >= float(current.get(key, 0.0))) or (det_delta < 0 and adv_value <= float(current.get(key, 0.0))):
            merged[key] = int(round(adv_value)) if key in {"upkeep_cost", "tasks_per_generation", "diversity_min_lineages"} else round(adv_value, 6)
            applied.append(key)
    source = "merged_deterministic_api" if applied else "deterministic_tuner"
    return merged, f"{source}; applied={applied}" if applied else source


def _classify_run_failure_bucket(metrics: Dict[str, object]) -> str:
    label = str(metrics.get("label", ""))
    diagnosis = set(metrics.get("diagnosis", []) or [])
    if "overshoot" in diagnosis or label == "failed_overshoot":
        return "overshoot"
    if "collapse" in diagnosis or label == "failed_collapse":
        return "collapse"
    if "early_dominance" in diagnosis or "late_dominance" in diagnosis or label == "failed_dominance":
        return "dominance"
    if "low_diversity" in diagnosis or label == "failed_low_diversity":
        return "low_diversity"
    if "instability" in diagnosis or label == "failed_instability":
        return "instability"
    if "weak_throughput" in diagnosis:
        return "weak_growth"
    if label in {"near_healthy", "healthy"}:
        return "near_success"
    return "weak_growth"


def _score_explanation(metrics: Dict[str, object]) -> Dict[str, object]:
    positives: List[str] = []
    negatives: List[str] = []
    gates = metrics.get("hard_gates", {}) if isinstance(metrics.get("hard_gates"), dict) else {}

    if gates.get("starts_near_25_ok"):
        positives.append("Started near the intended 25-agent baseline.")
    else:
        negatives.append("Start population drifted away from the intended 25-agent baseline.")

    if gates.get("reaches_target_by_generation_80_ok"):
        positives.append("Reached the target population band by generation 80.")
    else:
        negatives.append("Did not reach the 90-110 target band by generation 80.")

    if gates.get("lineage_count_ok"):
        positives.append("Maintained enough lineages to preserve ecosystem resilience.")
    else:
        negatives.append("Lineage count was too low, suggesting weak diversity.")

    if gates.get("top_lineage_share_ok") and gates.get("top_3_lineage_share_ok"):
        positives.append("Dominance stayed bounded (no lineage captured excessive energy share).")
    else:
        negatives.append("Dominance pressure was high; one lineage or top-3 lineages captured too much energy.")

    if gates.get("late_stability_ok"):
        positives.append("Late generations stayed relatively stable.")
    else:
        negatives.append("Late-run population volatility remained high.")

    diagnosis = metrics.get("diagnosis", []) or []
    biggest_failure = diagnosis[0] if diagnosis else _classify_run_failure_bucket(metrics)
    if diagnosis:
        negatives.append(f"Primary failure mode: {biggest_failure}.")

    return {
        "score": metrics.get("score"),
        "label": metrics.get("label"),
        "positives": positives,
        "negatives": negatives,
        "biggest_failure_mode": biggest_failure,
    }


def _build_run_observability(result: Dict[str, object]) -> Dict[str, object]:
    agents = list(result.get("agents", []))
    timeline = list(result.get("timeline", []))
    generation_count = len(timeline)
    final_generation = int(timeline[-1].get("generation", generation_count)) if timeline else 0

    by_id = {str(a.get("agent_id")): a for a in agents}
    offspring_counts: Counter[str] = Counter()
    for agent in agents:
        parent_id = agent.get("parent_id")
        if parent_id:
            offspring_counts[str(parent_id)] += 1

    helpers_given: Counter[str] = Counter()
    helpers_received: Counter[str] = Counter()
    tasks_completed: Counter[str] = Counter()
    help_rows: List[Dict[str, object]] = []

    for problem in result.get("problems", []):
        gen = int(problem.get("generation", 0) or 0)
        participants = list(dict.fromkeys(problem.get("agents_involved", []) or []))
        if problem.get("solved"):
            for aid in participants:
                tasks_completed[str(aid)] += 1
        chain = problem.get("contribution_chain", []) or []
        solver_id = participants[0] if participants else None
        for step in chain:
            step_type = str(step.get("type", ""))
            helper_id = str(step.get("agent_id", ""))
            if step_type in {"collaboration", "verification", "critique", "subtask", "integration"} and helper_id:
                recipient_id = str(solver_id or "")
                helpers_given[helper_id] += 1
                if recipient_id:
                    helpers_received[recipient_id] += 1
                help_rows.append(
                    {
                        "generation": gen,
                        "helper_agent_id": helper_id,
                        "recipient_agent_id": recipient_id,
                        "task_or_context": str(problem.get("problem_id", "")),
                        "contribution_type": step_type,
                        "reward_split_or_transfer": round(float((problem.get("reward_split") or {}).get(helper_id, 0.0)), 4),
                    }
                )

    survivors_table: List[Dict[str, object]] = []
    lineage_births: Counter[str] = Counter()
    lineage_deaths: Counter[str] = Counter()

    for agent in agents:
        aid = str(agent.get("agent_id"))
        lineage = str(agent.get("lineage_id"))
        lineage_births[lineage] += 1
        survivors_table.append(
            {
                "agent_id": aid,
                "lineage_id": lineage,
                "role": agent.get("role"),
                "age": final_generation - int(agent.get("generation_born", 0) or 0),
                "final_energy": round(float(agent.get("energy", 0.0)), 4),
                "offspring_count": int(offspring_counts.get(aid, 0)),
                "helps_given": int(helpers_given.get(aid, 0)),
                "helps_received": int(helpers_received.get(aid, 0)),
                "tasks_completed": int(tasks_completed.get(aid, 0)),
            }
        )

    deaths_table: List[Dict[str, object]] = []
    known_ids = set(by_id.keys())
    for evt in result.get("board_messages", []):
        if evt.get("message_type") != "death":
            continue
        aid = str(evt.get("agent_id"))
        if not aid or aid in known_ids:
            continue
        lineage = aid.split("-")[0]
        lineage_deaths[lineage] += 1
        deaths_table.append(
            {
                "agent_id": aid,
                "lineage_id": lineage,
                "role": None,
                "generation_of_death": int(evt.get("generation", 0) or 0),
                "likely_cause": "energy_depleted",
                "peak_energy": None,
                "offspring_count": int(offspring_counts.get(aid, 0)),
            }
        )

    reproduction_log = [
        {
            "generation": int(e.get("generation", 0) or 0),
            "parent_agent_id": e.get("parent_id"),
            "parent_lineage": e.get("parent_lineage_id"),
            "energy_before": e.get("parent_energy_before"),
            "energy_after": e.get("parent_energy_after"),
            "child_agent_id": e.get("child_id"),
            "child_lineage": e.get("child_lineage_id"),
            "mutation_summary": f"mutation_rate={e.get('mutation_rate', 0.0)}; applied={bool(e.get('mutation_applied'))}",
        }
        for e in (result.get("report", {}).get("observability", {}) and [])
    ]
    # Fallback to exported reproduction events if available.
    if not reproduction_log:
        path = Path(str(result.get("run_dir", ""))) / "reproduction_events.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for e in payload.get("events", []):
                reproduction_log.append(
                    {
                        "generation": int(e.get("generation", 0) or 0),
                        "parent_agent_id": e.get("parent_id"),
                        "parent_lineage": e.get("parent_lineage_id"),
                        "energy_before": e.get("parent_energy_before"),
                        "energy_after": e.get("parent_energy_after"),
                        "child_agent_id": e.get("child_id"),
                        "child_lineage": e.get("child_lineage_id"),
                        "mutation_summary": f"mutation_rate={e.get('mutation_rate', 0.0)}; applied={bool(e.get('mutation_applied'))}",
                    }
                )
                lineage_births[str(e.get("child_lineage_id"))] += 1

    late_cutoff = max(1, final_generation - 20)
    lineage_survivors: Counter[str] = Counter(str(a.get("lineage_id")) for a in agents)
    lineage_energy: Counter[str] = Counter()
    for a in agents:
        lineage_energy[str(a.get("lineage_id"))] += float(a.get("energy", 0.0))
    total_energy = sum(lineage_energy.values())

    lineage_peak_dominance: defaultdict[str, float] = defaultdict(float)
    for step in timeline:
        lineages = step.get("lineages", {}) or {}
        pop = max(1, int(step.get("population", 0) or 0))
        for lid, count in lineages.items():
            share = float(count) / float(pop)
            if share > lineage_peak_dominance[str(lid)]:
                lineage_peak_dominance[str(lid)] = round(share, 6)

    births_by_parent_lineage: Counter[str] = Counter(str(row.get("parent_lineage")) for row in reproduction_log if row.get("parent_lineage"))
    late_births_by_lineage: Counter[str] = Counter(
        str(row.get("parent_lineage"))
        for row in reproduction_log
        if row.get("parent_lineage") and int(row.get("generation", 0) or 0) >= late_cutoff
    )

    top_reproducer_by_lineage: Dict[str, str] = {}
    repro_by_parent: Counter[str] = Counter(str(row.get("parent_agent_id")) for row in reproduction_log if row.get("parent_agent_id"))
    for row in reproduction_log:
        lid = str(row.get("parent_lineage"))
        pid = str(row.get("parent_agent_id"))
        if not lid or not pid:
            continue
        curr = top_reproducer_by_lineage.get(lid)
        if curr is None or repro_by_parent[pid] > repro_by_parent[curr]:
            top_reproducer_by_lineage[lid] = pid

    lineage_ids = set(lineage_survivors) | set(lineage_births) | set(lineage_deaths) | set(births_by_parent_lineage)
    lineage_summary: List[Dict[str, object]] = []
    for lid in sorted(lineage_ids):
        extinct_gen = None
        if lineage_survivors.get(lid, 0) == 0:
            death_gens = [int(d.get("generation_of_death", 0)) for d in deaths_table if str(d.get("lineage_id")) == lid]
            extinct_gen = max(death_gens) if death_gens else None
        lineage_summary.append(
            {
                "lineage_id": lid,
                "births": int(births_by_parent_lineage.get(lid, 0)),
                "deaths": int(lineage_deaths.get(lid, 0)),
                "survivors": int(lineage_survivors.get(lid, 0)),
                "late_births": int(late_births_by_lineage.get(lid, 0)),
                "top_reproducer": top_reproducer_by_lineage.get(lid),
                "total_energy_share": round(float(lineage_energy.get(lid, 0.0)) / max(1e-9, total_energy), 6),
                "peak_dominance_share": round(float(lineage_peak_dominance.get(lid, 0.0)), 6),
                "extinction_generation_if_any": extinct_gen,
            }
        )

    return {
        "survivors_table": sorted(survivors_table, key=lambda row: row["final_energy"], reverse=True),
        "deaths_table": sorted(deaths_table, key=lambda row: row["generation_of_death"]),
        "reproduction_log": sorted(reproduction_log, key=lambda row: row["generation"]),
        "help_interaction_log": sorted(help_rows, key=lambda row: row["generation"]),
        "lineage_summary": lineage_summary,
    }


def _render_run_human_report(run_record: Dict[str, object]) -> Dict[str, object]:
    metrics = dict(run_record.get("metrics", {}))
    params = dict(run_record.get("params", {}))
    timeline = list(run_record.get("timeline", []))
    populations = [int(s.get("population", 0)) for s in timeline]
    peak_pop = max(populations) if populations else 0
    peak_gen = (populations.index(peak_pop) + 1) if populations else 0
    growth_stop_gen = next((idx + 1 for idx, p in enumerate(populations) if p < populations[max(0, idx - 1)] and idx > 0), None) if populations else None
    late_births = sum(int(s.get("births", 0) or 0) for s in timeline[-20:])

    diagnosis = metrics.get("diagnosis", []) or []
    failure_bucket = _classify_run_failure_bucket(metrics)
    what_happened = [
        f"Started with {params.get('agents', 25)} agents.",
        f"Peak population was {peak_pop} at generation {peak_gen}.",
        f"Final population ended at {metrics.get('final_population')} with {metrics.get('lineage_count')} lineages.",
        f"Late births over final 20 generations: {late_births}.",
    ]
    if growth_stop_gen:
        what_happened.append(f"Growth softened after approximately generation {growth_stop_gen}.")
    if diagnosis:
        what_happened.append(f"Observed failure signals: {', '.join(diagnosis)}.")

    why_likely = []
    if "overshoot" in diagnosis:
        why_likely.append("Reproduction pressure exceeded ecosystem carrying capacity.")
    if "collapse" in diagnosis:
        why_likely.append("Agents likely burned energy faster than they could replenish it.")
    if "early_dominance" in diagnosis or "late_dominance" in diagnosis:
        why_likely.append("One lineage captured too much reward/energy share and suppressed alternatives.")
    if "low_diversity" in diagnosis:
        why_likely.append("Mutation/diversity pressure was insufficient to sustain multiple viable families.")
    if "instability" in diagnosis:
        why_likely.append("Population oscillation remained too high in late generations.")
    if "weak_throughput" in diagnosis:
        why_likely.append("Task-solving throughput was too weak to fuel healthy reproduction.")
    if not why_likely:
        why_likely.append("Run remained mixed: some gates passed, but stability and lineage balance were not both sustained.")

    score_explanation = _score_explanation(metrics)
    return {
        "run_number": run_record.get("run_in_batch"),
        "result": metrics.get("label"),
        "failure_bucket": failure_bucket,
        "what_happened": what_happened,
        "why_it_likely_happened": why_likely,
        "score_explanation": score_explanation,
        "next_change": run_record.get("adjustment_reason", "No next change recorded."),
        "observability": _build_run_observability(run_record.get("result", {})),
    }


def _render_human_summary(*, final_outcome: str, best_run: Dict[str, object] | None, failure_modes: Dict[str, int], run_records: List[Dict[str, object]], advisory_usage: Dict[str, object]) -> Dict[str, object]:
    run_summaries = [_render_run_human_report(rec) for rec in run_records]
    best_metrics = (best_run or {}).get("metrics", {})
    best_run_summary = _render_run_human_report(best_run) if best_run else None

    failure_counts = {
        "overshoot": 0,
        "collapse": 0,
        "dominance": 0,
        "low_diversity": 0,
        "instability": 0,
        "weak_growth": 0,
        "near_success": 0,
    }
    for rec in run_records:
        bucket = _classify_run_failure_bucket(dict(rec.get("metrics", {})))
        failure_counts[bucket] = failure_counts.get(bucket, 0) + 1

    parameter_journey: List[Dict[str, object]] = []
    prev_params: Dict[str, object] | None = None
    for rec in run_records:
        params = dict(rec.get("params", {}))
        delta = {}
        explanation = []
        if prev_params is not None:
            for key in ["reproduction_threshold", "mutation_rate", "upkeep_cost", "tasks_per_generation", "diversity_bonus", "diversity_min_lineages"]:
                if prev_params.get(key) != params.get(key):
                    delta[key] = {"from": prev_params.get(key), "to": params.get(key)}
            diag = set(rec.get("metrics", {}).get("diagnosis", []) or [])
            if "overshoot" in diag and "reproduction_threshold" in delta:
                explanation.append("Reproduction threshold increased to curb early over-birth and overshoot.")
            if "collapse" in diag and ("upkeep_cost" in delta or "reproduction_threshold" in delta):
                explanation.append("Survival pressure was relaxed to reduce collapse risk.")
            if "low_diversity" in diag and ("mutation_rate" in delta or "diversity_bonus" in delta):
                explanation.append("Mutation/diversity pressure increased to rebuild lineage variety.")
            if "instability" in diag and ("mutation_rate" in delta or "tasks_per_generation" in delta):
                explanation.append("Volatility controls were tightened to smooth late-run swings.")
        parameter_journey.append(
            {
                "run": rec.get("run_in_batch"),
                "score": rec.get("score"),
                "label": rec.get("label"),
                "changes": delta,
                "plain_english_reason": rec.get("adjustment_reason") or ("; ".join(explanation) if explanation else "Bounded local step update."),
            }
        )
        prev_params = params

    recommendation = "keep tuning"
    if str(final_outcome).lower().startswith("stable swarm achieved"):
        recommendation = "freeze this preset as current best"
    elif best_run and float(best_run.get("score", 0.0)) >= 75:
        recommendation = "freeze this preset as current best"
    elif failure_counts["collapse"] + failure_counts["overshoot"] > max(1, len(run_records) // 2):
        recommendation = "abandon this region of parameter space"

    return {
        "executive_summary": {
            "stable_swarm_achieved": final_outcome == "Stable swarm achieved",
            "outcome": final_outcome,
            "runs_attempted": len(run_records),
            "best_run_number": best_run.get("run_in_batch") if best_run else None,
            "best_score": best_run.get("score") if best_run else None,
            "overall_learning": "Bounded deterministic tuning mapped which levers shift growth, dominance, and late-run stability.",
        },
        "best_run_summary": best_run_summary,
        "failure_breakdown": failure_counts,
        "parameter_evolution": parameter_journey,
        "run_level_reports": run_summaries,
        "final_recommendation": {
            "action": recommendation,
            "next_likely_lever": "reproduction_threshold + diversity_bonus" if failure_counts.get("dominance", 0) else "upkeep_cost + tasks_per_generation",
            "api_note": advisory_usage.get("operator_summaries", []),
        },
        "best_run_card": {
            "best_score": (best_run or {}).get("score"),
            "final_population": best_metrics.get("final_population"),
            "lineage_count": best_metrics.get("lineage_count"),
            "diversity_score": best_metrics.get("diversity_score"),
            "outcome_plain_english": PLAIN_LABELS.get(str((best_run or {}).get("label", "unknown")), "unknown"),
        },
        "advanced_details": {"raw_json_available": True},
    }


def _phase2_adaptive_update(params: Dict[str, object], score: Dict[str, object]) -> Dict[str, object]:
    if isinstance(score, dict) and "gates" in score:
        adapted = dict(params)
        gates = score.get("gates", {})
        population = score.get("component_metrics", {}).get("population_stability", {})
        if not gates.get("late_population_ok", True):
            adapted["initial_energy"] = min(180, int(adapted.get("initial_energy", 100)) + 8)
            adapted["upkeep_cost"] = max(2, int(adapted.get("upkeep_cost", 6)) - 1)
            adapted["reproduction_threshold"] = max(95.0, float(adapted.get("reproduction_threshold", 130.0)) - 4.0)
        if not gates.get("target_band_stability_ok", True) or not gates.get("population_volatility_ok", True):
            adapted["reproduction_cooldown_enabled"] = True
            adapted["reproduction_cooldown_generations"] = min(5, int(adapted.get("reproduction_cooldown_generations", 2)) + 1)
            adapted["mutation_rate"] = max(0.05, float(adapted.get("mutation_rate", 0.15)) - 0.01)
        if not gates.get("late_lineage_count_ok", True) or not gates.get("top_lineage_share_ok", True):
            adapted["anti_dominance_enabled"] = True
            adapted["lineage_size_penalty_enabled"] = True
            adapted["lineage_energy_share_penalty_enabled"] = True
            adapted["diversity_bonus"] = min(3.0, float(adapted.get("diversity_bonus", 1.0)) + 0.2)
            adapted["immigrant_injection_count"] = min(8, int(adapted.get("immigrant_injection_count", 2)) + 1)
        if not gates.get("solve_rate_ok", True):
            adapted["tasks_per_generation"] = max(12, int(adapted.get("tasks_per_generation", 15)) - 2)
        if population.get("late_avg_population", 0.0) > 120:
            adapted["upkeep_cost"] = min(12, int(adapted.get("upkeep_cost", 6)) + 1)
            adapted["reproduction_threshold"] = min(180.0, float(adapted.get("reproduction_threshold", 130.0)) + 2.0)
        return adapted
    return adjust_parameters(params, score)[0]


def discover_presets() -> List[Dict[str, object]]:
    presets: List[Dict[str, object]] = []
    seen_ids: set[str] = set()
    search_roots = [CONFIGS_PATH, CONFIG_PATH]
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if path.name.startswith("_"):
                continue
            cfg = json.loads(path.read_text(encoding="utf-8"))
            # Skip experiment-spec JSON files that cannot be executed as a direct simulation run.
            if {"target_preset", "search", "sweep", "base"}.issubset(cfg.keys()):
                continue
            rel_id = str(path.relative_to(root)).replace("\\", "/")
            if rel_id in seen_ids:
                continue
            label = path.stem.replace("_", " ").title()
            if "simulation" in cfg:
                sim = cfg.get("simulation", {})
                diversity = cfg.get("diversity", {})
                anti = cfg.get("anti_dominance", {})
                diminishing = anti.get("diminishing_rewards", {}) if isinstance(anti, dict) else {}
                lineage = anti.get("lineage_penalty", {}) if isinstance(anti, dict) else {}
                reproduction = anti.get("reproduction", {}) if isinstance(anti, dict) else {}
                tier_mix = cfg.get("tier_mix", {})
                preset = {
                    "id": rel_id,
                    "label": label,
                    "preset_name": cfg.get("preset_name", path.stem),
                    "agents": sim.get("agents"),
                    "generations": sim.get("generations"),
                    "initial_energy": sim.get("initial_energy"),
                    "upkeep_cost": sim.get("upkeep_cost"),
                    "tasks_per_generation": sim.get("tasks_per_generation"),
                    "reproduction_threshold": sim.get("reproduction_threshold"),
                    "mutation_rate": sim.get("mutation_rate"),
                    "diversity_bonus": diversity.get("bonus", 1.0),
                    "diversity_min_lineages": diversity.get("min_lineages", 4),
                    "immigrant_injection_count": diversity.get("immigrant_injection_count", 2),
                    "tier_mix": {"1": tier_mix.get("t1", 0.34), "2": tier_mix.get("t2", 0.31), "3": tier_mix.get("t3", 0.21), "4": tier_mix.get("t4", 0.14)},
                    "anti_dominance_enabled": anti.get("enabled", False),
                    "diminishing_reward_enabled": diminishing.get("enabled", False),
                    "diminishing_reward_k": diminishing.get("k", 250.0),
                    "lineage_size_penalty_enabled": lineage.get("enabled", False),
                    "lineage_size_penalty_threshold": lineage.get("threshold", 45),
                    "lineage_size_penalty_multiplier": lineage.get("strength", 0.85),
                    "lineage_energy_share_penalty_enabled": lineage.get("energy_share_enabled", False),
                    "lineage_energy_share_penalty_threshold": lineage.get("energy_share_threshold", 0.30),
                    "lineage_energy_share_penalty_multiplier": lineage.get("energy_share_strength", 0.80),
                    "reproduction_cooldown_enabled": reproduction.get("cooldown_enabled", False),
                    "reproduction_cooldown_generations": reproduction.get("cooldown_generations", 2),
                    "reproduction_cost": reproduction.get("cost", 32.0),
                    "child_energy_fraction": reproduction.get("energy_split", 0.5),
                    "schema": cfg,
                }
            else:
                preset = {
                    "id": rel_id,
                    "label": label,
                    "preset_name": cfg.get("preset_name", path.stem),
                    "seed": cfg.get("seed"),
                    "agents": cfg.get("agents"),
                    "generations": cfg.get("generations"),
                    "tasks_per_generation": cfg.get("tasks_per_generation"),
                    "initial_energy": cfg.get("initial_energy"),
                    "upkeep_cost": cfg.get("upkeep_cost"),
                    "reproduction_threshold": cfg.get("reproduction_threshold", 130),
                    "mutation_rate": cfg.get("mutation_rate", 0.15),
                    "diversity_bonus": cfg.get("diversity_bonus", 1.0),
                    "diversity_min_lineages": cfg.get("diversity_min_lineages", 4),
                    "immigrant_injection_count": cfg.get("immigrant_injection_count", 2),
                    "tier_mix": cfg.get("tier_mix", {"1": 0.34, "2": 0.31, "3": 0.21, "4": 0.14}),
                    "anti_dominance_enabled": cfg.get("anti_dominance_enabled", False),
                    "diminishing_reward_enabled": cfg.get("diminishing_reward_enabled", False),
                    "diminishing_reward_k": cfg.get("diminishing_reward_k", 250.0),
                    "lineage_size_penalty_enabled": cfg.get("lineage_size_penalty_enabled", False),
                    "lineage_size_penalty_threshold": cfg.get("lineage_size_penalty_threshold", 45),
                    "lineage_size_penalty_multiplier": cfg.get("lineage_size_penalty_multiplier", 0.85),
                    "lineage_energy_share_penalty_enabled": cfg.get("lineage_energy_share_penalty_enabled", False),
                    "lineage_energy_share_penalty_threshold": cfg.get("lineage_energy_share_penalty_threshold", 0.30),
                    "lineage_energy_share_penalty_multiplier": cfg.get("lineage_energy_share_penalty_multiplier", 0.80),
                    "reproduction_cooldown_enabled": cfg.get("reproduction_cooldown_enabled", False),
                    "reproduction_cooldown_generations": cfg.get("reproduction_cooldown_generations", 2),
                    "reproduction_cost": cfg.get("reproduction_cost", 32.0),
                    "child_energy_fraction": cfg.get("child_energy_fraction", 0.5),
                    "schema": None,
                }
            presets.append(preset)
            seen_ids.add(rel_id)
    if not presets and LEGACY_DEFAULT_CONFIG.exists():
        presets.insert(
            0,
            {
                "id": "default.json",
                "label": "Default",
                "preset_name": "default",
                "seed": 42,
                "agents": 10,
                "generations": 50,
                "tasks_per_generation": 15,
                "initial_energy": 100,
                "upkeep_cost": 6,
                "reproduction_threshold": 130,
                "mutation_rate": 0.15,
                "diversity_bonus": 1.0,
                "diversity_min_lineages": 4,
                "immigrant_injection_count": 2,
                "tier_mix": {"1": 0.34, "2": 0.31, "3": 0.21, "4": 0.14},
                "anti_dominance_enabled": False,
                "diminishing_reward_enabled": False,
                "diminishing_reward_k": 250.0,
                "lineage_size_penalty_enabled": False,
                "lineage_size_penalty_threshold": 45,
                "lineage_size_penalty_multiplier": 0.85,
                "lineage_energy_share_penalty_enabled": False,
                "lineage_energy_share_penalty_threshold": 0.30,
                "lineage_energy_share_penalty_multiplier": 0.80,
                "reproduction_cooldown_enabled": False,
                "reproduction_cooldown_generations": 2,
                "reproduction_cost": 32.0,
                "child_energy_fraction": 0.5,
                "schema": None,
            },
        )

    return presets


def config_from_request(payload: dict):
    preset = payload.get("preset", "ecosystem_optimal_v1.json")
    candidate_paths = [(CONFIGS_PATH / preset).resolve(), (CONFIG_PATH / preset).resolve()]
    preset_path = None
    for candidate in candidate_paths:
        if candidate.exists():
            preset_path = candidate
            break
    if preset_path is None:
        preset_path = DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else LEGACY_DEFAULT_CONFIG
    allowed_roots = [CONFIGS_PATH.resolve(), CONFIG_PATH.resolve()]
    if not any(str(preset_path).startswith(str(root)) for root in allowed_roots):
        raise ValueError("invalid preset path")
    try:
        cfg = load_config(preset_path)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"preset '{preset}' is not a runnable simulation config") from exc

    cfg.preset_name = str(payload.get("preset_name") or Path(preset).stem or cfg.preset_name)
    integer_fields = [
        "seed",
        "agents",
        "generations",
        "tasks_per_generation",
        "initial_energy",
        "upkeep_cost",
        "diversity_min_lineages",
        "immigrant_injection_count",
        "lineage_size_penalty_threshold",
        "reproduction_cooldown_generations",
    ]
    float_fields = [
        "reproduction_threshold",
        "mutation_rate",
        "diversity_bonus",
        "diminishing_reward_k",
        "lineage_size_penalty_multiplier",
        "lineage_energy_share_penalty_threshold",
        "lineage_energy_share_penalty_multiplier",
        "reproduction_cost",
        "child_energy_fraction",
    ]
    bool_fields = [
        "anti_dominance_enabled",
        "diminishing_reward_enabled",
        "lineage_size_penalty_enabled",
        "lineage_energy_share_penalty_enabled",
        "reproduction_cooldown_enabled",
    ]

    for key in integer_fields:
        if key in payload and payload[key] is not None:
            setattr(cfg, key, int(payload[key]))

    for key in float_fields:
        if key in payload and payload[key] is not None:
            setattr(cfg, key, float(payload[key]))

    for key in bool_fields:
        if key in payload and payload[key] is not None:
            setattr(cfg, key, bool(payload[key]))

    if "tier_mix" in payload and isinstance(payload["tier_mix"], dict):
        cfg.tier_mix = {str(k): float(v) for k, v in payload["tier_mix"].items()}

    if "run_label" in payload and payload["run_label"] is not None:
        cfg.run_label = str(payload["run_label"])

    if "overwrite" in payload:
        cfg.overwrite = bool(payload["overwrite"])

    return cfg


def _run_find_stable_swarm(max_runs: int, advisory_settings: Dict[str, object] | None = None) -> None:
    session_id = f"tuning_{uuid.uuid4().hex[:10]}"
    started = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()
    wall_clock_limit_seconds = 3600
    params = _stable_swarm_baseline()
    params["experiment_id"] = session_id
    params["log_dir"] = f"runs/tuning_sessions/{session_id}"

    session_dir = ROOT / "runs" / "tuning_sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    run_records: List[Dict[str, object]] = []
    failure_modes: Dict[str, int] = {}
    best_run: Dict[str, object] | None = None
    candidate_configs: List[Dict[str, object]] = []
    advisory_settings = {**_advisory_defaults(), **(advisory_settings or {})}
    advisory_calls = 0
    advisory_accepted = 0
    operator_summaries: List[str] = []

    TUNING_STATE["status"] = "running"
    TUNING_STATE["mode"] = "find_stable_swarm"
    TUNING_STATE["tuning_session_id"] = session_id
    TUNING_STATE["message"] = "Adaptive tuning rig in progress"
    TUNING_STATE["elapsed_seconds"] = 0
    TUNING_STATE["batch_number"] = 1
    TUNING_STATE["run_in_batch"] = 0
    TUNING_STATE["current_run"] = 0
    TUNING_STATE["max_runs"] = max_runs
    TUNING_STATE["current_parameters"] = None
    TUNING_STATE["best_score"] = 0.0
    TUNING_STATE["score_progression"] = []
    TUNING_STATE["best_run"] = None
    TUNING_STATE["winning_config"] = None
    TUNING_STATE["latest_outcome_label"] = None
    TUNING_STATE["latest_adjustment_reason"] = None
    TUNING_STATE["early_stop_reason"] = None
    TUNING_STATE["repeatability"] = None
    TUNING_STATE["candidate_configs"] = []
    TUNING_STATE["final_outcome"] = None
    TUNING_STATE["final_summary_path"] = None
    TUNING_STATE["error"] = None
    TUNING_STATE["goal_profile"] = GOAL_PROFILE
    TUNING_STATE["advisory_settings"] = advisory_settings
    TUNING_STATE["advisory_usage"] = {"calls": 0, "accepted": 0, "operator_summaries": []}
    TUNING_STATE["human_readable_summary"] = None
    TUNING_STATE["human_readable_report_path"] = None
    TUNING_STATE["session_diagnostics"] = {
        "improving_parameters": {},
        "dominance_worsening_parameters": {},
        "repeated_configs": [],
        "stall_events": [],
        "search_status": "progressing",
    }

    run_queue: Deque[Dict[str, object]] = deque([dict(params)])
    seen_signatures: Dict[tuple, int] = {}
    no_improvement_streak = 0

    try:
        for i in range(1, max_runs + 1):
            elapsed = int(time.monotonic() - started)
            TUNING_STATE["elapsed_seconds"] = elapsed
            if elapsed >= wall_clock_limit_seconds:
                TUNING_STATE["early_stop_reason"] = "1 hour wall-clock budget reached"
                break

            TUNING_STATE["current_run"] = i
            TUNING_STATE["run_in_batch"] = i
            current = dict(run_queue.popleft() if run_queue else params)
            current["run_label"] = f"{session_id}_r{i}"
            current["experiment_id"] = session_id
            current["log_dir"] = str(session_dir)
            current["agents"] = 25

            signature = canonical_config_signature(current)
            if seen_signatures.get(signature, 0) >= 1:
                TUNING_STATE["session_diagnostics"]["repeated_configs"].append({"run": i, "signature": str(signature)})
                if run_queue:
                    continue
                anchor = dict(best_run["params"] if best_run else params)
                anchor["agents"] = 25
                run_queue.extend(generate_local_variants(anchor, 3))
                continue
            seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
            TUNING_STATE["current_parameters"] = current

            cfg = config_from_request({"preset": "ecosystem_optimal_v1.json", **current})
            result = SimulationEngine(cfg).run()
            metrics = score_and_label_run(result)
            timeline = list(result.get("timeline", []))
            peak = max(timeline, key=lambda item: int(item.get("population", 0))) if timeline else {}
            metrics["peak_population"] = int(peak.get("population", 0) or 0)
            metrics["generation_of_peak_population"] = int(peak.get("generation", 0) or 0)

            run_record = {
                "run_id": uuid.uuid4().hex,
                "simulation_run_id": result.get("run_id", ""),
                "run_dir": result.get("run_dir", ""),
                "batch_number": 1,
                "run_in_batch": i,
                "params": current,
                "score": metrics["score"],
                "label": metrics["label"],
                "metrics": metrics,
                "timeline": timeline,
                "result": result,
                "adjustment_reason": "pending",
            }
            prior_score = float(run_records[-1]["score"]) if run_records else None
            run_record["score_delta"] = round(float(metrics["score"]) - prior_score, 4) if prior_score is not None else 0.0
            run_records.append(run_record)
            failure_modes[metrics["label"]] = failure_modes.get(metrics["label"], 0) + 1
            TUNING_STATE["latest_outcome_label"] = metrics["label"]
            TUNING_STATE["score_progression"].append({
                "run": i,
                "score": metrics["score"],
                "label": metrics["label"],
                "run_id": result.get("run_id", ""),
                "run_dir": result.get("run_dir", ""),
            })

            improved = best_run is None or metrics["score"] > float(best_run.get("score", 0.0))
            if improved:
                best_run = run_record
                TUNING_STATE["best_run"] = run_record
                TUNING_STATE["best_score"] = metrics["score"]
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1

            if metrics["healthy"] or metrics["near_healthy"]:
                candidate_configs.append({
                    "score": metrics["score"],
                    "label": metrics["label"],
                    "params": current,
                    "simulation_run_id": result.get("run_id", ""),
                })
                candidate_configs = sorted(candidate_configs, key=lambda c: c["score"], reverse=True)[:3]
                TUNING_STATE["candidate_configs"] = candidate_configs

            if metrics["healthy"]:
                repeatability_results = _run_repeatability_validation(current, session_id)
                TUNING_STATE["repeatability"] = repeatability_results
                if repeatability_results["passed"]:
                    TUNING_STATE["status"] = "success"
                    TUNING_STATE["message"] = "Stable swarm achieved"
                    TUNING_STATE["winning_config"] = current
                    TUNING_STATE["final_outcome"] = "Stable swarm achieved"
                    break
                TUNING_STATE["status"] = "near_miss"
                TUNING_STATE["message"] = "Promising config found, repeatability not yet proven"
                TUNING_STATE["winning_config"] = current
                TUNING_STATE["final_outcome"] = "Promising config found, repeatability not yet proven"
                break

            deterministic_updated, reason = adjust_parameters(current, metrics)
            deterministic_updated["agents"] = 25
            advice_record = {
                "advisory_input": None,
                "raw_advisory_response": None,
                "parsed_advisory_response": None,
                "advice_accepted": False,
                "applied_config_source": "deterministic_tuner",
                "merge_explanation": "deterministic baseline",
            }
            updated = deterministic_updated

            can_call_advisory = bool(advisory_settings.get("advisory_api_enabled")) and advisory_calls < int(advisory_settings.get("advisory_max_calls_per_session", 0))
            if can_call_advisory:
                advisory_payload = _build_advisory_payload(
                    session_id=session_id,
                    run_number=i,
                    elapsed_seconds=elapsed,
                    current=current,
                    result=result,
                    metrics=metrics,
                    run_records=run_records,
                )
                advisory_result = _call_advisory_api(advisory_payload, advisory_settings)
                advisory_calls += 1
                parsed = advisory_result.get("parsed") if isinstance(advisory_result, dict) else _default_advisory_response()
                merged, merge_explanation = _merge_advisory_with_deterministic(current, deterministic_updated, parsed if isinstance(parsed, dict) else _default_advisory_response())
                merged["agents"] = 25
                advice_record = {
                    "advisory_input": advisory_payload,
                    "raw_advisory_response": advisory_result.get("raw") if isinstance(advisory_result, dict) else None,
                    "parsed_advisory_response": parsed,
                    "advice_accepted": merge_explanation.startswith("merged_deterministic_api"),
                    "applied_config_source": "api_assisted_tuner" if merge_explanation.startswith("merged_deterministic_api") else "deterministic_tuner",
                    "merge_explanation": merge_explanation if not advisory_result.get("error") else f"{merge_explanation}; api_error={advisory_result.get('error')}",
                }
                if advice_record["advice_accepted"]:
                    advisory_accepted += 1
                if isinstance(parsed, dict) and parsed.get("operator_summary"):
                    operator_summaries.append(str(parsed.get("operator_summary")))
                updated = merged

            params = updated
            changed_parameters = {
                k: {"from": current.get(k), "to": updated.get(k)}
                for k in ["reproduction_threshold", "mutation_rate", "upkeep_cost", "tasks_per_generation", "diversity_bonus", "diversity_min_lineages"]
                if current.get(k) != updated.get(k)
            }
            run_records[-1]["changed_parameters"] = changed_parameters
            run_records[-1]["advisory"] = advice_record
            run_records[-1]["final_applied_config"] = updated
            TUNING_STATE["latest_adjustment_reason"] = reason
            run_records[-1]["adjustment_reason"] = reason
            TUNING_STATE["advisory_usage"] = {"calls": advisory_calls, "accepted": advisory_accepted, "operator_summaries": operator_summaries[-5:]}

            diagnoses = set(metrics.get("diagnosis", []))
            if improved:
                for p in ["reproduction_threshold", "upkeep_cost", "reproduction_cooldown_generations", "mutation_rate", "tasks_per_generation", "diversity_bonus", "diversity_min_lineages"]:
                    if current.get(p) != updated.get(p):
                        TUNING_STATE["session_diagnostics"]["improving_parameters"][p] = TUNING_STATE["session_diagnostics"]["improving_parameters"].get(p, 0) + 1
            if diagnoses.intersection({"early_dominance", "late_dominance", "low_diversity"}):
                for p in ["mutation_rate", "diversity_bonus", "diversity_min_lineages", "reproduction_cooldown_generations"]:
                    if current.get(p) != updated.get(p):
                        TUNING_STATE["session_diagnostics"]["dominance_worsening_parameters"][p] = TUNING_STATE["session_diagnostics"]["dominance_worsening_parameters"].get(p, 0) + 1

            run_queue.append(updated)
            if metrics["score"] >= 70:
                for variant in generate_local_variants(current, 3):
                    variant["agents"] = 25
                    variant["seed"] = int(current.get("seed", 42)) + (i * 7) + len(run_queue)
                    run_queue.append(variant)

            stalled = no_improvement_streak >= 4
            if stalled:
                TUNING_STATE["session_diagnostics"]["search_status"] = "stalled"
                TUNING_STATE["session_diagnostics"]["stall_events"].append({"run": i, "reason": "no score improvement or repeated configs"})
                anchor = dict(best_run["params"] if best_run else current)
                anchor["agents"] = 25
                for idx, variant in enumerate(generate_local_variants(anchor, 5), start=1):
                    variant["agents"] = 25
                    variant["seed"] = int(anchor.get("seed", 42)) + (i * 11) + idx
                    run_queue.append(variant)
                no_improvement_streak = 0
                TUNING_STATE["session_diagnostics"]["search_status"] = "recovering"

        elapsed = int(time.monotonic() - started)
        TUNING_STATE["elapsed_seconds"] = elapsed

        if TUNING_STATE["status"] == "running":
            if elapsed >= wall_clock_limit_seconds:
                TUNING_STATE["status"] = "timeout"
                TUNING_STATE["message"] = "No equilibrium found in 1 hour"
                TUNING_STATE["final_outcome"] = "No equilibrium found in 1 hour"
            elif candidate_configs:
                TUNING_STATE["status"] = "near_miss"
                TUNING_STATE["message"] = "Promising config found, repeatability not yet proven"
                TUNING_STATE["final_outcome"] = "Promising config found, repeatability not yet proven"
                TUNING_STATE["winning_config"] = candidate_configs[0]["params"]
            else:
                TUNING_STATE["status"] = "failed"
                TUNING_STATE["message"] = "Tuning completed without achieving equilibrium"
                TUNING_STATE["final_outcome"] = "No equilibrium found in 1 hour" if TUNING_STATE.get("early_stop_reason") else "Promising config found, repeatability not yet proven"

        summary = _build_tuning_summary(
            session_id=session_id,
            started_at=started_iso,
            elapsed_seconds=elapsed,
            run_records=run_records,
            best_run=best_run,
            candidate_configs=candidate_configs,
            failure_modes=failure_modes,
            final_outcome=TUNING_STATE.get("final_outcome") or "No equilibrium found in 1 hour",
            repeatability=TUNING_STATE.get("repeatability"),
            early_stop_reason=TUNING_STATE.get("early_stop_reason"),
            session_diagnostics=TUNING_STATE.get("session_diagnostics"),
            advisory_settings=advisory_settings,
            advisory_usage={"calls": advisory_calls, "accepted": advisory_accepted, "operator_summaries": operator_summaries[-5:]},
        )
        human_summary = _render_human_summary(
            final_outcome=TUNING_STATE.get("final_outcome") or "No Equilibrium Found",
            best_run=best_run,
            failure_modes=failure_modes,
            run_records=run_records,
            advisory_usage={"operator_summaries": operator_summaries[-5:]},
        )
        summary["human_readable_summary"] = human_summary
        summary_path = session_dir / "final_session_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        markdown_report_path = session_dir / "final_session_report.md"
        markdown_report_path.write_text(_build_markdown_report(summary), encoding="utf-8")

        runs_jsonl = session_dir / "runs.jsonl"
        with runs_jsonl.open("w", encoding="utf-8") as handle:
            for record in run_records:
                handle.write(json.dumps(record) + "\n")

        TUNING_STATE["final_summary_path"] = str(summary_path)
        TUNING_STATE["human_readable_summary"] = human_summary
        TUNING_STATE["human_readable_report_path"] = str(markdown_report_path)
    except Exception as exc:  # pragma: no cover
        TUNING_STATE["status"] = "error"
        TUNING_STATE["error"] = str(exc)
        TUNING_STATE["message"] = "Tuning failed"


def _run_repeatability_validation(base_params: Dict[str, object], session_id: str) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    pass_count = 0
    catastrophic = 0
    for i in range(5):
        cfg_params = dict(base_params)
        cfg_params["agents"] = 25
        cfg_params["seed"] = int(base_params.get("seed", 42)) + 100 + i
        cfg_params["run_label"] = f"{session_id}_repeatability_{i + 1}"
        cfg = config_from_request({"preset": "ecosystem_optimal_v1.json", **cfg_params})
        result = SimulationEngine(cfg).run()
        metrics = score_and_label_run(result)
        if metrics["healthy"]:
            pass_count += 1
        if metrics["label"] in {"failed_collapse", "failed_overshoot"}:
            catastrophic += 1
        results.append({
            "run_id": uuid.uuid4().hex,
            "simulation_run_id": result.get("run_id", ""),
            "run_dir": result.get("run_dir", ""),
            "seed": cfg_params["seed"],
            "score": metrics["score"],
            "label": metrics["label"],
        })

    return {
        "required_passes": 3,
        "pass_count": pass_count,
        "repeat_runs": 5,
        "catastrophic_failures": catastrophic,
        "passed": pass_count >= 3 and catastrophic == 0,
        "results": results,
    }


def _build_tuning_summary(*, session_id: str, started_at: str, elapsed_seconds: int, run_records: List[Dict[str, object]], best_run: Dict[str, object] | None, candidate_configs: List[Dict[str, object]], failure_modes: Dict[str, int], final_outcome: str, repeatability: Dict[str, object] | None, early_stop_reason: str | None, session_diagnostics: Dict[str, object] | None = None, advisory_settings: Dict[str, object] | None = None, advisory_usage: Dict[str, object] | None = None) -> Dict[str, object]:
    dominant = sorted(failure_modes.items(), key=lambda kv: kv[1], reverse=True)
    top_candidates = sorted(candidate_configs, key=lambda c: c["score"], reverse=True)[:3]
    suggested = "Increase anti-dominance pressure and reduce mutation volatility" if failure_modes.get("failed_overshoot", 0) > failure_modes.get("failed_collapse", 0) else "Increase initial energy and lower reproduction threshold to avoid collapse"
    human = _render_human_summary(
        final_outcome=final_outcome,
        best_run=best_run,
        failure_modes=failure_modes,
        run_records=run_records,
        advisory_usage=advisory_usage or {"calls": 0, "accepted": 0, "operator_summaries": []},
    )
    return {
        "tuning_session_id": session_id,
        "mode": "find_stable_swarm",
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "total_runs_executed": len(run_records),
        "best_score": best_run.get("score", 0.0) if best_run else 0.0,
        "best_config": best_run.get("params") if best_run else None,
        "top_candidate_configs": top_candidates,
        "dominant_failure_modes": [{"label": label, "count": count} for label, count in dominant],
        "repeatability": repeatability,
        "session_diagnostics": session_diagnostics or {},
        "early_stop_reason": early_stop_reason,
        "suggested_next_tuning_direction": suggested,
        "final_outcome": final_outcome,
        "goal_profile": GOAL_PROFILE,
        "advisory_settings": advisory_settings or _advisory_defaults(),
        "advisory_usage": advisory_usage or {"calls": 0, "accepted": 0, "operator_summaries": []},
        "starting_population_root_cause": "Find Stable Swarm UI copy still said 100 agents; tuning execution always now force-clamps agents=25 in baseline, run loop, and repeatability path.",
        "human_readable_summary": human,
    }


def _build_markdown_report(summary: Dict[str, object]) -> str:
    best = summary.get("best_config") or {}
    candidates = summary.get("top_candidate_configs") or []
    failures = summary.get("dominant_failure_modes") or []
    advisory = summary.get("advisory_usage") or {}
    human = summary.get("human_readable_summary") or {}
    exec_summary = human.get("executive_summary") or {}
    best_run_summary = human.get("best_run_summary") or {}

    lines = [
        f"# Tuning Session Report - {summary.get('tuning_session_id')}",
        "",
        "## A. Executive Summary",
        f"- Stable swarm achieved: **{exec_summary.get('stable_swarm_achieved', False)}**",
        f"- Outcome: **{summary.get('final_outcome')}**",
        f"- Runs attempted: **{summary.get('total_runs_executed')}**",
        f"- Best run: **Run {exec_summary.get('best_run_number')}** (score={summary.get('best_score')})",
        f"- Overall learning: {exec_summary.get('overall_learning', 'See run-level summaries below.')}",
        "",
        "## B. Best Run Summary",
        f"- Why it was best: {best_run_summary.get('score_explanation', {}).get('positives', ['Highest score under current gates.'])[0]}",
        f"- Biggest remaining failure mode: {best_run_summary.get('score_explanation', {}).get('biggest_failure_mode', 'n/a')}",
        f"- Config: `{json.dumps(best, sort_keys=True)}`",
        "",
        "## C. Failure Breakdown Across Runs",
    ]
    for bucket, count in (human.get("failure_breakdown") or {}).items():
        lines.append(f"- {bucket}: {count}")

    lines += ["", "## D. Parameter Evolution"]
    for row in human.get("parameter_evolution", [])[:20]:
        lines.append(f"- Run {row.get('run')}: {row.get('plain_english_reason')}")

    lines += ["", "## E. Final Recommendation", f"- {human.get('final_recommendation', {}).get('action', summary.get('suggested_next_tuning_direction'))}"]

    lines += ["", "## Run-Level Human Reports"]
    for run in human.get("run_level_reports", []):
        lines.append(f"### Run {run.get('run_number')} Summary")
        lines.append("- What happened:")
        for item in run.get("what_happened", []):
            lines.append(f"  - {item}")
        lines.append("- Why it likely happened:")
        for item in run.get("why_it_likely_happened", []):
            lines.append(f"  - {item}")
        lines.append(f"- What tuner changed next: {run.get('next_change')}")

    lines += [
        "",
        "## Top Candidate Configs",
    ]
    for idx, row in enumerate(candidates[:3], start=1):
        lines.append(f"- {idx}. score={row.get('score')} label={row.get('label')} params={json.dumps(row.get('params', {}), sort_keys=True)}")

    lines += ["", "## Failure Distribution (Raw Labels)"]
    for row in failures:
        lines.append(f"- {row.get('label')}: {row.get('count')}")
    lines += [
        "",
        "## Advisory API Usage",
        f"- Calls: {advisory.get('calls', 0)}",
        f"- Accepted merges: {advisory.get('accepted', 0)}",
        f"- Operator summaries: {advisory.get('operator_summaries', [])}",
        f"- API key env var: {summary.get('advisory_settings', {}).get('advisory_api_key_env', 'GENESIS2_ADVISORY_API_KEY')}",
        "",
        "## Raw JSON",
        f"- Session summary JSON: `final_session_summary.json`",
        f"- Runs JSONL: `runs.jsonl`",
    ]
    return "\n".join(lines) + "\n"



class GenesisHandler(BaseHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ["/", "/index.html"]:
            html = UI_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/api/state":
            if LAST_RESULT.get("status") == "running" and LAST_RESULT.get("run_dir"):
                LAST_RESULT["problem_board_events"] = events_from_problem_metrics(LAST_RESULT["run_dir"])
            self._json(LAST_RESULT)
            return

        if self.path == "/api/presets":
            self._json({"presets": discover_presets()})
            return

        if self.path == "/api/tuning/state":
            self._json(TUNING_STATE)
            return

        self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path == "/api/tuning/start":
            with RUN_LOCK:
                if LAST_RESULT["status"] == "running" or TUNING_STATE["status"] == "running":
                    self._json({"error": "A run or tuning session is already in progress"}, status=409)
                    return
                content_len = int(self.headers.get("Content-Length", 0))
                incoming = json.loads(self.rfile.read(content_len) or b"{}")
                mode = str(incoming.get("mode", "find_stable_swarm"))
                max_runs = int(incoming.get("max_runs", 8))
                if mode != "find_stable_swarm":
                    self._json({"error": "Only find_stable_swarm mode is currently implemented"}, status=400)
                    return

                advisory_settings = {**_advisory_defaults()}
                for key in advisory_settings:
                    if key in incoming:
                        advisory_settings[key] = incoming[key]
                thread = threading.Thread(target=_run_find_stable_swarm, args=(max(1, max_runs), advisory_settings), daemon=True)
                thread.start()
            self._json({"status": "running"}, status=202)
            return

        if self.path == "/api/run":
            with RUN_LOCK:
                if LAST_RESULT["status"] == "running":
                    self._json({"error": "A run is already in progress"}, status=409)
                    return

                content_len = int(self.headers.get("Content-Length", 0))
                incoming = json.loads(self.rfile.read(content_len) or b"{}")

                try:
                    cfg = config_from_request(incoming)
                except ValueError as exc:
                    self._json({"error": str(exc)}, status=400)
                    return

                LAST_RESULT["status"] = "running"
                LAST_RESULT["result"] = None
                LAST_RESULT["error"] = None
                LAST_RESULT["run_dir"] = None
                LAST_RESULT["problem_board_events"] = []
                LAST_RESULT["progress"] = {
                    "generation": 0,
                    "population": cfg.agents,
                    "births": 0,
                    "deaths": 0,
                }

                def worker() -> None:
                    try:
                        engine = SimulationEngine(cfg)

                        def progress_callback(step: Dict) -> None:
                            LAST_RESULT["progress"] = {
                                "generation": step.get("generation", 0),
                                "population": step.get("population", 0),
                                "births": step.get("births", 0),
                                "deaths": step.get("deaths", 0),
                                "avg_energy": step.get("energy_distribution", {}).get("mean", 0),
                            }

                        result = engine.run(progress_callback=progress_callback)
                        LAST_RESULT["status"] = "done"
                        LAST_RESULT["result"] = result
                        LAST_RESULT["run_dir"] = result.get("run_dir")
                        LAST_RESULT["problem_board_events"] = events_from_problem_metrics(LAST_RESULT.get("run_dir") or "")
                    except Exception as exc:  # pragma: no cover
                        LAST_RESULT["status"] = "error"
                        LAST_RESULT["error"] = str(exc)

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()

            self._json({"status": "running"}, status=202)
            return

        self._json({"error": "not found"}, status=404)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), GenesisHandler)
    print(f"Genesis2 UI: http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.host, args.port)
