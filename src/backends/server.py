from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Dict, List
import uuid

from src.analytics.bvl import events_from_problem_metrics
from src.engine.simulation import SimulationEngine, load_config
from src.tuner.adaptive_rig import adjust_parameters, build_find_stable_swarm_baseline, score_and_label_run


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
}


def _stable_swarm_baseline() -> Dict[str, object]:
    source = DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else LEGACY_DEFAULT_CONFIG
    params = build_find_stable_swarm_baseline(source)
    params.update({"seed": 42, "run_label": "adaptive_tuning_rig_baseline", "experiment_id": "adaptive_tuning_rig", "log_dir": "runs/tuning_sessions"})
    return params


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


def _run_find_stable_swarm(max_runs: int) -> None:
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

    try:
        for i in range(1, max_runs + 1):
            elapsed = int(time.monotonic() - started)
            TUNING_STATE["elapsed_seconds"] = elapsed
            if elapsed >= wall_clock_limit_seconds:
                TUNING_STATE["early_stop_reason"] = "1 hour wall-clock budget reached"
                break

            TUNING_STATE["current_run"] = i
            TUNING_STATE["run_in_batch"] = i
            current = dict(params)
            current["run_label"] = f"{session_id}_r{i}"
            current["experiment_id"] = session_id
            current["log_dir"] = str(session_dir)
            TUNING_STATE["current_parameters"] = current

            cfg = config_from_request({"preset": "ecosystem_optimal_v1.json", **current})
            result = SimulationEngine(cfg).run()
            metrics = score_and_label_run(result)

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
            }
            run_records.append(run_record)
            failure_modes[metrics["label"]] = failure_modes.get(metrics["label"], 0) + 1
            TUNING_STATE["latest_outcome_label"] = metrics["label"]

            progression_entry = {
                "run": i,
                "score": metrics["score"],
                "label": metrics["label"],
                "run_id": result.get("run_id", ""),
                "run_dir": result.get("run_dir", ""),
            }
            TUNING_STATE["score_progression"].append(progression_entry)

            if best_run is None or metrics["score"] > float(best_run.get("score", 0.0)):
                best_run = run_record
                TUNING_STATE["best_run"] = run_record
                TUNING_STATE["best_score"] = metrics["score"]

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

            updated, reason = adjust_parameters(current, metrics)
            params = updated
            TUNING_STATE["latest_adjustment_reason"] = reason

            if metrics["label"] in {"failed_collapse", "failed_overshoot"} and i >= 3:
                last_labels = [r["label"] for r in run_records[-3:]]
                if len(set(last_labels)) == 1:
                    TUNING_STATE["early_stop_reason"] = f"Early stop on clearly bad direction: repeated {metrics['label']}"
                    break

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
        )
        summary_path = session_dir / "final_session_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        runs_jsonl = session_dir / "runs.jsonl"
        with runs_jsonl.open("w", encoding="utf-8") as handle:
            for record in run_records:
                handle.write(json.dumps(record) + "\n")

        TUNING_STATE["final_summary_path"] = str(summary_path)
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


def _build_tuning_summary(*, session_id: str, started_at: str, elapsed_seconds: int, run_records: List[Dict[str, object]], best_run: Dict[str, object] | None, candidate_configs: List[Dict[str, object]], failure_modes: Dict[str, int], final_outcome: str, repeatability: Dict[str, object] | None, early_stop_reason: str | None) -> Dict[str, object]:
    dominant = sorted(failure_modes.items(), key=lambda kv: kv[1], reverse=True)
    top_candidates = sorted(candidate_configs, key=lambda c: c["score"], reverse=True)[:3]
    suggested = "Increase anti-dominance pressure and reduce mutation volatility" if failure_modes.get("failed_overshoot", 0) > failure_modes.get("failed_collapse", 0) else "Increase initial energy and lower reproduction threshold to avoid collapse"
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
        "early_stop_reason": early_stop_reason,
        "suggested_next_tuning_direction": suggested,
        "final_outcome": final_outcome,
    }



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

                thread = threading.Thread(target=_run_find_stable_swarm, args=(max(1, max_runs),), daemon=True)
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
