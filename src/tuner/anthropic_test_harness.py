from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error, request

from src.engine.simulation import SimulationEngine, load_config
from src.tuner.adaptive_rig import build_find_stable_swarm_baseline, score_and_label_run

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GOAL: Dict[str, Any] = {
    "target_label": "healthy",
    "stop_on_labels": [
        "failed_overshoot",
        "failed_collapse",
        "failed_low_diversity",
        "failed_dominance",
        "failed_instability",
        "failed_wrong_start",
    ],
    "stop_on_extinction": True,
    "stop_on_non_target_success": True,
}

LEVER_SPECS: Dict[str, Dict[str, float]] = {
    "initial_energy": {"min": 40, "max": 240, "max_step": 20},
    "upkeep_cost": {"min": 2, "max": 14, "max_step": 2},
    "tasks_per_generation": {"min": 20, "max": 120, "max_step": 10},
    "reproduction_threshold": {"min": 90, "max": 220, "max_step": 10},
    "mutation_rate": {"min": 0.05, "max": 0.5, "max_step": 0.04},
    "diversity_bonus": {"min": 0.05, "max": 0.8, "max_step": 0.06},
    "diversity_min_lineages": {"min": 8, "max": 28, "max_step": 2},
    "lineage_size_penalty_threshold": {"min": 10, "max": 80, "max_step": 5},
    "lineage_energy_share_penalty_threshold": {"min": 0.2, "max": 0.8, "max_step": 0.05},
    "reproduction_cooldown_generations": {"min": 1, "max": 8, "max_step": 2},
}


def _extract_json_object(text: str) -> Dict[str, object] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _call_anthropic_advisor(*, api_key: str, model: str, temperature: float, system_prompt: str, user_prompt: str, timeout_seconds: float) -> Dict[str, object]:
    body = {
        "model": model,
        "max_tokens": 900,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    req = request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            raw_text = resp.read().decode("utf-8")
        payload = json.loads(raw_text)
        content = payload.get("content", [])
        text = "\n".join([str(x.get("text", "")) for x in content if isinstance(x, dict)])
        parsed = _extract_json_object(text)
        return {"error": None if parsed else "invalid_json", "raw": raw_text, "parsed": parsed}
    except (error.HTTPError, error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"error": f"request_failed: {exc}", "raw": None, "parsed": None}


def _clamp_recommendation(current: Dict[str, Any], recommendation: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    accepted = dict(current)
    notes: List[str] = []
    for key, value in recommendation.items():
        if key not in LEVER_SPECS:
            notes.append(f"ignored {key}: unknown lever")
            continue
        spec = LEVER_SPECS[key]
        try:
            proposed = float(value)
        except (TypeError, ValueError):
            notes.append(f"ignored {key}: non-numeric")
            continue
        current_value = float(current.get(key, proposed))
        delta = proposed - current_value
        if abs(delta) > float(spec["max_step"]):
            proposed = current_value + (float(spec["max_step"]) if delta > 0 else -float(spec["max_step"]))
            notes.append(f"clamped {key}: step")
        proposed = max(float(spec["min"]), min(float(spec["max"]), proposed))
        accepted[key] = int(round(proposed)) if isinstance(current.get(key), int) else round(proposed, 6)
    return accepted, notes


def _stop_reason(goal: Dict[str, Any], metrics: Dict[str, Any], result: Dict[str, Any]) -> str | None:
    label = str(metrics.get("label", ""))
    final_pop = int(result.get("final_population", 0))
    target_label = str(goal.get("target_label", "healthy"))

    if goal.get("stop_on_extinction", True) and final_pop <= 0:
        return "extinction"
    if label in set(goal.get("stop_on_labels", [])):
        return f"failure_label:{label}"
    if goal.get("stop_on_non_target_success", True) and label in {"healthy", "near_healthy"} and label != target_label:
        return f"other_goal_met:{label}"
    if label == target_label:
        return f"target_met:{label}"
    return None


def _report_markdown(run_events: List[Dict[str, Any]], goal: Dict[str, Any], stop_reason: str, recommendation_notes: List[str]) -> str:
    if run_events:
        best = max(run_events, key=lambda event: float(event["metrics"].get("score", 0.0)))
        best_metrics = best["metrics"]
        score_history = [float(event["metrics"].get("score", 0.0)) for event in run_events]
        score_delta = score_history[-1] - score_history[0]
    else:
        best = None
        best_metrics = {}
        score_history = []
        score_delta = 0.0

    lines = [
        "# Anthropic Tuning Harness Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Goal target_label: `{goal.get('target_label')}`",
        f"Stop reason: `{stop_reason}`",
        "",
        "## Session summary",
        "",
        f"- Total runs: **{len(run_events)}**",
        f"- Score trend (first→last): **{score_history[0]:.2f} → {score_history[-1]:.2f}** (Δ {score_delta:+.2f})" if score_history else "- Score trend (first→last): n/a",
        (
            f"- Best run: **Run {best['run_index']}** with score **{best_metrics.get('score', 0.0):.2f}** and label `{best_metrics.get('label', 'unknown')}`"
            if best
            else "- Best run: n/a"
        ),
        "",
        "## Run results table",
        "",
        "| Run | Cycle | Score | Label | Final Pop | Lineages | Diversity | Diagnosis |",
        "|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for event in run_events:
        metrics = event["metrics"]
        diagnosis = ", ".join(metrics.get("diagnosis", [])) or "none"
        lines.append(
            "| {run_index} | {cycle} | {score:.2f} | `{label}` | {final_population} | {lineage_count} | {diversity_score:.3f} | {diagnosis} |".format(
                run_index=event["run_index"],
                cycle=event["cycle"],
                score=float(metrics.get("score", 0.0)),
                label=metrics.get("label", "unknown"),
                final_population=event["result"].get("final_population", 0),
                lineage_count=metrics.get("lineage_count", 0),
                diversity_score=float(metrics.get("diversity_score", 0.0)),
                diagnosis=diagnosis,
            )
        )

    lines.extend([
        "",
        "## Run log",
    ])
    for event in run_events:
        lines.extend(
            [
                f"- Run {event['run_index']}: label=`{event['metrics']['label']}` score={event['metrics']['score']:.2f} final_population={event['result']['final_population']} run_dir=`{event['result'].get('run_dir', '')}`",
                f"  - diagnosis: {', '.join(event['metrics'].get('diagnosis', [])) or 'none'}",
            ]
        )
    if recommendation_notes:
        lines.extend(["", "## Recommendation clamp notes"] + [f"- {note}" for note in recommendation_notes])
    return "\n".join(lines)


def _write_live_status(session_dir: Path, run_events: List[Dict[str, Any]]) -> None:
    if not run_events:
        return
    csv_lines = ["run_index,cycle,score,label,final_population,lineage_count,diversity_score,diagnosis"]
    for event in run_events:
        metrics = event["metrics"]
        diagnosis = "|".join(metrics.get("diagnosis", [])) or "none"
        csv_lines.append(
            "{run_index},{cycle},{score:.4f},{label},{final_population},{lineage_count},{diversity_score:.6f},{diagnosis}".format(
                run_index=event["run_index"],
                cycle=event["cycle"],
                score=float(metrics.get("score", 0.0)),
                label=metrics.get("label", "unknown"),
                final_population=event["result"].get("final_population", 0),
                lineage_count=metrics.get("lineage_count", 0),
                diversity_score=float(metrics.get("diversity_score", 0.0)),
                diagnosis=diagnosis,
            )
        )
    (session_dir / "run_results.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def run_harness(spec_path: Path) -> Dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    session_id = f"anthropic_harness_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    session_dir = ROOT / "runs" / "anthropic_harness" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = ROOT / str(spec.get("base_config", "config/default.json"))
    params = build_find_stable_swarm_baseline(base_cfg_path)
    params.update(spec.get("initial_parameters", {}))
    goal = {**DEFAULT_GOAL, **spec.get("goal", {})}

    api_key_env = str(spec.get("anthropic", {}).get("api_key_env", "ANTHROPIC_API_KEY"))
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing Anthropic API key in env var {api_key_env}")

    anthropic = spec.get("anthropic", {})
    model = str(anthropic.get("model", "claude-3-5-sonnet-20241022"))
    temperature = float(anthropic.get("temperature", 0.1))
    timeout_seconds = float(anthropic.get("timeout_seconds", 30))

    max_cycles = int(spec.get("max_cycles", 3))
    max_runs_per_cycle = int(spec.get("max_runs_per_cycle", 10))
    run_events: List[Dict[str, Any]] = []
    last_stop_reason = "max_runs_reached"

    for cycle in range(1, max_cycles + 1):
        advisor_payload = {
            "goal": goal,
            "adjustable_parameters": [{"name": k, **v, "current": params.get(k)} for k, v in LEVER_SPECS.items()],
            "history": [{"run": e["run_index"], "label": e["metrics"]["label"], "score": e["metrics"]["score"]} for e in run_events[-8:]],
            "request": "Recommend next bounded config in JSON: {recommended_config, rationale, predicted_effects, avoid}.",
        }
        system_prompt = "You tune Genesis2 simulator parameters. Return only JSON."
        advisor = _call_anthropic_advisor(
            api_key=api_key,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=json.dumps(advisor_payload),
            timeout_seconds=timeout_seconds,
        )
        parsed = advisor.get("parsed") if isinstance(advisor, dict) else None
        recommendation = parsed.get("recommended_config", {}) if isinstance(parsed, dict) else {}
        params, recommendation_notes = _clamp_recommendation(params, recommendation if isinstance(recommendation, dict) else {})

        for _ in range(max_runs_per_cycle):
            run_index = len(run_events) + 1
            run_params = deepcopy(params)
            run_params["run_label"] = f"{session_id}_r{run_index}"
            run_params["experiment_id"] = session_id
            run_params["log_dir"] = str(session_dir)
            cfg = load_config(base_cfg_path)
            for key, value in run_params.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
            result = SimulationEngine(cfg).run()
            metrics = score_and_label_run(result)
            event = {
                "cycle": cycle,
                "run_index": run_index,
                "params": run_params,
                "advisor": advisor,
                "result": {"run_id": result.get("run_id"), "run_dir": result.get("run_dir"), "final_population": result.get("final_population")},
                "metrics": metrics,
            }
            run_events.append(event)
            _write_live_status(session_dir, run_events)
            reason = _stop_reason(goal, metrics, result)
            if reason:
                last_stop_reason = reason
                report = _report_markdown(run_events, goal, reason, recommendation_notes)
                report_path = session_dir / "incident_report.md"
                report_path.write_text(report, encoding="utf-8")

                follow_up_payload = {
                    "goal": goal,
                    "incident": {
                        "stop_reason": reason,
                        "latest_run": event,
                        "report": report,
                    },
                    "request": "Given this incident, return next recommended_config JSON to tune toward the target goal.",
                }
                follow_up = _call_anthropic_advisor(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(follow_up_payload),
                    timeout_seconds=timeout_seconds,
                )
                output = {
                    "session_id": session_id,
                    "goal": goal,
                    "stop_reason": reason,
                    "runs": run_events,
                    "report_path": str(report_path),
                    "follow_up_advisor": follow_up,
                }
                (session_dir / "session_output.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
                return output

    report = _report_markdown(run_events, goal, last_stop_reason, [])
    report_path = session_dir / "incident_report.md"
    report_path.write_text(report, encoding="utf-8")
    output = {
        "session_id": session_id,
        "goal": goal,
        "stop_reason": last_stop_reason,
        "runs": run_events,
        "report_path": str(report_path),
        "follow_up_advisor": None,
    }
    (session_dir / "session_output.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic-driven Genesis2 test harness")
    parser.add_argument("--spec", required=True, help="Path to harness spec JSON")
    args = parser.parse_args()
    result = run_harness(Path(args.spec))
    print(json.dumps({"session_id": result["session_id"], "stop_reason": result["stop_reason"], "report_path": result["report_path"]}, indent=2))


if __name__ == "__main__":
    main()
