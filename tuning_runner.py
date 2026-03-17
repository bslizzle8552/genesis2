from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List
from urllib import error, request


DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_OBJECTIVE = {
    "description": "Tune Genesis2 to a healthy swarm profile.",
    "targets": [
        "start ~25 agents",
        "grow to ~100 by generation ~80",
        "remain in population band 90-110",
        "maintain multiple viable lineages",
        "avoid lineage dominance",
        "preserve late reproduction",
        "preserve collaboration and specialization",
    ],
}
DEFAULT_LEVERS = {
    "agents": {"min": 20, "max": 30, "max_step": 2, "effect": "starting population"},
    "generations": {"min": 80, "max": 140, "max_step": 10, "effect": "simulation horizon"},
    "initial_energy": {"min": 80, "max": 220, "max_step": 15, "effect": "survival buffer"},
    "reproduction_threshold": {"min": 90, "max": 240, "max_step": 12, "effect": "birth strictness"},
    "mutation_rate": {"min": 0.05, "max": 0.5, "max_step": 0.04, "effect": "exploration volatility"},
    "upkeep_cost": {"min": 2, "max": 14, "max_step": 2, "effect": "resource pressure"},
    "tasks_per_generation": {"min": 10, "max": 120, "max_step": 10, "effect": "workload pressure"},
    "diversity_bonus": {"min": 0.0, "max": 1.0, "max_step": 0.08, "effect": "incentive for lineage spread"},
    "diversity_min_lineages": {"min": 2, "max": 30, "max_step": 3, "effect": "minimum lineage target"},
}


@dataclass
class RunnerSettings:
    max_runs: int
    max_seconds: int
    model: str
    simulator_cmd: str
    output_dir: Path
    api_key: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _call_anthropic(*, api_key: str, model: str, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any] | None:
    body = {
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.1,
        "system": system_prompt,
        "messages": [{"role": "user", "content": json.dumps(user_payload)}],
    }
    req = request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    text = ""
    for part in payload.get("content", []):
        if part.get("type") == "text":
            text += part.get("text", "")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None




def _log_advisor_event(session_dir: Path, event: Dict[str, Any]) -> None:
    path = session_dir / "advisor_trace.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **event}) + "\n")

def _clip(value: Any, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _validate_config(candidate: Dict[str, Any], current: Dict[str, Any], levers: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    bounded = dict(current)
    for name, spec in levers.items():
        if name not in candidate:
            continue
        current_value = float(current.get(name, 0))
        min_v = float(spec["min"])
        max_v = float(spec["max"])
        max_step = float(spec["max_step"])
        requested = _clip(candidate[name], min_v, max_v)
        requested = max(current_value - max_step, min(current_value + max_step, requested))
        bounded[name] = round(requested, 6)

    for field in ["agents", "generations", "initial_energy", "reproduction_threshold", "upkeep_cost", "tasks_per_generation", "diversity_min_lineages"]:
        if field in bounded:
            bounded[field] = int(round(float(bounded[field])))
    return bounded


def _fallback_adjustment(current: Dict[str, Any], last_output: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(current)
    mode = last_output.get("failure_mode")
    if mode == "overshoot":
        updated["reproduction_threshold"] = int(updated.get("reproduction_threshold", 130)) + 8
        updated["upkeep_cost"] = int(updated.get("upkeep_cost", 6)) + 1
    elif mode == "collapse":
        updated["initial_energy"] = int(updated.get("initial_energy", 100)) + 10
        updated["reproduction_threshold"] = int(updated.get("reproduction_threshold", 130)) - 6
    elif mode in {"dominance", "low_diversity"}:
        updated["diversity_bonus"] = float(updated.get("diversity_bonus", 0.2)) + 0.05
        updated["mutation_rate"] = float(updated.get("mutation_rate", 0.15)) + 0.02
    else:
        updated["mutation_rate"] = max(0.05, float(updated.get("mutation_rate", 0.15)) - 0.01)
    return updated


def _run_simulator(simulator_cmd: str, config_path: Path, output_json: Path) -> Dict[str, Any]:
    cmd = [*simulator_cmd.split(), "--config", str(config_path), "--output-json", str(output_json)]
    subprocess.run(cmd, check=True)
    return json.loads(output_json.read_text(encoding="utf-8"))


def _run_summary(run_number: int, output: Dict[str, Any], config: Dict[str, Any]) -> str:
    explanation = f"Failure mode was {output['failure_mode']}."
    if output["failure_mode"] == "success":
        explanation = "Run stayed near target band without major instability."
    return "\n".join([
        f"Run {run_number} Summary:",
        f"- config used: {json.dumps(config)}",
        f"- start population: {output['start_population']}",
        f"- max population: {output['max_population']}",
        f"- final population: {output['final_population']}",
        f"- generations in target band: {output['generations_in_target_band']}",
        f"- viable lineages: {output['viable_lineages']}",
        f"- dominance %: {round(float(output['dominance_share']) * 100.0, 2)}",
        f"- late births: {output['late_births']}",
        f"- result: {output['failure_mode']}",
        f"- explanation: {explanation}",
    ])


def _session_report(history: List[Dict[str, Any]]) -> str:
    failures: Dict[str, int] = {}
    for run in history:
        mode = run["output"]["failure_mode"]
        failures[mode] = failures.get(mode, 0) + 1
    best = max(history, key=lambda h: h["output"].get("generations_in_target_band", 0))
    closest = min(history, key=lambda h: abs(h["output"].get("final_population", 0) - 100))
    return "\n".join([
        "SESSION REPORT:",
        f"- total runs: {len(history)}",
        f"- best run: {best['run_number']} ({best['output']['failure_mode']})",
        f"- closest to target: run {closest['run_number']} (final pop {closest['output']['final_population']})",
        f"- failure breakdown: {json.dumps(failures)}",
        "- what changes helped: higher generations_in_target_band and lower dominance share.",
        "- what changes hurt: repeated overshoot/collapse transitions.",
        f"- final recommendation: use run {best['run_number']} config as next baseline.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="External Genesis2 tuning controller")
    parser.add_argument("--runner-config", default="config/tuning_runner_default.json")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--simulator-cmd", default=f"{sys.executable} -m src.main")
    args = parser.parse_args()

    payload = _load_json(Path(args.runner_config))
    objective = payload.get("objective", DEFAULT_OBJECTIVE)
    levers = payload.get("levers", DEFAULT_LEVERS)
    current_config = payload.get("initial_config", {
        "agents": 25,
        "generations": 100,
        "initial_energy": 100,
        "reproduction_threshold": 130,
        "mutation_rate": 0.15,
        "upkeep_cost": 6,
        "tasks_per_generation": 30,
        "diversity_bonus": 0.2,
        "diversity_min_lineages": 6,
    })

    api_key = args.api_key or payload.get("api_key") or ""
    if not api_key and sys.stdin.isatty():
        api_key = getpass("Anthropic API key (optional, press enter for fallback mode): ").strip()

    settings = RunnerSettings(
        max_runs=int(args.max_runs),
        max_seconds=int(args.max_seconds),
        model=str(args.model or payload.get("model", DEFAULT_MODEL)),
        simulator_cmd=str(args.simulator_cmd or payload.get("simulator_cmd", f"{sys.executable} -m src.main")),
        output_dir=Path(payload.get("output_dir", "runs/tuning_runner")),
        api_key=api_key,
    )

    session_id = datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ")
    session_dir = settings.output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=False)

    system_prompt = "You are a swarm tuning advisor. Return only JSON with recommended_config, rationale, predicted_effects, confidence."

    history: List[Dict[str, Any]] = []
    start = time.time()
    collapse_streak = 0

    advisor_reply = None
    if settings.api_key:
        _log_advisor_event(session_dir, {"phase": "initial_request", "status": "attempt"})
        advisor_reply = _call_anthropic(
            api_key=settings.api_key,
            model=settings.model,
            system_prompt=system_prompt,
            user_payload={"objective": objective, "levers": levers, "history": [], "latest_run_summary": "none"},
        )
        _log_advisor_event(session_dir, {"phase": "initial_request", "status": "success" if advisor_reply else "failed"})
        if advisor_reply and isinstance(advisor_reply.get("recommended_config"), dict):
            current_config = _validate_config(advisor_reply["recommended_config"], current_config, levers)

    for run_number in range(1, settings.max_runs + 1):
        if time.time() - start > settings.max_seconds:
            break

        bounded_config = _validate_config(current_config, current_config, levers)
        run_config_path = session_dir / f"run_{run_number:03d}_config.json"
        run_output_path = session_dir / f"run_{run_number:03d}_output.json"
        run_config_path.write_text(json.dumps(bounded_config, indent=2), encoding="utf-8")

        output = _run_simulator(settings.simulator_cmd, run_config_path, run_output_path)
        summary = _run_summary(run_number, output, bounded_config)
        (session_dir / f"run_{run_number:03d}_summary.txt").write_text(summary, encoding="utf-8")

        history.append({"run_number": run_number, "config": bounded_config, "output": output, "summary": summary})

        if output["failure_mode"] == "success":
            break
        collapse_streak = collapse_streak + 1 if output["failure_mode"] == "collapse" else 0
        if collapse_streak >= 3:
            break

        compact_history = [
            {
                "run": h["run_number"],
                "failure_mode": h["output"]["failure_mode"],
                "final_population": h["output"]["final_population"],
                "dominance_share": h["output"]["dominance_share"],
                "late_births": h["output"]["late_births"],
            }
            for h in history[-8:]
        ]
        advisor_reply = None
        if settings.api_key:
            _log_advisor_event(session_dir, {"phase": "iterative_request", "run_number": run_number, "status": "attempt"})
            advisor_reply = _call_anthropic(
                api_key=settings.api_key,
                model=settings.model,
                system_prompt=system_prompt,
                user_payload={
                    "objective": objective,
                    "levers": [{"name": k, "current": bounded_config.get(k), **v} for k, v in levers.items()],
                    "run_history": compact_history,
                    "latest_run_summary": summary,
                },
            )

        _log_advisor_event(session_dir, {"phase": "iterative_request", "run_number": run_number, "status": "success" if advisor_reply else "failed"})
        if advisor_reply and isinstance(advisor_reply.get("recommended_config"), dict):
            next_candidate = advisor_reply["recommended_config"]
        else:
            next_candidate = _fallback_adjustment(bounded_config, output)

        current_config = _validate_config(next_candidate, bounded_config, levers)

    report = _session_report(history) if history else "SESSION REPORT:\n- total runs: 0"
    (session_dir / "session_report.txt").write_text(report, encoding="utf-8")
    (session_dir / "session_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(report)
    print(f"Session directory: {session_dir}")


if __name__ == "__main__":
    main()
