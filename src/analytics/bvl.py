from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


ACTION_MAP = {
    "claim": "propose",
    "decomposition": "propose",
    "subtask": "refine",
    "collaboration": "refine",
    "verification": "critique",
    "critique": "critique",
    "catch_incorrect": "critique",
    "solve": "solve",
    "artifact_reuse": "artifact_usage",
}


def speech_for_action(action_type: str, metadata: Dict[str, object]) -> str:
    artifact_id = metadata.get("artifact_id", "known artifact")
    templates = {
        "propose": "Let’s break this into steps...",
        "critique": "This fails because assumptions are not yet validated...",
        "refine": "Refining the current approach with additional structure...",
        "solve": "Final answer is ready and submitted.",
        "artifact_usage": f"Using artifact {artifact_id} to accelerate this path...",
    }
    return templates.get(action_type, "Progress update recorded.")


def _jsonl_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_problem_board_events(problem_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []

    for row in problem_rows:
        chain = row.get("contribution_chain", [])
        if not isinstance(chain, list):
            continue

        for idx, step in enumerate(chain):
            raw_type = str(step.get("type", ""))
            action_type = ACTION_MAP.get(raw_type, "refine")
            metadata = {
                "raw_action": raw_type,
                "detail": step.get("detail", ""),
                "tier": row.get("tier"),
                "domain": row.get("domain"),
                "artifact_id": step.get("artifact_id"),
            }
            events.append(
                {
                    "timestamp": f"g{row.get('generation', 0)}:s{idx}",
                    "generation": row.get("generation"),
                    "problem_id": row.get("problem_id"),
                    "problem_thread_id": row.get("problem_id"),
                    "domain": row.get("domain"),
                    "agent_id": step.get("agent_id"),
                    "lineage": step.get("lineage_id"),
                    "role": step.get("role", "unknown"),
                    "action_type": action_type,
                    "speech": speech_for_action(action_type, metadata),
                    "metadata": metadata,
                }
            )

    events.sort(key=lambda e: (int(e.get("generation") or 0), str(e.get("problem_id") or ""), str(e.get("timestamp") or "")))
    return events


def events_from_problem_metrics(run_dir: str | Path) -> List[Dict[str, object]]:
    stream = Path(run_dir) / "problem_metrics.jsonl"
    rows = _jsonl_rows(stream)
    return build_problem_board_events(rows)
