import json
from pathlib import Path

from src.tuner import anthropic_test_harness as harness


class _FakeEngine:
    calls = 0

    def __init__(self, _cfg):
        pass

    def run(self):
        _FakeEngine.calls += 1
        pop = 100 if _FakeEngine.calls == 1 else 30
        timeline = [
            {
                "generation": i + 1,
                "population": pop,
                "lineage_count": 12,
                "diversity_score": 0.2,
                "dominance_metrics": {
                    "top_lineage_energy_share": 0.4,
                    "top_3_lineage_energy_share": 0.65,
                },
            }
            for i in range(100)
        ]
        return {
            "run_id": f"run-{_FakeEngine.calls}",
            "run_dir": f"runs/fake/run-{_FakeEngine.calls}",
            "final_population": pop,
            "timeline": timeline,
            "tasks_solved": 10,
        }


def test_stop_reason_target_met():
    goal = {"target_label": "healthy", "stop_on_labels": [], "stop_on_extinction": True, "stop_on_non_target_success": True}
    reason = harness._stop_reason(goal, {"label": "healthy"}, {"final_population": 100})
    assert reason == "target_met:healthy"


def test_run_harness_writes_report_and_followup(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    responses = [
        {"error": None, "raw": "{}", "parsed": {"recommended_config": {"tasks_per_generation": 66}}},
        {"error": None, "raw": "{}", "parsed": {"recommended_config": {"mutation_rate": 0.24}}},
    ]

    def fake_call(**_kwargs):
        return responses.pop(0)

    monkeypatch.setattr(harness, "_call_anthropic_advisor", fake_call)
    monkeypatch.setattr(harness, "SimulationEngine", _FakeEngine)

    spec = {
        "base_config": "config/default.json",
        "initial_parameters": {"agents": 25, "generations": 100},
        "goal": {
            "target_label": "healthy",
            "stop_on_labels": ["failed_collapse"],
            "stop_on_extinction": True,
            "stop_on_non_target_success": False,
        },
        "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "model": "claude-3-5-sonnet-20241022", "temperature": 0.1, "timeout_seconds": 5},
        "max_cycles": 1,
        "max_runs_per_cycle": 2,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = harness.run_harness(spec_path)

    assert result["stop_reason"] in {"target_met:healthy", "failure_label:failed_collapse"}
    report_path = Path(result["report_path"])
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Anthropic Tuning Harness Report" in report_text
    assert "## Run results table" in report_text
    assert "| Run | Cycle | Score | Label | Final Pop | Lineages | Diversity | Diagnosis |" in report_text

    run_results_csv = report_path.parent / "run_results.csv"
    assert run_results_csv.exists()
    csv_text = run_results_csv.read_text(encoding="utf-8")
    assert "run_index,cycle,score,label,final_population,lineage_count,diversity_score,diagnosis" in csv_text

    session_output = report_path.parent / "session_output.json"
    assert session_output.exists()
