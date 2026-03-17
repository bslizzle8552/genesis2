import json
from pathlib import Path

from src.tuner import run_tuning_orchestrator


def test_tuning_orchestrator_runs_and_persists_metrics(tmp_path):
    spec = {
        "batch_id": "phase1_test_batch",
        "run_count": 2,
        "output_root": str(tmp_path / "tuner"),
        "base_config": {
            "seed": 7,
            "agents": 6,
            "generations": 4,
            "tasks_per_generation": 3,
            "log_dir": str(tmp_path / "sim_runs"),
        },
    }
    cfg = tmp_path / "tuner.json"
    cfg.write_text(json.dumps(spec), encoding="utf-8")

    result = run_tuning_orchestrator(cfg)

    assert result["run_count_completed"] == 2
    assert Path(result["paths"]["runs_jsonl"]).exists()
    assert Path(result["paths"]["summary"]).exists()
    assert len(result["runs"]) == 2

    row = result["runs"][0]
    assert "final_population" in row
    assert "lineage_count" in row
    assert "diversity_score" in row
    assert "avg_energy" in row
    assert "extinction_events" in row


def test_tuner_run_ids_are_unique(tmp_path):
    spec = {
        "batch_id": "unique_id_batch",
        "run_count": 3,
        "output_root": str(tmp_path / "tuner"),
        "base_config": {
            "seed": 3,
            "agents": 5,
            "generations": 3,
            "tasks_per_generation": 2,
        },
    }
    cfg = tmp_path / "tuner_unique.json"
    cfg.write_text(json.dumps(spec), encoding="utf-8")

    result = run_tuning_orchestrator(cfg)
    run_ids = [row["run_id"] for row in result["runs"]]

    assert len(run_ids) == len(set(run_ids))
    for run_id in run_ids:
        assert (tmp_path / "tuner" / "unique_id_batch" / f"run_{run_id}.json").exists()
