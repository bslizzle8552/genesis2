import json
from pathlib import Path

from src.engine.experiments import run_anti_dominance_experiments, run_experiment_batch


def test_experiment_batch_produces_comparison(tmp_path):
    spec = {
        "experiment_id": "test_batch",
        "output_root": str(tmp_path / "runs"),
        "base": {
            "agents": 6,
            "generations": 5,
            "tasks_per_generation": 3,
            "log_dir": str(tmp_path / "runs"),
        },
        "sweep": {
            "seed": [1, 2],
            "tasks_per_generation": [3],
        },
    }
    cfg_path = tmp_path / "batch.json"
    cfg_path.write_text(json.dumps(spec), encoding="utf-8")

    result = run_experiment_batch(cfg_path)
    summary_path = Path(result["comparison_summary"])

    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "test_batch"
    assert len(payload["runs"]) == 2


def test_anti_dominance_matrix_produces_six_runs(tmp_path):
    spec = {
        "experiment_id": "anti_dom_test",
        "output_root": str(tmp_path / "runs"),
        "baseline": {
            "seed": 42,
            "agents": 10,
            "generations": 4,
            "initial_energy": 100,
            "upkeep_cost": 6,
            "tasks_per_generation": 4,
            "reproduction_threshold": 120,
            "mutation_rate": 0.2,
            "diversity_bonus": 0.1,
            "diversity_min_lineages": 3,
            "immigrant_injection_count": 1,
            "tier_mix": {"1": 0.34, "2": 0.31, "3": 0.21, "4": 0.14},
            "api_access": False,
        },
        "experiments": [
            {"id": "baseline_optimal_control", "overrides": {"anti_dominance_enabled": False}},
            {"id": "diminishing_reward_only", "overrides": {"anti_dominance_enabled": True, "diminishing_reward_enabled": True}},
            {"id": "lineage_size_penalty_only", "overrides": {"anti_dominance_enabled": True, "lineage_size_penalty_enabled": True}},
            {"id": "reproduction_cooldown_only", "overrides": {"anti_dominance_enabled": True, "reproduction_cooldown_enabled": True}},
            {
                "id": "diminishing_reward_plus_cooldown",
                "overrides": {"anti_dominance_enabled": True, "diminishing_reward_enabled": True, "reproduction_cooldown_enabled": True},
            },
            {
                "id": "lineage_penalty_plus_cooldown",
                "overrides": {"anti_dominance_enabled": True, "lineage_size_penalty_enabled": True, "reproduction_cooldown_enabled": True},
            },
        ],
    }
    cfg_path = tmp_path / "anti_dom.json"
    cfg_path.write_text(json.dumps(spec), encoding="utf-8")

    result = run_anti_dominance_experiments(cfg_path)
    summary_path = Path(result["comparison_summary"])

    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "anti_dom_test"
    assert len(payload["runs"]) == 6
    assert all("dominance_improved_vs_baseline" in row for row in payload["runs"])
    assert all("throughput_degraded_vs_baseline" in row for row in payload["runs"])
