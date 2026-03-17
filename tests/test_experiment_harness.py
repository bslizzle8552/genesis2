import json
from pathlib import Path

from src.engine.experiments import run_experiment_batch


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
