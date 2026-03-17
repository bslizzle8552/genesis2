from pathlib import Path

from src.analytics.bvl import events_from_problem_metrics
from src.engine.simulation import SimulationConfig, SimulationEngine


def test_problem_board_events_include_agent_lineage_role(tmp_path):
    cfg = SimulationConfig(seed=13, agents=6, generations=3, tasks_per_generation=3, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()
    run_dir = Path(result["summary_path"]).parent

    events = events_from_problem_metrics(run_dir)

    assert events
    sample = events[0]
    assert "timestamp" in sample
    assert "problem_id" in sample
    assert "generation" in sample
    assert "agent_id" in sample
    assert "lineage" in sample
    assert "role" in sample
    assert sample["action_type"] in {"propose", "critique", "refine", "solve", "artifact_usage"}
