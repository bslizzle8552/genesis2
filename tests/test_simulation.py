from pathlib import Path
from src.engine.simulation import SimulationConfig, SimulationEngine


def test_simulation_runs_and_logs(tmp_path):
    cfg = SimulationConfig(seed=1, agents=5, generations=5, tasks_per_generation=4, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()

    assert result["final_population"] >= 0
    assert len(result["timeline"]) >= 1
    assert (tmp_path / "run_seed1_g5_default" / "summary.json").exists()


def test_agents_have_genomes_after_run(tmp_path):
    cfg = SimulationConfig(seed=2, agents=4, generations=3, tasks_per_generation=3, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()
    for agent in result["agents"]:
        assert "genome" in agent
        assert "specialization" in agent["genome"]


def test_simulation_outputs_report_and_markdown(tmp_path):
    cfg = SimulationConfig(seed=3, agents=6, generations=4, tasks_per_generation=4, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()

    assert "report" in result
    assert "warning_flags" in result["report"]
    assert "no_api_capability_ceiling" in result["report"]
    assert "ecosystem_diagnostics" in result["report"]
    assert "role_reproduction" in result["report"]
    assert Path(result["markdown_summary_path"]).exists()
    assert len(result["board_messages"]) >= 1


def test_problem_ledger_fields_are_exposed(tmp_path):
    cfg = SimulationConfig(seed=5, agents=6, generations=5, tasks_per_generation=4, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()

    problem = result["problems"][0]
    assert "prompt_text" in problem
    assert "status" in problem
    assert "agents_involved" in problem
    assert "contribution_chain" in problem
    assert "reward_split" in problem


def test_observability_report_contains_required_sections(tmp_path):
    cfg = SimulationConfig(seed=7, agents=8, generations=8, tasks_per_generation=5, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()

    obs = result["report"].get("observability", {})
    assert "energy" in obs
    assert "reproduction" in obs
    assert "contribution" in obs
    assert "collaboration" in obs
    assert "diversity" in obs
    assert "plateau" in obs
    assert "histogram" in obs["energy"]
    assert "top_10_richest" in obs["energy"]
    assert "births_by_generation" in obs["reproduction"]
    assert "births_by_role" in obs["reproduction"]
    assert "births_by_lineage" in obs["reproduction"]
    assert "births_per_agent" in obs["reproduction"]


def test_jsonl_metric_streams_are_emitted(tmp_path):
    cfg = SimulationConfig(seed=9, agents=6, generations=4, tasks_per_generation=3, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()
    run_dir = Path(result["summary_path"]).parent
    for name in ["generation_metrics.jsonl", "lineage_metrics.jsonl", "role_metrics.jsonl", "problem_metrics.jsonl"]:
        assert (run_dir / name).exists()

