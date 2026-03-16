from pathlib import Path
from src.engine.simulation import SimulationConfig, SimulationEngine


def test_simulation_runs_and_logs(tmp_path):
    cfg = SimulationConfig(seed=1, agents=5, generations=5, tasks_per_generation=4, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()

    assert result["final_population"] >= 0
    assert len(result["timeline"]) >= 1
    assert (tmp_path / "run_seed1_g5" / "summary.json").exists()


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
