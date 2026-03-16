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
    assert Path(result["markdown_summary_path"]).exists()
    assert len(result["board_messages"]) >= 1



def test_simulation_reports_contribution_ecology(tmp_path):
    cfg = SimulationConfig(seed=7, agents=8, generations=8, tasks_per_generation=5, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()

    assert "contribution_chains" in result
    assert len(result["contribution_chains"]) > 0
    assert "role_reward_totals" in result["report"]
    assert "solver_dominance_diagnostic" in result["report"]
    assert "reproduced_roles" in result["report"]
    assert "support_accounting" in result["timeline"][0]
