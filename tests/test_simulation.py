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


def test_report_includes_role_and_reward_diagnostics(tmp_path):
    cfg = SimulationConfig(seed=9, agents=12, generations=12, tasks_per_generation=8, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()

    report = result["report"]
    assert "role_contribution_totals" in report
    assert "role_based_fitness" in report
    assert "reward_sources" in report
    assert "solver_dominance_risk" in report

    reward_sources = report["reward_sources"]
    assert reward_sources["verification"] > 0
    assert reward_sources["decomposition"] > 0
    assert reward_sources["subtasks"] > 0


def test_support_roles_remain_viable_in_population(tmp_path):
    cfg = SimulationConfig(seed=14, agents=18, generations=16, tasks_per_generation=10, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()

    final_roles = [agent["role"] for agent in result["agents"]]
    assert final_roles
    assert any(role != "solver" for role in final_roles)
    assert "verifier" in final_roles
    assert "decomposer" in final_roles
