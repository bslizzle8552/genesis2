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
