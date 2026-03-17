from src.analytics.swarm_scoring import HealthySwarmScorer
from src.engine.simulation import SimulationConfig, SimulationEngine


def test_swarm_scoring_generates_components(tmp_path):
    cfg = SimulationConfig(
        preset_name="test",
        agents=8,
        generations=5,
        tasks_per_generation=3,
        log_dir=str(tmp_path / "runs"),
        run_label="score_test",
        overwrite=True,
    )
    result = SimulationEngine(cfg).run()
    score = HealthySwarmScorer().score_run(result)

    assert "composite_score" in score
    assert set(score["component_scores"]) == {
        "population_stability",
        "diversity",
        "dominance_control",
        "intergenerational_health",
        "throughput",
        "efficiency",
    }
    assert "gates" in score
    assert {"start_population_ok", "target_reach_generation_ok", "target_band_stability_ok", "population_volatility_ok"}.issubset(set(score["gates"]))
