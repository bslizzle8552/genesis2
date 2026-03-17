import json
from pathlib import Path
from src.engine.simulation import SimulationConfig, SimulationEngine


def test_simulation_runs_and_logs(tmp_path):
    cfg = SimulationConfig(seed=1, agents=5, generations=5, tasks_per_generation=4, log_dir=str(tmp_path))
    result = SimulationEngine(cfg).run()

    assert result["final_population"] >= 0
    assert len(result["timeline"]) >= 1
    run_dir = Path(result["summary_path"]).parent
    assert run_dir.name.startswith("default__")
    assert (run_dir / "summary.json").exists()


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



def test_repeated_runs_use_unique_folders_and_manifest(tmp_path):
    cfg = SimulationConfig(seed=11, agents=5, generations=3, tasks_per_generation=3, log_dir=str(tmp_path), run_label="same")

    result_one = SimulationEngine(cfg).run()
    result_two = SimulationEngine(cfg).run()

    run_dir_one = Path(result_one["summary_path"]).parent
    run_dir_two = Path(result_two["summary_path"]).parent

    assert run_dir_one != run_dir_two
    assert run_dir_one.exists()
    assert run_dir_two.exists()

    expected_files = [
        "artifact_metrics.jsonl",
        "config.json",
        "generation_metrics.jsonl",
        "lineage_metrics.jsonl",
        "problem_metrics.jsonl",
        "role_metrics.jsonl",
        "run_summary.md",
        "summary.json",
    ]

    manifest_path = run_dir_two / "run_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == result_two["run_id"]
    assert manifest["label"] == "same"
    assert manifest["started_at"]
    assert manifest["finished_at"]
    assert manifest["config_snapshot"]["seed"] == 11
    for filename in expected_files:
        assert filename in manifest["files"]
        assert (run_dir_two / filename).exists()


def test_run_label_is_sanitized_in_folder_name(tmp_path):
    cfg = SimulationConfig(seed=12, agents=4, generations=2, tasks_per_generation=2, log_dir=str(tmp_path), run_label="My Run / Label")
    result = SimulationEngine(cfg).run()
    run_dir = Path(result["summary_path"]).parent

    assert run_dir.name.startswith("My-Run-Label__")


def test_generation_metrics_include_dominance_fields(tmp_path):
    cfg = SimulationConfig(seed=13, agents=8, generations=5, tasks_per_generation=4, log_dir=str(tmp_path), anti_dominance_enabled=True, diminishing_reward_enabled=True)
    result = SimulationEngine(cfg).run()
    run_dir = Path(result["summary_path"]).parent
    metrics_path = run_dir / "generation_metrics.jsonl"
    lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = json.loads(lines[-1])

    for field in [
        "top_lineage_population_share",
        "top_lineage_energy_share",
        "top_3_lineage_energy_share",
        "energy_inequality_proxy",
        "births_from_top_lineage",
        "births_from_top_3_lineages",
        "reproduction_concentration",
        "lineage_extinction_count",
        "surviving_lineage_count",
        "reward_concentration_by_lineage",
        "births_blocked_by_cooldown",
        "reward_multiplier_stats",
    ]:
        assert field in row


def test_generation_log_tracks_anti_dominance_logging(tmp_path):
    cfg = SimulationConfig(
        seed=21,
        agents=8,
        generations=5,
        tasks_per_generation=4,
        log_dir=str(tmp_path),
        anti_dominance_enabled=True,
        diminishing_reward_enabled=True,
        reproduction_cooldown_enabled=True,
        reproduction_cooldown_generations=1,
    )
    result = SimulationEngine(cfg).run()
    latest = result["timeline"][-1]

    assert "top_3_lineage_energy_share" in latest
    assert "births_blocked_by_cooldown" in latest
    assert "reward_multiplier_stats" in latest
    assert "applied" in latest["reward_multiplier_stats"]
