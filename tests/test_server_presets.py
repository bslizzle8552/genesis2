from src.backends.server import _advisory_failure_reason, _build_markdown_report, _build_tuning_summary, _phase2_adaptive_update, _stable_swarm_baseline, config_from_request, discover_presets


def test_discover_presets_lists_experiment_configs():
    preset_ids = {p["id"] for p in discover_presets()}
    assert "default.json" in preset_ids
    assert "experiment_fast.json" in preset_ids
    assert "experiment_default.json" in preset_ids
    assert "experiment_stress.json" in preset_ids
    assert "ecosystem_optimal_v1.json" in preset_ids
    assert "stabilization_balanced.json" in preset_ids
    assert "stabilization_strong.json" in preset_ids


def test_config_from_request_uses_preset_values():
    cfg = config_from_request({"preset": "experiment_fast.json"})
    assert cfg.agents == 8
    assert cfg.generations == 20


def test_config_from_request_overrides_values():
    cfg = config_from_request({"preset": "experiment_fast.json", "agents": 11, "generations": 13})
    assert cfg.agents == 11
    assert cfg.generations == 13




def test_config_from_request_can_set_api_access():
    cfg = config_from_request({"preset": "experiment_fast.json", "api_access": True})
    assert cfg.api_access is True

def test_config_from_request_supports_extended_fields():
    cfg = config_from_request({
        "preset": "experiment_fast.json",
        "initial_energy": 120,
        "reproduction_threshold": 150,
        "mutation_rate": 0.22,
        "upkeep_cost": 4,
        "tier_mix": {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4},
    })
    assert cfg.initial_energy == 120
    assert cfg.reproduction_threshold == 150
    assert cfg.mutation_rate == 0.22
    assert cfg.upkeep_cost == 4
    assert cfg.tier_mix["4"] == 0.4


def test_config_from_request_supports_diversity_controls():
    cfg = config_from_request({
        "preset": "experiment_fast.json",
        "diversity_bonus": 2.0,
        "diversity_min_lineages": 5,
        "immigrant_injection_count": 3,
    })
    assert cfg.diversity_bonus == 2.0
    assert cfg.diversity_min_lineages == 5
    assert cfg.immigrant_injection_count == 3


def test_config_from_request_supports_anti_dominance_controls():
    cfg = config_from_request({
        "preset": "ecosystem_optimal_v1.json",
        "anti_dominance_enabled": True,
        "diminishing_reward_enabled": True,
        "diminishing_reward_k": 300.0,
        "lineage_size_penalty_enabled": True,
        "lineage_size_penalty_threshold": 40,
        "lineage_size_penalty_multiplier": 0.75,
        "reproduction_cooldown_enabled": True,
        "reproduction_cooldown_generations": 3,
        "reproduction_cost": 36.0,
        "child_energy_fraction": 0.4,
    })
    assert cfg.anti_dominance_enabled is True
    assert cfg.diminishing_reward_enabled is True
    assert cfg.diminishing_reward_k == 300.0
    assert cfg.lineage_size_penalty_enabled is True
    assert cfg.lineage_size_penalty_threshold == 40
    assert cfg.lineage_size_penalty_multiplier == 0.75
    assert cfg.reproduction_cooldown_enabled is True
    assert cfg.reproduction_cooldown_generations == 3
    assert cfg.reproduction_cost == 36.0
    assert cfg.child_energy_fraction == 0.4


def test_discover_presets_exposes_anti_dominance_ui_fields():
    presets = discover_presets()
    fast = next(p for p in presets if p["id"] == "experiment_fast.json")
    for key in [
        "anti_dominance_enabled",
        "diminishing_reward_enabled",
        "diminishing_reward_k",
        "lineage_size_penalty_enabled",
        "lineage_size_penalty_threshold",
        "lineage_size_penalty_multiplier",
        "reproduction_cooldown_enabled",
        "reproduction_cooldown_generations",
        "child_energy_fraction",
    ]:
        assert key in fast


def test_stable_swarm_baseline_uses_internal_rig_baseline():
    baseline = _stable_swarm_baseline()
    assert baseline["agents"] == 25
    assert baseline["generations"] == 100


def test_phase2_adaptive_update_adjusts_for_low_population():
    params = {
        "initial_energy": 100,
        "upkeep_cost": 6,
        "reproduction_threshold": 130.0,
        "mutation_rate": 0.15,
        "tasks_per_generation": 20,
        "diversity_bonus": 1.0,
        "immigrant_injection_count": 2,
    }
    score = {
        "gates": {
            "late_population_ok": False,
            "target_band_stability_ok": False,
            "population_volatility_ok": False,
            "late_lineage_count_ok": False,
            "top_lineage_share_ok": False,
            "solve_rate_ok": False,
        },
        "component_metrics": {"population_stability": {"late_avg_population": 50}},
    }
    updated = _phase2_adaptive_update(params, score)
    assert updated["initial_energy"] > params["initial_energy"]
    assert updated["upkeep_cost"] < params["upkeep_cost"]
    assert updated["anti_dominance_enabled"] is True
    assert updated["reproduction_cooldown_enabled"] is True


def test_tuning_summary_contains_goal_and_advisory_fields():
    summary = _build_tuning_summary(
        session_id="s1",
        started_at="2025-01-01T00:00:00Z",
        elapsed_seconds=12,
        run_records=[],
        best_run=None,
        candidate_configs=[],
        failure_modes={},
        final_outcome="No Equilibrium Found",
        repeatability=None,
        early_stop_reason=None,
        session_diagnostics={},
        advisory_settings={"advisory_api_enabled": True},
        advisory_usage={"calls": 1, "accepted": 1, "operator_summaries": ["ok"]},
        advisor_enabled_ui=True,
        advisor_active_runtime=True,
        advisor_failure_reason=None,
        advisory_calls_made=1,
        advisory_events=[{"run": 1, "call_made": True}],
    )
    assert summary["goal_profile"]["starting_agents_target"] == 25
    assert summary["advisory_usage"]["calls"] == 1
    assert summary["advisor_enabled_ui"] is True
    assert summary["advisor_active_runtime"] is True
    assert summary["advisory_calls_made"] == 1


def test_markdown_report_includes_advisory_section():
    report = _build_markdown_report({
        "tuning_session_id": "s1",
        "final_outcome": "Promising Candidate Found",
        "total_runs_executed": 2,
        "best_score": 72.5,
        "best_config": {"agents": 25},
        "top_candidate_configs": [],
        "dominant_failure_modes": [],
        "advisory_usage": {"calls": 1, "accepted": 0, "operator_summaries": ["fallback"]},
        "advisor_enabled_ui": True,
        "advisor_active_runtime": True,
        "advisor_failure_reason": None,
        "advisory_calls_made": 1,
        "advisory_events": [{"run": 1, "call_made": True, "request_summary": "x", "response_summary": "y", "recommendations_accepted": True, "recommendations_rejected": False, "failure_reason": None}],
        "suggested_next_tuning_direction": "Tune diversity",
    })
    assert "## Advisory API Usage" in report
    assert "Calls: 1" in report
    assert "Active at runtime: True" in report
    assert "## Advisory Event Log" in report


def test_tuning_summary_includes_human_readable_sections():
    run = {
        "run_in_batch": 1,
        "score": 55.0,
        "label": "failed_low_diversity",
        "params": {"agents": 25, "reproduction_threshold": 130, "mutation_rate": 0.2},
        "metrics": {
            "score": 55.0,
            "label": "failed_low_diversity",
            "final_population": 54,
            "lineage_count": 3,
            "diagnosis": ["low_diversity", "late_dominance"],
            "hard_gates": {"starts_near_25_ok": True, "reaches_target_by_generation_80_ok": False, "lineage_count_ok": False, "top_lineage_share_ok": False, "top_3_lineage_share_ok": False, "late_stability_ok": False},
        },
        "timeline": [{"generation": 1, "population": 25, "births": 1, "lineages": {"L1": 25}}],
        "result": {"agents": [], "timeline": [], "problems": [], "board_messages": [], "run_dir": ""},
        "adjustment_reason": "increase diversity bonus",
    }
    summary = _build_tuning_summary(
        session_id="s1",
        started_at="2025-01-01T00:00:00Z",
        elapsed_seconds=12,
        run_records=[run],
        best_run=run,
        candidate_configs=[],
        failure_modes={"failed_low_diversity": 1},
        final_outcome="No Equilibrium Found",
        repeatability=None,
        early_stop_reason=None,
        session_diagnostics={},
        advisory_settings={"advisory_api_enabled": True},
        advisory_usage={"calls": 1, "accepted": 0, "operator_summaries": ["fallback"]},
    )
    human = summary["human_readable_summary"]
    assert "executive_summary" in human
    assert "run_level_reports" in human
    assert human["failure_breakdown"]["dominance"] >= 1 or human["failure_breakdown"]["low_diversity"] >= 1


def test_advisory_failure_reason_mapping():
    assert _advisory_failure_reason("no_api_key") == "no API key"
    assert _advisory_failure_reason("invalid_endpoint") == "invalid endpoint"
    assert _advisory_failure_reason("request_failed: timeout") == "request failed"
