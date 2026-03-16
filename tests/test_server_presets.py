from src.backends.server import config_from_request, discover_presets


def test_discover_presets_lists_experiment_configs():
    preset_ids = {p["id"] for p in discover_presets()}
    assert "default.json" in preset_ids
    assert "experiment_fast.json" in preset_ids
    assert "experiment_default.json" in preset_ids
    assert "experiment_stress.json" in preset_ids


def test_config_from_request_uses_preset_values():
    cfg = config_from_request({"preset": "experiment_fast.json"})
    assert cfg.agents == 8
    assert cfg.generations == 20


def test_config_from_request_overrides_values():
    cfg = config_from_request({"preset": "experiment_fast.json", "agents": 11, "generations": 13})
    assert cfg.agents == 11
    assert cfg.generations == 13


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
