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
