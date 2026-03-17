from src.tuner.adaptive_rig import adjust_parameters, score_and_label_run


def _run_result(populations, lineages=12, diversity=0.2):
    timeline = [{"generation": i + 1, "population": p, "lineage_count": lineages, "diversity_score": diversity} for i, p in enumerate(populations)]
    return {"timeline": timeline, "final_population": populations[-1] if populations else 0}


def test_scoring_labels_healthy_case():
    result = _run_result([100] * 100, lineages=14, diversity=0.2)
    metrics = score_and_label_run(result)
    assert metrics["label"] == "healthy"
    assert metrics["score"] >= 80


def test_scoring_labels_collapse_case():
    result = _run_result([120] * 50 + [30] * 50, lineages=4, diversity=0.05)
    metrics = score_and_label_run(result)
    assert metrics["label"] == "collapsed"
    assert metrics["score"] < 50


def test_adjust_parameters_for_low_diversity():
    updated, reason = adjust_parameters(
        {"diversity_bonus": 0.175, "diversity_min_lineages": 14, "mutation_rate": 0.28},
        {"label": "low_diversity", "final_population": 100},
    )
    assert updated["diversity_bonus"] > 0.175
    assert updated["anti_dominance_enabled"] is True
    assert "diversity" in reason
