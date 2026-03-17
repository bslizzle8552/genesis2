from src.tuner.adaptive_rig import adjust_parameters, score_and_label_run


def _run_result(populations, lineages=12, diversity=0.2, top1=0.4, top3=0.65):
    timeline = [
        {
            "generation": i + 1,
            "population": p,
            "lineage_count": lineages,
            "diversity_score": diversity,
            "dominance_metrics": {
                "top_lineage_energy_share": top1,
                "top_3_lineage_energy_share": top3,
            },
        }
        for i, p in enumerate(populations)
    ]
    return {"timeline": timeline, "final_population": populations[-1] if populations else 0}


def test_scoring_labels_healthy_case():
    trajectory = [25 + int((75 * (i + 1)) / 80) for i in range(80)] + [100] * 20
    result = _run_result(trajectory, lineages=15, diversity=0.2)
    metrics = score_and_label_run(result)
    assert metrics["label"] == "healthy"
    assert metrics["score"] >= 65


def test_scoring_labels_collapse_case():
    result = _run_result([120] * 50 + [30] * 50, lineages=4, diversity=0.05)
    metrics = score_and_label_run(result)
    assert metrics["label"] == "failed_collapse"
    assert metrics["score"] < 50


def test_adjust_parameters_for_low_diversity():
    updated, reason = adjust_parameters(
        {"diversity_bonus": 0.175, "diversity_min_lineages": 14, "mutation_rate": 0.28},
        {"label": "failed_low_diversity", "final_population": 100},
    )
    assert updated["diversity_bonus"] > 0.175
    assert updated["anti_dominance_enabled"] is True
    assert "diversity" in reason


def test_hard_gate_dominance_failure():
    trajectory = [25 + int((75 * (i + 1)) / 80) for i in range(80)] + [100] * 20
    result = _run_result(trajectory, lineages=16, diversity=0.3, top1=0.7, top3=0.9)
    metrics = score_and_label_run(result)
    assert metrics["label"] == "failed_dominance"
    assert metrics["passed_hard_gates"] is False
    assert metrics["score"] <= 70


def test_penalizes_wrong_start_profile():
    result = _run_result([100] * 100, lineages=15, diversity=0.2)
    metrics = score_and_label_run(result)
    assert metrics["label"] == "failed_wrong_start"
