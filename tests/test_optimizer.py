"""Tests for the Day 1 optimization engine."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from optimizer import (
    PREFERENCE_PROFILES,
    add_derived_metrics,
    explain_recommendation,
    normalize_weights,
    pareto_frontier,
    profile_sensitivity,
    rank_routes,
)


def route(
    name: str,
    cost: float,
    transit: float,
    congestion: float = 0.0,
    geopolitical: float = 0.0,
    delay: float = 0.0,
    impact: float = 0.0,
    route_id: str | None = None,
) -> dict:
    return {
        "shipment_id": "SHP-TEST",
        "origin": "Mumbai, India",
        "destination": "Rotterdam, Netherlands",
        "route_id": route_id or f"SHP-TEST-{name.replace(' ', '')}",
        "route_name": name,
        "intermediate_hubs": "Colombo",
        "legs": "Mumbai → Colombo → Rotterdam",
        "cost_usd": cost,
        "transit_days": transit,
        "delay_probability": delay,
        "delay_impact_days": impact,
        "congestion_risk": congestion,
        "geopolitical_risk": geopolitical,
    }


def tradeoff_frame() -> pd.DataFrame:
    """Cheap/high-risk, fast/medium-risk, and safe/slow options.

    Expected times are 12, 8, and 10 days. Risk scores are 55, 28, and 5.5.
    """
    return pd.DataFrame([
        route("Cheap", 100, 12, congestion=100, geopolitical=100),
        route("Fast", 150, 8, congestion=60, geopolitical=40),
        route("Safe", 130, 10, congestion=10, geopolitical=10),
    ])


# --- 1-2. Derived metrics -------------------------------------------------

def test_expected_transit_time_formula():
    result = add_derived_metrics(pd.DataFrame([route("R", 100, 25, delay=0.30, impact=5)]))
    assert result.loc[0, "expected_transit_days"] == pytest.approx(26.5)


def test_risk_score_formula_and_bounds():
    result = add_derived_metrics(pd.DataFrame([route(
        "R", 100, 10, congestion=40, geopolitical=20, delay=0.30,
    )]))
    assert result.loc[0, "risk_score"] == pytest.approx(30.5)

    extreme = add_derived_metrics(pd.DataFrame([route(
        "R", 100, 10, congestion=100, geopolitical=100, delay=1.0,
    )]))
    assert extreme.loc[0, "risk_score"] == pytest.approx(100.0)
    assert 0 <= extreme.loc[0, "risk_score"] <= 100


# --- 3-4. Weights ---------------------------------------------------------

def test_weights_are_normalized_to_one():
    normalized = normalize_weights({"cost": 2, "time": 1, "risk": 1})
    assert sum(normalized.values()) == pytest.approx(1.0)
    assert normalized == pytest.approx({"cost": 0.5, "time": 0.25, "risk": 0.25})


def test_default_weights_are_normalized():
    assert sum(normalize_weights().values()) == pytest.approx(1.0)
    assert normalize_weights({"cost": 0.40, "time": 0.35}) == pytest.approx(
        {"cost": 0.40 / 0.75, "time": 0.35 / 0.75, "risk": 0.0}
    )

@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"cost": 0, "time": 0, "risk": 0}, "greater than zero"),
        ({"cost": -0.1, "time": 0.5, "risk": 0.5}, "non-negative"),
        ({"cost": 0.4, "time": 0.3, "risk": 0.3, "speed": 0.1}, "Unknown weight"),
        ({"cost": "high", "time": 0.5, "risk": 0.5}, "must be a number"),
        ({"cost": float("nan"), "time": 0.5, "risk": 0.5}, "finite"),
        ({"cost": True, "time": 0.5, "risk": 0.5}, "boolean"),
        ({}, "greater than zero"),
        (["cost", "time", "risk"], "dictionary"),
    ],
)
def test_invalid_weights_are_rejected(weights, message):
    with pytest.raises(ValueError, match=message):
        normalize_weights(weights)


# --- 5-6. Min-max normalization and ranking ------------------------------

def test_minmax_components_and_weighted_ranking():
    ranked = rank_routes(tradeoff_frame())

    components = ranked.set_index("route_name")
    assert components.loc["Cheap", "cost_component"] == pytest.approx(0.0)
    assert components.loc["Cheap", "time_component"] == pytest.approx(1.0)
    assert components.loc["Cheap", "risk_component"] == pytest.approx(1.0)
    assert components.loc["Fast", "cost_component"] == pytest.approx(1.0)
    assert components.loc["Fast", "time_component"] == pytest.approx(0.0)
    assert components.loc["Safe", "cost_component"] == pytest.approx(0.6)
    assert components.loc["Safe", "time_component"] == pytest.approx(0.5)
    assert components.loc["Safe", "risk_component"] == pytest.approx(0.0)

    assert components.loc["Safe", "score"] == pytest.approx(0.415)
    assert components.loc["Fast", "score"] == pytest.approx(0.513636, abs=1e-5)
    assert components.loc["Cheap", "score"] == pytest.approx(0.6)
    assert ranked["route_name"].tolist() == ["Safe", "Fast", "Cheap"]
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_identical_metric_values_do_not_divide_by_zero():
    frame = pd.DataFrame([
        route("A", 100, 10, congestion=20),
        route("B", 100, 10, congestion=20),
    ])
    ranked = rank_routes(frame)

    assert ranked["cost_component"].tolist() == [0.0, 0.0]
    assert ranked["time_component"].tolist() == [0.0, 0.0]
    assert ranked["risk_component"].tolist() == [0.0, 0.0]
    assert ranked["score"].tolist() == [0.0, 0.0]
    assert ranked["pareto_optimal"].all()
    assert all(math.isfinite(score) for score in ranked["score"])


@pytest.mark.parametrize(
    ("weights", "expected_order"),
    [
        ({"cost": 0.70, "time": 0.20, "risk": 0.10}, ["Cheap", "Safe", "Fast"]),
        ({"cost": 0.40, "time": 0.35, "risk": 0.25}, ["Safe", "Fast", "Cheap"]),
        ({"cost": 0.15, "time": 0.20, "risk": 0.65}, ["Safe", "Fast", "Cheap"]),
        ({"cost": 0.15, "time": 0.70, "risk": 0.15}, ["Fast", "Safe", "Cheap"]),
    ],
)
def test_preference_profiles_rank_expected_routes(weights, expected_order):
    ranked = rank_routes(tradeoff_frame(), weights)
    assert ranked["route_name"].tolist() == expected_order


# --- 7-8. Hard constraints -------------------------------------------------

def test_budget_constraint_marks_routes_infeasible():
    ranked = rank_routes(tradeoff_frame(), max_budget=120)
    feasible = ranked[ranked["feasible"]]

    assert feasible["route_name"].tolist() == ["Cheap"]
    assert ranked.set_index("route_name").loc["Fast", "feasible"] == False  # noqa: E712
    assert "budget" in ranked.set_index("route_name").loc["Fast", "recommendation_status"]


def test_deadline_constraint_marks_routes_infeasible():
    ranked = rank_routes(tradeoff_frame(), max_days=9)
    feasible = ranked[ranked["feasible"]]

    assert feasible["route_name"].tolist() == ["Fast"]
    assert "deadline" in ranked.set_index("route_name").loc["Safe", "recommendation_status"]


def test_constraint_boundaries_are_inclusive():
    ranked = rank_routes(tradeoff_frame(), max_budget=100, max_days=12)
    assert ranked.set_index("route_name").loc["Cheap", "feasible"] == True  # noqa: E712


def test_no_feasible_route_returns_clear_result():
    ranked = rank_routes(tradeoff_frame(), max_budget=50, max_days=5)

    assert not ranked["feasible"].any()
    assert ranked["rank"].isna().all()
    assert ranked["score"].isna().all()
    assert not ranked["pareto_optimal"].any()
    assert all("Infeasible" in status for status in ranked["recommendation_status"])

    explanation = explain_recommendation(ranked)
    assert "No route satisfies all hard constraints" in explanation
    assert "budget" in explanation
    assert "transit time" in explanation
    assert "not recommended" in explanation


def test_infeasible_route_is_never_recommended():
    # Risk-averse weights prefer Safe when all routes are eligible...
    risk_averse = {"cost": 0.15, "time": 0.20, "risk": 0.65}
    assert rank_routes(tradeoff_frame(), risk_averse).iloc[0]["route_name"] == "Safe"

    # ...but the budget removes Safe, so the recommendation must change.
    ranked = rank_routes(tradeoff_frame(), risk_averse, max_budget=120)
    by_name = ranked.set_index("route_name")
    assert ranked.iloc[0]["route_name"] == "Cheap"
    assert by_name.loc["Safe", "feasible"] == False  # noqa: E712
    assert by_name.loc["Safe", "score"] != by_name.loc["Safe", "score"]  # NaN
    assert ranked.iloc[0]["route_name"] != "Safe"


# --- 9-10. Pareto frontier -------------------------------------------------

def test_pareto_frontier_and_dominated_routes():
    frame = pd.concat([
        tradeoff_frame(),
        pd.DataFrame([route(
            "Dominated", 160, 13, congestion=100, geopolitical=100, delay=1.0,
        )]),
    ], ignore_index=True)
    ranked = rank_routes(frame)
    by_name = ranked.set_index("route_name")

    assert by_name.loc["Safe", "pareto_optimal"] == True  # noqa: E712
    assert by_name.loc["Fast", "pareto_optimal"] == True  # noqa: E712
    assert by_name.loc["Cheap", "pareto_optimal"] == True  # noqa: E712
    assert by_name.loc["Dominated", "pareto_optimal"] == False  # noqa: E712


def test_infeasible_routes_cannot_dominate_feasible_routes():
    ranked = rank_routes(tradeoff_frame(), max_budget=110)
    by_name = ranked.set_index("route_name")

    assert by_name.loc["Cheap", "pareto_optimal"] == True  # noqa: E712
    assert not ranked.loc[~ranked["feasible"], "pareto_optimal"].any()


def test_equal_routes_do_not_dominate_each_other():
    frame = pd.DataFrame([
        route("A", 100, 10, congestion=20),
        route("B", 100, 10, congestion=20),
        route("Worse", 120, 11, congestion=30),
    ])
    ranked = rank_routes(frame).set_index("route_name")

    assert ranked.loc["A", "pareto_optimal"] == True  # noqa: E712
    assert ranked.loc["B", "pareto_optimal"] == True  # noqa: E712
    assert ranked.loc["Worse", "pareto_optimal"] == False  # noqa: E712


def test_pareto_is_distinct_from_weighted_ranking():
    # With cost-only weights A and B tie on score, but B dominates A.
    frame = pd.DataFrame([
        route("A", 100, 20, congestion=80),
        route("B", 100, 10, congestion=10),
    ])
    ranked = rank_routes(frame, {"cost": 1, "time": 0, "risk": 0})
    by_name = ranked.set_index("route_name")

    assert by_name.loc["A", "rank"] == 1
    assert by_name.loc["A", "pareto_optimal"] == False  # noqa: E712
    assert by_name.loc["B", "pareto_optimal"] == True  # noqa: E712
    assert "not Pareto-optimal" in explain_recommendation(ranked)


# --- 11-12. Explanation and sensitivity -----------------------------------

def test_explanation_is_dynamic_and_complete():
    ranked = rank_routes(tradeoff_frame())
    explanation = explain_recommendation(ranked)

    assert "Recommended Safe" in explanation
    assert "$130.00" in explanation
    assert "10.0 days" in explanation
    assert "5.5/100" in explanation
    assert "Fast" in explanation
    assert "faster by 2.0 days" in explanation
    assert "cheaper by $20.00" in explanation
    assert "Pareto-optimal" in explanation
    assert "No candidate routes were rejected" in explanation


@pytest.mark.parametrize(
    ("weights", "expected_route", "fragment"),
    [
        (
            {"cost": 0.70, "time": 0.20, "risk": 0.10},
            "Cheap",
            "because it is cheaper by $30.00",
        ),
        (
            {"cost": 0.15, "time": 0.70, "risk": 0.15},
            "Fast",
            "lower risk by 22.5 points",
        ),
    ],
)
def test_explanation_remains_correct_when_recommendation_is_cheaper_or_riskier(
    weights, expected_route, fragment,
):
    ranked = rank_routes(tradeoff_frame(), weights)
    explanation = explain_recommendation(ranked)

    assert ranked.iloc[0]["route_name"] == expected_route
    assert explanation.startswith(f"Recommended {expected_route}")
    assert fragment in explanation


def test_explanation_reports_rejected_alternatives():
    ranked = rank_routes(tradeoff_frame(), max_budget=140)
    explanation = explain_recommendation(ranked)

    assert "1 candidate route(s) were rejected" in explanation
    assert "not eligible for recommendation" in explanation


def test_explanation_handles_empty_input():
    assert "No candidate routes" in explain_recommendation(pd.DataFrame())


def test_sensitivity_analysis_returns_all_profiles():
    sensitivity = profile_sensitivity(tradeoff_frame())

    assert sensitivity["profile"].tolist() == [
        "Cost focused", "Balanced", "Risk averse", "Time critical",
    ]
    assert sensitivity["recommended_route"].tolist() == [
        "Cheap", "Safe", "Safe", "Fast",
    ]
    for weights in PREFERENCE_PROFILES.values():
        assert sum(weights.values()) == pytest.approx(1.0)


def test_sensitivity_respects_hard_constraints():
    sensitivity = profile_sensitivity(tradeoff_frame(), max_budget=110)
    assert sensitivity["recommended_route"].tolist() == ["Cheap"] * 4

    none_feasible = profile_sensitivity(tradeoff_frame(), max_budget=50, max_days=5)
    assert none_feasible["recommended_route"].tolist() == ["No feasible route"] * 4
    assert none_feasible["score"].isna().all()


# --- 13. Input guards ------------------------------------------------------

def test_rank_routes_rejects_missing_input_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        rank_routes(pd.DataFrame([{"cost_usd": 100}]))


def test_empty_routes_are_returned_unchanged():
    empty = pd.DataFrame()
    assert rank_routes(empty).empty


def test_pareto_frontier_accepts_manual_frame():
    frame = pd.DataFrame({
        "cost_usd": [100, 120],
        "expected_transit_days": [10, 11],
        "risk_score": [20, 30],
    })
    assert pareto_frontier(frame).tolist() == [True, False]
