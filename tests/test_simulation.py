"""Tests for the Monte Carlo reliability simulator."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from optimizer import rank_routes
from simulation import MAX_SIMULATIONS, simulate_routes


def route(
    name: str = "Route A",
    *,
    route_id: str = "R1",
    cost: float = 100.0,
    transit: float = 10.0,
    delay: float = 0.3,
    impact: float = 4.0,
) -> dict:
    return {
        "shipment_id": "SHP-TEST",
        "route_id": route_id,
        "route_name": name,
        "cost_usd": cost,
        "transit_days": transit,
        "delay_probability": delay,
        "delay_impact_days": impact,
        "congestion_risk": 20.0,
        "geopolitical_risk": 10.0,
    }


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- Core output and reproducibility -------------------------------------

def test_returns_expected_number_of_trials():
    result = simulate_routes(
        frame(route("A", route_id="R1"), route("B", route_id="R2")),
        n_simulations=250,
        seed=7,
    )

    assert len(result.summary) == 2
    assert result.summary["simulation_count"].tolist() == [250, 250]
    assert result.samples.groupby("route_id").size().tolist() == [250, 250]
    assert result.samples.groupby("route_id")["trial"].max().tolist() == [249, 249]


def test_same_seed_is_reproducible():
    routes = frame(route("A", route_id="R1"), route("B", route_id="R2", delay=0.6))
    first = simulate_routes(routes, n_simulations=500, seed=123)
    second = simulate_routes(routes, n_simulations=500, seed=123)

    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.samples, second.samples)
    assert first.simulation_count == second.simulation_count
    assert first.seed == second.seed == 123


def test_different_seeds_can_change_outcomes():
    routes = frame(route("A", route_id="R1", delay=0.5, impact=5.0))
    first = simulate_routes(routes, n_simulations=500, seed=1)
    second = simulate_routes(routes, n_simulations=500, seed=2)

    assert not first.samples["simulated_transit_days"].equals(
        second.samples["simulated_transit_days"]
    )


def test_input_dataframe_is_not_mutated():
    routes = frame(route("A", route_id="R1"), route("B", route_id="R2"))
    before = routes.copy(deep=True)

    simulate_routes(routes, n_simulations=100, seed=42)

    pd.testing.assert_frame_equal(routes, before)


# --- Delay behaviour ------------------------------------------------------

def test_zero_delay_probability_produces_no_delay_events():
    result = simulate_routes(
        frame(route("A", route_id="R1", transit=10.0, delay=0.0, impact=4.0)),
        n_simulations=500,
        seed=42,
    )

    assert not result.samples["delay_occurred"].any()
    assert (result.samples["delay_days"] == 0).all()
    assert (result.samples["simulated_transit_days"] == 10.0).all()
    assert result.summary.loc[0, "mean_transit_days"] == pytest.approx(10.0)


def test_certain_delay_produces_events_in_every_trial():
    result = simulate_routes(
        frame(route("A", route_id="R1", transit=10.0, delay=1.0, impact=4.0)),
        n_simulations=500,
        seed=42,
    )

    assert result.samples["delay_occurred"].all()
    assert (result.samples["delay_days"] > 0).all()
    assert (result.samples["simulated_transit_days"] > 10.0).all()


def test_zero_delay_impact_keeps_scheduled_transit():
    result = simulate_routes(
        frame(route("A", route_id="R1", transit=8.5, delay=1.0, impact=0.0)),
        n_simulations=200,
        seed=42,
    )

    assert result.samples["delay_occurred"].all()
    assert (result.samples["delay_days"] == 0).all()
    assert (result.samples["simulated_transit_days"] == 8.5).all()


# --- Statistics and physical validity ------------------------------------

def test_on_time_probability_is_bounded_and_uses_deadline():
    result = simulate_routes(
        frame(route("A", route_id="R1", transit=10.0, delay=0.3, impact=4.0)),
        n_simulations=5_000,
        seed=42,
        deadline=10.0,
    )

    probability = result.summary.loc[0, "on_time_probability"]
    assert 0.0 <= probability <= 1.0
    assert result.deadline == 10.0
    # On-time at the scheduled date happens only when no delay occurs.
    assert probability == pytest.approx(0.7, abs=0.05)


def test_percentiles_are_ordered():
    result = simulate_routes(
        frame(route("A", route_id="R1", delay=0.4, impact=6.0)),
        n_simulations=2_000,
        seed=42,
        deadline=12.0,
    )
    row = result.summary.loc[0]

    assert row["median_transit_days"] <= row["p90_transit_days"] <= row["p95_transit_days"]


def test_transit_times_remain_physically_valid():
    result = simulate_routes(
        frame(route("A", route_id="R1", transit=0.1, delay=0.5, impact=0.01)),
        n_simulations=500,
        seed=42,
    )

    samples = result.samples
    assert (samples["simulated_transit_days"] >= 0.1).all()
    assert (samples["delay_days"] >= 0).all()
    assert samples["simulated_transit_days"].map(math.isfinite).all()


def test_multiple_routes_are_simulated_independently():
    result = simulate_routes(
        frame(
            route("No delay", route_id="R1", cost=100.0, delay=0.0, impact=4.0),
            route("Always late", route_id="R2", cost=200.0, delay=1.0, impact=4.0),
        ),
        n_simulations=300,
        seed=42,
    )
    by_id = result.summary.set_index("route_id")

    assert by_id.loc["R1", "mean_transit_days"] == pytest.approx(10.0)
    assert by_id.loc["R2", "mean_transit_days"] > 10.0
    assert by_id.loc["R1", "mean_cost"] == pytest.approx(100.0)
    assert by_id.loc["R2", "mean_cost"] == pytest.approx(200.0)
    assert result.samples.groupby("route_id")["trial"].count().tolist() == [300, 300]


# --- Validation -----------------------------------------------------------

@pytest.mark.parametrize("count", [0, -1, 1.5, "100", True, MAX_SIMULATIONS + 1])
def test_invalid_simulation_counts_are_rejected(count):
    with pytest.raises(ValueError, match="n_simulations"):
        simulate_routes(frame(route()), n_simulations=count)


@pytest.mark.parametrize("seed", [-1, 1.5, "42", True, 2**32])
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        simulate_routes(frame(route()), n_simulations=10, seed=seed)


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), "invalid"])
def test_invalid_probabilities_are_rejected(probability):
    routes = frame(route())
    routes["delay_probability"] = routes["delay_probability"].astype(object)
    routes.loc[0, "delay_probability"] = probability

    with pytest.raises(ValueError, match="delay_probability"):
        simulate_routes(routes, n_simulations=10)


def test_missing_required_columns_are_rejected():
    routes = frame(route()).drop(columns=["delay_probability"])

    with pytest.raises(ValueError, match="missing required columns"):
        simulate_routes(routes, n_simulations=10)


def test_duplicate_route_ids_are_rejected():
    routes = frame(route("A", route_id="R1"), route("B", route_id="R1"))

    with pytest.raises(ValueError, match="route_id must be unique"):
        simulate_routes(routes, n_simulations=10)


@pytest.mark.parametrize("deadline", [-1, float("nan"), "soon"])
def test_invalid_deadlines_are_rejected(deadline):
    with pytest.raises(ValueError, match="deadline"):
        simulate_routes(frame(route()), n_simulations=10, deadline=deadline)


@pytest.mark.parametrize("budget", [-1, float("inf"), "cheap"])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError, match="budget"):
        simulate_routes(frame(route()), n_simulations=10, budget=budget)


@pytest.mark.parametrize("cv", [-0.1, float("nan"), "high"])
def test_invalid_cost_uncertainty_is_rejected(cv):
    with pytest.raises(ValueError, match="cost_uncertainty_cv"):
        simulate_routes(frame(route()), n_simulations=10, cost_uncertainty_cv=cv)


def test_empty_routes_return_empty_results():
    result = simulate_routes(pd.DataFrame(), n_simulations=100, seed=42)

    assert result.summary.empty
    assert result.samples.empty
    assert result.simulation_count == 100


# --- Deadline and cost behaviour -----------------------------------------

def test_missing_deadline_leaves_on_time_probability_unavailable():
    result = simulate_routes(
        frame(route("A", route_id="R1")),
        n_simulations=100,
        seed=42,
        deadline=None,
    )

    assert result.deadline is None
    assert result.summary["on_time_probability"].isna().all()
    assert result.summary["mean_transit_days"].notna().all()


def test_default_cost_is_deterministic_and_budget_probability_is_unavailable():
    result = simulate_routes(
        frame(route("A", route_id="R1", cost=123.45)),
        n_simulations=100,
        seed=42,
        budget=100.0,
    )

    assert (result.samples["simulated_cost_usd"] == 123.45).all()
    assert result.summary.loc[0, "mean_cost"] == pytest.approx(123.45)
    assert pd.isna(result.summary.loc[0, "probability_over_budget"])


def test_optional_cost_uncertainty_produces_bounded_probability():
    result = simulate_routes(
        frame(route("A", route_id="R1", cost=100.0)),
        n_simulations=20_000,
        seed=42,
        budget=100.0,
        cost_uncertainty_cv=0.20,
    )

    assert result.summary.loc[0, "mean_cost"] == pytest.approx(100.0, rel=0.03)
    probability = result.summary.loc[0, "probability_over_budget"]
    assert 0.0 <= probability <= 1.0


# --- Optimizer integration ------------------------------------------------

def test_results_integrate_with_the_existing_optimizer():
    routes = frame(
        route("Cheap", route_id="R1", cost=100.0, transit=12.0, delay=0.2, impact=3.0),
        route("Fast", route_id="R2", cost=150.0, transit=8.0, delay=0.1, impact=2.0),
        route("Safe", route_id="R3", cost=130.0, transit=10.0, delay=0.05, impact=1.0),
    )
    before = routes.copy(deep=True)
    ranked = rank_routes(routes)
    simulated = simulate_routes(routes, n_simulations=500, seed=42, deadline=11.0)

    pd.testing.assert_frame_equal(routes, before)
    assert set(simulated.summary["route_id"]) == set(ranked["route_id"])
    assert ranked["rank"].tolist() == [1, 2, 3]
    assert simulated.summary["on_time_probability"].between(0, 1).all()
    # Simulation outputs are separate from deterministic score columns.
    assert "score" not in simulated.summary.columns
    assert "risk_score" not in simulated.summary.columns
