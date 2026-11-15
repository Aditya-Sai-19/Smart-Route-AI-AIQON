"""Monte Carlo reliability simulation for shipment routes.

The simulator is deliberately separate from the deterministic optimizer. It
does not change risk scores, weighted rankings, or hard-constraint results.

Model assumptions:
- Scheduled transit time is deterministic.
- A delay occurs with the route's ``delay_probability`` (Bernoulli event).
- Conditional delay days follow a bounded triangular distribution:
  low = 0.5 x expected impact, mode = expected impact, high = 1.5 x expected
  impact. ``delay_impact_days`` is an input assumption, not a measured
  distribution.
- Cost is deterministic unless ``cost_uncertainty_cv`` is explicitly supplied.
  When supplied, cost is multiplied by a mean-one lognormal factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np
import pandas as pd

DEFAULT_SIMULATIONS = 10_000
DEFAULT_SEED = 42
MAX_SIMULATIONS = 1_000_000
_MAX_SEED = 2**32 - 1

_REQUIRED_COLUMNS = (
    "route_id", "cost_usd", "transit_days", "delay_probability",
    "delay_impact_days",
)
_SUMMARY_COLUMNS = (
    "route_id", "route_name", "mean_transit_days", "median_transit_days",
    "p90_transit_days", "p95_transit_days", "on_time_probability", "mean_cost",
    "probability_over_budget", "simulation_count",
)
_SAMPLE_COLUMNS = (
    "route_id", "route_name", "trial", "simulated_transit_days",
    "delay_occurred", "delay_days", "simulated_cost_usd",
)


@dataclass(frozen=True)
class SimulationResult:
    """Per-route summary statistics plus the underlying simulated trials."""

    summary: pd.DataFrame
    samples: pd.DataFrame
    simulation_count: int
    seed: int
    deadline: float | None
    budget: float | None
    cost_uncertainty_cv: float | None


def _validate_int(value: int, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}; got {value}.")
    return value


def _validate_limit(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number or None; got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")
    return value


def _numeric_column(routes: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(routes[column], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values)
    if invalid.any():
        ids = ", ".join(routes.loc[invalid, "route_id"].astype(str).head(3))
        raise ValueError(
            f"{column} must contain finite numeric values; invalid route(s): {ids}."
        )
    return values.astype(float)


def _empty_result(
    n_simulations: int,
    seed: int,
    deadline: float | None,
    budget: float | None,
    cost_uncertainty_cv: float | None,
) -> SimulationResult:
    return SimulationResult(
        summary=pd.DataFrame(columns=_SUMMARY_COLUMNS),
        samples=pd.DataFrame(columns=_SAMPLE_COLUMNS),
        simulation_count=n_simulations,
        seed=seed,
        deadline=deadline,
        budget=budget,
        cost_uncertainty_cv=cost_uncertainty_cv,
    )


def simulate_routes(
    routes: pd.DataFrame,
    n_simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
    deadline: float | None = None,
    budget: float | None = None,
    cost_uncertainty_cv: float | None = None,
) -> SimulationResult:
    """Simulate delivery time and, optionally, cost for each route.

    Parameters
    ----------
    routes:
        One shipment's route rows. Required columns: ``route_id``, ``cost_usd``,
        ``transit_days``, ``delay_probability`` and ``delay_impact_days``.
        ``route_name`` is used in labels when present. The input is not mutated.
    n_simulations:
        Trials per route. Default 10,000.
    seed:
        Non-negative seed for independent, reproducible route-level streams.
    deadline:
        Maximum delivery time for on-time probability. ``None`` yields NaN.
    budget:
        Budget used for probability-over-budget when cost uncertainty is on.
    cost_uncertainty_cv:
        Optional coefficient of variation for a mean-one lognormal cost
        multiplier. ``None`` (default) keeps cost deterministic.

    Returns
    -------
    SimulationResult
        ``summary`` has one row per route. ``samples`` is long-form trial data.
    """
    n_simulations = _validate_int(
        n_simulations, "n_simulations", minimum=1, maximum=MAX_SIMULATIONS
    )
    seed = _validate_int(seed, "seed", minimum=0, maximum=_MAX_SEED)
    deadline = _validate_limit(deadline, "deadline")
    budget = _validate_limit(budget, "budget")

    if cost_uncertainty_cv is not None:
        if isinstance(cost_uncertainty_cv, bool) or not isinstance(
            cost_uncertainty_cv, numbers.Real
        ):
            raise ValueError(
                "cost_uncertainty_cv must be a number or None; "
                f"got {cost_uncertainty_cv!r}."
            )
        cost_uncertainty_cv = float(cost_uncertainty_cv)
        if not math.isfinite(cost_uncertainty_cv) or cost_uncertainty_cv < 0:
            raise ValueError(
                "cost_uncertainty_cv must be finite and non-negative; "
                f"got {cost_uncertainty_cv!r}."
            )

    if routes.empty:
        return _empty_result(n_simulations, seed, deadline, budget, cost_uncertainty_cv)

    missing = [col for col in _REQUIRED_COLUMNS if col not in routes.columns]
    if missing:
        raise ValueError(
            f"Simulation is missing required columns: {', '.join(missing)}."
        )

    if routes["route_id"].isna().any():
        raise ValueError("route_id must not contain missing values.")
    if routes["route_id"].duplicated().any():
        duplicates = ", ".join(
            routes.loc[routes["route_id"].duplicated(keep=False), "route_id"]
            .astype(str).unique()[:3]
        )
        raise ValueError(f"route_id must be unique; duplicates: {duplicates}.")

    costs = _numeric_column(routes, "cost_usd")
    scheduled = _numeric_column(routes, "transit_days")
    probabilities = _numeric_column(routes, "delay_probability")
    impacts = _numeric_column(routes, "delay_impact_days")

    if (costs <= 0).any():
        raise ValueError("cost_usd must be greater than 0.")
    if (scheduled < 0).any():
        raise ValueError("transit_days must be non-negative.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("delay_probability must be between 0 and 1.")
    if (impacts < 0).any():
        raise ValueError("delay_impact_days must be non-negative.")

    has_names = "route_name" in routes.columns
    root = np.random.SeedSequence(seed)
    children = root.spawn(len(routes))
    summary_rows: list[dict] = []
    sample_frames: list[pd.DataFrame] = []

    for position, (index, row) in enumerate(routes.iterrows()):
        rng = np.random.default_rng(children[position])
        route_id = str(row["route_id"])
        route_name = (
            str(row["route_name"])
            if has_names and pd.notna(row["route_name"])
            else route_id
        )
        cost = float(costs.loc[index])
        transit = float(scheduled.loc[index])
        probability = float(probabilities.loc[index])
        impact = float(impacts.loc[index])

        delay_occurred = rng.random(n_simulations) < probability
        if impact > 0:
            delay_days = np.where(
                delay_occurred,
                rng.triangular(0.5 * impact, impact, 1.5 * impact, n_simulations),
                0.0,
            )
        else:
            delay_days = np.zeros(n_simulations)

        simulated_transit = transit + delay_days

        if cost_uncertainty_cv is None:
            simulated_cost = np.full(n_simulations, cost)
            probability_over_budget = float("nan")
        else:
            sigma = math.sqrt(math.log1p(cost_uncertainty_cv**2))
            simulated_cost = cost * rng.lognormal(
                mean=-0.5 * sigma**2, sigma=sigma, size=n_simulations
            )
            probability_over_budget = (
                float(np.mean(simulated_cost > budget))
                if budget is not None else float("nan")
            )

        summary_rows.append({
            "route_id": route_id,
            "route_name": route_name,
            "mean_transit_days": float(np.mean(simulated_transit)),
            "median_transit_days": float(np.percentile(simulated_transit, 50)),
            "p90_transit_days": float(np.percentile(simulated_transit, 90)),
            "p95_transit_days": float(np.percentile(simulated_transit, 95)),
            "on_time_probability": (
                float(np.mean(simulated_transit <= deadline))
                if deadline is not None else float("nan")
            ),
            "mean_cost": float(np.mean(simulated_cost)),
            "probability_over_budget": probability_over_budget,
            "simulation_count": n_simulations,
        })
        sample_frames.append(pd.DataFrame({
            "route_id": route_id,
            "route_name": route_name,
            "trial": np.arange(n_simulations),
            "simulated_transit_days": simulated_transit,
            "delay_occurred": delay_occurred,
            "delay_days": delay_days,
            "simulated_cost_usd": simulated_cost,
        }))

    return SimulationResult(
        summary=pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS),
        samples=pd.concat(sample_frames, ignore_index=True),
        simulation_count=n_simulations,
        seed=seed,
        deadline=deadline,
        budget=budget,
        cost_uncertainty_cv=cost_uncertainty_cv,
    )
