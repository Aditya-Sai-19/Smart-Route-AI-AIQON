"""Multi-objective shipment-route evaluation, constraints, Pareto, and explanations.

All scores are relative decision-support indicators for one shipment. They are
not probabilities or calibrated financial estimates.
"""

from __future__ import annotations

import math
from typing import Dict

import pandas as pd

METRICS = ("cost_usd", "expected_transit_days", "risk_score")
WEIGHT_KEYS = ("cost", "time", "risk")
DEFAULT_WEIGHTS = {"cost": 0.40, "time": 0.35, "risk": 0.25}
PREFERENCE_PROFILES = {
    "Cost focused": {"cost": 0.70, "time": 0.20, "risk": 0.10},
    "Balanced": {"cost": 0.40, "time": 0.35, "risk": 0.25},
    "Risk averse": {"cost": 0.15, "time": 0.20, "risk": 0.65},
    "Time critical": {"cost": 0.15, "time": 0.70, "risk": 0.15},
}
_REQUIRED_INPUT_COLUMNS = (
    "cost_usd", "transit_days", "delay_probability", "delay_impact_days",
    "congestion_risk", "geopolitical_risk",
)
_TOL = 1e-9


def add_derived_metrics(routes: pd.DataFrame) -> pd.DataFrame:
    """Add expected transit time and the composite operational risk score.

    Expected transit time = scheduled transit days
                            + delay probability x delay impact days.

    Risk score = 0.45 x delay probability x 100
                 + 0.30 x congestion risk
                 + 0.25 x geopolitical risk.

    Delay risk is weighted highest because schedule uncertainty directly
    affects delivery commitments. Congestion is the next strongest signal of
    operational variability. Geopolitical risk matters but is rarer and less
    directly observable. Weights are module constants so a future version can
    make them configurable per shipment type or stakeholder.
    """
    df = routes.copy()
    df["expected_transit_days"] = (
        df["transit_days"] + df["delay_probability"] * df["delay_impact_days"]
    )
    # Inputs are on a 0-100 scale; clipping keeps the indicator in range even
    # if a caller supplies out-of-range values directly.
    df["risk_score"] = (
        0.45 * df["delay_probability"] * 100
        + 0.30 * df["congestion_risk"]
        + 0.25 * df["geopolitical_risk"]
    ).clip(0, 100)
    return df


def normalize_weights(weights: Dict[str, float] | None = None) -> Dict[str, float]:
    """Validate preference weights and normalize them to sum to 1.

    Missing keys count as zero. Unknown, non-numeric, negative, non-finite,
    or all-zero weights are rejected rather than silently clamped.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if not isinstance(weights, dict):
        raise ValueError("Weights must be provided as a dictionary.")

    unknown = sorted(set(weights) - set(WEIGHT_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown weight key(s): {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(WEIGHT_KEYS)}."
        )

    cleaned: dict[str, float] = {}
    for key in WEIGHT_KEYS:
        raw = weights.get(key, 0)
        if isinstance(raw, bool):
            raise ValueError(f"Weight '{key}' must be a number, not a boolean.")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Weight '{key}' must be a number; got {raw!r}.") from None
        if not math.isfinite(value):
            raise ValueError(f"Weight '{key}' must be finite; got {raw!r}.")
        if value < 0:
            raise ValueError(f"Weight '{key}' must be non-negative; got {value}.")
        cleaned[key] = value

    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("At least one preference weight must be greater than zero.")
    return {key: value / total for key, value in cleaned.items()}


def _minmax(series: pd.Series) -> pd.Series:
    """Scale a metric to [0, 1], returning zeros when every value is equal."""
    low, high = series.min(), series.max()
    if high == low:
        return pd.Series(0.0, index=series.index)
    return (series - low) / (high - low)


def pareto_frontier(df: pd.DataFrame) -> pd.Series:
    """Mark routes not dominated on cost, expected time, and risk.

    A route is dominated when another route is at least as good on all three
    metrics and strictly better on at least one. Identical routes do not
    dominate each other. Callers should pass feasible routes only so an
    infeasible option can never dominate a feasible one.
    """
    values = df[list(METRICS)].to_numpy()
    keep = []
    for i, candidate in enumerate(values):
        dominated = any(
            j != i
            and (other <= candidate).all()
            and (other < candidate).any()
            for j, other in enumerate(values)
        )
        keep.append(not dominated)
    return pd.Series(keep, index=df.index)


def rank_routes(
    routes: pd.DataFrame,
    weights: Dict[str, float] | None = None,
    max_budget: float | None = None,
    max_days: float | None = None,
) -> pd.DataFrame:
    """Evaluate and rank one shipment's routes. Lower composite score is better.

    Cost, expected transit time, and risk are min-max normalized across the
    feasible routes for this shipment. Budget and expected-time limits are hard
    constraints: infeasible routes stay visible but receive no score, rank, or
    Pareto status and can never be recommended.
    """
    if routes.empty:
        return routes.copy()

    missing = [col for col in _REQUIRED_INPUT_COLUMNS if col not in routes.columns]
    if missing:
        raise ValueError(f"Routes are missing required columns: {', '.join(missing)}.")

    weights = normalize_weights(weights)
    df = add_derived_metrics(routes)

    df["violates_budget"] = (
        df["cost_usd"] > max_budget if max_budget is not None else False
    )
    df["violates_deadline"] = (
        df["expected_transit_days"] > max_days if max_days is not None else False
    )
    df["feasible"] = ~(df["violates_budget"] | df["violates_deadline"])

    def constraint_status(row: pd.Series) -> str:
        reasons = []
        if row["violates_budget"]:
            reasons.append(
                f"cost ${row['cost_usd']:,.2f} exceeds budget ${max_budget:,.2f}"
            )
        if row["violates_deadline"]:
            reasons.append(
                f"expected {row['expected_transit_days']:.1f} days "
                f"exceeds deadline {max_days:.1f} days"
            )
        return "Infeasible: " + " and ".join(reasons) if reasons else "Feasible"

    df["recommendation_status"] = df.apply(constraint_status, axis=1)

    for col in ("cost_component", "time_component", "risk_component", "score", "rank"):
        df[col] = float("nan")
    df["pareto_optimal"] = False

    feasible = df[df["feasible"]]
    if not feasible.empty:
        # Normalize within this shipment's feasible set: 0 = best, 1 = worst.
        cost_component = _minmax(feasible["cost_usd"])
        time_component = _minmax(feasible["expected_transit_days"])
        risk_component = _minmax(feasible["risk_score"])
        df.loc[feasible.index, "cost_component"] = cost_component
        df.loc[feasible.index, "time_component"] = time_component
        df.loc[feasible.index, "risk_component"] = risk_component
        df.loc[feasible.index, "score"] = (
            weights["cost"] * cost_component
            + weights["time"] * time_component
            + weights["risk"] * risk_component
        )
        df.loc[feasible.index, "rank"] = (
            df.loc[feasible.index, "score"].rank(method="first").astype(int)
        )
        df.loc[feasible.index, "pareto_optimal"] = (
            pareto_frontier(feasible).astype(bool)
        )

    # Feasible routes first, then by weighted score. Infeasible routes remain
    # visible for comparison at the bottom of the table.
    result = df.sort_values(
        ["feasible", "score", "cost_usd"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    result["rank"] = result["rank"].astype("Int64")
    return result


def _metric_edges(best: pd.Series, other: pd.Series) -> tuple[list[str], list[str]]:
    """Return phrases for where the best route wins and where the other wins."""
    best_edges: list[str] = []
    other_edges: list[str] = []

    cost_gap = float(other["cost_usd"]) - float(best["cost_usd"])
    if cost_gap > _TOL:
        best_edges.append(f"cheaper by ${cost_gap:,.2f}")
    elif cost_gap < -_TOL:
        other_edges.append(f"cheaper by ${-cost_gap:,.2f}")

    time_gap = float(other["expected_transit_days"]) - float(best["expected_transit_days"])
    if time_gap > _TOL:
        best_edges.append(f"faster by {time_gap:.1f} days")
    elif time_gap < -_TOL:
        other_edges.append(f"faster by {-time_gap:.1f} days")

    risk_gap = float(other["risk_score"]) - float(best["risk_score"])
    if risk_gap > _TOL:
        best_edges.append(f"lower risk by {risk_gap:.1f} points")
    elif risk_gap < -_TOL:
        other_edges.append(f"lower risk by {-risk_gap:.1f} points")

    return best_edges, other_edges


def _route_label(row: pd.Series) -> str:
    route_id = row.get("route_id")
    if route_id is not None and pd.notna(route_id):
        return f"{row['route_name']} ({route_id})"
    return str(row["route_name"])


def explain_recommendation(ranked: pd.DataFrame) -> str:
    """Build a plain-language explanation from the actual evaluation results."""
    if ranked.empty or "feasible" not in ranked.columns:
        return "No candidate routes are available, so no recommendation can be made."

    feasible = ranked[ranked["feasible"]].sort_values("score")
    if feasible.empty:
        total = len(ranked)
        budget_fail = (
            int(ranked["violates_budget"].sum())
            if "violates_budget" in ranked.columns else 0
        )
        deadline_fail = (
            int(ranked["violates_deadline"].sum())
            if "violates_deadline" in ranked.columns else 0
        )
        clauses = []
        if budget_fail:
            clauses.append(
                f"{budget_fail} of {total} candidate routes exceed the maximum budget"
            )
        if deadline_fail:
            clauses.append(
                f"{deadline_fail} of {total} exceed the maximum expected transit time"
            )
        detail = " and ".join(clauses) or "every candidate violates a hard constraint"
        review = (
            "review the budget"
            if budget_fail and not deadline_fail
            else "review the delivery deadline"
            if deadline_fail and not budget_fail
            else "review the budget or delivery deadline"
        )
        return (
            f"No route satisfies all hard constraints: {detail}. The cheapest option "
            f"costs ${ranked['cost_usd'].min():,.2f} and the fastest needs "
            f"{ranked['expected_transit_days'].min():.1f} expected days. Please "
            f"{review}. Infeasible routes remain visible in the comparison but were "
            "not recommended."
        )

    best = feasible.iloc[0]
    label = _route_label(best)
    parts = [
        f"Recommended {label} at an estimated cost of "
        f"${best['cost_usd']:,.2f}, expected transit time of "
        f"{best['expected_transit_days']:.1f} days, and composite risk score of "
        f"{best['risk_score']:.1f}/100."
    ]

    if len(feasible) == 1:
        parts.append(
            "It is the only route that satisfies the selected hard constraints, "
            "so it also has the lowest weighted score."
        )
    else:
        second = feasible.iloc[1]
        second_label = _route_label(second)
        best_edges, other_edges = _metric_edges(best, second)
        best_score = float(best["score"])
        second_score = float(second["score"])
        if abs(best_score - second_score) <= _TOL:
            score_phrase = (
                f"the weighted scores tie at {best_score:.3f}, and this route is "
                "listed first by a deterministic tie-break"
            )
        else:
            score_phrase = (
                f"the weighted score still favors {label} "
                f"({best_score:.3f} vs {second_score:.3f})"
            )

        if other_edges:
            reason = (
                f" because it is {' and '.join(best_edges)}"
                if best_edges else ""
            )
            parts.append(
                f"Although {second_label} is {' and '.join(other_edges)}, "
                f"{score_phrase}{reason}."
            )
        elif best_edges:
            parts.append(
                f"It matches or beats {second_label} on every objective "
                f"({', '.join(best_edges)}), and {score_phrase}."
            )
        else:
            parts.append(
                f"It has identical objectives to {second_label}, and {score_phrase}."
            )

    if bool(best["pareto_optimal"]):
        parts.append(
            "It is Pareto-optimal: no other feasible route is at least as good on "
            "all three metrics and strictly better on at least one."
        )
    else:
        parts.append(
            "It is not Pareto-optimal: another feasible route is at least as good "
            "on all three metrics and strictly better on at least one, although the "
            "selected weights still give this route the lowest score."
        )

    excluded = int((~ranked["feasible"]).sum())
    if excluded:
        parts.append(
            f"{excluded} candidate route(s) were rejected by the hard budget/deadline "
            "constraints; they remain visible for comparison but were not eligible "
            "for recommendation."
        )
    else:
        parts.append("No candidate routes were rejected by the hard constraints.")

    return " ".join(parts)


def profile_sensitivity(
    routes: pd.DataFrame,
    max_budget: float | None = None,
    max_days: float | None = None,
) -> pd.DataFrame:
    """Recommend a route under the four default stakeholder preference profiles."""
    rows = []
    for name, weights in PREFERENCE_PROFILES.items():
        ranked = rank_routes(routes, weights, max_budget, max_days)
        feasible = ranked[ranked["feasible"]].sort_values("score")
        if feasible.empty:
            rows.append({
                "profile": name,
                "recommended_route": "No feasible route",
                "route_id": None,
                "cost_usd": None,
                "expected_transit_days": None,
                "risk_score": None,
                "score": None,
                "pareto_optimal": None,
            })
        else:
            best = feasible.iloc[0]
            rows.append({
                "profile": name,
                "recommended_route": best["route_name"],
                "route_id": best.get("route_id"),
                "cost_usd": float(best["cost_usd"]),
                "expected_transit_days": float(best["expected_transit_days"]),
                "risk_score": float(best["risk_score"]),
                "score": round(float(best["score"]), 3),
                "pareto_optimal": bool(best["pareto_optimal"]),
            })
    return pd.DataFrame(rows)
