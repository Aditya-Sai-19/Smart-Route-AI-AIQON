"""Reproducible synthetic shipment-route dataset and its validation rules.

All values are illustrative. They are not live carrier quotations, schedules,
or risk intelligence.
"""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

ORIGINS = [
    ("Shanghai", "China"), ("Mumbai", "India"), ("Singapore", "Singapore"),
    ("Los Angeles", "USA"), ("Santos", "Brazil"), ("Rotterdam", "Netherlands"),
    ("Jebel Ali", "UAE"), ("Busan", "South Korea"),
]
DESTINATIONS = [
    ("Rotterdam", "Netherlands"), ("Hamburg", "Germany"), ("New York", "USA"),
    ("Singapore", "Singapore"), ("Mumbai", "India"), ("Antwerp", "Belgium"),
    ("Felixstowe", "UK"), ("Dubai", "UAE"),
]
HUBS = [
    "Colombo", "Port Said", "Suez Canal", "Jebel Ali", "Singapore",
    "Tangier Med", "Algeciras", "Durban", "Piraeus", "Salalah",
]
ROUTE_NAMES = [
    "Direct Express", "Regional Hub", "Economy Corridor",
    "Resilient Route", "Balanced Corridor",
]
# (cost factor, transit-time factor, risk factor) per candidate route.
ROUTE_PROFILES = [
    (0.88, 0.82, 1.28),  # faster, often riskier
    (1.05, 1.12, 0.72),  # safer, slower and costlier
    (0.96, 1.00, 1.00),  # middle ground
    (1.15, 0.92, 0.62),  # expensive resilience option
]

ROUTES_PER_SHIPMENT = 4
MIN_ROUTES_PER_SHIPMENT = 3
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "shipments.csv"

REQUIRED_COLUMNS = (
    "shipment_id", "origin", "destination", "route_id", "route_name",
    "intermediate_hubs", "legs", "cost_usd", "transit_days",
    "delay_probability", "delay_impact_days", "congestion_risk",
    "geopolitical_risk",
)
NUMERIC_COLUMNS = (
    "cost_usd", "transit_days", "delay_probability", "delay_impact_days",
    "congestion_risk", "geopolitical_risk",
)
_ID_COLUMNS = ("shipment_id", "route_id")

# (column, validity predicate, human-readable rule)
_VALUE_RULES = (
    ("cost_usd", lambda s: s > 0, "must be greater than 0"),
    ("transit_days", lambda s: s > 0, "must be greater than 0"),
    ("delay_probability", lambda s: s.between(0, 1), "must be between 0 and 1"),
    ("delay_impact_days", lambda s: s >= 0, "must be non-negative"),
    ("congestion_risk", lambda s: s.between(0, 100), "must be between 0 and 100"),
    ("geopolitical_risk", lambda s: s.between(0, 100), "must be between 0 and 100"),
)


def _examples(df: pd.DataFrame, mask: pd.Series, limit: int = 3) -> str:
    """Return a short, readable list of route IDs for an error message."""
    ids = df.loc[mask, "route_id"].astype(str).tolist()
    shown = ", ".join(ids[:limit])
    remaining = len(ids) - limit
    return f"{shown} and {remaining} more" if remaining > 0 else shown


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate a shipment-route dataset and raise one aggregated ValueError.

    Rules cover required columns, nulls, numeric ranges, unique IDs, unique
    shipment-route pairs, distinct endpoints, and at least three routes per
    shipment. The message names the offending column and example route IDs.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}.")
    if df.empty:
        raise ValueError("Dataset contains no records.")

    errors: list[str] = []

    null_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    for col, count in null_counts[null_counts > 0].items():
        errors.append(
            f"{col} has {int(count)} missing value(s): "
            f"{_examples(df, df[col].isna())}."
        )

    numeric: dict[str, pd.Series] = {}
    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_numeric = df[col].notna() & coerced.isna()
        if non_numeric.any():
            errors.append(
                f"{col} must be numeric; invalid route(s): {_examples(df, non_numeric)}."
            )
        numeric[col] = coerced

    for col, is_valid, rule in _VALUE_RULES:
        values = numeric[col]
        invalid = values.notna() & ~is_valid(values)
        if invalid.any():
            errors.append(
                f"{col} {rule}; invalid route(s): {_examples(df, invalid)}."
            )

    duplicate_routes = df["route_id"].duplicated(keep=False)
    if duplicate_routes.any():
        errors.append(f"route_id must be unique; duplicate route_id(s): {_examples(df, duplicate_routes)}.")

    duplicate_combinations = df.duplicated(["shipment_id", "route_id"], keep=False)
    if duplicate_combinations.any():
        errors.append(
            "Duplicate shipment_id-route_id combination(s): "
            f"{_examples(df, duplicate_combinations)}."
        )

    same_endpoints = (
        df["origin"].astype(str).str.strip().str.casefold()
        == df["destination"].astype(str).str.strip().str.casefold()
    )
    if same_endpoints.any():
        errors.append(
            "origin and destination must differ; invalid route(s): "
            f"{_examples(df, same_endpoints)}."
        )

    route_counts = df.groupby("shipment_id").size()
    under_served = route_counts[route_counts < MIN_ROUTES_PER_SHIPMENT]
    if not under_served.empty:
        details = ", ".join(
            f"{shipment_id} has {count}" for shipment_id, count in under_served.items()
        )
        errors.append(
            f"Every shipment needs at least {MIN_ROUTES_PER_SHIPMENT} route options; {details}."
        )

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))


def generate_dataset(seed: int = 42, shipments: int = 30) -> pd.DataFrame:
    """Generate 30 reproducible shipments with four candidate routes each.

    The fixed seed makes the output identical across runs. Values are
    illustrative and are validated before the frame is returned.
    """
    rng = random.Random(seed)
    rows = []
    for n in range(1, shipments + 1):
        origin = rng.choice(ORIGINS)
        destination = rng.choice([d for d in DESTINATIONS if d[0] != origin[0]])
        base_cost = rng.randint(2500, 8500)
        base_days = rng.randint(15, 38)
        for option in range(1, ROUTES_PER_SHIPMENT + 1):
            cost_factor, time_factor, risk_factor = ROUTE_PROFILES[option - 1]
            cost = round(base_cost * cost_factor * rng.uniform(.94, 1.06), 2)
            transit = round(base_days * time_factor * rng.uniform(.94, 1.06), 1)
            delay = round(min(.85, max(.03, rng.uniform(.08, .35) * risk_factor)), 3)
            congestion = round(min(95, max(5, rng.uniform(15, 65) * risk_factor)), 1)
            geopolitical = round(min(95, max(3, rng.uniform(8, 55) * risk_factor)), 1)
            hubs = rng.sample(HUBS, k=rng.randint(1, 3))
            rows.append({
                "shipment_id": f"SHP-{n:03d}",
                "origin": f"{origin[0]}, {origin[1]}",
                "destination": f"{destination[0]}, {destination[1]}",
                "route_id": f"SHP-{n:03d}-R{option}",
                "route_name": ROUTE_NAMES[option - 1],
                "intermediate_hubs": " → ".join(hubs),
                "legs": " → ".join([origin[0], *hubs, destination[0]]),
                "cost_usd": cost,
                "transit_days": transit,
                "delay_probability": delay,
                "delay_impact_days": round(rng.uniform(2, 8), 1),
                "congestion_risk": congestion,
                "geopolitical_risk": geopolitical,
            })

    dataset = pd.DataFrame(rows)
    validate_dataset(dataset)
    return dataset


def main() -> None:
    """Regenerate and validate data/shipments.csv."""
    dataset = generate_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATA_PATH, index=False, encoding="utf-8")
    print(
        f"Generated {len(dataset)} route options for "
        f"{dataset['shipment_id'].nunique()} shipments at {DATA_PATH}"
    )


if __name__ == "__main__":
    main()
