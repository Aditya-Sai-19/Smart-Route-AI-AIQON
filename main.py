"""Command-line evaluation of one shipment's candidate routes.

Prompts for a shipment ID from the validated dataset, ranks its routes with the
default balanced weights, and prints the comparison table plus the generated
recommendation explanation. Hard constraints are available in the dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from data_generator import DATA_PATH, generate_dataset
from optimizer import explain_recommendation, rank_routes


def load_data() -> pd.DataFrame:
    """Load the checked-in dataset, regenerating it when the CSV is absent."""
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else generate_dataset()


def main() -> None:
    data = load_data()
    shipment_id = input(f"Shipment ID (e.g. {data['shipment_id'].iloc[0]}): ").strip()
    shipment = data[data["shipment_id"] == shipment_id]
    if shipment.empty:
        raise SystemExit(f"Shipment ID '{shipment_id}' not found in the dataset.")
    ranked = rank_routes(shipment)
    print(ranked[[
        "rank", "route_name", "cost_usd", "expected_transit_days",
        "risk_score", "score", "feasible",
    ]].to_string(index=False))
    print("\n" + explain_recommendation(ranked))


if __name__ == "__main__":
    main()
