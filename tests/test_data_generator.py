"""Tests for synthetic dataset generation and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from data_generator import (
    MIN_ROUTES_PER_SHIPMENT,
    REQUIRED_COLUMNS,
    generate_dataset,
    validate_dataset,
)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset()


def test_dataset_has_30_shipments_and_120_routes(dataset):
    assert len(dataset) == 120
    assert dataset["shipment_id"].nunique() == 30
    assert dataset["route_id"].nunique() == 120
    assert dataset.groupby("shipment_id").size().eq(4).all()
    assert dataset["route_id"].is_unique


def test_dataset_contains_all_required_fields(dataset):
    assert list(dataset.columns) == list(REQUIRED_COLUMNS)
    assert dataset.notna().all().all()


def test_dataset_values_are_within_valid_ranges(dataset):
    assert (dataset["cost_usd"] > 0).all()
    assert (dataset["transit_days"] > 0).all()
    assert dataset["delay_probability"].between(0, 1).all()
    assert (dataset["delay_impact_days"] >= 0).all()
    assert dataset["congestion_risk"].between(0, 100).all()
    assert dataset["geopolitical_risk"].between(0, 100).all()


def test_origin_and_destination_differ(dataset):
    endpoints_differ = dataset["origin"] != dataset["destination"]
    assert endpoints_differ.all()


def test_every_shipment_has_at_least_three_routes(dataset):
    assert dataset.groupby("shipment_id").size().min() >= MIN_ROUTES_PER_SHIPMENT


def test_generation_is_reproducible_with_fixed_seed():
    pd.testing.assert_frame_equal(generate_dataset(seed=123), generate_dataset(seed=123))


def test_different_seeds_change_the_dataset():
    assert not generate_dataset(seed=1).equals(generate_dataset(seed=2))


def test_csv_round_trip_remains_valid(dataset, tmp_path):
    path = tmp_path / "shipments.csv"
    dataset.to_csv(path, index=False, encoding="utf-8")
    reloaded = pd.read_csv(path)

    validate_dataset(reloaded)
    assert len(reloaded) == 120
    assert reloaded["intermediate_hubs"].str.contains("→").any()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("cost_usd", 0, "greater than 0"),
        ("cost_usd", -1, "greater than 0"),
        ("transit_days", 0, "greater than 0"),
        ("transit_days", -2, "greater than 0"),
        ("delay_probability", -0.01, "between 0 and 1"),
        ("delay_probability", 1.01, "between 0 and 1"),
        ("delay_impact_days", -1, "non-negative"),
        ("congestion_risk", -1, "between 0 and 100"),
        ("congestion_risk", 101, "between 0 and 100"),
        ("geopolitical_risk", -1, "between 0 and 100"),
        ("geopolitical_risk", 101, "between 0 and 100"),
        ("cost_usd", "not-a-number", "must be numeric"),
        ("congestion_risk", None, "missing value"),
    ],
)
def test_invalid_dataset_values_raise_meaningful_errors(dataset, column, value, message):
    data = dataset.copy()
    data[column] = data[column].astype(object)
    data.loc[0, column] = value

    with pytest.raises(ValueError, match=message) as error:
        validate_dataset(data)

    assert column in str(error.value)
    assert "SHP-001-R1" in str(error.value)


def test_duplicate_route_ids_are_rejected(dataset):
    data = dataset.copy()
    data.loc[1, "route_id"] = data.loc[0, "route_id"]

    with pytest.raises(ValueError, match="route_id must be unique"):
        validate_dataset(data)


def test_duplicate_shipment_route_combinations_are_rejected(dataset):
    data = dataset.copy()
    data.loc[1, "route_id"] = data.loc[0, "route_id"]

    with pytest.raises(ValueError, match="Duplicate shipment_id-route_id"):
        validate_dataset(data)


def test_fewer_than_three_routes_per_shipment_is_rejected(dataset):
    data = dataset[~(
        (dataset["shipment_id"] == "SHP-001")
        & (dataset["route_id"].isin(["SHP-001-R3", "SHP-001-R4"]))
    )]

    with pytest.raises(ValueError, match="at least 3 route options"):
        validate_dataset(data)


def test_identical_origin_and_destination_is_rejected(dataset):
    data = dataset.copy()
    data.loc[0, "destination"] = data.loc[0, "origin"]

    with pytest.raises(ValueError, match="origin and destination must differ"):
        validate_dataset(data)


def test_missing_required_columns_are_rejected(dataset):
    data = dataset.drop(columns=["delay_probability"])

    with pytest.raises(ValueError, match="missing required columns: delay_probability"):
        validate_dataset(data)


def test_empty_dataset_is_rejected(dataset):
    with pytest.raises(ValueError, match="no records"):
        validate_dataset(dataset.iloc[0:0])
