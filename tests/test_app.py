"""Streamlit integration tests using Streamlit's AppTest harness."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def run_app() -> AppTest:
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=60)
    return app


def test_dashboard_renders_core_sections():
    app = run_app()

    assert not app.exception
    assert len(app.tabs) == 4
    assert app.title[0].value == "SmartRoute AI"

    labels = [metric.label for metric in app.metric]
    for expected in (
        "Shipments", "Route alternatives", "Feasible routes", "Weighted score",
        "Mean transit", "P90", "On-time probability", "Simulations",
    ):
        assert expected in labels

    assert len(app.get("plotly_chart")) == 5
    assert len(app.dataframe) == 3
    assert not app.error


def test_no_feasible_route_state_never_shows_a_success_recommendation():
    app = run_app()
    app.number_input[0].set_value(1.0).run(timeout=60)

    assert not app.exception
    assert any("No route satisfies all hard constraints" in item.value for item in app.error)
    assert all("Recommended" not in item.value for item in app.success)
    assert any("No feasible route exists" in item.value for item in app.warning)


def test_deadline_enables_on_time_probability():
    app = run_app()
    app.number_input[1].set_value(20.0).run(timeout=60)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["On-time probability"] != "—"
    assert "%" in metrics["On-time probability"]


def test_invalid_custom_weights_show_a_friendly_error():
    app = run_app()
    app.selectbox[1].set_value("Custom").run(timeout=60)
    for slider in app.slider:
        slider.set_value(0)
    app.run(timeout=60)

    assert not app.exception
    assert any("At least one preference weight" in item.value for item in app.error)
