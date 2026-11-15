"""SmartRoute AI — interactive route decision support dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_generator import DATA_PATH, generate_dataset, validate_dataset
from optimizer import (
    PREFERENCE_PROFILES,
    add_derived_metrics,
    explain_recommendation,
    normalize_weights,
    profile_sensitivity,
    rank_routes,
)
from simulation import simulate_routes

st.set_page_config(
    page_title="SmartRoute AI | Route Decision Support",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "ink": "#0f172a",
    "muted": "#94a3b8",
    "primary": "#0f766e",
    "blue": "#2563eb",
    "accent": "#d97706",
    "danger": "#b91c1c",
}
CHART_COLORS = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#be123c", "#0891b2"]
PROFILE_ORDER = ["Balanced", "Cost focused", "Risk averse", "Time critical"]

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp { background-color: #f8fafc; }
    .block-container { padding-top: 5rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { color: #0f172a; letter-spacing: -0.02em; }
    div[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }
    div[data-testid="stMetricLabel"] { color: #64748b; }
    div[data-testid="stMetricValue"] { color: #0f172a; font-weight: 650; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { padding-bottom: 0.7rem; font-weight: 600; }
    div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 10px; }
    .sr-kicker {
        color: #0f766e; font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.2rem;
    }
    .sr-badge {
        display: inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 650; margin-right: 0.35rem;
        border: 1px solid transparent;
    }
    .sr-badge.good { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
    .sr-badge.warn { background: #fffbeb; color: #b45309; border-color: #fde68a; }
    .sr-badge.neutral { background: #f1f5f9; color: #475569; border-color: #cbd5e1; }
    .sr-note { color: #64748b; font-size: 0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def badge(text: str, kind: str = "good") -> str:
    return f'<span class="sr-badge {kind}">{text}</span>'


def score_text(value: float) -> str:
    return "—" if pd.isna(value) else f"{float(value):.3f}"


def percent_text(value: float) -> str:
    return "—" if pd.isna(value) else f"{float(value):.1%}"


def style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=64, b=12),
        font=dict(family='Inter, "Segoe UI", sans-serif', color=PALETTE["ink"], size=12),
        title=dict(
            x=0, xanchor="left", y=0.99, yref="container", yanchor="top",
            font=dict(size=15, color=PALETTE["ink"]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white", font_size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    return fig


def hover_rows(df: pd.DataFrame) -> list[list[str]]:
    return [
        [
            str(row["route_id"]),
            str(row["route_name"]),
            f"{float(row['risk_score']):.1f}",
            score_text(row["score"]),
            "Yes" if bool(row["pareto_optimal"]) else "No",
            str(row["recommendation_status"]),
        ]
        for _, row in df.iterrows()
    ]


def cost_time_chart(ranked: pd.DataFrame, recommended_id: str | None) -> go.Figure:
    fig = go.Figure()
    feasible = ranked[ranked["feasible"]]
    infeasible = ranked[~ranked["feasible"]]
    recommended = feasible[feasible["route_id"] == recommended_id] if recommended_id else feasible.iloc[0:0]
    pareto_others = feasible[
        feasible["pareto_optimal"] & ~feasible["route_id"].isin(recommended["route_id"])
    ]
    dominated = feasible[~feasible["pareto_optimal"]]

    def add(subset: pd.DataFrame, name: str, color: str, symbol: str, size: int) -> None:
        if subset.empty:
            return
        fig.add_trace(go.Scatter(
            x=subset["expected_transit_days"],
            y=subset["cost_usd"],
            mode="markers",
            name=name,
            customdata=hover_rows(subset),
            marker=dict(
                color=color, size=size, symbol=symbol,
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Cost $%{y:,.0f} · expected %{x:.1f} days<br>"
                "Risk %{customdata[2]}/100 · score %{customdata[3]}<br>"
                "Pareto %{customdata[4]}<br>%{customdata[5]}<extra></extra>"
            ),
        ))

    add(infeasible, "Infeasible (excluded)", PALETTE["muted"], "x-thin", 11)
    add(dominated, "Feasible · dominated", PALETTE["blue"], "circle", 12)
    add(pareto_others, "Feasible · Pareto-optimal", PALETTE["primary"], "diamond", 14)
    add(recommended, "Recommended", PALETTE["accent"], "star", 20)

    fig.update_xaxes(title="Expected transit time (days)")
    fig.update_yaxes(title="Cost (USD)")
    fig.update_layout(title="Cost vs expected transit time")
    return style_figure(fig)


def risk_chart(ranked: pd.DataFrame, recommended_id: str | None) -> go.Figure:
    data = ranked.sort_values("risk_score")
    colors = [
        PALETTE["accent"] if row["route_id"] == recommended_id
        else PALETTE["primary"] if row["feasible"]
        else PALETTE["muted"]
        for _, row in data.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=data["risk_score"],
        y=data["route_name"],
        orientation="h",
        marker_color=colors,
        text=[f"{value:.1f}" for value in data["risk_score"]],
        textposition="outside",
        customdata=hover_rows(data),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>Risk %{x:.1f}/100<br>"
            "%{customdata[5]}<extra></extra>"
        ),
    ))
    fig.update_xaxes(title="Deterministic risk score (0–100)", range=[0, 105])
    fig.update_yaxes(title="")
    fig.update_layout(title="Route risk comparison", showlegend=False)
    return style_figure(fig, height=330)


def ranking_chart(ranked: pd.DataFrame, recommended_id: str | None) -> go.Figure:
    data = ranked[ranked["feasible"]].sort_values("rank", ascending=False)
    colors = [
        PALETTE["accent"] if row["route_id"] == recommended_id else PALETTE["primary"]
        for _, row in data.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=data["score"],
        y=data["route_name"],
        orientation="h",
        marker_color=colors,
        text=[f"#{int(row['rank'])} · {row['score']:.3f}" for _, row in data.iterrows()],
        textposition="outside",
        customdata=hover_rows(data),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>Weighted score %{x:.3f}<br>"
            "Pareto %{customdata[4]}<extra></extra>"
        ),
    ))
    fig.update_xaxes(title="Weighted score (lower is better)", range=[0, 1.08])
    fig.update_yaxes(title="")
    fig.update_layout(title="Optimization ranking", showlegend=False)
    return style_figure(fig, height=330)


def distribution_chart(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    route_ids: list[str],
    recommended_id: str | None,
    deadline: float | None,
) -> go.Figure:
    fig = go.Figure()
    for position, route_id in enumerate(route_ids):
        subset = samples[samples["route_id"] == route_id]
        if subset.empty:
            continue
        fig.add_trace(go.Histogram(
            x=subset["simulated_transit_days"],
            name=str(subset["route_name"].iloc[0]),
            nbinsx=55,
            opacity=0.45,
            histnorm="probability",
            marker_color=CHART_COLORS[position % len(CHART_COLORS)],
        ))

    if recommended_id and not summary.empty:
        row = summary[summary["route_id"] == recommended_id]
        if not row.empty:
            p50 = float(row["median_transit_days"].iloc[0])
            p90 = float(row["p90_transit_days"].iloc[0])
            fig.add_vline(
                x=p50, line_dash="dot", line_color=PALETTE["ink"],
                annotation_text=f"P50 {p50:.1f}d", annotation_position="top left",
            )
            fig.add_vline(
                x=p90, line_dash="dash", line_color=PALETTE["primary"],
                annotation_text=f"P90 {p90:.1f}d", annotation_position="top right",
            )
    if deadline is not None:
        fig.add_vline(
            x=deadline, line_dash="dash", line_color=PALETTE["danger"],
            annotation_text=f"Deadline {deadline:.1f}d", annotation_position="top",
        )

    fig.update_layout(
        title="Simulated transit-time distribution",
        barmode="overlay",
        xaxis_title="Simulated delivery time (days)",
        yaxis_title="Share of trials",
    )
    return style_figure(fig, height=390)


def on_time_chart(summary: pd.DataFrame, feasible_ids: list[str], recommended_id: str | None) -> go.Figure:
    data = summary[summary["route_id"].isin(feasible_ids)].sort_values("on_time_probability")
    colors = [
        PALETTE["accent"] if row["route_id"] == recommended_id else PALETTE["primary"]
        for _, row in data.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=data["on_time_probability"],
        y=data["route_name"],
        orientation="h",
        marker_color=colors,
        text=[f"{value:.1%}" for value in data["on_time_probability"]],
        textposition="outside",
        customdata=[[str(row["route_id"])] for _, row in data.iterrows()],
        hovertemplate=(
            "<b>%{y}</b><br>On-time probability %{x:.1%}<extra></extra>"
        ),
    ))
    fig.update_xaxes(title="Probability of meeting the deadline", range=[0, 1.12], tickformat=".0%")
    fig.update_yaxes(title="")
    fig.update_layout(title="Simulated on-time probability", showlegend=False)
    return style_figure(fig, height=330)


def sensitivity_chart(sensitivity: pd.DataFrame) -> go.Figure | None:
    scored = sensitivity.dropna(subset=["score"])
    if scored.empty:
        return None
    route_names = list(scored["recommended_route"].unique())
    color_map = {name: CHART_COLORS[i % len(CHART_COLORS)] for i, name in enumerate(route_names)}
    fig = go.Figure(go.Bar(
        x=scored["profile"],
        y=scored["score"],
        marker_color=[color_map[name] for name in scored["recommended_route"]],
        text=scored["recommended_route"],
        textposition="outside",
        customdata=[
            [row["recommended_route"], f"{row['cost_usd']:,.0f}",
             f"{row['expected_transit_days']:.1f}", f"{row['risk_score']:.1f}"]
            for _, row in scored.iterrows()
        ],
        hovertemplate=(
            "<b>%{x}</b><br>Recommended: %{customdata[0]}<br>"
            "Cost $%{customdata[1]} · expected %{customdata[2]} days<br>"
            "Risk %{customdata[3]}/100 · score %{y:.3f}<extra></extra>"
        ),
    ))
    top = max(0.1, float(scored["score"].max()) * 1.3)
    fig.update_yaxes(title="Weighted score (lower is better)", range=[0, top])
    fig.update_xaxes(title="")
    fig.update_layout(title="Recommendation by preference profile", showlegend=False)
    return style_figure(fig, height=390)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        data = generate_dataset()
    validate_dataset(data)
    return data


@st.cache_data(show_spinner=False)
def cached_simulation(
    routes: pd.DataFrame, n_simulations: int, seed: int, deadline: float | None,
):
    return simulate_routes(
        routes,
        n_simulations=n_simulations,
        seed=seed,
        deadline=deadline,
    )


# --- Data and controls -----------------------------------------------------

try:
    data = load_data()
except Exception as exc:  # noqa: BLE001 - keep the UI free of tracebacks
    st.markdown('<div class="sr-kicker">Logistics decision support</div>', unsafe_allow_html=True)
    st.title("SmartRoute AI")
    st.error("The route dataset could not be loaded or validated.")
    st.code(str(exc), language=None)
    st.stop()

if data.empty:
    st.error("The dataset contains no shipments, so there is nothing to evaluate.")
    st.stop()

shipment_ids = sorted(data["shipment_id"].unique())
labels = {}
for shipment_id, group in data.groupby("shipment_id"):
    first = group.iloc[0]
    labels[shipment_id] = f"{shipment_id} · {first['origin']} → {first['destination']}"

with st.sidebar:
    st.markdown('<div class="sr-kicker">SmartRoute AI</div>', unsafe_allow_html=True)
    st.markdown("### Shipment controls")
    st.caption("Synthetic demonstration data. No live carrier feeds.")
    selected_id = st.selectbox("Shipment", shipment_ids, format_func=labels.get)

    shipment = data[data["shipment_id"] == selected_id].copy()
    if shipment.empty:
        st.error("The selected shipment has no candidate routes.")
        st.stop()

    st.divider()
    st.markdown("**Hard constraints**")
    max_budget = st.number_input(
        "Maximum budget (USD)", min_value=0.0, value=0.0, step=500.0,
        help="Set to 0 to disable the budget constraint.",
    )
    max_days = st.number_input(
        "Maximum expected transit (days)", min_value=0.0, value=0.0, step=1.0,
        help="Set to 0 to disable the deadline constraint. A deadline also enables on-time simulation metrics.",
    )
    budget = max_budget if max_budget > 0 else None
    deadline = max_days if max_days > 0 else None

    st.divider()
    st.markdown("**Preference profile**")
    profile = st.selectbox("Priority", PROFILE_ORDER + ["Custom"], index=0)

    if profile == "Custom":
        c1, c2, c3 = st.columns(3)
        raw_weights = {
            "cost": c1.slider("Cost", 0, 100, 40),
            "time": c2.slider("Time", 0, 100, 35),
            "risk": c3.slider("Risk", 0, 100, 25),
        }
    else:
        raw_weights = PREFERENCE_PROFILES[profile]

    try:
        weights = normalize_weights(raw_weights)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.caption(
        f"Normalized · cost {weights['cost']:.0%} · time {weights['time']:.0%} "
        f"· risk {weights['risk']:.0%}"
    )

    st.divider()
    st.markdown("**Monte Carlo simulation**")
    n_simulations = st.select_slider(
        "Trials per route", options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.caption("Bernoulli delay event; bounded triangular delay days.")

try:
    ranked = rank_routes(shipment, weights, budget, deadline)
except ValueError as exc:
    st.error(f"Route evaluation failed: {exc}")
    st.stop()

feasible = ranked[ranked["feasible"]].sort_values("rank")
recommended = feasible.iloc[0] if not feasible.empty else None
recommended_id = str(recommended["route_id"]) if recommended is not None else None
origin = shipment.iloc[0]["origin"]
destination = shipment.iloc[0]["destination"]

# --- Header and executive overview ----------------------------------------

st.markdown('<div class="sr-kicker">Logistics decision support</div>', unsafe_allow_html=True)
st.title("SmartRoute AI")
st.caption(
    "Multi-objective shipment routing with deterministic ranking and Monte Carlo reliability evidence."
)

overview = add_derived_metrics(data)
avg_cost = overview.groupby("shipment_id")["cost_usd"].min().mean()
avg_transit = overview["expected_transit_days"].mean()
avg_risk = overview["risk_score"].mean()

st.markdown("#### Network overview")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Shipments", f"{data['shipment_id'].nunique():,}")
k2.metric("Route alternatives", f"{len(overview):,}")
k3.metric("Avg shipment cost", f"${avg_cost:,.0f}", help="Average lowest-cost option per shipment.")
k4.metric("Avg expected transit", f"{avg_transit:.1f} d", help="Mean across all route options.")
k5.metric("Avg deterministic risk", f"{avg_risk:.1f}/100", help="Mean across all route options.")
k6.metric(
    "Feasible routes", f"{len(feasible)} / {len(ranked)}",
    help="Feasible options for the selected shipment under the current hard constraints.",
)

st.markdown(f"### {origin} → {destination}")
st.caption(
    f"Shipment **{selected_id}** · {len(shipment)} candidate routes · "
    f"{len(feasible)} feasible · {profile} priorities"
)

# --- Recommendation --------------------------------------------------------

if recommended is not None:
    with st.container(border=True):
        r1, r2, r3, r4, r5 = st.columns([2.2, 1.1, 1.1, 1.1, 1.1])
        with r1:
            st.markdown(f"#### {recommended['route_name']}")
            st.caption(str(recommended["legs"]))
            st.markdown(
                badge("Recommended", "good")
                + badge(
                    "Pareto-optimal" if bool(recommended["pareto_optimal"]) else "Dominated",
                    "good" if bool(recommended["pareto_optimal"]) else "warn",
                )
                + badge("Feasible", "good"),
                unsafe_allow_html=True,
            )
        r2.metric("Cost", f"${recommended['cost_usd']:,.0f}")
        r3.metric("Expected transit", f"{recommended['expected_transit_days']:.1f} d")
        r4.metric("Risk score", f"{recommended['risk_score']:.1f}/100")
        r5.metric("Weighted score", f"{recommended['score']:.3f}", help="Lower is better.")
        st.success(explain_recommendation(ranked))
else:
    st.error(explain_recommendation(ranked))
    st.info(
        "No route is eligible for recommendation. Infeasible alternatives remain "
        "visible below so the planner can review the budget and deadline."
    )

tab_compare, tab_reliability, tab_sensitivity, tab_method = st.tabs(
    ["Route comparison", "Reliability analysis", "Sensitivity", "Methodology & assumptions"]
)

# --- Route comparison ------------------------------------------------------

with tab_compare:
    display = pd.DataFrame({
        "Route": ranked["route_name"],
        "Cost (USD)": ranked["cost_usd"],
        "Expected transit (days)": ranked["expected_transit_days"],
        "Risk (0–100)": ranked["risk_score"],
        "Score": ranked["score"].map(score_text),
        "Rank": ranked["rank"].map(lambda value: "—" if pd.isna(value) else str(int(value))),
        "Pareto": ranked["pareto_optimal"].map({True: "Yes", False: "No"}),
        "Feasible": ranked["feasible"].map({True: "Yes", False: "No"}),
        "Constraint status": ranked["recommendation_status"],
    })

    def highlight_route(row: pd.Series) -> list[str]:
        if recommended is not None and row["Route"] == recommended["route_name"]:
            return ["background-color: #ecfdf5; color: #065f46; font-weight: 600"] * len(row)
        if row["Feasible"] == "No":
            return ["color: #94a3b8"] * len(row)
        return [""] * len(row)

    styled = (
        display.style
        .apply(highlight_route, axis=1)
        .format({
            "Cost (USD)": "${:,.0f}",
            "Expected transit (days)": "{:.1f}",
            "Risk (0–100)": "{:.1f}",
        }, na_rep="—")
    )
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={
            "Route": st.column_config.TextColumn(width="medium"),
            "Constraint status": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "Infeasible routes show no optimization score or rank and are never recommended. "
        "The highlighted row is the deterministic recommendation."
    )

    st.plotly_chart(
        cost_time_chart(ranked, recommended_id), width="stretch",
        config={"displayModeBar": False},
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            risk_chart(ranked, recommended_id), width="stretch",
            config={"displayModeBar": False},
        )
    with chart_right:
        if feasible.empty:
            st.info("Optimization ranking is unavailable because no route is feasible.")
        else:
            st.plotly_chart(
                ranking_chart(ranked, recommended_id), width="stretch",
                config={"displayModeBar": False},
            )

    st.download_button(
        "Download route evaluation (CSV)",
        ranked.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_id}_route_evaluation.csv",
        mime="text/csv",
    )

# --- Reliability -----------------------------------------------------------

with tab_reliability:
    st.markdown("#### Simulated delivery reliability")
    st.caption(
        "Monte Carlo evidence is separate from the deterministic score. It informs "
        "uncertainty but does not replace the weighted recommendation."
    )

    try:
        simulation = cached_simulation(shipment, n_simulations, int(seed), deadline)
    except (ValueError, TypeError) as exc:
        simulation = None
        st.error(f"Simulation could not run with the current settings: {exc}")

    if simulation is not None and simulation.summary.empty:
        st.info("The simulation returned no results for this shipment.")

    if simulation is not None and not simulation.summary.empty:
        summary = simulation.summary.copy()
        summary_by_id = summary.set_index("route_id")
        feasible_ids = list(feasible["route_id"].astype(str))
        chart_ids = feasible_ids if feasible_ids else list(shipment["route_id"].astype(str))

        if recommended_id and recommended_id in summary_by_id.index:
            reference = summary_by_id.loc[recommended_id]
            reference_label = str(reference["route_name"])
        else:
            reference = summary.sort_values("mean_transit_days").iloc[0]
            reference_label = str(reference["route_name"])
            st.warning(
                "No feasible route exists. Reliability is shown for reference only; "
                "the reference route is not a recommendation."
            )

        st.caption(
            f"Reference: **{reference_label}** · {simulation.simulation_count:,} trials per route "
            f"· seed {simulation.seed} · "
            + ("deadline active" if deadline is not None else "no deadline set")
        )
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Mean transit", f"{reference['mean_transit_days']:.1f} d")
        m2.metric("P50", f"{reference['median_transit_days']:.1f} d")
        m3.metric("P90", f"{reference['p90_transit_days']:.1f} d")
        m4.metric("P95", f"{reference['p95_transit_days']:.1f} d")
        m5.metric(
            "On-time probability",
            percent_text(reference["on_time_probability"]),
            help="Set a maximum expected transit deadline to calculate this.",
        )
        m6.metric("Simulations", f"{int(reference['simulation_count']):,}")

        st.plotly_chart(
            distribution_chart(
                simulation.samples, summary, chart_ids, recommended_id, deadline,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

        if deadline is None:
            st.info(
                "On-time probability is unavailable because no deadline is set. "
                "Enter a maximum expected transit time in the sidebar."
            )
        elif feasible.empty:
            st.info("On-time probability is not ranked because no route is feasible.")
        else:
            st.plotly_chart(
                on_time_chart(summary, feasible_ids, recommended_id),
                width="stretch",
                config={"displayModeBar": False},
            )

        with st.expander("Simulation summary by route"):
            summary_display = summary.rename(columns={
                "route_name": "Route",
                "mean_transit_days": "Mean transit (days)",
                "median_transit_days": "P50 (days)",
                "p90_transit_days": "P90 (days)",
                "p95_transit_days": "P95 (days)",
                "on_time_probability": "On-time probability",
                "mean_cost": "Mean cost (USD)",
                "probability_over_budget": "P(over budget)",
                "simulation_count": "Simulations",
            })
            st.dataframe(
                summary_display[[
                    "Route", "Mean transit (days)", "P50 (days)", "P90 (days)",
                    "P95 (days)", "On-time probability", "Mean cost (USD)",
                    "Simulations",
                ]].style.format({
                    "Mean transit (days)": "{:.1f}",
                    "P50 (days)": "{:.1f}",
                    "P90 (days)": "{:.1f}",
                    "P95 (days)": "{:.1f}",
                    "On-time probability": "{:.1%}",
                    "Mean cost (USD)": "${:,.0f}",
                    "Simulations": "{:,.0f}",
                }, na_rep="—"),
                width="stretch", hide_index=True,
            )

        st.caption(
            "Assumption: delay is a Bernoulli event with the route's delay probability; "
            "conditional delay days follow a triangular distribution bounded at "
            "0.5×, 1.0× and 1.5× the stated delay impact. Cost is deterministic by default. "
            "These are modelled outcomes, not carrier performance guarantees."
        )
        st.download_button(
            "Download simulation summary (CSV)",
            summary.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_id}_simulation_summary.csv",
            mime="text/csv",
        )

# --- Sensitivity -----------------------------------------------------------

with tab_sensitivity:
    sensitivity = profile_sensitivity(shipment, budget, deadline)

    sens_display = pd.DataFrame({
        "Profile": sensitivity["profile"],
        "Recommended route": sensitivity["recommended_route"],
        "Cost (USD)": sensitivity["cost_usd"],
        "Expected transit (days)": sensitivity["expected_transit_days"],
        "Risk (0–100)": sensitivity["risk_score"],
        "Score": sensitivity["score"],
        "Pareto": sensitivity["pareto_optimal"].map({True: "Yes", False: "No"}),
    })

    def highlight_profile(row: pd.Series) -> list[str]:
        if row["Profile"] == profile:
            return ["background-color: #f0fdfa; color: #115e59; font-weight: 600"] * len(row)
        return [""] * len(row)

    st.dataframe(
        sens_display.style.apply(highlight_profile, axis=1).format({
            "Cost (USD)": "${:,.0f}",
            "Expected transit (days)": "{:.1f}",
            "Risk (0–100)": "{:.1f}",
            "Score": "{:.3f}",
        }, na_rep="—"),
        width="stretch", hide_index=True,
    )
    st.caption("Hard budget and deadline constraints remain active in every profile.")

    chart = sensitivity_chart(sensitivity)
    if chart is None:
        st.info("Sensitivity comparison is unavailable because no profile has a feasible route.")
    else:
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

    unique_routes = sensitivity.loc[
        sensitivity["recommended_route"] != "No feasible route", "recommended_route"
    ].nunique()
    if unique_routes == 0:
        st.error("No route is feasible under the current hard constraints in any profile.")
    elif unique_routes == 1:
        only_route = sensitivity.loc[
            sensitivity["recommended_route"] != "No feasible route", "recommended_route"
        ].iloc[0]
        st.success(
            f"The recommendation is stable across all four profiles: **{only_route}**."
        )
    else:
        st.warning(
            f"The recommendation changes across **{unique_routes}** routes as stakeholder "
            "priorities shift. Review the trade-offs before committing."
        )

# --- Methodology -----------------------------------------------------------

with tab_method:
    st.markdown("#### Methodology and assumptions")
    with st.expander("Deterministic multi-objective optimization", expanded=True):
        st.markdown(
            """
            **Expected transit time**

            `scheduled transit days + delay probability × delay impact days`

            **Composite operational risk (0–100)**

            `0.45 × delay probability × 100 + 0.30 × congestion risk + 0.25 × geopolitical risk`

            **Min-max normalization**

            `(value − minimum) / (maximum − minimum)` across the selected shipment's
            feasible routes. If every route has the same value, the normalized component is 0.

            **Weighted score**

            `cost weight × normalized cost + time weight × normalized expected time + risk weight × normalized risk`

            Lower is better. Weights are validated and normalized to sum to 1. The
            deterministic weighted score remains the primary recommendation mechanism.
            """
        )

    with st.expander("Hard constraints and Pareto optimality"):
        st.markdown(
            """
            Budget and expected transit limits are **hard constraints**. A route that
            violates either limit is infeasible, receives no score or rank, cannot be
            Pareto-optimal, and is never recommended.

            A feasible route is **Pareto-optimal** when no other feasible route is at
            least as good on cost, expected transit time, and risk and strictly better
            on at least one. Pareto status and weighted ranking answer different
            questions and are reported separately.
            """
        )

    with st.expander("Monte Carlo reliability model"):
        st.markdown(
            """
            For every route and trial:

            1. Start with the scheduled transit time.
            2. Draw a Bernoulli delay event using `delay_probability`.
            3. If delayed, draw delay days from a **triangular distribution** with
               lower bound `0.5 × delay_impact_days`, mode `delay_impact_days`, and
               upper bound `1.5 × delay_impact_days`.
            4. Add the delay to the scheduled time and record the outcome.

            The triangular distribution is a modelling assumption, not a measured
            distribution. Zero impact produces no added days. On-time probability is
            the share of trials with simulated delivery time at or below the selected
            deadline; it is unavailable when no deadline is set.

            Simulation uses independent, reproducible per-route random streams from
            the configured seed. Cost is deterministic by default. Optional cost
            uncertainty is available only through the engine API and is not enabled
            in this dashboard.
            """
        )

    with st.expander("Data and model limitations"):
        st.markdown(
            """
            - All dataset values are synthetic and illustrative, not live market data.
            - Delay probability, congestion, and geopolitical indicators are assumed inputs.
            - The risk score is a decision-support indicator, not a calibrated probability.
            - Monte Carlo metrics are only as reliable as the stated input assumptions.
            - Normalization is relative to one shipment; scores are not comparable across shipments.
            - The dashboard assumes candidate routes already exist and does not model vessel
              schedules, live weather, customs, emissions, or carrier guarantees.
            """
        )
