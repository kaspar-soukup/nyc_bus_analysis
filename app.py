import os

import pandas as pd
import streamlit as st

from run_did_model import run_did_pipeline

st.set_page_config(page_title="NYC Bus Peak Speeds", page_icon="🚌", layout="wide")
st.title("🚌 NYC Bus Peak-Speed Analysis")
st.caption("Difference-in-differences view of congestion pricing impacts")


@st.cache_data(show_spinner=False)
def load_pipeline_results() -> dict:
    return run_did_pipeline(app_token=os.getenv("APP_TOKEN"))


def format_mph(value: float | None) -> str:
    return f"{value:.2f} mph" if value is not None else "N/A"


def build_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    focus = df[df["group"].isin(["CBD routes", "Manhattan routes"])]
    pivot = focus.pivot(index="month", columns="group", values="avg_speed_mph")
    pivot = pivot.sort_index()
    pivot.index.name = "Month"
    return pivot


def render_dashboard(results: dict) -> None:
    summary = results["model_summary"]
    latest = results["latest_month"]
    fun = results["fun_stats"]
    timeseries = build_timeseries(results["group_monthly_avg"])

    with st.container():
        st.subheader("Difference-in-Differences Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("DiD coefficient", format_mph(summary["coefficient"]))
        c2.metric("p-value", f"{summary['p_value']:.4f}")
        c3.metric("Observations", summary["n_obs"])

    st.divider()

    with st.container():
        st.subheader("Latest Month Overview")
        month_label = latest["month"].strftime("%B %Y") if latest["month"] is not None else "N/A"
        st.write(f"Most recent data month: **{month_label}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("CBD peak speed", format_mph(latest["cbd_avg"]))
        c2.metric("Manhattan routes", format_mph(latest["manhattan_routes_avg"]))
        c3.metric("All Manhattan", format_mph(latest["manhattan_all_avg"]))

    st.divider()

    if not timeseries.empty:
        st.subheader("Monthly Peak Speeds")
        st.line_chart(timeseries)
    else:
        st.info("Not enough data to draw the monthly trend.")

    st.divider()

    st.subheader("Fun Stats")
    col1, col2 = st.columns(2)
    if fun["fastest_route"]:
        col1.metric(
            "Fastest route",
            f"{fun['fastest_route']} ({format_mph(fun['fastest_speed'])})",
            help=f"Group: {fun['fastest_group']}"
        )
    else:
        col1.write("No fastest route identified.")

    if fun["slowest_route"]:
        col2.metric(
            "Slowest route",
            f"{fun['slowest_route']} ({format_mph(fun['slowest_speed'])})",
            help=f"Group: {fun['slowest_group']}"
        )
    else:
        col2.write("No slowest route identified.")

    st.divider()

    st.subheader("Pre vs Post Changes")
    st.dataframe(results["pre_post_summary"], use_container_width=True)

    with st.expander("View model summary table"):
        model = results.get("model")
        if model is None:
            st.info("Model output not available.")
        else:
            coeffs = pd.DataFrame({
                "feature": model.params.index,
                "coefficient": model.params.values,
                "p_value": model.pvalues.values,
            }).reset_index(drop=True)
            st.dataframe(coeffs, use_container_width=True)

    with st.expander("Raw speed change panel"):
        st.dataframe(results["speed_changes"], use_container_width=True)


results = load_pipeline_results()
if not results:
    st.error("Unable to load pipeline results. Check that the raw data files are available.")
else:
    render_dashboard(results)
