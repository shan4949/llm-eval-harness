"""Streamlit dashboard — eval run history, scores, cost, latency, retrieval drift."""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

st.set_page_config(page_title="EvalHarness Dashboard", layout="wide")

DB_URL = os.environ.get("DATABASE_URL", "postgresql://eval:eval@localhost:5432/evalharness")
engine = create_engine(DB_URL, pool_pre_ping=True)


@st.cache_data(ttl=30)
def load_runs() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT r.run_id, r.started_at, r.finished_at, r.eval_set_name, r.is_baseline,
                       p.name as pipeline, p.model_name, p.git_commit_sha
                FROM eval_runs r
                JOIN pipelines p ON p.pipeline_id = r.pipeline_id
                ORDER BY r.run_id DESC
                LIMIT 50
            """),
            conn,
        )


@st.cache_data(ttl=30)
def load_scores(run_id: int) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT er.test_case_id, er.latency_ms, er.estimated_cost_usd,
                       er.retrieval_sim_p50, es.metric_name, es.score, es.passed, es.judge_reasoning
                FROM eval_results er
                JOIN eval_scores es ON es.result_id = er.result_id
                WHERE er.run_id = :run_id
            """),
            conn,
            params={"run_id": run_id},
        )


@st.cache_data(ttl=30)
def load_trend() -> pd.DataFrame:
    """Pass rates and cost per run across all runs."""
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT er.run_id, r.started_at, es.metric_name,
                       AVG(CASE WHEN es.passed THEN 1.0 ELSE 0.0 END) AS pass_rate,
                       AVG(er.estimated_cost_usd::float) AS avg_cost,
                       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY er.latency_ms) AS p95_latency,
                       AVG(er.retrieval_sim_p50::float) AS avg_retrieval_sim
                FROM eval_results er
                JOIN eval_runs r ON r.run_id = er.run_id
                JOIN eval_scores es ON es.result_id = er.result_id
                WHERE es.metric_name = 'keyword_match'
                GROUP BY er.run_id, r.started_at, es.metric_name
                ORDER BY er.run_id
            """),
            conn,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("EvalHarness")
st.sidebar.caption("FOMC RAG Pipeline — Eval Dashboard")

try:
    runs_df = load_runs()
except Exception as e:
    st.error(f"Cannot connect to database: {e}")
    st.stop()

if runs_df.empty:
    st.info("No eval runs yet. Run `evalharness run` to get started.")
    st.stop()

run_options = {
    f"Run {row.run_id} — {row.started_at.date()} ({row.eval_set_name})"
    + (" [baseline]" if row.is_baseline else ""): row.run_id
    for row in runs_df.itertuples()
}
selected_label = st.sidebar.selectbox("Select run", list(run_options.keys()))
selected_run_id = run_options[selected_label]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("EvalHarness Dashboard")
selected_run = runs_df[runs_df.run_id == selected_run_id].iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Run ID", selected_run_id)
col2.metric("Model", selected_run["model_name"])
col3.metric("Eval set", selected_run["eval_set_name"])
col4.metric("Git SHA", selected_run["git_commit_sha"] or "–")

# ── Score summary for selected run ───────────────────────────────────────────
scores_df = load_scores(selected_run_id)

if scores_df.empty:
    st.warning("No scores found for this run.")
else:
    st.subheader("Score summary")

    pivot = scores_df.groupby("metric_name")["passed"].mean().reset_index()
    pivot.columns = ["Metric", "Pass rate"]
    pivot["Pass rate"] = (pivot["Pass rate"] * 100).round(1)

    fig_bar = px.bar(
        pivot, x="Metric", y="Pass rate", text="Pass rate",
        color="Pass rate", color_continuous_scale="RdYlGn", range_color=[0, 100],
        title="Pass rate by metric (%)",
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(coloraxis_showscale=False, yaxis_range=[0, 110])
    st.plotly_chart(fig_bar, use_container_width=True)

    # Per-test-case table
    st.subheader("Per-test-case results")
    kw_scores = scores_df[scores_df.metric_name == "keyword_match"][
        ["test_case_id", "latency_ms", "estimated_cost_usd", "retrieval_sim_p50", "passed"]
    ].rename(columns={"passed": "keyword_match"})
    st.dataframe(kw_scores, use_container_width=True)

# ── Trend charts ──────────────────────────────────────────────────────────────
st.subheader("Trends across all runs")
trend_df = load_trend()

if not trend_df.empty:
    col_a, col_b = st.columns(2)

    with col_a:
        fig_pass = px.line(
            trend_df, x="run_id", y="pass_rate", markers=True,
            title="Keyword match pass rate over time",
            labels={"pass_rate": "Pass rate", "run_id": "Run"},
        )
        fig_pass.update_yaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(fig_pass, use_container_width=True)

    with col_b:
        fig_lat = px.line(
            trend_df, x="run_id", y="p95_latency", markers=True,
            title="P95 latency over time (ms)",
            labels={"p95_latency": "P95 latency (ms)", "run_id": "Run"},
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig_cost = px.line(
            trend_df, x="run_id", y="avg_cost", markers=True,
            title="Avg cost per query (USD)",
            labels={"avg_cost": "USD", "run_id": "Run"},
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_d:
        fig_drift = px.line(
            trend_df, x="run_id", y="avg_retrieval_sim", markers=True,
            title="Retrieval similarity (drift monitor)",
            labels={"avg_retrieval_sim": "Avg retrieval sim p50", "run_id": "Run"},
        )
        fig_drift.add_hline(y=0.3, line_dash="dash", line_color="orange",
                            annotation_text="Drift warning threshold")
        st.plotly_chart(fig_drift, use_container_width=True)

# ── Raw run table ─────────────────────────────────────────────────────────────
with st.expander("All runs"):
    st.dataframe(runs_df, use_container_width=True)
