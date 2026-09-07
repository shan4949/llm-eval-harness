"""Regression gating — two-proportion z-test and latency/cost comparison."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sqlalchemy import func, select, text

from evalharness.config import settings
from evalharness.storage.db import get_session
from evalharness.storage.models import EvalResult, EvalRun, EvalScore


@dataclass
class RunSummary:
    run_id: int
    n_cases: int
    pass_rate: float           # fraction of keyword_match passed
    latency_p95_ms: float
    cost_per_query_usd: float
    retrieval_sim_median: float


@dataclass
class RegressionReport:
    baseline: RunSummary
    candidate: RunSummary
    pass_rate_delta: float     # candidate - baseline (negative = regression)
    pass_rate_p_value: float
    latency_increase_pct: float
    cost_increase_pct: float
    is_regression: bool
    reasons: list[str]


def _get_run_summary(run_id: int) -> RunSummary:
    with get_session() as session:
        # keyword_match pass rate
        scores = (
            session.execute(
                select(EvalScore.passed)
                .join(EvalResult, EvalScore.result_id == EvalResult.result_id)
                .where(EvalResult.run_id == run_id, EvalScore.metric_name == "keyword_match")
            )
            .scalars()
            .all()
        )
        n = len(scores)
        pass_rate = sum(bool(s) for s in scores) / n if n else 0.0

        # latency p95
        latencies = (
            session.execute(
                select(EvalResult.latency_ms).where(
                    EvalResult.run_id == run_id, EvalResult.latency_ms.isnot(None)
                )
            )
            .scalars()
            .all()
        )
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

        # cost per query
        costs = (
            session.execute(
                select(EvalResult.estimated_cost_usd).where(
                    EvalResult.run_id == run_id, EvalResult.estimated_cost_usd.isnot(None)
                )
            )
            .scalars()
            .all()
        )
        avg_cost = float(np.mean([float(c) for c in costs])) if costs else 0.0

        # retrieval sim median
        sims = (
            session.execute(
                select(EvalResult.retrieval_sim_p50).where(
                    EvalResult.run_id == run_id, EvalResult.retrieval_sim_p50.isnot(None)
                )
            )
            .scalars()
            .all()
        )
        sim_median = float(np.median([float(s) for s in sims])) if sims else 0.0

    return RunSummary(
        run_id=run_id,
        n_cases=n,
        pass_rate=round(pass_rate, 4),
        latency_p95_ms=round(p95, 1),
        cost_per_query_usd=round(avg_cost, 6),
        retrieval_sim_median=round(sim_median, 4),
    )


def compare(baseline_id: int, candidate_id: int) -> RegressionReport:
    base = _get_run_summary(baseline_id)
    cand = _get_run_summary(candidate_id)

    delta = cand.pass_rate - base.pass_rate

    # two-proportion z-test for pass rate (H1: candidate < baseline)
    count_b = int(base.pass_rate * base.n_cases)
    count_c = int(cand.pass_rate * cand.n_cases)
    n_b, n_c = base.n_cases, cand.n_cases
    p_pool = (count_b + count_c) / (n_b + n_c)
    se = (p_pool * (1 - p_pool) * (1 / n_b + 1 / n_c)) ** 0.5
    z = (cand.pass_rate - base.pass_rate) / se if se > 0 else 0.0
    p_value = float(stats.norm.cdf(z))  # one-tailed: P(Z < z)

    latency_increase = (
        (cand.latency_p95_ms - base.latency_p95_ms) / base.latency_p95_ms
        if base.latency_p95_ms > 0 else 0.0
    )
    cost_increase = (
        (cand.cost_per_query_usd - base.cost_per_query_usd) / base.cost_per_query_usd
        if base.cost_per_query_usd > 0 else 0.0
    )

    reasons: list[str] = []
    if delta < -settings.regression_pass_rate_delta and p_value < settings.regression_alpha:
        reasons.append(
            f"Pass rate dropped {abs(delta):.1%} (p={p_value:.3f} < {settings.regression_alpha})"
        )
    if latency_increase > settings.latency_p95_increase_pct:
        reasons.append(f"P95 latency increased {latency_increase:.1%}")
    if cost_increase > settings.cost_increase_pct:
        reasons.append(f"Cost/query increased {cost_increase:.1%}")

    return RegressionReport(
        baseline=base,
        candidate=cand,
        pass_rate_delta=round(delta, 4),
        pass_rate_p_value=round(float(p_value), 4),
        latency_increase_pct=round(latency_increase, 4),
        cost_increase_pct=round(cost_increase, 4),
        is_regression=len(reasons) > 0,
        reasons=reasons,
    )
