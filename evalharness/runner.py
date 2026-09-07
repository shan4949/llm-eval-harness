"""Core eval runner — loads test cases, calls SUT, scores, stores to DB."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

from evalharness.config import settings
from evalharness.evaluators import deterministic, llm_judge
from evalharness.storage.db import get_session
from evalharness.storage.models import EvalResult, EvalRun, EvalScore, Pipeline

load_dotenv()

SUT_URL = "http://localhost:8000/query"


def _load_eval_set(name: str) -> list[dict]:
    path = Path("eval_sets") / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    return yaml.safe_load(path.read_text())


def _get_git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def _get_or_create_pipeline(session, model_name: str) -> Pipeline:
    sha = _get_git_sha()
    pipeline = Pipeline(
        name="fomc-rag",
        git_commit_sha=sha,
        model_name=model_name,
        created_at=datetime.now(timezone.utc),
    )
    session.add(pipeline)
    session.flush()
    return pipeline


def run_eval(eval_set_name: str, is_baseline: bool = False, use_llm_judge: bool = True) -> int:
    """
    Run the full eval suite. Returns the run_id.
    Requires the SUT FastAPI server to be running at SUT_URL.
    """
    test_cases = _load_eval_set(eval_set_name)
    client = httpx.Client(timeout=30.0)

    with get_session() as session:
        pipeline = _get_or_create_pipeline(session, settings.sut_model)

        run = EvalRun(
            pipeline_id=pipeline.pipeline_id,
            started_at=datetime.now(timezone.utc),
            eval_set_name=eval_set_name,
            is_baseline=is_baseline,
        )
        session.add(run)
        session.flush()

        for tc in test_cases:
            tc_id = tc["id"]
            question = tc["question"]
            expected = tc.get("expected", "")
            keywords = tc.get("keywords", [])
            expected_idk = tc.get("expected_idk", False)

            # Call SUT
            try:
                resp = client.post(SUT_URL, json={"question": question})
                resp.raise_for_status()
                sut = resp.json()
            except Exception as e:
                # Store failure as a result with empty answer
                sut = {
                    "answer": f"[SUT ERROR: {e}]",
                    "model": settings.sut_model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "estimated_cost_usd": 0.0,
                    "retrieval_sim_p50": 0.0,
                    "retrieved_context": "",
                }

            result = EvalResult(
                run_id=run.run_id,
                test_case_id=tc_id,
                question=question,
                retrieved_context=sut.get("retrieved_context", ""),
                model_response=sut["answer"],
                input_tokens=sut.get("input_tokens"),
                output_tokens=sut.get("output_tokens"),
                latency_ms=sut.get("latency_ms"),
                retrieval_sim_p50=sut.get("retrieval_sim_p50"),
                estimated_cost_usd=sut.get("estimated_cost_usd"),
            )
            session.add(result)
            session.flush()

            # Deterministic scores
            det_scores = deterministic.run_all(
                answer=sut["answer"],
                keywords=keywords,
                expected_idk=expected_idk,
                latency_ms=sut.get("latency_ms", 0),
            )
            for s in det_scores:
                session.add(EvalScore(
                    result_id=result.result_id,
                    metric_name=s.metric_name,
                    score=s.score,
                    judge_reasoning=s.reasoning,
                    passed=s.passed,
                ))

            # LLM judge scores (skip for idk cases — no factual content to judge)
            if use_llm_judge and not expected_idk:
                judge_scores = llm_judge.run_all(
                    question=question,
                    context=sut.get("retrieved_context", ""),
                    answer=sut["answer"],
                    expected=expected,
                )
                for s in judge_scores:
                    session.add(EvalScore(
                        result_id=result.result_id,
                        metric_name=s.metric_name,
                        score=s.score,
                        judge_reasoning=s.reasoning,
                        passed=s.passed,
                    ))

        run.finished_at = datetime.now(timezone.utc)
        run_id = run.run_id  # capture before session closes

    return run_id
