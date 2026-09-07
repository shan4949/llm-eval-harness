"""Deterministic evaluators — no LLM calls, instant and free."""
from dataclasses import dataclass


@dataclass
class EvalScore:
    metric_name: str
    score: float          # 0.0 or 1.0 for deterministic
    passed: bool
    reasoning: str


def keyword_match(answer: str, keywords: list[str]) -> EvalScore:
    """All keywords must appear in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    missing = [kw for kw in keywords if kw.lower() not in answer_lower]
    passed = len(missing) == 0
    return EvalScore(
        metric_name="keyword_match",
        score=1.0 if passed else round(1 - len(missing) / len(keywords), 2),
        passed=passed,
        reasoning=f"Missing keywords: {missing}" if missing else "All keywords found",
    )


def idk_check(answer: str, expected_idk: bool) -> EvalScore:
    """Check whether the model correctly said 'I don't know' for out-of-corpus questions."""
    says_idk = "i don't know" in answer.lower() or "i do not know" in answer.lower()
    if expected_idk:
        passed = says_idk
        reasoning = "Correctly declined to answer" if passed else "Should have said I don't know"
    else:
        passed = not says_idk
        reasoning = "Correctly gave an answer" if passed else "Incorrectly said I don't know"
    return EvalScore(
        metric_name="idk_check",
        score=1.0 if passed else 0.0,
        passed=passed,
        reasoning=reasoning,
    )


def latency_check(latency_ms: int, threshold_ms: int = 5000) -> EvalScore:
    """Flag responses that took longer than threshold_ms."""
    passed = latency_ms <= threshold_ms
    return EvalScore(
        metric_name="latency_ok",
        score=1.0 if passed else 0.0,
        passed=passed,
        reasoning=f"{latency_ms}ms {'<=' if passed else '>'} {threshold_ms}ms threshold",
    )


def run_all(
    answer: str,
    keywords: list[str],
    expected_idk: bool,
    latency_ms: int,
) -> list[EvalScore]:
    scores = [
        keyword_match(answer, keywords),
        idk_check(answer, expected_idk),
        latency_check(latency_ms),
    ]
    return scores
