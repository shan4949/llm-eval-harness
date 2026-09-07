"""LLM-as-judge evaluators — call Claude to score faithfulness, relevance, correctness."""
import json
import statistics
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from evalharness.config import settings
from evalharness.evaluators.deterministic import EvalScore

load_dotenv()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


_JUDGE_SYSTEM = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Score the given answer on the requested metric. Respond ONLY with valid JSON:
{"score": <1-5>, "reasoning": "<one sentence>"}

Scoring scale: 1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent."""


def _call_judge(prompt: str) -> dict:
    """Call the judge model once and parse JSON response."""
    resp = _get_client().messages.create(
        model=settings.judge_model,
        max_tokens=256,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _majority_score(prompts: list[str]) -> tuple[float, str]:
    """Call judge N times (self-consistency) and average scores."""
    results = [_call_judge(p) for p in prompts]
    avg_score = statistics.mean(r["score"] for r in results)
    reasoning = results[0]["reasoning"]  # take first reasoning as representative
    return avg_score, reasoning


def faithfulness(question: str, context: str, answer: str) -> EvalScore:
    """Is the answer supported by the retrieved context? (hallucination check)"""
    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        "Score how well the answer is supported by the context. "
        "5=fully grounded in context, 1=contradicts or ignores context."
    )
    prompts = [prompt] * settings.judge_self_consistency_n
    score, reasoning = _majority_score(prompts)
    return EvalScore(
        metric_name="faithfulness",
        score=round(score / 5, 2),
        passed=score >= 3,
        reasoning=reasoning,
    )


def relevance(question: str, answer: str) -> EvalScore:
    """Does the answer actually address the question?"""
    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Score how relevant and on-topic the answer is to the question. "
        "5=directly answers the question, 1=completely off-topic."
    )
    prompts = [prompt] * settings.judge_self_consistency_n
    score, reasoning = _majority_score(prompts)
    return EvalScore(
        metric_name="relevance",
        score=round(score / 5, 2),
        passed=score >= 3,
        reasoning=reasoning,
    )


def correctness(question: str, answer: str, expected: str) -> EvalScore:
    """Is the answer factually correct compared to the expected answer?"""
    prompt = (
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"Actual answer: {answer}\n\n"
        "Score factual correctness: does the actual answer convey the same key facts as the expected answer? "
        "5=fully correct, 1=completely wrong. "
        "If the actual answer says 'I don't know' but the expected answer has content, score 1."
    )
    prompts = [prompt] * settings.judge_self_consistency_n
    score, reasoning = _majority_score(prompts)
    return EvalScore(
        metric_name="correctness",
        score=round(score / 5, 2),
        passed=score >= 3,
        reasoning=reasoning,
    )


def run_all(
    question: str,
    context: str,
    answer: str,
    expected: str,
) -> list[EvalScore]:
    return [
        faithfulness(question, context, answer),
        relevance(question, answer),
        correctness(question, answer, expected),
    ]
