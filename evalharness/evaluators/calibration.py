"""Judge calibration — Cohen's kappa between LLM judge and human labels."""
from pathlib import Path
from typing import NamedTuple

import yaml
from scipy.stats import chi2_contingency


class KappaResult(NamedTuple):
    kappa: float
    agreement_pct: float
    n: int
    interpretation: str


def _interpret(kappa: float) -> str:
    if kappa < 0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight agreement"
    if kappa < 0.40:
        return "fair agreement"
    if kappa < 0.60:
        return "moderate agreement"
    if kappa < 0.80:
        return "substantial agreement"
    return "almost perfect agreement"


def cohen_kappa(judge_labels: list[bool], human_labels: list[bool]) -> KappaResult:
    """Compute Cohen's kappa between two binary label sequences."""
    assert len(judge_labels) == len(human_labels), "Label lists must be same length"
    n = len(judge_labels)

    # contingency counts
    tp = sum(j and h for j, h in zip(judge_labels, human_labels))
    fp = sum(j and not h for j, h in zip(judge_labels, human_labels))
    fn = sum(not j and h for j, h in zip(judge_labels, human_labels))
    tn = sum(not j and not h for j, h in zip(judge_labels, human_labels))

    p_o = (tp + tn) / n  # observed agreement

    p_pos_judge = (tp + fp) / n
    p_pos_human = (tp + fn) / n
    p_neg_judge = (fn + tn) / n
    p_neg_human = (fp + tn) / n
    p_e = p_pos_judge * p_pos_human + p_neg_judge * p_neg_human  # expected agreement

    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

    return KappaResult(
        kappa=round(kappa, 4),
        agreement_pct=round(p_o * 100, 1),
        n=n,
        interpretation=_interpret(kappa),
    )


def load_human_labels(path: Path) -> dict[str, bool]:
    """
    Load human labels from a YAML file.
    Format:
      - id: fomc_2022_03_hike
        passed: true
    """
    data = yaml.safe_load(path.read_text())
    return {item["id"]: bool(item["passed"]) for item in data}


def calibrate_from_results(
    results: list[dict],  # [{id, judge_passed, human_passed}, ...]
) -> KappaResult:
    judge_labels = [r["judge_passed"] for r in results]
    human_labels = [r["human_passed"] for r in results]
    return cohen_kappa(judge_labels, human_labels)
