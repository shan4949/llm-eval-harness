from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TIMESTAMP

from evalharness.storage.db import Base


class Pipeline(Base):
    __tablename__ = "pipelines"

    pipeline_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    git_commit_sha: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))

    runs: Mapped[list["EvalRun"]] = relationship(back_populates="pipeline")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pipeline_id: Mapped[int] = mapped_column(ForeignKey("pipelines.pipeline_id"))
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    eval_set_name: Mapped[str] = mapped_column(Text, nullable=False)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)

    pipeline: Mapped["Pipeline"] = relationship(back_populates="runs")
    results: Mapped[list["EvalResult"]] = relationship(back_populates="run")


class EvalResult(Base):
    __tablename__ = "eval_results"

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.run_id"))
    test_case_id: Mapped[str] = mapped_column(Text, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_context: Mapped[str | None] = mapped_column(Text)
    model_response: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    retrieval_sim_p50: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    run: Mapped["EvalRun"] = relationship(back_populates="results")
    scores: Mapped[list["EvalScore"]] = relationship(back_populates="result")


class EvalScore(Base):
    __tablename__ = "eval_scores"

    score_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("eval_results.result_id"))
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    judge_reasoning: Mapped[str | None] = mapped_column(Text)
    passed: Mapped[bool | None] = mapped_column(Boolean)

    result: Mapped["EvalResult"] = relationship(back_populates="scores")
