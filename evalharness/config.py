from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql://eval:eval@localhost:5432/evalharness"

    # LLM APIs
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Judge model (default: use a cheaper model for judging)
    judge_model: str = "claude-haiku-4-5-20251001"
    judge_self_consistency_n: int = 3       # call judge N times and take majority

    # SUT model
    sut_model: str = "claude-haiku-4-5-20251001"

    # Regression gate thresholds
    regression_pass_rate_delta: float = 0.05    # flag if pass rate drops > 5pp AND p < alpha
    regression_alpha: float = 0.05              # significance level for z-test
    latency_p95_increase_pct: float = 0.20      # flag if p95 latency rises > 20%
    cost_increase_pct: float = 0.25             # flag if cost/query rises > 25%

    # Embedding drift
    drift_ks_alpha: float = 0.05               # KS test significance level for retrieval drift

    # Model pricing (USD per 1k tokens)
    pricing: dict[str, dict[str, float]] = {
        "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
    }


settings = Settings()
