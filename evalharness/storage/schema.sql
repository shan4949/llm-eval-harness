-- EvalHarness schema
-- Write DDL by hand so every decision is defensible in an interview.
-- Mount this file as docker-entrypoint-initdb.d/schema.sql for auto-init.

CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id     SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    git_commit_sha  TEXT,
    model_name      TEXT NOT NULL,          -- e.g. 'claude-sonnet-4-5', 'gpt-4o-mini'
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- One execution of the full eval suite against a pipeline version.
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id          SERIAL PRIMARY KEY,
    pipeline_id     INTEGER REFERENCES pipelines(pipeline_id),
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ,
    eval_set_name   TEXT NOT NULL,
    is_baseline     BOOLEAN DEFAULT FALSE
);

-- One test case's raw execution result within a run.
-- Kept separate from scores so we can re-score with a new judge without re-running the SUT.
CREATE TABLE IF NOT EXISTS eval_results (
    result_id           SERIAL PRIMARY KEY,
    run_id              INTEGER REFERENCES eval_runs(run_id),
    test_case_id        TEXT NOT NULL,
    question            TEXT NOT NULL,
    retrieved_context   TEXT,
    model_response      TEXT NOT NULL,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    latency_ms          INTEGER,
    retrieval_sim_p50   NUMERIC(5,4),       -- median cosine sim of retrieved chunks (drift monitoring)
    estimated_cost_usd  NUMERIC(10,6)
);

-- One judge/deterministic score per metric per result (many-to-one).
-- Separate table: adding a metric = insert rows, not ALTER TABLE.
CREATE TABLE IF NOT EXISTS eval_scores (
    score_id         SERIAL PRIMARY KEY,
    result_id        INTEGER REFERENCES eval_results(result_id),
    metric_name      TEXT NOT NULL,         -- 'faithfulness', 'relevance', 'correctness', 'latency_ok'
    score            NUMERIC(4,2),          -- normalized 0-1
    judge_reasoning  TEXT,                  -- always store reasoning, not just the number
    passed           BOOLEAN
);

-- Indexes for the queries we'll actually run
CREATE INDEX IF NOT EXISTS idx_eval_results_run_id ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_scores_result_id ON eval_scores(result_id);
CREATE INDEX IF NOT EXISTS idx_eval_scores_metric ON eval_scores(metric_name);
CREATE INDEX IF NOT EXISTS idx_eval_runs_pipeline ON eval_runs(pipeline_id);
