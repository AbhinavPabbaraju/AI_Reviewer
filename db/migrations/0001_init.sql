-- Argus initial schema.
-- Forward-only. Applied by Alembic; kept as plain SQL because the interesting
-- parts here are constraints and index strategy, and those are unreadable
-- through an ORM migration DSL.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- title similarity during dedup

-- --------------------------------------------------------------------------
-- Enumerated types
-- Mirrors app.domain.contracts. Native enums (rather than CHECK-constrained
-- text) so that adding a value is a deliberate, reviewed migration -- the
-- contract is the spine of the system and should be hard to change casually.
-- --------------------------------------------------------------------------

CREATE TYPE severity AS ENUM ('info', 'low', 'medium', 'high', 'critical');

CREATE TYPE finding_category AS ENUM (
    'correctness', 'security', 'performance', 'concurrency',
    'api_contract', 'error_handling', 'testing', 'maintainability', 'style'
);

CREATE TYPE finding_source AS ENUM ('llm', 'semgrep', 'ruff', 'eslint', 'tsc');

CREATE TYPE verification_status AS ENUM ('pending', 'verified', 'demoted', 'rejected');

CREATE TYPE run_status AS ENUM (
    'queued', 'indexing', 'retrieving', 'analyzing', 'reviewing',
    'verifying', 'publishing', 'succeeded', 'failed', 'cancelled'
);

CREATE TYPE symbol_kind AS ENUM (
    'function', 'method', 'class', 'interface', 'type_alias',
    'variable', 'module', 'enum'
);

CREATE TYPE edge_kind AS ENUM (
    'defines', 'calls', 'imports', 'inherits', 'references', 'tests'
);

CREATE TYPE snapshot_status AS ENUM ('pending', 'running', 'ready', 'failed');

-- --------------------------------------------------------------------------
-- Tenancy
-- --------------------------------------------------------------------------

CREATE TABLE installations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    github_installation_id BIGINT NOT NULL UNIQUE,
    account_login       TEXT NOT NULL,
    account_type        TEXT NOT NULL CHECK (account_type IN ('User', 'Organization')),
    suspended_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE repositories (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    installation_id     UUID NOT NULL REFERENCES installations(id) ON DELETE CASCADE,
    github_repo_id      BIGINT NOT NULL UNIQUE,
    owner               TEXT NOT NULL,
    name                TEXT NOT NULL,
    default_branch      TEXT NOT NULL DEFAULT 'main',
    is_private          BOOLEAN NOT NULL DEFAULT TRUE,
    -- Per-repo overrides: findings budget, disabled categories, path ignores.
    config              JSONB NOT NULL DEFAULT '{}'::jsonb,
    review_enabled      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (owner, name)
);

CREATE INDEX idx_repositories_installation ON repositories (installation_id);

-- --------------------------------------------------------------------------
-- Index snapshots: one per (repo, commit) indexing attempt.
-- Incremental indexing diffs against the previous ready snapshot.
-- --------------------------------------------------------------------------

CREATE TABLE index_snapshots (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id       UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    commit_sha          CHAR(40) NOT NULL,
    parent_snapshot_id  UUID REFERENCES index_snapshots(id) ON DELETE SET NULL,
    status              snapshot_status NOT NULL DEFAULT 'pending',
    files_indexed       INTEGER NOT NULL DEFAULT 0,
    chunks_embedded     INTEGER NOT NULL DEFAULT 0,
    chunks_reused       INTEGER NOT NULL DEFAULT 0,   -- proves incrementality works
    embedding_model     TEXT NOT NULL,
    parser_version      TEXT NOT NULL,
    duration_ms         INTEGER,
    error               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repository_id, commit_sha, embedding_model, parser_version)
);

-- Hot path: "what is the newest usable index for this repo?"
CREATE INDEX idx_snapshots_repo_ready
    ON index_snapshots (repository_id, created_at DESC)
    WHERE status = 'ready';

-- --------------------------------------------------------------------------
-- Code corpus
-- --------------------------------------------------------------------------

CREATE TABLE files (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id       UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    snapshot_id         UUID NOT NULL REFERENCES index_snapshots(id) ON DELETE CASCADE,
    path                TEXT NOT NULL,
    blob_sha            CHAR(40) NOT NULL,
    language            TEXT NOT NULL,
    size_bytes          INTEGER NOT NULL CHECK (size_bytes >= 0),
    line_count          INTEGER NOT NULL CHECK (line_count >= 0),
    is_test             BOOLEAN NOT NULL DEFAULT FALSE,
    is_generated        BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (snapshot_id, path),
    CONSTRAINT files_path_is_relative CHECK (path !~ '^/' AND path !~ '(^|/)\.\.(/|$)')
);

CREATE INDEX idx_files_snapshot_lang ON files (snapshot_id, language);
CREATE INDEX idx_files_blob ON files (repository_id, blob_sha);

CREATE TABLE symbols (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id       UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_id             UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    fqn                 TEXT NOT NULL,          -- e.g. app.files.FileStore.read
    name                TEXT NOT NULL,
    kind                symbol_kind NOT NULL,
    line_start          INTEGER NOT NULL CHECK (line_start >= 1),
    line_end            INTEGER NOT NULL,
    signature           TEXT,
    docstring           TEXT,
    is_exported         BOOLEAN NOT NULL DEFAULT TRUE,
    CONSTRAINT symbols_span_ordered CHECK (line_end >= line_start)
);

CREATE INDEX idx_symbols_file ON symbols (file_id);
CREATE INDEX idx_symbols_fqn ON symbols (repository_id, fqn);
CREATE INDEX idx_symbols_name ON symbols (repository_id, name);

CREATE TABLE symbol_edges (
    id                  BIGSERIAL PRIMARY KEY,
    repository_id       UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    src_symbol_id       UUID NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    dst_symbol_id       UUID REFERENCES symbols(id) ON DELETE CASCADE,
    -- Unresolved targets keep their textual name so the graph degrades
    -- gracefully instead of silently losing edges. Dynamic dispatch in Python
    -- and structural typing in TS both defeat exact resolution; pretending
    -- otherwise would be the first hallucination in the pipeline.
    dst_unresolved_name TEXT,
    kind                edge_kind NOT NULL,
    confidence          REAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT edge_has_a_target CHECK (
        dst_symbol_id IS NOT NULL OR dst_unresolved_name IS NOT NULL
    )
);

-- Graph expansion walks both directions (callers and callees), so index both.
CREATE INDEX idx_edges_src ON symbol_edges (src_symbol_id, kind);
CREATE INDEX idx_edges_dst ON symbol_edges (dst_symbol_id, kind)
    WHERE dst_symbol_id IS NOT NULL;

CREATE TABLE chunks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id       UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_id             UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    symbol_id           UUID REFERENCES symbols(id) ON DELETE SET NULL,
    -- sha256(repo_id | path | symbol_fqn | normalized_body). The whole point of
    -- incremental indexing: a one-line change re-embeds one chunk, not 50k.
    content_hash        CHAR(64) NOT NULL,
    line_start          INTEGER NOT NULL CHECK (line_start >= 1),
    line_end            INTEGER NOT NULL,
    token_count         INTEGER NOT NULL CHECK (token_count > 0),
    content             TEXT NOT NULL,
    embedding           vector(1536),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunks_span_ordered CHECK (line_end >= line_start),
    UNIQUE (repository_id, content_hash)
);

CREATE INDEX idx_chunks_file ON chunks (file_id);

-- HNSW over IVFFlat: IVFFlat needs retraining as the corpus grows and degrades
-- on incremental inserts, which is exactly our write pattern. Every ANN query
-- is filtered by repository_id, so the partial-scan cost is bounded.
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_chunks_repo ON chunks (repository_id);

-- --------------------------------------------------------------------------
-- Reviews
-- --------------------------------------------------------------------------

CREATE TABLE pull_requests (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id       UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    number              INTEGER NOT NULL CHECK (number >= 1),
    title               TEXT NOT NULL,
    author_login        TEXT NOT NULL,
    base_ref            TEXT NOT NULL,
    head_ref            TEXT NOT NULL,
    state               TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repository_id, number)
);

CREATE TABLE review_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id       UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    pull_request_id     UUID NOT NULL REFERENCES pull_requests(id) ON DELETE CASCADE,
    snapshot_id         UUID REFERENCES index_snapshots(id) ON DELETE SET NULL,

    head_sha            CHAR(40) NOT NULL,
    base_sha            CHAR(40) NOT NULL,

    status              run_status NOT NULL DEFAULT 'queued',
    ruleset_version     TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    model               TEXT NOT NULL,

    -- GitHub redelivers webhooks. Idempotency lives here, in the durable store,
    -- not in the broker: a redelivery must be a no-op, not a second paid review.
    idempotency_key     CHAR(64) NOT NULL UNIQUE,

    risk_score          REAL CHECK (risk_score BETWEEN 0 AND 1),
    security_score      REAL CHECK (security_score BETWEEN 0 AND 1),
    performance_score   REAL CHECK (performance_score BETWEEN 0 AND 1),
    maintainability_score REAL CHECK (maintainability_score BETWEEN 0 AND 1),
    completeness_score  REAL CHECK (completeness_score BETWEEN 0 AND 1),

    findings_posted     INTEGER NOT NULL DEFAULT 0 CHECK (findings_posted >= 0),
    findings_suppressed INTEGER NOT NULL DEFAULT 0 CHECK (findings_suppressed >= 0),

    cost_usd            NUMERIC(10, 6) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    tokens_in           INTEGER NOT NULL DEFAULT 0 CHECK (tokens_in >= 0),
    tokens_out          INTEGER NOT NULL DEFAULT 0 CHECK (tokens_out >= 0),

    trace_id            TEXT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    error               TEXT,

    CONSTRAINT run_diff_is_nonempty CHECK (head_sha <> base_sha),
    CONSTRAINT run_terminal_has_finished_at CHECK (
        (status IN ('succeeded', 'failed', 'cancelled')) = (finished_at IS NOT NULL)
    ),
    CONSTRAINT run_error_iff_failed CHECK (
        (status = 'failed') = (error IS NOT NULL)
    )
);

CREATE INDEX idx_runs_pr ON review_runs (pull_request_id, started_at DESC);

-- Feeds the stuck-run sweeper (Redis is not a durable broker; the row is the
-- source of truth -- see ADR-004).
CREATE INDEX idx_runs_active ON review_runs (status, started_at)
    WHERE status NOT IN ('succeeded', 'failed', 'cancelled');

CREATE TABLE findings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID NOT NULL REFERENCES review_runs(id) ON DELETE CASCADE,

    severity            severity NOT NULL,
    category            finding_category NOT NULL,
    source              finding_source NOT NULL,

    title               TEXT NOT NULL CHECK (char_length(title) BETWEEN 8 AND 160),
    explanation         TEXT NOT NULL CHECK (char_length(explanation) >= 20),

    path                TEXT NOT NULL,
    line_start          INTEGER NOT NULL CHECK (line_start >= 1),
    line_end            INTEGER NOT NULL,

    suggested_fix       TEXT,
    improved_code       TEXT,

    confidence          REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    confidence_calibrated BOOLEAN NOT NULL DEFAULT FALSE,

    rule_id             TEXT,
    references_json     JSONB NOT NULL DEFAULT '[]'::jsonb,

    verification_status verification_status NOT NULL DEFAULT 'pending',
    gates_failed        TEXT[] NOT NULL DEFAULT '{}',

    -- Stable across runs; lets a dismissed comment stay dismissed even after a
    -- prompt tweak rewords the explanation.
    fingerprint         CHAR(16) NOT NULL,
    dedup_group         TEXT,

    prompt_version      TEXT,
    posted_comment_id   BIGINT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT findings_span_ordered CHECK (line_end >= line_start),
    CONSTRAINT findings_patch_needs_prose CHECK (
        improved_code IS NULL OR suggested_fix IS NOT NULL
    ),
    -- Mirrors the domain validators. Belt and braces: model output reaches this
    -- table, and a constraint violation is a loud failure rather than a silent
    -- bad comment.
    CONSTRAINT findings_analyzer_has_rule CHECK (
        source = 'llm' OR rule_id IS NOT NULL
    ),
    CONSTRAINT findings_llm_has_prompt_version CHECK (
        source <> 'llm' OR prompt_version IS NOT NULL
    ),
    CONSTRAINT findings_verified_has_no_failed_gates CHECK (
        verification_status <> 'verified' OR cardinality(gates_failed) = 0
    )
);

CREATE INDEX idx_findings_run ON findings (run_id, severity DESC, confidence DESC);
CREATE INDEX idx_findings_fingerprint ON findings (fingerprint);
CREATE INDEX idx_findings_title_trgm ON findings USING gin (title gin_trgm_ops);

-- Evidence is a child table, not a JSONB blob: the retrieval inspector joins
-- against it, and the M6 harness aggregates over evidence roles to answer
-- "which retrieval sources actually produce true positives?"
CREATE TABLE finding_evidence (
    id                  BIGSERIAL PRIMARY KEY,
    finding_id          UUID NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    role                TEXT NOT NULL,
    path                TEXT NOT NULL,
    line_start          INTEGER NOT NULL CHECK (line_start >= 1),
    line_end            INTEGER NOT NULL,
    excerpt             TEXT NOT NULL,
    symbol_fqn          TEXT,
    CONSTRAINT evidence_span_ordered CHECK (line_end >= line_start)
);

CREATE INDEX idx_evidence_finding ON finding_evidence (finding_id);

CREATE TABLE run_events (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              UUID NOT NULL REFERENCES review_runs(id) ON DELETE CASCADE,
    stage               run_status NOT NULL,
    message             TEXT,
    payload             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_run_events_run ON run_events (run_id, created_at);

-- --------------------------------------------------------------------------
-- Evaluation corpus (M6). Defined here so the schema is stable before the
-- harness needs it -- retrofitting eval storage after the fact is how eval
-- harnesses end up as one-off scripts.
-- --------------------------------------------------------------------------

CREATE TABLE eval_cases (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug                TEXT NOT NULL UNIQUE,
    repo_url            TEXT NOT NULL,
    base_sha            CHAR(40) NOT NULL,
    head_sha            CHAR(40) NOT NULL,
    origin              TEXT NOT NULL CHECK (origin IN ('seeded', 'historical')),
    -- Ground truth: the defects a competent human reviewer should catch.
    expected_defects    JSONB NOT NULL,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE eval_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id             UUID NOT NULL REFERENCES eval_cases(id) ON DELETE CASCADE,
    run_id              UUID REFERENCES review_runs(id) ON DELETE SET NULL,
    prompt_version      TEXT NOT NULL,
    ruleset_version     TEXT NOT NULL,
    model               TEXT NOT NULL,
    true_positives      INTEGER NOT NULL DEFAULT 0,
    false_positives     INTEGER NOT NULL DEFAULT 0,
    false_negatives     INTEGER NOT NULL DEFAULT 0,
    verification_drops  JSONB NOT NULL DEFAULT '{}'::jsonb,  -- gate -> count
    cost_usd            NUMERIC(10, 6) NOT NULL DEFAULT 0,
    duration_ms         INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_eval_results_config
    ON eval_results (prompt_version, ruleset_version, model, created_at DESC);

COMMIT;
