-- Quorum Postgres init. Runs once on a fresh ./data/postgres volume.
-- Schemas: facts, concept_aliases, trace_events.
-- LangGraph checkpointer manages its own tables on first connect; not declared here.

BEGIN;

-- 1. XBRL facts. The (cik, concept, period, unit) PK enforces "one canonical
-- value per company-concept-period-currency". iter_facts dedups upstream by
-- collapsing companyfacts restatement clones to the latest accession, so the
-- PK never conflicts at write time. Unit is part of the PK because the same
-- concept can be reported in both USD and a foreign currency for ADRs.
CREATE TABLE IF NOT EXISTS facts (
  cik          TEXT NOT NULL,
  concept      TEXT NOT NULL,
  period       TEXT NOT NULL,
  unit         TEXT NOT NULL,
  value        NUMERIC,
  accession    TEXT NOT NULL,
  PRIMARY KEY (cik, concept, period, unit)
);
CREATE INDEX IF NOT EXISTS idx_facts_lookup ON facts (cik, concept, period);

-- 2. Concept aliases (normalized concept dictionary, see ARCHITECTURE 4.3).
-- Ordering = position in the fallback chain; resolver returns first non-null.
CREATE TABLE IF NOT EXISTS concept_aliases (
  axis_metric_key    TEXT NOT NULL,
  ticker_or_default  TEXT NOT NULL,
  ordering           INT  NOT NULL,
  concept            TEXT NOT NULL,
  PRIMARY KEY (axis_metric_key, ticker_or_default, ordering)
);
CREATE INDEX IF NOT EXISTS idx_aliases_lookup
  ON concept_aliases (axis_metric_key, ticker_or_default, ordering);

-- 3. Trace events. Canonical schema lives in docs/ARCHITECTURE.md 4.2.
-- Any column rename or addition lands in ARCHITECTURE.md first; this DDL follows.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'trace_error_kind') THEN
    CREATE TYPE trace_error_kind AS ENUM ('none', 'transient', 'terminal');
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS trace_events (
  id                       BIGSERIAL PRIMARY KEY,
  request_id               UUID NOT NULL,
  trace_id                 UUID NOT NULL,
  node_name                TEXT NOT NULL,
  attempt_number           INT  NOT NULL DEFAULT 1,
  "timestamp"              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  duration_ms              INT,
  input_shape              JSONB,
  output_shape             JSONB,
  tokens_in                INT  NOT NULL DEFAULT 0,
  tokens_out               INT  NOT NULL DEFAULT 0,
  cache_read_tokens        INT  NOT NULL DEFAULT 0,
  cost_dollars_billed      NUMERIC(14, 6) NOT NULL DEFAULT 0,
  cost_dollars_effective   NUMERIC(14, 6) NOT NULL DEFAULT 0,
  error_kind               trace_error_kind NOT NULL DEFAULT 'none',
  error_reason             TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace_request ON trace_events (request_id, "timestamp");
CREATE INDEX IF NOT EXISTS idx_trace_node    ON trace_events (node_name,  "timestamp");

COMMIT;
