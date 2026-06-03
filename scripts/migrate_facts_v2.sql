-- One-shot migration from the (cik, concept, period, accession) PK to the
-- (cik, concept, period, unit) PK introduced when iter_facts moved to
-- duration-based period classification. TRUNCATE is safe here because the
-- next ingest run rebuilds the table from companyfacts.

BEGIN;

ALTER TABLE facts DROP CONSTRAINT IF EXISTS facts_pkey;
TRUNCATE TABLE facts;
ALTER TABLE facts ALTER COLUMN unit SET NOT NULL;
ALTER TABLE facts ADD PRIMARY KEY (cik, concept, period, unit);

COMMIT;
