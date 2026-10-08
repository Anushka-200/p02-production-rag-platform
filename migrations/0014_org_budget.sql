-- Per-tenant spend control. One organisation must not be able to run up the bill for everyone, whether
-- by heavy use, a bug that loops, or an attack that asks expensive questions in a loop.
--
-- `orgs.daily_budget_usd`  the organisation's own daily cap (NULL = the server default).
-- `org_usage`              what the organisation has spent today, across member chat, the widget and
--                          WhatsApp. Every fresh answer adds to it; `alert_level` makes each alert
--                          fire once per day (0 none, 1 warning at the alert fraction, 2 exhausted).

ALTER TABLE orgs ADD COLUMN daily_budget_usd numeric(10, 4) CHECK (daily_budget_usd >= 0);

CREATE TABLE org_usage (
    org_id        uuid NOT NULL REFERENCES orgs (id) ON DELETE CASCADE,
    day           date NOT NULL DEFAULT current_date,
    requests      integer NOT NULL DEFAULT 0,
    input_tokens  bigint NOT NULL DEFAULT 0,
    output_tokens bigint NOT NULL DEFAULT 0,
    cost_usd      numeric(10, 6) NOT NULL DEFAULT 0,
    alert_level   smallint NOT NULL DEFAULT 0,
    PRIMARY KEY (org_id, day)
);

ALTER TABLE org_usage ENABLE ROW LEVEL SECURITY;
-- Our own code reads it before answering (also for anonymous visitors, who never run SQL themselves)
-- and adds to it after answering. Only the organisation's own row, never another's.
CREATE POLICY own_org ON org_usage FOR ALL
    USING (org_id = app_org()) WITH CHECK (org_id = app_org());
