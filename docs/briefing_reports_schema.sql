-- Recommended schema for BriefingGenerationAgent persistence.
-- Move this into the backend Flyway migration path when DB ownership is settled.

CREATE TABLE IF NOT EXISTS briefing_reports (
    id VARCHAR(40) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    briefing_type VARCHAR(20) NOT NULL,
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    requested_by_user_id BIGINT,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    progress NUMERIC(3,2) DEFAULT 0.0,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    confidence NUMERIC(3,2),
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    CONSTRAINT chk_briefing_reports_type
        CHECK (briefing_type IN ('daily', 'weekly', 'monthly'))
);

CREATE INDEX IF NOT EXISTS idx_briefing_reports_status
    ON briefing_reports(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_briefing_reports_user
    ON briefing_reports(requested_by_user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS briefing_report_cards (
    briefing_report_id VARCHAR(40) NOT NULL REFERENCES briefing_reports(id) ON DELETE CASCADE,
    card_news_id TEXT NOT NULL REFERENCES card_news(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (briefing_report_id, card_news_id)
);

CREATE TABLE IF NOT EXISTS briefing_report_articles (
    briefing_report_id VARCHAR(40) NOT NULL REFERENCES briefing_reports(id) ON DELETE CASCADE,
    raw_article_id BIGINT NOT NULL REFERENCES raw_articles(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (briefing_report_id, raw_article_id)
);

