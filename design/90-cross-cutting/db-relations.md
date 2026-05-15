# DB Relations — Article/Card/Briefing Provenance

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `DB Relations` |
| **범위** | raw_articles 중심 관계, 카드뉴스/브리핑/분석/로그 FK 정비 |
| **상태** | ✅ 반영 예정 — backend Flyway `V20`, `V21` |
| **Owner** | Backend + Ingestion |
| **Version** | v1 (2026-05-15) |

## 2. 원칙

운영 중인 `raw_articles`, `article_images`, `card_news` 데이터는 삭제하지 않는다. 기존 writer가 사용하는 legacy 컬럼도 즉시 rename/drop 하지 않는다.

관계 정비는 다음 순서만 허용한다.

1. nullable 컬럼 또는 매핑 테이블 추가
2. 기존 JSON/legacy 컬럼에서 백필
3. orphan 검증
4. index 추가
5. FK 추가
6. application writer 전환
7. legacy 컬럼 정리는 별도 후속 migration

## 3. 기준 관계

```text
raw_articles.id
  ├── article_images.raw_article_id
  ├── card_news_articles.raw_article_id
  ├── article_peer_companies.raw_article_id
  ├── briefing_report_articles.raw_article_id
  ├── weak_signal_cards.source_raw_article_id
  └── raw_articles.crawl_run_id -> crawl_runs.id

card_news.id
  ├── card_news_articles.card_news_id
  ├── article_images.card_news_id
  ├── evidence_chain.card_news_id
  ├── briefing_report_cards.card_news_id
  ├── briefing_history_cards.card_news_id
  ├── feedback.card_news_id
  ├── weak_signal_cards.card_news_id
  └── analysis_ledger_card_news.card_news_id

peer_companies.id
  ├── article_peer_companies.peer_company_id
  ├── card_news.peer_company_id
  ├── peer_financials.peer_company_id
  ├── job_postings.peer_company_id
  ├── weak_signal_cards.peer_company_id
  └── analysis_ledger_peer_companies.peer_company_id

briefing_reports.id
  ├── briefing_history.briefing_report_id
  ├── briefing_report_cards.briefing_report_id
  ├── briefing_report_articles.briefing_report_id
  └── briefing_recipients.briefing_report_id

crawl_runs.id
  ├── raw_articles.crawl_run_id
  ├── crawl_logs.crawl_run_id
  └── pipeline_logs.crawl_run_id
```

## 4. Legacy Compatibility

| Legacy | 신규 기준 | 전환 정책 |
|---|---|---|
| `article_images.article_id` | `article_images.raw_article_id` | V20 백필, 기존 컬럼 유지 |
| `article_images.issue_card_id` | `article_images.card_news_id` | V20 백필, 기존 컬럼 유지 |
| `evidence_chain.issue_card_id` | `evidence_chain.card_news_id` | V20 백필, 기존 PK 유지 |
| `card_news.company` | `card_news.peer_company_id` | V20 백필, writer는 후속 전환 |
| `raw_articles.company` JSONB | `article_peer_companies` | V20 alias/name/keyword 매칭 백필 |
| `briefing_history.card_ids` | `briefing_history_cards` | V21 백필, 기존 배열 유지 |
| `feedback.artifact_type/artifact_id` | typed nullable FKs | V21 백필, polymorphic 컬럼 유지 |
| `analysis_ledger.peer_ids/source_card_ids` JSONB | mapping tables | V21 백필, JSONB 유지 |

## 5. Migration Slots

- `axis-backend/src/main/resources/db/migration/V20__normalize_article_card_peer_relations.sql`
  - `crawl_runs`, `crawl_cursors` Flyway ownership 보강
  - `raw_articles.crawl_run_id`
  - `article_images.raw_article_id`, `article_images.card_news_id`
  - `evidence_chain.card_news_id`
  - `card_news_articles`
  - `article_peer_companies`
  - peer FK alias columns

- `axis-backend/src/main/resources/db/migration/V21__add_briefing_feedback_log_relation_tables.sql`
  - briefing report/history/card/article/recipient mappings
  - feedback typed FK columns
  - crawl/pipeline/usage log FKs
  - weak signal typed links
  - analysis ledger peer/card mapping tables

## 6. Application Follow-up

Writer 전환은 별도 PR에서 진행한다.

- axis-ai `save_articles`: `raw_articles.crawl_run_id` 직접 저장
- axis-ai `save_card_news`: `card_news.peer_company_id` 저장
- axis-ai `save_evidence_chain`: `evidence_chain.card_news_id` 저장
- axis-ai evidence/card flow: `card_news_articles` 직접 upsert
- backend `ArticleImage`: `rawArticleId`, `cardNewsId` 신규 컬럼 기준 조회
- backend `RawArticle`: `company` JSONB를 문자열이 아닌 JSON/List 타입으로 매핑
- feedback/briefing/analysis writer: typed FK와 mapping table 동시 쓰기

## 7. Changelog

- **2026-05-15** — V20/V21 기준 관계 정비 문서 추가. 기존 데이터 보존과 rollout compatibility를 최우선 원칙으로 확정.
