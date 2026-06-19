# Raw Content VDB Pipeline: Current Flow And Redesign

## Purpose

뉴스 기사, IR/DART 자료, 리서치 자료의 본문이 `raw_articles.content`에 누적되면서 다음 문제가 생겼다.

- RDB가 원문 저장소처럼 커지고, `SELECT content`가 여러 경로로 확산된다.
- LLM 프롬프트 조립 시 원문 전체가 실수로 들어갈 수 있다.
- Qdrant는 현재 카드 요약 검색과 일부 DART 청크 검색에만 쓰여, 원문 근거 검색 저장소 역할을 충분히 하지 못한다.
- 이미 RDB에 적재된 본문을 VectorDB로 백필하고, 이후 파이프라인은 본문 청크를 VDB에서 가져오고 메타데이터/정형 자료만 RDB에서 조합해야 한다.

이 문서는 현재 데이터 흐름을 분석하고, 원문 본문을 VDB 중심으로 전환하기 위한 목표 구조와 단계별 파이프라인을 정의한다.

## Current State

### Storage

| Layer | Current role | Main tables/collections | Notes |
| --- | --- | --- | --- |
| PostgreSQL | 원문, 메타데이터, 분석 결과, 카드 결과 저장 | `raw_articles`, `raw_article_parse_results`, `integrated_issues`, `card_news`, `raw_article_*` projection tables | `raw_articles.content`가 원문 본문 저장소 역할을 한다. |
| Qdrant `axis_main` | 카드/이슈 요약 검색 | card title + summary vectors | `src/rag/vector_index.py`, `src/rag/hybrid_search.py`가 사용한다. 원문 청크 검색용이 아니다. |
| Qdrant `axis_documents` | DART 일부 청크와 assistant knowledge 저장 | DART chunks, integrated issue/card/peer derived knowledge | 원문 전체 범용 저장소가 아니라 혼합 목적 컬렉션이다. |

스키마상 `raw_articles.qdrant_vector_id`는 존재하지만 단일 UUID라서 다중 청크 문서에는 부족하다. 원문 하나가 여러 청크로 분할되는 구조에서는 별도 chunk mapping 테이블이 필요하다.

### Ingestion Flow

현재 핵심 흐름은 다음과 같다.

1. Crawler/parser가 뉴스, IR, DART, 리서치 자료를 수집한다.
2. `src/db/article_store.py::save_articles()`가 `raw_articles.content`에 본문을 저장한다.
3. 분류, 파서 결과, business signal, financial metric 등은 RDB projection table에 저장된다.
4. 카드 생성 또는 이슈 통합 단계에서 `get_articles_by_ids()`가 `content`까지 조회한다.
5. `SourceSummarizer`는 현재 원문 전체를 프롬프트에 넣지 않고 rule-based `evidence_snippets`만 추출하지만, snippet 후보는 여전히 RDB 본문에서 읽는다.
6. `CardNewsComposer.generate_from_cluster()` 경로는 `_format_articles()`에서 `content` 전체를 프롬프트에 넣는 fallback이 남아 있다.
7. 생성된 카드의 title/summary는 `axis_main`에 임베딩된다.
8. DART 일부 섹션은 선택적으로 `axis_documents`에 청크로 인덱싱된다.

### Query And LLM Context Flow

| Consumer | Current retrieval | Risk |
| --- | --- | --- |
| Search API | `axis_main` hybrid search | 카드 요약 검색만 가능하다. 원문 근거 검색이 아니다. |
| Chat orchestrator | `axis_main` + assistant knowledge + RDB raw article fallback | fallback이 `raw_articles.content ILIKE`와 `SELECT content`를 사용한다. |
| Integration agent | `InputBundle.items`에 full article dict 포함 | downstream이 원문 필드를 실수로 프롬프트에 넣을 수 있다. |
| SourceSummarizer | RDB content에서 local evidence sentence 추출 | 프롬프트는 제한되어 있지만 RDB 원문 의존은 남아 있다. |
| Card composer fallback | full content를 articles text로 포맷 | 토큰 폭증의 직접 위험 경로다. |
| Analysis context builder | Qdrant precedent/card context + RDB facts | 원문 청크 검색은 하지 않는다. |

## Current Problems

1. `raw_articles.content`가 canonical body store다.
   - RDB는 메타데이터와 정형 projection을 위한 저장소여야 하는데, 현재는 긴 원문 보관과 검색까지 맡는다.
   - RDB snapshot 기준 `raw_articles`가 15,299건이고, PDF/IR 텍스트와 긴 뉴스 본문이 섞여 있다.

2. `get_articles_by_ids()`가 기본적으로 `content`를 반환한다.
   - 호출자는 메타데이터만 필요해도 full body를 받는다.
   - LLM prompt composer로 full body dict가 전달될 수 있다.

3. Qdrant collection 역할이 분리되어 있지 않다.
   - `axis_main`은 카드 검색용이다.
   - `axis_documents`는 DART 청크와 assistant knowledge가 섞여 있다.
   - 범용 raw content retrieval 전용 collection이 없다.

4. 단일 `qdrant_vector_id`로는 chunk-level provenance가 불가능하다.
   - 원문 하나가 여러 point로 나뉘면 `raw_article_id -> chunk_uid -> qdrant_point_id` 매핑이 필요하다.

5. 일부 카드/레거시 데이터의 source provenance가 빈 상태다.
   - DB snapshot에서 `card_news.source_raw_article_ids`, `source_articles`, `evidence_payload`가 비어 있는 샘플이 확인된다.
   - 이후 retrieval을 정확히 하려면 source URL/title 기반 보정 또는 재생성이 필요하다.

6. 시간 기준 검색이 불완전하다.
   - backtest/as-of 분석에서는 미래 문서가 검색되면 안 된다.
   - 현재 `AnalysisContextBuilder`는 look-ahead를 피하려고 Qdrant RAG 자체를 skip하는 경우가 있다.
   - 원문 VDB 검색은 `published_at <= as_of` filter를 지원해야 한다.

## Target Architecture

### Storage Principle

| Data type | Target storage |
| --- | --- |
| 원문 본문 | Qdrant raw content chunks |
| 본문 chunk text | Qdrant payload에 bounded chunk로 저장 |
| 원문 metadata | PostgreSQL `raw_articles` |
| chunk mapping/status | PostgreSQL new mapping table |
| parser/metric/signal/fact | PostgreSQL projection tables |
| card/issue summary | PostgreSQL + `axis_main` card vectors |
| derived assistant knowledge | `axis_documents` or separate derived collection |

Qdrant payload에는 문서 전체 원문을 저장하지 않는다. 다만 LLM 근거 조합을 위해 검색 결과 point가 반환할 수 있는 bounded chunk text는 필요하다. 권장 chunk 크기는 source type별로 다르게 두되, point payload의 `text`는 약 800-1,500자 범위로 제한한다.

### New Qdrant Collection

새 collection을 권장한다.

```text
axis_raw_content
```

기존 `axis_documents`를 재사용할 수도 있지만, DART/assistant derived knowledge와 범용 원문 청크가 섞이면 filter와 운영이 복잡해진다. `axis_raw_content`를 분리하면 TTL, 재인덱싱, 삭제, payload schema migration을 독립적으로 운영할 수 있다.

권장 payload:

```json
{
  "raw_article_id": "uuid",
  "chunk_uid": "uuid-or-stable-string",
  "source_type": "news|ir|dart|research|...",
  "source_name": "string",
  "publisher": "string",
  "title": "string",
  "url": "string",
  "published_at": "iso8601",
  "published_at_ts": 1710000000,
  "company_ids": ["..."],
  "matched_companies": ["..."],
  "matched_sectors": ["..."],
  "section_key": "body|business|risk|financial|qa|...",
  "section_title": "string",
  "page": 12,
  "chunk_index": 0,
  "text": "bounded chunk text",
  "text_hash": "sha256",
  "content_hash": "sha256",
  "token_estimate": 320,
  "index_version": "raw-content-v1"
}
```

### New PostgreSQL Mapping Table

`raw_articles.qdrant_vector_id`는 유지하되, chunk-level mapping은 별도 테이블로 관리한다.

```sql
CREATE TABLE raw_article_content_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_article_id UUID NOT NULL REFERENCES raw_articles(id) ON DELETE CASCADE,
    qdrant_collection TEXT NOT NULL DEFAULT 'axis_raw_content',
    qdrant_point_id UUID NOT NULL,
    chunk_uid TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    section_key TEXT,
    section_title TEXT,
    char_start INTEGER,
    char_end INTEGER,
    token_estimate INTEGER,
    text_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    index_version TEXT NOT NULL,
    index_status TEXT NOT NULL DEFAULT 'indexed',
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    error_message TEXT,
    UNIQUE (raw_article_id, chunk_uid, index_version)
);
```

이 테이블에는 긴 본문 text를 저장하지 않는다. 운영 화면이나 디버깅이 필요하면 `text_preview VARCHAR(300)` 정도만 선택적으로 둔다.

### RDB Content Column Transition

전환 중에는 `raw_articles.content`를 바로 삭제하지 않는다. 다음 순서가 안전하다.

1. `content_storage_status` 또는 metadata field로 `rdb`, `vdb_indexed`, `vdb_verified`, `rdb_compacted` 상태를 기록한다.
2. 신규 ingestion은 원문을 임시로 RDB에 저장한 뒤 VDB index 성공 시 compact/null 처리한다.
3. 기존 backfill이 완료되고 모든 reader가 metadata-only/VDB retrieval로 바뀐 뒤 `content`를 null 또는 preview-only로 줄인다.
4. fallback read는 feature flag 뒤에만 둔다.

## Target Data Flow

```text
Crawler/parser
  -> raw article metadata + transient body
  -> ContentChunker
  -> ContentIndexService
       -> Qdrant axis_raw_content points
       -> raw_article_content_chunks mapping
  -> raw_articles metadata/projection tables

Analysis/chat/search request
  -> RDB metadata/facts/signals/metrics
  -> ContentRetrievalService
       -> Qdrant axis_raw_content filtered search
       -> top-k bounded chunks
  -> Context assembler
       -> metadata + facts + selected chunks
  -> LLM prompt
```

`axis_main`은 계속 카드/이슈 요약 검색용으로 유지한다. 원문 근거 검색은 `axis_raw_content`가 담당한다.

## Agent Retrieval Decision

레포 조사 기준으로 agent가 필요한 본문 호출은 대부분 LLM context 조립용이다. 따라서 agent-facing retrieval API를 "전체 본문 복원"과 "LLM context" 두 갈래로 노출하지 않는다.

권장 원칙:

- agent는 항상 token budget이 있는 `retrieve_context()`만 호출한다.
- `retrieve_context()`는 RDB metadata filter와 VDB top-k chunk search를 결합한다.
- `raw_article_id`, `integrated_issue_id`, `card_news_id`가 이미 있어도 전체 chunk를 모두 이어 붙이지 않는다.
- source id는 VDB 검색 범위를 좁히는 filter로 쓰고, LLM에는 질문/작업 목적에 맞는 chunk만 넣는다.
- 전체 chunk 복원은 백필 검증, 관리자 원문 확인, content compaction rollback 같은 internal utility로만 둔다.

확인된 현재 상태:

| Area | Repo finding | Target change |
| --- | --- | --- |
| Backend raw article list | list query는 `content`를 내려주지 않는다. | 유지한다. |
| Backend raw article detail | `/api/raw-articles/{id}`는 `ra.content`를 내려준다. | RDB content compact 후에는 VDB chunk exact read로 대체하거나 admin/debug 용도로 제한한다. |
| Frontend raw articles page | 현재 `RawArticlesView`는 raw article API가 아니라 card news 기반 mixer 화면이다. | 일반 사용자 flow에서 full raw body 필요성은 낮다. |
| Chat agent | 후보 source snippet만 LLM prompt에 넣지만, raw article fallback은 RDB `content` 600자를 읽는다. | raw article fallback을 VDB chunk retrieval로 교체한다. |
| SourceSummarizer | LLM prompt에는 `evidence_snippets`만 넣지만 snippet 후보를 RDB `content`에서 뽑는다. | VDB evidence chunk에서 snippet을 받도록 변경한다. |
| Card composer fallback | `_format_articles()`가 full `content`를 LLM prompt에 넣는다. | fallback도 `retrieve_context()` 결과만 쓰도록 변경한다. |
| Integrated/insight/action knowledge | `integrated_issues`, `card_news.evidence_payload`, implication/action payload를 RDB에서 다시 읽는 경로가 있다. | 긴 narrative field는 `knowledge_type/source_field` 단위로 VDB 인덱싱하고 context retrieval로 호출한다. |

즉 비용 절감의 핵심 API는 다음 하나다.

```python
retrieve_context(
    query: str,
    *,
    knowledge_types: list[str],
    source_record_ids: list[str] | None = None,
    company_ids: list[str] | None = None,
    source_types: list[str] | None = None,
    as_of_ts: int | None = None,
    max_chunks: int,
    max_chars: int,
) -> RetrievalContext
```

`source_record_ids`가 주어져도 full body를 복원하지 않고, 해당 id 집합 안에서 query와 관련 있는 chunk만 고른다. LLM 비용은 저장 위치가 아니라 prompt token 수로 결정되므로, agent 경로는 항상 `max_chunks/max_chars`를 강제한다.

## Required Services

### ContentChunker

역할:

- source type별 본문을 deterministic chunk로 분할한다.
- chunk_uid를 stable하게 만든다.
- 너무 작은/중복/boilerplate chunk를 제거한다.
- parser_result가 있는 문서는 section metadata를 보존한다.

초기 정책:

| Source type | Chunking |
| --- | --- |
| news | 문장 단위 window, 600-1,000자, overlap 1-2문장 |
| IR/PDF | parser_result document_chunks 우선, 없으면 page/heading 기반 fallback |
| DART | 기존 `document_index.index_dart_chunks()` 로직을 일반화하고 section filter 확장 |
| research | heading/paragraph 기반 |

### ContentIndexService

역할:

- chunk를 BGE-M3 dense+sparse embedding으로 변환한다.
- deterministic point id로 Qdrant upsert를 idempotent하게 만든다.
- `raw_article_content_chunks`에 mapping/status를 저장한다.
- 실패한 chunk는 retry 가능한 상태로 남긴다.

초기 파일 위치:

```text
src/rag/content_index.py
```

### ContentRetrievalService

역할:

- semantic query 또는 source raw_article_ids 기준으로 `axis_raw_content`를 검색한다.
- RDB에서 metadata, metrics, business signals, parser-derived facts를 가져온다.
- chunk text와 metadata를 하나의 context item으로 조합한다.
- token/char budget, source diversity, dedupe를 적용한다.

초기 파일 위치:

```text
src/rag/content_retrieval.py
```

검색 모드:

| Mode | Use case | Required filters |
| --- | --- | --- |
| `article_basis` | 이미 source raw_article_ids가 정해진 카드/이슈 | `raw_article_id in (...)` |
| `semantic_query` | chat/search 질의 | query text, company/source filters |
| `cluster_basis` | issue integration source bundle | cluster article ids + query expansion |
| `point_in_time` | backtest/as-of 분석 | `published_at_ts <= as_of_ts` |

### Metadata-Only Article Store

`article_store.get_articles_by_ids()`는 현재 full `content`를 반환한다. 바로 default를 바꾸면 영향 범위가 크므로 다음 순서를 권장한다.

1. `get_article_metadata_by_ids()`를 추가한다.
2. `get_articles_by_ids(include_content=False)` 옵션을 추가하되, 기존 호출부는 단계적으로 명시한다.
3. LLM prompt 경로는 `include_content=False` 또는 metadata API만 쓰도록 테스트로 고정한다.
4. legacy/local processing에서만 `include_content=True`를 허용한다.

## Migration And Backfill Pipeline

### Phase 0: Audit And Guardrails

- `content`를 SELECT하거나 LLM prompt로 전달하는 호출부를 목록화한다.
- `CardNewsComposer.generate_from_cluster()` fallback의 full content prompt를 제거하거나 hard cap을 적용한다.
- chat orchestrator의 RDB raw article fallback을 feature flag로 감싼다.
- prompt assembly 테스트에 `content:` 또는 긴 원문이 들어가지 않는지 assertion을 추가한다.

### Phase 1: Schema And Collection

- `raw_article_content_chunks` migration을 추가한다.
- Qdrant `axis_raw_content` collection을 생성한다.
- payload index를 설정한다.
  - `raw_article_id`
  - `source_type`
  - `published_at_ts`
  - `company_ids` 또는 `matched_companies`
  - `matched_sectors`
  - `index_version`

### Phase 2: Backfill

Backfill candidate query:

```sql
SELECT
    id,
    source_type,
    source_name,
    publisher,
    title,
    url,
    published_at,
    company,
    matched_companies,
    matched_sectors,
    metadata,
    content
FROM raw_articles r
WHERE content IS NOT NULL
  AND length(content) > 0
  AND NOT EXISTS (
      SELECT 1
      FROM raw_article_content_chunks c
      WHERE c.raw_article_id = r.id
        AND c.index_version = :index_version
        AND c.index_status = 'indexed'
        AND c.deleted_at IS NULL
  )
ORDER BY published_at NULLS LAST, id
LIMIT :batch_size;
```

Backfill steps:

1. batch로 `raw_articles.content`를 읽는다.
2. source type별 chunker를 적용한다.
3. chunk별 embedding을 생성한다.
4. deterministic point id로 Qdrant에 upsert한다.
5. `raw_article_content_chunks`에 mapping을 upsert한다.
6. sample query로 article id filter retrieval을 검증한다.
7. 검증이 끝난 article은 metadata에 content hash/length/storage status를 기록한다.

RDB content compact/null은 모든 reader migration 후 별도 phase에서 실행한다.

### Phase 3: Runtime Retrieval Migration

1. `SourceSummarizer`
   - 기존 local evidence sentence extraction 대신 `ContentRetrievalService` 결과 chunk를 받는다.
   - prompt에는 chunk text, source title, published_at, url, section metadata만 들어간다.

2. `IntegrationAgent`
   - `InputBundle.items`에 full `content`를 싣지 않는다.
   - source ids와 metadata를 넘기고, 근거 본문은 retrieval service로 요청한다.

3. `ChatOrchestrator`
   - `_raw_article_search()`의 `content ILIKE` fallback을 제거한다.
   - raw article 후보는 VDB semantic search + RDB metadata enrichment로 만든다.

4. `CardNewsComposer`
   - `generate_from_cluster()`의 `_format_articles()`가 full content를 쓰지 않도록 변경한다.
   - fallback도 retrieved evidence chunk만 사용한다.

5. `AnalysisContextBuilder`
   - as-of가 있어도 Qdrant를 통째로 skip하지 않고 `published_at_ts <= as_of_ts` 필터로 검색한다.

### Phase 4: RDB Content Compaction

전제 조건:

- 신규 ingestion이 VDB indexing을 성공해야 complete로 처리된다.
- LLM prompt 경로가 metadata-only API와 content retrieval API만 사용한다.
- raw article search가 RDB `content`를 검색하지 않는다.
- backfill coverage와 retrieval smoke test가 통과한다.

Compaction policy:

```sql
UPDATE raw_articles
SET
    content = NULL,
    metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
        'content_storage', 'qdrant',
        'content_index_version', :index_version,
        'content_hash', :content_hash,
        'content_length', :content_length,
        'content_compacted_at', now()
    )
WHERE id = :raw_article_id;
```

## Current Vs Target Comparison

| Area | Current | Target |
| --- | --- | --- |
| 원문 저장 | `raw_articles.content` | `axis_raw_content` chunk payload |
| 원문 mapping | `raw_articles.qdrant_vector_id` 단일 값 | `raw_article_content_chunks` N:1 mapping |
| 카드 검색 | `axis_main` title+summary | 유지 |
| 원문 검색 | RDB content read/ILIKE, 일부 DART only VDB | Qdrant raw content hybrid search |
| LLM context | 일부 경로에서 full content 위험 | bounded top-k chunks only |
| metadata/facts | RDB | 유지 |
| as-of 분석 | Qdrant skip 가능 | timestamp filter |
| 신규 ingestion | RDB content 먼저 저장 | transient body -> VDB index -> metadata-only |
| legacy data | RDB content 대량 존재 | backfill 후 compact/null |

## Implementation Order

1. Add schema migration for `raw_article_content_chunks`.
2. Add `axis_raw_content` collection setup to Qdrant client.
3. Implement `ContentChunker` and `ContentIndexService`.
4. Add dry-run backfill script with batch size, source type filter, and verification report.
5. Implement `ContentRetrievalService`.
6. Replace chat raw article fallback with VDB retrieval.
7. Replace `SourceSummarizer` input path to use retrieved chunks.
8. Remove full-content prompt fallback in `CardNewsComposer`.
9. Add metadata-only article read API and migrate LLM-facing paths.
10. Run full backfill, then compact/null RDB content after verification.

## Acceptance Criteria

- LLM prompt tests prove no full `raw_articles.content` field is inserted.
- `SourceSummarizer` receives bounded evidence chunks, not full article bodies.
- `ChatOrchestrator` raw article retrieval no longer uses SQL `content ILIKE`.
- `CardNewsComposer.generate_from_cluster()` cannot format full body into the prompt.
- `axis_raw_content` retrieval supports filters for `raw_article_id`, source type, company, sector, and `published_at_ts <= as_of_ts`.
- Backfill is idempotent for the same `index_version`.
- Each indexed chunk has a RDB mapping row and stable provenance.
- Cards generated after migration have populated `source_raw_article_ids` and evidence references.
- RDB content compaction only runs after retrieval smoke tests pass.

## Test Plan

Unit tests:

- chunker creates stable chunk ids for the same input.
- chunker respects max char/token bounds.
- content indexer is idempotent for the same article/version.
- retrieval service dedupes repeated chunks by `text_hash`.
- prompt builders reject or ignore `content` fields.

Integration tests:

- backfill indexes a fixture article into mocked or test Qdrant and writes chunk mappings.
- retrieval by `raw_article_id` returns only that article's chunks.
- retrieval with `as_of` excludes future chunks.
- chat raw article path works without selecting `raw_articles.content`.
- summarizer prompt contains evidence chunks but not full body labels.

Operational checks:

- count articles with content but no indexed chunks.
- count indexed chunks per source type.
- sample retrieval precision by source id and query.
- Qdrant payload size distribution.
- prompt token usage before/after migration.

## Open Decisions

1. Whether Qdrant payload may contain bounded chunk text.
   - Recommended: yes. Without text payload, another object store is needed to fetch evidence text after vector search.

2. Whether to reuse `axis_documents`.
   - Recommended: no. Create `axis_raw_content` to separate raw evidence retrieval from derived assistant knowledge.

3. When to null `raw_articles.content`.
   - Recommended: after reader migration and backfill verification, not during initial indexing.

4. How to repair legacy cards without source ids.
   - Recommended: match by `source_articles.url`, source title, and generated date; unresolved rows should be flagged instead of guessed silently.
