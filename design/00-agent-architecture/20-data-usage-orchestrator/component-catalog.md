# DataUsageOrchestrator Component Catalog

상태: Draft v0.1

## Reasoning Agents

### MixerAgent

목적: 여러 카드/이슈를 비교하고 종합한다.

call input:

- card ids
- comparison ratio/options
- user context

DB read:

- `card_news`
- `evidence_payload`
- related raw sources

return output:

- comparison result
- synthesized insight
- source/evidence refs

DB write:

- optional analysis ledger / usage log

does_not:

- 새 raw 데이터를 수집하지 않는다.
- 카드뉴스를 새로 생성하지 않는다.

failure policy:

- 일부 card load 실패 시 partial comparison.

### InsightAgent

목적: 저장된 카드와 trend/profile context에서 흐름 단위 인사이트를 만든다.

call input:

- peer/sector/event filters
- time range
- optional selected cards

DB read:

- `card_news`
- `peer_companies`
- trend context storage
- raw metric/signal tables

return output:

- insight narrative
- supporting cards
- evidence refs

DB write:

- optional insight cache or report payload

does_not:

- 1건 raw article의 통합 이슈를 새로 만들지 않는다.

failure policy:

- insufficient evidence면 empty insight + reason.

### ReportAgent

목적: 저장된 카드/인사이트/검색 결과를 보고서 구조로 조립하고 narrative를 생성한다.

call input:

- report type
- sections
- selected cards or filters
- audience context

DB read:

- `card_news`
- trend context storage
- profile context
- optional search results

return output:

- report outline
- sections
- source refs

DB write:

- report history if product requires.

does_not:

- briefing delivery scheduling을 담당하지 않는다.

failure policy:

- section 단위 partial generation.

### ChatbotAgent

목적: 저장소 기반 Q&A를 수행한다.

call input:

- user message
- session context
- filters

DB read:

- raw/card/profile/trend storage
- Qdrant search index

return output:

- answer
- retrieved sources
- related cards
- confidence/evidence metadata

DB write:

- optional chat session/history.

does_not:

- 출처 없는 답변을 확정적으로 말하지 않는다.

failure policy:

- retrieval 실패 시 "근거 부족" 응답.

## Generators

### BriefingGenerator

target name for legacy `BriefingGenerationAgent`.

목적: 사용자/조직별 briefing 문서를 생성한다.

DB read:

- `card_news`
- profile/trend context
- user preferences

DB write:

- briefing history / delivery log

failure policy:

- section별 partial generation, delivery 전 validation.

### AnswerGenerator

target name for legacy `AnswerAgent`.

목적: retrieved/reranked evidence를 사용자 답변으로 생성한다.

DB read:

- direct DB read 최소화. `HybridSearchService`/`RerankService` 결과를 우선 사용.

failure policy:

- evidence 없으면 answer abstain.

## Services

| Component | Purpose | DB read | DB write |
|---|---|---|---|
| `HybridSearchService` | BM25/vector/hybrid retrieval | raw/card/Qdrant | None |
| `RerankService` | retrieved hits 재정렬 | optional model/index | None |
| `KeywordExtractionService` | keyword extraction/cache | raw/card | optional cache |
| `SearchSuggestService` | prefix suggestion | keyword/cache/log | optional cache |
| `DerivedMetricsService` | dashboard/overview metric 계산 | raw/card/metrics | optional cache |

Planned code path: `src/services/data_usage/`.

## Builders

| Component | Purpose | DB read | DB write |
|---|---|---|---|
| `KeywordGraphBuilder` | keyword graph artifact 생성 | card/raw/keyword | optional graph cache |
| `PeerWordCloudBuilder` | peer별 word cloud artifact 생성 | keyword/card/raw | optional cache |
| `EvidenceBuilder` | evidence payload/source refs 조립 | raw/card/metrics | evidence payload target |

Planned ingestion/common builder path: `src/services/ingestion/`.

## Open Questions

- `EvidenceBuilder`는 1단계 card writer 내부 책임인지, 2단계 usage 공통 builder인지.
- `DerivedMetricsService` cache table 필요 여부.
- `KeywordGraphBuilder` 기존 `KeywordGraphAgent` import alias 유지 기간.
