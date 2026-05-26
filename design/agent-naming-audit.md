# Agent Naming Audit

작성일: 2026-05-26

이 문서는 `Agent`로 계획되었지만 실제 구현에서는 서비스/빌더/잡이 더 적절한 컴포넌트의 명칭과 구현 상태를 정리한다.

| 기존/검토명 | 유지/변경명 | 실제 코드 상태 | 기준 파일 |
|---|---|---|---|
| SectorPulseAggregator / SectorPulseAggregatorAgent | SectorPulseAggregator 또는 SectorPulseRefreshJob | 코드 미구현. `sector_pulse` MV와 K8s CronJob manifest만 있음 | `axis-backend/.../V33__context_engineering.sql`, `axis-infra/k8s/base/cronjob-sector-pulse.yaml` |
| EventChainDiscoveryAgent | EventChainDiscoveryJob | 미구현 | 계획만 존재 |
| BriefingGenerationAgent | BriefingGenerationService | 미구현. `/gen-search`와 별개로 user briefing 생성 서비스는 fixture/계획 상태 | `src/pipeline/delivery_graph.py`는 daily delivery body builder |
| HybridSearchAgent | HybridSearchService | 함수형 구현 | `src/rag/hybrid_search.py` |
| RerankAgent | RerankService | 함수형 구현 | `src/rag/reranker.py` |
| AnswerAgent | AnswerService | 미구현. `ChatbotAgent.answer()`는 structure-only stub, `/gen-search`도 TODO | `src/agents/chatbot_agent.py`, `src/api/router.py` |
| KeywordGraphAgent / KeywordGraphBuilderAgent | KeywordGraphBuilder | structure-only stub은 있음. 실제 builder 로직은 미구현 | `src/agents/keyword_graph_agent.py` |
| KeywordExtractionAgent | KeywordExtractionService | 미구현 | 계획만 존재 |
| SearchSuggestAgent | SearchSuggestService | 미구현 | 계획만 존재 |
| PeerWordCloudAgent | PeerWordCloudBuilder | 미구현 | 계획만 존재 |
| DerivedMetricsAgent | DerivedMetricsService | 미구현 | 계획만 존재 |
| EvidenceAgent | EvidenceBuilder | 미구현. 독립 evidence builder 없음 | `card_news` 저장/조회 경로 일부만 존재 |
| FinancialLinkerAgent | FinancialLinkerService | 미구현. financial metric 추출/조회 기반만 있음 | `raw_article_financial_metrics`, `peer_financial_trend` |
| EmbedIndexAgent | EmbedIndexService | 함수형 구현 | `src/rag/embedder.py`, `src/rag/vector_index.py` |
| RelevanceAgent | RelevanceService | 구현됨. Agent 클래스가 아니라 preprocessing service/evaluator | `src/preprocessing/relevance.py` |
| DedupAgent | DedupService | 구현됨. Agent 클래스가 아니라 preprocessing deduplicator | `src/preprocessing/dedup.py` |
| ClassificationAgent | ClassificationService | 구현됨. Agent 클래스가 아니라 preprocessing classifier | `src/preprocessing/classification.py` |
| ParserAgent | ParserService | 구현됨. Agent 클래스가 아니라 parser router + parser modules | `src/parsers/parser_router.py`, `src/parsers/*_parser.py` |
| CrawlerAgent | CrawlerJob 또는 CrawlerService | 구현됨. 단일 Agent 클래스가 아니라 scheduler/batch/source crawler 구조 | `src/crawler/scheduler.py`, `src/crawler/batch_processor.py`, `src/crawler/sources/*.py` |

## Design 반영 원칙

- LLM reasoning 주체만 `Agent` 명칭을 유지한다.
- DB 조회/전처리/검색/인덱싱처럼 deterministic 하거나 외부 모델 래퍼인 컴포넌트는 `Service`를 쓴다.
- 캐시/그래프/evidence처럼 산출물을 조립하는 컴포넌트는 `Builder`를 쓴다.
- CronJob 또는 batch entrypoint 성격은 `Job`을 쓴다.
