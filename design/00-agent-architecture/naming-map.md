# Naming Map

상태: Draft v0.1  
원칙: 이름은 컴포넌트의 실제 책임과 실패 정책을 드러내야 한다.

## Naming Rule

| Suffix | 사용 기준 | 예시 |
|---|---|---|
| `Agent` | LLM reasoning 또는 여러 입력을 판단해 구조화된 결론을 만드는 컴포넌트 | `AnalysisAgent`, `ImplicationAgent`, `ProfileAgent` |
| `Runner` | fixed graph/pipeline 실행자. child node를 순서대로 실행하지만 직접 추론하지 않음 | `AnalysisGraphRunner` |
| `Orchestrator` | 저장된 결과를 task별 컴포넌트에 라우팅하는 coordinator | `DataUsageOrchestrator` |
| `Service` | deterministic business logic, search, parser, API helper, reusable operation | `HybridSearchService`, `ParserService` |
| `Builder` | 구조화 payload, graph, word cloud, evidence chain 등 artifact를 조립 | `KeywordGraphBuilder`, `EvidenceBuilder` |
| `Generator` | 사용자에게 노출되는 텍스트/문서/답변 생성 | `AnswerGenerator`, `BriefingGenerator` |
| `Job` | cron/batch/refresh/discovery 작업. 주기 실행이 본질 | `SectorPulseRefreshJob`, `EventChainDiscoveryJob` |

## Canonical Names

### 유지할 Agent

| 이름 | 이유 |
|---|---|
| `IssueIntegrationAgent` | 여러 source fact를 통합 이슈로 정리하는 LLM/판단 중심 책임 |
| `AnalysisAgent` | multi-source content analysis와 strategic meaning 판단 |
| `ImplicationAgent` | SK AX 관점 시사점 생성 |
| `ProfileAgent` | 회사별 strategic profile을 LLM으로 합성 |
| `CapabilityEvolutionAgent` | capability 변화 narrative를 합성하는 context agent |
| `ITTrendAgent` | 글로벌/산업 IT 흐름을 TrendContext로 합성 |
| `EvaluatorAgent` | 현재는 rule-based + 향후 LLM judge를 포함할 수 있어 Agent 유지 |
| `CardNewsAgent` | 카드뉴스 구조와 문구 생성 책임이 LLM 중심 |
| `MixerAgent` | 여러 카드/이슈를 비교·종합하는 reasoning agent |
| `InsightAgent` | 흐름 단위 인사이트 추론 |
| `ReportAgent` | 보고서 구조와 narrative 생성 |
| `ChatbotAgent` | 대화형 질의응답 reasoning |

### 이름 변경 대상

| 기존 이름 | 유지/변경명 | 비고 |
|---|---|---|
| `DataAnalysisSupervisorAgent` | `AnalysisGraphRunner` | 기존 클래스는 backward compatibility alias로 유지 |
| `SectorPulseAggregatorAgent` | `SectorPulseRefreshJob` 또는 `SectorPulseAggregator` | MV refresh 중심이면 Job, 집계 함수면 Aggregator |
| `EventChainDiscoveryAgent` | `EventChainDiscoveryService` 또는 `EventChainDiscoveryJob` | batch 실행이면 Job |
| `BriefingGenerationAgent` | `BriefingGenerator` 또는 `BriefingGenerationService` | 사용자-facing 생성이면 Generator |
| `HybridSearchAgent` | `HybridSearchService` | deterministic retrieval |
| `RerankAgent` | `RerankService` | reranker 호출 wrapper |
| `AnswerAgent` | `AnswerGenerator` 또는 `AnswerService` | 답변 생성이면 Generator |
| `KeywordGraphAgent` | `KeywordGraphBuilder` | graph artifact 조립 |
| `KeywordExtractionAgent` | `KeywordExtractionService` | extraction/caching 중심 |
| `SearchSuggestAgent` | `SearchSuggestService` | autocomplete service |
| `PeerWordCloudAgent` | `PeerWordCloudBuilder` | word cloud artifact 조립 |
| `DerivedMetricsAgent` | `DerivedMetricsService` | deterministic metric 계산 |
| `EvidenceAgent` | `EvidenceBuilder` 또는 `EvidenceService` | evidence payload 조립 |
| `FinancialLinkerAgent` | `FinancialLinkerService` | metric/source link deterministic matching |
| `EmbedIndexAgent` | `EmbedIndexService` | vector index write service |
| `RelevanceAgent` | `RelevanceService` | 전처리 relevance 판단. LLM 사용 여부와 별개로 pipeline service |
| `DedupAgent` | `DedupService` | clustering/dedup service |
| `ClassificationAgent` | `ClassificationService` | company/sector/event label service |
| `ParserAgent` | `ParserService` | source parser router |
| `CrawlerAgent` | `CrawlerService` 또는 `CrawlerJob` | crawler 실행 단위면 Job, reusable fetch면 Service |

## Code Migration Policy

1. 먼저 문서와 신규 import alias를 만든다.
2. 기존 public class/function은 최소 1 sprint 동안 alias로 유지한다.
3. API, cronjob, script entrypoint 이름은 운영 배포와 묶어서 별도 PR로 바꾼다.
4. DB 컬럼명 변경은 migration cost가 있으므로 이름만 바꾸지 않는다.
5. 테스트명은 canonical name 기준으로 새로 만들고, legacy 테스트는 점진 제거한다.
