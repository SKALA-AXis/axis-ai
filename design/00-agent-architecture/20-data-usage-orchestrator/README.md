# 2단계 DataUsageOrchestrator

상태: Draft v0.1  
현재 코드: `src/pipeline/data_usage_orchestrator.py`

## Purpose

1단계에서 저장된 raw/card/profile/trend 결과를 활용해 비교, 리포트, 인사이트,
챗봇, 키워드 그래프, 검색/답변을 제공한다.

DataUsageOrchestrator는 새 데이터를 수집하거나, 새로운 `AnalysisPackage`를 만들지 않는다.

## Inputs

- task type
- user query or request payload
- selected card ids
- peer/company filters
- sector/event filters
- time range
- user context

## DB Read

- `raw_articles`
- `raw_article_parse_results`
- `raw_article_financial_metrics`
- `raw_article_business_signals`
- `card_news`
- `card_news_articles`
- `evidence_chain`
- `peer_companies`
- trend context storage
- Qdrant index

## DB Write

Task별로 다르다.

- report/briefing history
- usage logs
- evaluation/feedback payload
- optional user session/chat history

## Routing

```text
DataUsageOrchestrator
  task=mixer / compare       -> MixerAgent
  task=insight               -> InsightAgent
  task=report                -> ReportAgent
  task=briefing              -> BriefingGenerator
  task=chatbot / qa          -> ChatbotAgent
  task=keyword_graph         -> KeywordGraphBuilder
  task=search                -> HybridSearchService -> RerankService -> AnswerGenerator
  task=keyword_extraction    -> KeywordExtractionService
  task=suggest               -> SearchSuggestService
  task=peer_word_cloud       -> PeerWordCloudBuilder
  task=derived_metrics       -> DerivedMetricsService
```

## Does

- 저장된 결과를 task별 컴포넌트에 전달한다.
- 검색/비교/리포트/인사이트/챗봇 요청의 공통 context를 만든다.
- source, related cards, trend context를 함께 넘긴다.

## Does Not

- raw article을 새로 분석하지 않는다.
- `IssueIntegrationAgent`, `AnalysisAgent`, `ImplicationAgent`를 직접 실행하지 않는다.
- 카드뉴스를 새로 저장하지 않는다. 단, report/briefing/chat history는 저장 가능하다.

## Failure Policy

- task별 컴포넌트 실패는 partial result와 error metadata를 반환한다.
- search/QA는 empty hits 또는 fallback answer를 허용한다.
- report/briefing 생성 실패는 저장하지 않고 retry 가능한 job result로 남긴다.

## Open Questions

- `InsightAgent`를 1단계 analysis graph에도 둘지.
- `ReportAgent`와 `BriefingGenerator`의 호출 관계.
- `KeywordGraphBuilder` 현재 코드명 `KeywordGraphAgent` alias 유지 기간.
