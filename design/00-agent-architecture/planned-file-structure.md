# Planned File Structure

상태: Draft v0.1  
목적: 추가 개발 예정 컴포넌트의 코드 파일 위치를 먼저 고정한다.

## AnalysisGraphRunner Future Stages

```text
src/agents/analysis_flow/
  summary_agent.py          # SummaryAgent, 추후 개발
  insight_agent.py          # InsightAgent / AnalysisFlowInsightAgent, 추후 개발
  skax_response_agent.py    # SkaxResponseAgent, 추후 개발
```

## Context Maintenance Jobs

```text
src/jobs/context/
  sector_pulse_refresh_job.py     # SectorPulseRefreshJob / SectorPulseAggregator, 추후 개발
  event_chain_discovery_job.py    # EventChainDiscoveryJob / EventChainDiscoveryService, 추후 개발
```

## DataUsageOrchestrator Services

```text
src/services/data_usage/
  hybrid_search_service.py        # HybridSearchService, 추후 개발
  rerank_service.py               # RerankService, 추후 개발
  answer_generator.py             # AnswerGenerator / AnswerService, 추후 개발
  briefing_generator.py           # BriefingGenerator / BriefingGenerationService, 추후 개발
  keyword_extraction_service.py   # KeywordExtractionService, 추후 개발
  search_suggest_service.py       # SearchSuggestService, 추후 개발
  peer_word_cloud_builder.py      # PeerWordCloudBuilder, 추후 개발
  derived_metrics_service.py      # DerivedMetricsService, 추후 개발
```

## Ingestion / Preprocessing Services

```text
src/services/ingestion/
  crawler_service.py              # CrawlerService / CrawlerJob, 추후 개발
  parser_service.py               # ParserService, 추후 개발
  relevance_service.py            # RelevanceService, 추후 개발
  dedup_service.py                # DedupService, 추후 개발
  classification_service.py       # ClassificationService, 추후 개발
  evidence_builder.py             # EvidenceBuilder / EvidenceService, 추후 개발
  financial_linker_service.py     # FinancialLinkerService, 추후 개발
  embed_index_service.py          # EmbedIndexService, 추후 개발
```

## Placeholder Policy

- placeholder는 운영 graph에 연결하지 않는다.
- class에는 `status = "planned"`, `note = "추후 개발"`을 둔다.
- method는 `NotImplementedError`를 던진다.
- 실제 구현 전까지 DB read/write를 수행하지 않는다.
- legacy 이름이 필요한 경우 alias class를 둔다.
