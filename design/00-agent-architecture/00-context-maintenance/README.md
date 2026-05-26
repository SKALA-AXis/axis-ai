# 0단계 Context Maintenance Layer

상태: Draft v0.1

## Purpose

Context Maintenance Layer는 1단계 분석 graph가 매 실행마다 과거 raw 데이터를 길게 읽지 않도록,
피어사와 산업 흐름 context를 주기적으로 압축·갱신한다.

핵심 산출물:

- `ProfileContext`: SK AX + Peer + Sector runtime context
- `peer_companies.profile_snapshot`: 회사별 전략 프로필 snapshot
- `peer_companies.peer_plus_payload["capability_evolution"]`: capability 변화 context
- `TrendContext` 또는 `global_industry_trends`: 글로벌/산업 IT 흐름 context

## Components

```text
ProfileAgent
  raw_articles + DART/IR + official newsroom + metrics/signals
  -> peer_companies.profile_snapshot

CapabilityEvolutionAgent
  raw_article_business_signals
  -> peer_companies.peer_plus_payload["capability_evolution"]

ITTrendAgent
  SPRi / BCG / Gartner / market report / global newsroom analysis
  -> TrendContext / global_industry_trends

ProfileContextLoader
  companies + sectors + event_type + IntegratedIssue
  -> ProfileContext
```

## Layer Boundary

Does:

- 주기적으로 profile/trend/capability context를 갱신한다.
- AnalysisGraphRunner가 사용할 compact context를 만든다.
- DB와 Qdrant에서 필요한 최근 신호를 읽어 context density를 보강한다.

Does not:

- 카드뉴스를 생성하지 않는다.
- 개별 이슈의 SK AX 대응 방향을 생성하지 않는다.
- raw article의 relevance/dedup/classification을 수행하지 않는다.

## Runtime Relationship

```text
AnalysisGraphRunner.issue_integrate
  -> IntegratedIssue.main_company / mentioned_peer_companies 확정
  -> ProfileContextLoader
  -> AnalysisContextBuilder
  -> AnalysisAgent / ImplicationAgent
```

## Code Mapping

| 설계명 | 현재 코드 |
|---|---|
| `ProfileAgent` | `src/agents/profile_agent.py` |
| `ProfileContextLoader` | `src/services/profile_context_loader.py` |
| `SKAXProfileLoader` | `src/services/skax_profile_context_loader.py` |
| `CapabilityEvolutionAgent` | `src/agents/context/capability_evolution_agent.py` |
| `ITTrendAgent` | `src/agents/it_trend_agent.py` |
| `AnalysisContextBuilder` | `src/services/analysis_context_builder.py` |
| `SectorPulseRefreshJob` | `src/jobs/context/sector_pulse_refresh_job.py` (추후 개발) |
| `EventChainDiscoveryJob` | `src/jobs/context/event_chain_discovery_job.py` (추후 개발) |
