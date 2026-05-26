# Open Questions

초안 기반으로 다음 라운드에서 결정해야 할 질문이다. 결정되면 각 대표 문서로 반영한다.

## Architecture

1. `InsightAgent`는 1단계 AnalysisGraphRunner 내부 node인가, 2단계 DataUsageOrchestrator 전용인가?
2. `ReportAgent`와 `BriefingGenerator`의 경계는 무엇인가?
3. `CardNewsAgent`가 계속 1단계 마지막 node인가, 카드 생성도 2단계 usage로 분리할 가능성이 있는가?

## Context Storage

1. `global_industry_trends`는 신규 테이블이 필요한가?
2. `TrendContext`는 카드 생성에 항상 주입할 것인가, 글로벌/산업 자료에서만 optional로 쓸 것인가?
3. `peer_companies.profile_snapshot` 계열 컬럼은 현재 DB에 맞춰 어떤 이름으로 확정할 것인가?

## Naming

1. `SectorPulseAggregator`와 `SectorPulseRefreshJob` 중 어떤 이름을 canonical로 둘 것인가?
2. `EvidenceBuilder`와 `EvidenceService` 중 어떤 이름이 더 맞는가?
3. `AnswerGenerator`와 `AnswerService` 중 어떤 이름이 API surface에 더 적합한가?

## Evaluation

1. `EvaluatorAgent`가 validate node 내부에서 동기 실행되어야 하는가?
2. LLM Judge는 카드 저장 전 hard gate인가, 저장 후 sidecar 평가인가?
3. `human_review` 라우팅 결과를 어떤 테이블/컬럼에 저장할 것인가?

## Implementation Sequence

1. 먼저 문서 구조만 확정할 것인가, code package 구조도 함께 바꿀 것인가?
2. legacy class alias 유지 기간은 어느 정도가 적당한가?
3. `DataAnalysisSupervisorAgent` import를 언제부터 `AnalysisGraphRunner`로 교체할 것인가?
