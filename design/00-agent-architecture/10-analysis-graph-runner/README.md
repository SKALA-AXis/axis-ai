# 1단계 AnalysisGraphRunner

상태: Draft v0.1  
현재 코드: `src/agents/analysis_supervisor_agent.py`, `src/pipeline/analysis_flow_graph.py`

## Purpose

수집/정제된 데이터에서 만들어진 `AnalysisInputBundle`을 입력으로 받아,
이슈 통합, context 구성, 분석, 시사점, 검증, 카드뉴스 저장까지 실행한다.

기존 `DataAnalysisSupervisorAgent` 이름은 backward compatibility로 유지하고,
canonical name은 `AnalysisGraphRunner`다.

## Flow

```text
AnalysisInputBundle
  -> issue_integrate
  -> profile_context
  -> build_analysis_context
  -> strategic_analyze
  -> implication
  -> validate
     -> pass: assemble -> card_writer -> save_card_news
     -> fail: human_review
```

## Runtime State

Canonical state name: `AnalysisFlowState`

주요 필드:

- `input_bundle`
- `classification`
- `integrated_issue`
- `profile_context`
- `analysis_context`
- `analysis`
- `implication`
- `validation`
- `analysis_package`
- `card_news_payload`
- `card_news_id`
- `stage_outputs`
- `human_review_flags`

`stage_outputs`는 향후 summary/insight/SK AX response node 확장을 위한 공통 계약이다.

```json
{
  "issue_integration": {},
  "content_analysis": {},
  "skax_implication": {}
}
```

## Entry Points

| Entry | Source | Use case |
|---|---|---|
| `AnalysisPipelineRunner.run_cluster()` | `raw_articles.cluster_id` | 뉴스 / 뉴스룸 cluster 분석 |
| `AnalysisPipelineRunner.run_raw_article()` | `raw_articles.id` | DART / IR / 리포트 / 단일 문서 분석 |
| `AnalysisPipelineRunner.run_input_bundle()` | caller-provided bundle | 테스트 / 재처리 / future orchestrator |

## Does

- `AnalysisInputBundle` 1건을 graph에 흘린다.
- 각 node output을 state에 저장한다.
- validation을 통과한 결과만 card writer로 보낸다.
- validation 실패는 human review route로 보낸다.

## Does Not

- raw data를 수집하지 않는다.
- profile snapshot을 새로 생성하지 않는다.
- 저장된 여러 카드의 비교/리포트/챗봇 답변을 만들지 않는다.

## Future Nodes

아래 node는 아직 확정 전이다. 추가 시 `stage_outputs`에 자기 산출물을 남기고,
기존 validation/evidence payload를 재사용한다.

```text
content_analysis
  -> summary_agent
  -> insight_agent
  -> skax_response_agent
  -> validate
```

## Code Mapping

| 설계명 | 현재 코드 |
|---|---|
| `AnalysisGraphRunner` | `src/agents/analysis_supervisor_agent.py` |
| `AnalysisFlowState` | `src/pipeline/analysis_flow_graph.py`의 `SupervisorState` alias |
| `AnalysisPipelineRunner` | `src/pipeline/analysis_pipeline.py` |
| `AnalysisDeliveryService` | `src/pipeline/analysis_delivery.py` |
| `IssueIntegrationAgent` | `src/agents/issue_integration_agent.py` |
| `AnalysisAgent` | `src/agents/analysis_agent.py`, `src/analysis/analyzer.py` |
| `ImplicationAgent` | `src/agents/implication_agent.py` |
| `CardNewsAgent` | `src/agents/card_news_agent.py` |
| `SummaryAgent` | `src/agents/analysis_flow/summary_agent.py` (추후 개발) |
| `AnalysisFlowInsightAgent` | `src/agents/analysis_flow/insight_agent.py` (추후 개발) |
| `SkaxResponseAgent` | `src/agents/analysis_flow/skax_response_agent.py` (추후 개발) |
