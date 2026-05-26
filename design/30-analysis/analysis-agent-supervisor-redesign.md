# AnalysisAgent / AnalysisSupervisor Redesign

작성일: 2026-05-26 KST  
대상: `src/agents/analysis_agent.py`, `src/analysis/analyzer.py`, `src/agents/analysis_supervisor_agent.py`, `src/pipeline/analysis_flow_graph.py`

## 1. 설계 개요

`analysis_agent`는 raw DB에 쌓인 자료를 분석하는 content analysis agent다. 입력 자료는 뉴스뿐 아니라 DART 공시, IR, 증권 리포트, 산업/글로벌 자료까지 포함하므로 뉴스 요약 프롬프트로 처리하면 안 된다.

새 설계의 핵심은 **Evidence-first Multi-source Document Intelligence**다.

1. `IssueIntegrationAgent`가 자료를 fact/number/signal/evidence 중심으로 통합한다.
2. `AnalysisAgent`가 출처 유형별 분석 렌즈를 고르고, claim ledger와 materiality matrix로 내용을 구조화한다.
3. `AnalysisSupervisor`는 단계별 산출물을 `stage_outputs`로 누적해 이후 요약, 시사점, SK AX 대응방향 agent가 같은 근거를 재사용하게 한다.

## 2. Agent 책임

### IssueIntegrationAgent

- raw article, parser_result, materialized metric/signal을 `AnalysisInputBundle`로 묶는다.
- 뉴스는 기존 cluster summarizer를 사용한다.
- DART/IR/리포트는 `document_chunks`, `topic_signals`, `sections`, `financial_metrics`, `business_signals`를 근거 fact로 승격한다.
- 해석이나 대응 방향은 만들지 않는다.

### AnalysisAgent

- 입력: `IntegratedIssue`, `classification`, `cluster_metadata`, `AnalysisContext`.
- 출력: 기존 호환 필드(`analysis_summary`, `strategic_meaning`, `impact_level`) + 신규 구조화 필드.
- SK AX 대응 권고는 만들지 않는다. 대응 agent가 사용할 관찰점만 `handoff.skax_response_agent_inputs`에 남긴다.

주요 출력 필드:

- `source_profile`: source family, source_type, analysis lens, materiality focus.
- `content_analysis`: 핵심 thesis, 전개, 전략 벡터, 재무/운영 readout, 리스크, 타이밍, 근거 공백.
- `detailed_findings`: strategy/financial/operation/market/technology/risk/governance 단위 세부 분석.
- `evidence_map`: 분석 claim과 fact/source id 연결.
- `uncertainty_notes`: 전망/계획/근거 부족 지점.
- `handoff`: 이후 summary/insight/SK AX response agent로 넘길 입력 후보.

### AnalysisSupervisor

현재는 fixed LangGraph DAG runner다. LLM-router형 동적 supervisor는 아니지만, supervisor 역할의 핵심인 stage orchestration과 quality gate는 담당한다.

현재 활성 flow:

```text
issue_integrate
  -> profile_context
  -> build_analysis_context
  -> strategic_analyze(content_analysis)
  -> implication
  -> validate
  -> assemble
  -> card_writer
```

향후 확장 flow:

```text
content_analysis
  -> summary_agent
  -> insight_agent
  -> skax_response_agent
  -> validate
  -> package/card_writer
```

## 3. 출처별 분석 렌즈

| Source family | 주요 자료 | 분석 초점 |
|---|---|---|
| `news` | 뉴스, 회사 보도자료, 글로벌 뉴스룸 | 사건, 주체, 고객/제품/계약, 일정, 노출 강도 |
| `filing` | DART, 공시 | 공시 항목, 수치, 사업 세그먼트 변화, 확정/예정 구분, 리스크 |
| `ir` | 실적 발표, IR PDF | 경영진 메시지, 실적 동인, 가이던스, 사업 우선순위 |
| `research` | 증권 리포트 | 애널리스트 주장, 밸류에이션/전망치, 근거와 가정 |
| `trend` | SPRi, BCG, 글로벌 산업 자료 | 산업 구조 변화, 기술/수요 신호, 지역/플랫폼 맥락 |
| `mixed` | 여러 유형 묶음 | claim ledger로 중복·상충·보강 관계 비교 |

## 4. 품질 원칙

- 분석 문장은 가능한 `fact_id`, `source_article_ids`, `evidence_text`와 연결한다.
- 입력에 없는 수치, 고객명, 제품명, 회사명, 날짜는 만들지 않는다.
- 전망/계획/가능성 표현은 확정 사실로 바꾸지 않는다.
- 근거가 약하면 `uncertainty_notes`와 낮은 `confidence`로 남긴다.
- SK AX 권고문은 implication/response 계층으로 미룬다.

## 5. 검토 및 보완 방법

1. Unit test: LLM을 fake로 대체해 `content_analysis`, `detailed_findings`, `evidence_map`, `handoff` 정규화가 유지되는지 확인한다.
2. Integration test: DART/IR parser_result가 `AnalysisInputBundle.facts`로 승격되는지 확인한다.
3. Quality gate: validate 노드가 출처 없는 수치와 provenance 누락을 차단하는지 유지한다.
4. Regression review: 카드 생성 결과의 `evidence_payload.analysis_package.stage_outputs`를 샘플링해 stage별 근거가 끊기지 않는지 본다.
5. Expansion review: summary/insight/SK AX response agent 추가 시 새 node는 `stage_outputs`에 자기 산출물을 쓰고, 기존 validation/evidence payload를 재사용한다.
