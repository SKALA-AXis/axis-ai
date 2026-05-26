# AnalysisGraphRunner Node Contracts

상태: Draft v0.1

## issue_integrate

calls: `IssueIntegrationAgent`

input:

- `input_bundle`

output:

- `integrated_issue`
- `stage_outputs.issue_integration`

does:

- 뉴스/뉴스룸 cluster는 `SourceSummarizer`로 fact summary를 만든다.
- DART/IR/리포트는 parser_result, metrics, business signals를 통합한다.
- main company, mentioned peer, facts, key numbers, evidence basis를 확정한다.

does_not:

- SK AX 시사점이나 대응 방향을 만들지 않는다.

failure policy:

- retry 대상.
- 실패 시 invalid integrated issue로 human review route를 타게 한다.

## profile_context

calls: `ProfileContextLoader`

input:

- `input_bundle`
- `integrated_issue`

output:

- `profile_context`

does:

- `integrated_issue.main_company` 우선으로 peer profile을 로드한다.
- `integrated_issue.mentioned_peer_companies`와 `input_bundle.companies`를 fallback으로 쓴다.
- 미리 저장된 profile snapshot과 최근 metric/signal을 조립한다.

does_not:

- profile snapshot을 새로 생성하지 않는다.

failure policy:

- primary loader 실패 시 legacy fallback.
- 최종 실패 시 empty context 허용.

## build_analysis_context

calls: `AnalysisContextBuilder`

input:

- `input_bundle`
- `integrated_issue`
- `profile_context`

output:

- `analysis_context`

does:

- timeline, capability, sector pulse, financial trend, similar cards를 compact context로 만든다.
- token budget을 지킨다.

does_not:

- LLM을 호출하지 않는다.
- 새로운 profile/trend snapshot을 만들지 않는다.

failure policy:

- layer별 fallback.
- 전체 실패 시 empty `AnalysisContext`.

## strategic_analyze

calls: `AnalysisAgent`

input:

- `integrated_issue`
- `classification`
- `cluster_metadata`
- `analysis_context`

output:

- `analysis`
- `stage_outputs.content_analysis`

does:

- source family별 분석 렌즈를 선택한다.
- `content_analysis`, `detailed_findings`, `evidence_map`, `uncertainty_notes`, `handoff`를 만든다.
- 기존 `analysis_summary`, `strategic_meaning`, `impact_level` 계약도 유지한다.

does_not:

- SK AX가 해야 할 일을 권고하지 않는다.
- 입력에 없는 수치/고객/제품/날짜를 만들지 않는다.

failure policy:

- retry 대상.
- 실패 시 `is_valid_analysis=false`로 human review route.

## implication

calls: `ImplicationAgent`

input:

- `input_bundle`
- `integrated_issue`
- `analysis`
- `profile_context`
- `analysis_context`
- `classification`

output:

- `implication`
- `stage_outputs.skax_implication`

does:

- SK AX 관점의 기회, 위협, 중요도, recommended actions를 만든다.
- evidence id와 provenance를 유지한다.

does_not:

- source fact를 새로 만들지 않는다.

failure policy:

- retry 대상.
- LLM 실패 시 heuristic fallback.

## validate

calls:

- hard validation
- `EvaluatorAgent`

input:

- `integrated_issue`
- `analysis`
- `implication`
- `evidence_payload`
- `sources`
- `analysis_context`

output:

- `validation`

branch:

- pass -> `assemble`
- fail -> `human_review`

does:

- 출처 없는 수치를 차단한다.
- evidence chain 누락을 차단한다.
- confidence, specificity, actionability 등 soft metric을 남긴다.

does_not:

- 카드뉴스 payload를 만들지 않는다.

failure policy:

- rule-based step이므로 retry 없음.
- hard fail은 human review.

## assemble

input:

- `input_bundle`
- `integrated_issue`
- `analysis`
- `implication`
- `validation`
- `evidence_payload`
- `classification`

output:

- `analysis_package`

does:

- downstream card writer가 읽을 단일 package로 묶는다.

does_not:

- DB write를 하지 않는다.

failure policy:

- fail-fast. package assembly 오류는 graph error로 기록한다.

## card_writer

calls:

- `CardNewsAgent.generate_from_analysis_package`
- `save_card_news`

input:

- `analysis_package`
- `classification`

output:

- `card_news_payload`
- `card_news_id`

does:

- 카드뉴스 payload를 생성한다.
- `peer_company_id`, `primary_keyword_category`, `source_raw_article_ids`, `evidence_payload`를 보강한다.
- `card_news`에 저장한다.

does_not:

- validation fail 결과를 저장하지 않는다.

failure policy:

- DB transient 실패는 retry.
- 최종 실패 시 `card_news_id=None`.
