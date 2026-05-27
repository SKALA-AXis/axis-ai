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
- `IntegratedIssue v2` 계약으로 `source_family`, `scope_type`, `claim_ledger`,
  `evidence_ledger`, `quality`, `source_coverage`를 남긴다.
- `raw_articles.id`는 `raw_article_id`로 보존하고, fact/evidence/claim/coverage
  전 구간에서 `raw_article_ids`를 함께 남긴다.
- 원문 본문은 `content_digest`로 압축해 남긴다.
  이 값은 뉴스/DART/IR/리포트 본문을 다음 `AnalysisAgent`가 읽을 수 있게 만든
  내용 설명 레이어이며, 증거 채택용 `claim_ledger`와 분리한다.
  - `content_digest`: bundle 통합 `summary`, `detailed_explanation`,
    `key_points`, `body_extracts`, `sections`, `sources`를 남긴다. 하나의
    `content_digest`는 여러 raw article 단위 source digest를 `sources`에
    포함할 수 있다.
  - `content_digest.sources`: source별 `source_index`, `id`, `title`, `source`,
    `summary`, `key_points`, `body_extracts`, `url`, `published_at`, `publisher`를
    남기는 source별 축약 view.
  - `content_digest_storage`: DB 저장을 위해 `content_summary`,
    `content_detailed_explanation`, `content_sources`, `content_sections`,
    `content_body_extracts`, `content_key_points`, `content_raw_article_ids` 등으로
    투영한 값이다.
  - `key_points`는 원문에서 뽑은 단일 핵심 내용이고, `sections`는 key point를
    parser section 또는 의미 카테고리/주요 키워드 기준으로 묶어 `summary`,
    `detailed_explanation`, `body_extracts`, `keywords`, `numbers_and_dates`,
    `source_count`를 제공하는 단위다.
- `source_map`은 `raw_article_id -> source_index`를 고정한다. 이후
  `CardNewsAgent`가 citation `[1]`과 출처 drawer를 만들 때 이 값을 기준으로
  원문 링크를 연결한다.
- `issue_frame`은 카드뉴스 문장이 아니라 정규화된 분석 재료다. `companies`,
  `sectors`, `event`, `topics`, `entities`, `key_numbers`, `timeline`,
  `relations`, `quality`를 rule-based로 만들고, 모든 항목은 가능한 경우
  `raw_article_ids`와 `source_indexes`를 남긴다.
- `processing_status`, `crawl_status`, `relevance_label`, `relevance_score`,
  `relevance_reason`은 source row quality로 보존한다. `SKIPPED` 또는
  `irrelevant` row만 있는 입력은 `is_valid_summary=false`로 처리해 human review
  route를 타게 한다.
- fact 선택은 단순 순번/개수 절단이 아니라 score + source/category coverage +
  prompt budget 정책으로 수행한다.

does_not:

- SK AX 시사점이나 대응 방향을 만들지 않는다.
- source_type 하나만으로 문서 유형을 단순 매핑하지 않는다. source_type,
  content_type, source_name, metadata, parser section 신호를 함께 사용한다.

failure policy:

- retry 대상.
- 실패 시 invalid integrated issue로 human review route를 타게 한다.

runtime modules:

- facade: `src/agents/issue_integration_agent.py`
- services:
  - `src/services/issue_integration/source_profile.py`
  - `src/services/issue_integration/fact_extractor.py`
  - `src/services/issue_integration/fact_ranker.py`
  - `src/services/issue_integration/evidence_ledger.py`
  - `src/services/issue_integration/source_map.py`
  - `src/services/issue_integration/issue_frame.py`
  - `src/services/issue_integration/issue_composer.py`
  - `src/services/issue_integration/quality.py`
  - `src/services/issue_integration/policy.py`
  - `src/services/issue_integration/lexicon.py`

`policy.py`는 token/char budget, quality weight, provenance quality 같은 조정값을
한곳에 모은다. 새 고정값이 필요하면 이 파일에 추가하고, 왜 필요한지 주석으로
근거를 남긴다.
`lexicon.py`는 source/profile/fact type 판정용 의미 단서를 모은다. 운영 샘플
오류 분석 결과가 생기면 이 목록을 교체한다.

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
