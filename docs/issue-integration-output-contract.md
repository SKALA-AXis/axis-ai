# IssueIntegrationAgent Output Contract

> 작성일: 2026-05-29  
> 대상 코드: `src/agents/issue_integration_agent.py`, `src/services/issue_integration/*`  
> 현재 schema: `integrated_issue_v3`

## 1. 목적

`IssueIntegrationAgent`는 raw DB row를 최종 요약문으로 작성하는 agent가 아니다. 이 agent의 목적은 뉴스, DART, IR, 리포트, 글로벌 자료 등 서로 다른 raw source를 후속 agent가 읽기 쉬운 **정규화된 분석 재료**로 바꾸는 것이다.

후속 agent는 이 output을 기반으로 각자 역할을 수행한다.

```text
IssueIntegrationAgent
  -> SummaryAgent
  -> AnalysisAgent
  -> ImplicationAgent
  -> ResponseDirectionAgent
  -> CardNewsAgent
  -> Mixer / KeywordGraph / Chatbot
```

`tests/output_issue_integration_cluster.json`은 테스트 산출물이다. 최상위에는 `db_input`과 `issue`가 있으며, 실제 통합 에이전트의 최종 output contract는 `issue` 객체와 동일하다.

```json
{
  "db_input": {},
  "issue": {
    "schema_version": "integrated_issue_v3"
  }
}
```

운영 코드에서는 `issue` 객체를 `integrated_issue`로 취급한다.

## 2. Top-Level Schema

`integrated_issue_v3`의 최상위 구조는 아래 순서를 기준으로 한다.

```json
{
  "schema_version": "integrated_issue_v3",
  "issue_brief": {},
  "analysis_ready_inputs": {},
  "content_digest": {},
  "issue_frame": {},
  "sources": [],
  "evidence": {},
  "quality": {},
  "metadata": {}
}
```

각 영역의 역할은 다르다. 후속 agent는 전체를 무조건 prompt에 넣기보다, 자기 역할에 맞는 영역만 선택해서 사용해야 한다.

| 필드 | 역할 | 주 사용자 |
|---|---|---|
| `issue_brief` | 이슈의 최소 식별 정보 | 모든 agent |
| `analysis_ready_inputs` | 공통 분석 착수 재료 | Summary, Analysis, Implication |
| `content_digest` | 본문 압축/추출 결과 | Summary, Analysis, CardNews |
| `issue_frame` | 구조화된 주체/섹터/이벤트/토픽 | Analysis, Implication, KeywordGraph |
| `sources` | 출처 목록 | CardNews, Chatbot, Evidence UI |
| `evidence` | fact와 근거 reference | 모든 LLM agent, 검증 |
| `quality` | 입력/추출 품질 신호 | Evaluator, Human Review |
| `metadata` | 실행/저장/추적 메타데이터 | Pipeline, Debug |

## 3. `issue_brief`

`issue_brief`는 후속 agent가 가장 먼저 읽어야 하는 헤더다.

```json
{
  "is_valid": true,
  "headline": "이슈 제목",
  "one_line_summary": "1문장 요약",
  "main_company": "lg_cns",
  "mentioned_peer_companies": ["lg_cns"],
  "event_type": "launch",
  "sectors": ["ax"],
  "source_family": "news",
  "scope_type": "peer_company",
  "analysis_scope": {
    "analyzed_source_ids": [1574]
  },
  "confidence": 0.7,
  "reason": ""
}
```

사용 원칙:

- `is_valid=false`면 downstream agent는 생성 품질을 낮추거나 human review로 보내야 한다.
- `main_company`, `event_type`, `sectors`는 분석 관점 선택에 사용한다.
- `analysis_scope.analyzed_source_ids`는 실제 분석 근거로 사용된 `raw_articles.id` 목록이다.
- `sources`에는 클러스터 전체 source가 있을 수 있으므로, 실제 분석 근거와 전체 출처를 혼동하지 않는다.

## 4. `analysis_ready_inputs`

`analysis_ready_inputs`는 agent별 지시문이 아니라, 후속 agent가 공통으로 참고할 수 있는 착수 재료다.

```json
{
  "core_question": "이 이슈가 보여주는 사업/시장 의미는 무엇인가?",
  "key_developments": [],
  "materiality_signals": [],
  "uncertainty_points": [],
  "suggested_sections": []
}
```

의미:

- `core_question`: 후속 분석이 답해야 할 중심 질문
- `key_developments`: 본문/fact에서 추출된 주요 전개
- `materiality_signals`: 수치, 사업 신호, 키워드, 중요 섹션
- `uncertainty_points`: 전망, 근거 부족, 불확실한 지점
- `suggested_sections`: 본문을 읽기 좋은 단위로 나눈 섹션 힌트

이 필드는 `handoff`를 대체한다. `handoff`처럼 "어떤 agent에게 무엇을 시켜라"가 아니라, 어떤 후속 agent도 재사용할 수 있는 분석 재료만 담는다.

## 5. `content_digest`

`content_digest`는 raw 본문을 그대로 넣지 않고, rule-based extractive compression으로 압축한 본문 재료다.

```json
{
  "summary": "본문 핵심 압축",
  "detailed_explanation": "세부 내용 설명",
  "key_points": [],
  "body_extracts": [],
  "sections": [],
  "basis_scope": "analysis_eligible_sources",
  "source_digest_count": 1,
  "section_count": 2,
  "has_content": true,
  "compression_method": "extractive_semantic_rank_v1"
}
```

현재 `content_digest`는 LLM이 새로 작성한 추상 요약이 아니다. 원문 title/content와 parser result에서 문장/청크를 추출하고, 아래 기준으로 랭킹한 뒤 압축한다.

- 제목과의 겹침
- 수치 포함 여부
- materiality keyword 포함 여부
- parser section 여부
- 본문 초반 위치
- 불확실성 표현 포함 여부

후속 agent 사용법:

- SummaryAgent: `summary`, `detailed_explanation`, `sections`, `key_points`를 우선 사용한다.
- AnalysisAgent: `detailed_explanation`, `sections`, `body_extracts`를 읽어 의미 분석을 한다.
- CardNewsAgent: `sections`와 `key_points`를 카드 문단 후보로 사용하되, 문장 작성은 자체 책임으로 한다.

주의:

- `content_digest`는 본문 압축 결과이므로 문장 자체를 유지한다.
- 모든 전략적 해석을 이 필드에서 기대하면 안 된다.
- 이 필드는 "본문이 무엇을 말했는가"에 가깝고, "그래서 SK AX가 무엇을 해야 하는가"는 후속 agent 책임이다.

## 6. `issue_frame`

`issue_frame`은 본문과 메타데이터에서 추출한 구조화 프레임이다.

주요 하위 필드:

```json
{
  "companies": {},
  "sectors": [],
  "event": {},
  "topics": [],
  "entities": [],
  "key_numbers": [],
  "timeline": [],
  "relations": [],
  "unresolved_candidates": {},
  "quality": {}
}
```

사용 원칙:

- `companies`: 회사 주체 판단에 사용한다.
- `sectors`: 섹터/도메인 context 선택에 사용한다.
- `event`: 이벤트 타입, trigger term, confidence 확인에 사용한다.
- `topics`: 키워드 그래프, 분석 소주제, 카드 섹션 후보에 사용한다.
- `key_numbers`: 수치 기반 분석과 검증에 사용한다.
- `entities`: 보조 색인/관계 그래프/검색용으로 사용한다.

`entities`는 분석 본문이라기보다 보조 구조화 정보다. 카드뉴스나 전략 분석의 핵심 입력으로 과도하게 사용하지 않는다.

## 7. `sources`

`sources`는 화면 표시, citation, 출처 링크, 챗봇 검색에 사용하는 최소 출처 목록이다.

```json
[
  {
    "id": 1574,
    "title": "기사 제목",
    "source_name": "naver_news",
    "source_type": "news",
    "publisher": "매체명",
    "published_at": "2026-05-07T05:06:00+00:00",
    "url": "https://...",
    "relevance_label": "relevant",
    "relevance_score": 0.82
  }
]
```

사용 원칙:

- `id`는 `raw_articles.id`다.
- `sources`는 전체 클러스터 출처를 담을 수 있다.
- 실제 분석에 사용된 source는 `issue_brief.analysis_scope.analyzed_source_ids`를 확인한다.
- 카드뉴스 UI의 출처 목록은 이 필드를 기준으로 표시한다.

## 8. `evidence`

`evidence`는 fact와 근거를 관리한다. 반복되는 긴 근거 문장은 `references`에 한 번만 저장하고, fact나 frame에서는 `evidence_ref_id`로 참조한다.

```json
{
  "references": [
    {
      "id": "ev1",
      "text": "근거 문장",
      "source_ids": [1574]
    }
  ],
  "by_section": [
    {
      "section": "main_event",
      "title": "핵심 사건",
      "facts": [
        {
          "fact_id": "c1574_a1574_f1",
          "fact": "핵심 fact",
          "source_ids": [1574],
          "section": "main_event",
          "fact_type": "general_fact",
          "summary_role": "main_event",
          "evidence_ref_id": "ev1",
          "confidence": "high"
        }
      ]
    }
  ],
  "claims": [],
  "uncertain_points": []
}
```

후속 agent 사용법:

- LLM agent는 claim을 만들 때 `facts[].fact`를 우선 사용한다.
- 근거 문장이 필요하면 `facts[].evidence_ref_id`로 `references[].text`를 찾아 사용한다.
- `source_ids`는 출처 연결과 evidence chain 저장에 사용한다.
- `uncertain_points`는 과도한 확정 표현을 피하는 데 사용한다.

reference 역참조 예시:

```python
refs = {item["id"]: item["text"] for item in issue["evidence"]["references"]}
for section in issue["evidence"]["by_section"]:
    for fact in section["facts"]:
        evidence_text = refs.get(fact.get("evidence_ref_id"))
```

## 9. `quality`

`quality`는 이 output을 후속 agent가 얼마나 신뢰해도 되는지 판단하는 품질 신호다.

주요 필드:

- `confidence`
- `eligible_source_count`
- `skipped_source_count`
- `irrelevant_source_count`
- `crawl_error_source_count`
- `selected_fact_count`
- `evidence_count`
- `review_flags`
- `selection`
- `source_row_statuses`

사용 원칙:

- `review_flags`가 있으면 Evaluator 또는 Human Review 조건으로 사용한다.
- `eligible_source_count=0`이면 실질 분석 근거가 없는 상태다.
- `contains_irrelevant_source_rows`, `no_analysis_eligible_source_rows`는 downstream 생성 품질을 낮춰야 한다.

현재 입력 eligibility는 allowlist가 아니라 denylist다.

```text
excluded:
- processing_status = SKIPPED
- relevance_label = irrelevant
- crawl_status not in success/ok/completed
- error_message exists
```

## 10. `metadata`

`metadata`는 pipeline 추적과 저장용 메타데이터다.

후속 LLM prompt에는 기본적으로 넣지 않는다. 디버깅, 저장, lineage 추적, batch 실행 확인에 사용한다.

예:

- `cluster_id`
- `representative_id`
- `bundle_id`
- `issue_component`
- `integration_component`
- `issue_source_type`
- `model`
- `integrated_at`

## 11. Agent별 권장 입력

### SummaryAgent

우선 사용:

- `issue_brief`
- `content_digest.summary`
- `content_digest.detailed_explanation`
- `content_digest.sections`
- `evidence.by_section`
- `evidence.references`
- `sources`

하지 말 것:

- raw DB를 다시 읽어 동일 요약을 만들지 않는다.
- `analysis_ready_inputs.core_question`만 보고 요약하지 않는다.

### AnalysisAgent

우선 사용:

- `issue_brief`
- `analysis_ready_inputs`
- `content_digest`
- `issue_frame.companies`
- `issue_frame.sectors`
- `issue_frame.event`
- `issue_frame.topics`
- `issue_frame.key_numbers`
- `evidence`

하지 말 것:

- evidence에 없는 사실을 만들지 않는다.
- SK AX 대응 방향까지 작성하지 않는다.

### ImplicationAgent

우선 사용:

- `issue_brief`
- `analysis` 결과
- `issue_frame.companies`
- `issue_frame.sectors`
- `evidence.by_section`
- `quality.review_flags`

하지 말 것:

- 원문 요약을 다시 작성하지 않는다.
- 분석 agent가 만든 의미와 evidence를 무시하지 않는다.

### ResponseDirectionAgent

우선 사용:

- `issue_brief`
- `analysis`
- `implication`
- SK AX profile/context
- peer profile/context
- `quality`

하지 말 것:

- 통합 에이전트 output만으로 직접 대응 전략을 확정하지 않는다.

### CardNewsAgent

우선 사용:

- `issue_brief.headline`
- `content_digest.sections`
- `content_digest.key_points`
- `evidence.by_section`
- `sources`

하지 말 것:

- `sources`의 모든 기사가 분석에 사용되었다고 표현하지 않는다.
- 출처 없는 문장을 카드에 넣지 않는다.

### KeywordGraph / Mixer / Chatbot

우선 사용:

- `issue_frame.topics`
- `issue_frame.entities`
- `issue_frame.relations`
- `issue_frame.key_numbers`
- `sources`
- `evidence.references`

## 12. Prompt 구성 권장 순서

LLM agent에 넘길 때는 전체 JSON을 그대로 넣기보다 아래 순서를 권장한다.

```text
1. issue_brief
2. analysis_ready_inputs
3. content_digest.summary / sections / detailed_explanation
4. issue_frame 핵심 subset
5. evidence.by_section facts
6. 필요한 evidence.references
7. quality flags
8. sources
```

역할별로 prompt projection이 필요하면 `src/services/issue_integration/agent_views.py`를 사용한다.

## 13. DB 저장 매핑

DBML 기준 문서:

```text
docs/axis_ai_schema.dbml
```

Flyway 마이그레이션:

```text
axis-backend/src/main/resources/db/migration/V40__add_integrated_issue_storage.sql
```

저장 구조는 아래 원칙을 따른다.

| output 영역 | 저장 위치 | 이유 |
|---|---|---|
| 전체 `issue` 객체 | `integrated_issues.payload` | 재처리, 디버깅, schema evolution 대비 |
| `issue_brief` 핵심값 | `integrated_issues` scalar columns | 목록 조회, 필터링, 카드 생성 대상 선택 |
| `analysis_ready_inputs` | `integrated_issues.analysis_ready_inputs` | 후속 agent 공통 입력 보존 |
| `content_digest` 전체 | `integrated_issues.content_digest` | 원본 구조 보존 |
| `content_digest.summary/details/count` | `integrated_issues.content_*` columns | 리스트/검색/품질 조회 |
| `content_digest.sections` | `integrated_issue_content_sections` | 섹션 단위 검색, 카드/요약 agent 입력 |
| `sources` | `integrated_issue_source_articles` + `integrated_issues.sources` | raw_articles FK와 원본 source payload 동시 보존 |
| `evidence.references` | `integrated_issue_evidence_references` | 중복 evidence text 제거, ref 기반 재사용 |
| `issue_frame` | `integrated_issues.issue_frame` | 구조화 분석 재료 보존 |
| `quality`, `metadata` | `integrated_issues.quality`, `integrated_issues.metadata` | 검증/운영 추적 |

`card_news`에는 `integrated_issue_id`를 nullable FK로 둔다. 카드뉴스는 통합 이슈를 복사해 저장하지 않고, 어떤 `IntegratedIssue`를 근거로 생성됐는지만 연결한다.

권장 저장 키:

```text
cluster:{cluster_id}:v3
raw:{raw_article_id}:v3
```

같은 `issue_key`를 재처리할 때는 기존 row를 `is_current=false`, `status='superseded'`로 바꾸고 새 row를 current로 저장한다.

## 14. 유지해야 할 경계

`IssueIntegrationAgent`가 해야 하는 일:

- raw row를 분석 가능한 구조로 정리
- 본문을 추출형으로 압축
- source, fact, evidence, frame을 연결
- 품질 신호와 review flag 제공

`IssueIntegrationAgent`가 하지 말아야 하는 일:

- 최종 요약문 확정
- 전략적 의미 확정
- SK AX 시사점 확정
- 대응 방향/실행 과제 제안
- 카드뉴스 문구 완성

이 경계를 유지해야 Summary/Analysis/Implication/Response/CardNews agent가 서로 역할을 침범하지 않는다.

## 15. 테스트와 확인

DB cluster 샘플 output 생성:

```bash
PYTHONPATH=. uv run python tests/test_issue_inte.py db-cluster
```

생성 파일:

```text
tests/output_issue_integration_cluster.json
```

실제 output 확인:

```python
import json

data = json.load(open("tests/output_issue_integration_cluster.json"))
issue = data["issue"]
assert issue["schema_version"] == "integrated_issue_v3"
```

관련 테스트:

```bash
PYTHONPATH=. uv run pytest tests/test_issue_integration_agent.py tests/test_issue_inte.py
```
