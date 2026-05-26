# ProfileAgent

## Status

- layer: Context Maintenance Layer
- type: Agent
- code: `src/agents/profile_agent.py`
- status: implemented / needs naming and storage alignment

## Purpose

피어사(SK AX 포함)별 전략 프로필을 생성한다. 단순 DB loader가 아니라 DART, IR,
공식 뉴스룸, 누적 뉴스, business signal, financial metric을 종합해 회사별 사업 영역,
역량, 전략 변화, 주요 근거를 정리한다.

## Call Input

- `peer_company_id`
- refresh window
- optional source filters

## DB Read

- `peer_companies`
- `raw_articles`
- `raw_article_parse_results`
- `raw_article_business_signals`
- `raw_article_financial_metrics`
- DART / IR / official newsroom / cumulative news rows

## Return Output

```json
{
  "peer_id": "samsung_sds",
  "profile_snapshot": {},
  "profile_snapshot_version": "profile-v1",
  "profile_snapshot_generated_at": "2026-05-26T00:00:00+09:00",
  "evidence_refs": []
}
```

## DB Write

Target, pending DB naming confirmation:

- `peer_companies.profile_snapshot`
- `peer_companies.profile_snapshot_version`
- `peer_companies.profile_snapshot_generated_at`

Current code may store equivalent payloads in existing JSONB fields. Exact migration is open.

## Does

- 회사별 사업 영역과 capability를 요약한다.
- 최근 전략 변화와 반복 등장하는 사업 신호를 정리한다.
- 근거가 된 raw article, metric, signal id를 보존한다.
- SK AX profile과 peer profile을 분리한다.

## Does Not

- 개별 이슈의 분석 결과를 만들지 않는다.
- 카드뉴스를 만들지 않는다.
- `AnalysisInputBundle`을 실행하지 않는다.

## Failure Policy

- LLM 실패 시 이전 snapshot을 유지한다.
- 이전 snapshot이 없으면 minimal empty profile을 반환하고 refresh error를 로깅한다.
- DB write는 peer_id + profile version 기준 idempotent하게 갱신한다.

## Open Questions

- `profile_snapshot` 계열 컬럼이 현재 DB에 있는지, `peer_plus_payload`에 넣을지 결정 필요.
- weekly refresh cadence와 partial refresh trigger 기준이 필요하다.
