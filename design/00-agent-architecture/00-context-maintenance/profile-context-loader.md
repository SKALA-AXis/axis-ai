# ProfileContextLoader

## Status

- layer: Context Maintenance Layer / AnalysisGraphRunner runtime dependency
- type: Service
- current code: `src/services/profile_context_loader.py`
- legacy function: `src/services/profile_context_v2.py`의 `build_profile_context_v2()`
- status: implemented as class wrapper / legacy function retained

## Purpose

AnalysisGraphRunner 실행 중 `IntegratedIssue`가 확정한 main company와 mentioned companies를
기준으로 `ProfileContext`를 조립한다. 이 컴포넌트는 LLM agent가 아니라 DB lookup과
context assembly service다.

## Call Input

- `companies`
- `sectors`
- `event_type`
- `integrated_issue.main_company`
- `integrated_issue.mentioned_peer_companies`

## DB Read

- `peer_companies.profile_snapshot` or equivalent profile JSONB
- `raw_article_business_signals`
- `raw_article_financial_metrics`
- optional SK AX profile source

## Return Output

```json
{
  "skax_profile": {},
  "peer_profiles": {},
  "sector_context": {}
}
```

## DB Write

None.

## Does

- main company 우선으로 peer profile을 로드한다.
- mentioned peer와 input bundle companies를 fallback으로 사용한다.
- 최근 business/financial context를 runtime에 가볍게 보강한다.
- `AnalysisAgent`와 `ImplicationAgent`가 공유할 `ProfileContext`를 만든다.

## Does Not

- 새로운 profile snapshot을 생성하지 않는다.
- LLM을 호출하지 않는다.
- 카드뉴스/시사점 문구를 만들지 않는다.

## Failure Policy

- primary loader 실패 시 legacy `ProfileAgent.build_context()` fallback을 허용한다.
- 둘 다 실패하면 empty `ProfileContext`를 반환하고 analysis confidence에 반영한다.

## Open Questions

- `build_profile_context_v2()` legacy function 제거 시점.
- 기존 `ProfileAgent.build_context()` legacy fallback 제거 시점.
