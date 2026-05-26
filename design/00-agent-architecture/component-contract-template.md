# Component Contract Template

모든 Agent/Service/Builder/Job 문서는 아래 형식을 따른다.  
목적은 이름보다 **입출력, DB 책임, failure policy**를 먼저 고정하는 것이다.

```md
# ComponentName

## Status

- layer:
- type: Agent | Runner | Orchestrator | Service | Builder | Generator | Job
- code:
- owner:
- status: draft | implemented | planned | deprecated

## Purpose

무엇을 해결하는 컴포넌트인가.

## Call Input

호출자가 넘겨주는 값.

## DB Read

직접 읽는 테이블/컬럼.

## Return Output

다음 단계로 넘기는 결과.

## DB Write

저장하는 테이블/컬럼. 없으면 `None`.

## Does

해야 하는 일.

## Does Not

하지 말아야 하는 일.

## Failure Policy

retry / fallback / empty result / human_review / fail-fast 여부.

## Observability

로그, trace, metric, evaluation payload.

## Open Questions

아직 결정되지 않은 지점.
```

## 최소 작성 규칙

- `Purpose`, `Call Input`, `Return Output`, `Does Not`, `Failure Policy`는 비워두지 않는다.
- DB write가 있으면 idempotency 기준을 함께 적는다.
- LLM을 호출하면 prompt version과 output schema를 적는다.
- LLM을 호출하지 않으면 Agent suffix를 쓰지 않는 것을 기본값으로 한다.
