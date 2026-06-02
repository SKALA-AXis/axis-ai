# CardNewsAgent — Design Plan

## 메타

| 항목 | 값 |
|---|---|
| 권장 이름 | `CardNewsAgent` |
| 단계 | 1단계 데이터 수집·정제·통합·분석·시사점·카드뉴스 생성 |
| 위치 | `src/agents/card_news_agent.py` |
| 입력 | `AnalysisPackage` |
| 출력 | 기존 카드뉴스 저장/API 구조 |

## 책임

`CardNewsAgent`는 최종 결과를 카드뉴스 형태로 재가공하는 presentation Agent이다.
원문 통합이나 전략 분석을 수행하지 않는다.

주요 책임:
- 카드 제목 생성
- 카드뉴스용 3줄 요약 생성
- 핵심 포인트 생성
- 시사점 문장 재가공
- sources / validation 구성
- 기존 `save_card_news` 경로에 맞는 payload 생성

## 입력

```python
class AnalysisPackage:
    bundle_id: str
    input_bundle: AnalysisInputBundle
    integrated_issue: dict
    analysis: dict
    implication: dict
    sources: list[dict]
    validation: dict
```

입력 의미:
- `integrated_issue`: IntegrationAgent가 만든 통합 이슈
- `analysis`: AnalysisAgent가 만든 전략 의미 분석
- `implication`: ImplicationAgent가 만든 SK AX 관점 시사점
- `sources`: 원문 출처
- `validation`: 분석 패키지 검증 정보

## 출력

기존 DB/API 저장 구조를 유지한다.

```python
class CardNews:
    card_id: str
    title: str
    summary_lines: list[str]
    key_points: list[str]
    implication: str
    sources: list[dict]
    validation: dict
    company: list[str]
    sector: list[str]
    event_type: str | None
```

## 처리 흐름

```text
IntegratedIssue
+ AnalysisResult
+ ImplicationResult
+ sources
+ validation
        │
        ▼
CardNewsAgent
        │
        ├─ title
        ├─ summary_lines
        ├─ key_points
        ├─ implication
        ├─ sources
        └─ validation
        │
        ▼
기존 save_card_news 경로
        │
        ▼
card_news / card_news_articles / evidence_chain
```

## 카드뉴스용 3줄 요약

카드뉴스용 3줄 요약은 여기서 생성한다.

기준:
1. 핵심 사건 또는 상태
2. 연결된 제품·서비스·기술·업무·고객·산업 영역
3. 수치·적용 사례·후속 단계·시장 반응·불확실성 중 가장 구체적인 사실

주의:
- `IntegrationAgent`의 `integrated_text`를 그대로 복사하지 않는다.
- 원문에 없는 수치, 제품명, 고객명, 원인을 만들지 않는다.
- 시사점 문장을 사실 요약에 섞지 않는다.
- 카드뉴스 문장은 사용자 화면에서 읽기 좋은 presentation 문장으로 재작성한다.

## 책임 NOT

- 데이터 수집
- 원문 저장
- 전처리·적합성 판단
- 기업·섹터·이벤트 매칭
- 이슈 통합
- 전략 분석
- 프로필 context 생성
- 시사점 원문 생성
- DB schema 변경
- `src/schemas.py` 변경

## Validation

검증 항목:
- 제목이 비어 있지 않은가
- `summary_lines`가 정확히 3문장인가
- 핵심 포인트가 `IntegratedIssue`, `AnalysisResult`, `ImplicationResult`와 일관되는가
- sources가 비어 있지 않은가
- 기존 카드뉴스 저장/API schema와 맞는가

## 설계 원칙

1. `CardNewsAgent`는 `AnalysisPackage`를 재가공한다.
2. raw item을 직접 분석하지 않는다.
3. 카드뉴스용 3줄 요약은 여기서 만든다.
4. 기존 DB schema와 API schema는 유지한다.
