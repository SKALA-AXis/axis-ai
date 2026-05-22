# ITTrendAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| 이름 | `ITTrendAgent` |
| 단계 | 2단계 데이터 활용·리포트·인사이트·키워드 그래프·챗봇 |
| Supervisor | `DataUsageOrchestrator` |
| 위치 | `src/agents/it_trend_agent.py` |
| 상태 | 구조 설계 우선, 세부 로직은 후속 정의 |

## 2. 책임

`ITTrendAgent`는 SPRi / BCG 자료와 글로벌 회사 뉴스룸의 분석 결과를 기반으로
글로벌 IT 트렌드와 산업 흐름을 `TrendContext`로 정리·갱신한다. 이 Agent는
카드뉴스 생성용 Agent가 아니다.

대상 데이터 예:
- SPRI 보고서
- BCG 자료
- 글로벌 회사 뉴스룸에서 생성된 `IntegratedIssue`
- 글로벌 회사 뉴스룸에서 생성된 `AnalysisResult`

한 줄 책임:

> SPRi / BCG 트렌드 자료, 글로벌 회사 뉴스룸 분석 결과, 과거 TrendContext를 함께
> 보고 AnalysisAgent가 참고할 글로벌·산업 흐름 context를 만든다.

## 3. 책임 NOT

- 원문 크롤링
- PDF/HTML 파싱
- 기업·섹터·이벤트 매칭
- 카드뉴스 생성
- 글로벌 회사별 뉴스룸 카드뉴스 생성
- SK AX 대응 전략 확정
- DB schema 생성

## 4. IT 트렌드 영역 입력 구분

IT 트렌드 영역 입력은 두 종류로 나눈다.

1. 글로벌 회사별 뉴스룸
2. SPRi / BCG 트렌드 자료

글로벌 회사별 뉴스룸은 카드뉴스 생성 대상이다. Microsoft, AWS, Google, NVIDIA,
OpenAI 같은 회사별 뉴스룸은 일반 이슈처럼 다음 흐름을 탄다.

```text
수집
→ 전처리 / 매칭
→ IntegratedIssue
→ AnalysisAgent
→ ImplicationAgent
→ CardNewsAgent
```

SPRi / BCG 자료는 카드뉴스 생성 대상이 아니다. 이 자료는 `ITTrendAgent`가 읽어
`TrendContext`를 생성하거나 갱신하는 데 사용한다.

글로벌 회사별 뉴스룸 결과도 `TrendContext` 갱신 입력으로 사용한다. 단, 이때
`ITTrendAgent`는 원문 뉴스룸이나 카드뉴스 화면 결과가 아니라 `IntegratedIssue`와
`AnalysisResult`를 실행 신호로 참고한다.

## 5. 입력 구조

```python
class ITTrendInput:
    trend_items: list[dict]
    period: str | None
    source_groups: list[str]
    previous_trend_context: dict | None
    reference_issue_results: list[dict]
    metadata: dict
```

`trend_items`는 이미 Parser / Extractor를 거친 SPRi / BCG 정규화 데이터다.
`reference_issue_results`는 글로벌 회사별 뉴스룸의 `IntegratedIssue` /
`AnalysisResult` 입력이다.

예:

```json
{
  "source_id": "spri_2026_05_ai_report",
  "source_type": "research_report",
  "publisher": "SPRI",
  "title": "2026 AI 산업 동향",
  "content": "...",
  "published_at": "2026-05-01",
  "keywords": ["AI", "Agent", "Cloud"],
  "sources": []
}
```

## 6. 출력 구조

```python
class ITTrendOutput:
    trend_summary: str
    trend_lines: list[str]
    signals: list[dict]
    sources: list[dict]
    validation: dict
```

예:

```json
{
  "trend_summary": "최근 IT 트렌드는 생성형 AI의 업무 적용, AI 인프라 투자, 산업별 자동화 사례 확대로 요약된다.",
  "trend_lines": [
    "리서치 자료에서는 생성형 AI가 파일럿 단계를 넘어 업무 프로세스 적용으로 이동하는 흐름이 반복된다.",
    "BCG 자료에서는 AI 투자와 운영 모델 변화가 함께 다뤄진다."
  ],
  "signals": [
    {
      "signal": "AI 업무 적용 확대",
      "source_ids": ["..."],
      "confidence": "medium"
    }
  ],
  "peer_trend_links": [],
  "sources": [],
  "validation": {
    "pass": true
  }
}
```

## 7. 처리 흐름

```text
Raw / 정제 데이터 저장소
  + SPRi / BCG 트렌드 문서
  + 과거 TrendContext
  + 글로벌 뉴스룸의 IntegratedIssue / AnalysisResult
        │
        ▼
DataUsageOrchestrator
        │
        ▼
ITTrendAgent
        │
        ├─ SPRi / BCG 트렌드 source 묶음 구성
        ├─ 글로벌 뉴스룸 분석 결과에서 실행 신호 정리
        ├─ 과거 TrendContext와 최신 자료 비교
        ├─ 일시적 이슈/구조적 변화 구분
        ├─ 출처 기반 TrendContext 생성 / 갱신
        └─ validation
```

## 8. 향후 구현 원칙

1. 특정 회사명이나 특정 기술명을 규칙에 하드코딩하지 않는다.
2. 출처 유형, 기간, 반복 신호, 키워드 변화, 적용 영역 같은 정보 유형 기준으로 판단한다.
3. 원문에 없는 트렌드를 만들지 않는다.
4. 리포트/인사이트/챗봇에서 재사용 가능한 구조로 출력한다.
5. `AnalysisAgent`는 이슈 분석 시 필요하면 `TrendContext`를 참고한다.
