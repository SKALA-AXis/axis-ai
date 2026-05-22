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

`ITTrendAgent`는 글로벌 IT 트렌드, 산업 동향, 기술 키워드를 Peer사 동향과 연결해 해석한다.

대상 데이터 예:
- SPRI 보고서
- BCG / McKinsey / Gartner 등 컨설팅·리서치 자료
- 글로벌 회사 뉴스룸
- 산업 동향 브리핑
- 시장·기술 트렌드 문서

한 줄 책임:

> 외부 트렌드 출처와 저장된 Peer사 동향을 연결해 현재 IT 흐름이 구조적 변화인지 판단한다.

## 3. 책임 NOT

- 원문 크롤링
- PDF/HTML 파싱
- 기업·섹터·이벤트 매칭
- 단일 Peer사 카드뉴스 생성
- SK AX 대응 전략 확정
- DB schema 생성

## 4. 글로벌 기업 동향 포함 기준

글로벌 기업 뉴스룸, 공식 발표, 빅테크 동향은 별도 글로벌 트렌드 전용 에이전트로 분리하지 않고
`ITTrendAgent`의 입력 소스 중 하나로 본다.

`ITTrendAgent`는 SPRI/BCG/뉴스룸/산업 브리핑에서 보이는 현재 IT 흐름과
Peer사 동향의 연결성을 정리한다.

## 5. 입력 구조

```python
class ITTrendInput:
    items: list[dict]
    period: str | None
    source_groups: list[str]
    metadata: dict
```

`items`는 이미 Parser / Extractor를 거친 정규화 데이터다.

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
    "글로벌 회사 뉴스룸에서는 AI 인프라와 데이터센터 투자 확대가 주요 신호로 확인된다.",
    "산업 브리핑에서는 제조·금융·공공 영역의 자동화 적용 사례가 증가하고 있다."
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
  + 외부 트렌드 문서
  + 글로벌 뉴스룸 데이터
        │
        ▼
DataUsageOrchestrator
        │
        ▼
ITTrendAgent
        │
        ├─ 트렌드 source 묶음 구성
        ├─ 반복 출현 키워드/주제 정리
        ├─ Peer사 동향과 트렌드 연결
        ├─ 일시적 이슈/구조적 변화 구분
        ├─ 출처 기반 trend_lines 생성
        └─ validation
```

## 8. 향후 구현 원칙

1. 특정 회사명이나 특정 기술명을 규칙에 하드코딩하지 않는다.
2. 출처 유형, 기간, 반복 신호, 키워드 변화, 적용 영역 같은 정보 유형 기준으로 판단한다.
3. 원문에 없는 트렌드를 만들지 않는다.
4. 리포트/인사이트/챗봇에서 재사용 가능한 구조로 출력한다.
5. SK AX 전략 시사점이 필요하면 `InsightAgent` 또는 `ImplicationAgent`와 결합한다.
