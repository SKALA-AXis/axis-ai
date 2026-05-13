# FinancialLinkerAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `FinancialLinkerAgent` |
| **Supervisor** | Ingestion (Evidence sub) + Enrichment (Home DART radar / Peer+ IR pack 재사용) |
| **상태** | ✅ 구현 — `src/agents/financial_linker_agent.py` |
| **호출 위치** | EvidenceAgent (Ingestion `evidence` 노드), Home dashboard, Peer+ IR pack |

## 2. 책임

**한 줄**: card 의 sector·event·title 키워드로 `peer_financials` 테이블의 segment 매칭하여 QoQ/YoY delta + DART rcept_no + IR page 참조 부착.

**구체적**:

1. card 의 (sector, event_type, title 키워드) → segment 후보 검색 (예: ITS/Cloud/Security)
2. peer_financials 의 동일 segment 분기별 데이터 fetch (당기 + 직전 + 전년 동기)
3. QoQ delta = (current - prev) / prev, YoY delta = (current - year_ago) / year_ago
4. DART 공시 번호 (rcept_no) + IR PDF page 매칭 (IRParserAgent 가 사전에 채움)
5. narrative 1줄 자동 생성 ("ITS 매출 증가와 기사 동향 연결" 등)

## 3. 책임 NOT

- IR PDF 파싱 자체 — IRParserAgent
- segment 분류 (어떤 사업부인지) — peer_financials seed (manual or W6 prompt)
- LLM 자유형 분석 — 본 agent 는 산식

## 4. 입력 스펙

```python
class FinancialLinkInput(TypedDict):
    peer_id: str            # 'samsung_sds', 'lg_cns' 등
    sector: str             # 'ax', 'infra', 'deal', ...
    event_type: str
    title: str
```

## 5. 출력 스펙

```python
class FinancialLinkOutput(TypedDict):
    link: dict | None       # {linked: bool, segment: str, highlights: list[str], headcount_delta: float | None}
    refs: list[dict]        # [{period, metric, metric_ko, value_krwbn, delta_pct_qoq, delta_pct_yoy, dart_rcept_no, ir_page, narrative}]
```

## 6. 알고리즘

### 6.1 segment 매칭 (산식)

```python
SECTOR_TO_SEGMENT = {
    "ax": ["ITS", "AX", "AI 솔루션", "Smart Logistics"],
    "security": ["IT Service", "Security"],
    "infra": ["Cloud", "Data Center", "IT Infrastructure"],
    "deal": ["*"],  # any segment
}

def find_segments(peer_id, sector, title):
    candidates = SECTOR_TO_SEGMENT.get(sector, ["*"])
    # title 키워드 매칭으로 narrow
    if "클라우드" in title or "cloud" in title.lower():
        candidates = ["Cloud", "Data Center"]
    if "AI" in title.upper() or "생성형" in title:
        candidates = ["AX", "AI 솔루션", "ITS"]
    return candidates
```

### 6.2 metric 산출 (산식)

```python
METRICS_KO = {
    "revenue": "매출",
    "operating_income": "영업이익",
    "ai_revenue_share": "AI 매출 비중",
    "headcount": "임직원",
}

def compute_deltas(segment_rows):
    current = segment_rows[0]   # 최근 분기
    prev_q = segment_rows[1]    # 직전 분기
    prev_y = segment_rows[4]    # 전년 동기

    return [{
        "period": current.period,
        "metric": metric,
        "metric_ko": METRICS_KO.get(metric, metric),
        "value_krwbn": current[metric],
        "delta_pct_qoq": round((current[metric] - prev_q[metric]) / prev_q[metric] * 100, 1),
        "delta_pct_yoy": round((current[metric] - prev_y[metric]) / prev_y[metric] * 100, 1),
        "dart_rcept_no": current.dart_rcept_no,
        "ir_page": current.ir_page,
        "narrative": f"{METRICS_KO.get(metric)} 변화 {delta_pct_qoq:+.1f}% QoQ ({segment} segment)",
    } for metric in ("revenue", "operating_income")]
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — 산식 only
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| peer_financials 에 segment 데이터 없음 | link=null, refs=[] |
| 분기 수 < 5 (전년 동기 없음) | YoY 계산 skip, QoQ 만 |
| peer_id == 'sk_ax' (자사) | link=null (자사 모니터링 안 함) |
| `dart_rcept_no` null | refs 에 포함, dart_rcept_no 만 null |

## 9. 외부 의존성

- **DB**: `peer_financials` (READ — IRParserAgent + manual seed 가 채움)
- **lib**: 없음 (산식)

## 10. State 흐름

본 agent 는 EvidenceAgent 내부 sub-call. LangGraph state 변경 X. EvidenceResult.financial_refs / financial_link 에 출력.

## 11. Provenance + Confidence

- **Provenance**: refs[*].dart_rcept_no 자체가 provenance (감사 추적)
- **Confidence**: linked=true 면 1.0 (deterministic). false 면 0.0

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | peer='samsung_sds', sector='ax', title='AX 플랫폼' | segment='AX 솔루션', linked=true, refs len ≥ 2 |
| Unit | peer='unknown', sector='other' | linked=false, refs=[] |
| Unit | 분기 데이터 1건만 | linked=true, refs len=1 (QoQ/YoY 없음) |
| Edge | event='partnership' + segment 없음 | linked=false (intentional, MOU 는 재무 직접 연결 약함) |

## 13. 모니터링

- **pipeline_logs.step**: 별도 step 아님 — `evidence` 의 sub
- **KPI**:
  - linked=true 비율 (event_type='ma'/'deal' 카드 중) ≥ 60%
  - segment 매칭 정확도 (sampling) ≥ 80%
- **token 예산**: 해당 없음

## 14. 구현 메모 + Changelog

### 핵심 파일

- `src/agents/financial_linker_agent.py`
- 데이터 source: `peer_financials` 테이블 (Track B 의 IR Parser 가 채움) + manual JSON stub (`data/peer_financials/*.json`)

### Changelog

- **v1 (2026-04-W3)** — segment 키워드 매칭 + QoQ/YoY
- **v2 (2026-05-W1, W5)** — IRParserAgent 연동 (DART rcept_no + IR page 자동 채움)
- **v3 (제안)** — segment 분류 LLM 보조 (현재 매칭 miss 시 fallback)
