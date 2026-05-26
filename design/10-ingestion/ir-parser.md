# IRParserService — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `IRParserService` |
| **Supervisor** | Ingestion (Evidence sub) |
| **상태** | ✅ 구현 — `src/parsers/ir_parser.py` (W5 활성) |
| **Trigger** | Track B (매일 02:00) — IR PDF 수집 후 / EvidenceBuilder 가 호출 |

## 2. 책임

**한 줄**: IR PDF (예: 네이버 금융 리서치 / 한경 컨센서스 발) 에서 PyMuPDF 로 텍스트 + 페이지 번호 + 표/이미지 후보 추출하여 `peer_financials` 테이블에 row 채움.

**구체적**:

1. PDF bytes → PyMuPDF `fitz.open()` 으로 page-by-page 로드
2. 각 page 의 텍스트 + bbox + 표 후보 (vector graphics 분석) + 이미지 후보 추출
3. segment 별 분기 metric (매출/영업이익/AI 비중/headcount) 패턴 매칭
4. `peer_financials` 의 (peer_id, segment, period) PK 로 UPSERT
5. `dart_rcept_no` 와 `ir_page` 컬럼 채움 (FinancialLinker 가 참조)

## 3. 책임 NOT

- 비-IR PDF (단순 보고서) — 현재 IR/Research/DART 첨부 PDF 만 처리
- LLM 호출 — 산식 (PyMuPDF + regex)
- segment 정의 — manual seed (W5 prompt 결정)

## 4. 입력 스펙

```python
class IRParserInput(TypedDict):
    pdf_bytes: bytes
    peer_id: str
    source_url: str
    published_at: datetime
```

## 5. 출력 스펙

```python
class IRParserOutput(TypedDict):
    pages: list[dict]                # [{page_no, text, bbox_count, table_candidates: list[bbox], image_candidates: list[bbox]}]
    extracted_metrics: list[dict]    # [{segment, period, metric, value_krwbn, page}]
```

DB INSERT (peer_financials):
- `(peer_id, segment, period, metric)` PK
- `value_krwbn: float`
- `dart_rcept_no: text | null`
- `ir_page: int` (PDF 의 어느 페이지)
- `source_url: text`

## 6. 알고리즘

### 6.1 PDF 로드 + 페이지 텍스트

```python
import fitz  # PyMuPDF

def extract_pages(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text("text")
        # 표 후보 — line/rect 가 많은 영역
        drawings = page.get_drawings()
        rect_count = sum(1 for d in drawings if d["type"] == "f" or d["type"] == "fl")
        # 이미지 후보
        image_xrefs = page.get_images(full=True)
        pages.append({
            "page_no": page_no,
            "text": text,
            "bbox_count": rect_count,
            "table_candidates": [d["rect"] for d in drawings if rect_count > 10],
            "image_candidates": [xref[0] for xref in image_xrefs],
        })
    return pages
```

### 6.2 metric 패턴 매칭 (regex)

```python
METRIC_PATTERNS = {
    "revenue": r"매출(액)?\s*[:|]\s*([\d,]+)\s*(억|조)",
    "operating_income": r"영업이익\s*[:|]\s*([\d,]+)\s*(억|조)",
    "ai_revenue_share": r"AI\s*매출\s*비중\s*[:|]\s*([\d.]+)\s*%",
    "headcount": r"(임직원|직원)\s*수\s*[:|]\s*([\d,]+)\s*명",
}

def extract_metrics(page_text, page_no):
    metrics = []
    for metric, pattern in METRIC_PATTERNS.items():
        for match in re.finditer(pattern, page_text):
            value_raw = match.group(1).replace(",", "")
            unit = match.group(2) if len(match.groups()) > 1 else None
            value_krwbn = float(value_raw) * (10000 if unit == "조" else 1)
            metrics.append({
                "metric": metric,
                "value_krwbn": value_krwbn,
                "page": page_no,
            })
    return metrics
```

### 6.3 period / segment 매칭

- period — 본 보고서 메타 (제목/표지) 에서 추출 "2026Q1", "FY2025"
- segment — 페이지 헤더/표 제목에서 "ITS", "Cloud Solutions", "Smart Logistics" 등 매칭

## 7. LLM 모델 + token 예산

- **LLM 미사용** — PyMuPDF + regex
- 토큰 예산: ₩0
- (W6+ 옵션) segment 분류 LLM 도움 시 ~₩30/일

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| PDF 깨짐 / 비표준 | log warning + raw bytes 보존 (재시도 가능) |
| 한글 폰트 임베딩 누락 (text 추출 빈문자) | OCR fallback (W7+) — 현재는 skip + log |
| metric pattern miss | 해당 페이지 skip, 다음 페이지 진행 |
| segment 명 unknown | "Unknown" 으로 저장 + admin manual review |

## 9. 외부 의존성

- **lib**: `pymupdf` (= `fitz`)
- **DB**: `peer_financials` (UPSERT)

## 10. State 흐름

본 parser 는 LangGraph node 아니라 EvidenceBuilder + Track B 크롤러가 호출. State 변경 X.

## 11. Provenance + Confidence

- **Provenance**: peer_financials row 의 source_url + ir_page (감사 추적)
- **Confidence**: deterministic, regex hit = 1.0

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | 삼성SDS IR PDF 12 페이지 | extracted_metrics len ≥ 8 (segment 별 metric 매칭) |
| Unit | text 추출 빈문자 (이미지 only) | warning log + metrics=[] |
| Edge | 깨진 PDF (header 손상) | `fitz.FileDataError` catch → log + skip |
| Integration | Track B 매일 02:00 IR 4개 (peer 4사) | peer_financials 16 row UPSERT |

## 13. 모니터링

- **pipeline_logs.step**: 별도 step 아님 — Track B crawl 또는 evidence 의 sub
- **KPI**:
  - metric 추출 정확도 (sampling) ≥ 85%
  - PDF parsing 성공률 ≥ 95%
- **token 예산**: 해당 없음

## 14. 구현 메모 + Changelog

### 의존 lib

```toml
pymupdf = ">=1.24"
```

### 핵심 파일

- `src/parsers/ir_parser.py`
- 호출자: EvidenceBuilder + `src/crawler/sources/ir.py` (Track B)

### Changelog

- **v1 (2026-05-W1, W5 활성)** — PyMuPDF + regex 기본 metric 추출
- **v2 (제안, W7+)** — OCR fallback (한글 폰트 없는 스캔 PDF 대응)
- **v3 (제안)** — segment 분류 LLM 보조
