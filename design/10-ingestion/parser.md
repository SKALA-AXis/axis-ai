# ParserAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `ParserAgent` (unified strategy) |
| **Supervisor** | Ingestion (sub of `crawl` node) |
| **상태** | ✅ 구현 — `axis-ai/src/agents/parser_agent.py` + `parser_quality_agent.py` + `dart_parser_agent.py` (현재 3 파일 분리, v2 통합 후보) |
| **Owner** | 심유정 |
| **Version** | v1 (현재 3 파일 분리) → v2 strategy 통합 권장 (P9 cleanup) |
| **Trigger** | CrawlerAgent 가 호출 (각 RawArticle 마다) |

## 2. 책임 (Single Responsibility)

**한 줄**: 외부 source 의 raw bytes (HTML / PDF / JSON / RSS) 를 정규화된 text + structured metadata 로 변환.

**구체적**:

1. **HTML 파싱** — BeautifulSoup + 본문 추출 (article body) + 광고/사이드바 제거
2. **PDF 파싱** — PyMuPDF 로 텍스트 + 페이지 매핑 + 표/이미지 후보
3. **DART HTML** — 표 마커 (`[표] ... [/표]`) 삽입 + `metadata.contains_tables` + `metadata.table_count` 부착
4. **품질 평가** — 본문 200자 미만 / 한글 비율 < 10% / 광고성 키워드 → `processing_status = 'SKIPPED_QUALITY'` 로 마킹 (Gate 1)
5. **Source 별 strategy** — naver_news.html ≠ dart_html ≠ ir_pdf 의 파싱 규칙 다름

## 3. 책임 NOT (out of scope)

- 외부 fetch — CrawlerAgent 가 담당
- 신뢰도 평가 — CredibilityAgent (다음 노드)
- 관련성 판별 — RelevanceAgent (다음 노드)
- 클러스터링 — DedupAgent
- LLM 호출 — 본 agent 는 산식 only

## 4. 입력 스펙

```python
class ParserInput(TypedDict):
    raw_bytes: bytes
    content_type: Literal["text/html", "application/pdf", "application/json", "application/rss+xml"]
    source_name: str                # 'naver_news' | 'dart' | 'ir_naver_finance' | ...
    source_type: str                # 'news' | 'official' | 'dart' | 'ir' | 'job' | ...
    url: str                        # 출처 URL (메타 추적용)
```

## 5. 출력 스펙

```python
class ParserOutput(TypedDict):
    content: str                    # 정규화된 본문 텍스트 (최대 10,000 chars)
    title: str | None
    published_at: datetime | None
    extracted_metadata: dict        # source 별 추가 메타 (예: DART → contains_tables, table_count)
    parse_strategy: str             # 'html_default' | 'dart_html' | 'pdf_pymupdf' | 'rss' | 'json'
    quality_status: Literal["OK", "SKIPPED_QUALITY", "PARSE_ERROR"]
    quality_reason: str | None      # SKIPPED_QUALITY 시 사유 (예: "content < 200 chars")
```

**post-conditions**:

- `quality_status == 'OK'` 이면 content ≥ 200자 + 한글 비율 ≥ 10%
- `quality_status == 'SKIPPED_QUALITY'` row 는 `raw_articles.processing_status` 도 동일하게 마킹 → 후속 노드가 skip
- `parse_strategy == 'dart_html'` 이면 content 내부에 `[표] ... [/표]` 마커 1+ 포함

## 6. 알고리즘

### 6.1 Strategy 선택 (v2 권장)

```python
class ParserAgent:
    STRATEGIES = {
        ("text/html", "dart"): DartHtmlStrategy(),         # 표 마커 + jsonb 메타
        ("text/html", "*"): GenericHtmlStrategy(),         # BeautifulSoup 기본
        ("application/pdf", "ir"): IrPdfStrategy(),        # PyMuPDF + 페이지/표 후보
        ("application/pdf", "*"): GenericPdfStrategy(),
        ("application/json", "*"): JsonStrategy(),         # Naver API 응답 등
        ("application/rss+xml", "*"): RssStrategy(),       # feedparser
    }

    def parse(self, raw_bytes, content_type, source_type, ...):
        strategy = self.STRATEGIES.get((content_type, source_type),
                                       self.STRATEGIES[(content_type, "*")])
        parsed = strategy.parse(raw_bytes, ...)
        quality = self._quality_check(parsed.content)
        return ParserOutput(..., quality_status=quality.status, quality_reason=quality.reason)
```

### 6.2 DART 표 마커 삽입 (DartHtmlStrategy)

```python
def _table_to_text(table: Tag) -> str:
    """<table> 을 셀 단위 공백/줄바꿈 텍스트로 변환"""
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        rows.append(" ".join(cells))
    return "\n".join(rows)

def parse(self, raw_bytes, ...):
    soup = BeautifulSoup(raw_bytes, "html.parser")
    tables = soup.find_all("table")
    content_parts = []
    table_count = 0
    for elem in soup.body.descendants:
        if elem.name == "table":
            content_parts.append(f"\n[표]\n{_table_to_text(elem)}\n[/표]\n")
            table_count += 1
        elif elem.name in ("p", "div") and not _is_inside_table(elem):
            content_parts.append(elem.get_text(" ", strip=True))
    return ParsedDocument(
        content="\n".join(content_parts),
        extracted_metadata={
            "contains_tables": table_count > 0,
            "table_count": table_count,
            "table_parse_strategy": "html_table_to_text",
        },
        parse_strategy="dart_html",
    )
```

### 6.3 품질 평가 (ParserQualityAgent → strategy 의 일부)

```python
def _quality_check(self, content: str) -> QualityResult:
    if len(content) < 200:
        return QualityResult("SKIPPED_QUALITY", "content < 200 chars")
    korean_ratio = sum(1 for c in content if "가" <= c <= "힣") / len(content)
    if korean_ratio < 0.10:
        return QualityResult("SKIPPED_QUALITY", f"한글 비율 {korean_ratio:.1%} < 10%")
    if any(kw in content for kw in AD_KEYWORDS):
        return QualityResult("SKIPPED_QUALITY", "광고성 키워드 포함")
    return QualityResult("OK", None)
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — 산식 only
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| `BeautifulSoup` 파싱 실패 (encoding 깨짐) | content 그대로 raw 저장 + `quality_status='PARSE_ERROR'` |
| PyMuPDF 가 PDF 못 읽음 | `quality_status='PARSE_ERROR'` + PDF bytes 보존 (재처리 가능) |
| DART 표 안에 nested table | inner 만 표 마커, outer 는 그냥 cell text (재귀 X — 무한루프 방지) |
| 한글 비율 < 10% (영문 기사) | `SKIPPED_QUALITY` — 후속 노드가 처리 안 함 |
| content > 10,000 chars | 앞 10,000 만 잘라서 저장 + `metadata.content_truncated = true` |

## 9. 외부 의존성

- **lib**: `beautifulsoup4>=4.12`, `lxml`, `PyMuPDF`, `feedparser>=6.0`
- **DB**: `raw_articles.content`, `raw_articles.metadata` (jsonb), `raw_articles.processing_status`
- **외부 API**: 없음 (순수 파싱)

## 10. State 흐름 (LangGraph)

본 agent 는 LangGraph node 가 아니라 **CrawlerAgent 내부에서 호출되는 sub-agent**. 따라서 state 변경 없음 — 단지 raw_articles row 업데이트.

```python
# crawler_agent.py 내부:
parsed = parser_agent.parse(raw_bytes=..., content_type=..., source_type=..., url=...)
article = RawArticle(
    content=parsed.content,
    title=parsed.title,
    published_at=parsed.published_at,
    metadata={**article.metadata, **parsed.extracted_metadata},
    crawl_status=parsed.quality_status,
)
```

## 11. Provenance + Confidence

- **Provenance**: `raw_articles.metadata.parse_strategy`, `raw_articles.metadata.parser_version`
- **Confidence**: N/A — quality_status 가 OK / SKIPPED / ERROR 3-state 로 표현

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | DART HTML 표 3개 포함 입력 | content 내 `[표]` 마커 3쌍 + `metadata.table_count=3` |
| Unit | 본문 100자 (200 미만) | `quality_status='SKIPPED_QUALITY'`, reason="content < 200 chars" |
| Unit | 영문 기사 (한글 0%) | `SKIPPED_QUALITY` |
| Unit | "광고 문의 02-1234-5678" 포함 | `SKIPPED_QUALITY` |
| Unit | PDF 페이지 5장 | content 5 페이지 합쳐짐 + `metadata.ir_page_count=5` |
| Integration | Naver Search API response 100건 → parse | 100건 중 ≥ 90건 OK · ≤ 10건 SKIPPED |
| Edge | 깨진 HTML (`<div>` close 누락) | BeautifulSoup 의 lxml 가 best-effort 복구 → content 추출 |
| Edge | 빈 본문 (`""`) | `PARSE_ERROR` |

## 13. 모니터링

- **pipeline_logs.step**: 별도 step 아님 — `crawl` 의 sub-metric
- **KPI**:
  - 파싱 성공률 ≥ 90%
  - SKIPPED_QUALITY 비율 ≤ 15% (높으면 source 별 룰 조정)
  - PARSE_ERROR 비율 ≤ 2%
  - DART 표 마커 정확도 (수동 sampling) ≥ 95%
- **token 예산**: 해당 없음

## 14. 구현 메모 + Changelog

### 의존 lib

```toml
beautifulsoup4 = ">=4.12"
lxml = ">=5.0"        # BeautifulSoup parser
pymupdf = ">=1.24"    # PDF
feedparser = ">=6.0"  # RSS
```

### 핵심 파일 (현재 v1 — 분리)

- `src/agents/parser_agent.py` — 일반 HTML/PDF
- `src/agents/parser_quality_agent.py` — Gate 1 (품질 평가)
- `src/agents/dart_parser_agent.py` — DART 표 마커

### v2 통합 권장 (P9 cleanup)

3 파일 → 1 file (`parser_agent.py`) + Strategy 패턴 (`src/agents/parser/strategies/{dart,generic_html,pdf,rss,json}.py`)

### Changelog

- **v1 (2026-04-W2)** — 3 파일 분리 구현
- **v1.1 (2026-04-W3)** — DART 표 마커 + PyMuPDF 추가
- **v2 (proposed, P9)** — Strategy 패턴 통합
