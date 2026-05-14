# CrawlerAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `CrawlerAgent` |
| **Supervisor** | Ingestion |
| **LangGraph node** | `crawl` (첫 노드) |
| **상태** | ✅ 구현 — `axis-ai/src/agents/crawler_agent.py` 외 분산 (`src/crawler/sources/*.py`) |
| **Owner** | 심유정 |
| **Version** | v3 (2026-04-W3 — Track A/B/C 분리) |
| **Trigger** | Spring `@Scheduled.triggerIngestionPipeline` 매시 정각 → `POST /pipeline/run?track=A|B|C|ALL` |

## 2. 책임 (Single Responsibility)

**한 줄**: peer ids + track 을 입력받아 외부 소스에서 raw 기사를 fetch 하여 `raw_articles` 테이블에 INSERT.

**구체적**:

1. peer 별 검색 키워드로 외부 API/RSS/Playwright 크롤
2. 응답 데이터 → `RawArticle` dataclass (url, title, content, source_name, published_at, ...)
3. 중복 URL 은 `INSERT ... ON CONFLICT (url) DO NOTHING` 으로 idempotent 보장
4. `crawl_run_id` UUID 발급 + `pipeline_logs` 에 메트릭 (수집 건수 / 실패율 / latency) 기록

## 3. 책임 NOT (out of scope)

- HTML/PDF 내용 파싱 — ParserAgent (별도)
- 신뢰도 분류 — CredibilityAgent (다음 노드)
- 관련성 분류 — RelevanceAgent (다음 노드)
- 중복 클러스터링 — DedupAgent (다음 노드, BGE-M3 임베딩 기반)
- 광고/품질 필터 — ParserQualityAgent (sub)

## 4. 입력 스펙

```python
class CrawlInput(TypedDict):
    company: list[str]              # peer ids: samsung_sds | lg_cns | hyundai_autoever | posco_dx
    track: Literal["a", "b", "c", "all"]
    trigger_type: Literal["scheduled", "manual"]
    collected_since: str | None      # ISO8601 — incremental fetch lower bound
```

## 5. 출력 스펙

```python
class CrawlOutput(TypedDict):
    raw_article_ids: list[int]       # 신규 INSERT 된 raw_articles.id
    crawl_run_id: str                # UUID — pipeline_logs trace
    errors: list[str]                # source 별 실패 (예: "naver: 429 rate limit")
```

**post-conditions**:

- 모든 raw_articles row 는 `source_type`, `source_name`, `url_hash` 채워짐
- 동일 url 중복은 skip (ON CONFLICT)
- `processing_status = 'RAW'` 초기값

## 6. 알고리즘

### Track A (1시간 간격 — 실시간 뉴스)

```python
sources_track_a = [
    NaverNewsAPI(peer_keywords),       # 0.75 신뢰도
    GoogleNewsRSS(peer_keywords),       # 0.65
    ETnewsRSS(sections=["it","industry","economy"]),
    ZDNetRSS, BloterRSS,                # 0.68
    YonhapIndustryRSS,                  # 0.85
]
```

### Track B (매일 02:00 — 배치)

```python
sources_track_b = [
    DartOpenAPI(corp_codes_4peer),      # 1.00 신뢰도
    KiprisPatent(applicant_names),      # 0.95
    OfficialNewsroom(samsung_sds, lg_cns),  # 0.90 Playwright
    OfficialNewsroomGeneric(hyundai_autoever, posco_dx),  # 0.90 best-effort
    HankyungConsensus(peer_codes),      # 0.80 Playwright
    NaverFinanceResearch(item_codes),   # 0.75 Playwright
    Work24Jobs(applicant_names),        # 0.50 약신호 input (work24.go.kr OpenAPI)
]
```

### Track C (별도 03:30 — 산업 보고서)

```python
sources_track_c = [
    BCGNewsroom, McKinseyInsights,      # MBB
]
```

### 예외 처리 정책

```python
# 각 source 별 try/except — 한 source 실패해도 다른 source 진행
for source in sources:
    try:
        articles = await source.fetch(peer_keywords)
        save_articles(articles, run_context=ctx)
    except (HTTPError, TimeoutError) as e:
        log.warning("[%s] 수집 실패: %s", source.name, e)
        errors.append(f"{source.name}: {e}")
        # 5분 대기 후 1회 재시도
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — 산식 only (외부 fetch + dataclass mapping + INSERT)
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| HTTP 403/429 | 5분 대기 후 1회 재시도 → 실패 시 source SKIP + log |
| 파싱 오류 | 원문 그대로 raw_articles 저장 (`processing_status = 'PARSE_ERROR'`) → ParserQualityAgent 가 Gate 1 에서 거름 |
| 타임아웃 10초 | source SKIP + log |
| Playwright 브라우저 부재 (현대오토에버 등) | `BrowserType.launch failed` → 해당 peer 만 skip, 다른 peer 진행 |
| DB INSERT 실패 | row 별 rollback + `db.commit()` 후속 row 계속 |
| robots.txt 위반 source | 크롤링 금지 (whitelist 만) |

## 9. 외부 의존성

- **DB**: `raw_articles` 테이블 (INSERT)
- **DB**: `pipeline_logs` 테이블 (metric trace via `_logged_step`)
- **외부 API**: Naver Search API · DART OpenAPI · KIPRIS · 고용24 work24.go.kr OpenAPI (각각 API key 필요 — env `NAVER_*`, `DART_API_KEY`, `KIPRIS_API_KEY`, `WORK24_API_KEY` + `WORK24_RETURN_TYPE`)
- **외부 RSS**: ETnews · ZDNet · Bloter · Yonhap · Google News
- **Playwright**: 공식 뉴스룸 (헤드리스 Chromium) — Pod 에 `playwright install` 필요

## 10. State 흐름 (LangGraph)

**소비**:
- `company: list[str]` (peer ids)
- `trigger_type`
- `collected_since`

**생산**:
- `raw_article_ids: list[int]` (이후 노드가 ID 로 fetch)
- `crawl_run_id: str`
- `errors` (append)

## 11. Provenance + Confidence

- **Provenance**: `raw_articles.metadata` jsonb 에 source-specific 메타 (api_endpoint, query_keywords, fetch_timestamp, retry_count)
- **Confidence**: N/A (raw 데이터 수집 단계, 신뢰도는 CredibilityAgent 가 부여)

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | NaverNewsAPI mock — 정상 응답 100건 | `save_articles` 가 100건 INSERT |
| Unit | NaverNewsAPI mock — 429 응답 | retry 1회 후 SKIP, `errors` 에 entry |
| Unit | 동일 url 중복 INSERT | ON CONFLICT 로 skip, `inserted=0` 반환 |
| Integration | Track A 전체 실행 (peer 4사 × 6 source) | `raw_article_ids` 100+ + `pipeline_logs` 7 step entry |
| Edge | Playwright 브라우저 미설치 | 해당 source 만 fail, 다른 source 진행 |
| Edge | DB 연결 끊김 | retry → 실패 → `pipeline_logs.status='failed'` |

## 13. 모니터링

- **pipeline_logs.step**: `crawl`
- **KPI**:
  - 일일 수집 raw 건수 ≥ 400
  - source 별 성공률 ≥ 95% (DART 100%, work24 80%, Playwright 75%)
  - 평균 latency 매시 cycle ≤ 5분
- **알람** (P3): 시간당 수집 0건 = source 전체 fail = 즉시 alert
- **token 예산**: 해당 없음 (LLM 미사용)

## 14. 구현 메모 + Changelog

### 의존 lib

```toml
httpx = ">=0.27"        # async HTTP
feedparser = ">=6.0"    # RSS
beautifulsoup4 = ">=4.12"
playwright = ">=1.40"   # Track B 공식 뉴스룸
sqlalchemy = ">=2.0"
```

### 핵심 파일

- 진입점: `src/api/router.py` 의 `_run_collection_track`
- 배치 처리: `src/crawler/batch_processor.py`
- source 별 구현: `src/crawler/sources/{naver,dart,kipris,saramin,bcg,company_news,global_newsroom,ir,jobs,naver_research}.py`
- save logic: `src/db/article_store.py` 의 `save_articles`

### Changelog

- **v1 (2026-04-W1)** — Track A 만 (naver + google news + ETnews)
- **v2 (2026-04-W2)** — DART/KIPRIS/고용24 (work24) 추가 (Track B)
- **v3 (2026-04-W3)** — Track C (BCG/McKinsey) 분리 + Playwright newsroom + IR PDF
- **v4 (2026-05-12)** — 044ac12 commit 후 backfill_runner 분리, retry 정책 강화
