# 크롤러 구현 현황 (axis-ai)

> 작성일: 2026-04-23  
> 브랜치: `feat/crawler-v4`  
> 담당: AI Engineer B (심유정)

---

## 1. 전체 구조

```
src/
├── crawler/
│   ├── base.py                  # RawArticle, DailyLimitGuard, BaseCrawler, RETRY_POLICY
│   ├── batch_processor.py       # Track A/B 오케스트레이터
│   ├── fast_filter.py           # Gate 1: 관련성·품질 필터 + 경량 중요도 사전분류
│   ├── scheduler.py             # APScheduler (Track A 1시간, Track B 새벽 2시)
│   ├── playwright_client.py     # Playwright 공유 클라이언트 (SPA 렌더링)
│   ├── sources/
│   │   ├── naver.py             # Naver News API (Tier 1)
│   │   ├── rss.py               # RSS + Google News RSS (Tier 2)
│   │   ├── bigkinds.py          # BigKinds REST API (Tier 2)
│   │   ├── dart.py              # DART 금융감독원 공시 (Tier 1)
│   │   ├── kipris.py            # KIPRIS 특허·실용신안 (Tier 1)
│   │   ├── official.py          # 삼성SDS·LG CNS 공식 뉴스룸 (Tier 1)
│   │   ├── jobs.py              # Saramin 채용공고 (Tier 5)
│   │   ├── consensus.py         # 한경 컨센서스 리포트 (Tier 2)
│   │   └── naver_research.py    # 네이버 증권 리서치 (Tier 2)
│   ├── parsers/
│   │   ├── content.py           # HTML 본문 추출 (BeautifulSoup)
│   │   └── dedup.py             # URL SHA-256 해시 인메모리 중복 제거
│   └── monitors/
│       └── urgent.py            # 상시 긴급 감지 모니터 (FastFilter + GPT-4o 2차 검증)
└── db/
    ├── postgres.py              # SQLAlchemy 연결 (SessionLocal, engine)
    └── article_store.py         # raw_articles 테이블 저장 레이어
```

---

## 2. 데이터 모델

### RawArticle (base.py)

```python
@dataclass
class RawArticle:
    url: str
    title: str
    content: str              # 요약 또는 본문 (최대 10,000자 저장)
    source_tier: int          # 1·2·3
    source_name: str
    credibility_score: float  # 0.0~1.0 (SOURCE_CREDIBILITY 기준표)
    peer_id: Optional[str]    # "samsung_sds" | "lg_cns"
    published_at: Optional[datetime]
    collected_at: datetime    # 수집 시각 (자동 생성)
    url_hash: str             # MD5(url), __post_init__ 자동 계산
    metadata: dict            # 소스별 추가 필드
```

### credibility_score 기준표

| 소스 | 점수 | 티어 |
|---|---|---|
| dart | 1.00 | 1 |
| kipris | 0.95 | 1 |
| samsung_sds_newsroom | 0.90 | 1 |
| lg_cns_newsroom | 0.90 | 1 |
| yonhap | 0.85 | 2 |
| hankyung_consensus | 0.80 | 2 |
| naver_news / naver_research | 0.75 | 2 |
| bigkinds / etnews | 0.70 | 2 |
| zdnet / itchosun | 0.65~0.68 | 2 |
| google_news | 0.65 | 2 |
| saramin | 0.50 | 5 |
| telegram | 0.30 | 3 |

---

## 3. 크롤 트랙 구조

### Track A — 실시간 수집 (1시간 간격)

| 크롤러 | 소스 | API/방식 | 비고 |
|---|---|---|---|
| `NaverNewsCrawler` | 네이버 뉴스 | REST API | `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` 필요 |
| `RssCrawler` | 전자신문, ZDNet, IT조선, 연합뉴스 | RSS (feedparser) | 키워드 제목 필터링 |
| `GoogleNewsRssCrawler` | Google News | RSS (feedparser) | 쿼리 기반 |
| `BigKindsCrawler` | BigKinds | REST API + Playwright | `BIGKINDS_API_KEY` 필요 |

**실행 흐름:**
```
각 peer_id × 크롤러 병렬 실행
    → FastFilter.filter()       # Gate 1: 관련성·품질 제거
    → DedupStore.filter_new()   # 인메모리 URL 중복 제거
    → save_articles()           # raw_articles 테이블 삽입
```

### Track B — 배치 수집 (매일 새벽 2시)

| 크롤러 | 소스 | API/방식 | 비고 |
|---|---|---|---|
| `DartCrawler` | 금융감독원 전자공시 | DART OpenAPI | `DART_API_KEY` 필요, 전일~당일 공시 |
| `OfficialNewsroomCrawler` | 삼성SDS 뉴스룸 | Playwright + HTML | URL 슬러그에서 날짜 파싱 |
| `OfficialNewsroomCrawler` | LG CNS 뉴스룸 | 내부 REST API 2단계 | fingerprint → fetch |
| `JobsCrawler` | Saramin | REST API | `SARAMIN_API_KEY` 필요 |
| `KiprisCrawler` | KIPRIS 특허 | 공공데이터포털 REST API | `KIPRIS_API_KEY` 필요, XML 응답 |

**실행 흐름:**
```
per-peer: DartCrawler, OfficialNewsroomCrawler, JobsCrawler 순차 실행
KiprisCrawler는 1회만 호출 (내부에서 두 peer_id 모두 처리)
    → FastFilter.filter()
    → DedupStore.filter_new()
    → save_articles()
```

---

## 4. 소스별 구현 세부

### Naver News API
- 엔드포인트: `https://openapi.naver.com/v1/search/news.json`
- 키워드 리스트를 순회하며 각 키워드마다 최신 20건 수집
- 날짜 파싱: RFC 2822 형식 (`%a, %d %b %Y %H:%M:%S +0900`)

### RSS (IT 미디어)
- `feedparser` 라이브러리 사용
- 제목(title)에 peer 키워드 포함 여부로 필터링
- Google News RSS는 쿼리 기반 (`?q=삼성SDS&hl=ko&gl=KR`)

### BigKinds
- 1단계: REST API로 요약·메타데이터 수집 (최근 1일, IT 전문 매체 코드 6개)
- 2단계: FastFilter에서 `urgent`/`notable` 판정 기사만 Playwright로 본문 전문 수집
- `BIGKINDS_API_KEY` 미설정 시 graceful skip

### DART 공시
- 법인코드: `samsung_sds=00126186`, `lg_cns=00139834`
- 파라미터: `crtfc_key` (API 키), 전일~당일 날짜 범위 조회
- `status != "000"` 응답 시 공시 없음으로 처리

### KIPRIS 특허
- 엔드포인트: `patUtiModInfoSearchSevice/applicantNameSearchInfo`
- 파라미터: `accessKey` (ServiceKey 아님)
- XML 태그: PascalCase (`PatentUtilityInfo`, `InventionName`, `ApplicationNumber` 등)
- AI 관련 IPC 코드 필터: `G06N, G06F, G06V, G06T, H04L`
- 두 출원인(`삼성에스디에스`, `엘지씨엔에스`) asyncio.gather로 병렬 수집

### 삼성SDS 공식 뉴스룸
- Playwright로 렌더링 후 `a[href*='/news/']` 선택자로 링크 추출
- `wait_for_selector` 옵션: `state="attached"` (JS 렌더 요소는 visible 불가)
- 날짜 파싱: URL 슬러그 패턴 `-YYMMDD.html` (예: `kkr-260415.html` → 2026-04-15)

### LG CNS 공식 뉴스룸
- JS 렌더 한계로 Playwright 대신 내부 REST API 직접 호출
- 1단계: fingerprint 획득 (`/bin/cf/fetch/fingerprint?cfDirectoryPath=/newsroom`)
- 2단계: 뉴스 목록 fetch (`/bin/cf/fetch?fp={fingerprint}&filterFieldValue=press`)
- URL 구성: `item.link` 또는 `{base}/kr/newsroom/press/detail.{jcrName}`

### Saramin 채용공고
- 회사명 필터: `삼성SDS`, `LG CNS`
- `SARAMIN_API_KEY` 미설정 시 graceful skip
- 약한 신호 감지용 (조직 변화, 기술 스택 급증 탐지)

---

## 5. 필터 파이프라인

### FastFilter (Gate 1)

```
관련성 체크: title + content에 PEER_KEYWORDS 포함 여부
품질 체크:
  - content 길이 ≥ 200자
  - 한글 비율 ≥ 10%
  - 광고성 키워드 미포함 (광고, PR, Sponsored, 부동산 등)
중요도 사전 분류 (0.1ms/건):
  - CRITICAL 키워드 (합병, M&A, 상장 등): +50점
  - HIGH 키워드 (MOU, AI, 파트너십 등): +20점
  - MEDIUM 키워드 (협력, 채용 등): +5점
  - 금액 정규식 (\d+조원, \d+억원 계약): +30~50점
  - credibility ≥ 0.9: +10점 보너스
```

| 점수 | 등급 |
|---|---|
| ≥ 60 | urgent |
| 30~59 | notable |
| < 30 | reference |

### DedupStore (parsers/dedup.py)
- URL SHA-256 해시 16자리로 인메모리 중복 제거
- LRU 방식 캐시 크기 제한: 10,000건 (재시작 시 초기화)
- PostgreSQL `ON CONFLICT (url) DO NOTHING`으로 DB 레벨 2차 중복 방지

---

## 6. DB 저장 레이어 (article_store.py)

```sql
INSERT INTO raw_articles (
    peer_id, source_tier, source_name, title, content, url,
    published_at, collected_at, credibility_score, processing_status, metadata
) VALUES (..., 'RAW', CAST(:metadata AS jsonb))
ON CONFLICT (url) DO NOTHING
RETURNING id
```

**Gate 1 최소 품질 검사 (`_is_valid`):**
- `url`, `title`, `peer_id` 모두 존재해야 통과
- `content < 10자` AND `title < 5자` 동시 충족 시 제외

**저장 필드 제한:**
- `title`: 최대 500자
- `content`: 최대 10,000자

---

## 7. 긴급 감지 모니터 (monitors/urgent.py)

배치 트랙과 별개로 상시 실행되는 독립 프로세스.

```
폴링 주기:
  - RSS/Google News: 60초
  - 공식 뉴스룸 페이지: 120초

파이프라인:
  새 항목 감지 (URL 해시 기반)
      → FastFilter (score ≥ 60: urgent)
      → GPT-4o 2차 검증 (LLM 호출은 urgent만, 하루 ~5~10건)
      → Slack Webhook 발송 (confirmed=true인 경우)
```

**GPT-4o 검증 판단 기준:**
- 삼성SDS·LG CNS의 M&A, 대형 수주, 전략적 파트너십, 경영진 교체
- SK AX 주요 사업(에이전틱AI·제조AX·MSP)에 직접적 위협/기회

---

## 8. 스케줄러 (scheduler.py)

```python
# Track A — 실시간 (1시간 간격)
IntervalTrigger(hours=1)
max_instances=1, coalesce=True, misfire_grace_time=600

# Track B — 배치 (매일 새벽 2시)
CronTrigger(hour=2, minute=0)
max_instances=1, coalesce=True, misfire_grace_time=3600
```

FastAPI lifespan에서 `scheduler.start()` / `scheduler.shutdown()` 호출.

---

## 9. 환경변수 요약

| 변수명 | 필수 | 비고 |
|---|---|---|
| `NAVER_CLIENT_ID` | Track A | 미설정 시 Naver 크롤러 skip |
| `NAVER_CLIENT_SECRET` | Track A | |
| `DART_API_KEY` | Track B | 미설정 시 DART 크롤러 skip |
| `KIPRIS_API_KEY` | Track B | 미설정 시 KIPRIS 크롤러 skip |
| `BIGKINDS_API_KEY` | Track A | 미설정 시 BigKinds 크롤러 skip |
| `SARAMIN_API_KEY` | Track B | 미설정 시 Jobs 크롤러 skip |
| `DATABASE_URL` | 전체 | `postgresql://axuser:axpass@localhost:5432/axis` |
| `OPENAI_API_KEY` | urgent-monitor | GPT-4o 2차 검증용 |

---

## 10. 실제 수집 검증 결과 (2026-04-22)

| 크롤러 | 수집 건수 | 상태 |
|---|---|---|
| NaverNewsCrawler | 20건 | ✅ |
| GoogleNewsRssCrawler | 수집됨 | ✅ |
| RssCrawler | 수집됨 | ✅ |
| BigKindsCrawler | — | ⏭ API 키 없음 |
| DartCrawler | 수집됨 | ✅ |
| KiprisCrawler | 60건 | ✅ |
| OfficialNewsroomCrawler (SDS) | 3건 | ✅ |
| OfficialNewsroomCrawler (LG CNS) | 15건 | ✅ |
| JobsCrawler | — | ⏭ API 키 없음 |
| **Track A DB 저장** | **26건** | ✅ |
| **Track B DB 저장** | **9건** | ✅ |

---

## 11. 다음 단계 (미구현)

| 항목 | 우선순위 | 관련 파일 |
|---|---|---|
| 골든셋 레이블링 (20건 이상) | 🔴 | `tests/` |
| Gate 2 신뢰도 필터 (CredibilityAgent) | 🔴 | `src/agents/credibility_agent.py` |
| Gate 3 BGE-M3 중복 클러스터링 (DeduplicationAgent) | 🔴 | `src/agents/dedup_agent.py` |
| ClassificationAgent (LLM 중요도 분류) | 🟡 | `src/agents/classification_agent.py` |
| IssueCardAgent (이슈 카드 생성) | 🟡 | `src/agents/issue_card_agent.py` |
| Qdrant 벡터 삽입 파이프라인 | 🟡 | `src/rag/embedder.py` |
| content.py 본문 전문 수집 연동 | 🟡 | `src/crawler/parsers/content.py` |
| UrgentMonitor Slack Webhook 연동 | 🟢 | `src/crawler/monitors/urgent.py` |
| 크롤러 단위 테스트 | 🟢 | `tests/test_crawler.py` |
