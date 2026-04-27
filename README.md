# axis-ai

AXIS 서비스의 Python AI 서버. **뉴스·공시·채용공고 크롤링 → LangGraph 5-노드 분석 파이프라인 → 이슈 카드 + 검증 첨부 4종 생성**까지 담당합니다.

> 전체 프로젝트 개요는 [axis-infra](https://github.com/SKALA-AXis/axis-infra) 참조.

---

## TL;DR — 이 브랜치(`feat/crawler-v4`)에서 바뀐 것

| 분류 | 변경 | 영향 |
|---|---|---|
| **모니터링 Peer 4사 확장** | `samsung_sds`, `lg_cns` → `+ hyundai_autoever`, `+ posco_dx` | 모든 크롤러·재무 모듈 4 peer 지원 |
| **BigKinds 완전 제거** | `sources/bigkinds.py` 삭제, `BIGKINDS_API_KEY` env 제거 | Track A에서 빠짐 |
| **버그 수정** | `naver_research.py`의 LG CNS itemCode `034730`(=SK Inc.) → `064400` | 그동안 LG CNS 리서치 페이지가 아니라 SK 지주 리서치를 긁고 있었음 |
| **현대오토에버·포스코DX 뉴스룸 추가** | `OfficialNewsroomCrawler`에 generic Playwright 방식 추가 | best-effort 셀렉터, W7에서 검증 보강 예정 |
| **v3 파이프라인 전환** | 5-node ingestion graph: crawl → credibility → dedup → classify → issue_card → evidence | ImplicationAgent / ValidationAgent / WeakSignalAgent → `_deprecated/`로 이관 |
| **EvidenceAgent 신설** | 검증 첨부 4종(source_links/provenance/financial_refs/mbb_refs) 자동 부착 | 환각 방지 1차 방어선이 SC → Evidence Chain으로 이동 |
| **FinancialLinkerAgent 신설** | 카드 ↔ 재무 segment QoQ/YoY 매칭 + vs SK AX 결정적 비교 4지표 | DART OpenAPI 실수치 기반 |
| **classification_agent v3** | 트렌드 섹터(5종) + **결정적 노출도** 산식 (LLM 5축 점수 폐기) | 추적·재현 가능한 점수 |
| **`data/peer_financials/` 신설** | 4개 peer + sk_ax JSON, DART OpenAPI 실수치 + segment/AI비중 stub | FinancialLinkerAgent 입력 |
| **IRParserAgent 추가 (스켈레톤)** | PyMuPDF 기반 IR PDF 텍스트 추출 | W5에서 본격 활성 |

---

## 이 레포의 책임

```
크롤러 (뉴스·공시·채용공고 수집)
  ↓
3단계 필터 게이트 (품질·신뢰도·중복)
  ↓
LangGraph v3 수집 파이프라인 (5-노드)
  ↓
이슈 카드 + Evidence Chain 4종 → PostgreSQL
  ↓
RAG 하이브리드 검색 (BGE-M3 + Qdrant)
  ↓
GPT-4o Generative Search
  ↓
FastAPI 내부 서버 (SpringBoot에서만 호출)
```

---

## 기술 스택

| 항목 | 내용 |
|---|---|
| 언어 | Python 3.11+ |
| 패키지 관리 | **uv** (pip 사용 금지) |
| 웹 프레임워크 | FastAPI 0.115.x |
| AI 파이프라인 | LangGraph 1.1.x |
| LLM | GPT-4o (langchain-openai 1.1.x) |
| 임베딩 | BGE-M3 (FlagEmbedding 1.2.x) — Dense + Sparse 원샷 |
| Reranker | BGE-reranker-v2-m3 |
| Vector DB | Qdrant 1.9.x |
| Raw DB | PostgreSQL 16.x (SQLAlchemy 2.x) |
| 헤드리스 브라우저 | Playwright (SDS·hyundai·posco 뉴스룸·한경 컨센서스) |
| 린트·포맷 | ruff |
| 타입 체크 | mypy |
| 테스트 | pytest + pytest-asyncio |

---

## 프로젝트 구조

```
axis-ai/
├── CLAUDE.md                       # Claude Code 컨텍스트
├── README.md                       # 이 파일
├── Dockerfile
├── pyproject.toml                  # uv 의존성 정의
├── uv.lock                         # 잠금 파일 — 반드시 커밋
├── run_pipeline_once.py            # 파이프라인 1회 실행 스크립트 (디버깅용)
├── .env.example
│
├── data/                           # ★ 신설: 재무 데이터 (FinancialLinkerAgent 입력)
│   ├── peer_financials/
│   │   ├── samsung_sds.json        # DART OpenAPI 실수치 + segment/AI비중 stub
│   │   ├── lg_cns.json
│   │   ├── hyundai_autoever.json
│   │   └── posco_dx.json
│   ├── sk_ax_financials.json       # SK 지주(holding) 매출 — _scope_warning 마킹
│   └── ir_samples/                 # IR PDF 샘플 (IRParserAgent 입력, W5)
│
├── src/
│   ├── api/
│   │   ├── main.py                 # uvicorn 진입점
│   │   └── router.py               # FastAPI 엔드포인트
│   │
│   ├── agents/                     # ── v3 에이전트 ──
│   │   ├── crawler_agent.py        # 4 peer 크롤러 오케스트레이션
│   │   ├── credibility_agent.py    # Gate 2: 출처 신뢰도 분류
│   │   ├── dedup_agent.py          # Gate 3: 중복 제거 + 클러스터링
│   │   ├── classification_agent.py # ★ v3: 트렌드 섹터 5종 + 결정적 노출도
│   │   ├── issue_card_agent.py     # GPT-4o로 3줄 요약 + 시사점 생성
│   │   ├── evidence_agent.py       # ★ 신설: 검증 첨부 4종 부착
│   │   ├── financial_linker_agent.py # ★ 신설: 카드 ↔ 재무 segment 매칭 + vs SK AX
│   │   ├── ir_parser_agent.py      # ★ 신설(스켈레톤): IR PDF 파싱 (W5)
│   │   ├── notification_agent.py   # 이메일 발송 (Slack Webhook 폐기)
│   │   ├── sector_keywords.py      # 섹터 분류 키워드 사전
│   │   └── _deprecated/            # ← v1 에이전트 보관소
│   │       ├── implication_agent.py    # SK AX 시사점 (보류)
│   │       ├── validation_agent_sc.py  # SC 검증 (Evidence Chain으로 대체)
│   │       └── weak_signal_agent.py    # 약한 신호 (W7에 부활 예정)
│   │
│   ├── crawler/                    # ── 크롤러 ──
│   │   ├── base.py                 # SOURCE_CREDIBILITY, DailyLimitGuard, RawArticle
│   │   ├── batch_processor.py      # Track A/B 오케스트레이션 + DART CORP_CODES (4 peer)
│   │   ├── scheduler.py            # APScheduler + PEER_KEYWORDS (4 peer)
│   │   ├── playwright_client.py    # 공통 Playwright 헤드리스 클라이언트
│   │   ├── fast_filter.py          # Gate 1 (품질) 사전 필터
│   │   ├── parsers/
│   │   │   ├── content.py          # readability 본문 추출
│   │   │   └── dedup.py            # URL 해시 중복 제거
│   │   ├── monitors/
│   │   │   └── urgent.py           # 긴급 키워드 모니터링
│   │   └── sources/                # 소스별 크롤러 (BigKinds 삭제됨)
│   │       ├── naver.py            # Naver News API
│   │       ├── rss.py              # ETnews / ZDNet / Bloter / 연합뉴스 / Google News RSS
│   │       ├── dart.py             # DART OpenAPI 공시
│   │       ├── kipris.py           # 특허 (4 peer)
│   │       ├── official.py         # 공식 뉴스룸 (SDS/LGCNS + hyundai/posco generic)
│   │       ├── consensus.py        # 한경 컨센서스 (4 peer)
│   │       ├── naver_research.py   # 네이버 금융 리서치 (4 peer, itemCode 버그 수정)
│   │       └── jobs.py             # 사람인 채용 (4 peer)
│   │
│   ├── pipeline/
│   │   ├── ingestion_graph.py      # ★ v3 5-노드 수집 파이프라인 (1시간)
│   │   └── delivery_graph.py       # 전달 파이프라인 (오전 8:30 이메일)
│   │
│   ├── rag/
│   │   ├── embedder.py             # BGE-M3 Dense+Sparse 원샷
│   │   ├── hybrid_search.py        # Qdrant RRF (Dense×Sparse Top-50 → Top-20)
│   │   └── reranker.py             # BGE-reranker-v2-m3 → Top-10
│   │
│   ├── db/
│   │   ├── article_store.py        # raw_articles · issue_cards · evidence_chain CRUD
│   │   ├── postgres.py             # SQLAlchemy 엔진·세션
│   │   └── qdrant_client.py        # Qdrant 클라이언트
│   │
│   └── schemas.py                  # Pydantic 모델 (ai-internal-api.yaml에서 생성)
│
└── tests/
```

---

## 모니터링 Peer 4사

| ID | 회사명 | 경쟁 강도 | 비고 |
|---|---|---|---|
| `samsung_sds` | 삼성SDS | 🔴 매우 높음 | AX 풀스택, OpenAI 리셀러 1호 |
| `lg_cns` | LG CNS | 🔴 매우 높음 | 팔란티어 파트너십, 에이전트웍스 |
| `hyundai_autoever` | 현대오토에버 | 🟡 높음 | 모빌리티 SI |
| `posco_dx` | 포스코DX | 🟡 높음 | 산업 DX |

비교 기준 — `sk_ax`: SK주식회사 지주(`corp_code 00181712`). **현재 SK 그룹 전체 매출**이라 매출 절대값 비교 시 스코프 차이 큼 (`_scope_warning` 마킹). 정확한 SK AX 부문 매출은 사업전략팀 내부 자료로 교체 필요.

---

## 크롤러 소스 — Track A / Track B

### Track A — 1시간 간격 (실시간 뉴스)

| 소스 | 신뢰도 | 수집 방법 | 일일 한도 |
|---|---|---|---|
| 네이버 뉴스 API | 0.75 | REST API + 본문 enrichment | 200 |
| Google News RSS | 0.65 | feedparser + httpx | 200 |
| ETnews (IT/산업/경제) | 0.70 | RSS, 3개 섹션 | 300(공유) |
| ZDNet Korea | 0.68 | RSS (feedburner) | 〃 |
| Bloter | 0.68 | RSS (feedburner) | 〃 |
| 연합뉴스 산업 | 0.85 | RSS | 200 |

### Track B — 매일 새벽 2시 (배치)

| 소스 | 신뢰도 | 수집 방법 | 4 peer 처리 |
|---|---|---|---|
| **DART** (금감원 공시) | 1.00 | OpenAPI `/api/list.json` | corp_code 4개 등록 |
| **KIPRIS** (특허) | 0.95 | 공공데이터 REST | 한글 출원인명 4개 |
| **공식 뉴스룸** SDS | 0.90 | Playwright + URL 슬러그 패턴 | samsung_sds 전용 |
| **공식 뉴스룸** LG CNS | 0.90 | 내부 fingerprint REST | lg_cns 전용 |
| **공식 뉴스룸** 현대오토에버 | 0.90 | Playwright generic ★ 신규 | best-effort 셀렉터 |
| **공식 뉴스룸** 포스코DX | 0.90 | Playwright generic ★ 신규 | 〃 |
| 한경 컨센서스 | 0.80 | Playwright (SPA) | 4 peer 검색 |
| 네이버 금융 리서치 | 0.75 | Playwright | itemCode 4개 (LG CNS 버그 수정) |
| 사람인 채용공고 | 0.50 | Saramin API | 4 peer 회사명 |

> **BigKinds, LinkedIn, 잡플래닛은 미구현** — BigKinds는 의도적으로 제외, LinkedIn/잡플래닛은 공식 API 미승인. Saramin이 채용공고 단일 소스.

### 크롤러 예외 처리 원칙
- HTTP 403/429 → 5분 대기 후 1회 재시도, 실패 시 SKIP + 로그
- 파싱 오류 → 원문 그대로 PostgreSQL 저장 (`processing_status='PARSE_ERROR'`)
- 타임아웃(10초) → SKIP + 로그
- 전역 일일 한도: 5,000건/일 (`DailyLimitGuard.GLOBAL_LIMIT`)
- robots.txt 위반 소스 크롤링 금지

---

## v3 수집 파이프라인 (`ingestion_graph.py`)

```
[ crawl ] → [ credibility ] → [ dedup ] → [ classify ] → [ issue_card ] → [ evidence ]
   ↓             ↓               ↓             ↓              ↓               ↓
원문 수집    Gate 2 신뢰도    Gate 3 중복    트렌드 섹터    GPT-4o 카드    검증 첨부 4종
PG 전량저장  Low/Unverified   코사인 0.90   + 결정적       3줄 요약        부착
            제외             클러스터링    노출도         + 시사점        (source/provenance/
                                                                          financial/mbb)
```

- 매시간 실행. 두 파이프라인은 PostgreSQL을 통해서만 데이터 교환.
- `_GPT_WORKERS=5`: 분류·카드 생성 GPT-4o 호출 병렬도 (rate limit 고려).
- 실패한 노드는 `state.errors`에 누적. `human_review_flags`에 ID 추가.

### 결정적 노출도 산식 (v3)

LLM 점수가 아닌 **추적·재현 가능**한 결정적 입력값 기반.

```
exposure_score = 0.40·cluster_size_norm
               + 0.30·credibility_max
               + 0.20·peer_mention_rate
               + 0.10·tier1_diversity

high     ≥ 0.70
medium   0.40 ~ 0.70
low      < 0.40
```

> v1의 LLM 5축(긴급·주목·참고)은 폐기. API 스키마는 호환을 위해 `importance` deprecated 표시 유지.

---

## v3 전달 파이프라인 (`delivery_graph.py`)

```
[ BriefingAgent ] → [ NotificationAgent ]
       ↓                     ↓
PostgreSQL에서        SMTP/SendGrid
이슈 카드 + Evidence   이메일 발송
조회·본문 구성        (오전 8:30)
```

> v3 변경: Slack Webhook → 이메일.

---

## 환각 방지 — Evidence Chain (v3 핵심)

모든 이슈 카드에 4종 검증 정보 자동 부착. 누락 시 `pass=false` + `human_review` 플래그.

| 컴포넌트 | 내용 | 생성 주체 |
|---|---|---|
| `source_links` | 원문 URL + 출처명 + credibility_score | EvidenceAgent |
| `provenance` | raw_article_ids, llm_model, prompt_version, run_at | EvidenceAgent |
| `financial_refs` | DART 공시번호 + QoQ/YoY 변화량 + segment narrative | **FinancialLinkerAgent** |
| `mbb_refs` | 컨설팅사 보고서 자동 매칭 | (W5에서 활성) |

추가 규칙:
- 출처에 없는 수치(금액·%·날짜) → 자동 Fail
- '확실하다' '반드시' 등 단정 표현 → 경고 플래그

> SC(Self-Consistency) 검증은 `/gen-search`에만 잔존(단발 응답이라 다중 생성 비교 가능). 동향 카드 환각 방지의 1차 방어선이 SC → Evidence Chain으로 이동.

---

## FinancialLinkerAgent — 차별화 핵심

뉴스 카드를 받아 Peer사 재무 시계열에서 관련 사업부·변화량을 찾아 첨부. **시사점(implication)을 대체하는 "숫자로 설명되는 팩트"**.

### 입력
- 카드의 `peer_id`, `sector`, `event_type`, `title`, `summary_lines`
- `data/peer_financials/{peer_id}.json` (DART OpenAPI 실수치 + segment 키워드)
- `data/sk_ax_financials.json` (비교 기준)

### 출력
```jsonc
{
  "linked": true,
  "segment": {"id": "ai", "name_ko": "AI/지능형", "match_keyword_hits": 2},
  "financial_refs": [
    {
      "period": "2025Q4",
      "metric_ko": "전체 매출",
      "value_krwbn": 19356,
      "delta_pct_qoq": 27.1,
      "delta_pct_yoy": -4.4,
      "dart_rcept_no": "20260316000832",
      "narrative": "2025Q4 매출 1.94조원 (QoQ +27.1%) / YoY -4.4%"
    }
  ],
  "vs_sk_ax": {
    "metrics": [
      {"key": "revenue_total",   "ratio_peer_over_skax": 0.06, "narrative": "..."},
      {"key": "revenue_yoy_pct", "gap_pp": -5.9, "narrative": "..."},
      {"key": "ai_engineers",    "ratio_peer_over_skax": 1.33, "narrative": "..."}
    ],
    "summary_one_liner": "vs SK AX (2025Q4): 매출 0.06x · YoY -5.9%p · AI 인력 1.33x"
  },
  "highlights": ["AI 인력 +310명", "vs SK AX: AI 인력 540→720명 (Peer 1.33x)"]
}
```

### vs SK AX 결정적 4지표
1. **전체 매출** — 절대 격차 + 배수
2. **매출 YoY 성장률** — %p 격차 (성장 속도)
3. **AI 매출 비중** — %p 격차 (현재 stub null, IR/PDF 파싱 후 채움)
4. **AI 엔지니어 수** — 배수·절대 격차

> PoC 단계: stub JSON 사용. W4 후반에 `peer_financials` 테이블이 axis-backend Flyway로 생성되면 DB 조회로 교체.

---

## 데이터 디렉토리 (`data/`)

### `data/peer_financials/{peer_id}.json`

DART OpenAPI 실수치 (총매출·영업이익 5분기) + segment/AI비중/headcount stub. 각 파일 구조:

```jsonc
{
  "peer_id": "samsung_sds",
  "_source": "dart_v1_total_only",
  "_dart_corp_code": "00126186",
  "_fetched_at": "2026-04-27",
  "policy": { "include_segments": ["its","cloud","ai"] },
  "segments": {
    "its":   {"name_ko": "IT서비스(ITS)", "keywords": ["ITS","SI","운영"]},
    "cloud": {"name_ko": "클라우드",       "keywords": ["클라우드","MSP"]},
    "ai":    {"name_ko": "AI/지능형",      "keywords": ["AI","FabriX","LLM"]}
  },
  "quarterly": [
    {
      "period": "2025Q4",
      "report_date": "2026-03-10",
      "dart_rcept_no": "20260310002989",
      "dart_url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260310002989",
      "_dart_fs_div": "CFS",
      "_derivation": "annual - (Q1+Q2+Q3)",
      "revenue_total_krwbn": 35368,
      "operating_profit_krwbn": 2261,
      "segment_revenue_krwbn": null,            // ← W5 PDF 파싱
      "ai_revenue_share_pct": null              // ← IR 자료
    }
  ],
  "headcount": [/* 추정치, W7 약한신호 감지기에서 보완 */]
}
```

| 필드 | 신뢰도 | 비고 |
|---|---|---|
| `revenue_total_krwbn` / `operating_profit_krwbn` | ✅ DART 검증 | 사업보고서 분기보고서 thstrm_amount 기반 |
| `segment_revenue_krwbn` | ❌ stub null | DART 정형 데이터에 없음 → W5에서 IR PDF 파싱 |
| `ai_revenue_share_pct` | ❌ stub null | 공식 공시 미존재 → IR 자료 입력 필요 |
| `headcount` | 🟡 추정치 | W7 잡공고 약한신호로 보완 |

### `data/sk_ax_financials.json` — 비교 기준

`corp_code 00181712` (SK주식회사 지주). **SK 그룹 전체 매출(에너지·통신 포함)**이라 SK AX 단독 부문 매출 아님. `_scope_warning` 마킹.

### `data/ir_samples/`

IR 분기·연간 PDF 샘플. IRParserAgent(W5) 입력.

---

## RAG 하이브리드 검색

```
쿼리 → BGE-M3 임베딩 (Dense + Sparse 원샷)
     → Qdrant 병렬 Prefetch
        ├── Dense 코사인 유사도 Top-50
        └── Sparse 내적 점수 Top-50
     → RRF Fusion: score(d) = 1/(k+rank_dense) + 1/(k+rank_sparse), k=60
     → 중복 제거 후 Top-20
     → 메타데이터 필터 (peer_id / event_type / sector / exposure_band / 날짜)
     → BGE-reranker-v2-m3 재랭킹 → Top-10
```

폴백:
- 임베딩 서버 타임아웃(3초) → BM25 폴백
- 결과 0건 → 기간 범위 2배 확장 후 재시도

### Qdrant 컬렉션 페이로드

원문 텍스트는 저장 금지. 메타데이터만.

```python
{
  "rdb_id": int,             # PostgreSQL FK
  "peer_id": str,            # samsung_sds | lg_cns | hyundai_autoever | posco_dx
  "event_type": str,         # 6종 taxonomy
  "sector": str,             # security | ai_tech | large_deal | sk_ax_biz | other
  "exposure_score": float,   # 0~1 결정적 산식
  "exposure_band": str,      # high | medium | low
  "credibility_score": float,
  "published_at": int,       # Unix timestamp
  "cluster_id": int,
  "source_name": str,
  "title": str, "summary": str,
}
```

- `axis_main` 컬렉션: 최근 3개월 RAG 검색용 (90일 TTL)
- `axis_history` 컬렉션: 1년치 시그널 히스토리 전용 (365일 TTL)

---

## FastAPI 내부 엔드포인트

SpringBoot에서만 호출. 8001 포트 외부 노출 금지.

| Method | Path | 설명 |
|---|---|---|
| `GET`  | `/health` | 헬스체크 (DB·Qdrant 연결 확인) |
| `POST` | `/pipeline/run` | 수집 파이프라인 실행 (비동기) |
| `POST` | `/pipeline/delivery` | 전달 파이프라인 실행 (이메일 발송) |
| `POST` | `/search` | BGE-M3 하이브리드 검색 |
| `POST` | `/gen-search` | Generative Search (RAG + GPT-4o + SC 검증) |
| `POST` | `/weak-signal/run` | 약한 신호 감지기 (W7 활성 예정) |

---

## 로컬 개발 세팅

```bash
# 1. 레포 클론
git clone https://github.com/SKALA-AXis/axis-ai.git
cd axis-ai

# 2. 환경변수 설정
cp .env.example .env
# OPENAI_API_KEY, DATABASE_URL, NAVER_CLIENT_ID/SECRET, DART_API_KEY 입력

# 3. 의존성 설치
uv sync

# 4. pre-commit 훅
uv run pre-commit install

# 5. DB·Qdrant 기동 (axis-infra 필요)
cd ../axis-infra && docker compose up -d postgres qdrant && cd ../axis-ai

# 6. AI 서버 실행
uv run uvicorn src.api.main:app --reload --port 8001

# 7. 헬스체크
curl http://localhost:8001/health

# (선택) 파이프라인 1회 실행 — 콘솔에 결과 출력
uv run python run_pipeline_once.py
```

---

## 환경 변수

```bash
# LLM
OPENAI_API_KEY=sk-...

# DB
DATABASE_URL=postgresql://axuser:axpass@localhost:5432/axis

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# 크롤러
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
DART_API_KEY=...
KIPRIS_API_KEY=...
SARAMIN_API_KEY=...           # 옵션 (없으면 채용공고 SKIP)

# 서버
AI_SERVER_PORT=8001
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 개발 명령어

```bash
uv run ruff format .          # 포맷
uv run ruff check .           # 린트
uv run mypy src/              # 타입 체크
uv run pytest tests/ -v       # 테스트
uv add 패키지명               # 의존성 추가 (pip install 금지)
```

---

## CI

GitHub Actions (`.github/workflows/ci.yml`) — push / PR 시 자동 실행.

```
uv sync
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```

> CI 상세 및 실패 대응: [axis-infra/docs/CI.md](https://github.com/SKALA-AXis/axis-infra/blob/develop/docs/CI.md)

---

## 절대 하지 말 것

- `pip install` 금지 → `uv add` 사용
- `uv.lock` 커밋 건너뛰기 금지 (환경 재현 보장)
- Qdrant 페이로드에 원문 전체 텍스트 저장 금지 (메타데이터만)
- 수집·전달 파이프라인을 같은 LangGraph 그래프에 묶기 금지
- Evidence Chain 4종 첨부 누락 시 카드 발행 금지 (`pass=false` 카드는 human_review로 분류)
- `.env` 파일 커밋 금지

---

## 참고: v1 → v3 전환 요약

| 영역 | v1 | v3 |
|---|---|---|
| 모니터링 Peer | 2사 | 4사 |
| 노출도 점수 | LLM 5축 (긴급/주목/참고) | 결정적 산식 (cluster·credibility·peer mention·tier diversity) |
| 환각 방지 | SC 검증 (3회 생성 후 2/3 일치) | **Evidence Chain 4종 부착** + SC는 `/gen-search`에만 |
| 시사점 | ImplicationAgent (LLM 생성) | **FinancialLinkerAgent** (재무 숫자 기반 팩트) |
| 약한 신호 | LLM 분석 | 결정적 트렌드 분석 (W7) |
| 알림 | Slack Webhook | 이메일 (SMTP/SendGrid) |
| 비교 기준 | 없음 | **vs SK AX 결정적 4지표** |
