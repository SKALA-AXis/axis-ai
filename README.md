# axis-ai

AXIS 서비스의 Python AI 서버. **뉴스·공시·채용공고 크롤링 → 로컬 JSON 저장/전처리 → LangGraph 5-노드 분석 파이프라인 → 이슈 카드 + 검증 첨부 4종 생성**까지 담당합니다.

> 전체 프로젝트 개요는 [axis-infra](https://github.com/SKALA-AXis/axis-infra) 참조.

---

## 🚀 실행 방법 (Cloud / Local 두 가지 모드)

axis-ai는 두 가지 DB 프로파일을 지원합니다. **둘 다 똑같은 코드**가 돌고, `.env` 파일만 다릅니다.

| 모드 | DB | Qdrant | 언제 쓰나 |
|---|---|---|---|
| **Cloud** (기본) | Supabase Postgres | Qdrant Cloud | 팀 공용 데이터, 데모, PR 검증 |
| **Local** | docker postgres | docker qdrant | 오프라인 작업, 스키마 실험, 비용 절감 |

### .env 파일 구성 (절대 커밋 금지)

```
axis-ai/
├── .env          ← Cloud 기본값 (= axis-infra/.env 와 동일 값)
└── .env.local    ← Local 컨테이너 모드 (호스트 → docker postgres/qdrant)
```

`.env`는 axis-infra/.env 의 Cloud 값과 **동일하게 유지**해야 합니다. (Single Source of Truth는 axis-infra)

### Mode 1 — Cloud 모드로 실행 (기본)

```bash
# 1. 의존성 설치
uv sync

# 2. .env 받기 (axis-infra/.env 의 Cloud 값을 그대로 복사)
#    팀 공용 .env는 노션/1Password 등에서 받으세요.

# 3. AI 서버 실행
uv run uvicorn src.api.main:app --reload --port 8001

# 4. 또는 단독 스크립트 (옵션 안 주면 .env = Cloud)
uv run python run_pipeline_once.py
uv run python run_crawler_once.py --track a
```

### Mode 2 — Local 컨테이너 모드로 실행

```bash
# 1. axis-infra 쪽에서 postgres·qdrant 컨테이너부터 띄우기
cd ../axis-infra
cp .env.local.example .env.local         # 처음 한 번만
docker compose --profile local --env-file .env.local up -d postgres qdrant
cd ../axis-ai

# 2. .env.local 준비 (.env 와 같은 값을 베이스로, DB·QDRANT만 localhost 로 교체)
#    템플릿은 .env.example 참고

# 3. --env local 플래그로 단독 스크립트 실행
uv run python run_pipeline_once.py --env local
uv run python run_crawler_once.py --track a --env local
```

> `--env local` 은 `.env.local` 을 `override=True` 로 로드합니다. `.env.local` 이 없으면 기본 `.env`(Cloud)로 폴백 — 자세한 동작은 [src/config/env_loader.py](src/config/env_loader.py) 참고.

### Mode 3 — Docker 컨테이너 안에서 ai 서비스로 실행

axis-infra 의 docker compose 가 env 를 컨테이너에 주입하므로 `.env` 파일은 무시됩니다.

```bash
# Cloud 모드 — Supabase + Qdrant Cloud 에 붙음
cd ../axis-infra
docker compose up -d ai

# Local 모드 — 같은 네트워크의 postgres/qdrant 컨테이너에 붙음
docker compose --profile local --env-file .env.local up -d
```

### 환경 설정 체크리스트 (팀원 신규 세팅)

- [ ] `uv sync` — Python 의존성 설치
- [ ] Cloud 용 `.env` 받기 (axis-infra/.env 와 동일 값)
- [ ] (옵션) Track B 크롤러 쓸 거면: `uv run playwright install chromium`
- [ ] (옵션) Local 모드 쓸 거면: `axis-infra` 에서 `--profile local` 컨테이너 기동 후 `.env.local` 작성
- [ ] `uv run pre-commit install` — ruff format/check 자동화

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
| **ParserAgent 추가** | PyMuPDF 기반 PDF/문서 payload 파싱 | IR·증권사 리포트·산업 동향 PDF에 공통 적용 |
| **크롤러 결과 JSON 통일** | `crawler_results/*.json` 배열 포맷 사용 | JSONL 대신 일반 JSON으로 저장/전처리 |
| **`company_tier` 추가** | `self`, `domestic`, `overseas` 구분 | SK AX=self, 기존 config 회사=domestic, 향후 global_companies=overseas |
| **전처리 runner 분리** | `run_preprocess_once.py`는 크롤링 없이 저장된 JSON만 처리 | `--source-type news`처럼 결과 파일 내부 source_type 기준 필터 |
| **homepage crawler 정리** | 미사용 `company_homepage` 계열 제거 | 공식/회사 뉴스는 `company_news` 흐름으로 관리 |
| **RSS 제거 / 글로벌 공식 뉴스룸 추가** | `rss` 소스 제거, `global_newsroom` 추가 | 해외 peer 공식 발표는 `source_type=official`로 수집 |
| **공통 PDF 파서 추가** | IR, 증권사 리포트, SPRi/BCG 등 PDF payload 추출 | page text/table/image 후보를 JSON extra에 보존 |
| **전처리 source별 라우팅** | 기사형/문서형/구조화 신호를 분리 | DART·IR·증권사 리포트는 관련도 판단 없이 parser quality 후 보존 |

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
├── run_crawler_once.py             # 크롤러 1회 실행 (Track A/B 선택)
├── run_all_once.py                 # DB 크롤링 → DB 전처리 순차 실행
├── run_local_crawler_once.py       # 크롤러 1회 실행 후 crawler_results/*.json 저장
├── run_preprocess_once.py          # 저장된 crawler JSON만 전처리하는 로컬 runner
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
│   └── ir_samples/                 # IR PDF 샘플 (ParserAgent 입력)
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
│   │   ├── parser_agent.py         # PDF/문서 payload 공통 파싱
│   │   ├── parser_quality_agent.py # 파서 결과 품질 점검
│   │   ├── notification_agent.py   # 이메일 발송 (Slack Webhook 폐기)
│   │   ├── sector_keywords.py      # 섹터 분류 키워드 사전
│   │   └── _deprecated/            # ← v1 에이전트 보관소
│   │       ├── implication_agent.py    # SK AX 시사점 (보류)
│   │       ├── validation_agent_sc.py  # SC 검증 (Evidence Chain으로 대체)
│   │       └── weak_signal_agent.py    # 약한 신호 (W7에 부활 예정)
│   │
│   ├── crawler/                    # ── 크롤러 ──
│   │   ├── base.py                 # SOURCE_CREDIBILITY, DailyLimitGuard, RawArticle
│   │   ├── base_crawler.py         # 크롤러 공통 부모 클래스
│   │   ├── batch_processor.py      # Track A/B 오케스트레이션 + DART CORP_CODES (4 peer)
│   │   ├── scheduler.py            # APScheduler + PEER_KEYWORDS (4 peer)
│   │   ├── result_writer.py        # crawler_results/*.json 저장 공통 유틸
│   │   ├── playwright_client.py    # 공통 Playwright 헤드리스 클라이언트
│   │   ├── article_filter.py       # HTML 문자열 정리 유틸
│   │   ├── parsers/
│   │   │   ├── article_content.py  # HTML 본문/이미지 추출
│   │   │   ├── content.py          # readability 본문 추출
│   │   │   ├── dedup.py            # URL 해시 중복 제거
│   │   │   ├── link_check.py       # URL 접근성 검사
│   │   │   └── pdf_payload.py      # PDF 텍스트/표 후보/이미지 후보 추출
│   │   ├── monitors/
│   │   │   └── keepalive.py        # Supabase/Qdrant keepalive
│   │   ├── local/                  # run_local_crawler_once.py 전용 로컬 실행 사본
│   │   └── sources/                # 파이프라인용 소스별 크롤러
│   │       ├── naver.py            # Naver News API
│   │       ├── dart.py             # DART OpenAPI 공시
│   │       ├── company_news.py     # 회사 공식 뉴스/뉴스룸
│   │       ├── global_newsroom.py  # 해외 peer 공식 뉴스룸(source_type=official)
│   │       ├── bcg.py              # BCG 산업 동향
│   │       ├── spri.py             # SPRi 산업 동향 PDF
│   │       ├── stock.py            # 시장 데이터
│   │       ├── keyword.py          # 네이버 데이터랩 검색 트렌드
│   │       ├── official.py         # 공식 뉴스룸 (SDS/LGCNS + hyundai/posco generic)
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

해외 peer는 [src/config/global_companies.py](src/config/global_companies.py)에 별도 정의되어 있으며 현재 `nvidia`, `apple`, `microsoft`, `google`, `amazon`, `meta`를 지원합니다. 해외 peer는 `company_tier=overseas`로 저장되고, 현재 수집 중심 소스는 `global_newsroom`입니다.

---

## 크롤러 소스 — Track A / Track B

### Track A — 1시간 간격 (실시간 뉴스)

| 소스 | 신뢰도 | 수집 방법 | 일일 한도 |
|---|---|---|---|
| 네이버 뉴스 API | 0.70 | REST API + 본문 enrichment + peer 필수 필터 | `news` 한도 |

RSS/Google News RSS는 현재 크롤러 흐름에서 제거되었습니다. 해외 peer 공식 발표는 Track B의 `global_newsroom`에서 `source_type=official`로 수집합니다.

### Track B — 매일 새벽 2시 (배치)

| 소스 | 신뢰도 | 수집 방법 | 4 peer 처리 |
|---|---|---|---|
| **DART** (금감원 공시) | 1.00 | OpenAPI 목록 + 원문 document XML 수집 | corp_code 등록 회사 |
| **공식 뉴스룸** SDS | 0.90 | Playwright + URL 슬러그 패턴 | samsung_sds 전용 |
| **공식 뉴스룸** LG CNS | 0.90 | 내부 fingerprint REST | lg_cns 전용 |
| **공식 뉴스룸** 현대오토에버 | 0.90 | Playwright generic ★ 신규 | best-effort 셀렉터 |
| **공식 뉴스룸** 포스코DX | 0.90 | Playwright generic ★ 신규 | 〃 |
| **글로벌 공식 뉴스룸** | 0.90 | NVIDIA/MS/Google 등 공식 뉴스룸 HTML | overseas peer |
| 네이버 금융 리서치 | 0.80 | PDF 링크 수집 + PDF payload 파싱 | 국내 peer |
| IR 자료 | 1.00 | 기업 IR PDF 수집 + PDF payload 파싱 | 국내 peer |
| 채용공고 | 0.60 | Work24/채용 API·페이지 | 국내 peer |
| 네이버 데이터랩 | 0.55 | 검색 트렌드 API | 구조화 신호 |
| 주가/시장 데이터 | 0.55 | 시장 데이터 API/페이지 | 구조화 신호 |
| BCG/SPRi 산업 동향 | 0.70 | HTML/PDF 산업 리포트 파싱 | 산업 동향 |

> **BigKinds, RSS, LinkedIn, 잡플래닛은 미사용** — BigKinds/RSS는 제거, LinkedIn/잡플래닛은 공식 API 미승인.

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

IR 분기·연간 PDF 샘플. ParserAgent 입력.

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

> Cloud / Local 모드 분기는 위 [🚀 실행 방법](#-실행-방법-cloud--local-두-가지-모드) 섹션 참조. 아래는 신규 팀원이 처음부터 환경을 세팅할 때 따라가는 절차입니다.

```bash
# 1. 레포 클론
git clone https://github.com/SKALA-AXis/axis-ai.git
cd axis-ai

# 2. 의존성 설치
uv sync

# 3. pre-commit 훅
uv run pre-commit install

# 4-A. Cloud 모드: 팀 공용 .env 받아서 axis-ai/.env 로 저장
#      (axis-infra/.env 와 같은 값 — Single Source of Truth는 axis-infra)

# 4-B. Local 모드: axis-infra 에서 컨테이너 기동 후 .env.local 작성
cd ../axis-infra
cp .env.local.example .env.local
docker compose --profile local --env-file .env.local up -d postgres qdrant
cd ../axis-ai
# axis-ai/.env.local 도 별도 작성 (호스트 → docker postgres/qdrant 용)

# 5. AI 서버 실행
uv run uvicorn src.api.main:app --reload --port 8001
curl http://localhost:8001/health

# 6. (선택) 파이프라인 1회 실행
uv run python run_pipeline_once.py            # Cloud
uv run python run_pipeline_once.py --env local # Local
```

---

## 단독 실행 스크립트 (로컬 디버깅)

크롤 → 전처리/파이프라인을 수동 실행할 수 있습니다. 운영 환경에서는 APScheduler가 자동으로 돌리지만, **새 변경사항을 한 번에 검증**할 때 유용합니다.

현재 로컬 디버깅 경로는 두 가지입니다.

| 경로 | 저장 위치 | 용도 |
|---|---|---|
| DB 경로 | PostgreSQL `raw_articles` | 실제 서비스 파이프라인 검증 |
| JSON 경로 | `src/crawler/crawler_results/*.json` | DB 저장 없이 크롤링 결과와 전처리 결과 확인 |

### 흐름 A — DB 기반 서비스 파이프라인

```
[1] run_crawler_once.py       →  raw_articles 테이블에 RAW 상태로 저장
                                  ↓
[2] run_pipeline_once.py      →  RAW 기사 → credibility → dedup → classify
                                  → issue_card → evidence → issue_cards 저장
```

크롤링과 전처리까지만 한 번에 실행하려면 `run_all_once.py`를 사용합니다.

```bash
uv run python run_all_once.py
uv run python run_all_once.py --track all --env local
uv run python run_all_once.py --company samsung_sds --company nvidia
```

`run_all_once.py`는 이슈카드/evidence/financial refs를 생성하지 않습니다. 최종 카드까지 만들고 싶을 때만 별도로 `run_pipeline_once.py`를 실행합니다.

### 흐름 B — JSON 기반 로컬 전처리

```
[1] run_local_crawler_once.py  →  src/crawler/crawler_results/*.json 저장
                                  ↓
[2] run_preprocess_once.py     →  저장된 JSON 로드
                                  → credibility → relevance → dedup → classification
                                  → src/crawler/crawler_results/preprocessed/*.json 저장
```

`run_preprocess_once.py`는 크롤링을 실행하지 않습니다. 이미 저장된 crawler JSON만 읽습니다.

### 1. DB 크롤러 단독 실행

```bash
uv run python run_crawler_once.py              # Track A만 (기본, 1~2분)
uv run python run_crawler_once.py --track a    # 명시적 Track A
uv run python run_crawler_once.py --track b    # Track B (5~10분)
uv run python run_crawler_once.py --track all  # A + B 순차

# DB 프로파일 전환 — 두 스크립트 모두 동일하게 지원
uv run python run_crawler_once.py --track a              # Cloud (기본 .env)
uv run python run_crawler_once.py --track a --env local  # .env.local 로드
```

| 트랙 | 소스 | 실행 시간 | 필요 환경 |
| --- | --- | --- | --- |
| **A** | 네이버 뉴스 | 1~2분 | NAVER_CLIENT_ID/SECRET |
| **B** | DART·IR·증권사 리포트·공식뉴스룸·글로벌 뉴스룸·채용·트렌드·시장 데이터 | 5~10분+ | DART_API_KEY, Playwright(`uv run playwright install chromium`), 소스별 API 키 |

출력: Peer별 / 소스별 신규 저장 건수 요약.

### 1-A. DB 크롤링 + 전처리 한 번에 실행

```bash
uv run python run_all_once.py                 # Track A+B 수집 후 전처리 실행
uv run python run_all_once.py --env local     # 로컬 DB에 저장 후 전처리 실행
uv run python run_all_once.py --track b       # Track B만 수집 후 전처리 실행
uv run python run_all_once.py --company nvidia
```

`run_all_once.py`는 내부에서 `run_crawler_once.py`를 먼저 실행하고, 성공한 경우에만 `run_pipeline_once.py --preprocess-only`를 이어서 실행합니다. 전처리 범위는 `credibility → source_type 라우팅 → relevance/parser quality → dedup/clustering → classification`까지입니다. DB에 쌓지 않는 JSON 검수 흐름(`run_local_crawler_once.py`/`run_preprocess_once.py`)과는 별개입니다.

### 2. JSON 크롤러 단독 실행

```bash
uv run python run_local_crawler_once.py
uv run python run_local_crawler_once.py --source naver_news --company samsung_sds
uv run python run_local_crawler_once.py --source dart --company lg_cns
uv run python run_local_crawler_once.py --source company_news
uv run python run_local_crawler_once.py --source global_newsroom --company nvidia
uv run python run_local_crawler_once.py --source official --company-tier overseas
uv run python run_local_crawler_once.py --source naver_news,global_newsroom,naver_research
uv run python run_local_crawler_once.py --source naver_datalab
```

결과는 `src/crawler/crawler_results/{source}_{YYYYMMDD_HHMMSS}.json` 형태로 저장됩니다. 각 row에는 공통적으로 `source_type`, `company`, `company_tier`, `title`, `content`, `url`, `published_at`, `collected_at` 등이 들어갑니다.

`--source`는 쉼표 구분을 지원합니다. `--company-tier overseas`를 같이 주면 `official`은 해외 공식 뉴스룸(`global_newsroom`)으로 해석됩니다. 예전 호환을 위해 `--source rss`도 `global_newsroom`으로 alias 처리되지만, 실제 RSS 크롤러는 사용하지 않습니다.

`company_tier`는 company별 구분값입니다.

| 값 | 의미 |
|---|---|
| `self` | 본인 회사. 현재 `sk_ax` |
| `domestic` | 현재 `companies.py` config에 있는 국내 peer |
| `overseas` | `global_companies.py`에 있는 해외 peer 또는 국내 config 밖 회사 |

### 3. JSON 전처리 단독 실행

```bash
# crawler_results/*.json 전체 전처리
uv run python run_preprocess_once.py

# crawler_results/*.json 전체에서 source_type=news만 전처리
uv run python run_preprocess_once.py --source-type news

# 여러 source_type 처리
uv run python run_preprocess_once.py --source-type news --source-type official

# 특정 JSON 파일만 전처리
uv run python run_preprocess_once.py \
  --input src/crawler/crawler_results/naver_news_20260507_172457.json
```

`--input`을 생략하면 `src/crawler/crawler_results` 바로 아래의 `.json` 파일을 모두 읽습니다. `preprocessed/` 안의 전처리 결과 파일은 다시 읽지 않습니다.

`--source-type`은 파일명이 아니라 JSON row 내부의 `"source_type"` 값을 기준으로 필터링합니다. 예를 들어 `naver_news_*.json` 안의 `"source_type": "news"` row만 처리하려면 `--source-type news`를 사용합니다.

전처리 단계에서 호출되는 agents:

| source_type | 전처리 흐름 |
|---|---|
| `news`, `official` | credibility → relevance → dedup/cluster → classification |
| `dart`, `ir`, `securities_report` | credibility → parser_agent → parser_quality_agent → auto relevant 보존 |
| `trend_report` | credibility → parser_agent → 산업 문서로 보존 |
| `job`, `market_data`, `search_trend`, `social` | credibility/source tagging 후 구조화 신호로 보존 |

`signal_agent.py`는 현재 전처리 흐름에서 제외되어 있습니다. 추후 급변/급증 탐지 단계에서 다시 사용할 예정입니다.

### 4. DB 파이프라인 단독 실행

```bash
uv run python run_pipeline_once.py             # Cloud (기본 .env)
uv run python run_pipeline_once.py --env local # .env.local 로드 (로컬 컨테이너 DB)
```

`processing_status='RAW'`인 기사를 모두 처리. **GPT-4o 호출이 클러스터 수만큼 발생**하므로 비용 주의 (대략 클러스터 1개당 ~₩30).

출력: 단계별 카운트(RAW→credible→cluster→cards) + 섹터·노출도 분포 + 카드별 상세(요약·검증체인·재무 highlights).

### 자주 만나는 상황

| 증상 | 원인 / 해결 |
|---|---|
| `RAW 기사 로드 \| count=0` | 새 기사 없음. `run_crawler_once.py` 먼저 실행 |
| `NAVER_CLIENT_ID 미설정` | `.env`에 키 입력 (없으면 해당 소스만 SKIP, 다른 소스는 계속 동작) |
| Playwright 미설치 (Track B) | `uv run playwright install chromium` |
| `소스별 수집 한도 초과` | DailyLimitGuard에 의한 정상 동작. 한도 조정은 [src/crawler/base.py](axis-ai/src/crawler/base.py) `SOURCE_TYPE_LIMITS` |
| 카드 0건인데 RAW는 있음 | dedup 단계에서 모두 기존 클러스터로 흡수됐을 가능성. 같은 RAW를 재처리하려면 DB에서 `processing_status='RAW'`로 리셋 필요 |

### DB 상태 빠른 확인

```bash
PGPASSWORD=axpass psql -h localhost -U axuser -d axis -c "
  SELECT processing_status, COUNT(*) FROM raw_articles GROUP BY processing_status;
"

PGPASSWORD=axpass psql -h localhost -U axuser -d axis -c "
  SELECT peer_id, COUNT(*) FROM issue_cards GROUP BY peer_id;
"
```

---

## 환경 변수

`.env` (Cloud 기본) 와 `.env.local` (Local 컨테이너) 의 차이는 DB·Qdrant 호스트뿐. 나머지는 동일.

```bash
# LLM
OPENAI_API_KEY=sk-...

# DB — Cloud (Supabase Transaction Pooler, sslmode=require)
DATABASE_URL=postgresql://postgres.<ref>:<pw>@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres?sslmode=require
# DB — Local (.env.local 에서 사용)
# DATABASE_URL=postgresql://axuser:axpass@localhost:5432/axis

# Qdrant — Cloud
QDRANT_HOST=https://<cluster-id>.<region>.aws.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=<qdrant-cloud-jwt>
# Qdrant — Local (.env.local 에서 사용; api_key 비움)
# QDRANT_HOST=localhost
# QDRANT_PORT=6333
# QDRANT_API_KEY=

# 크롤러
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
DART_API_KEY=...
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
