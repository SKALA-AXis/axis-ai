# axis-ai

AXIS 서비스의 Python AI 서버입니다. 뉴스·공시·채용공고 크롤링, LangGraph 기반 AI 파이프라인, BGE-M3 하이브리드 검색, GPT-4o 이슈 카드 생성을 담당합니다.

> 전체 프로젝트 개요는 [axis-infra](https://github.com/skala-ai-13/axis-infra)를 참조하세요.

---

## 기술 스택

| 항목 | 내용 |
|---|---|
| 언어 | Python 3.11+ |
| 패키지 관리 | uv |
| 웹 프레임워크 | FastAPI 0.115.x |
| AI 파이프라인 | LangGraph 0.1.x |
| LLM | GPT-4o (langchain-openai) |
| 임베딩 | BGE-M3 (FlagEmbedding) — Dense + Sparse 원샷 |
| Reranker | BGE-reranker-v2-m3 (FlagEmbedding) |
| Vector DB | Qdrant 1.9.x |
| Raw DB | PostgreSQL 16.x (SQLAlchemy 2.x) |
| 린트·포맷 | ruff |
| 타입 체크 | mypy |
| 테스트 | pytest + pytest-asyncio |

---

## 프로젝트 구조

```
src/
├── api/
│   ├── main.py             uvicorn 진입점 (src.api.main:app)
│   └── router.py           FastAPI 엔드포인트 정의
├── agents/
│   ├── crawler_agent.py    뉴스·공시·채용공고 수집
│   ├── credibility_agent.py출처 신뢰도 분류 (High/Medium/Low/Unverified)
│   ├── dedup_agent.py      중복 제거 + 이슈 클러스터링
│   ├── classification_agent.py 중요도 분류 (긴급/주목/참고)
│   ├── issue_card_agent.py 이슈 카드 생성
│   ├── implication_agent.py SK AX 시사점 초안 생성
│   ├── validation_agent.py SC 검증 (환각 방지)
│   ├── weak_signal_agent.py약한 신호 감지
│   └── notification_agent.py Slack 알림 발송
├── pipeline/
│   ├── ingestion_graph.py  수집 파이프라인 (1시간마다)
│   └── delivery_graph.py   전달 파이프라인 (오전 8:30)
├── rag/
│   ├── embedder.py         BGE-M3 Dense+Sparse 임베딩
│   ├── hybrid_search.py    Qdrant RRF 하이브리드 검색
│   └── reranker.py         BGE-reranker-v2-m3 재랭킹
├── crawler/
│   ├── base_crawler.py     크롤러 베이스 클래스
│   ├── naver_crawler.py    네이버 뉴스 API
│   ├── dart_crawler.py     DART 공시 API
│   ├── rss_crawler.py      전자신문·ZDNet·IT조선 RSS
│   └── job_crawler.py      채용공고 (LinkedIn·잡플래닛)
├── db/
│   ├── postgres.py         SQLAlchemy 엔진·세션
│   └── qdrant_client.py    Qdrant 클라이언트
└── schemas.py              Pydantic 모델 (ai-internal-api.yaml 기반)
```

---

## 로컬 개발 세팅

```bash
# 1. 레포 클론
git clone https://github.com/skala-ai-13/axis-ai.git
cd axis-ai

# 2. 환경변수 설정
cp .env.example .env
# .env에서 OPENAI_API_KEY, DATABASE_URL, NAVER_CLIENT_ID 등 입력

# 3. 의존성 설치
uv sync

# 4. pre-commit 훅 설치
uv run pre-commit install

# 5. DB·Qdrant Docker 실행 (axis-infra 레포 필요)
cd ../axis-infra && docker compose up -d postgres qdrant && cd ../axis-ai

# 6. AI 서버 실행
uv run uvicorn src.api.main:app --reload --port 8001

# 7. 헬스체크 확인
curl http://localhost:8001/health
```

---

## FastAPI 내부 엔드포인트

SpringBoot에서만 호출합니다. 8001 포트는 외부 직접 노출 금지.

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/health` | 헬스체크 (DB·Qdrant 연결 확인) |
| `POST` | `/pipeline/run` | 수집 파이프라인 실행 (비동기) |
| `POST` | `/pipeline/delivery` | 전달 파이프라인 실행 (브리핑 생성) |
| `POST` | `/search` | BGE-M3 하이브리드 검색 |
| `POST` | `/gen-search` | Generative Search (RAG + GPT-4o) |
| `POST` | `/weak-signal/run` | 약한 신호 감지기 실행 |

---

## LangGraph 파이프라인 구조

```
수집 파이프라인 (ingestion_graph.py)  — 매시간 실행
SupervisorAgent
  ├── CrawlerAgent        → 원문 수집 → PostgreSQL 전량 저장
  ├── CredibilityAgent    → Gate 2: 신뢰도 분류
  ├── DeduplicationAgent  → Gate 3: 중복 제거·클러스터링
  ├── ClassificationAgent → 중요도 점수 산출
  ├── IssueCardAgent      → 이슈 카드 생성 (3줄 요약)
  ├── ImplicationAgent    → SK AX 시사점 초안
  └── ValidationAgent     → SC 검증 (3회 생성 후 2/3 일치 확인)

전달 파이프라인 (delivery_graph.py)  — 평일 오전 8:30 실행
  ├── BriefingAgent       → PostgreSQL에서 이슈 카드 조회
  └── NotificationAgent   → Slack Webhook 발송
```

두 파이프라인은 PostgreSQL을 통해서만 데이터를 교환합니다. 직접 호출 금지.

---

## RAG 하이브리드 검색 흐름

```
쿼리 → BGE-M3 임베딩 (Dense + Sparse 원샷)
     → Qdrant Dense Prefetch (Top-50) + Sparse Prefetch (Top-50)
     → RRF Fusion → Top-20
     → BGE-reranker-v2-m3 재랭킹 → Top-10 반환
```

폴백: 임베딩 타임아웃(3초) → BM25 폴백 / 결과 0건 → 기간 2배 확장 재시도

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

# 서버
AI_SERVER_PORT=8001
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

GitHub Actions (`.github/workflows/ci.yml`) — push / PR 시 자동 실행

```
uv sync
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```

> CI 상세 설명 및 실패 대응 방법: [axis-infra/docs/CI.md](https://github.com/SKALA-AXis/axis-infra/blob/develop/docs/CI.md)

---

## 주의사항

- `pip install` 사용 금지 — `uv add` 사용
- `uv.lock` 커밋 건너뛰기 금지 (환경 재현 보장)
- Qdrant 페이로드에 원문 전체 텍스트 저장 금지 (메타데이터만)
- 수집·전달 파이프라인 같은 LangGraph 그래프에 묶기 금지
- SC 검증(환각 방지) 생략 금지
- `.env` 파일 커밋 금지
