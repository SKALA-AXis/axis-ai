# AXIS — AI 파이프라인 컨텍스트 (axis-ai)

> 이 레포는 AXIS 서비스의 Python AI 서버입니다.
> LangGraph 에이전트, RAG 파이프라인, 크롤러, 임베딩을 담당합니다.
> 전체 프로젝트 맥락은 axis-infra/CLAUDE.md를 참조하세요.

---

## 이 레포의 책임

```
axis-ai가 하는 일:
├── 크롤러 (뉴스·공시·채용공고 수집)
├── 전처리·정제 (품질 필터, 신뢰도 분류, 중복 제거)
├── LangGraph AI 파이프라인 (수집→이슈카드 생성)
├── BGE-M3 임베딩 (Dense + Sparse 원샷)
├── Qdrant 하이브리드 검색 (RRF + Reranker)
├── GPT-4o API 호출 (분류·요약·시사점·Generative Search)
└── FastAPI 내부 서버 (SpringBoot에서만 호출 가능)
```

---

## 프로젝트 구조 (2026-06-11 실측)

```
axis-ai/
├── pyproject.toml / uv.lock     ← uv 관리 (pip 금지). torch 는 CPU 인덱스 핀
├── Dockerfile                   ← API 서버 (torch+cpu, Playwright). builder 스테이지는 PR CI 가 스모크
├── Dockerfile.cron              ← CronJob 경량 이미지 (--only-group cron, ~80MB)
├── src/
│   ├── api/                     ← FastAPI router(898줄) + 도메인별 *_schemas.py
│   ├── agents/                  ← LLM 에이전트 (주요: strategic_insight 9.2k줄·briefing_generation 7.3k줄
│   │   │                          ·mixer 2.7k·today_insight 2.6k·chat_orchestrator 2.4k·it_trend 1.5k·integration 1.2k)
│   │   └── context/             ← capability_evolution, weekly_digest 등 보조 에이전트
│   ├── pipeline/                ← analysis_flow_graph(분석 DAG)·delivery_graph(브리핑 본문)·supervisor_graph
│   ├── rag/                     ← embedder(BGE-M3)·hybrid_search·reranker·문서 인덱스
│   ├── crawler/                 ← base + sources/(매체별)·parsers/·monitors/
│   ├── preprocessing/           ← relevance(신뢰도)·dedup(클러스터링)·classification(섹터/노출도)
│   ├── analysis/ services/ composers/ extractors/ evaluators/ parsers/
│   ├── db/                      ← postgres·article_store·qdrant_client 등
│   ├── config/                  ← companies·sectors·event_types·openai_policy·env_loader
│   ├── middleware/ observability/
│   └── schemas.py
├── scripts/                     ← cron 진입점 (evaluate_recent_cards, postprocess_singleton_clusters) 외
└── tests/                       ← 36+ 파일. 표적 실행 권장 (전체는 CI)
```

> 분해 계획: 거대 에이전트 2개는 axis-infra/docs/structure-tasks/agent-split-design.md 참조.

---

## 기술 스택

```
Python           3.11+
패키지 관리       uv (pip 사용 금지)
웹 프레임워크     FastAPI 0.13x (lock 실측)
AI 파이프라인     LangGraph 1.2.x (lock 실측)
LangChain        langchain 1.2.x + langchain-core 1.2.x
LLM 연동         langchain-openai 1.1.x (GPT-4o)
LLM 모델         OpenAI GPT-4o (gpt-4o)
임베딩           BGE-M3 (BAAI/bge-m3) via FlagEmbedding 1.2.x
Reranker         BGE-reranker-v2-m3 via FlagEmbedding
Vector DB        Qdrant 1.9.x
Raw DB           PostgreSQL 16.x via SQLAlchemy 2.x
스케줄러          APScheduler 3.x
린트·포맷         ruff
타입 체크         mypy
테스트           pytest + pytest-asyncio
```

---

## FastAPI 내부 엔드포인트 (2026-06-11 실측)

SpringBoot에서만 호출합니다. 외부 직접 접근 불가 (8001 포트 외부 노출 금지).

```
GET  /healthz /health                          헬스체크 (경량/상세)
POST /pipeline/run                             수집 파이프라인 (매시간, 202 비동기)
POST /pipeline/delivery                        브리핑 본문 데이터 (발송은 backend SES)
POST /briefing/generate                        브리핑 생성
GET  /api/cards /api/cards/today               카드 조회
POST /chat /chat/pdf                           어시스턴트 (RAG 실구현 경로)
POST /today-insight/generate /insight/generate 인사이트
POST /mixer/analyze /mixer/analyze/stream      믹서 (SSE)
POST /global/trends/run                        글로벌 트렌드 (ITTrendAgent 5-phase)
POST /link/verify                              링크 검증
POST /weak-signal/run                          약한 신호 (주 1회)
POST /search /gen-search                       ⚠️ TODO 스텁 (빈 응답 — 실검색은 /chat 경로)
```

---

## LangGraph 에이전트 구조

### 분석 파이프라인 (pipeline/analysis_flow_graph.py — 실측 8노드)

```
issue_integrate → profile_context → build_analysis_context → strategic_insight → validate
  validate pass → assemble → card_writer → END
  validate fail → human_review → END

LLM 노드(issue_integrate/strategic_insight/card_writer)만 retry (1s 시작, 2배 backoff, 최대 2회)
수집(크롤링→relevance→dedup→classification)은 preprocessing/ 모듈이 담당, /pipeline/run 이 트리거
```

> ⚠️ 과거 문서의 "ingestion_graph.py 5노드"는 v3 설계안 — 해당 파일은 존재하지 않음 (2026-06-11 확인).

### 전달 파이프라인 (delivery_graph.py — v4: backend SES 통합)

```
BriefingAgent     PostgreSQL에서 동향 카드 + evidence_chain 조회 + 이메일 본문 구성
EmailAgent (v4)   본문 데이터 (subject / html / text / recipients) 반환 — 직접 발송 안 함
```

`/pipeline/delivery` endpoint 가 본문 데이터 반환 → **axis-backend 의 SesMailService** 가
AWS SES V2 SDK (IRSA + ses-mailer-sa) 로 발송. axis-ai 의 smtplib 발송 코드 폐기.

> v3 → v4 변경: Python smtplib SMTP 발송 → backend SES SDK 통합 (IRSA 인증).
> sender: `noreply@skala-ai.com` (매니저 SES verified domain).
> 자세한 spec: `axis-infra/docs/SES_INTEGRATION.md`.
> v3 변경 (history): Slack Webhook → 이메일. SC 검증 → evidence chain 첨부 4종.
> SC 검증은 /gen-search 에만 잔존.

### 두 파이프라인은 반드시 분리
```python
# ✅ 올바른 구조
ingestion_graph = StateGraph(IngestionState)   # 1시간마다
delivery_graph  = StateGraph(DeliveryState)    # 오전 8:30

# ❌ 절대 금지
# 수집과 전달을 같은 그래프에 묶는 것
```

---

## PipelineState 구조

```python
class IngestionState(TypedDict):
    peer_ids: List[str]
    trigger_type: str               # 'scheduled' | 'manual'
    raw_article_ids: List[int]      # PostgreSQL ID 목록
    credible_ids: List[int]
    cluster_map: dict               # {cluster_id: [article_ids]}
    representative_ids: List[int]
    classified_clusters: List[dict]
    card_news: List[dict]
    implications: List[dict]
    validation_results: List[dict]
    errors: Annotated[List[str], operator.add]
    human_review_flags: List[int]
```

---

## RAG 하이브리드 검색 흐름

```
1. 자연어 쿼리 수신
2. BGE-M3 원샷 → Dense 벡터(768차원) + Sparse 벡터 동시 생성
3. Qdrant 병렬 실행
   - Dense Prefetch: 코사인 유사도 상위 50건
   - Sparse Prefetch: 내적 점수 상위 50건
4. RRF Fusion: score(d) = 1/(k+rank_dense) + 1/(k+rank_sparse), k=60
5. 중복 제거 후 Top-20
6. 메타데이터 필터 (peer_id, event_type, importance, 날짜)
7. BGE-reranker-v2-m3 재랭킹 → Top-10
8. 반환

폴백:
- 임베딩 서버 타임아웃(3초) → BM25 폴백
- 결과 0건 → 기간 범위 2배 확장 후 재시도
```

---

## Qdrant 컬렉션 구조

```python
# main 컬렉션 (최근 3개월, RAG 검색용)
collection_name = "axis_main"
vectors: Dense(768, COSINE) + Sparse
TTL: 90일

# history 컬렉션 (1년치, 시그널 히스토리 전용)
collection_name = "axis_history"
vectors: Dense(768, COSINE) + Sparse
TTL: 365일

# 페이로드 구조 (메타데이터만, 원문 텍스트 저장 금지)
payload = {
    "rdb_id": int,              # PostgreSQL FK (원문 조회용)
    "peer_id": str,             # samsung_sds | lg_cns | hyundai_autoever | posco_dx
    "event_type": str,          # 6개 taxonomy
    "sector": str,              # 트렌드 섹터(코드 정본): ax | security | infra | deal | other
    "exposure_score": float,    # v3 결정적 산식 (0~1)
    "exposure_band": str,       # v3 노출도 밴드: high | medium | low
    "credibility_score": float,
    "published_at": int,        # Unix timestamp
    "cluster_id": int,
    "source_name": str,
    "title": str,               # 제목만 (본문 X)
    "summary": str,             # 3줄 요약만
}
```

---

## 크롤러 소스 계층

모니터링 대상 Peer 4사: `samsung_sds`, `lg_cns`, `hyundai_autoever`, `posco_dx`.

### Track A — 1시간 간격 (실시간 뉴스)

| 소스 | 신뢰도 | 수집 방법 | 비고 |
|---|---|---|---|
| 네이버 뉴스 API | 0.75 | REST API | peer 키워드별 검색 |
| Google News RSS | 0.65 | RSS | peer 키워드별 검색 |
| ETnews (IT/산업/경제) | 0.70 | RSS (feedparser) | 3개 섹션 |
| ZDNet Korea | 0.68 | RSS (feedburner) | |
| Bloter | 0.68 | RSS (feedburner) | |
| 연합뉴스 산업 | 0.85 | RSS | |

### Track B — 매일 새벽 2시 (배치)

| 소스 | 신뢰도 | 수집 방법 | 4 peer 처리 |
|---|---|---|---|
| DART (금감원 공시) | 1.00 | OpenAPI | corp_code 4개 등록 |
| KIPRIS (특허) | 0.95 | 공공데이터 REST | 한글 출원인명 4개 |
| 공식 뉴스룸 (SDS/LG CNS) | 0.90 | Playwright(SDS), 내부 API(LG CNS) | 매체별 전용 로직 |
| 공식 뉴스룸 (현대오토에버/포스코DX) | 0.90 | Playwright generic | best-effort 셀렉터 |
| 한경 컨센서스 | 0.80 | Playwright (SPA) | 4 peer 검색 |
| 네이버 금융 리서치 | 0.75 | Playwright | itemCode 4개 등록 |
| 고용24 (work24.go.kr) | 0.50 | work24 OpenAPI (`WORK24_API_KEY`) | 약한 신호 감지용 — 공채속보 + 채용공고 |

> BigKinds, LinkedIn, 잡플래닛, Saramin 은 미구현 — 의도적으로 제외.
> 고용24 (work24) 가 채용공고 단일 소스. 향후 LinkedIn 공식 API 승인 받으면 추가 검토.

### 크롤러 예외 처리 원칙
- HTTP 403/429 → 5분 대기 후 1회 재시도, 실패 시 SKIP + 로그
- 파싱 오류 → 원문 그대로 PostgreSQL 저장 (processing_status='PARSE_ERROR')
- 타임아웃(10초) → SKIP + 로그
- robots.txt 위반 소스 크롤링 금지

---

## 3단계 필터 게이트

```
Gate 1 (품질):
  - 본문 200자 미만 → SKIPPED_QUALITY
  - 인코딩 깨짐 (한글 비율 < 10%) → SKIPPED_QUALITY
  - 광고성 키워드 포함 → SKIPPED_QUALITY

Gate 2 (신뢰도):
  - Low 또는 Unverified 출처 → SKIPPED_CREDIBILITY
  - Qdrant 미삽입, PostgreSQL만 보관

Gate 3 (중복):
  - 코사인 유사도 ≥ 0.80 → 클러스터링 (rule-first 키 + 이벤트 버킷/시그니처 게이트 + LLM judge 병행)
  - 클러스터 내 대표 기사 1건만 Qdrant 삽입
  - 나머지는 cluster_id 부여 후 PostgreSQL만 보관

결과:
  수집 ~500건/일 → 필터 후 ~50건만 Qdrant 삽입 (10%)
```

---

## 노출도 산식 (구현 실측 — 2026-06-11)

```
exposure_score = 0.70·cluster_size_score + 0.30·company_mention_score
high ≥ 0.65   (src/preprocessing/classification.py)
```

> ⚠️ 1차 미팅 확정 스펙(0.40~0.50/0.30/0.20 다항)과 다름 — "코드가 맞다(문서 갱신)" vs
> "스펙 이탈(코드 수정)" **팀 결정 대기** (axis-infra PROJECT_STRUCTURE_PLAN §2.6).

---

## 환각 방지 — v3: Evidence Chain + SC

### 수집 파이프라인 (메인) — Evidence Chain 4종 첨부

모든 동향 카드에 자동 부착. 누락 시 `pass=false` + human_review 플래그.

```
1. source_links     원문 URL + 출처명 + credibility_score
2. provenance       raw_article_ids, llm_model, prompt_version, evidence_version, run_at
3. financial_refs   FinancialLinkerAgent: card sector·event·title 키워드로 segment 매칭
                    → QoQ/YoY 매출·영업익·AI 비중·인력 delta + DART 공시번호 + IR 페이지
4. mbb_refs         컨설팅사 보고서 자동 매칭 (W5)
```

추가 규칙:
- 출처에 없는 수치(금액·%·날짜) → 자동 Fail
- '확실하다' '반드시' 등 단정 표현 → 경고 플래그

### Generative Search (/gen-search) — SC 잔존

```python
# Self-Consistency: gen-search 응답에만 적용
1. 동일 쿼리로 답변 3회 독립 생성
2. 3개 결과의 핵심 주장 비교 (LLM 판정)
3. 일치율 ≥ 2/3 → Pass, 최빈 버전 선택
4. 일치율 < 2/3 → Fail → 해당 항목 공란 + Human 검토 플래그
```

> v3 변경: 동향 카드 환각 방지의 1차 방어선이 SC → Evidence Chain 4종으로 이동.
> SC는 응답이 단발성(생성 1회)인 /gen-search에만 적용.

---

## 이슈 카드 JSON 스키마

```python
IssueCard = {
    "id": "IC-20260420-001",          # 날짜 기반 ID
    "peer_id": "lg_cns",
    "cluster_id": 789,
    "title": "LG CNS, 팔란티어와 국내 제조·에너지 AX 파트너십 체결",
    "summary_lines": [                 # 반드시 3줄
        "1. ...",
        "2. ...",
        "3. ...",
    ],
    "event_type": "partnership",       # 6개 taxonomy 중 하나
    "sector": "ai_tech",               # v3 트렌드 섹터 5종 중 하나
    "exposure_score": 0.74,            # v3 결정적 산식 (0~1)
    "exposure_band": "high",           # v3 노출도 밴드: high | medium | low
    "implication": {
        "why_important": str,
        "potential_impact": str,
        "follow_up_questions": List[str],
        "suggested_actions": List[str],
        "evidence_chain": {            # v3 검증 첨부 4종 — 환각 방지의 핵심
            "source_links": [...],     # url + source_name + credibility_score
            "provenance": {...},       # raw_article_ids, llm_model, prompt_version, run_at
            "financial_refs": [...],   # FinancialLinkerAgent: segment QoQ/YoY + DART/IR refs
            "mbb_refs": [...],         # 컨설팅사 보고서 자동 매칭 (W5)
            "financial_link": {...},   # {linked, segment, highlights, headcount_delta}
            "pass": bool,              # 4종 모두 첨부됐는지
            "missing": List[str],      # pass=false 시 누락 항목
        },
    },
    "created_at": "ISO8601",
}
```

---

## 약한 신호 감지기 (차별화 기능)

AlphaSense·빅카인즈·딥서치가 '보도된 것'을 분석한다면,
AXIS는 '보도되기 6~12개월 전'을 감지합니다.

```
감지 패턴:
1. 채용공고 급증: 특정 기술 스택 언급 이번 주 > 지난 4주 평균 × 3배
2. 신규 직군 출현: 지난 3개월 0건 → 이번 달 3건 이상
3. 부서 집중 패턴: 특정 사업부 채용 비중 20% 이상 증가
4. 직급 패턴: 주니어보다 시니어/리더십 비중 갑자기 증가

출력:
- 신호 강도: weak | medium | strong
- 근거 원문 링크
- 해석 텍스트 + confidence
```

---

## LLM 비용 예측 (GPT-4o 기준)

| 작업 | 건수/일 | 토큰/건 (입출력) | 일일 비용 |
|---|---|---|---|
| 중요도 분류 | ~80건 | ~1,500 | ~$0.5 |
| 이슈 카드 생성 | ~80건 | ~2,000 | ~$0.65 |
| 시사점 생성 | ~50건 | ~3,000 | ~$0.6 |
| Generative Search | ~20회 | ~5,000/회 | ~$0.4 |
| **합계** | | | **~$2.15 (₩3,100)** |

> GPT-4o 기준: Input $2.5/1M tokens, Output $10/1M tokens
> LLM 호출 전 항상 토큰 수 추정 후 예산 초과 여부 확인

---

## 환경 변수 (.env)

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

# 내부 서버
AI_SERVER_PORT=8001

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 코드 스타일

```
push 전 CI 게이트 (전부 통과 후 push — 일부만 돌리고 push 금지):
  ruff check src/ && ruff format --check src/
  mypy src/ --ignore-missing-imports
  lint-imports                  # 모듈 경계 (agents→api 신규 유입 차단)
  pytest tests/<관련 파일> -v   # 표적 실행 (전체는 CI)
패키지 추가:  uv add 패키지명 (pip install 금지)
```

### 무거운 임포트 규칙 (2026-06-11 도입)

`from langchain_openai import ChatOpenAI` 를 **모듈 레벨에 두지 말 것** — langchain_core 가
transformers 풀체인(분 단위)을 끌어옴. ChatOpenAI 를 호출하는 함수 안에서 지연 임포트하고,
타입 어노테이션은 `if TYPE_CHECKING:` 블록 사용 (기존 17개 파일 전환 완료, FlagEmbedding 도 동일 패턴).

### Docstring 스타일 (Google 스타일)
```python
def embed_text(text: str) -> dict:
    """BGE-M3로 텍스트를 임베딩합니다.

    Args:
        text: 임베딩할 텍스트

    Returns:
        {'dense': np.ndarray, 'sparse': dict} 형태의 벡터

    Raises:
        EmbeddingTimeoutError: 임베딩 서버 응답 3초 초과 시
    """
```

### LLM 호출 패턴 (langchain-openai 사용)

```python
# openai SDK 직접 호출 대신 langchain-openai 사용
# LangGraph와 통합이 자연스럽고 LangSmith 트레이싱 자동 지원

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=2000,
)

# LangGraph 에이전트 내부에서
response = await llm.ainvoke(messages)
result = response.content
```

### pyproject.toml (최신 버전 기준)

```toml
[project]
name = "axis-ai"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.27",
    "langchain>=1.2",
    "langchain-core>=1.2",
    "langchain-openai>=1.1",      # GPT-4o 연동 필수
    "langgraph>=1.1",
    "FlagEmbedding>=1.2",          # BGE-M3, Reranker
    "qdrant-client>=1.9",
    "sqlalchemy>=2.0",
    "psycopg2-binary>=2.9",
    "openai>=1.30",
    "apscheduler>=3.10",
    "feedparser>=6.0",
    "beautifulsoup4>=4.12",
]

[dependency-groups]
dev = [
    "ruff>=0.5",
    "mypy>=1.10",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pre-commit>=3.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict_optional = true
```

---

## 절대 하지 말 것

- `pip install` 사용 금지 → `uv add` 사용
- 모듈 레벨 `from langchain_openai import ...` 금지 (위 지연 임포트 규칙)
- `git add -A` 금지 — 명시적 파일 목록만 (untracked WIP 휩쓸림 사고 2026-06-11)
- CronJob 신설 시 23:00–07:00 KST 창 금지 (노드 야간 셧다운 — infra ADR 0007)
- `uv.lock` 커밋 건너뛰기 금지
- Qdrant 페이로드에 원문 전체 텍스트 저장 금지
- 수집 파이프라인과 전달 파이프라인 같은 그래프에 묶기 금지
- 팀장 인터뷰 전 FR-006(시사점 방향) 코드 확정 금지
- 환각 방지 SC 검증 생략 금지
- `.env` 파일 커밋 금지
