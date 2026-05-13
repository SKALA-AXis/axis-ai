# CardComposerAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `CardComposerAgent` (3-phase: summarize → analyze → compose) |
| **Supervisor** | Ingestion |
| **LangGraph node** | `card_news` (#6) |
| **상태** | ✅ 구현 (v1: 3 파일 분리 — IssueCardAgent + NewsSummaryAgent + NewsAnalysisAgent) → v2 통합 권장 |
| **현재 클래스** | `IssueCardAgent` (`src/agents/issue_card_agent.py`) — DB 테이블만 V9 에서 `card_news` 로 rename, **클래스 명은 function-named 라 유지** |
| **Trigger** | ClassificationAgent 후 — cluster 마다 1회 |

## 2. 책임

**한 줄**: classified cluster 로부터 frontend 가 표시할 card (title / 3줄 summary / implication / sources) 를 LLM 으로 생성하여 `card_news` 테이블에 INSERT.

**구체적 (3-phase)**:

1. **Phase 1 — Summarize** (`NewsSummaryAgent.summarize`) — 클러스터 article 3~5건의 사실만 추출. validation 플래그 (`is_valid_summary`) 부여.
2. **Phase 2 — Analyze** (`NewsAnalysisAgent.analyze`) — summary + classification + cluster metadata → SK AX 관점 가설/영향도 평가.
3. **Phase 3 — Compose** (`IssueCardAgent.generate`) — 위 결과를 frontend display schema (`CardNewsItem`) 로 조립.

산출물 = `card_news` row + 이후 EvidenceAgent 가 evidence_chain 부착.

## 3. 책임 NOT

- evidence_chain 부착 — EvidenceAgent (다음 노드)
- DB INSERT 자체는 EvidenceAgent 가 `save_card_news` 호출 (transactional)
- frontend display 변환 — **`CardNewsAgent` (`src/agents/card_news_agent.py`)** 가 별개 책임. CardComposerAgent (= 본 design 의 IssueCardAgent 3-phase) 가 DB row 를 만들고, CardNewsAgent 는 그 row 를 frontend `CardNewsItem` 스키마 (`axis-frontend/src/features/card-news/model/cardNews.ts`) 로 직렬화. 두 클래스 동시 존재 — 합치지 않음 (책임 분리)
- backend 의 CardNewsResponse 는 frontend 와 정합 보장

## 4. 입력 스펙

```python
class CardComposerInput(TypedDict):
    classified_clusters: list[ClassifiedCluster]   # ClassificationAgent output
    cluster_map: dict[int, list[int]]              # DedupAgent output (article_ids)
```

내부 fetch:
- 각 cluster 의 article 3건 (`get_articles_by_ids`, credibility DESC + published_at DESC 5건 까지)

## 5. 출력 스펙

```python
class CardNewsRow(TypedDict):
    id: str                          # 'IC-YYYYMMDD-NNN' 또는 'CN-YYYYMMDD-NNN'
    cluster_id: int
    company: str                     # peer_id (jsonb varchar)
    title: str                       # ≤ 500 chars
    summary_lines: list[str]         # 정확히 3줄
    event_type: str
    importance: str                  # exposure_band (legacy field name)
    importance_score: float          # exposure_score
    implication: dict                # v3 통합 jsonb (아래 §6.4)
    sources: list[dict]              # [{title, url, source_name, credibility, ...}]
    validation_pass: bool            # 첫 단계 validation (Phase 1)
    validation_sc_score: float       # 0 (current — SC 폐기됨, evidence_chain 으로 대체)

class CardComposerOutput(TypedDict):
    card_news: list[CardNewsRow]
```

## 6. 알고리즘

### 6.1 ID 생성

```python
def generate_card_id(date: date) -> str:
    """Format: 'CN-YYYYMMDD-NNN' — N is sequence within day"""
    prefix = f"CN-{date.strftime('%Y%m%d')}"
    seq = count_today_cards(prefix) + 1
    return f"{prefix}-{seq:03d}"
```

### 6.2 Phase 1 — Summarize (LLM gpt-4o-mini)

```text
PROMPT:
다음 클러스터의 기사들 (같은 이벤트 보도) 을 사실 위주로 3줄로 요약하라.

대표 기사: {title}
참고 기사 (최대 3): {others_titles}

3줄 요약 규칙:
1. 1번째 줄: WHO + WHAT (예: "삼성SDS가 LG CNS와 AX 협약 체결")
2. 2번째 줄: WHEN + WHERE / HOW (예: "5월 13일 서울 본사에서 발표")
3. 3번째 줄: 핵심 수치 또는 차별점 (예: "3년간 1,000억원 규모 공동 R&D")

본문 (대표): {content_preview_1500}

JSON 반환:
{
  "summary_lines": ["1번째 줄", "2번째 줄", "3번째 줄"],
  "is_valid_summary": true | false,    # 본문에 명확한 사실이 부족하면 false
  "extracted_facts": {
    "amounts": ["1,000억원"],         # 본문에 등장한 수치 (검증용)
    "dates": ["5월 13일"],
    "entities": ["삼성SDS", "LG CNS"]
  }
}
```

### 6.3 Phase 2 — Analyze (LLM gpt-4o-mini)

```text
PROMPT:
다음 사실 요약 + 분류 결과를 보고 SK AX 사업전략팀 관점의 시사점을 생성하라.

요약: {summary_lines}
event_type: {event_type}
sector: {sector}
exposure_band: {exposure_band}
클러스터 메타: cluster_size={N}, source_count={M}, credibility_max={c}

JSON 반환:
{
  "why_important": "한 문장 (왜 SK AX 가 주목해야 하는가)",
  "potential_impact": "한 문장 (어떤 영향을 줄 수 있는가)",
  "follow_up_questions": ["1", "2"],   # 추가 조사 질문 1~3개
  "suggested_actions": ["a", "b", "c"], # SK AX 액션 아이디어 2~4개
  "confidence": 0.0~1.0,               # 분석 신뢰도 (출처 부족하면 ↓)
  "out_of_evidence": ["불확실한 주장"]  # 본문에 없는 가정 (검증 X)
}
```

### 6.4 Phase 3 — Compose

```python
def compose(cluster, summary, analysis, articles) -> CardNewsRow:
    return {
        "id": generate_card_id(today),
        "cluster_id": cluster.cluster_id,
        "company": cluster.company[0] if cluster.company else "unknown",
        "title": summary["summary_lines"][0][:500],
        "summary_lines": summary["summary_lines"],
        "event_type": cluster.event_type,
        "importance": cluster.exposure_band,        # legacy field name
        "importance_score": cluster.exposure_score,
        "implication": {
            "sector": cluster.sector,
            "sectors": cluster.sectors,
            "exposure_score": cluster.exposure_score,
            "exposure_band": cluster.exposure_band,
            "why_important": analysis["why_important"],
            "potential_impact": analysis["potential_impact"],
            "follow_up_questions": analysis["follow_up_questions"],
            "suggested_actions": analysis["suggested_actions"],
            "confidence": analysis["confidence"],
            "signals": cluster.signals,             # exposure_score 산식 입력 보존
            # evidence_chain 은 다음 노드 (EvidenceAgent) 가 채움
        },
        "sources": [
            {
                "title": a.title,
                "url": a.url,
                "source_name": a.source_name,
                "credibility_score": a.credibility_score,
                "published_at": a.published_at.isoformat(),
            }
            for a in articles
        ],
        "validation_pass": summary["is_valid_summary"],
        "validation_sc_score": 0.0,  # SC 폐기됨, evidence_chain 으로 대체
    }
```

## 7. LLM 모델 + token 예산

| Phase | 모델 | token (in+out) | 일일 호출 | 일일 비용 |
|---|---|---|---|---|
| Summarize | gpt-4o-mini | ~2,000 | ~80 (cluster 수) | ₩200 |
| Analyze | gpt-4o-mini | ~1,500 | ~80 | ₩150 |
| Compose | (LLM 없음, 산식) | — | — | ₩0 |
| **합계** | | | | **₩350/일** |

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| Summarize 가 `is_valid_summary=false` | `validation_pass=false` 로 마킹 + downstream (BriefingService) 가 skip 또는 human_review |
| Analyze JSON parse 실패 | retry 1회 → 실패 시 minimal stub (`why_important="(분석 실패)"`, confidence=0.0) |
| LLM 타임아웃 (5초/phase) | phase 별 retry 1회 → 실패 시 stub |
| `extracted_facts` 가 본문에 없는 수치 포함 | EvidenceAgent 의 validation 단계에서 `out_of_evidence` 표면화 — 카드는 생성하되 confidence ↓ |
| 클러스터 article 0건 | skip (예외 상황) |

## 9. 외부 의존성

- **DB**: `raw_articles` (READ), `card_news` (INSERT — EvidenceAgent 가 호출)
- **외부 API**: OpenAI gpt-4o-mini
- **lib**: `openai`, `langchain_openai`

## 10. State 흐름 (LangGraph)

**소비**: `classified_clusters`, `cluster_map`
**생산**: `card_news: list[CardNewsRow]` (다음 노드 EvidenceAgent 가 evidence_chain 부착 + DB INSERT)

```python
@_logged_step("card_news", "classified_clusters", "card_news")
def card_news_node(state):
    from src.agents.issue_card_agent import IssueCardAgent
    from src.agents.news_summary_agent import PeerNewsSummaryAgent
    from src.agents.news_analysis_agent import PeerNewsAnalysisAgent
    agent = IssueCardAgent()
    summary_agent = PeerNewsSummaryAgent()
    analysis_agent = PeerNewsAnalysisAgent()
    ...
    return {**state, "card_news": cards}
```

## 11. Provenance + Confidence

- **Provenance**:
  ```json
  {
    "raw_article_ids": [123, 124, 125],
    "llm_model": "gpt-4o-mini",
    "prompt_version": "v3.0",
    "evidence_version": "v3.0",
    "run_at": "2026-05-13T09:00:00Z",
    "agent": "CardComposerAgent"
  }
  ```
- **Confidence**: `analysis["confidence"]` (Phase 2 출력). UI 가 < 0.6 일 때 경고 표시.

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit (Summarize) | "삼성SDS-LG CNS MOU" 본문 3건 | summary_lines 3줄 + is_valid_summary=true + amounts/dates 추출 |
| Unit (Analyze) | summary + sector='deal' | why_important / potential_impact 비어있지 않음, confidence ≥ 0.5 |
| Unit (Compose) | Phase 1+2 결과 입력 | CardNewsRow schema 완벽 매핑 |
| Edge | 본문이 광고/SKIPPED_QUALITY 가 섞임 | is_valid_summary=false |
| Edge | Analyze 가 본문에 없는 수치 (환각) 포함 | `out_of_evidence` 에 등록, EvidenceAgent 가 처리 |
| Integration | cluster 5개 → 5 card 생성 | 모든 card 의 implication 필드 채워짐 |

## 13. 모니터링

- **pipeline_logs.step**: `card_news`
- **KPI**:
  - 카드 생성 성공률 ≥ 95%
  - `is_valid_summary=false` 비율 ≤ 10%
  - 평균 confidence ≥ 0.70
  - `out_of_evidence` 발생 비율 ≤ 5% (환각 모니터)
- **token 예산**: ₩350/일

## 14. 구현 메모 + Changelog

### 의존 lib

```toml
openai = ">=1.30"
langchain-openai = ">=1.1"
```

### 핵심 파일 (v1 — 3 파일 분리)

- `src/agents/issue_card_agent.py` — Phase 3 (Compose) + 진입점
- `src/agents/news_summary_agent.py` — Phase 1 (`PeerNewsSummaryAgent`)
- `src/agents/news_analysis_agent.py` — Phase 2 (`PeerNewsAnalysisAgent`)
- LangGraph 호출 → `src/pipeline/ingestion_graph.py` `card_news_node`

### v2 통합 권장 (P9)

3 phase 를 단일 `CardComposerAgent` class 의 internal method 로 통합 — boilerplate 감소 + 일관 LLM client 관리. 다만 v1 처럼 phase 별 unit test 가능성 유지.

### Changelog

- **v1 (2026-04-W2)** — 3-phase 분리 구현
- **v2 (2026-04-W3)** — v3 메타데이터 (implication jsonb 통합)
- **v3 (2026-05-12)** — DB 테이블 `issue_cards` → `card_news` rename (V9). **클래스 명 `IssueCardAgent` 는 function-named 라 유지**, `card_news_agent.py` (frontend display 생성기) 와 collision 회피
- **v4 (proposed, P9)** — 3 파일 → 1 CardComposerAgent 통합
