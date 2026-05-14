# EvidenceAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `EvidenceAgent` |
| **Supervisor** | Ingestion |
| **LangGraph node** | `evidence` (#7) |
| **상태** | ✅ 구현 — `src/agents/evidence_agent.py` |
| **Trigger** | CardComposerAgent 후 — card_news row 마다 1회 |

## 2. 책임

**한 줄**: 각 card_news row 에 **evidence_chain 4종** (source_links / provenance / financial_refs / mbb_refs) 을 부착하여 환각 방지 + 추적 가능성 확보.

**구체적**:

1. **source_links** (산식) — sources[] 를 trim 하여 evidence_chain.source_links 에 복제
2. **provenance** (산식) — LLM model + prompt_version + run_at + raw_article_ids + git_sha 기록
3. **financial_refs** — FinancialLinkerAgent (sub) 호출 → segment QoQ/YoY delta + DART rcept_no + IR page
4. **mbb_refs** — 컨설팅사 보고서 자동 매칭 (W5+, BCG/McKinsey)
5. **pass 평가** — 4종 모두 부착 = pass=true, 누락 = missing[] 마킹 + human_review_flag
6. `save_card_news` + `save_evidence_chain` 호출하여 DB 영속

## 3. 책임 NOT

- card 본문 생성 — CardComposerAgent (이전 노드)
- LLM 자유형 사실 추출 — 산식 + sub-agent 의 deterministic 책임
- Qdrant 인덱싱 — EmbedIndexAgent (다음 노드)

## 4. 입력 스펙

```python
class EvidenceInput(TypedDict):
    card_news: list[CardNewsRow]    # CardComposer output
    cluster_map: dict[int, list[int]]
```

## 5. 출력 스펙

```python
class EvidenceResult(TypedDict):
    card_id: str
    source_links: list[dict]        # [{title, url, source_name, credibility_score}]
    provenance: dict                # llm_model, prompt_version, run_at, raw_article_ids, git_sha, evidence_version
    financial_refs: list[dict]      # [{period, metric, value_krwbn, delta_pct_qoq, delta_pct_yoy, dart_rcept_no, ir_page, narrative}]
    mbb_refs: list[dict]            # [{firm, title, url, published_date}]
    financial_link: dict | None     # {linked, segment, highlights, headcount_delta}
    evidence_version: str           # 'v3.0'
    pass: bool
    missing: list[str]              # 누락 항목 (예: ["mbb_refs"])

class EvidenceOutput(TypedDict):
    evidence_results: list[EvidenceResult]
    human_review_flags: list[int]   # pass=false 인 cluster_id
```

DB 영속:
- `card_news` row INSERT (card_news 테이블)
- `evidence_chain` row INSERT/UPDATE (PK = issue_card_id (legacy column 명), card 1:1)

## 6. 알고리즘

### 6.1 attach() — 진입점

```python
def attach(card: CardNewsRow, cluster_article_ids: list[int]) -> EvidenceResult:
    # 1) source_links — sources[] 복제 + trim
    source_links = [{
        "title": src["title"][:200],
        "url": src["url"],
        "source_name": src["source_name"],
        "credibility_score": src.get("credibility_score", 0.0),
    } for src in card["sources"][:5]]

    # 2) provenance
    provenance = {
        "raw_article_ids": cluster_article_ids[:10],
        "llm_model": "gpt-4o-mini",  # or "gpt-4o" depending on phase
        "prompt_version": "v3.0",
        "evidence_version": "v3.0",
        "run_at": datetime.utcnow().isoformat(),
        "git_sha": GIT_SHA,
        "agent": "EvidenceAgent",
        "cluster_id": card["cluster_id"],
    }

    # 3) financial_refs (sub-agent)
    financial_link = None
    financial_refs = []
    if card.get("event_type") in ("ma", "tech", "new_biz") or card.get("sector") == "deal":
        financial_linker = FinancialLinkerAgent()
        result = financial_linker.link(
            peer_id=card["company"],
            sector=card["sector"],
            event=card["event_type"],
            title=card["title"],
        )
        financial_link = result.get("link")           # {linked, segment, highlights, headcount_delta}
        financial_refs = result.get("refs", [])

    # 4) mbb_refs (sub-agent, W5+)
    mbb_refs = []
    if card.get("sector") in ("ax", "infra"):
        mbb_matcher = MbbMatcherAgent()
        mbb_refs = mbb_matcher.match(card)

    # 5) pass / missing
    missing = []
    if not source_links: missing.append("source_links")
    if not provenance: missing.append("provenance")
    if not financial_refs and card.get("event_type") in ("ma", "deal"): missing.append("financial_refs")
    if not mbb_refs and card.get("sector") in ("ax", "infra"): missing.append("mbb_refs")

    return EvidenceResult(
        card_id=card["id"],
        source_links=source_links,
        provenance=provenance,
        financial_refs=financial_refs,
        mbb_refs=mbb_refs,
        financial_link=financial_link,
        evidence_version="v3.0",
        pass=(len(missing) == 0),
        missing=missing,
    )
```

### 6.2 환각 방지 룰

- `extracted_facts.amounts` 의 수치가 source_links[*].url 의 원문에 등장하지 않으면 → `out_of_evidence` 마킹 (CardComposer 가 미리 검증 가능)
- "확실하다 / 반드시 / 분명히" 단정 표현 포함 시 warning flag (현재 manual review 만, 자동 거부 X)
- 동일 cluster_id 가 이전 cycle 에 이미 evidence_chain row 가지면 UPDATE (ON CONFLICT)

### 6.3 Prompt audit — 02-prompt-design-checklist 17 요소

Evidence 는 LLM 미사용 (산식 + sub-agent 위임). 필수 1, 4, 11, 12, 14.

| # | 요소 | 충족 위치 | 비고 |
|---|---|---|---|
| **1** | 역할 정의 | (LLM 미사용) | sub-agent (FinancialLinker/IRParser) 가 자기 prompt 에서 충족 |
| **4** | 출처 우선순위 | source_links[].credibility_score + tier1_diversity carry | Tier1>Tier2>Tier3 우선 |
| **11** | 정량 수치 우선 | financial_refs[] segment QoQ/YoY delta + headcount_delta | DART/IR 수치만 carry |
| **12** | 공식 vs 추정 구분 | provenance.evidence_version='v3.0' + source_links[].source_name 마킹 | DART = 공식, 자체 산식 = 추정 |
| **14** | 출력 형식 | EvidenceChain TypedDict + pass/missing 필드 | 4종 missing 강제 표면화 |

→ **5/5 필수 충족** (LLM 미사용이라 1번은 sub-agent 가 책임). 환각 방지의 v3 핵심 — checklist 11/12 의 "정량 + 공식 vs 추정" 강제 적용 지점.

## 7. LLM 모델 + token 예산

- **본 agent: LLM 미사용** — 산식 + sub-agent 위임
- 토큰 예산 (sub 포함): ~₩200/일

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| FinancialLinkerAgent 가 segment 못 찾음 | financial_link=null, financial_refs=[] + missing 에 추가 |
| MbbMatcherAgent W5 미활성 | mbb_refs=[] (의도된 미구현) |
| DB INSERT 실패 (evidence_chain) | rollback + retry 1회 → 실패 시 `errors` append, card 보존 (다음 cycle 재시도) |
| sources[] 가 0개 | source_links=[] + missing=["source_links"] + pass=false |

## 9. 외부 의존성

- **DB**: `card_news` (INSERT), `evidence_chain` (UPSERT), `raw_articles` (READ)
- **Sub-agent**: FinancialLinkerAgent (`src/agents/financial_linker_agent.py`)
- **Sub-agent (W5+)**: MbbMatcherAgent / IRParserAgent

## 10. State 흐름 (LangGraph)

**소비**: `card_news`, `cluster_map`
**생산**: `evidence_results`, `human_review_flags`

## 11. Provenance + Confidence

- **Provenance**: 본 agent 가 provenance 자체를 채움 (evidence_chain.provenance jsonb)
- **Confidence**: `pass: bool` 이 거시 confidence. UI 가 `pass=false` 시 ⚠️ 인 human_review 마킹.

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | 모든 sub-agent OK | pass=true, missing=[] |
| Unit | event_type='ma' + financial_link=null | missing=['financial_refs'], pass=false |
| Unit | sector='ax' + mbb_refs=[] (W5 미활성) | missing=['mbb_refs'], pass=false (의도) |
| Integration | 5 card → 5 evidence_chain INSERT | DB row 5 + pipeline_logs 1 entry |
| Edge | DB unique constraint violation | UPDATE (ON CONFLICT) |

## 13. 모니터링

- **pipeline_logs.step**: `evidence`
- **KPI**:
  - pass=true 비율 ≥ 70%
  - missing 분포 (sampling): financial_refs 부재 비율 ~30% (improvement target — segment 매칭 강화)
  - `out_of_evidence` 누적 ≤ 5% (CardComposer 단계에서 검증)
- **token 예산**: 본 agent ₩0, sub-agent 포함 ~₩200/일

## 14. 구현 메모 + Changelog

### 핵심 파일

- `src/agents/evidence_agent.py` — 본 agent
- `src/agents/financial_linker_agent.py` — sub
- `src/agents/ir_parser_agent.py` — sub (W5 부터)
- DB save: `src/db/article_store.py` 의 `save_card_news`, `save_evidence_chain`

### Changelog

- **v1 (2026-04-W2)** — source_links + provenance 만
- **v2 (2026-04-W3)** — financial_refs + financial_link 추가 (FinancialLinkerAgent 분리)
- **v3 (2026-05-W1)** — mbb_refs 추가 (W5)
- **v3.1 (2026-05-12)** — V9 column rename 후에도 evidence_chain.issue_card_id 컬럼 유지 (V10 분리). Python SQL 의 column 이름 그대로 (placeholder 만 :card_news_id 로 변경)
