# CredibilityAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `CredibilityAgent` |
| **Supervisor** | Ingestion |
| **LangGraph node** | `credibility` (#2) |
| **상태** | ✅ 구현 — `src/agents/credibility_agent.py` |
| **Trigger** | Ingestion `crawl` 노드 직후 |

## 2. 책임

**한 줄**: 각 raw_article 에 출처 기반 신뢰도 등급 (High / Medium / Low / Unverified) 부여 — LLM 없음, 산식 only.

**구체적**:

1. `source_name` → 출처 화이트리스트에서 base credibility_score lookup
2. 휴리스틱 보정 (예: 공식 도메인 + HTTPS + 게재일 명시 → boost)
3. `credibility_grade` enum 으로 매핑 (≥0.8=High, 0.5~0.8=Medium, 0.3~0.5=Low, <0.3=Unverified)
4. Low / Unverified row → `processing_status = 'SKIPPED_CREDIBILITY'` (Gate 2)

## 3. 책임 NOT

- 내용 사실 확인 (factual check) — 본 agent 는 출처 기반 only
- 광고/품질 필터 — ParserQualityAgent 가 Gate 1
- 관련성 (peer 와 관련 있는지) — RelevanceService 가 다음 노드

## 4. 입력 스펙

```python
class CredibilityInput(TypedDict):
    raw_article_ids: list[int]      # crawl 단계 output
```

내부적으로 `raw_articles` 의 `source_name`, `source_type`, `publisher`, `url`, `language` 조회.

## 5. 출력 스펙

```python
class CredibilityOutput(TypedDict):
    credible_ids: list[int]                 # credibility_grade in (High, Medium) 만
    skipped_credibility_ids: list[int]      # Low / Unverified
```

`raw_articles` 컬럼 갱신:

- `credibility_score: float` (0.0~1.0)
- `credibility_grade: str` (`'High' | 'Medium' | 'Low' | 'Unverified'`)
- `processing_status: 'CREDIBILITY_CHECKED' | 'SKIPPED_CREDIBILITY'`

## 6. 알고리즘

```python
SOURCE_BASE_SCORE = {
    # Tier 1 — 공식 채널
    "dart": 1.00,
    "kipris": 0.95,
    "samsung_sds_newsroom": 0.90,
    "lg_cns_newsroom": 0.90,

    # Tier 2 — 주요 언론
    "yonhap": 0.85,
    "naver_news": 0.75,
    "google_news": 0.65,

    # Tier 3 — IT 전문
    "etnews": 0.70,
    "bloter": 0.68,
    "zdnet_kr": 0.68,

    # Tier 4 — 약신호
    "saramin": 0.50,

    # Unknown
    "default": 0.30,
}

def credibility_score(article) -> float:
    base = SOURCE_BASE_SCORE.get(article.source_name, SOURCE_BASE_SCORE["default"])

    # 휴리스틱 보정
    bonus = 0
    if article.url.startswith("https://") and is_official_domain(article.url):
        bonus += 0.05
    if article.published_at is not None:
        bonus += 0.03
    if article.publisher and is_known_publisher(article.publisher):
        bonus += 0.02

    return min(base + bonus, 1.0)

def credibility_grade(score: float) -> str:
    if score >= 0.80: return "High"
    if score >= 0.50: return "Medium"
    if score >= 0.30: return "Low"
    return "Unverified"
```

## 7. LLM 모델 + token 예산

- **LLM 미사용** — 산식 only
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| `source_name` 화이트리스트 miss | `default=0.30` → `Low` 또는 `Unverified` 자동 처리 |
| `publisher` null | bonus 0, base 만 |
| Pattern: 갑자기 새 source 가 자주 등장 | admin alert (월간 통계) — 화이트리스트에 추가 검토 |

## 9. 외부 의존성

- **DB**: `raw_articles` (UPDATE credibility_score · credibility_grade · processing_status)
- **lib**: 없음 (산식 only)

## 10. State 흐름 (LangGraph)

**소비**: `raw_article_ids`
**생산**: `credible_ids` (Gate 2 통과만)

## 11. Provenance + Confidence

- **Provenance**: `raw_articles.metadata.credibility_version` (예: "v3.0")
- **Confidence**: `credibility_score` 자체가 confidence proxy (downstream agent 가 가중치로 사용 — 예: ClassificationService 의 `credibility_max`)

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | source_name='dart' | grade='High', score≥0.95 |
| Unit | source_name='unknown_blog' | grade='Unverified', score=0.30 |
| Unit | https + 공식 domain | bonus +0.05 적용 |
| Integration | raw_articles 100건 | credible_ids 비율 ≥ 60% |

## 13. 모니터링

- **pipeline_logs.step**: `credibility`
- **KPI**: credible 비율 60~80% (너무 낮으면 source 화이트리스트 점검)
- **token 예산**: 해당 없음

## 14. 구현 메모 + Changelog

### 핵심 파일

- `src/agents/credibility_agent.py`
- 화이트리스트: `src/config/source_credibility.py` (제안 — 현재는 dict literal in agent file)

### Changelog

- **v1 (2026-04-W1)** — 기본 화이트리스트
- **v2 (2026-04-W3)** — 휴리스틱 보정 추가
- **v3 (proposed)** — 화이트리스트를 별도 config 로 분리 + admin 페이지에서 GUI 수정
