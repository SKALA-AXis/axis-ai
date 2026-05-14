# ProvenanceTrackerMiddleware — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `ProvenanceTrackerMiddleware` (cross-cutting decorator + storage spec) |
| **Supervisor** | (none — 모든 agent 에 주입) |
| **상태** | ✅ 부분 구현 — `EvidenceAgent._build_provenance()` (`src/agents/evidence_agent.py`) 가 evidence_chain 한정 부착 중. 다른 agent 출력에는 미부착 → P9 표준화 |
| **Trigger** | 모든 LLM-derived 출력 생성 시점 |

## 2. 책임

**한 줄**: 모든 AI 산출물에 "어떤 모델·prompt·코드 버전·입력으로 만들었는지" 추적 메타데이터 (provenance) 를 자동 부착하여 reproducibility + 환각 감사 보장.

**구체적**:

1. **자동 수집** — runtime 변수 (LLM model, prompt version, git_sha, run_at, agent class name)
2. **입력 추적** — raw_article_ids, cluster_id, source_card_ids 등 산출 origin
3. **JSON 직렬화** — jsonb 컬럼에 저장 가능한 형태 (timestamps ISO8601, ids primitive)
4. **persist** — `evidence_chain.provenance` (현재) + `briefing_reports.provenance` / `insight_results.provenance` (확장)
5. **read** — admin /api/admin/audit-logs + per-card detail UI 의 "출처 확인" 패널

## 3. 책임 NOT

- **본문 fact 검증** — EvidenceAgent 의 source_links / out_of_evidence 책임
- **점수화** — ConfidenceScoreMiddleware
- **사용자 활동 추적** — AuditLogMiddleware (admin / login 등 user action)

## 4. 입력 스펙

```python
class ProvenanceContext(TypedDict):
    agent_class: str                # "CardComposerAgent"
    agent_method: str               # "compose" / "summarize" / "analyze"
    inputs: dict                    # {"raw_article_ids": [...], "cluster_id": 7, ...}
    llm_model: str | None           # "gpt-4o" / "gpt-4o-mini" / None (deterministic)
    prompt_version: str | None      # "ic-v3.0" / "anal-v2.1"
    extra: dict                     # agent-specific (예: parser_strategy='dart_html')
```

## 5. 출력 스펙 — 표준 Provenance schema

```python
class Provenance(TypedDict):
    # Identification (5 base fields — README §5 spec)
    llm_model: str | None
    prompt_version: str | None
    run_at: str                     # ISO8601 UTC
    raw_article_ids: list[int]      # 산출 origin
    git_sha: str                    # CI/CD env 에서 capture (`AXIS_GIT_SHA`)

    # Observability link (spec: axis-infra/docs/OBSERVABILITY_LANGFUSE.md)
    langfuse_trace_id: str | None       # Langfuse root trace (drill-down 진입점)
    langfuse_observation_id: str | None # 본 산출의 핵심 LLM span (LangChain handler 의 generation id)
    request_id: str | None              # FastAPI middleware UUID — BE 의 MDC request_id 와 일치

    # Extensions (agent-specific, optional)
    agent: str                      # "CardComposerAgent"
    agent_method: str | None        # "summarize" / "analyze" / "compose"
    evidence_version: str | None    # evidence chain spec 버전 (v3.0)
    parser_strategy: str | None     # Parser 가 사용한 strategy
    parser_version: str | None      # parser_agent.py 버전 태그
    cluster_id: int | None          # Dedup 결과 cluster
    source_card_ids: list[str] | None  # Mixer / Insight 입력 카드
    confidence_components: dict | None # ConfidenceScoreMiddleware 와 연동
    sc_iter: int | None             # Answer agent
    retry_count: int                # 0 이 정상
```

저장 위치별 mapping:

| 저장 위치 | 채우는 agent | 형태 |
|---|---|---|
| `evidence_chain.provenance` jsonb | EvidenceAgent (ingestion 매시) | Provenance 전체 |
| `briefing_reports.provenance` jsonb | BriefingGenerationAgent (user POST) | Provenance + briefing_type 필드 |
| `mixer_results.provenance` jsonb (V12+) | MixerAnalysisAgent | Provenance + source_card_ids |
| `insight_results.provenance` jsonb (V12+) | InsightCascadeAgent | Provenance + level (4단계) |
| `weak_signal_cards.metadata.provenance` | WeakSignalAgent | Provenance + detection_version |

## 6. 알고리즘

### 6.1 git_sha 캡처

```python
import os, subprocess

def _get_git_sha() -> str:
    # 우선순위: env (CI) > git command (local) > "unknown"
    if sha := os.environ.get("AXIS_GIT_SHA"):
        return sha[:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

# Pod 부팅 시 1회 캡처 후 module-level cache
_GIT_SHA_CACHE = _get_git_sha()
```

### 6.2 Decorator wrapping

```python
from functools import wraps

def with_provenance(agent_class: str, method: str = "run"):
    def deco(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            t0 = time.time()
            result = await func(*args, **kwargs)
            prov = {
                "agent": agent_class,
                "agent_method": method,
                "llm_model": kwargs.get("_llm_model"),
                "prompt_version": kwargs.get("_prompt_version"),
                "run_at": datetime.now(UTC).isoformat(),
                "git_sha": _GIT_SHA_CACHE,
                "raw_article_ids": kwargs.get("raw_article_ids", []),
                "retry_count": kwargs.get("_retry_count", 0),
            }
            if isinstance(result, dict):
                result.setdefault("provenance", {}).update(prov)
            return result
        return wrapper
    return deco

# 사용
@with_provenance("CardComposerAgent", "analyze")
async def analyze(cluster, **kwargs): ...
```

### 6.3 prompt_version 관리

각 agent 파일 상단:

```python
PROMPT_VERSION = "ic-v3.0"     # agent + version 형태, 변경 시 +0.1
# major 변경 시 +1 (예: prompt 골격 자체 교체)
```

Prompt 변경 + commit 시 PR template 에 `prompt_version_bump: [ic-v3.0 → ic-v3.1]` 체크박스 강제.

### 6.4 LangfuseTraceLinker — 3-tier observability 의 Tier 3 → output 매핑

`02-prompt-design-checklist.md` §4 의 3-tier observability 표준 (trail / steps / trace_id) 중 **Tier 3 (langfuse_trace_id)** 를 runtime 에 매핑해 출력 schema 에 박는 sub-middleware. analysis agent (Mixer / Insight / PeerComparison / GlobalTrends / Briefing) 의 출력에 `langfuse_trace_id` + `reasoning_steps[].langfuse_observation_id` 자동 채움.

```python
from langfuse import get_client

def link_langfuse_trace(output: dict) -> dict:
    """analysis agent output 에 Tier 3 trace_id + Tier 1/2 step 별 observation_id 매핑.
    LLM call 직후 (with_provenance decorator 안에서) 호출됨.

    - single LLM call agent (Mixer / Insight / Peer / Global): 모든 trail/steps 의
      observation_id = 현재 generation_id (동일).
    - multi LLM call agent (Briefing 의 section 별 call): linker 가 각 section
      generation 시점에 sub-call 되어 section 별 trace_id 매핑.
    """
    client = get_client()
    trace_id = client.get_current_trace_id()
    if not trace_id:
        return output  # Langfuse 비활성 (local dev) — silently skip

    output["langfuse_trace_id"] = trace_id
    obs_id = client.get_current_observation_id()

    for step in output.get("reasoning_steps", []):
        step.setdefault("langfuse_observation_id", obs_id)
    for trail_step in output.get("reasoning_trail", []):
        trail_step.setdefault("langfuse_observation_id", obs_id)
    return output
```

`with_provenance` decorator (§6.2) 가 LLM call 종료 직후 `link_langfuse_trace(result)` 호출.

### 6.5 사용자 노출 vs admin 노출 분리

3-tier 중 Tier 3 (langfuse_trace_id) 는 **admin only** — 시스템 프롬프트 / 전체 응답 / token 비용이 Langfuse trace 에 보이기 때문. FastAPI response serializer 에서:

```python
def serialize_for_user(output: dict, is_admin: bool) -> dict:
    """일반 사용자 응답에서 trace_id 제거 (시스템 프롬프트 노출 방지)."""
    out = dict(output)
    if not is_admin:
        out.pop("langfuse_trace_id", None)
        # reasoning_steps / trail 안의 langfuse_observation_id 도 strip
        for step in out.get("reasoning_steps", []):
            step.pop("langfuse_observation_id", None)
        for t in out.get("reasoning_trail", []):
            t.pop("langfuse_observation_id", None)
    return out
```

admin 역할 판정은 BE 가 JWT claim 기반 처리 후 axis-ai 에 `X-Axis-Role: admin` 헤더 전달.

## 7. LLM 모델 + token 예산

- **LLM 미사용** — pure middleware
- 토큰: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| `git_sha` 캡처 실패 (pod 내 git 없음) | "unknown" 저장 + 로그 경고 1회 |
| jsonb serialize 실패 (non-primitive) | 해당 필드만 drop + 로그 |
| Provenance 누락된 카드가 DB 에 존재 (legacy) | 마이그레이션 X — read 시점에 `provenance = {}` 처리 |
| `prompt_version` 미선언 | "unversioned" 저장 + 로그 |

## 9. 외부 의존성

- 환경변수: `AXIS_GIT_SHA` (Helm chart 가 image build SHA 주입)
- DB column: `evidence_chain.provenance` (✅ 존재) + 신규 5 테이블 (briefing_reports / mixer_results / insight_results / chat_sessions / weak_signal_cards 의 metadata jsonb)

## 10. State 흐름

```python
# IngestionState 의 모든 결과 dict (card_news[i], evidence_results[i]) 가
# provenance 필드를 키로 가짐.
state["card_news"] = [
    {
        "id": "CN-...",
        ...,
        "implication": {
            ...,
            "evidence_chain": {
                "source_links": [...],
                "provenance": {...},   # ← here
                ...
            }
        }
    }
]
```

## 11. Provenance + Confidence

본 middleware 의 출력 자체가 다른 agent 의 `provenance` 필드. ConfidenceScoreMiddleware 의 `confidence_components` 도 본 provenance dict 안에 nested 저장.

## 12. 테스트 시나리오

| 유형 | 시나리오 | 검증 |
|---|---|---|
| Unit | `_get_git_sha` (CI env) | env 값 우선 |
| Unit | `_get_git_sha` (env 없음, git 있음) | git rev-parse 결과 |
| Unit | `@with_provenance` decorator | result dict 에 provenance 키 자동 추가 |
| Unit | non-dict 반환 agent | 무영향 (그대로 return) |
| Integration | EvidenceAgent → DB 저장 → 재읽기 | 모든 field round-trip OK |
| Edge | prompt_version 미정의 | "unversioned" 저장 |
| Edge | jsonb 직렬화 실패 (numpy array) | 해당 필드 drop, 나머지 보존 |

## 13. 모니터링

- KPI:
  - provenance 누락 row 비율 ≤ 1% (1시간 batch 후 SELECT count(*) WHERE provenance IS NULL OR provenance = '{}')
  - `git_sha = 'unknown'` 비율 0% (배포 시 환경변수 주입 확인)
- Grafana: `axis_ai_provenance_missing_total` counter

## 14. 구현 메모 + Changelog

### 핵심 파일

- 현재: `src/agents/evidence_agent.py` 의 `_build_provenance()` (evidence 한정)
- 신규 P9: `src/middleware/provenance.py` — decorator + git_sha cache
- 신규 P9: 각 agent 파일 상단 `PROMPT_VERSION = "..."` 상수 통일

### Helm chart 변경

```yaml
# axis-infra/helm/axis-ai/values.yaml
env:
  - name: AXIS_GIT_SHA
    value: {{ .Values.image.tag }}   # CI 에서 commit SHA 를 image tag 로 사용
```

### Backend 연동

- `GET /api/admin/audit-logs` — provenance 의 git_sha · llm_model · prompt_version 컬럼 표시
- `GET /api/cards/{id}` 응답의 `evidence_chain.provenance` (이미 jsonb 그대로 전달 — schema 보존 확인 필요)

### Changelog

- **v1 (2026-04-W3)** — EvidenceAgent 단독 구현
- **v2 (제안, P9)** — 표준 decorator + 5 신규 jsonb column 확장
