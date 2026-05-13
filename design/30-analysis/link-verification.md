# LinkVerificationAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `LinkVerificationAgent` |
| **Supervisor** | Analysis |
| **상태** | 🟡 backend fixture (`POST /api/cards/{id}/verify-link`), axis-ai 신규 |
| **Trigger** | User 카드 상세에서 "출처 검증" 클릭 |

## 2. 책임

**한 줄**: 카드의 source URL 들을 HTTP HEAD/GET 으로 liveness 체크 + content hash diff 로 변경 감지.

**구체적**:

1. card.sources[*].url 각각 HTTP HEAD (timeout 5초)
2. 200 OK → live, 404/410 → dead, 3xx → redirect (final url 기록), 4xx/5xx → error
3. (옵션) live 인 경우 GET 으로 body fetch → SHA256 hash vs 원래 저장된 hash 비교
4. content 변경됐으면 (예: 기사가 수정됨) → diff 마킹
5. 결과 `link_verification_logs` 테이블에 저장 (감사 추적)

## 3. 책임 NOT

- 원문 내용 재크롤 — CrawlerAgent (다음 cycle 에서)
- LLM 분석 — 산식 only
- 변경 감지 후 자동 재처리 — manual review 흐름

## 4. 입력 스펙

```python
class LinkVerificationInput(TypedDict):
    card_id: str
```

내부 fetch: `card_news.sources` (JSONB)

## 5. 출력 스펙

```python
class LinkStatus(TypedDict):
    url: str
    status: Literal["live","dead","redirected","error"]
    http_code: int | None
    final_url: str | None       # redirect 시
    content_changed: bool       # GET 시 hash diff
    last_modified: str | None
    checked_at: datetime

class LinkVerificationOutput(TypedDict):
    card_id: str
    sources: list[LinkStatus]
    overall_status: Literal["all_live","some_dead","all_dead","content_changed"]
    verified_at: datetime
```

frontend `POST /api/cards/{id}/verify-link` 응답.

## 6. 알고리즘

```python
import httpx
import hashlib

async def verify(card_id):
    card = fetch_card(card_id)
    statuses = []
    for src in card.sources:
        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
                head = await client.head(src["url"])
                final = src["url"]
                http_code = head.status_code
                if 300 <= http_code < 400:
                    final = head.headers.get("location", src["url"])
                    head = await client.head(final)
                    http_code = head.status_code
                status = "live" if http_code == 200 else "dead" if http_code in (404,410) else "error"

                # content diff (옵션)
                content_changed = False
                if status == "live" and src.get("original_content_hash"):
                    body = (await client.get(final)).text
                    current_hash = hashlib.sha256(body.encode()).hexdigest()[:16]
                    content_changed = current_hash != src["original_content_hash"]

                statuses.append({
                    "url": src["url"],
                    "status": status if not content_changed else "live (content_changed)",
                    "http_code": http_code,
                    "final_url": final if final != src["url"] else None,
                    "content_changed": content_changed,
                    "last_modified": head.headers.get("last-modified"),
                    "checked_at": datetime.utcnow(),
                })
        except (httpx.TimeoutException, httpx.RequestError) as e:
            statuses.append({
                "url": src["url"], "status": "error", "http_code": None,
                "final_url": None, "content_changed": False,
                "last_modified": None, "checked_at": datetime.utcnow(),
            })

    overall = "all_live" if all(s["status"].startswith("live") and not s["content_changed"] for s in statuses) \
        else "content_changed" if any(s["content_changed"] for s in statuses) \
        else "all_dead" if all(s["status"] in ("dead","error") for s in statuses) \
        else "some_dead"

    save_verification_log(card_id, statuses)  # link_verification_logs
    return {"card_id": card_id, "sources": statuses, "overall_status": overall, "verified_at": datetime.utcnow()}
```

## 7. LLM 모델 + token 예산

- **LLM 미사용**
- 토큰 예산: ₩0

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| timeout 5초 | status='error' (request 실패) |
| robots.txt 차단 | HEAD 차단 시 status='error' + log |
| TLS error | status='error' |
| 대량 요청 (10+ sources) | 병렬 httpx.AsyncClient 활용 |

## 9. 외부 의존성

- **lib**: `httpx>=0.27`
- **DB**: `card_news.sources` (READ), `link_verification_logs` (INSERT, 신규 V14 migration)

## 10. State 흐름

stateless (on-demand request).

## 11. Provenance + Confidence

- **Provenance**: link_verification_logs.checked_at + agent_version
- **Confidence**: deterministic (HTTP 결과)

## 12. 테스트 시나리오

| Unit | 200 OK + hash unchanged | status='live', content_changed=false |
| Unit | 404 | status='dead' |
| Unit | 301 → 200 | status='live', final_url 채워짐 |
| Unit | timeout | status='error' |
| Integration | 5 sources, 1 dead | overall='some_dead' |

## 13. 모니터링

- KPI:
  - 일일 호출 latency 평균 ≤ 3초
  - dead link 비율 (전체 카드 중 sampling) ≤ 5%
- token: ₩0

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/link_verification_agent.py` (신규 P7)
- 신규 테이블: `link_verification_logs` (V14 migration)

### Changelog

- **v1 (제안, P7)** — HTTP HEAD + 옵션 GET hash diff
