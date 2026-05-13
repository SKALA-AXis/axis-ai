# WeakSignalAgent — Design Plan

## 1. 메타

| 항목 | 값 |
|---|---|
| **이름** | `WeakSignalAgent` (PatternDetect + AnomalyDetection + AlertRouting 통합 3-phase) |
| **Supervisor** | WeakSignal (단일 agent, 자체 supervisor) |
| **상태** | 🔴 W7+ 재구축 — `_deprecated/weak_signal_agent.py` 일부 로직 재활용 |
| **Trigger** | Spring @Scheduled 월 09:00 KST + on-demand POST `/weak-signal/run` |

## 2. 책임

**한 줄**: 채용공고 / 특허 / MOU 공시 / 발표 톤 시계열에서 **선행지표 패턴** + **이상 탐지** 후 사용자 정의 Alert Rule 매칭하여 알림 발송.

**구체적 (3-phase)**:

1. **PatternDetect** — 산식 (이동평균 + 임계값)
   - 채용 급증 (특정 기술 스택 언급 이번 주 > 지난 4주 평균 × 3배)
   - 신규 직군 출현 (지난 3개월 0건 → 이번 달 3건 이상)
   - 부서 집중 (특정 사업부 채용 비중 +20%)
   - 직급 패턴 (시니어/리더십 비중 급증)
2. **AnomalyDetection** — z-score (산식) + LLM 해석 (mini)
   - 발표 톤/방향의 7일 이동평균 vs 30일 평균 → z > 2.5 → 이상치
   - LLM 으로 1줄 해석 ("XX 기업이 갑자기 보안 강조 → 침해 사고 의심")
3. **AlertRouting** — 사용자 정의 rule (`/api/alerts/rules`) 매칭
   - rule = {keyword, peer, threshold, channels: ['email', 'in_app']}
   - 감지된 signal 의 strength × keyword 매칭 → 알람 대상 사용자 결정
   - 채널별 발송 (이메일 = SesMailService 재사용, in_app = `alerts` 테이블 INSERT)

## 3. 책임 NOT

- 데이터 수집 (채용/특허/공시) — Ingestion supervisor 의 Track B (sub-source: saramin, kipris, dart)
- 정상 카드 생성 — IngestionSupervisor (W7+ 의 약신호는 별도 `weak_signal_cards` 테이블 권장)

## 4. 입력 스펙

```python
class WeakSignalInput(TypedDict):
    window_days: int           # 기본 90 (3개월)
    peer_id: str | None        # None = 전체
    signal_types: list[str]    # ['hiring','patent','mou','announcement'] 기본 all
```

## 5. 출력 스펙

```python
class DetectedSignal(TypedDict):
    signal_type: Literal["hiring_surge","new_role","dept_concentration","seniority_shift","tone_anomaly"]
    peer_id: str
    strength: Literal["weak","medium","strong"]
    confidence: float
    detected_at: datetime
    evidence: list[dict]       # raw_article_ids + 산식 값
    interpretation: str         # LLM 1줄 해석

class WeakSignalOutput(TypedDict):
    signals: list[DetectedSignal]
    alerts_dispatched: int
    routing_log: list[dict]    # [{rule_id, user_id, channel, signal_id}]
    provenance: dict
```

신규 spec endpoint: `GET /api/weak-signals?since=2026-04-01&peer=samsung_sds`

## 6. 알고리즘

### 6.1 PatternDetect (산식)

```python
def detect_hiring_surge(peer_id, window_days):
    weekly_counts = fetch_job_postings_by_week(peer_id, weeks=8)
    if len(weekly_counts) < 4:
        return None
    last_4_weeks_avg = np.mean(weekly_counts[:-1])
    this_week = weekly_counts[-1]
    if this_week >= last_4_weeks_avg * 3:
        return DetectedSignal(
            signal_type="hiring_surge",
            peer_id=peer_id,
            strength="strong" if this_week >= last_4_weeks_avg * 5 else "medium",
            confidence=0.85,
            detected_at=datetime.utcnow(),
            evidence=[{"weekly_counts": list(weekly_counts), "ratio": this_week/last_4_weeks_avg}],
            interpretation="",  # AnomalyDetection 단계에서 채움
        )

def detect_new_role(peer_id):
    # 지난 3개월 0건 → 이번 달 3+ 건
    ...

def detect_dept_concentration(peer_id):
    # 부서 별 채용 비중 변화 +20%p
    ...

def detect_seniority_shift(peer_id):
    # 시니어/리더십 비중 변화
    ...
```

### 6.2 AnomalyDetection (산식 + LLM)

```python
def detect_tone_anomaly(peer_id):
    # 7일 이동평균 vs 30일 평균 z-score
    cards = fetch_peer_cards(peer_id, since=30d)
    daily_tone = [sentiment_score(c) for c in cards]  # -1~+1
    rolling_7d = np.mean(daily_tone[-7:])
    rolling_30d = np.mean(daily_tone)
    z = (rolling_7d - rolling_30d) / np.std(daily_tone)
    if abs(z) > 2.5:
        # LLM 해석
        interp = llm_interpret_anomaly(peer_id, cards[-7:], z)
        return DetectedSignal(
            signal_type="tone_anomaly",
            peer_id=peer_id,
            strength="strong" if abs(z) > 3.0 else "medium",
            confidence=min(abs(z) / 3.0, 1.0),
            ...
            interpretation=interp,
        )
```

LLM 해석 prompt:
```text
{peer} 의 최근 7일 발표 톤이 지난 30일 평균에서 z={z:.2f} 만큼 벗어남.

최근 7일 카드 제목:
{recent_titles}

한 줄로 해석 (50자 이내):
"왜 이런 변화가 일어났는지 + SK AX 의 시사점"

JSON: {"interpretation": "..."}
```

### 6.3 AlertRouting

```python
def route_alerts(detected_signals):
    rules = fetch_alert_rules()
    routing_log = []
    for sig in detected_signals:
        for rule in rules:
            if matches_rule(sig, rule):
                for channel in rule["channels"]:
                    if channel == "email":
                        SesMailService.send_alert(rule["user_email"], sig)
                    elif channel == "in_app":
                        insert_alert(rule["user_id"], sig)
                    routing_log.append({"rule_id": rule.id, "user_id": rule.user_id, "channel": channel, "signal_id": sig.id})
    return routing_log
```

`matches_rule`:
```python
def matches_rule(sig, rule):
    if rule["peer_id"] and sig.peer_id != rule["peer_id"]: return False
    if rule["min_strength"] and STRENGTH_ORDER[sig.strength] < STRENGTH_ORDER[rule["min_strength"]]: return False
    if rule["keywords"] and not any(kw in sig.interpretation for kw in rule["keywords"]): return False
    return True
```

## 7. LLM 모델 + token 예산

- **모델**: gpt-4o-mini (해석 phase 만)
- 호출수: ~5/주 (이상치 detected 만, 평균)
- 토큰: ~2,000 (in 1,500 + out 500)
- **주간 비용**: ~₩50 → 일평균 ₩7

## 8. 에러 처리

| 시나리오 | 대응 |
|---|---|
| 채용 데이터 0건 (Saramin fail) | hiring_surge skip + log |
| z-score 계산 시 std=0 (단조 톤) | z=0 처리, anomaly 없음 |
| LLM 해석 fail | "(해석 실패)" stub |
| Alert rule 매칭 0 | alerts_dispatched=0, log only |
| SesMailService fail | retry 1회 → 실패 시 in_app 만 |

## 9. 외부 의존성

- **DB**: `raw_articles` (Track B의 job postings), `card_news` (발표 톤), `weak_signal_cards` (신규 **V11** migration), `alerts` (INSERT), `alert_rules` (READ)
- **외부 API**: OpenAI gpt-4o-mini (해석)
- **재사용**: SesMailService (BE Java, REST 호출 또는 직접 DB INSERT)

## 10. State 흐름

WeakSignalSupervisor 의 단일 LangGraph 그래프 (3-phase sequential):

```python
class WeakSignalState(TypedDict):
    window_days: int
    peer_ids: list[str]
    detected_patterns: list[DetectedSignal]   # phase 1
    detected_anomalies: list[DetectedSignal]  # phase 2
    routing_log: list[dict]                    # phase 3
    errors: list[str]
```

## 11. Provenance + Confidence

- **Provenance**: `weak_signal_cards.metadata.detection_version='v1', phase_versions={...}`
- **Confidence**: signal_type 별 산식 + LLM 신뢰도

## 12. 테스트 시나리오

| Unit | hiring 평균 5건/주, 이번주 20건 | hiring_surge strong, ratio=4.0 |
| Unit | hiring 평균 5건/주, 이번주 6건 | None (3배 미달) |
| Unit | tone z=3.5 | tone_anomaly strong + LLM 해석 |
| Unit | rule={peer:'samsung_sds', keywords:['보안']}, signal interpretation 에 '보안' 포함 | rule match → alert dispatch |
| Edge | LLM fail | interpretation="(해석 실패)" + signal 보존 |

## 13. 모니터링

- KPI:
  - 주간 detected signals 수 (~5건 예상)
  - false positive 비율 (사용자 피드백 기반, sampling) ≤ 20%
  - Alert 발송 성공률 ≥ 95%
- token: ~₩7/일

## 14. 구현 메모 + Changelog

- 핵심 파일: `src/agents/weak_signal_agent.py` (신규 P8) — `src/agents/_deprecated/weak_signal_agent.py` 의 패턴 매칭 로직 재활용
- 신규 테이블: `weak_signal_cards` (**V11** migration — 현재 master V9 → V10 chat_sessions → V11), `alert_rules` (이미 backend spec 존재)

### Changelog

- **v1 (구, _deprecated/)** — 단순 키워드 기반 패턴 (2026-04-W2 폐기)
- **v2 (제안, P8)** — 3-phase (Pattern + Anomaly + Routing) 통합 + LLM 해석 + AlertRule 매칭
