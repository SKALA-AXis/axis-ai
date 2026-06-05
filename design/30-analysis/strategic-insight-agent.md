# StrategicInsightAgent

`StrategicInsightAgent`는 기존 `StrategicAnalyzer`와 `ImplicationAgent`의 기본 실행 경로를
하나의 LLM agent로 통합한다. 단, 후속 저장과 검증 호환성을 위해 출력은 반드시
`analysis`와 `implication` 두 블록으로 분리한다.

## 목적

- `IntegrationAgent`가 만든 `integrated_issue`를 기반으로 피어사/산업 관점의 전략적
  의미를 분석한다.
- `ProfileContext`와 `AnalysisContext`를 참고해 SK AX 관점의 시사점과 대응 방향을 만든다.
- 피어 관점 분석과 SK AX 대응을 섞지 않고, 같은 근거에서 나온 두 결과를 분리 저장한다.

## Input

```text
integrated_issue
classification
input_bundle.metadata
profile_context
analysis_context
```

| 입력 | 의미 |
|---|---|
| `integrated_issue` | IntegrationAgent가 원문/클러스터를 통합한 분석용 이슈. 핵심 사실, 수치, 사업 신호, 근거 출처를 포함한다. |
| `classification` | 섹터, 이벤트 타입, 중요도 등 분류 결과. 문장에 그대로 노출하기보다 분석 참고값으로 사용한다. |
| `input_bundle.metadata` | 대표 기사, 클러스터 기사, 생성 시각 등 실행 메타데이터. |
| `profile_context` | 피어사 프로필과 SK AX 프로필. 분석/시사점의 해석 보조 맥락이다. |
| `analysis_context` | 최근 이벤트, 섹터 흐름, 재무 추세, 유사 카드 등 추가 맥락. 새 사실 생성 근거가 아니다. |

## Output

```json
{
  "is_valid_strategic_insight": true,
  "analysis": {
    "is_valid_analysis": true,
    "analysis_scope": "peer_and_industry",
    "analysis_summary": "피어사의 전략적 의미 1문장",
    "strategic_meaning": ["의미 1", "의미 2"],
    "market_signal": "시장/산업 흐름 1문장",
    "impact_level": "high",
    "impact_reason": "영향도 판단 근거",
    "risk_or_opportunity": "opportunity",
    "confidence": 0.8,
    "reason": "분석 근거"
  },
  "implication": {
    "is_valid_implication": true,
    "implication_scope": "peer_and_skax",
    "peer_implication": {
      "company_id": "integrated_issue.main_company",
      "company_name_ko": "피어사명",
      "peer_meaning": "피어사 관점 의미",
      "capability_change": "역량 변화",
      "sourced_evidence_ids": ["입력에 존재하는 fact_id"]
    },
    "skax_implication": {
      "why_important": "SK AX에 중요한 이유",
      "potential_impact": "예상 영향",
      "opportunities": ["기회 1"],
      "threats": ["위협 1"],
      "recommended_actions": ["실행 권고 1"],
      "business_line_mapping": ["SK AX 프로필의 사업라인 후보 중 실제 관련 있는 값"]
    },
    "follow_up_questions": ["추가 확인 질문"],
    "watch_points": ["관찰 포인트"],
    "confidence": 0.75,
    "evidence_label": "moderate",
    "provenance": {
      "generator": "StrategicInsightAgent",
      "prompt_version": "strategic-insight-v1.4-specific-actions",
      "model": "gpt-4o",
      "used_fact_ids": ["입력에 존재하는 fact_id"],
      "used_context_layers": ["실제로 사용한 context layer명"],
      "run_at": "ISO-8601 timestamp"
    }
  }
}
```

## Evidence Rules

- `integrated_issue`에 없는 사실, 수치, 회사명, 제품명, 고객명은 생성하지 않는다.
- 수치/날짜/정량 표현은 `fact_basis`, `key_numbers`, `representative_sources` 중 하나에
  근거가 있어야 한다.
- `profile_context`와 `analysis_context`는 해석 보조 맥락이다. 여기서 본 내용을 새 사건처럼
  쓰지 않는다.
- `sourced_evidence_ids`와 `used_fact_ids`는 입력에 존재하는 `fact_id`만 사용한다.
- 시장점유율, 매출 영향, 고객 확보, 수주 가능성, 가격 경쟁 같은 결과 표현은
  `integrated_issue`에 직접 근거가 있을 때만 쓴다.
- `business_line_mapping`은 `profile_context.skax_profile.business_lines` 또는 SK AX 프로필의
  사업영역 후보 중 실제 관련 있는 값만 선택한다. 후보의 이름뿐 아니라 사업영역 설명,
  핵심 역량, 최근 방향성이 `integrated_issue.business_signals`/`fact_summary`와 연결될 때만
  선택한다. 출력 schema의 예시값을 기본값처럼 복사하지 않는다.
- SK AX 프로필에 사업라인/사업영역 후보가 없으면 `business_line_mapping`은 `[]`로 둔다.
- `profile_context`는 배경 맥락이다. 현재 사건의 회사명, 제품명, 고객명, 수치, 날짜는
  `integrated_issue` 근거에서만 가져온다.
- 채택/선정/PoC/판매 권한 확보 수준의 근거를 시장 선점, 점유율 확대, 매출 기여처럼
  과대해석하지 않는다.
- `analysis.reason`과 `skax_implication.potential_impact`는 성과 예측보다 평가 기준,
  제안 방식, 레퍼런스 구성, 운영 책임 설명 방식의 변화로 작성한다. 직접 근거 없이
  시장점유율, 매출, 수주, 경쟁력 강화 같은 결과를 예측하지 않는다.
- 프롬프트는 중복 규칙을 줄이고, 통합 이슈 근거성·프로필 직접 관련성·구체적 실행 권고만
  강조하는 `strategic-insight-v1.4-specific-actions` 기준을 사용한다.
- "강화", "입지", "경쟁력" 같은 넓은 표현은 단독으로 쓰지 않고, 판매 접점,
  적용 레퍼런스, 운영 지원 범위, 검증 기준, 제안 메시지 중 무엇이 어떻게 바뀌는지
  함께 설명한다.
- LLM 출력의 `recommended_actions`가 "강조/강화/확대/개발/구축/수립" 같은 일반론으로
  끝나면 후처리에서 제안서, PoC, 레퍼런스 등 산출물 단위의 실행 문장으로 보정한다.

## Pipeline Position

```text
IntegrationAgent
→ ProfileContextLoader
→ AnalysisContextBuilder
→ StrategicInsightAgent
→ validate
→ AnalysisPackage
→ CardNewsComposer
```

기존 `StrategicAnalyzer`와 `ImplicationAgent`는 LLM 실패 또는 legacy 테스트 주입 시의
fallback/compat 경로로 유지한다.
