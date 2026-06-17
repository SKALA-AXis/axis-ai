"""Frontend-ready prompt templates for StrategicInsightAgent."""

FRONTEND_READY_REPAIR_SYSTEM_PROMPT = """\
당신은 카드뉴스 화면에 직접 표시할 frontend_ready 문장만 작성하는 repair agent입니다.
전체 analysis/implication 문장을 고치지 말고 frontend_ready JSON 만 출력합니다.
새 사실을 만들지 말고 IntegratedIssue, ProfileContext,
Machine linkage hints에 있는 근거만 사용합니다.
JSON 외 텍스트를 출력하지 마세요.
"""


FRONTEND_READY_REPAIR_USER_PROMPT_TEMPLATE_LEGACY = """\
## IntegratedIssue
{integrated_issue_json}

## ProfileContext
{profile_json}

## Machine linkage hints
{profile_linkage_json}

## Response artifact guidance
{action_artifact_plan_json}

## Issue execution slots
{issue_execution_slots_json}

## frontend_ready 위반
{violations_json}

## 작성 기준
1. 출력은 frontend_ready 객체 하나만 반환합니다.
2. source와 각 block.source는 반드시 "frontend_repair_direct"입니다.
3. 먼저 문장을 쓰지 말고 insight_basis/action_basis에 화면 문장의 뼈대를 정리합니다.
   - insight_basis: event_anchor, observed_change, cause_or_background,
     comparison_context, strategic_reading
   - action_basis: skax_question, check_target, evaluation_basis,
     execution_angle, required_condition
4. 그 다음 basis를 바탕으로 sentence/evidence_sentence를 작성합니다.
   sentence는 결론만 담은 짧은 1문장입니다.
   evidence_sentence는 보통 그 결론을 뒷받침하는 구체 근거/설명 1문장입니다.
   다만 사용자별 SK AX 보강 프로필이 현재 사건과 직접 맞물려 suggested_action을
   공통 대응방향보다 구체화하는 경우에는 suggested_action.evidence_sentence를
   3~4문장으로 써도 됩니다.
   suggested_action.sentence는 action_basis 조각을 이어 붙이지 말고,
   현재 사건이 SK AX의 후속 판단에 남기는 변화를 자연스러운 문장으로 씁니다.
   입력 fact에서 직접 확인된 대상·수치·기능·적용 방식·관계 구조를 사용해
   문장의 초점을 좁힙니다.
   sentence/evidence_sentence는 사용자 화면에 그대로 노출되므로
   내부 검증 메모가 아니라 독자가 바로 읽을 수 있는 완성문으로 씁니다.
   제품명·서비스명·행사명은 현재 사건의 중심이면 그대로 사용합니다.
   근거/설명 문장은 결론을 다시 선언하는 데서 멈추지 말고,
   기사 사실이 어떤 차이, 범위, 단계, 구조를 드러내는지 직접 말합니다.
   evidence_sentence는 일반 판단 문장으로 끝내지 말고 integrated_text,
   fact_summary, consolidated_facts, fact_basis, strategic_evidence_inventory,
   cluster_fact_intelligence에 있는 제품명, 수치, 적용처, 고객/계약 관계,
   기능, 운영 범위 같은 구체 기사 표현을
   1개 이상 다시 연결합니다.
5. 현재 사건에서 가장 중요한 신호 하나를 선택하되, 통합 이슈에 나온
   고유 명사·수치·제품/서비스·고객/계약 관계·기능 설명 중 최소 두 가지를 문장에 반영합니다.
   상위 개념으로 바꿔 말하기 전에 fact_summary의 구체 표현을 먼저 유지합니다.
6. profile linkage가 충분하고 실제 문장에 프로필 근거를 쓸 때만 profile_based로 씁니다.
   그렇지 않으면 event_based 또는 generic_monitoring으로 낮춥니다.
7. key_implication은 현재 사건이 피어사 또는 해당 기업에 남기는 영향과 의미를 씁니다.
   SK AX 행동은 넣지 않습니다.
   key_implication.sentence는 현재 사실을 다시 말하거나 시장 일반론으로 넓히는 데서
   멈추지 말고, 기사 속 계약·출시·수주·도입·협력·수치 변화가 그 기업의 사업,
   고객 대응, 운영 방식, 경쟁 위치에 어떤 의미를 남기는지 해석합니다.
   여러 회사가 함께 등장하면 현재 카드의 피어사와 직접 연결된 사실만 근거로 삼고,
   다른 회사의 검증 규모·도입 서비스·계약 사실을 현재 피어사의 성과처럼 쓰지 않습니다.
   시장 변화는 피어사 의미를 설명하는 배경으로만 사용합니다.
   입력 근거로 직접 설명 가능한 축만 선택합니다.
   입력 사실이나 profile linkage에 성과/협약/출시/수치 변화의 원인 또는 배경이 있으면
   그 원인도 strategic_reading에 반영합니다. 단, 입력에 없는 AI/AX/클라우드/플랫폼
   원인을 새로 붙이지 않습니다.
   입력 근거 없이 밸류에이션, 캡티브 트랩, 공시 체계, 오케스트레이션, SaaS,
   반복 매출, 고객 락인 같은 표현을 새로 만들지 않습니다.
   단정하지 말고 근거 범위에 맞춰 "~가 부각될 수 있다", "~로 이동하는 신호다",
   "~를 암시한다" 정도의 강도로 작성합니다.
   key_implication.evidence_sentence도 대응방향 문장이 아닙니다.
   독자에게 행동을 지시하지 말고, 기사 사실이 왜 key_implication.sentence를
   뒷받침하는지만 설명합니다.
8. suggested_action은 key_implication과 구분해, 현재 사건이 SK AX의 후속 방향에 남기는
   대응 관점을 씁니다.
   SK AX의 사업, 운영 모델, 협력 방식, 고객 대응 방식 중 실제 근거와 연결되는
   한 축을 골라 구체적으로 작성합니다.
   대응방향은 지시문이나 체크리스트가 아니라 전략적 해석 문장입니다.
   ProfileContext.skax_profile.user_strategy_overlays가 있으면 사용자별 SK AX 보강 프로필입니다.
   현재 사건 fact와 직접 관련되는 의미 단위만 사용해 공통 대응방향을 더 구체화합니다.
   관련이 없으면 overlay를 사용하지 않습니다.
   관련 있더라도 overlay 문장을 그대로 반복하지 말고,
   현재 사건에서 드러난 업무·시스템·운영 구조와 연결해 SK AX식 대응 관점으로 재해석합니다.
   overlay에 있는 내부 initiative, 현재 범위, 목표 방향, 운영 조건은 현재 사건과 겹칠 때만
   suggested_action의 구체화 근거로 사용하고, key_implication에는 넣지 않습니다.
   overlay를 쓰는 경우 suggested_action.sentence는 한 줄 핵심 대응으로 유지하고,
   suggested_action.evidence_sentence는 왜 그 대응이 공통 대응보다 구체화되는지 설명합니다.
   overlay의 목표 방향이나 제품/운영 범위가 현재 사건과 직접 겹치면,
   suggested_action.sentence는 "점검/검토/필요" 수준으로 끝내지 말고
   SK AX가 기존 범위에 머물지 않고 overlay에 실제로 제시된 목표 방향으로
   어떻게 나아가야 하는지 한 문장으로 씁니다.
   핵심 대응은 준비나 구체화 작업 자체가 아니라,
   현재 범위에서 목표 방향으로 나아가야 한다는 방향 전환 문장으로 씁니다.
   명명이나 재정의 자체를 결론으로 삼지 말고,
   현재 범위에 머물지 않고 목표 방향으로 나아가는 대응을 결론으로 씁니다.
   설명에는 현재 사건이 높인 고객 기대/경쟁 기준, overlay의 현재 범위가 그대로이면 부족해지는 지점,
   overlay에 실제로 있는 목표 방향과 실행 조건, 그렇게 조정할 때의 사업적 의미를
   현재 입력 안에서 확인되는 근거만으로 연결합니다.
   근거/설명 중간 문장은 "필요가 있다" 같은 일반 표현보다
   기존 범위에 머물면 어떤 고객 기대나 실행 범위를 설명하기 어려운지,
   그래서 overlay의 어떤 목표 방향과 실행 조건으로 묶어야 하는지를 말합니다.
   마지막 설명은 단순 확인 과제로 끝내지 말고, 그렇게 접근했을 때 SK AX 제안이
   overlay에 실제로 제시된 고객 도입 기준이나 차별화 근거 중 무엇을
   확보하거나 강화할 수 있는지 입력 근거 안에서 정리합니다.
   특히 overlay에 차별화 기준이 있으면 피어 기능을 단순 추격하는 것이 아니라
   고객 도입 기준에서 어떤 차별화 근거를 만들 수 있는지까지 설명합니다.
   overlay 내용을 요약해 붙이지 말고, 현재 사건과 맞닿는 부분을 SK AX 대응 논리로 다시 구성합니다.
   특히 협력/MOU/공동개발 이슈에서는 각 참여자가 보유한 역량이 어떻게 맞물리는지
   먼저 읽습니다. 기사에 한쪽의 현장·시스템·운영 역량과 다른 쪽의 AI 모델·기술
   역량이 함께 나오면, 두 역량이 어떤 적용 단계에서 맞물리는지 해석하고,
   대응방향은 SK AX의 데이터, 검증 환경, 파트너 역량이 어떤 후속 사업 형태와
   연결될 수 있는지로 씁니다. 단, 기사에 없는 약점이나 단점은 만들지 않습니다.
   frontend_ready 생성 가능성은 fact 충분성으로 판단합니다.
   이미 IntegratedIssue/analysis_package가 만들어졌다면 앞단에서 카드 후보로 유효하다고
   판단된 상태입니다. 따라서 고객명·계약·구축 범위가 없다는 이유만으로 닫지 말고,
   integrated_text, fact_summary, consolidated_facts, fact_basis,
   strategic_evidence_inventory, cluster_fact_intelligence에서
   이 이슈가 통합된 이유가 되는 anchor를 먼저 찾습니다.
   그 anchor로 변화/차이를 설명할 수 있고 SK AX의 대응 방향을 만들 수 있으면,
   claim strength를 cautious 또는 moderate로 낮춰 frontend_ready를 작성합니다.
   정말 변화 anchor나 판단 축을 만들 수 없을 때만 needs_review/watch_only로 둡니다.
   계약·협약·수주·구축·고객 적용·PoC처럼 실행 대상과 범위가 확인되면
   strong actionable signal입니다.
   고객명·계약·구축 범위가 부족해도 매출액/매출 비중/사업 비중/성장률 같은 사업 구조 fact와
   특정 서비스·플랫폼·솔루션·시스템의 변화 또는 고도화 fact, 기능·적용 방식·운영 방식·자동화·
   데이터 활용 같은 실행 fact가 함께 있으면 moderate actionable signal입니다.
   moderate에서는 직접 구축/확보/운영을 단정하지 말고, 입력 fact에서 확인된
   대상과 변화 범위를 낮은 강도의 판단 축으로 씁니다.
   frontend_ready 위반이 있으면 현재 사건 fact에서 suggested_action을 다시 씁니다.
   프로필 연결이 약하면
   suggested_action.evidence_mode를 event_based 또는 generic_monitoring으로 낮춥니다.
   이때 skax_anchor_terms는 비워도 되지만, sentence에는 반드시 "SK AX는" 주어와
   현재 사건에서 나온 대상/시스템/서비스/수치/기능 중 하나를 포함합니다.
   skax_anchor_terms가 비어 있거나 evidence_mode가 generic_monitoring이면
   business_line_mapping 후보명, SK AX 프로필 조각, "기회" 같은 내부 라벨을
   sentence에 붙이지 않습니다. 이 경우 대응방향은 현재 사건의 산업·업무 변화가
   SK AX의 후속 움직임에 주는 의미만 씁니다.
   profile_based 과강도 위반이면 integrated_text/fact 근거에서 다시 작성합니다.
   SK AX 프로필 근거가 직접 보이지 않으면 claim_strength를 낮추고
   event_based 또는 generic_monitoring으로 전환합니다.
   먼저 실제로 기사에 나온 대상과 변화 지점을 찾고, 그 대상이 SK AX의 후속 움직임에서
   어떤 준비 형태로 이어지는지 낮은 강도로 작성합니다.
   고객명·계약·구축 범위가 부족하면 직접 수행이나 사업 확대가 아니라
   입력 fact에서 확인된 판단 축으로 낮춥니다.
   claim이 과강도라는 위반이 있으면
   매출 비중, 서비스 고도화, 탐지·보완·대응, 운영 방식처럼 관찰 가능한 사실 중심으로 낮춥니다.
   서비스 변화·수치·계약·출시·구축·적용 범위가 부족한 단순 홍보/웨비나/경진대회는
   weak signal로 보고 억지 frontend_ready를 만들지 않습니다.
   다만 전시·행사라도 참가 규모, 참여 주체, 기술 테마, 적용 분야가 함께 있으면
   단순 개최 사실로 닫지 말고 산업 채택 신호로 읽어 시사점/대응방향을 작성합니다.
   generic_monitoring이면 SK AX의 구체 사업명/역량명을 새로 붙이지 말고,
   현재 사건의 대상·고객군·도입 흐름이 SK AX의 후속 판단에 주는 관점을 씁니다.
   현재 사건 anchor와 직접 연결되지 않는 SK AX 사업명은 문장에 넣지 않습니다.
   먼저 SK AX profile linkage 강도에 따라 action mode를 고릅니다.
   - direct_business_match: SK AX 프로필에 현재 사건의 사업/역량/서비스와 직접 접점이 있을 때만
     기존 사업/역량 안에서 보완할 운영 기준이나 사업 판단 기준을 씁니다.
   - adjacent_opportunity_probe: 직접 근거는 약하지만 고객군, 자동화, 운영 시스템,
     AI/클라우드 운영 같은 인접 접점이 있을 때는 낮은 강도의 관찰 문장으로 씁니다.
   - watch_or_monitor: 연결 근거가 거의 없으면
     피어 동향/산업 신호 모니터링 또는 needs_review 수준으로 둡니다.
   key_implication의 핵심 명사 조합을 그대로 반복하지 말고,
   SK AX의 업무 범위, 데이터 조건, 파트너 조건, 검증 환경 중
   무엇이 후속 변화의 핵심인지 정합니다.
   action_basis에는 대응 대상, 선택 근거, 후속 변화 방향을 각각 명확히 적습니다.
   실행 관점은 입력 사건에 맞춰 필요한 축만 선택합니다.
   SK AX의 사업이나 운영 방식이 어떤 형태로 달라질 수 있는지
   action_basis.required_condition에 포함합니다.
   연결 근거가 강하면 더 실행적인 방향을 쓸 수 있습니다. 연결 근거가 약하면 수요 변화,
   적용 가능성, 기존 시스템 접점, 파트너십 필요성 같은 관찰 관점으로 낮춥니다.
9. sentence는 근거를 길게 붙이지 말고 핵심 결론만 씁니다.
   evidence_sentence에서 무엇이 어떤 방식으로 바뀌었는지 설명합니다.
   evidence_sentence는 기사 사실을 반복만 하지 말고 그 사실이 왜 결론을 뒷받침하는지 설명합니다.
   sentence/evidence_sentence는 현재 사건의 구체 사실을 우선합니다.
   수치·비교군, 제품/서비스명, 작동 방식, 적용 현장, 협약/거래 관계 중
   현재 사건에서 확인된 구체 anchor를 넣어 다시 씁니다.
10. claim_type/evidence_mode/anchor_terms는 문장에 실제로 쓴 표현에서 추출합니다.
    claim_type이 애매하면 event_based_signal 또는 internal_strategy_check로 낮춥니다.
11. current fact에 없는 제품명, 고객명, 기술명, 숫자, 성과를 추가하지 않습니다.
12. 기존 analysis/implication 문장은 참고하지 말고,
    IntegratedIssue와 linkage 구조만 보고 새로 작성합니다.
13. suggested_action은 SK AX의 후속 움직임을 현재 사건의 고유 신호에서 끌어옵니다.
    key_implication을 단순히 SK AX 주어로 바꿔 말하지 않습니다.
    action evidence_sentence는 key_implication evidence_sentence와 같은 사실 설명을
    반복을 줄이고, 그 사실이 SK AX의 후속 변화나 준비 관점으로 이어지는 이유를 설명합니다.
    대응방향 근거는 "피어사가 했기 때문"에서 멈추지 말고,
    현재 사건이 SK AX에게 어떤 사업 대상, 파트너 조건, 검증 환경을 생각하게 하는지 연결합니다.
    대응방향 sentence는 SK AX의 다음 움직임이 어떤 형태로 전개될 수 있는지 보이게 씁니다.
    실행 축은 입력 근거에 실제로 나온 대상·수치·기능·관계 구조에서 가져옵니다.
14. 재무·거래구조·지배구조 성격의 사건은 내부거래 비중, 매출 구성, 고객 구성,
    협약 구조처럼 기사에 나온 구조 변화 자체를 중심으로 씁니다.
    현재 사건에 없는 AI·클라우드·DX 역량으로 연결하지 않습니다.
    SPC·컨소시엄·민관 합작 인프라 사업은 단순 참여 사실로 낮추지 말고,
    지분율, 대표 선임, 주도 기업 여부, 정부·파트너 지분 구조, 구축·운영 범위,
    착공/서비스 일정, GPU·데이터센터 규모가 해당 기업의 역할과 영향력에
    어떤 의미를 주는지 먼저 읽습니다.
    기사에 재무 안정성, 사업 관리, 주주 간 이해 조율, 정부 사업 관리 같은
    운영·거버넌스 근거가 있으면 기술 역량보다 사업 실행 구조의 의미를 우선합니다.
15. 결론 문장은 효과를 단정하지 말고, 관찰된 지표·적용 방식·관계 구조가
    어떤 평가 기준을 만들었는지 씁니다.
16. 재무/거래구조 사건은 기술/사업역량으로 건너뛰지 말고, 거래 비중·매출 구성·비교군을
    기준으로 작성합니다.
    key_implication.sentence에는 수치·비율·비교군을 길게 반복하지 말고,
    그 수치가 의미하는 매출 구성, 거래 의존도, 평가 기준 변화를 씁니다.
    수치·비율·비교 대상은 evidence_sentence에 배치해 결론을 뒷받침합니다.
    IntegratedIssue.issue_frame이 peer_comparison 성격이면 comparison_facts,
    risk_facts, market_structure_facts를 ProfileContext보다 먼저 읽습니다.
    strategic_evidence_inventory가 있으면 core_facts뿐 아니라 supporting_facts,
    background_facts, cause_or_driver_facts, risk_facts,
    uncertainty_or_limitation_facts, strategic_tensions, actionable_questions까지 훑습니다.
    핵심 사건을 직접 만들지 않는 보조 사실이라도 시사점의 두께, 한계, 대응 기준을
    설명하는 데 필요하면 evidence_sentence나 suggested_action 근거에 반영합니다.
    특히 cause_or_driver_facts, uncertainty_or_limitation_facts, strategic_tensions 중
    값이 있으면 단순 비교 결과만 쓰지 말고 적어도 하나의 층위를 결론 또는 근거에 연결합니다.
    원인/드라이버는 왜 피어 간 차이가 생겼는지, 불확실성/한계는 무엇을 별도로 설명하거나
    추가 검증해야 하는지, strategic_tensions는 어떤 상충 관계가 의사결정 기준이 되는지
    보여주는 데 사용합니다.
    uncertainty_or_limitation_facts가 있으면 suggested_action은 단순 모니터링이 아니라
    어떤 사실을 별도 지표로 설명하거나 외부 검증 가능하게 만들지까지 포함합니다.
    먼저 comparison_axis가 무엇을 비교하는 축인지 입력 fact 안에서 정의하고,
    각 comparison_facts의 metric/value/evidence_text가 보여주는 차이와 방향을 읽습니다.
    그 다음 risk_facts와 market_structure_facts가 있으면 비교 차이가 왜
    평가 기준, 설명 책임, 운영 판단, 리스크 관리와 연결되는지 해석합니다.
    비교 수치·비율은 단순 순위가 아니라 각 metric이 나타내는 사업 구조, 의존도,
    고객 기반, 리스크 노출, 평가 기준을 설명하는 근거로만 씁니다.
    수치가 낮거나 높다는 이유만으로 경쟁력 우위나 성과를 단정하지 말고,
    입력에 함께 제공된 업계 기준, 규제/리스크 맥락, 시장 구조 fact가 있을 때만
    그 의미를 연결합니다. key_implication은 피어/시장 평가 기준의 변화를,
    suggested_action은 SK AX의 후속 판단에 남는 지표, 고객군, 수익성,
    지속성, 외부 검증 가능성 같은 관점을 씁니다.
    현재 사건에 없는 기술·사업 성과, 원인, 대응 전략으로 건너뛰지 않습니다.
17. 서비스/제품 출시 사건은 시장 지위보다 적용 대상, 처리 범위, 연결되는 업무/시스템을
    기준으로 작성합니다.
18. 협약/구축 사건은 혁신/경쟁력보다 적용 현장, 사용 플랫폼, 운영 방식, 역할 구조를
    기준으로 작성합니다.
    고객 규모, 거점 수, 시장 규모, 사업 규모는 적용 환경을 설명하는 보조 근거로만 쓰고,
    SK AX 대응 기준으로 바로 연결하지 않습니다. 대응 기준은 현재 사건의 실행 구조에서 가져옵니다.
    피어사 고유 제품명은 key_implication.evidence_sentence에서 사실 근거로 쓸 수 있지만,
    SK AX 프로필에 같은 제품/역량 근거가 없으면 suggested_action의 직접 기준으로 쓰지 않습니다.
   피어사 제품명을 SK AX의 직접 기준으로 옮기기보다, 해당 제품이 맡는 기능과
   적용 업무를 현재 사건의 표현으로 풀어 씁니다.
   예시 축은 닫힌 목록이 아니며, 입력 사건이 다른 구조라면 IntegratedIssue의
   실제 표현에서 판단 축을 다시 도출합니다.
    고객 규모·거점 수·시장 규모는 suggested_action.sentence에는 넣지 말고,
    꼭 필요할 때만 evidence_sentence에서 적용 환경의 무게를 설명하는 보조 근거로 제한합니다.
19. 성과·지위 변화에 관한 표현은 현재 사건 fact 또는 profile linkage가
    그 효과를 직접 뒷받침할 때만 씁니다.
    근거가 관계 체결·선정·도입 흐름 수준이면 관계 구조, 적용 대상, 실행 범위,
    후속 확인 기준으로 낮춰 작성합니다.
20. frontend_ready 위반에 효과성 표현/강한 주장/실행 조건 부족이 있으면
    Issue execution slots의 counterparty, target_system, product_or_service,
    execution_scope 중 실제 값이 있는 슬롯을 우선 사용합니다.
    슬롯 값으로 설명할 수 없는 효과성 문장은 버립니다.

## anchor 선택 기준
고정 문장형을 만들지 말고 현재 입력에서 확인된 anchor 조합으로 씁니다.
- 재무·거래구조 성격이면 수치/비율, 비교 대상, 매출·거래 구성, 고객군 중 2개 이상을 사용합니다.
  단, 핵심 결론 문장은 숫자를 반복하는 문장이 아니라 그 숫자가 만드는 평가 기준을 말합니다.
  수치와 비교 대상은 근거/설명 문장에 두고 결론을 뒷받침하게 합니다.
- 제품·서비스 출시 성격이면 제품명, 기능, 적용 대상, 사용 방식 중 2개 이상을 사용합니다.
- 협약·계약·구축 성격이면 협력 대상, 적용처, 제품·기술명, 실행 범위 중 2개 이상을 사용합니다.
- 대응방향은 현재 사건 anchor와 SK AX의 후속 변화, 준비 관점, 운영 전제를 함께 씁니다.

## claim_type 후보
- event_based_signal
- profile_based_signal
- financial_structure_signal
- governance_exposure_signal
- self_or_market_signal
- market_adoption_signal
- operational_shift
- workflow_execution_signal
- internal_strategy_check

## evidence_mode 후보
- profile_based
- event_based
- generic_monitoring

## 출력
{{
  "frontend_ready": {{
    "source": "frontend_repair_direct",
    "insight_basis": {{
      "event_anchor": ["현재 사건의 핵심 anchor"],
      "observed_change": "현재 사건에서 관찰된 변화",
      "comparison_context": "비교할 산업/피어/고객군 맥락",
      "strategic_reading": "시사점 결론으로 이어지는 해석",
      "confidence": "strong|moderate|cautious"
    }},
    "action_basis": {{
      "skax_question": "SK AX가 이번 사건에서 도출할 대응 질문",
      "check_target": ["대응 대상 또는 제안 구성 요소"],
      "response_angle": "실행 관점",
      "required_condition": "실행 전 확보하거나 설명해야 할 조건",
      "confidence": "moderate|cautious"
    }},
    "key_implication": {{
      "source": "frontend_repair_direct",
      "frame": "분석 관점",
      "claim_type": "event_based_signal",
      "claim_strength": "cautious",
      "evidence_mode": "event_based",
      "event_anchor_terms": ["현재 사건에서 실제로 쓴 표현"],
      "profile_anchor_terms": [],
      "unsupported_claims_removed": [],
      "sentence": "피어사/시장 관점 결론 1문장",
      "evidence_sentence": "현재 사건 사실과 결론의 연결 설명 1문장"
    }},
    "suggested_action": {{
      "source": "frontend_repair_direct",
      "frame": "SK AX 대응 관점",
      "claim_type": "internal_strategy_check",
      "claim_strength": "cautious",
      "evidence_mode": "generic_monitoring",
      "event_anchor_terms": ["현재 사건에서 실제로 쓴 표현"],
      "skax_anchor_terms": [],
      "unsupported_claims_removed": [],
      "sentence": "SK AX 대응방향 1문장",
      "evidence_sentence": "현재 사건 신호와 대응방향의 연결 설명. overlay 구체화 시 3~4문장 가능"
    }}
  }}
}}
"""


FRONTEND_READY_REPAIR_USER_PROMPT_TEMPLATE = """\
## IntegratedIssue
{integrated_issue_json}

## ProfileContext
{profile_json}

## Machine linkage hints
{profile_linkage_json}

## Response artifact guidance
{action_artifact_plan_json}

## Issue execution slots
{issue_execution_slots_json}

## frontend_ready 위반
{violations_json}

## 역할
frontend_ready 객체 하나만 작성합니다.

## 작성 원칙
- IntegratedIssue에서 확인된 사실을 근거로 씁니다.
- key_implication은 피어사/시장 관점, suggested_action은 SK AX 관점으로 분리합니다.
- 각 block은 결론 sentence 1문장과 근거 evidence_sentence를 씁니다.
  evidence_sentence는 보통 1문장이지만, user_strategy_overlays가 현재 사건과 직접 맞물려
  suggested_action을 구체화하는 경우 suggested_action.evidence_sentence는 3~4문장까지 허용합니다.
- ProfileContext.skax_profile.user_strategy_overlays는 사용자별 SK AX 보강 프로필입니다.
  현재 사건 fact와 직접 관련되는 의미 단위만 suggested_action에 사용하고,
  관련이 없으면 overlay를 사용하지 않습니다.
  관련 있더라도 overlay 문장을 그대로 반복하지 말고,
  현재 사건에서 드러난 업무·시스템·운영 구조와 연결해 SK AX식 대응 관점으로 재해석합니다.
  overlay에 있는 내부 initiative, 현재 범위, 목표 방향, 운영 조건은 현재 사건과 겹칠 때만
  suggested_action의 구체화 근거로 사용하고, key_implication에는 넣지 않습니다.
  overlay를 쓰는 경우 suggested_action.sentence는 한 줄 핵심 대응으로 유지하고,
  suggested_action.evidence_sentence는 왜 그 대응이 공통 대응보다 구체화되는지 설명합니다.
  overlay의 목표 방향이나 제품/운영 범위가 현재 사건과 직접 겹치면,
  suggested_action.sentence는 "점검/검토/필요" 수준으로 끝내지 말고
  SK AX가 기존 범위에 머물지 않고 overlay에 실제로 제시된 목표 방향으로
  어떻게 나아가야 하는지 한 문장으로 씁니다.
  핵심 대응은 준비나 구체화 작업 자체가 아니라,
  현재 범위에서 목표 방향으로 나아가야 한다는 방향 전환 문장으로 씁니다.
  명명이나 재정의 자체를 결론으로 삼지 말고,
  현재 범위에 머물지 않고 목표 방향으로 나아가는 대응을 결론으로 씁니다.
  설명에는 현재 사건이 높인 고객 기대/경쟁 기준, overlay의 현재 범위가 그대로이면 부족해지는 지점,
  overlay에 실제로 있는 목표 방향과 실행 조건, 그렇게 조정할 때의 사업적 의미를
  현재 입력 안에서 확인되는 근거만으로 연결합니다.
  근거/설명 중간 문장은 "필요가 있다" 같은 일반 표현보다
  기존 범위에 머물면 어떤 고객 기대나 실행 범위를 설명하기 어려운지,
  그래서 overlay의 어떤 목표 방향과 실행 조건으로 묶어야 하는지를 말합니다.
  마지막 설명은 단순 확인 과제로 끝내지 말고, 그렇게 접근했을 때 SK AX 제안이
  overlay에 실제로 제시된 고객 도입 기준이나 차별화 근거 중 무엇을
  확보하거나 강화할 수 있는지 입력 근거 안에서 정리합니다.
  특히 overlay에 차별화 기준이 있으면 피어 기능을 단순 추격하는 것이 아니라
  고객 도입 기준에서 어떤 차별화 근거를 만들 수 있는지까지 설명합니다.
  overlay 내용을 요약해 붙이지 말고, 현재 사건과 맞닿는 부분을 SK AX 대응 논리로 다시 구성합니다.
- ProfileContext 접점이 약하면 evidence_mode를 event_based 또는 generic_monitoring으로 둡니다.
- source와 각 block.source는 "frontend_repair_direct"입니다.

## 출력
{{
  "frontend_ready": {{
    "source": "frontend_repair_direct",
    "insight_basis": {{
      "event_anchor": ["현재 사건에서 실제 확인된 anchor"],
      "observed_change": "현재 사건에서 관찰된 변화",
      "comparison_context": "필요할 때만 쓰는 비교/시장 맥락",
      "strategic_reading": "시사점 문장으로 이어지는 해석",
      "confidence": "strong|moderate|cautious"
    }},
    "action_basis": {{
      "skax_question": "SK AX가 이번 사건에서 확인할 질문",
      "check_target": ["대응 대상 또는 판단 요소"],
      "response_angle": "대응 관점",
      "required_condition": "실행 전 확인해야 할 조건",
      "confidence": "moderate|cautious"
    }},
    "key_implication": {{
      "source": "frontend_repair_direct",
      "frame": "시사점 관점",
      "claim_type": "event_based_signal|profile_based_signal|financial_structure_signal|...",
      "claim_strength": "strong|moderate|cautious",
      "evidence_mode": "profile_based|event_based|generic_monitoring",
      "event_anchor_terms": ["문장에 실제로 쓴 현재 사건 표현"],
      "profile_anchor_terms": ["문장에 실제로 쓴 ProfileContext 표현"],
      "unsupported_claims_removed": [],
      "sentence": "피어사/시장 관점 결론 1문장",
      "evidence_sentence": "현재 사건 사실과 결론의 연결 설명 1문장"
    }},
    "suggested_action": {{
      "source": "frontend_repair_direct",
      "frame": "SK AX 대응 관점",
      "claim_type": "internal_strategy_check|event_based_signal|self_or_market_signal|...",
      "claim_strength": "moderate|cautious",
      "evidence_mode": "profile_based|event_based|generic_monitoring",
      "event_anchor_terms": ["문장에 실제로 쓴 현재 사건 표현"],
      "skax_anchor_terms": ["문장에 실제로 쓴 SK AX/ProfileContext 표현"],
      "unsupported_claims_removed": [],
      "sentence": "SK AX 대응방향 1문장",
      "evidence_sentence": "현재 사건 신호와 대응방향의 연결 설명. overlay 구체화 시 3~4문장 가능"
    }},
    "needs_review": false,
    "reason": ""
  }}
}}
"""
