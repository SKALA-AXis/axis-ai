"""peer_swot prompts — extracted from facade (move-only)."""

# ruff: noqa: E501  — long prompt-string lines (exempt in original facade)

from __future__ import annotations

import os
import re

PROMPT_VERSION = "peer_swot_clean_v2"


DEFAULT_MODEL = os.getenv("PEER_SWOT_LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"


SK_AX_ID = "sk_ax"


TARGET_PEER_IDS = (
    SK_AX_ID,
    "samsung_sds",
    "lg_cns",
    "hyundai_autoever",
    "posco_dx",
)


COMPETITOR_PEER_IDS = tuple(peer_id for peer_id in TARGET_PEER_IDS if peer_id != SK_AX_ID)


COMPETITOR_PEER_NAMES = (
    "삼성SDS",
    "삼성 SDS",
    "LG CNS",
    "엘지 CNS",
    "현대오토에버",
    "현대 오토에버",
    "포스코DX",
    "포스코 DX",
)


COMPARISON_LABELS = ("사업 신호", "기술 신호", "리스크")


SWOT_LABELS = ("Strength", "Weakness", "Opportunity", "Threat")


GENERIC_CHANGE_OBJECT_BY_LABEL = {
    "사업 신호": "구체 사업명 확인 제한",
    "기술 신호": "구체 기술명 확인 제한",
    "리스크": "구체 리스크 원인 확인 제한",
}


GENERIC_SWOT_TITLE_BY_LABEL = {
    "Strength": "내부 역량 재사용성",
    "Weakness": "내부 실행 제약",
    "Opportunity": "외부 수요 활용 가능성",
    "Threat": "외부 환경 압박",
}


SWOT_FACTOR_TYPE_BY_LABEL = {
    "Strength": "internal_controllable",
    "Weakness": "internal_controllable",
    "Opportunity": "external_uncontrollable",
    "Threat": "external_uncontrollable",
}


COMPARISON_SIGNAL_TYPES = {
    "business_overview",
    "business_update",
    "product_service",
    "service_launch",
    "strategy",
    "rd",
    "risk",
}


COMPARISON_SOURCE_TYPES = {"dart", "ir", "securities_report"}


SWOT_SIGNAL_TYPES = {
    "business_overview",
    "business_update",
    "product_service",
    "service_launch",
    "strategy",
    "rd",
    "risk",
    "forecast",
    "investment",
    "valuation",
}


SWOT_METRIC_NAMES = {
    "revenue_yoy",
    "revenue_qoq",
    "operating_profit_yoy",
    "operating_profit_qoq",
    "operating_margin",
    "operating_margin_yoy",
    "operating_margin_qoq",
    "net_income",
    "net_profit",
}


FINANCIAL_NUMBER_PATTERN = re.compile(
    r"(?i)("
    r"(?:매출|영업이익|순이익|당기순이익|수주|잔고|revenue|profit|yoy|qoq).{0,24}"
    r"[+-]?\d[\d,]*(?:\.\d+)?\s*(?:%|억|억원|조|조원)?"
    r"|"
    r"[+-]?\d[\d,]*(?:\.\d+)?\s*(?:%|억|억원|조|조원).{0,24}"
    r"(?:매출|영업이익|순이익|당기순이익|수주|잔고|revenue|profit|yoy|qoq)"
    r")"
)


NOISE_PHRASES = (
    "최근 공개 원문에서는",
    "최근 공개 원문에서",
    "이 peer사는",
    "이 Peer사는",
    "해당 peer사는",
    "해당 Peer사는",
    "현재 입력 근거만으로 해당 축을 정의하기 어렵다",
)


COMPARISON_PROMPT = """You are a Korean IT service industry analyst.

Write exactly three Peer+ core comparison points from the evidence pack.

Purpose:
- These are recent observed movements, not SWOT judgments.
- Write in the form "what changed / what is being attempted / what risk appeared recently".
- Do not diagnose whether the company is strong, weak, advantaged, or threatened structurally.
- If peer.id is "all", write industry-level IT service trends across peer companies, not a single company's story.
- If peer.id is "all", prefer themes appearing in at least two peer companies; if only one peer supports a theme, say evidence is limited.
- Prefer a named business, service, platform, customer domain, or technology when it is visible in evidence.
- If a concrete name is not visible, do not invent one; use a cautious category-level object and set insufficient_evidence=true.
- Do not use financial numbers, revenue/profit/order figures, YoY/QoQ, %, 억원, 조원.
- Do not write Strength/Weakness/Opportunity/Threat style conclusions.

Required labels:
1. 사업 신호: what business/customer/domain movement is visible.
2. 기술 신호: what platform/technology/product implementation is visible.
3. 리스크: what execution, market, customer, regulation, schedule, competition, or dependency risk is visible.

Evidence rules:
- Use only evidence_ref values from allowed_comparison_evidence_refs.
- If evidence is thin, still write the best cautious observation and set insufficient_evidence=true.

Return strict JSON only:
{{
  "comparison_points": [
    {{
      "label": "사업 신호",
      "change_object": "specific name if visible, otherwise cautious category",
      "body": "1-2 Korean sentences",
      "evidence_refs": ["signal:123"],
      "source_urls": [],
      "confidence": 0.0,
      "evidence_summary": "short Korean evidence summary",
      "reasoning_summary": "short Korean reasoning",
      "insufficient_evidence": false
    }},
    {{
      "label": "기술 신호",
      "change_object": "specific name if visible, otherwise cautious category",
      "body": "1-2 Korean sentences",
      "evidence_refs": ["signal:123"],
      "source_urls": [],
      "confidence": 0.0,
      "evidence_summary": "short Korean evidence summary",
      "reasoning_summary": "short Korean reasoning",
      "insufficient_evidence": false
    }},
    {{
      "label": "리스크",
      "change_object": "specific risk object if visible, otherwise cautious category",
      "body": "1-2 Korean sentences",
      "evidence_refs": ["signal:123"],
      "source_urls": [],
      "confidence": 0.0,
      "evidence_summary": "short Korean evidence summary",
      "reasoning_summary": "short Korean reasoning",
      "insufficient_evidence": false
    }}
  ]
}}

Evidence Pack:
{pack_json}
"""


SWOT_PROMPT = """You are a Korean strategy analyst.

Write an actual SWOT diagnosis for the target company or competitor group.

Important distinction:
- Core comparison points are recent observed movements.
- SWOT is a diagnosis of internal capability/constraint and external opportunity/threat.
- If peer.id is "all", diagnose the peer group or IT service industry trend, not SamsungSDS or any single company.
- Do not restate the comparison points with different labels.
- Do not reuse the same title, change object, or event sentence from observed_comparison_points_for_non_repetition.
- If the same evidence is relevant, reinterpret it through a different SWOT lens:
  Strength = reusable capability, Weakness = internal constraint, Opportunity = external opening, Threat = external pressure.
- Do not explain what SWOT is. Do not write about "axis", "criteria", or "monitoring".
- Write what this company is strong/weak/exposed/positioned for.

SWOT definitions:
- Strength: internal capability or differentiator that the company can control and use competitively.
- Weakness: internal limitation, execution burden, cost/profitability issue, concentration, or capability gap that the company can improve.
- Opportunity: external favorable market/customer/policy/technology condition the company can exploit.
- Threat: external unfavorable competition/regulation/market/customer/technology condition that can hurt the company.

Evidence rules:
- Use only evidence_ref values from allowed_swot_evidence_refs.
- Prefer diagnostic evidence matching the SWOT label.
- Use diagnostic_question, judgment_basis, and basis_points as the main guide.
- If evidence is thin, write a cautious diagnosis without inventing names and set insufficient_evidence=true.

Return strict JSON only:
{{
  "swot": [
    {{
      "label": "Strength",
      "title": "specific title if visible, otherwise cautious category title",
      "body": "1-2 Korean sentences; use concrete names only when visible in evidence",
      "factor_type": "internal_controllable",
      "check_point": "what to check next",
      "evidence_refs": ["diagnostic:peer:strength:abc"],
      "source_urls": [],
      "confidence": 0.0,
      "evidence_summary": "short Korean evidence summary",
      "reasoning_summary": "short Korean reasoning",
      "insufficient_evidence": false
    }},
    {{
      "label": "Weakness",
      "title": "specific title if visible, otherwise cautious category title",
      "body": "1-2 Korean sentences; use concrete names only when visible in evidence",
      "factor_type": "internal_controllable",
      "check_point": "what to check next",
      "evidence_refs": ["diagnostic:peer:weakness:abc"],
      "source_urls": [],
      "confidence": 0.0,
      "evidence_summary": "short Korean evidence summary",
      "reasoning_summary": "short Korean reasoning",
      "insufficient_evidence": false
    }},
    {{
      "label": "Opportunity",
      "title": "specific title if visible, otherwise cautious category title",
      "body": "1-2 Korean sentences; use concrete names only when visible in evidence",
      "factor_type": "external_uncontrollable",
      "check_point": "what to check next",
      "evidence_refs": ["diagnostic:peer:opportunity:abc"],
      "source_urls": [],
      "confidence": 0.0,
      "evidence_summary": "short Korean evidence summary",
      "reasoning_summary": "short Korean reasoning",
      "insufficient_evidence": false
    }},
    {{
      "label": "Threat",
      "title": "specific title if visible, otherwise cautious category title",
      "body": "1-2 Korean sentences; use concrete names only when visible in evidence",
      "factor_type": "external_uncontrollable",
      "check_point": "what to check next",
      "evidence_refs": ["diagnostic:peer:threat:abc"],
      "source_urls": [],
      "confidence": 0.0,
      "evidence_summary": "short Korean evidence summary",
      "reasoning_summary": "short Korean reasoning",
      "insufficient_evidence": false
    }}
  ]
}}

Evidence Pack:
{pack_json}
"""
