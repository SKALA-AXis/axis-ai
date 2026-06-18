"""Fast action-only regeneration for user strategy context apply."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any

from src.llm import LLMSpec, build_chat_llm
from src.services.profile_context_loader import ProfileContextLoader

_DEFAULT_MODEL = "gpt-5.5"
_MODEL = os.getenv(
    "FRONTEND_READY_REPAIR_MODEL",
    os.getenv("FRONTEND_READY_MODEL", os.getenv("STRATEGIC_INSIGHT_MODEL", _DEFAULT_MODEL)),
)
_TIMEOUT_SECONDS = float(os.getenv("STRATEGY_CONTEXT_ACTION_TIMEOUT_SECONDS", "120"))
_MAX_OUTPUT_TOKENS = int(os.getenv("STRATEGY_CONTEXT_ACTION_MAX_TOKENS", "3000"))


def regenerate_strategy_context_action(
    analysis_package: dict[str, Any],
    *,
    card_news_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Regenerate only action fields for personalized strategy context.

    This path intentionally does not call StrategicInsightAgent. The stored
    card already has common summary/insight; personalized apply only replaces
    the SK AX action copy.
    """
    package = _json_dict(analysis_package)
    integrated_issue = _json_dict(package.get("integrated_issue") or package.get("summary"))
    classification = _json_dict(package.get("classification"))
    existing_analysis = _json_dict(package.get("analysis"))
    existing_implication = _json_dict(package.get("implication"))
    if not integrated_issue:
        raise ValueError("integrated_issue is required for strategy context action regeneration")
    if not existing_implication:
        raise ValueError(
            "existing implication is required for strategy context action regeneration"
        )

    profile_context = (
        ProfileContextLoader()
        .load(
            companies=_companies_from_issue(integrated_issue),
            sectors=_sectors_from_issue(integrated_issue, classification),
            event_type=str(
                classification.get("event_type")
                or integrated_issue.get("cluster_event_type")
                or integrated_issue.get("event_type")
                or ""
            )
            or None,
            user_id=user_id,
            issue_scope=_issue_scope(
                package=package,
                integrated_issue=integrated_issue,
                classification=classification,
                card_news_id=card_news_id,
            ),
            strict=False,
            require_skax_profile=False,
        )
        .to_dict()
    )
    profile_for_prompt = _profile_for_prompt(profile_context)
    prompt = _build_prompt(
        integrated_issue=integrated_issue,
        classification=classification,
        existing_implication=existing_implication,
        profile_context=profile_for_prompt,
    )
    content = _invoke_json_llm(prompt)
    action = _extract_action(content)
    if not action.get("sentence") or not action.get("evidence_sentence"):
        raise ValueError("strategy context action response is missing sentence/evidence_sentence")

    next_implication = _merge_action(existing_implication, action)
    return {
        "analysis": existing_analysis,
        "implication": next_implication,
        "sentence_grounding": package.get("sentence_grounding") or {},
        "diagnostics": {
            "strategy_context_action_only": True,
            "strategy_context_action_model": _MODEL,
        },
    }


def _invoke_json_llm(prompt: str) -> str:
    llm = build_chat_llm(
        LLMSpec(
            model=_MODEL,
            temperature=0.2,
            max_tokens=_MAX_OUTPUT_TOKENS,
            max_tokens_reasoning=_MAX_OUTPUT_TOKENS,
            json_object=True,
            reasoning_effort="low",
            timeout=_TIMEOUT_SECONDS,
            max_retries=0,
        )
    )
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You write Korean executive card-news action copy. "
                    "Return valid JSON only. Do not invent facts."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    return response.content if isinstance(response.content, str) else str(response.content)


def _build_prompt(
    *,
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    existing_implication: dict[str, Any],
    profile_context: dict[str, Any],
) -> str:
    fact_lines = _fact_lines(integrated_issue)[:12]
    existing_frontend = _json_dict(existing_implication.get("frontend"))
    existing_ready = _json_dict(existing_implication.get("frontend_ready"))
    return "\n".join(
        [
            "## 목적",
            (
                "기존 카드뉴스의 요약과 시사점은 유지하고, 맞춤 전략자료가 "
                "관련될 때 대응방안만 고도화합니다."
            ),
            "관련 없는 맞춤 전략자료는 사용하지 않습니다.",
            "",
            "## 현재 카드 fact",
            _json_dumps(
                {
                    "headline": integrated_issue.get("headline")
                    or integrated_issue.get("title")
                    or "",
                    "main_company": integrated_issue.get("main_company"),
                    "event_type": classification.get("event_type")
                    or integrated_issue.get("cluster_event_type")
                    or integrated_issue.get("event_type"),
                    "fact_lines": fact_lines,
                }
            ),
            "",
            "## 기존 카드 대응방안",
            _json_dumps(
                {
                    "suggested_actions": _string_list(
                        existing_frontend.get("suggested_actions"), max_items=3
                    ),
                    "frontend_ready_suggested_action": _json_dict(
                        existing_ready.get("suggested_action")
                    ),
                }
            ),
            "",
            "## SK AX profile and user strategy overlays",
            _json_dumps(profile_context),
            "",
            "## 작성 규칙",
            "- suggested_action만 작성합니다. key_implication은 작성하지 않습니다.",
            "- 모든 문장은 카드뉴스 문체인 '~다' 체로 씁니다.",
            "- '하세요', '하십시오', '합니다', '보여줍니다' 같은 대화형 존댓말을 쓰지 않습니다.",
            "- sentence는 'SK AX는'으로 시작해 어떤 방향으로 나아가야 하는지 한 문장으로 씁니다.",
            (
                "- 맞춤 전략자료에 현재 사건과 직접 관련된 내부 제품명이나 initiative명이 "
                "명시되어 있으면 sentence에 한 번 반영합니다."
            ),
            "- evidence_sentence는 3~4문장으로 씁니다.",
            "- 설명은 현재 사건이 보여준 변화, 기존 대응 범위가 부족해질 수 있는 이유,",
            (
                "  맞춤 전략자료와 연결되는 제품·운영·통제·연계 방향, "
                "그렇게 접근했을 때의 사업적 의미를 연결합니다."
            ),
            (
                "- 맞춤 전략자료 문장을 그대로 복사하지 말고 현재 사건에 비춰 "
                "SK AX 대응 논리로 재구성합니다."
            ),
            "- 입력에 없는 고객명, 계약, 수치, 레퍼런스, 성과, 시장 지위는 만들지 않습니다.",
            "- 피어 제품명은 SK AX가 따라야 할 기준처럼 쓰지 않고 사건 근거로만 사용합니다.",
            "- 일본어 한자나 깨진 문자를 쓰지 않습니다.",
            "- JSON 외 텍스트는 출력하지 않습니다.",
            "",
            "## 반환 JSON",
            _json_dumps(
                {
                    "suggested_action": {
                        "source": "frontend_repair_direct",
                        "frame": "string",
                        "claim_type": "internal_strategy_check",
                        "claim_strength": "cautious",
                        "evidence_mode": "profile_based",
                        "event_anchor_terms": ["현재 카드 fact에서 실제 사용한 표현"],
                        "skax_anchor_terms": ["SK AX profile/overlay에서 실제 사용한 표현"],
                        "unsupported_claims_removed": [],
                        "sentence": "핵심 대응 1문장",
                        "evidence_sentence": "근거/설명 3~4문장",
                    }
                }
            ),
        ]
    )


def _extract_action(content: str) -> dict[str, Any]:
    data = _parse_json(content)
    action = _json_dict(data.get("suggested_action"))
    if not action:
        action = _json_dict(_json_dict(data.get("frontend_ready")).get("suggested_action"))
    if not action:
        return {}
    return {
        "source": "frontend_repair_direct",
        "frame": str(action.get("frame") or "string").strip(),
        "claim_type": str(action.get("claim_type") or "internal_strategy_check").strip(),
        "claim_strength": str(action.get("claim_strength") or "cautious").strip(),
        "evidence_mode": str(action.get("evidence_mode") or "profile_based").strip(),
        "event_anchor_terms": _string_list(action.get("event_anchor_terms"), max_items=8),
        "skax_anchor_terms": _string_list(action.get("skax_anchor_terms"), max_items=8),
        "unsupported_claims_removed": _string_list(
            action.get("unsupported_claims_removed"), max_items=8
        ),
        "sentence": _clean_sentence(action.get("sentence")),
        "evidence_sentence": _clean_sentence(action.get("evidence_sentence")),
    }


def _merge_action(implication: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(implication)
    frontend_ready = deepcopy(_json_dict(out.get("frontend_ready")))
    frontend_ready.setdefault("source", "llm_direct")
    frontend_ready["suggested_action"] = action
    out["frontend_ready"] = frontend_ready

    sentence = str(action.get("sentence") or "").strip()
    evidence = str(action.get("evidence_sentence") or "").strip()
    item = {"main": sentence, "detail": evidence}
    frontend = deepcopy(_json_dict(out.get("frontend")))
    frontend["suggested_actions"] = [sentence]
    frontend["response_direction_blocks"] = [item]
    frontend["suggested_action_items"] = [item]
    frontend["skax_checkpoint_blocks"] = [item]
    out["frontend"] = frontend
    out["suggested_actions"] = [sentence]
    out["response_direction_blocks"] = [item]
    out["suggested_action_items"] = [item]
    out["skax_checkpoint_blocks"] = [item]

    diagnostics = _json_dict(out.get("frontend_ready_diagnostics"))
    diagnostics["strategy_context_action_only"] = True
    diagnostics["strategy_context_action_model"] = _MODEL
    out["frontend_ready_diagnostics"] = diagnostics
    return out


def _profile_for_prompt(profile_context: dict[str, Any]) -> dict[str, Any]:
    skax = _json_dict(profile_context.get("skax_profile"))
    profile = {
        "one_liner": skax.get("one_liner"),
        "company_summary": skax.get("company_summary"),
        "strategic_focus": skax.get("strategic_focus"),
        "core_capabilities": skax.get("core_capabilities"),
        "key_products_services": skax.get("key_products_services"),
        "priority_initiatives": skax.get("priority_initiatives"),
        "user_strategy_overlays": skax.get("user_strategy_overlays"),
    }
    return {
        "skax_profile": _compact(profile, depth=0),
        "profile_linkage": _compact(profile_context.get("profile_linkage"), depth=0),
    }


def _issue_scope(
    *,
    package: dict[str, Any],
    integrated_issue: dict[str, Any],
    classification: dict[str, Any],
    card_news_id: str | None,
) -> dict[str, Any]:
    metadata = _json_dict(package.get("metadata"))
    return {
        "card_news_id": card_news_id or metadata.get("card_news_id"),
        "integrated_issue_id": (
            package.get("integrated_issue_id")
            or integrated_issue.get("integrated_issue_id")
            or integrated_issue.get("id")
        ),
        "cluster_id": integrated_issue.get("cluster_id"),
        "headline": integrated_issue.get("headline") or integrated_issue.get("main_issue"),
        "title": integrated_issue.get("title") or integrated_issue.get("headline"),
        "event_type": classification.get("event_type")
        or integrated_issue.get("cluster_event_type")
        or integrated_issue.get("event_type"),
        "main_company": integrated_issue.get("main_company"),
        "fact_summary": integrated_issue.get("fact_summary"),
        "consolidated_facts": integrated_issue.get("consolidated_facts"),
    }


def _companies_from_issue(issue: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("main_company", "company", "peer_company"):
        value = str(issue.get(key) or "").strip()
        if value and value not in values:
            values.append(value)
    return values


def _sectors_from_issue(issue: dict[str, Any], classification: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in _string_list(classification.get("sectors"), max_items=8):
        if item not in values:
            values.append(item)
    for key in ("sector", "industry"):
        value = str(classification.get(key) or issue.get(key) or "").strip()
        if value and value not in values:
            values.append(value)
    return values


def _fact_lines(issue: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("fact_summary", "summary_lines"):
        for item in _string_list(issue.get(key), max_items=8):
            if item not in lines:
                lines.append(item)
    for key in ("consolidated_facts", "extracted_facts"):
        for item in _listish(issue.get(key))[:8]:
            text = _fact_text(item)
            if text and text not in lines:
                lines.append(text)
    integrated_text = str(issue.get("integrated_text") or "").strip()
    if integrated_text:
        for sentence in re.split(r"(?:[.!?。]\s+|다\.\s+)", integrated_text):
            sentence = sentence.strip()
            if 15 <= len(sentence) <= 260 and sentence not in lines:
                lines.append(sentence)
            if len(lines) >= 12:
                break
    return lines


def _fact_text(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("fact", "text", "sentence", "summary", "description"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        return _json_dumps(item)[:260]
    return str(item or "").strip()


def _json_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _listish(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any, *, max_items: int = 10) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _compact(value: Any, *, depth: int) -> Any:
    if depth > 5:
        return str(value)[:360]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if item in ({}, [], "", None):
                continue
            out[str(key)] = _compact(item, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_compact(item, depth=depth + 1) for item in value[:6]]
    text = str(value)
    return text[:900] if len(text) > 900 else value


def _clean_sentence(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _parse_json(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    return data if isinstance(data, dict) else {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
