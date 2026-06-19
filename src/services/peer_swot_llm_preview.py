# 작성일: 2026-06-11
# 작성자: 안가은
# 변경이력:
#   2026-06-11 안가은 — Peer 키워드·SWOT 프리뷰 파이프라인 추가 및 ruff/lint/mypy 정리
#   2026-06-14 최종민 — peer_swot·router gen-search 팩토리 이행
"""Generate Peer+ comparison points and SWOT snapshots.

This service intentionally keeps the generation surface small:

1. Core comparison points describe recently observed business, technology, and
   risk movements. They use only recent business-signal evidence and do not use
   financial numbers.
2. SWOT describes a company diagnosis. It uses a separate diagnostic evidence
   layer built from profile, latest signals, and financial indicators.

The output payload is kept compatible with Peer+:
`comparison_points` labels are `사업 신호`, `기술 신호`, `리스크`.
`swot` labels are `Strength`, `Weakness`, `Opportunity`, `Threat`.
"""

# ruff: noqa: E402, E501

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config.env_loader import load_profile
from src.config.openai_policy import openai_calls_enabled, openai_disabled_reason
from src.db.postgres import reconfigure_from_env
from src.llm import LLMSpec, build_chat_llm
from src.services.peer_swot.data_fetcher import (  # noqa: F401
    annotate_peer_context,
    fetch_business_signals,
    fetch_financial_metrics,
    fetch_peers,
)
from src.services.peer_swot.evidence_builder import (  # noqa: F401
    build_company_evidence,
    build_diagnostic_basis_points,
    build_diagnostic_evidence,
    build_evidence_packs,
    build_pack,
    build_peer_coverage,
    top_values,
)
from src.services.peer_swot.prompts import (  # noqa: F401
    COMPARISON_LABELS,
    COMPARISON_PROMPT,
    COMPARISON_SIGNAL_TYPES,
    COMPARISON_SOURCE_TYPES,
    COMPETITOR_PEER_IDS,
    COMPETITOR_PEER_NAMES,
    DEFAULT_MODEL,
    FINANCIAL_NUMBER_PATTERN,
    GENERIC_CHANGE_OBJECT_BY_LABEL,
    GENERIC_SWOT_TITLE_BY_LABEL,
    NOISE_PHRASES,
    PROMPT_VERSION,
    SK_AX_ID,
    SWOT_FACTOR_TYPE_BY_LABEL,
    SWOT_LABELS,
    SWOT_METRIC_NAMES,
    SWOT_PROMPT,
    SWOT_SIGNAL_TYPES,
    TARGET_PEER_IDS,
)
from src.services.peer_swot.result_normalizer import (  # noqa: F401
    build_comparison_evidence_summary,
    build_comparison_reasoning_summary,
    build_diagnostic_swot_body,
    build_overall_change_object,
    build_overall_comparison_body,
    build_overall_swot_body,
    build_swot_evidence_summary,
    build_swot_reasoning_summary,
    ensure_overall_multi_peer_signal_refs,
    fallback_comparison_item,
    fallback_swot_item,
    filter_swot_diagnostics_for_comparison,
    normalize_comparison_points,
    normalize_swot_items,
    separate_swot_body_from_comparison,
    signals_for_label,
    swot_text_overlaps_comparison,
    themes_from_diagnostics,
    themes_from_signal_refs,
)
from src.services.peer_swot.result_persister import (  # noqa: F401
    average_confidence,
    build_analysis_trace,
    collect_metric_ids,
    collect_peer_ids,
    collect_raw_article_ids,
    collect_signal_ids,
    save_result_to_db,
    save_results_to_db,
)
from src.services.peer_swot.utils import (  # noqa: F401
    build_metric_source_citation,
    build_overall_check_point,
    build_overall_reasoning_coverage_note,
    build_overall_rules,
    build_provenance,
    build_signal_source_citation,
    canonical_comparison_label,
    canonical_swot_label,
    choose_diagnostic_refs_for_label,
    choose_signal_refs_for_label,
    clamp_confidence,
    clean_display_text,
    collect_evidence_ids_from_result,
    collect_urls_for_refs,
    compact_date,
    compact_text,
    content_overlaps_comparison,
    count_distinct_peers_for_refs,
    diagnostic_for_label,
    extract_numeric_ids,
    extract_similarity_tokens,
    extract_specific_phrase,
    extract_theme_terms,
    fallback_check_point,
    fallback_swot_title,
    flatten_limited,
    has_negative_metric_signal,
    has_risk_term,
    has_tech_term,
    hash_payload,
    human_source_name,
    infer_change_object,
    is_generic_change_object,
    is_overall_pack,
    is_semantically_close,
    is_usable_signal_text,
    is_weak_info_text,
    iso_date,
    iter_evidence_items,
    joined_item_text,
    normalize_peer,
    normalize_refs,
    normalize_similarity_text,
    parse_json_object,
    print_or_write_json,
    ref_matches_label,
    remove_financial_number_sentences,
    rough_token_count,
    sanitize_overall_company_names,
    summarize_profile,
    to_float,
    unique_evidence_items,
)

log = logging.getLogger("generate_peer_swot_llm_preview")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Peer+ comparison and SWOT snapshots.")
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--company", action="append", choices=COMPETITOR_PEER_IDS, default=None)
    parser.add_argument(
        "--overall-only", action="store_true", help="Generate only the overall competitor snapshot."
    )
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--fallback-days", type=int, default=365)
    parser.add_argument("--signal-limit", type=int, default=12)
    parser.add_argument("--metric-limit", type=int, default=6)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Build evidence packs only.")
    parser.add_argument(
        "--estimate-tokens", action="store_true", help="Estimate prompt tokens only."
    )
    parser.add_argument(
        "--save-db", action="store_true", help="Persist results to peer_llm_analysis_snapshots."
    )
    parser.add_argument("--output", default=None, help="Optional JSON output file path.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    load_profile(args.env)
    reconfigure_from_env()

    if args.overall_only and args.company:
        raise RuntimeError("--overall-only cannot be used with --company")

    target_ids = (
        COMPETITOR_PEER_IDS if args.overall_only else tuple(args.company or COMPETITOR_PEER_IDS)
    )
    packs = build_evidence_packs(
        target_ids,
        include_overall=args.overall_only or args.company is None,
        include_companies=not args.overall_only,
        days=args.days,
        fallback_days=args.fallback_days,
        signal_limit=args.signal_limit,
        metric_limit=args.metric_limit,
    )

    if args.dry_run or args.estimate_tokens:
        if args.save_db:
            raise RuntimeError("--save-db cannot be used with --dry-run or --estimate-tokens")
        if args.estimate_tokens:
            payload: dict[str, Any] = {
                "mode": "estimate_tokens",
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "successful_llm_calls": len(packs) * 2,
                "estimates": [estimate_pack_tokens(pack) for pack in packs],
            }
            payload["total_prompt_tokens"] = sum(
                item["comparison_prompt_tokens"] + item["swot_prompt_tokens"]
                for item in payload["estimates"]
            )
        else:
            payload = {
                "mode": "dry_run",
                "prompt_version": PROMPT_VERSION,
                "evidence_packs": packs,
            }
    else:
        if not openai_calls_enabled():
            raise RuntimeError(openai_disabled_reason())
        agent = PeerSwotAgent(model=args.model)
        results = [agent.generate(pack) for pack in packs]
        payload = {
            "mode": "llm_preview",
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "results": results,
        }
        if args.save_db:
            payload["saved_count"] = save_results_to_db(packs, results, model=args.model)

    print_or_write_json(payload, args.output)


class PeerSwotAgent:
    def __init__(self, *, model: str) -> None:
        self.model = model
        # env 로 모델 지정 가능 → gpt-5 라도 reasoning_effort 미전달(기존 동작) 위해 None.
        self.llm = build_chat_llm(
            LLMSpec(model=model, temperature=0.1, max_tokens=1800, reasoning_effort=None)
        )

    def generate(self, evidence_pack: dict[str, Any]) -> dict[str, Any]:
        comparison_raw = self._invoke_json(
            COMPARISON_PROMPT, build_comparison_prompt_pack(evidence_pack)
        )
        comparison_points = normalize_comparison_points(comparison_raw, evidence_pack)
        evidence_pack["_normalized_comparison_points"] = comparison_points

        swot_pack = build_swot_prompt_pack(evidence_pack, comparison_points)
        swot_raw = self._invoke_json(SWOT_PROMPT, swot_pack)
        swot_items = normalize_swot_items(
            swot_raw,
            evidence_pack,
            diagnostics_override=swot_pack["diagnostic_evidence"],
        )
        evidence_pack.pop("_normalized_comparison_points", None)

        result = {
            "peer_id": evidence_pack["peer"]["id"],
            "peer_name": evidence_pack["peer"]["name"],
            "model_name": self.model,
            "prompt_version": PROMPT_VERSION,
            "comparison_mode": evidence_pack["comparison_mode"],
            "evidence_hash": evidence_pack["evidence_hash"],
            "comparison_points": comparison_points,
            "swot": swot_items,
            "swot_monitoring_axes": swot_items,
            "analysis_trace": build_analysis_trace(comparison_points, swot_items),
            "overall_check_point": build_overall_check_point(comparison_points, swot_items),
        }
        return result

    def _invoke_json(self, prompt_template: str, prompt_pack: dict[str, Any]) -> dict[str, Any]:
        from src.observability import tracing_config

        prompt = prompt_template.format(
            pack_json=json.dumps(prompt_pack, ensure_ascii=False, indent=2, default=str)
        )
        response = self.llm.invoke(
            prompt,
            config=tracing_config(agent="PeerSwotLlmPreview", phase="invoke_json"),
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return parse_json_object(str(content))


def build_comparison_prompt_pack(evidence_pack: dict[str, Any]) -> dict[str, Any]:
    comparison_input = evidence_pack["comparison_input"]
    return {
        "peer": evidence_pack["peer"],
        "comparison_mode": evidence_pack["comparison_mode"],
        "allowed_comparison_evidence_refs": comparison_input["allowed_comparison_evidence_refs"],
        "signals": comparison_input["signals"],
    }


def build_swot_prompt_pack(
    evidence_pack: dict[str, Any],
    comparison_points: list[dict[str, Any]],
) -> dict[str, Any]:
    swot_input = evidence_pack["swot_input"]
    diagnostic_evidence = filter_swot_diagnostics_for_comparison(
        swot_input["diagnostic_evidence"],
        comparison_points,
    )
    return {
        "peer": evidence_pack["peer"],
        "comparison_mode": evidence_pack["comparison_mode"],
        "allowed_swot_evidence_refs": [item["evidence_id"] for item in diagnostic_evidence],
        "diagnostic_evidence": diagnostic_evidence,
        "required_swot_lens": {
            "Strength": "recent event가 아니라 내부에서 반복 활용 가능한 역량",
            "Weakness": "recent risk가 아니라 내부 개선이 필요한 제약",
            "Opportunity": "recent activity가 아니라 외부 시장·고객·정책이 열어주는 유리한 조건",
            "Threat": "recent issue가 아니라 회사가 통제하기 어려운 외부 압박",
        },
        "forbidden_recent_event_texts": [
            compact_text(
                " / ".join(str(item.get(key) or "") for key in ("change_object", "body")), 220
            )
            for item in comparison_points
        ],
        "observed_comparison_points_for_non_repetition": [
            {
                "label": item.get("label"),
                "change_object": item.get("change_object"),
                "body": item.get("body"),
            }
            for item in comparison_points
        ],
    }


def estimate_pack_tokens(pack: dict[str, Any]) -> dict[str, Any]:
    comparison_prompt = COMPARISON_PROMPT.format(
        pack_json=json.dumps(build_comparison_prompt_pack(pack), ensure_ascii=False, default=str)
    )
    swot_prompt = SWOT_PROMPT.format(
        pack_json=json.dumps(build_swot_prompt_pack(pack, []), ensure_ascii=False, default=str)
    )
    return {
        "peer_id": pack["peer"]["id"],
        "comparison_prompt_tokens": rough_token_count(comparison_prompt),
        "swot_prompt_tokens": rough_token_count(swot_prompt),
    }


if __name__ == "__main__":
    main()
