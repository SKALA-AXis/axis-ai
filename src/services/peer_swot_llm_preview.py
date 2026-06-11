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
import hashlib
import json
import logging
import os
import re
import sys
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.config.env_loader import load_profile
from src.config.openai_policy import openai_calls_enabled, openai_disabled_reason
from src.db.postgres import SessionLocal, reconfigure_from_env

log = logging.getLogger("generate_peer_swot_llm_preview")

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


def build_evidence_packs(
    peer_ids: tuple[str, ...],
    *,
    include_overall: bool,
    include_companies: bool,
    days: int,
    fallback_days: int,
    signal_limit: int,
    metric_limit: int,
) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        requested_ids = tuple(dict.fromkeys((SK_AX_ID, *peer_ids)))
        peers = fetch_peers(db, requested_ids)
        peer_by_id = {peer["id"]: peer for peer in peers}
        if SK_AX_ID not in peer_by_id:
            raise RuntimeError("SK AX peer row is required")

        company_evidence_by_id = {
            peer["id"]: build_company_evidence(
                db,
                peer,
                days=days,
                fallback_days=fallback_days,
                signal_limit=signal_limit,
                metric_limit=metric_limit,
            )
            for peer in peers
        }

    packs: list[dict[str, Any]] = []
    if include_overall:
        competitor_evidence = [
            company_evidence_by_id[peer_id]
            for peer_id in COMPETITOR_PEER_IDS
            if peer_id in company_evidence_by_id
        ]
        packs.append(
            build_pack(
                peer={"id": "all", "name": "전체 경쟁사", "profile_snapshot": None},
                comparison_mode="overall_competitors_vs_sk_ax",
                companies=competitor_evidence,
            )
        )

    if not include_companies:
        return packs

    for peer_id in peer_ids:
        company_evidence = company_evidence_by_id.get(peer_id)
        if company_evidence is None:
            log.warning("요청 peer를 찾지 못했습니다 | peer_id=%s", peer_id)
            continue
        packs.append(
            build_pack(
                peer=company_evidence["peer"],
                comparison_mode="peer_vs_sk_ax",
                companies=[company_evidence],
            )
        )
    return packs


def build_company_evidence(
    db: Any,
    peer: dict[str, Any],
    *,
    days: int,
    fallback_days: int,
    signal_limit: int,
    metric_limit: int,
) -> dict[str, Any]:
    peer_id = str(peer["id"])
    comparison_signals = fetch_business_signals(
        db,
        peer_id,
        days=days,
        limit=signal_limit,
        signal_types=COMPARISON_SIGNAL_TYPES,
        source_types=COMPARISON_SOURCE_TYPES,
    )
    if not comparison_signals and fallback_days > days:
        comparison_signals = fetch_business_signals(
            db,
            peer_id,
            days=fallback_days,
            limit=signal_limit,
            signal_types=COMPARISON_SIGNAL_TYPES,
            source_types=COMPARISON_SOURCE_TYPES,
        )

    swot_signals = fetch_business_signals(
        db,
        peer_id,
        days=fallback_days,
        limit=signal_limit,
        signal_types=SWOT_SIGNAL_TYPES,
        source_types=None,
    )
    metrics = fetch_financial_metrics(
        db,
        peer_id,
        limit=metric_limit,
        metric_names=SWOT_METRIC_NAMES,
    )
    peer_ref = normalize_peer(peer)
    annotate_peer_context(comparison_signals, peer_ref)
    annotate_peer_context(swot_signals, peer_ref)
    annotate_peer_context(metrics, peer_ref)
    diagnostics = build_diagnostic_evidence(peer, swot_signals, metrics)
    return {
        "peer": peer_ref,
        "comparison_signals": comparison_signals,
        "swot_source_signals": swot_signals,
        "financial_metrics": metrics,
        "diagnostic_evidence": diagnostics,
    }


def build_pack(
    *,
    peer: dict[str, Any],
    comparison_mode: str,
    companies: list[dict[str, Any]],
) -> dict[str, Any]:
    comparison_signals = flatten_limited(
        [company["comparison_signals"] for company in companies],
        limit=max(12, len(companies) * 4),
    )
    diagnostic_evidence = flatten_limited(
        [company["diagnostic_evidence"] for company in companies],
        limit=max(16, len(companies) * 4),
    )
    pack: dict[str, Any] = {
        "peer": normalize_peer(peer),
        "reference_peer": {"id": SK_AX_ID, "name": "SK AX"},
        "comparison_mode": comparison_mode,
        "prompt_version": PROMPT_VERSION,
        "comparison_input": {
            "purpose": "recent observed movements only; no financial numbers",
            "allowed_comparison_evidence_refs": [
                item["evidence_id"] for item in comparison_signals
            ],
            "signals": comparison_signals,
            "overall_rules": build_overall_rules(peer, companies),
            "overall_peer_coverage": build_peer_coverage(companies),
        },
        "swot_input": {
            "purpose": "company diagnosis; separate from recent comparison points",
            "allowed_swot_evidence_refs": [item["evidence_id"] for item in diagnostic_evidence],
            "diagnostic_evidence": diagnostic_evidence,
            "overall_rules": build_overall_rules(peer, companies),
            "overall_peer_coverage": build_peer_coverage(companies),
        },
        "companies": companies,
    }
    pack["evidence_hash"] = hash_payload(
        {
            "peer": pack["peer"],
            "comparison_mode": comparison_mode,
            "comparison_refs": pack["comparison_input"]["allowed_comparison_evidence_refs"],
            "swot_refs": pack["swot_input"]["allowed_swot_evidence_refs"],
        }
    )
    return pack


def build_overall_rules(peer: dict[str, Any], companies: list[dict[str, Any]]) -> list[str]:
    if peer.get("id") != "all":
        return []
    return [
        "전체 결과는 특정 1개 회사가 아니라 peer 그룹의 IT 서비스 산업 흐름을 요약한다.",
        "가능하면 2개 이상 peer에서 반복되는 AI, 클라우드, 보안, 데이터, 자동화, 산업 DX 흐름을 우선한다.",
        "한 회사 근거만 있는 경우 전체 트렌드로 단정하지 말고 근거 제한을 명시한다.",
    ]


def build_peer_coverage(companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for company in companies:
        peer = company.get("peer") or {}
        signals = company.get("comparison_signals") or []
        coverage.append(
            {
                "peer_id": peer.get("id"),
                "peer_name": peer.get("name"),
                "signal_count": len(signals),
                "top_business_areas": top_values(signals, "business_area", limit=4),
                "top_signal_types": top_values(signals, "signal_type", limit=4),
            }
        )
    return coverage


def top_values(items: list[dict[str, Any]], key: str, *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        value = compact_text(item.get(key), 60)
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return [
        value
        for value, _count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    ]


def annotate_peer_context(items: list[dict[str, Any]], peer: dict[str, Any]) -> None:
    for item in items:
        item["peer_id"] = peer.get("id")
        item["peer_name"] = peer.get("name")


def fetch_peers(db: Any, peer_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT id, name, profile_snapshot
            FROM peer_companies
            WHERE id = ANY(:peer_ids)
            ORDER BY CASE id
                WHEN 'sk_ax' THEN 0
                WHEN 'samsung_sds' THEN 1
                WHEN 'lg_cns' THEN 2
                WHEN 'hyundai_autoever' THEN 3
                WHEN 'posco_dx' THEN 4
                ELSE 99
            END
            """
        ),
        {"peer_ids": list(peer_ids)},
    ).mappings()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "profile_snapshot": row["profile_snapshot"],
        }
        for row in rows
    ]


def fetch_business_signals(
    db: Any,
    peer_id: str,
    *,
    days: int,
    limit: int,
    signal_types: set[str] | None,
    source_types: set[str] | None,
) -> list[dict[str, Any]]:
    filters = [
        "rabs.peer_id = :peer_id",
        "COALESCE(ra.published_at, ra.collected_at, ra.created_at, rabs.created_at) >= "
        "NOW() - (:days || ' days')::interval",
        "COALESCE(rabs.summary, '') <> ''",
    ]
    params: dict[str, Any] = {
        "peer_id": peer_id,
        "days": days,
        "limit": max(limit * 5, limit),
    }
    if signal_types:
        filters.append("rabs.signal_type = ANY(:signal_types)")
        params["signal_types"] = list(signal_types)
    if source_types:
        filters.append("rabs.source_type = ANY(:source_types)")
        params["source_types"] = list(source_types)

    rows = db.execute(
        text(
            f"""
            SELECT
                rabs.id,
                rabs.raw_article_id,
                rabs.source_type,
                rabs.business_area,
                rabs.signal_type,
                rabs.sentiment,
                rabs.summary,
                rabs.evidence_text,
                rabs.confidence,
                rabs.created_at,
                rabs.updated_at,
                ra.title,
                ra.url,
                COALESCE(ra.published_at, ra.collected_at, ra.created_at, rabs.created_at) AS evidence_at,
                ra.source_name
            FROM raw_article_business_signals rabs
            JOIN raw_articles ra ON ra.id = rabs.raw_article_id
            WHERE {" AND ".join(filters)}
            ORDER BY
                rabs.updated_at DESC NULLS LAST,
                rabs.confidence DESC NULLS LAST,
                COALESCE(ra.published_at, ra.collected_at, ra.created_at, rabs.created_at) DESC NULLS LAST,
                rabs.id DESC
            LIMIT :limit
            """
        ),
        params,
    ).mappings()

    signals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        summary = compact_text(row["summary"], 300)
        evidence_text = compact_text(row["evidence_text"], 260)
        if not is_usable_signal_text(summary, evidence_text) or summary in seen:
            continue
        seen.add(summary)
        signals.append(
            {
                "evidence_id": f"signal:{row['id']}",
                "article_id": row["raw_article_id"],
                "source_type": row["source_type"],
                "source_name": row["source_name"],
                "business_area": row["business_area"],
                "signal_type": row["signal_type"],
                "sentiment": row["sentiment"],
                "title": compact_text(row["title"], 120),
                "summary": summary,
                "evidence_text": evidence_text,
                "url": row["url"],
                "date": iso_date(row["evidence_at"]),
                "updated_at": iso_date(row["updated_at"]),
                "confidence": to_float(row["confidence"]),
            }
        )
    return signals[:limit]


def fetch_financial_metrics(
    db: Any,
    peer_id: str,
    *,
    limit: int,
    metric_names: set[str],
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT
                rafm.id,
                rafm.raw_article_id,
                rafm.period,
                rafm.metric_name,
                rafm.metric_label,
                rafm.metric_scope,
                rafm.business_area,
                rafm.value_numeric,
                rafm.value_krwbn,
                rafm.unit,
                rafm.evidence_text,
                rafm.confidence,
                rafm.updated_at,
                ra.title,
                ra.url,
                COALESCE(ra.published_at, ra.collected_at, ra.created_at) AS evidence_at,
                ra.source_name
            FROM raw_article_financial_metrics rafm
            LEFT JOIN raw_articles ra ON ra.id = rafm.raw_article_id
            WHERE rafm.peer_id = :peer_id
              AND rafm.metric_name = ANY(:metric_names)
            ORDER BY
                rafm.updated_at DESC NULLS LAST,
                rafm.period_year DESC NULLS LAST,
                rafm.period_quarter DESC NULLS LAST,
                rafm.confidence DESC NULLS LAST,
                rafm.id DESC
            LIMIT :limit
            """
        ),
        {
            "peer_id": peer_id,
            "metric_names": list(metric_names),
            "limit": max(limit * 4, limit),
        },
    ).mappings()

    metrics: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for row in rows:
        key = (row["period"], row["metric_name"], row["metric_scope"])
        if key in seen:
            continue
        seen.add(key)
        metrics.append(
            {
                "evidence_id": f"metric:{row['id']}",
                "article_id": row["raw_article_id"],
                "period": row["period"],
                "metric_name": row["metric_name"],
                "metric_label": row["metric_label"],
                "metric_scope": row["metric_scope"],
                "business_area": row["business_area"],
                "value_numeric": to_float(row["value_numeric"]),
                "value_krwbn": to_float(row["value_krwbn"]),
                "unit": row["unit"],
                "evidence_text": compact_text(row["evidence_text"], 240),
                "updated_at": iso_date(row["updated_at"]),
                "title": compact_text(row["title"], 120),
                "url": row["url"],
                "date": iso_date(row["evidence_at"]),
                "source_name": row["source_name"],
                "confidence": to_float(row["confidence"]),
            }
        )
    return metrics[:limit]


def build_diagnostic_evidence(
    peer: dict[str, Any],
    signals: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    peer_id = str(peer["id"])
    profile_text = summarize_profile(peer.get("profile_snapshot"))
    business_signals = unique_evidence_items(
        [s for s in signals if str(s.get("signal_type")) not in {"rd", "risk"}]
    )
    tech_signals = unique_evidence_items(
        [s for s in signals if str(s.get("signal_type")) == "rd" or has_tech_term(s)]
    )
    risk_signals = unique_evidence_items(
        [s for s in signals if str(s.get("signal_type")) == "risk" or has_risk_term(s)]
    )
    external_signals = unique_evidence_items(
        [
            s
            for s in signals
            if str(s.get("signal_type")) in {"forecast", "investment", "valuation", "risk"}
            or str(s.get("source_type")) == "securities_report"
        ]
    )
    weakness_metrics = [
        m
        for m in metrics
        if has_negative_metric_signal(m) or str(m.get("metric_name", "")).startswith("operating")
    ]

    specs = [
        (
            "Strength",
            "strength",
            "Internal capability/differentiator",
            "반복적으로 활용 가능한 내부 역량이나 차별화 자산은 무엇인가?",
            "내부 통제 가능 요소이며, 여러 고객·산업·서비스로 재사용될 수 있는 역량인지 판단",
            [*tech_signals[:2], *business_signals[:2]],
            [],
            profile_text,
        ),
        (
            "Weakness",
            "weakness",
            "Internal limitation or improvement area",
            "성과를 제약하는 내부 비용, 실행 부담, 수익성, 집중도 문제는 무엇인가?",
            "회사가 개선하거나 관리해야 하는 내부 제약인지 판단",
            risk_signals[:2],
            weakness_metrics[:2],
            profile_text,
        ),
        (
            "Opportunity",
            "opportunity",
            "External favorable market/customer/technology condition",
            "외부 시장·고객·정책·기술 변화 중 활용 가능한 기회는 무엇인가?",
            "회사가 통제할 수는 없지만 사업 확장에 유리하게 작용할 외부 조건인지 판단",
            external_signals[:3] or business_signals[:2],
            [],
            "",
        ),
        (
            "Threat",
            "threat",
            "External unfavorable competition/regulation/market condition",
            "외부 경쟁, 수요, 규제, 기술 변화 중 성과를 압박할 요인은 무엇인가?",
            "회사가 통제하기 어렵고 사업 속도나 수익성을 낮출 수 있는 외부 조건인지 판단",
            risk_signals[:3] or external_signals[:2],
            [],
            "",
        ),
    ]

    diagnostics: list[dict[str, Any]] = []
    for (
        label,
        slug,
        diagnosis_type,
        diagnostic_question,
        judgment_basis,
        selected_signals,
        selected_metrics,
        profile,
    ) in specs:
        source_refs = [item["evidence_id"] for item in selected_signals + selected_metrics]
        digest = hashlib.sha1(
            json.dumps(
                [peer_id, label, source_refs, profile], ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()[:10]
        basis_points = build_diagnostic_basis_points(
            label, selected_signals, selected_metrics, profile
        )
        source_signal_citations = [build_signal_source_citation(item) for item in selected_signals]
        source_metric_citations = [build_metric_source_citation(item) for item in selected_metrics]
        source_citations = [*source_signal_citations, *source_metric_citations]
        diagnostics.append(
            {
                "evidence_id": f"diagnostic:{peer_id}:{slug}:{digest}",
                "peer_id": peer_id,
                "peer_name": peer.get("name"),
                "label": label,
                "diagnosis_type": diagnosis_type,
                "diagnostic_question": diagnostic_question,
                "judgment_basis": judgment_basis,
                "basis_points": basis_points,
                "source_citations": [citation for citation in source_citations if citation],
                "source_signal_citations": [
                    citation for citation in source_signal_citations if citation
                ],
                "source_metric_citations": [
                    citation for citation in source_metric_citations if citation
                ],
                "factor_type": SWOT_FACTOR_TYPE_BY_LABEL[label],
                "profile_hint": compact_text(profile, 320),
                "source_signal_refs": [item["evidence_id"] for item in selected_signals],
                "source_metric_refs": [item["evidence_id"] for item in selected_metrics],
                "signal_summaries": [
                    compact_text(
                        " / ".join(
                            str(value or "")
                            for value in (
                                item.get("business_area"),
                                item.get("signal_type"),
                                item.get("summary"),
                            )
                        ),
                        260,
                    )
                    for item in selected_signals
                ],
                "metric_summaries": [
                    compact_text(
                        " / ".join(
                            str(value or "")
                            for value in (
                                item.get("period"),
                                item.get("metric_label") or item.get("metric_name"),
                                item.get("business_area"),
                                item.get("evidence_text"),
                            )
                        ),
                        220,
                    )
                    for item in selected_metrics
                ],
            }
        )
    return diagnostics


class PeerSwotAgent:
    def __init__(self, *, model: str) -> None:
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0.1, max_completion_tokens=1800)

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
        prompt = prompt_template.format(
            pack_json=json.dumps(prompt_pack, ensure_ascii=False, indent=2, default=str)
        )
        response = self.llm.invoke(prompt)
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


def filter_swot_diagnostics_for_comparison(
    diagnostics: list[dict[str, Any]],
    comparison_points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparison_refs = {
        str(ref)
        for item in comparison_points
        if isinstance(item, dict)
        for ref in item.get("evidence_refs") or []
    }
    comparison_texts = [
        compact_text(
            " / ".join(str(item.get(key) or "") for key in ("change_object", "body")),
            260,
        )
        for item in comparison_points
        if isinstance(item, dict)
    ]
    filtered_diagnostics: list[dict[str, Any]] = []
    for diagnosis in diagnostics:
        filtered = dict(diagnosis)
        source_signal_refs = [str(ref) for ref in diagnosis.get("source_signal_refs") or []]
        signal_summaries = [str(summary) for summary in diagnosis.get("signal_summaries") or []]
        signal_citations = [
            str(citation) for citation in diagnosis.get("source_signal_citations") or []
        ]
        kept_refs: list[str] = []
        kept_summaries: list[str] = []
        kept_signal_citations: list[str] = []
        removed_count = 0
        for index, summary in enumerate(signal_summaries):
            ref = source_signal_refs[index] if index < len(source_signal_refs) else ""
            if ref in comparison_refs or content_overlaps_comparison(summary, comparison_texts):
                removed_count += 1
                continue
            if ref:
                kept_refs.append(ref)
            kept_summaries.append(summary)
            if index < len(signal_citations):
                kept_signal_citations.append(signal_citations[index])

        basis_points = [str(point) for point in diagnosis.get("basis_points") or []]
        kept_basis_points = [
            point
            for point in basis_points
            if not content_overlaps_comparison(point, comparison_texts)
        ]
        removed_count += len(basis_points) - len(kept_basis_points)

        filtered["source_signal_refs"] = kept_refs
        filtered["signal_summaries"] = kept_summaries
        filtered["source_signal_citations"] = kept_signal_citations
        filtered["source_citations"] = [
            *kept_signal_citations,
            *[str(citation) for citation in diagnosis.get("source_metric_citations") or []],
        ]
        filtered["basis_points"] = kept_basis_points
        if not kept_basis_points and not kept_summaries and not filtered.get("source_metric_refs"):
            filtered["basis_points"] = [
                "비교 포인트와 독립적으로 사용할 수 있는 SWOT 진단 근거가 제한적입니다."
            ]
            filtered["insufficient_independent_evidence"] = True
        if removed_count > 0:
            filtered["comparison_content_excluded"] = True
            filtered["diagnosis_guardrail"] = (
                "비교 포인트와 같은 원문 근거·주제는 제외했으며, 남은 근거로만 SWOT을 판단한다."
            )
        filtered_diagnostics.append(filtered)
    return filtered_diagnostics


def content_overlaps_comparison(value: str, comparison_texts: list[str]) -> bool:
    if not value:
        return False
    return any(
        is_semantically_close(value, comparison_text) for comparison_text in comparison_texts
    )


def is_overall_pack(evidence_pack: dict[str, Any]) -> bool:
    return evidence_pack.get("peer", {}).get("id") == "all"


def ensure_overall_multi_peer_signal_refs(
    label: str,
    refs: list[str],
    signals: list[dict[str, Any]],
) -> list[str]:
    selected = [signal for signal in signals if str(signal.get("evidence_id")) in set(refs)]
    selected_peer_ids = {str(signal.get("peer_id")) for signal in selected if signal.get("peer_id")}
    if len(selected_peer_ids) >= 2:
        return refs[:3]

    candidates = signals_for_label(label, signals)
    expanded_refs = list(refs)
    for signal in candidates:
        peer_id = str(signal.get("peer_id") or "")
        evidence_id = str(signal.get("evidence_id") or "")
        if not evidence_id or evidence_id in expanded_refs:
            continue
        if peer_id in selected_peer_ids and len(selected_peer_ids) >= 1:
            continue
        expanded_refs.append(evidence_id)
        if peer_id:
            selected_peer_ids.add(peer_id)
        if len(selected_peer_ids) >= 2 or len(expanded_refs) >= 3:
            break
    return expanded_refs[:3]


def signals_for_label(label: str, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if label == "기술 신호":
        selected = [
            signal
            for signal in signals
            if str(signal.get("signal_type")) == "rd" or has_tech_term(signal)
        ]
    elif label == "리스크":
        selected = [
            signal
            for signal in signals
            if str(signal.get("signal_type")) == "risk" or has_risk_term(signal)
        ]
    else:
        selected = [
            signal for signal in signals if str(signal.get("signal_type")) not in {"rd", "risk"}
        ]
    return selected or signals


def build_overall_change_object(
    label: str,
    refs: list[str],
    signals: list[dict[str, Any]],
    fallback: str,
) -> str:
    themes = themes_from_signal_refs(refs, signals)
    if themes:
        return "·".join(themes[:2])
    defaults = {
        "사업 신호": "AI·클라우드 사업 흐름",
        "기술 신호": "AI·클라우드 기술 흐름",
        "리스크": "클라우드·AI 경쟁 리스크",
    }
    return defaults.get(label, fallback)


def build_overall_comparison_body(
    label: str,
    refs: list[str],
    signals: list[dict[str, Any]],
    change_object: str,
) -> str:
    themes = themes_from_signal_refs(refs, signals)
    theme_text = "·".join(themes[:3]) if themes else change_object
    bodies = {
        "사업 신호": f"{theme_text}를 중심으로 고객 산업과 서비스 확장 방향이 재편되고 있습니다. 이는 국내 IT서비스 시장의 수요가 단순 SI보다 AI·클라우드 기반 실행 영역으로 이동하고 있음을 보여줍니다.",
        "기술 신호": f"{theme_text} 관련 기술 적용과 플랫폼화가 경쟁 축으로 부상하고 있습니다. AI, 클라우드, 데이터 기반 역량이 산업 공통의 기술 차별화 요소로 이동하는 흐름입니다.",
        "리스크": f"{theme_text} 확대와 함께 실행 부담과 경쟁 압력이 커지고 있습니다. IT서비스 산업의 수요·투자·경쟁 변동성을 함께 관리해야 하는 상황으로 해석됩니다.",
    }
    return bodies[label]


def build_overall_swot_body(
    label: str,
    diagnostics: list[dict[str, Any]],
    evidence_pack: dict[str, Any],
) -> str:
    themes = themes_from_diagnostics(diagnostics)
    theme_text = "·".join(themes[:3]) if themes else "AI·클라우드·산업 DX"
    bodies = {
        "Strength": f"{theme_text} 역량이 고객 산업 전반으로 확장될 수 있는 실행 기반으로 축적되고 있습니다. 이는 국내 IT서비스 시장에서 반복 적용 가능한 기술·운영 역량이 강점으로 작용할 수 있음을 의미합니다.",
        "Weakness": f"{theme_text} 확대 과정에서 비용, 인력, 납기, 수익성 관리 부담이 함께 커질 수 있습니다. 서비스 고도화 속도만큼 내부 운영 효율과 프로젝트 관리 체계를 개선해야 하는 과제로 해석됩니다.",
        "Opportunity": f"{theme_text} 수요가 공공, 금융, 제조 등 여러 고객 산업으로 확장되고 있습니다. 이는 IT서비스 산업이 활용할 수 있는 외부 수요 조건이 넓어지고 있다는 의미입니다.",
        "Threat": f"{theme_text} 시장에서 경쟁 강도와 고객 투자 변동성이 동시에 커지고 있습니다. 성장 속도와 수익성을 압박할 수 있는 외부 조건으로 관리가 필요합니다.",
    }
    return bodies[label]


def themes_from_signal_refs(refs: list[str], signals: list[dict[str, Any]]) -> list[str]:
    by_ref = {str(signal.get("evidence_id")): signal for signal in signals}
    texts = [
        " ".join(str(signal.get(key) or "") for key in ("business_area", "summary", "title"))
        for ref in refs
        for signal in [by_ref.get(ref)]
        if signal
    ]
    return extract_theme_terms(" ".join(texts))


def themes_from_diagnostics(diagnostics: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for diagnosis in diagnostics:
        texts.extend(str(point) for point in diagnosis.get("basis_points") or [])
        texts.extend(str(summary) for summary in diagnosis.get("signal_summaries") or [])
    return extract_theme_terms(" ".join(texts))


def extract_theme_terms(text_value: str) -> list[str]:
    lowered = str(text_value or "").lower()
    candidates = [
        ("AI", ("ai", "생성형", "llm", "agent", "에이전트")),
        ("클라우드", ("cloud", "클라우드", "msp", "gpu", "데이터센터")),
        ("데이터", ("데이터", "data")),
        ("보안", ("보안", "security", "인증", "암호")),
        ("자동화", ("자동화", "automation", "robot", "로봇")),
        ("산업 DX", ("dx", "스마트팩토리", "제조", "산업")),
        ("ERP", ("erp",)),
        ("모빌리티", ("모빌리티", "차량", "ota", "자동차")),
    ]
    themes: list[str] = []
    for label, terms in candidates:
        if any(term in lowered for term in terms):
            themes.append(label)
    return themes[:4]


def count_distinct_peers_for_refs(refs: list[str], signals: list[dict[str, Any]]) -> int:
    by_ref = {str(signal.get("evidence_id")): signal for signal in signals}
    return len(
        {
            str(signal.get("peer_id"))
            for ref in refs
            for signal in [by_ref.get(ref)]
            if signal and signal.get("peer_id")
        }
    )


def sanitize_overall_company_names(value: str) -> str:
    text_value = str(value or "")
    for name in COMPETITOR_PEER_NAMES:
        text_value = text_value.replace(name, "일부 기업")
    return re.sub(r"(일부 기업[,·/ ]*){2,}", "일부 기업 ", text_value).strip()


def normalize_comparison_points(
    raw: dict[str, Any], evidence_pack: dict[str, Any]
) -> list[dict[str, Any]]:
    raw_items = raw.get("comparison_points") if isinstance(raw, dict) else []
    if not isinstance(raw_items, list):
        raw_items = []

    allowed_refs = set(evidence_pack["comparison_input"]["allowed_comparison_evidence_refs"])
    signals = evidence_pack["comparison_input"]["signals"]
    by_label: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        label = canonical_comparison_label(item.get("label"))
        if label and label not in by_label:
            by_label[label] = item

    normalized: list[dict[str, Any]] = []
    for label in COMPARISON_LABELS:
        item = dict(by_label.get(label) or fallback_comparison_item(label, signals))
        body = clean_display_text(str(item.get("body") or ""))
        body = remove_financial_number_sentences(body)
        if not body:
            body = str(fallback_comparison_item(label, signals)["body"])
        refs = normalize_refs(item.get("evidence_refs"), allowed_refs)
        if not refs:
            refs = choose_signal_refs_for_label(label, signals)
        if is_overall_pack(evidence_pack):
            refs = ensure_overall_multi_peer_signal_refs(label, refs, signals)
        source_urls = collect_urls_for_refs(refs, signals)
        change_object = compact_text(item.get("change_object"), 80) or infer_change_object(
            label, refs, signals
        )
        if is_generic_change_object(change_object):
            change_object = GENERIC_CHANGE_OBJECT_BY_LABEL[label]
        evidence_summary = build_comparison_evidence_summary(label, refs, signals)
        reasoning_summary = build_comparison_reasoning_summary(
            label,
            change_object,
            refs,
            signals,
            evidence_pack=evidence_pack,
        )
        if is_overall_pack(evidence_pack):
            body = build_overall_comparison_body(label, refs, signals, change_object)
            change_object = build_overall_change_object(label, refs, signals, change_object)
            reasoning_summary = sanitize_overall_company_names(reasoning_summary)
            evidence_summary = sanitize_overall_company_names(evidence_summary)
        normalized.append(
            {
                "label": label,
                "change_object": change_object,
                "body": compact_text(body, 420),
                "evidence_refs": refs,
                "source_urls": source_urls,
                "confidence": clamp_confidence(item.get("confidence")),
                "evidence_summary": evidence_summary,
                "reasoning_summary": reasoning_summary,
                "insufficient_evidence": bool(item.get("insufficient_evidence"))
                or not refs
                or change_object == GENERIC_CHANGE_OBJECT_BY_LABEL[label],
            }
        )
    return normalized


def normalize_swot_items(
    raw: dict[str, Any],
    evidence_pack: dict[str, Any],
    *,
    diagnostics_override: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw_items: list[Any] = []
    if isinstance(raw, dict):
        raw_items = raw.get("swot") or raw.get("swot_monitoring_axes") or []
    if not isinstance(raw_items, list):
        raw_items = []

    diagnostics = diagnostics_override or evidence_pack["swot_input"]["diagnostic_evidence"]
    allowed_refs = {str(item.get("evidence_id")) for item in diagnostics if item.get("evidence_id")}
    by_label: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        label = canonical_swot_label(item.get("label"))
        if label is None and index < len(SWOT_LABELS):
            label = SWOT_LABELS[index]
        if label and label not in by_label:
            by_label[label] = item

    normalized: list[dict[str, Any]] = []
    for label in SWOT_LABELS:
        item = dict(by_label.get(label) or fallback_swot_item(label, diagnostics))
        refs = normalize_refs(item.get("evidence_refs"), allowed_refs)
        preferred_refs = choose_diagnostic_refs_for_label(label, diagnostics)
        if not refs or not ref_matches_label(refs[0], label):
            refs = preferred_refs or refs[:1]
        body = clean_display_text(str(item.get("body") or ""))
        if not body:
            body = str(fallback_swot_item(label, diagnostics)["body"])
        title = clean_display_text(item.get("title") or item.get("axis_name") or "")
        if not title:
            title = fallback_swot_title(label, diagnostics)
        if swot_text_overlaps_comparison(title, evidence_pack):
            title = GENERIC_SWOT_TITLE_BY_LABEL[label]
        if is_overall_pack(evidence_pack):
            title = GENERIC_SWOT_TITLE_BY_LABEL[label]
        diagnosis = diagnostic_for_label(label, diagnostics)
        body = separate_swot_body_from_comparison(label, body, title, evidence_pack, diagnostics)
        if is_overall_pack(evidence_pack):
            body = build_overall_swot_body(label, diagnostics, evidence_pack)
        evidence_summary = build_swot_evidence_summary(diagnosis)
        reasoning_summary = build_swot_reasoning_summary(label, title, diagnosis)
        if is_overall_pack(evidence_pack):
            reasoning_summary = sanitize_overall_company_names(reasoning_summary)
        normalized.append(
            {
                "label": label,
                "title": compact_text(title, 80),
                "body": compact_text(body, 430),
                "factor_type": SWOT_FACTOR_TYPE_BY_LABEL[label],
                "check_point": compact_text(clean_display_text(item.get("check_point")), 180)
                or fallback_check_point(label),
                "evidence_refs": refs,
                "source_urls": [],
                "confidence": clamp_confidence(item.get("confidence")),
                "evidence_summary": evidence_summary,
                "reasoning_summary": reasoning_summary,
                "insufficient_evidence": bool(item.get("insufficient_evidence")) or not refs,
            }
        )
    return normalized


def fallback_comparison_item(label: str, signals: list[dict[str, Any]]) -> dict[str, Any]:
    refs = choose_signal_refs_for_label(label, signals)
    object_name = infer_change_object(label, refs, signals)
    lacks_specific_object = object_name in GENERIC_CHANGE_OBJECT_BY_LABEL.values()
    if label == "사업 신호":
        body = (
            f"{object_name} 수준의 사업 움직임만 확인되며, 구체 사업명은 추가 근거 확인이 필요합니다."
            if lacks_specific_object
            else f"{object_name} 관련 사업 움직임이 최근 근거에서 확인됩니다."
        )
    elif label == "기술 신호":
        body = (
            f"{object_name} 수준의 기술 움직임만 확인되며, 구체 플랫폼명은 추가 근거 확인이 필요합니다."
            if lacks_specific_object
            else f"{object_name} 관련 기술 적용 또는 플랫폼화 움직임이 확인됩니다."
        )
    else:
        body = (
            f"{object_name} 수준의 리스크만 확인되며, 구체 원인은 추가 근거 확인이 필요합니다."
            if lacks_specific_object
            else f"{object_name} 관련 실행 부담이나 시장 불확실성을 함께 점검해야 합니다."
        )
    return {
        "label": label,
        "change_object": object_name,
        "body": body,
        "evidence_refs": refs,
        "confidence": 0.5 if refs else 0.2,
        "evidence_summary": "입력 근거에서 확인 가능한 최신 신호를 기준으로 보수적으로 작성했습니다.",
        "reasoning_summary": "핵심 비교 포인트는 최근 관찰된 변화만 요약했습니다.",
        "insufficient_evidence": not refs or lacks_specific_object,
    }


def fallback_swot_item(label: str, diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    refs = choose_diagnostic_refs_for_label(label, diagnostics)
    title = fallback_swot_title(label, diagnostics)
    diagnosis = diagnostic_for_label(label, diagnostics)
    bodies = {
        "Strength": f"{title}은 내부적으로 경쟁사 대비 활용 가능한 역량입니다.",
        "Weakness": f"{title}은 내부 개선 과제로 남아 있어 실행 품질과 수익성 관리가 필요합니다.",
        "Opportunity": f"{title}은 외부 시장 변화가 유리하게 작용할 수 있는 지점입니다.",
        "Threat": f"{title}은 외부 환경 변화가 사업 성과를 압박할 수 있는 지점입니다.",
    }
    return {
        "label": label,
        "title": title,
        "body": bodies[label],
        "factor_type": SWOT_FACTOR_TYPE_BY_LABEL[label],
        "check_point": fallback_check_point(label),
        "evidence_refs": refs,
        "confidence": 0.45 if refs else 0.2,
        "evidence_summary": build_swot_evidence_summary(diagnosis),
        "reasoning_summary": build_swot_reasoning_summary(label, title, diagnosis),
        "insufficient_evidence": not refs,
    }


def build_diagnostic_basis_points(
    label: str,
    signals: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    profile: str,
) -> list[str]:
    points: list[str] = []
    if profile and label in {"Strength", "Weakness"}:
        points.append(f"회사 프로필상 내부 사업·역량 설명: {compact_text(profile, 120)}")
    for signal in signals[:2]:
        area = compact_text(signal.get("business_area"), 40) or "사업 영역 미상"
        signal_type = compact_text(signal.get("signal_type"), 30) or "신호 유형 미상"
        summary = compact_text(signal.get("summary"), 120)
        points.append(f"{area}/{signal_type} 근거: {summary}")
    for metric in metrics[:2]:
        metric_name = compact_text(metric.get("metric_label") or metric.get("metric_name"), 50)
        period = compact_text(metric.get("period"), 20)
        evidence = compact_text(metric.get("evidence_text"), 120)
        points.append(f"{period} {metric_name} 근거: {evidence}")
    if not points:
        points.append("해당 SWOT 항목을 강하게 뒷받침하는 독립 근거가 제한적입니다.")
    return points[:4]


def diagnostic_for_label(label: str, diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return next((item for item in diagnostics if item.get("label") == label), {})


def build_comparison_evidence_summary(
    label: str, refs: list[str], signals: list[dict[str, Any]]
) -> str:
    by_ref = {item.get("evidence_id"): item for item in signals}
    picked = [by_ref[ref] for ref in refs if ref in by_ref]
    if not picked:
        return "출처: 해당 판단에 직접 연결되는 공개 원문 출처가 부족합니다."
    citations = [build_signal_source_citation(item) for item in picked[:2]]
    citations = [citation for citation in citations if citation]
    if not citations:
        return "출처: 원문 제목·발행일·URL 정보가 제한적입니다."
    return compact_text("출처: " + " | ".join(citations), 260)


def build_comparison_reasoning_summary(
    label: str,
    change_object: str,
    refs: list[str],
    signals: list[dict[str, Any]],
    *,
    evidence_pack: dict[str, Any] | None = None,
) -> str:
    by_ref = {item.get("evidence_id"): item for item in signals}
    picked = [by_ref[ref] for ref in refs if ref in by_ref]
    checked = (
        compact_text(
            " / ".join(
                str(value or "")
                for item in picked[:1]
                for value in (item.get("business_area"), item.get("summary"))
            ),
            110,
        )
        or change_object
    )
    lens = {
        "사업 신호": "사업·고객·서비스 실행 움직임",
        "기술 신호": "기술·플랫폼·제품 적용 움직임",
        "리스크": "실행·시장·경쟁·규제상 최근 부담",
    }[label]
    meaning = {
        "사업 신호": "최근 사업 방향이나 고객 접점이 이동하고 있다는 의미",
        "기술 신호": "기술 적용 영역이나 플랫폼화 방향이 드러난다는 의미",
        "리스크": "현재 실행 과정에서 관리해야 할 불확실성이 드러난다는 의미",
    }[label]
    coverage_note = build_overall_reasoning_coverage_note(evidence_pack, refs)
    return compact_text(
        f"추론 과정: '{checked}' 내용을 확인했고, 이는 {meaning}입니다. {coverage_note}그래서 '{change_object}' 항목을 {lens}으로 판단했습니다.",
        280,
    )


def build_overall_reasoning_coverage_note(
    evidence_pack: dict[str, Any] | None,
    refs: list[str],
) -> str:
    if not evidence_pack or evidence_pack.get("peer", {}).get("id") != "all":
        return ""
    ref_set = set(refs)
    covered_peer_names: list[str] = []
    for company in evidence_pack.get("companies") or []:
        peer = company.get("peer") if isinstance(company, dict) else None
        peer_name = peer.get("name") if isinstance(peer, dict) else None
        company_refs = {
            str(item.get("evidence_id"))
            for item in company.get("comparison_signals") or []
            if isinstance(item, dict)
        }
        if ref_set & company_refs and peer_name:
            covered_peer_names.append(str(peer_name))
    if len(covered_peer_names) >= 2:
        return "여러 기업 근거가 같은 방향을 가리켜 국내 IT서비스 시장의 공통 흐름으로 보았습니다. "
    if len(covered_peer_names) == 1:
        return "다만 직접 근거가 일부 기업에 집중되어 시장 전체 판단에는 제한이 있습니다. "
    return "시장 공통 흐름으로 볼 직접 근거는 제한적입니다. "


def build_swot_evidence_summary(diagnosis: dict[str, Any]) -> str:
    citations = diagnosis.get("source_citations") if isinstance(diagnosis, dict) else None
    if isinstance(citations, list) and citations:
        return compact_text("출처: " + " | ".join(str(citation) for citation in citations[:2]), 280)
    return "출처: 비교 포인트와 독립적으로 사용할 수 있는 공개 원문 출처가 부족합니다."


def build_swot_reasoning_summary(label: str, title: str, diagnosis: dict[str, Any]) -> str:
    question = compact_text(
        diagnosis.get("diagnostic_question") if isinstance(diagnosis, dict) else "", 90
    )
    basis = compact_text(
        diagnosis.get("judgment_basis") if isinstance(diagnosis, dict) else "", 110
    )
    basis_points = diagnosis.get("basis_points") if isinstance(diagnosis, dict) else None
    checked = (
        compact_text(str(basis_points[0]), 120)
        if isinstance(basis_points, list) and basis_points
        else title
    )
    label_lens = {
        "Strength": "내부 통제 가능한 경쟁 역량",
        "Weakness": "내부에서 개선해야 할 제약",
        "Opportunity": "외부에서 유리하게 열린 조건",
        "Threat": "외부에서 불리하게 작용할 압박",
    }[label]
    meaning = basis or label_lens
    return compact_text(
        f"추론 과정: '{checked}' 내용을 확인했고, 이는 {meaning} 기준에 해당합니다. 그래서 '{question or title}' 관점에서 {label_lens}으로 판단했습니다.",
        300,
    )


def build_signal_source_citation(item: dict[str, Any]) -> str:
    source_name = human_source_name(item.get("source_name") or item.get("source_type"))
    title = compact_text(item.get("title"), 90) or "제목 미확인 원문"
    date_value = compact_date(item.get("date") or item.get("updated_at"))
    url = compact_text(item.get("url"), 120)
    parts = [source_name, f"'{title}'"]
    if date_value:
        parts.append(date_value)
    citation = " ".join(part for part in parts if part)
    if url:
        citation = f"{citation}, {url}"
    return compact_text(citation, 240)


def build_metric_source_citation(item: dict[str, Any]) -> str:
    source_name = human_source_name(item.get("source_name") or "공시/IR 원문")
    title = compact_text(item.get("title"), 90) or "재무 지표 원문"
    date_value = compact_date(item.get("date") or item.get("updated_at"))
    metric = compact_text(item.get("metric_label") or item.get("metric_name"), 50)
    period = compact_text(item.get("period"), 20)
    url = compact_text(item.get("url"), 120)
    detail = " ".join(value for value in (period, metric) if value)
    citation = " ".join(part for part in (source_name, f"'{title}'", date_value, detail) if part)
    if url:
        citation = f"{citation}, {url}"
    return compact_text(citation, 240)


def human_source_name(value: Any) -> str:
    raw = str(value or "").strip()
    aliases = {
        "dart": "DART 공시",
        "ir": "IR 자료",
        "securities_report": "증권사 리포트",
        "naver_research": "네이버 증권 리서치",
        "공시/IR 원문": "공시/IR 원문",
    }
    return aliases.get(raw, raw or "공개 원문")


def compact_date(value: Any) -> str:
    text_value = str(value or "").strip()
    if not text_value:
        return ""
    match = re.search(r"\d{4}-\d{2}-\d{2}", text_value)
    return match.group(0) if match else compact_text(text_value, 20)


def separate_swot_body_from_comparison(
    label: str,
    body: str,
    title: str,
    evidence_pack: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> str:
    comparison_items = evidence_pack.get("_normalized_comparison_points") or []
    if not comparison_items:
        return body
    comparison_texts = [
        " ".join(str(item.get(key) or "") for key in ("label", "change_object", "body"))
        for item in comparison_items
        if isinstance(item, dict)
    ]
    candidate = f"{title} {body}"
    if not any(is_semantically_close(candidate, text_value) for text_value in comparison_texts):
        return body
    diagnosis = diagnostic_for_label(label, diagnostics)
    return build_diagnostic_swot_body(label, title, diagnosis)


def build_diagnostic_swot_body(label: str, title: str, diagnosis: dict[str, Any]) -> str:
    summaries = diagnosis.get("signal_summaries") if isinstance(diagnosis, dict) else None
    basis_points = diagnosis.get("basis_points") if isinstance(diagnosis, dict) else None
    profile_hint = compact_text(
        diagnosis.get("profile_hint") if isinstance(diagnosis, dict) else "", 120
    )
    checked = ""
    if isinstance(summaries, list) and summaries:
        checked = compact_text(str(summaries[0]), 130)
    elif isinstance(basis_points, list) and basis_points:
        checked = compact_text(str(basis_points[0]), 130)
    else:
        checked = profile_hint or title

    object_name = extract_specific_phrase(
        " ".join(str(item) for item in (summaries or [])) or profile_hint or title
    )
    if is_generic_change_object(object_name):
        object_name = title

    bodies = {
        "Strength": (
            f"근거에서는 {checked} 내용이 확인됩니다. "
            f"{object_name} 역량은 서비스·플랫폼 운영에 반복 적용할 수 있는 내부 자산이므로 강점으로 판단합니다."
        ),
        "Weakness": (
            f"근거에서는 {checked} 내용이 확인됩니다. "
            f"{object_name} 이슈는 비용·일정·인력 또는 수익성 측면에서 내부 관리가 필요한 제약이므로 약점으로 판단합니다."
        ),
        "Opportunity": (
            f"근거에서는 {checked} 내용이 확인됩니다. "
            f"{object_name} 흐름은 회사가 직접 만든 변화는 아니지만 시장·고객 투자 확대를 활용할 수 있는 기회로 판단합니다."
        ),
        "Threat": (
            f"근거에서는 {checked} 내용이 확인됩니다. "
            f"{object_name} 압박은 경쟁·규제·수요 변화처럼 직접 통제하기 어려운 외부 조건이므로 위협으로 판단합니다."
        ),
    }
    return compact_text(bodies[label], 430)


def swot_text_overlaps_comparison(value: str, evidence_pack: dict[str, Any]) -> bool:
    comparison_items = evidence_pack.get("_normalized_comparison_points") or []
    comparison_texts = [
        " ".join(str(item.get(key) or "") for key in ("change_object", "body"))
        for item in comparison_items
        if isinstance(item, dict)
    ]
    return content_overlaps_comparison(value, comparison_texts)


def build_analysis_trace(
    comparison_points: list[dict[str, Any]],
    swot_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for item in comparison_points:
        label = str(item.get("label") or "비교 포인트")
        refs = [str(ref) for ref in item.get("evidence_refs") or []]
        reasoning = compact_text(str(item.get("reasoning_summary") or ""), 300)
        evidence = compact_text(str(item.get("evidence_summary") or ""), 280)
        trace.append(
            {
                "step": f"{label} 판단",
                "summary": reasoning,
                "reasoning": reasoning,
                "evidence": evidence,
                "evidence_refs": refs,
            }
        )
    for item in swot_items:
        label = str(item.get("label") or "SWOT")
        refs = [str(ref) for ref in item.get("evidence_refs") or []]
        reasoning = compact_text(str(item.get("reasoning_summary") or ""), 320)
        evidence = compact_text(str(item.get("evidence_summary") or ""), 300)
        trace.append(
            {
                "step": f"{label} 판단",
                "summary": reasoning,
                "reasoning": reasoning,
                "evidence": evidence,
                "evidence_refs": refs,
            }
        )
    return trace


def build_overall_check_point(
    comparison_points: list[dict[str, Any]],
    swot_items: list[dict[str, Any]],
) -> str:
    objects = [item.get("change_object") for item in comparison_points if item.get("change_object")]
    titles = [item.get("title") for item in swot_items if item.get("title")]
    values = [str(value) for value in [*objects, *titles] if value]
    return (
        compact_text(" / ".join(values[:4]), 180)
        or "다음 분기에도 사업, 기술, 리스크, SWOT 진단 근거를 분리해 확인합니다."
    )


def save_results_to_db(
    evidence_packs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    model: str,
) -> int:
    pack_by_peer_id = {str(pack.get("peer", {}).get("id")): pack for pack in evidence_packs}
    saved_count = 0
    with SessionLocal() as db:
        for result in results:
            peer_id = str(result.get("peer_id") or "")
            evidence_pack = pack_by_peer_id.get(peer_id)
            if not peer_id or evidence_pack is None:
                log.warning("DB 저장 건너뜀 | peer_id=%s evidence_pack 없음", peer_id)
                continue
            save_result_to_db(db, evidence_pack, result, model=model)
            saved_count += 1
        db.commit()
    log.info("LLM 분석 스냅샷 DB 저장 완료 | count=%s", saved_count)
    return saved_count


def save_result_to_db(
    db: Any, evidence_pack: dict[str, Any], result: dict[str, Any], *, model: str
) -> None:
    peer = evidence_pack["peer"]
    peer_id = str(peer["id"])
    comparison_mode = str(
        evidence_pack.get("comparison_mode") or result.get("comparison_mode") or "peer_vs_sk_ax"
    )
    scope = "all" if peer_id == "all" else "company"
    evidence_refs = collect_evidence_ids_from_result(result)
    source_signal_ids = sorted(
        extract_numeric_ids(evidence_refs, "signal:") | collect_signal_ids(evidence_pack)
    )
    source_metric_ids = sorted(
        extract_numeric_ids(evidence_refs, "metric:") | collect_metric_ids(evidence_pack)
    )
    source_raw_article_ids = sorted(collect_raw_article_ids(evidence_pack))
    peer_ids = collect_peer_ids(evidence_pack)
    confidence = average_confidence(result)
    analysis_trace = (
        result.get("analysis_trace") if isinstance(result.get("analysis_trace"), list) else []
    )
    params = {
        "scope": scope,
        "peer_id": peer_id,
        "reference_peer_id": SK_AX_ID,
        "comparison_mode": comparison_mode,
        "prompt_version": result.get("prompt_version") or PROMPT_VERSION,
        "model_name": result.get("model_name") or model,
        "evidence_hash": result.get("evidence_hash") or evidence_pack.get("evidence_hash"),
        "input_snapshot": json.dumps(evidence_pack, ensure_ascii=False, default=str),
        "output_payload": json.dumps(result, ensure_ascii=False, default=str),
        "analysis_trace": json.dumps(analysis_trace, ensure_ascii=False, default=str),
        "provenance": json.dumps(build_provenance(result), ensure_ascii=False, default=str),
        "confidence": confidence,
        "source_raw_article_ids": source_raw_article_ids,
        "source_signal_ids": source_signal_ids,
        "source_metric_ids": source_metric_ids,
        "peer_ids": peer_ids,
    }

    update_result = db.execute(
        text(
            """
            UPDATE peer_llm_analysis_snapshots
            SET
                scope = :scope,
                reference_peer_id = :reference_peer_id,
                schema_version = 'peer_swot_comparison_v1',
                status = 'active',
                evidence_hash = :evidence_hash,
                input_snapshot = CAST(:input_snapshot AS jsonb),
                output_payload = CAST(:output_payload AS jsonb),
                analysis_trace = CAST(:analysis_trace AS jsonb),
                provenance = CAST(:provenance AS jsonb),
                confidence = :confidence,
                source_raw_article_ids = :source_raw_article_ids,
                source_signal_ids = :source_signal_ids,
                source_metric_ids = :source_metric_ids,
                peer_ids = :peer_ids,
                generated_at = NOW(),
                updated_at = NOW()
            WHERE analysis_type = 'peer_swot_comparison'
              AND peer_id = :peer_id
              AND comparison_mode = :comparison_mode
              AND prompt_version = :prompt_version
              AND COALESCE(model_name, '') = COALESCE(:model_name, '')
            """
        ),
        params,
    )
    if update_result.rowcount and update_result.rowcount > 0:
        return

    db.execute(
        text(
            """
            INSERT INTO peer_llm_analysis_snapshots (
                analysis_type,
                scope,
                peer_id,
                reference_peer_id,
                comparison_mode,
                schema_version,
                prompt_version,
                model_name,
                status,
                evidence_hash,
                input_snapshot,
                output_payload,
                analysis_trace,
                provenance,
                confidence,
                source_raw_article_ids,
                source_signal_ids,
                source_metric_ids,
                peer_ids,
                generated_at
            )
            VALUES (
                'peer_swot_comparison',
                :scope,
                :peer_id,
                :reference_peer_id,
                :comparison_mode,
                'peer_swot_comparison_v1',
                :prompt_version,
                :model_name,
                'active',
                :evidence_hash,
                CAST(:input_snapshot AS jsonb),
                CAST(:output_payload AS jsonb),
                CAST(:analysis_trace AS jsonb),
                CAST(:provenance AS jsonb),
                :confidence,
                :source_raw_article_ids,
                :source_signal_ids,
                :source_metric_ids,
                :peer_ids,
                NOW()
            )
            ON CONFLICT (
                analysis_type,
                peer_id,
                comparison_mode,
                evidence_hash,
                prompt_version,
                (COALESCE(model_name, ''))
            )
            DO UPDATE SET
                scope = EXCLUDED.scope,
                reference_peer_id = EXCLUDED.reference_peer_id,
                schema_version = EXCLUDED.schema_version,
                status = 'active',
                input_snapshot = EXCLUDED.input_snapshot,
                output_payload = EXCLUDED.output_payload,
                analysis_trace = EXCLUDED.analysis_trace,
                provenance = EXCLUDED.provenance,
                confidence = EXCLUDED.confidence,
                source_raw_article_ids = EXCLUDED.source_raw_article_ids,
                source_signal_ids = EXCLUDED.source_signal_ids,
                source_metric_ids = EXCLUDED.source_metric_ids,
                peer_ids = EXCLUDED.peer_ids,
                generated_at = NOW(),
                updated_at = NOW()
            """
        ),
        params,
    )


def collect_evidence_ids_from_result(result: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for section_name in ("comparison_points", "swot", "analysis_trace"):
        for item in result.get(section_name) or []:
            if not isinstance(item, dict):
                continue
            for ref in item.get("evidence_refs") or []:
                if isinstance(ref, str) and ref.strip():
                    refs.add(ref.strip())
    return refs


def collect_signal_ids(evidence_pack: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for item in iter_evidence_items(evidence_pack):
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id.startswith("signal:"):
            raw_id = evidence_id.removeprefix("signal:")
            if raw_id.isdigit():
                ids.add(int(raw_id))
        for ref in item.get("source_signal_refs") or []:
            if (
                isinstance(ref, str)
                and ref.startswith("signal:")
                and ref.removeprefix("signal:").isdigit()
            ):
                ids.add(int(ref.removeprefix("signal:")))
    return ids


def collect_metric_ids(evidence_pack: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for item in iter_evidence_items(evidence_pack):
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id.startswith("metric:"):
            raw_id = evidence_id.removeprefix("metric:")
            if raw_id.isdigit():
                ids.add(int(raw_id))
        for ref in item.get("source_metric_refs") or []:
            if (
                isinstance(ref, str)
                and ref.startswith("metric:")
                and ref.removeprefix("metric:").isdigit()
            ):
                ids.add(int(ref.removeprefix("metric:")))
    return ids


def collect_raw_article_ids(evidence_pack: dict[str, Any]) -> set[int]:
    article_ids: set[int] = set()
    for item in iter_evidence_items(evidence_pack):
        article_id = item.get("article_id")
        if isinstance(article_id, int):
            article_ids.add(article_id)
        elif isinstance(article_id, str) and article_id.isdigit():
            article_ids.add(int(article_id))
    return article_ids


def collect_peer_ids(evidence_pack: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for company in evidence_pack.get("companies") or []:
        peer = company.get("peer") if isinstance(company, dict) else None
        peer_id = peer.get("id") if isinstance(peer, dict) else None
        if isinstance(peer_id, str) and peer_id not in ids:
            ids.append(peer_id)
    peer_id = evidence_pack.get("peer", {}).get("id")
    if isinstance(peer_id, str) and peer_id != "all" and peer_id not in ids:
        ids.append(peer_id)
    return ids


def iter_evidence_items(evidence_pack: dict[str, Any]):
    for company in evidence_pack.get("companies") or []:
        if not isinstance(company, dict):
            continue
        for key in (
            "comparison_signals",
            "swot_source_signals",
            "financial_metrics",
            "diagnostic_evidence",
        ):
            for item in company.get(key) or []:
                if isinstance(item, dict):
                    yield item
    for key in ("comparison_input", "swot_input"):
        section = evidence_pack.get(key)
        if not isinstance(section, dict):
            continue
        for nested_key in ("signals", "diagnostic_evidence"):
            for item in section.get(nested_key) or []:
                if isinstance(item, dict):
                    yield item


def build_provenance(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "generator": "src/services/peer_swot_llm_preview.py",
        "prompt_version": result.get("prompt_version") or PROMPT_VERSION,
        "model_name": result.get("model_name"),
        "evidence_hash": result.get("evidence_hash"),
    }


def average_confidence(result: dict[str, Any]) -> float | None:
    values: list[float] = []
    for section_name in ("comparison_points", "swot"):
        for item in result.get(section_name) or []:
            value = item.get("confidence") if isinstance(item, dict) else None
            numeric = to_float(value)
            if numeric is not None:
                values.append(numeric)
    if not values:
        return None
    return sum(values) / len(values)


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


def rough_token_count(text_value: str) -> int:
    return max(1, int(len(text_value) / 2.2))


def print_or_write_json(payload: dict[str, Any], output: str | None) -> None:
    text_value = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_value + "\n", encoding="utf-8")
        log.info("JSON written: %s", path)
    else:
        print(text_value)


def parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def canonical_comparison_label(value: Any) -> str | None:
    label = str(value or "").strip()
    aliases = {
        "사업 변화": "사업 신호",
        "비즈니스 신호": "사업 신호",
        "기술 변화": "기술 신호",
        "테크 신호": "기술 신호",
        "리스크 변화": "리스크",
        "위험": "리스크",
    }
    label = aliases.get(label, label)
    return label if label in COMPARISON_LABELS else None


def canonical_swot_label(value: Any) -> str | None:
    label = str(value or "").strip()
    aliases = {
        "강점": "Strength",
        "Strengths": "Strength",
        "약점": "Weakness",
        "Weaknesses": "Weakness",
        "기회": "Opportunity",
        "Opportunities": "Opportunity",
        "위협": "Threat",
        "Threats": "Threat",
    }
    label = aliases.get(label, label)
    return label if label in SWOT_LABELS else None


def normalize_refs(raw_refs: Any, allowed_refs: set[str]) -> list[str]:
    if not isinstance(raw_refs, list):
        return []
    refs: list[str] = []
    for ref in raw_refs:
        if not isinstance(ref, str):
            continue
        ref = ref.strip()
        if ref in allowed_refs and ref not in refs:
            refs.append(ref)
    return refs[:3]


def choose_signal_refs_for_label(label: str, signals: list[dict[str, Any]]) -> list[str]:
    if label == "기술 신호":
        selected = [s for s in signals if str(s.get("signal_type")) == "rd" or has_tech_term(s)]
    elif label == "리스크":
        selected = [s for s in signals if str(s.get("signal_type")) == "risk" or has_risk_term(s)]
    else:
        selected = [s for s in signals if str(s.get("signal_type")) not in {"rd", "risk"}]
    if not selected:
        selected = signals
    return [str(item["evidence_id"]) for item in selected[:2] if item.get("evidence_id")]


def choose_diagnostic_refs_for_label(label: str, diagnostics: list[dict[str, Any]]) -> list[str]:
    selected = [item for item in diagnostics if item.get("label") == label]
    return [str(item["evidence_id"]) for item in selected[:1] if item.get("evidence_id")]


def ref_matches_label(ref: str, label: str) -> bool:
    lower = ref.lower()
    label_slug = {
        "Strength": ":strength:",
        "Weakness": ":weakness:",
        "Opportunity": ":opportunity:",
        "Threat": ":threat:",
    }[label]
    return label_slug in lower


def collect_urls_for_refs(refs: list[str], signals: list[dict[str, Any]]) -> list[str]:
    by_ref = {item.get("evidence_id"): item for item in signals}
    urls: list[str] = []
    for ref in refs:
        url = by_ref.get(ref, {}).get("url")
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)
    return urls[:3]


def infer_change_object(label: str, refs: list[str], signals: list[dict[str, Any]]) -> str:
    by_ref = {item.get("evidence_id"): item for item in signals}
    candidates = [by_ref[ref] for ref in refs if ref in by_ref] or signals
    for item in candidates:
        for key in ("business_area", "title", "summary"):
            value = compact_text(item.get(key), 80)
            extracted = extract_specific_phrase(value)
            if extracted:
                return extracted
    defaults = {
        "사업 신호": GENERIC_CHANGE_OBJECT_BY_LABEL["사업 신호"],
        "기술 신호": GENERIC_CHANGE_OBJECT_BY_LABEL["기술 신호"],
        "리스크": GENERIC_CHANGE_OBJECT_BY_LABEL["리스크"],
    }
    return defaults[label]


def extract_specific_phrase(text_value: str | None) -> str:
    text_value = compact_text(text_value, 90)
    if not text_value:
        return ""
    quoted = re.findall(
        r"[A-Za-z][A-Za-z0-9+.#/-]{2,}|[가-힣A-Za-z0-9+.#/-]{2,}(?:AI|AX|DX|클라우드|플랫폼|서비스|솔루션|팩토리|자동화)",
        text_value,
    )
    if quoted:
        return compact_text(quoted[0], 60)
    parts = re.split(r"[,/·\s]+", text_value)
    for part in parts:
        part = part.strip()
        if len(part) >= 3 and not part.isdigit():
            return part[:60]
    return text_value[:60]


def is_generic_change_object(value: str) -> bool:
    normalized = re.sub(r"\s+", "", str(value or "").lower())
    generic_values = {
        "",
        "사업",
        "사업활동",
        "비즈니스",
        "서비스",
        "플랫폼",
        "솔루션",
        "기술",
        "고객",
        "시장",
        "리스크",
        "위험",
        "실행리스크",
        "시장리스크",
    }
    return normalized in {re.sub(r"\s+", "", item.lower()) for item in generic_values}


def is_semantically_close(left: str, right: str) -> bool:
    left_norm = normalize_similarity_text(left)
    right_norm = normalize_similarity_text(right)
    if not left_norm or not right_norm:
        return False
    ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    if ratio >= 0.58:
        return True
    left_tokens = set(extract_similarity_tokens(left_norm))
    right_tokens = set(extract_similarity_tokens(right_norm))
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.62


def normalize_similarity_text(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣]+", " ", str(value or "").lower())
    stopwords = {
        "관련",
        "최근",
        "확인",
        "확인됩니다",
        "움직임",
        "신호",
        "기준",
        "판단",
        "요인",
        "수준",
        "가능",
        "필요",
        "있습니다",
        "합니다",
    }
    tokens = [token for token in value.split() if token not in stopwords and len(token) > 1]
    return " ".join(tokens)


def extract_similarity_tokens(value: str) -> list[str]:
    return [token for token in value.split() if len(token) >= 2]


def fallback_swot_title(label: str, diagnostics: list[dict[str, Any]]) -> str:
    diagnostic = next((item for item in diagnostics if item.get("label") == label), None)
    summaries = diagnostic.get("signal_summaries") if isinstance(diagnostic, dict) else []
    phrase = ""
    if summaries:
        phrase = extract_specific_phrase(str(summaries[0]))
    defaults = {
        "Strength": GENERIC_SWOT_TITLE_BY_LABEL["Strength"],
        "Weakness": GENERIC_SWOT_TITLE_BY_LABEL["Weakness"],
        "Opportunity": GENERIC_SWOT_TITLE_BY_LABEL["Opportunity"],
        "Threat": GENERIC_SWOT_TITLE_BY_LABEL["Threat"],
    }
    if phrase:
        return f"{phrase} 기반 {defaults[label]}"
    return defaults[label]


def fallback_check_point(label: str) -> str:
    return {
        "Strength": "동일 역량이 다른 고객·산업으로 반복 확장되는지 확인",
        "Weakness": "비용, 인력, 일정 부담이 실적과 납품 품질에 미치는 영향 확인",
        "Opportunity": "외부 수요와 고객 투자 흐름이 실제 계약으로 전환되는지 확인",
        "Threat": "경쟁, 규제, 수요 둔화가 사업 속도를 낮추는지 확인",
    }[label]


def has_tech_term(item: dict[str, Any]) -> bool:
    text_value = joined_item_text(item).lower()
    return any(
        term in text_value
        for term in (
            "ai",
            "ax",
            "dx",
            "cloud",
            "클라우드",
            "플랫폼",
            "데이터",
            "보안",
            "자동화",
            "llm",
            "생성형",
        )
    )


def has_risk_term(item: dict[str, Any]) -> bool:
    text_value = joined_item_text(item).lower()
    return any(
        term in text_value
        for term in (
            "risk",
            "리스크",
            "위험",
            "부담",
            "지연",
            "둔화",
            "경쟁",
            "규제",
            "수익성",
            "비용",
            "하락",
            "감소",
        )
    )


def has_negative_metric_signal(metric: dict[str, Any]) -> bool:
    text_value = joined_item_text(metric).lower()
    value = metric.get("value_numeric")
    return (
        any(
            term in text_value
            for term in ("감소", "하락", "둔화", "적자", "손실", "마진", "수익성")
        )
        or bool(re.search(r"(?i)(yoy|qoq|전년|전분기).{0,20}-\d", text_value))
        or (isinstance(value, int | float) and value < 0)
    )


def joined_item_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "business_area",
            "signal_type",
            "summary",
            "evidence_text",
            "title",
            "metric_name",
            "metric_label",
        )
    )


def summarize_profile(profile_snapshot: Any) -> str:
    if profile_snapshot is None:
        return ""
    if isinstance(profile_snapshot, str):
        try:
            profile_snapshot = json.loads(profile_snapshot)
        except json.JSONDecodeError:
            return compact_text(profile_snapshot, 360)
    if isinstance(profile_snapshot, dict):
        parts: list[str] = []
        for key in (
            "summary",
            "businessOverview",
            "business_overview",
            "description",
            "mainBusiness",
        ):
            value = profile_snapshot.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        for key in ("keywords", "businessKeywords", "technologyKeywords"):
            value = profile_snapshot.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value[:8])
        return compact_text(" / ".join(parts), 420)
    return compact_text(str(profile_snapshot), 360)


def clean_display_text(value: Any) -> str:
    text_value = compact_text(value, 1000)
    for phrase in NOISE_PHRASES:
        text_value = text_value.replace(phrase, "")
    text_value = re.sub(r"\s+", " ", text_value).strip()
    return text_value


def is_weak_info_text(value: str) -> bool:
    text_value = clean_display_text(value)
    if len(text_value) < 28:
        return True
    weak_phrases = (
        "근거를 기준으로",
        "보수적으로 작성",
        "확인 가능한",
        "분류했습니다",
        "판단했습니다",
        "short korean",
        "reasoning",
        "evidence",
    )
    if any(phrase in text_value.lower() for phrase in weak_phrases):
        return len(text_value) < 80
    return False


def remove_financial_number_sentences(text_value: str) -> str:
    sentences = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+", text_value)
    kept = [sentence for sentence in sentences if not FINANCIAL_NUMBER_PATTERN.search(sentence)]
    result = " ".join(kept).strip()
    if result:
        return result
    return FINANCIAL_NUMBER_PATTERN.sub("", text_value).strip()


def normalize_peer(peer: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": peer.get("id"),
        "name": peer.get("name"),
        "profile_summary": summarize_profile(peer.get("profile_snapshot")),
    }


def flatten_limited(groups: list[list[dict[str, Any]]], *, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    index = 0
    while len(items) < limit:
        added = False
        for group in groups:
            if index >= len(group):
                continue
            item = group[index]
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                items.append(item)
                added = True
                if len(items) >= limit:
                    break
        if not added:
            break
        index += 1
    return items


def unique_evidence_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        unique_items.append(item)
    return unique_items


def is_usable_signal_text(summary: str, evidence_text: str) -> bool:
    combined = f"{summary} {evidence_text}".strip()
    if len(combined) < 24:
        return False
    if re.search(r"^(그림|표|table|figure)\s*\d*", summary.strip(), re.IGNORECASE):
        return False
    korean_alpha = len(re.findall(r"[가-힣A-Za-z]", combined))
    noisy_chars = len(re.findall(r"[{}<>『』|=]", combined))
    if korean_alpha < 16:
        return False
    if noisy_chars >= 6 and noisy_chars > korean_alpha * 0.15:
        return False
    return True


def extract_numeric_ids(evidence_refs: set[str], prefix: str) -> set[int]:
    ids: set[int] = set()
    for ref in evidence_refs:
        if not ref.startswith(prefix):
            continue
        raw_id = ref.removeprefix(prefix)
        if raw_id.isdigit():
            ids.add(int(raw_id))
    return ids


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]


def compact_text(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    text_value = re.sub(r"\s+", " ", str(value)).strip()
    if len(text_value) <= limit:
        return text_value
    return text_value[: max(0, limit - 1)].rstrip() + "…"


def iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp_confidence(value: Any) -> float:
    numeric = to_float(value)
    if numeric is None:
        return 0.6
    return max(0.0, min(1.0, numeric))


if __name__ == "__main__":
    main()
