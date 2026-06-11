"""Generate Peer+ quarterly overview keywords with an LLM.

The output is stored in ``peer_llm_analysis_snapshots`` with
``analysis_type='peer_overview_keywords'``. The backend reads that snapshot for
the Peer+ overview table and falls back to deterministic SQL only when the LLM
snapshot is missing.

Usage:
    uv run python scripts/generate_peer_overview_keywords.py --env local --dry-run
    ENABLE_OPENAI_CALLS=true uv run python scripts/generate_peer_overview_keywords.py --env local --period 2026Q1 --save-db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from src.config.env_loader import load_profile
from src.config.openai_policy import openai_calls_enabled, openai_disabled_reason
from src.db.postgres import SessionLocal, reconfigure_from_env

log = logging.getLogger("generate_peer_overview_keywords")

PROMPT_VERSION = "peer_overview_keywords_v1"
SCHEMA_VERSION = "peer_overview_keywords_v1"
ANALYSIS_TYPE = "peer_overview_keywords"
COMPARISON_MODE = "quarterly_keyword_selection"
DEFAULT_MODEL = (
    os.getenv("PEER_OVERVIEW_KEYWORD_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
)

TARGET_PEER_IDS = (
    "sk_ax",
    "samsung_sds",
    "lg_cns",
    "hyundai_autoever",
    "posco_dx",
)
FORBIDDEN_LABELS = {
    "ai",
    "ax",
    "dx",
    "ai/dx",
    "ai·dx",
    "ai dx",
    "cloud",
    "ai_ax",
    "company_total",
    "enterprise_it",
    "vehicle_sw",
    "smart_factory",
    "logistics",
    "growth",
    "risk",
    "forecast",
    "strategy",
    "orders_pipeline",
    "investment",
    "valuation",
    "business_overview",
    "product_service",
}

NON_IT_DOMAIN_TERMS = {
    "물류",
    "logistics",
    "운송",
    "배송",
    "창고",
    "풀필먼트",
    "유통",
    "scm",
    "supply chain",
    "supply-chain",
    "제조",
    "철강",
    "배터리",
    "이차전지",
}
PURE_DOMAIN_OPERATION_TERMS = {
    "매출",
    "운임",
    "물동량",
    "운송량",
    "창고 운영",
    "물류센터 운영",
    "센터 운영",
    "국제운송",
    "내륙운송",
    "운송 서비스",
    "배송 서비스",
    "풀필먼트",
    "시장 점유율",
}
SK_AX_SOURCE_NAMES = {
    "sk ax site",
    "sk ax newsroom",
}
SK_AX_ENTITY_TERMS = {
    "sk ax",
    "sk에이엑스",
    "에스케이 ax",
    "sk c&c",
    "sk c & c",
    "sk㈜ c&c",
    "sk주식회사 c&c",
    "에스케이씨앤씨",
    "c&c부문",
    "c&c 부문",
}
SK_AX_BUSINESS_CONTEXT_TERMS = {
    "ax",
    "ai",
    "인공지능",
    "생성형",
    "클라우드",
    "cloud",
    "데이터센터",
    "data center",
    "it 컨설팅",
    "시스템 구축",
    "아웃소싱",
    "outsourcing",
    "agentic",
    "erp",
    "scm",
    "자동화",
    "디지털전환",
    "digital transformation",
    "it서비스",
    "it 서비스",
    "enterprise it",
}
SK_GROUP_NON_SK_AX_TERMS = {
    "sk하이닉스",
    "sk hynix",
    "하이닉스",
    "sk텔레콤",
    "sk telecom",
    "skt",
    "sk스퀘어",
    "sk square",
    "sk이노베이션",
    "sk innovation",
    "sk온",
    "sk on",
    "sk엔무브",
    "sk e&s",
    "sk바이오팜",
    "sk biopharmaceuticals",
    "sk실트론",
    "sk네트웍스",
    "skc",
    "투자부문",
    "계열회사",
    "관계회사",
    "자회사",
    "포트폴리오",
    "약물",
    "신약",
    "바이오",
    "제약",
    "drug discovery",
    "drug design",
    "pharma",
    "biopharma",
}
SK_AX_PROVIDER_ACTION_TERMS = {
    "구축",
    "제공",
    "운영",
    "개발",
    "도입",
    "적용",
    "수주",
    "계약",
    "협력",
    "파트너",
    "서비스",
    "솔루션",
    "플랫폼",
    "컨설팅",
    "outsourcing",
    "managed service",
}
SK_HOLDING_VALUATION_TERMS = {
    "nav",
    "순자산가치",
    "상장 계열사",
    "비상장자회사",
    "지분가치",
    "배당수익",
    "목표주가",
    "상승여력",
    "할인율",
}

IT_RELEVANCE_TERMS = {
    "ai",
    "ax",
    "dx",
    "it",
    "sw",
    "소프트웨어",
    "시스템",
    "플랫폼",
    "saas",
    "클라우드",
    "cloud",
    "데이터",
    "자동화",
    "최적화",
    "관제",
    "솔루션",
    "보안",
    "생성형",
    "llm",
    "rag",
    "msp",
    "디지털",
}
PROFILE_STRONG_IT_TERMS = {
    "ai",
    "ax",
    "dx",
    "it",
    "sw",
    "소프트웨어",
    "시스템",
    "플랫폼",
    "saas",
    "클라우드",
    "cloud",
    "데이터",
    "자동화",
    "관제",
    "보안",
    "생성형",
    "llm",
    "rag",
    "msp",
    "디지털",
    "로봇",
}

SYSTEM_PROMPT = """\
너는 Peer+ 분기 키워드 선별기다.
반드시 제공된 evidence pack 안의 정보만 사용한다.
profile_context는 회사의 정적 사업·역량 기준선이다. 최종 키워드는 business_signals의 분기 원문 근거를 우선한다.
business_area, signal_type은 내부 분류값이므로 그대로 최종 키워드로 쓰지 않는다.
사업 키워드는 고객/시장에 제공되는 사업, 서비스, 오퍼링, 수주/확장 방향이다.
기술 키워드는 그 사업을 가능하게 하는 제품, 플랫폼, 기술, 자동화/AI/클라우드 구현 축이다.
Peer+는 SK AX가 봐야 할 IT 서비스/AX/클라우드/AI/데이터/보안/운영 자동화 경쟁 신호만 선별한다.
물류, 제조, 철강, 배터리 같은 단어 자체를 금지하지 않는다.
다만 해당 내용이 운임, 물동량, 매출, 창고 운영 같은 산업 본업 성과라면 제외하고, 플랫폼, SaaS, AI, 클라우드, 데이터, 보안, 자동화, 시스템 구축/운영 역량으로 명확히 표현된 경우만 포함한다.
근거가 부족하면 null을 반환한다.
AI, AX, DX 단독 표현은 금지한다.
growth, risk, forecast 같은 signal_type 이벤트 분류값은 금지한다.
company_total, cloud, ai_ax 같은 내부 business_area 값 그대로 사용은 금지한다.
증권사 주가 전망이나 투자의견 문장만 근거로 선택하지 않는다.
SK AX는 SK 주식회사 전체/비IT 계열 사업 문맥을 근거로 삼지 않는다.
SK AX의 경우 SK그룹 계열사나 투자 포트폴리오의 관심사를 SK AX 관심사로 간주하지 않는다.
예를 들어 약물 설계 플랫폼, 바이오/신약, 반도체, 통신, 배터리처럼 SK바이오팜·SK하이닉스·SK텔레콤·SK온 등 특정 계열사 문맥이면 제외한다.
원문 문장을 그대로 복사하지 말고 한국어로 자연스럽게 바꿔 설명한다.
출력은 반드시 지정된 JSON 형식만 따른다.
"""

USER_PROMPT_TEMPLATE = """\
아래 evidence pack을 바탕으로 {peer_name}의 Peer+ 한눈 비교 표에 넣을 분기 키워드를 생성해줘.

분석 대상:
- peer_id: {peer_id}
- peer_name: {peer_name}
- period: {period}

목표:
- 사업 키워드 1개와 기술 키워드 1개를 선택한다.
- 각 키워드는 반드시 evidence_refs로 근거 signal을 1개 이상 연결한다.
- business_area/signal_type 내부 태그를 그대로 쓰지 말고, summary/evidence_text/title에 담긴 실제 사업명, 서비스명, 제품명, 기술명으로 정규화한다.
- profile_context는 회사의 기본 사업축과 역량 기준선을 이해하는 보조 정보로만 사용한다.
- profile_context만으로 키워드를 만들지 말고 business_signals의 evidence_refs를 연결한다.
- profile_context에 물류/제조/철강/배터리 같은 도메인 항목이 있더라도, IT 서비스/AI/클라우드/플랫폼/자동화/시스템 역량으로 연결되는 부분만 선택한다.
- Peer가 하는 모든 사업을 요약하지 말고, SK AX의 IT 서비스/AX 사업 판단에 직접 관련되는 신호만 선택한다.
- 물류/운송/창고/유통/제조/철강/배터리 같은 단어 자체를 금지하지 않는다.
- 단, 해당 내용이 운임, 물동량, 매출, 창고 운영 같은 산업 본업 성과라면 제외하고, 플랫폼/SaaS/AI/클라우드/데이터/보안/자동화/시스템 구축·운영 역량으로 명확히 연결된 경우만 포함한다.
- 근거가 부족하면 해당 keyword 객체의 label을 null로 둔다.
- info 팝오버에 들어갈 top_keyword_evidence는 추론 과정이 아니라 각 키워드의 판단 근거 설명문으로 작성한다.
- top_keyword_evidence에는 원문 날짜, 원문에서 확인한 실제 내용, 그 내용을 보아 키워드로 선정한 이유가 모두 들어가야 한다.
- "원문에서 어떤 사업명/기술명/서비스명/제품명과 관련한 진행 방향이 확인되었고, 이를 보아 왜 이 키워드로 판단했는지"가 구체적으로 드러나야 한다.
- 판단 근거는 "원문 날짜와 내용 → 이 내용을 보아 키워드로 판단한 이유" 순서로 쓴다.

사업 키워드 기준:
- 고객에게 제공하는 사업, 서비스, 오퍼링, 수주/확장 활동, 고객 산업을 나타낸다.
- 예: AI 인프라, 클라우드 MSP, SCM 플랫폼, 차량 SW 플랫폼, 로봇 자동화.

기술 키워드 기준:
- 사업을 가능하게 하는 제품, 플랫폼, 기술, 구현 역량을 나타낸다.
- 예: FabriX, Brity Copilot, GPUaaS, 생성형 AI, OTA, 로봇 관제.

금지:
- AI, AX, DX 단독 표현
- company_total, cloud, ai_ax, enterprise_it 같은 내부 business_area 값 그대로 사용
- growth, risk, forecast, strategy 같은 signal_type 값 그대로 사용
- 근거에 없는 브랜드/제품명 추정
- 물류, 운송, 창고, 유통, 제조, 철강, 배터리처럼 IT 구현과 분리된 산업 본업 키워드
- 증권사 투자의견/주가 전망만으로 키워드 선택
- peer_id가 sk_ax일 때 SK그룹 계열사, 투자 포트폴리오, 약물 설계/바이오/신약/반도체/통신/배터리 등 비-SK AX 문맥을 SK AX 사업·기술 키워드로 선택

출력 JSON:
{{
  "peer_id": "{peer_id}",
  "peer_name": "{peer_name}",
  "period": "{period}",
  "prompt_version": "{prompt_version}",
  "business_keyword": {{
    "label": "키워드 또는 null",
    "reason": "어떤 사업 활동을 진행한다는 근거 때문에 이 키워드를 선택했는지 1~2문장",
    "reasoning": "확인된 활동을 왜 사업 활동/고객 산업/수주·확장 축으로 해석했는지 1문장",
    "confidence": 0.0,
    "evidence_refs": ["signal:1"],
    "source_urls": ["https://example.com"],
    "evidence_summary": "구체적인 진행 내용과 원문 근거 요약 1문장"
  }},
  "technology_keyword": {{
    "label": "키워드 또는 null",
    "reason": "이 키워드를 선택한 판단 요약 1~2문장",
    "reasoning": "확인된 활동을 왜 기술 구현/제품/플랫폼 축으로 해석했는지 1문장",
    "confidence": 0.0,
    "evidence_refs": ["signal:2"],
    "source_urls": ["https://example.com"],
    "evidence_summary": "구체적인 진행 내용과 원문 근거 요약 1문장"
  }},
  "top_keyword": "사업 키워드\\n기술 키워드",
  "top_keyword_reason": "분기 원문 기반 사업 신호에서 사업 방향과 기술 구현 축을 분리해 선택했다는 설명",
  "top_keyword_basis": "LLM grounded selection · signals N · articles M · confidence X",
  "top_keyword_evidence": [
    "{peer_name} 사업 키워드 기준: 키워드. 근거 내용: [YYYY-MM-DD] 원문 제목/요약에서 사업명·서비스명·수주·확장 방향과 관련해 확인된 실제 내용. 판단 이유: 이 원문 내용을 보아 왜 이 표현을 사업 키워드로 선정했는지.",
    "{peer_name} 기술 키워드 기준: 키워드. 근거 내용: [YYYY-MM-DD] 원문 제목/요약에서 기술명·제품명·플랫폼명·구현 방향과 관련해 확인된 실제 내용. 판단 이유: 이 원문 내용을 보아 왜 이 표현을 기술 키워드로 선정했는지."
  ],
  "top_keyword_evidence_urls": ["https://example.com"],
  "rejected_candidates": [
    {{"label": "growth", "reason": "signal_type 이벤트 분류값이라 제외"}}
  ],
  "analysis_trace": [
    {{"step": "근거 확인", "summary": "입력 근거에서 확인한 사업/기술 신호", "reasoning": "확인한 신호를 사업/기술 후보로 읽은 방식", "evidence": "입력 근거에서 실제 확인한 내용", "evidence_refs": ["signal:1"]}},
    {{"step": "후보 정제", "summary": "제외한 내부 태그와 부적합 근거", "reasoning": "후보를 제외하거나 남긴 판단 기준", "evidence": "후보 정제에 사용한 근거", "evidence_refs": ["signal:1"]}},
    {{"step": "최종 판단", "summary": "사업 키워드와 기술 키워드를 선택한 이유", "reasoning": "최종 키워드로 압축한 판단 과정", "evidence": "최종 선택에 연결된 핵심 근거", "evidence_refs": ["signal:1", "signal:2"]}}
  ]
}}

Evidence Pack:
{evidence_pack_json}
"""

VALIDATION_RETRY_PROMPT_TEMPLATE = """\
이전 응답은 검증에 실패했다.

검증 실패 사유:
{validation_errors}

다시 작성하라.
- 반드시 같은 JSON 형식만 출력한다.
- business_keyword와 technology_keyword에는 reason, reasoning, evidence_summary를 모두 반드시 포함한다.
- analysis_trace의 모든 항목에는 summary, reasoning, evidence를 모두 반드시 포함한다.
- evidence_refs는 입력 evidence id 안에 있는 값만 사용한다.
- business_area/signal_type 내부 태그를 최종 label로 쓰지 않는다.
- label은 최신 이벤트가 아니라 Peer 한눈 비교 표의 명사형 포지션 태그로 쓴다.
- 확대, 강화, 추진, 성장, 전망, 리스크 같은 흐름/판단 단어를 label에 쓰지 않는다.
- 물류/운송/창고/유통/제조/철강/배터리 같은 산업 본업 키워드는 제외한다.
- 단, 플랫폼/SaaS/AI/클라우드/데이터/보안/자동화/시스템 구축·운영 역량으로 명확히 연결된 경우만 IT 서비스 키워드로 바꿔 쓴다.
- info 문구는 추론 단계명을 나열하지 말고 2~3문장 선정 이유로 작성한다.
- info 문구에는 원문 날짜, 원문에서 확인한 실제 내용, 그 내용을 보아 키워드로 선정한 이유를 포함한다.
- 사업명, 기술명, 서비스명, 제품명, 수주/확장/구축/출시/도입 같은 진행 방향 중 근거에 있는 내용을 포함한다.
- 입력에 없는 사실은 쓰지 않는다.

원래 요청:
{original_prompt}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Peer+ overview keywords.")
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument(
        "--period",
        default=None,
        help="Quarter such as 2026Q1. Defaults to latest 5-peer common signal quarter.",
    )
    parser.add_argument("--company", action="append", choices=TARGET_PEER_IDS, default=None)
    parser.add_argument("--signal-limit", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Build evidence packs only.")
    parser.add_argument(
        "--estimate-tokens",
        action="store_true",
        help="Estimate prompt tokens without calling the LLM.",
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Persist validated results to peer_llm_analysis_snapshots.",
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

    peer_ids = tuple(args.company or TARGET_PEER_IDS)
    with SessionLocal() as db:
        period = args.period or resolve_latest_common_signal_period(db)
        if not period:
            raise RuntimeError("No common business signal period found.")
        packs = build_evidence_packs(db, peer_ids, period=period, signal_limit=args.signal_limit)

    if args.dry_run or args.estimate_tokens:
        if args.save_db:
            raise RuntimeError("--save-db cannot be used with --dry-run or --estimate-tokens")
        if args.estimate_tokens:
            agent = PeerOverviewKeywordAgent(model=args.model)
            estimates = [
                agent.estimate_prompt_tokens(pack) | {"peer_id": pack["peer"]["id"]}
                for pack in packs
            ]
            payload: dict[str, Any] = {
                "mode": "estimate_tokens",
                "period": period,
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "model": args.model,
                "successful_llm_calls": len(packs),
                "estimates": estimates,
                "total_prompt_tokens": sum(item["prompt_tokens"] for item in estimates),
                "note": "Completion tokens are not included. Add roughly 600-1000 output tokens per peer on successful generation.",
            }
        else:
            payload = {
                "mode": "dry_run",
                "period": period,
                "prompt_version": PROMPT_VERSION,
                "evidence_packs": packs,
            }
    else:
        if not openai_calls_enabled():
            raise RuntimeError(openai_disabled_reason())
        agent = PeerOverviewKeywordAgent(model=args.model)
        results = [agent.generate(pack) for pack in packs]
        payload = {
            "mode": "llm_preview",
            "period": period,
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "results": results,
        }
        if args.save_db:
            saved_count = save_results_to_db(packs, results, model=args.model)
            payload["saved_count"] = saved_count

    print_or_write_json(payload, args.output)


def resolve_latest_common_signal_period(db: Any) -> str | None:
    row = (
        db.execute(
            text(
                """
            SELECT period
            FROM raw_article_business_signals
            WHERE peer_id IN ('sk_ax', 'samsung_sds', 'lg_cns', 'hyundai_autoever', 'posco_dx')
              AND period ~ '^[0-9]{4}Q[1-4]$'
            GROUP BY period
            HAVING COUNT(DISTINCT peer_id) = 5
            ORDER BY
                MAX(COALESCE(period_year, NULLIF(SUBSTRING(period FROM '^([0-9]{4})'), '')::int, 0)) DESC,
                MAX(COALESCE(period_quarter, NULLIF(SUBSTRING(period FROM 'Q([1-4])$'), '')::int, 0)) DESC,
                period DESC
            LIMIT 1
            """
            )
        )
        .mappings()
        .first()
    )
    return str(row["period"]) if row else None


def build_evidence_packs(
    db: Any,
    peer_ids: tuple[str, ...],
    *,
    period: str,
    signal_limit: int,
) -> list[dict[str, Any]]:
    peers = fetch_peers(db, peer_ids)
    packs = []
    for peer in peers:
        signals = fetch_business_signals(db, peer["id"], period=period, limit=signal_limit)
        pack = {
            "peer": {"id": peer["id"], "name": peer["name"]},
            "period": period,
            "profile_context": compact_profile_snapshot(peer.get("profile_snapshot")),
            "business_signals": signals,
            "candidate_hints": build_candidate_hints(signals),
        }
        pack["evidence_hash"] = evidence_hash(pack)
        packs.append(pack)
    return packs


def fetch_peers(db: Any, peer_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT id, name, profile_snapshot
            FROM peer_companies
            WHERE id IN ('sk_ax', 'samsung_sds', 'lg_cns', 'hyundai_autoever', 'posco_dx')
            ORDER BY CASE id
                WHEN 'sk_ax' THEN 0
                WHEN 'samsung_sds' THEN 1
                WHEN 'lg_cns' THEN 2
                WHEN 'hyundai_autoever' THEN 3
                WHEN 'posco_dx' THEN 4
                ELSE 99
            END
            """
        )
    ).mappings()
    requested = set(peer_ids)
    return [
        {"id": row["id"], "name": row["name"], "profile_snapshot": row["profile_snapshot"]}
        for row in rows
        if row["id"] in requested
    ]


def fetch_business_signals(
    db: Any, peer_id: str, *, period: str, limit: int
) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            WITH source_rows AS (
                SELECT
                    rabs.id,
                    rabs.raw_article_id,
                    rabs.source_type,
                    rabs.source_name,
                    rabs.business_area,
                    rabs.signal_type,
                    rabs.sentiment,
                    rabs.summary,
                    rabs.evidence_text,
                    rabs.confidence,
                    ra.title,
                    ra.url,
                    COALESCE(ra.published_at, ra.collected_at, ra.created_at, rabs.created_at) AS evidence_at,
                    CASE
                        WHEN rabs.source_type IN ('ir', 'dart') THEN 4
                        WHEN rabs.signal_type IN ('product_service', 'orders_pipeline', 'strategy', 'business_update') THEN 3
                        WHEN rabs.signal_type IN ('growth', 'investment') THEN 2
                        WHEN rabs.signal_type IN ('risk', 'forecast', 'valuation') THEN 0
                        ELSE 1
                    END AS source_priority,
                    CASE
                        WHEN rabs.business_area = 'company_total'
                         AND COALESCE(rabs.summary, rabs.evidence_text, '') !~* '(AI|AX|DX|클라우드|cloud|로봇|소프트웨어|SW|플랫폼|솔루션|자동화|데이터센터|agent|LLM|GPU|FabriX|Brity|OTA|커넥티드)'
                        THEN 1
                        ELSE 0
                    END AS weak_context
                FROM raw_article_business_signals rabs
                JOIN raw_articles ra
                  ON ra.id = rabs.raw_article_id
                WHERE rabs.peer_id = :peer_id
                  AND rabs.period = :period
                  AND COALESCE(rabs.summary, rabs.evidence_text, '') <> ''
                  AND COALESCE(rabs.confidence, 0.5) >= 0.68
            ),
            ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(NULLIF(business_area, ''), 'unknown'), COALESCE(NULLIF(signal_type, ''), 'unknown')
                        ORDER BY source_priority DESC, confidence DESC NULLS LAST, evidence_at DESC NULLS LAST, id DESC
                    ) AS category_rank
                FROM source_rows
                WHERE weak_context = 0
            )
            SELECT *
            FROM ranked
            WHERE category_rank <= 4
            ORDER BY source_priority DESC, confidence DESC NULLS LAST, evidence_at DESC NULLS LAST, id DESC
            LIMIT :limit
            """
        ),
        {"peer_id": peer_id, "period": period, "limit": limit * 4},
    ).mappings()

    signals = []
    seen_claims: set[str] = set()
    for row in rows:
        claim = compact_text(row["summary"], 420)
        evidence_text = compact_text(row["evidence_text"], 520)
        title = compact_text(row["title"], 150)
        source_name = str(row["source_name"] or "")
        if not is_it_relevant_signal(
            row["business_area"],
            claim,
            evidence_text,
            peer_id=peer_id,
            title=title,
            source_name=source_name,
        ):
            continue
        dedup_key = f"{row['business_area']}|{row['signal_type']}|{claim or evidence_text}"
        if dedup_key in seen_claims:
            continue
        seen_claims.add(dedup_key)
        signals.append(
            {
                "evidence_id": f"signal:{row['id']}",
                "signal_id": row["id"],
                "raw_article_id": row["raw_article_id"],
                "title": title,
                "url": row["url"],
                "date": iso_date(row["evidence_at"]),
                "source_type": row["source_type"],
                "source_name": source_name,
                "business_area": row["business_area"],
                "signal_type": row["signal_type"],
                "sentiment": row["sentiment"],
                "summary": claim,
                "evidence_text": evidence_text,
                "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
            }
        )
    return signals[:limit]


def is_it_relevant_signal(
    business_area: Any,
    summary: str,
    evidence_text: str,
    *,
    peer_id: str | None = None,
    title: str = "",
    source_name: str = "",
) -> bool:
    area_text = str(business_area or "").lower()
    combined_text = f"{area_text} {title} {summary} {evidence_text}".lower()
    if peer_id == "sk_ax" and not is_sk_ax_owned_signal(combined_text, source_name=source_name):
        return False
    has_domain = any(term in combined_text for term in NON_IT_DOMAIN_TERMS)
    if has_domain:
        has_it_implementation = any(term in combined_text for term in PROFILE_STRONG_IT_TERMS)
        if not has_it_implementation:
            return False
        has_pure_operation = any(term in combined_text for term in PURE_DOMAIN_OPERATION_TERMS)
        if has_pure_operation and not has_it_implementation:
            return False
    return True


def is_sk_ax_owned_signal(text_value: str, *, source_name: str = "") -> bool:
    lowered_source = " ".join(str(source_name or "").lower().split())
    if lowered_source in SK_AX_SOURCE_NAMES:
        return True

    lowered = " ".join(str(text_value or "").lower().split())
    if not lowered:
        return False

    has_sk_ax_entity = any(term in lowered for term in SK_AX_ENTITY_TERMS)
    has_non_sk_ax_group_context = any(term in lowered for term in SK_GROUP_NON_SK_AX_TERMS)
    has_provider_action = any(term in lowered for term in SK_AX_PROVIDER_ACTION_TERMS)

    if has_non_sk_ax_group_context:
        return has_sk_ax_entity and has_provider_action

    if has_sk_ax_entity:
        return True

    if any(term in lowered for term in SK_HOLDING_VALUATION_TERMS):
        return False

    business_hits = sum(1 for term in SK_AX_BUSINESS_CONTEXT_TERMS if term in lowered)
    has_sk_business_context = (
        "sk주식회사" in lowered
        or "sk 주식회사" in lowered
        or "sk inc" in lowered
        or "sk㈜" in lowered
    )
    return business_hits >= 2 or (has_sk_business_context and business_hits >= 1)


def compact_profile_snapshot(snapshot: Any) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None

    business_areas = [
        compact_profile_business_area(item)
        for item in snapshot.get("business_areas") or []
        if isinstance(item, dict) and is_it_relevant_profile_item(item)
    ]
    business_areas = [item for item in business_areas if item][:4]

    strategic_focus = [
        compact_profile_focus(item)
        for item in snapshot.get("strategic_focus") or []
        if isinstance(item, dict) and is_it_relevant_profile_item(item)
    ]
    strategic_focus = [item for item in strategic_focus if item][:5]

    core_capabilities = [
        compact_text(item, 90)
        for item in snapshot.get("core_capabilities") or []
        if item and is_it_relevant_text(str(item))
    ][:6]
    cautions = [
        compact_text(item, 110)
        for item in snapshot.get("cautions") or []
        if item and is_it_relevant_text(str(item))
    ][:4]

    profile_context = {
        "source": "peer_companies.profile_snapshot",
        "usage": "정적 회사 프로필 기준선이다. 최종 키워드는 business_signals의 분기 원문 근거를 우선한다.",
        "generated_at": snapshot.get("generated_at"),
        "schema_version": snapshot.get("schema_version"),
        "one_liner": compact_profile_scalar(snapshot.get("one_liner"), 120),
        "company_summary": compact_profile_scalar(snapshot.get("company_summary"), 180),
        "business_areas": business_areas,
        "strategic_focus": strategic_focus,
        "core_capabilities": core_capabilities,
        "cautions": cautions,
    }
    return {key: value for key, value in profile_context.items() if value not in (None, "", [], {})}


def compact_profile_business_area(item: dict[str, Any]) -> dict[str, Any]:
    evidence_texts = item.get("evidence_texts") or []
    return {
        "name": compact_text(item.get("name"), 60),
        "summary": compact_text(item.get("summary"), 140),
        "recent_direction": compact_text(item.get("recent_direction"), 140),
        "evidence_texts": [
            compact_text(value.get("text") if isinstance(value, dict) else value, 130)
            for value in evidence_texts[:2]
            if value
        ],
    }


def compact_profile_focus(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "period": item.get("period"),
        "business_area": compact_text(item.get("business_area"), 50),
        "signal_type": compact_text(item.get("signal_type"), 40),
        "summary": compact_text(item.get("summary"), 150),
    }


def is_it_relevant_profile_item(item: dict[str, Any]) -> bool:
    text_value = json.dumps(item, ensure_ascii=False, default=str).lower()
    has_domain = any(term in text_value for term in NON_IT_DOMAIN_TERMS)
    has_it_relevance = any(term in text_value for term in PROFILE_STRONG_IT_TERMS)
    if has_domain:
        return has_it_relevance
    return any(term in text_value for term in IT_RELEVANCE_TERMS)


def is_it_relevant_text(value: str) -> bool:
    text_value = value.lower()
    return any(term in text_value for term in IT_RELEVANCE_TERMS)


def compact_profile_scalar(value: Any, max_chars: int) -> str:
    text_value = compact_text(value, max_chars)
    lowered = text_value.lower()
    if any(term in lowered for term in NON_IT_DOMAIN_TERMS) and not any(
        term in lowered for term in PROFILE_STRONG_IT_TERMS
    ):
        return ""
    return text_value


def build_candidate_hints(signals: list[dict[str, Any]]) -> dict[str, Any]:
    business_area_counts = Counter(str(item.get("business_area") or "") for item in signals)
    signal_type_counts = Counter(str(item.get("signal_type") or "") for item in signals)
    source_counts = Counter(str(item.get("source_type") or "") for item in signals)
    frequent_terms = extract_frequent_terms(signals)
    return {
        "business_area_counts": dict(business_area_counts.most_common(12)),
        "signal_type_counts": dict(signal_type_counts.most_common(12)),
        "source_counts": dict(source_counts.most_common(8)),
        "frequent_terms_from_evidence": frequent_terms,
        "note": "Hints are dynamic terms from this quarter's evidence, not a fixed company dictionary.",
    }


def extract_frequent_terms(signals: list[dict[str, Any]]) -> list[str]:
    text_value = " ".join(
        f"{item.get('title') or ''} {item.get('summary') or ''} {item.get('evidence_text') or ''}"
        for item in signals
    )
    patterns = [
        r"[A-Za-z][A-Za-z0-9+./-]{2,}(?:\s+[A-Za-z][A-Za-z0-9+./-]{1,}){0,2}",
        r"[가-힣A-Za-z0-9]+(?:AI|AX|DX|SW|SaaS|GPUaaS|MSP|센터|플랫폼|솔루션|자동화|로봇|클라우드|데이터센터|팩토리|물류|소프트웨어)[가-힣A-Za-z0-9]*",
    ]
    terms: list[str] = []
    for pattern in patterns:
        terms.extend(match.group(0).strip() for match in re.finditer(pattern, text_value))
    blocked = {item.lower() for item in FORBIDDEN_LABELS}
    cleaned = []
    for term in terms:
        normalized = re.sub(r"\s+", " ", term).strip(".,:;()[]{}·")
        if len(normalized) < 3:
            continue
        if normalized.lower() in blocked:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
            continue
        cleaned.append(normalized)
    counter = Counter(cleaned)
    return [term for term, _ in counter.most_common(18)]


class PeerOverviewKeywordAgent:
    def __init__(self, *, model: str) -> None:
        self.model = model
        self._llm: Any | None = None

    def generate(self, evidence_pack: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(evidence_pack)
        validation_error: ValueError | None = None
        for attempt in range(3):
            request_prompt = prompt
            if attempt > 0 and validation_error is not None:
                request_prompt = VALIDATION_RETRY_PROMPT_TEMPLATE.format(
                    validation_errors=str(validation_error),
                    original_prompt=prompt,
                )
            response = self._get_llm().invoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", request_prompt),
                ]
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            result = parse_json_response(content)
            normalize_result(result, evidence_pack, model=self.model)
            try:
                validate_result(result, evidence_pack)
                return result
            except ValueError as exc:
                validation_error = exc
                log.warning(
                    "LLM keyword validation failed | peer=%s attempt=%s errors=%s",
                    evidence_pack["peer"]["id"],
                    attempt + 1,
                    exc,
                )
        if validation_error is not None:
            raise validation_error
        raise RuntimeError("LLM response validation failed")

    def estimate_prompt_tokens(self, evidence_pack: dict[str, Any]) -> dict[str, int]:
        return {
            "prompt_tokens": estimate_text_tokens(self._build_prompt(evidence_pack), self.model)
        }

    def _build_prompt(self, evidence_pack: dict[str, Any]) -> str:
        peer = evidence_pack["peer"]
        return USER_PROMPT_TEMPLATE.format(
            peer_id=peer["id"],
            peer_name=peer["name"],
            period=evidence_pack["period"],
            prompt_version=PROMPT_VERSION,
            evidence_pack_json=json.dumps(evidence_pack, ensure_ascii=False, indent=2, default=str),
        )

    def _get_llm(self) -> Any:
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.model,
                temperature=0.1,
                max_completion_tokens=2400,
            )
        return self._llm


def normalize_result(result: dict[str, Any], evidence_pack: dict[str, Any], *, model: str) -> None:
    peer = evidence_pack["peer"]
    result["peer_id"] = peer["id"]
    result["peer_name"] = peer["name"]
    result["period"] = evidence_pack["period"]
    result["prompt_version"] = result.get("prompt_version") or PROMPT_VERSION
    result["model_name"] = model
    result["schema_version"] = SCHEMA_VERSION
    result["evidence_hash"] = evidence_pack.get("evidence_hash")

    business = (
        result.get("business_keyword") if isinstance(result.get("business_keyword"), dict) else {}
    )
    technology = (
        result.get("technology_keyword")
        if isinstance(result.get("technology_keyword"), dict)
        else {}
    )
    business_label = clean_label(business.get("label"))
    technology_label = clean_label(technology.get("label"))
    business["label"] = business_label
    technology["label"] = technology_label
    normalize_keyword_reasoning_fields(business)
    normalize_keyword_reasoning_fields(technology)
    result["business_keyword"] = business
    result["technology_keyword"] = technology
    result["top_keyword"] = (
        "\n".join(label for label in (business_label, technology_label) if label) or None
    )
    result.setdefault(
        "top_keyword_reason",
        f"{evidence_pack['period']} 원문 기반 사업 신호에서 사업 방향과 기술 구현 축을 분리해 선택했습니다.",
    )
    result.setdefault("top_keyword_basis", build_basis(result, evidence_pack))
    result["top_keyword_evidence"] = normalize_evidence_lines(result, evidence_pack)
    result["top_keyword_evidence_urls"] = normalize_evidence_urls(result)
    normalize_trace_reasoning_fields(result)


def normalize_keyword_reasoning_fields(item: dict[str, Any]) -> None:
    reason = str(item.get("reason") or "").strip()
    reasoning = str(item.get("reasoning") or "").strip()
    evidence_summary = str(item.get("evidence_summary") or "").strip()
    if not reasoning:
        item["reasoning"] = reason or evidence_summary
    if not evidence_summary:
        item["evidence_summary"] = reason or reasoning


def normalize_trace_reasoning_fields(result: dict[str, Any]) -> None:
    for item in result.get("analysis_trace") or []:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        if not str(item.get("reasoning") or "").strip() and summary:
            item["reasoning"] = summary
        if not str(item.get("evidence") or "").strip() and summary:
            item["evidence"] = summary


def validate_result(result: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    evidence_ids = collect_evidence_ids(evidence_pack)
    errors: list[str] = []
    for key in ("business_keyword", "technology_keyword"):
        item = result.get(key)
        if not isinstance(item, dict):
            errors.append(f"{key} is not an object")
            continue
        label = clean_label(item.get("label"))
        if not label:
            errors.append(f"{key}.label is empty")
        elif label.lower() in FORBIDDEN_LABELS:
            errors.append(f"{key}.label is forbidden: {label}")
        relevance_error = validate_sk_ax_it_relevance(label, item)
        if relevance_error:
            errors.append(f"{key}.{relevance_error}")
        refs = [ref for ref in item.get("evidence_refs") or [] if isinstance(ref, str)]
        if not refs:
            errors.append(f"{key}.evidence_refs is empty")
        if not item.get("reasoning"):
            errors.append(f"{key}.reasoning is empty")
        if not item.get("evidence_summary"):
            errors.append(f"{key}.evidence_summary is empty")
        for ref in refs:
            if ref not in evidence_ids:
                errors.append(f"{key}.evidence_refs has unknown ref: {ref}")
        confidence = safe_float(item.get("confidence"))
        if confidence is not None and not 0 <= confidence <= 1:
            errors.append(f"{key}.confidence is out of range")
    trace_steps = [
        item.get("step") for item in result.get("analysis_trace") or [] if isinstance(item, dict)
    ]
    for expected_step in ("근거 확인", "후보 정제", "최종 판단"):
        if expected_step not in trace_steps:
            errors.append(f"analysis_trace missing step: {expected_step}")
    for item in result.get("analysis_trace") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("reasoning"):
            errors.append(f"analysis_trace/{item.get('step')} reasoning is empty")
        if not item.get("evidence"):
            errors.append(f"analysis_trace/{item.get('step')} evidence is empty")
    evidence_lines = [
        str(line).strip() for line in result.get("top_keyword_evidence") or [] if str(line).strip()
    ]
    if len(evidence_lines) < 2:
        errors.append("top_keyword_evidence must include business and technology explanation lines")
    if errors:
        raise ValueError("; ".join(errors))


def validate_sk_ax_it_relevance(label: str | None, item: dict[str, Any]) -> str | None:
    if not label:
        return None
    text = " ".join(
        str(item.get(key) or "") for key in ("label", "reason", "reasoning", "evidence_summary")
    ).lower()
    has_non_it_domain = any(term in text for term in NON_IT_DOMAIN_TERMS)
    has_it_relevance = any(term in text for term in IT_RELEVANCE_TERMS)
    label_lower = label.lower()

    if any(term in label_lower for term in NON_IT_DOMAIN_TERMS) and not has_it_relevance:
        return f"label uses non-IT domain term as keyword: {label}"
    if has_non_it_domain and not has_it_relevance:
        return f"evidence lacks IT-service qualifier for non-IT domain term: {label}"
    return None


def normalize_evidence_lines(result: dict[str, Any], evidence_pack: dict[str, Any]) -> list[str]:
    lines = [
        str(line).strip() for line in result.get("top_keyword_evidence") or [] if str(line).strip()
    ]
    peer_name = evidence_pack["peer"]["name"]
    generated = []
    for axis_index, (axis_name, key) in enumerate(
        (("사업", "business_keyword"), ("기술", "technology_keyword"))
    ):
        item = result.get(key) if isinstance(result.get(key), dict) else {}
        label = clean_label(item.get("label")) or "-"
        refs = [ref for ref in item.get("evidence_refs") or [] if isinstance(ref, str)]
        evidence_details = [lookup_evidence_detail(evidence_pack, ref) for ref in refs[:2]]
        evidence_details = [detail for detail in evidence_details if detail]
        evidence_texts = [format_evidence_detail(detail) for detail in evidence_details]
        evidence_summary = str(
            item.get("evidence_summary")
            or " / ".join(filter(None, evidence_texts))
            or "근거 요약 없음"
        )
        reason = str(item.get("reason") or "")
        reasoning = str(item.get("reasoning") or "")
        source_basis = compact_text(
            " / ".join(filter(None, evidence_texts)) or evidence_summary, 260
        )
        fallback_reason = (
            reason
            or reasoning
            or f"이 원문 내용을 보아 {label}이 {peer_name}의 {axis_name} 방향을 직접 보여줘 {axis_name} 키워드로 선정했습니다."
        )
        canonical_line = (
            f"{peer_name} {axis_name} 키워드 기준: {label}. "
            f"근거 내용: {source_basis}. "
            f"판단 이유: {compact_text(fallback_reason, 240)}"
        )
        stored_line = lines[axis_index] if axis_index < len(lines) else ""
        if has_required_evidence_detail(stored_line):
            generated.append(stored_line)
        else:
            generated.append(canonical_line)
    return generated


def normalize_evidence_urls(result: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("business_keyword", "technology_keyword"):
        item = result.get(key) if isinstance(result.get(key), dict) else {}
        source_urls = [
            str(url).strip() for url in item.get("source_urls") or [] if str(url).strip()
        ]
        urls.append(source_urls[0] if source_urls else "")
    return urls


def has_required_evidence_detail(line: str) -> bool:
    if not line:
        return False
    return (
        "근거 내용:" in line
        and "판단 이유:" in line
        and bool(re.search(r"\[[0-9]{4}-[0-9]{2}-[0-9]{2}\]", line))
    )


def lookup_evidence_detail(evidence_pack: dict[str, Any], ref: str) -> dict[str, str]:
    for item in evidence_pack.get("business_signals") or []:
        if item.get("evidence_id") == ref:
            return {
                "date": compact_text(item.get("date") or "날짜 미확인", 20),
                "title": compact_text(item.get("title") or "제목 미확인", 120),
                "summary": compact_text(
                    item.get("summary") or item.get("evidence_text") or "원문 요약 없음", 260
                ),
            }
    return {}


def format_evidence_detail(detail: dict[str, str]) -> str:
    date_value = detail.get("date") or "날짜 미확인"
    title = detail.get("title") or "제목 미확인"
    summary = detail.get("summary") or "원문 요약 없음"
    date_prefix = (
        f"[{date_value}]"
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date_value)
        else f"[{date_value}]"
    )
    return compact_text(f"{date_prefix} {title} - {summary}", 320)


def collect_evidence_ids(evidence_pack: dict[str, Any]) -> set[str]:
    return {
        str(item.get("evidence_id"))
        for item in evidence_pack.get("business_signals") or []
        if item.get("evidence_id")
    }


def save_results_to_db(
    evidence_packs: list[dict[str, Any]], results: list[dict[str, Any]], *, model: str
) -> int:
    pack_by_peer_id = {str(pack.get("peer", {}).get("id")): pack for pack in evidence_packs}
    saved_count = 0
    with SessionLocal() as db:
        for result in results:
            peer_id = str(result.get("peer_id") or "")
            evidence_pack = pack_by_peer_id.get(peer_id)
            if not peer_id or evidence_pack is None:
                log.warning("DB save skipped | peer_id=%s evidence_pack missing", peer_id)
                continue
            save_result_to_db(db, evidence_pack, result, model=model)
            saved_count += 1
        db.commit()
    log.info("Peer overview keyword snapshots saved | count=%s", saved_count)
    return saved_count


def save_result_to_db(
    db: Any, evidence_pack: dict[str, Any], result: dict[str, Any], *, model: str
) -> None:
    evidence_refs = collect_evidence_ids_from_result(result)
    source_signal_ids = sorted(extract_numeric_ids(evidence_refs, "signal:"))
    source_raw_article_ids = sorted(collect_raw_article_ids(evidence_pack, source_signal_ids))
    confidence = average_confidence(result)
    analysis_trace = (
        result.get("analysis_trace") if isinstance(result.get("analysis_trace"), list) else []
    )
    peer_id = str(result["peer_id"])
    params = {
        "analysis_type": ANALYSIS_TYPE,
        "peer_id": peer_id,
        "comparison_mode": COMPARISON_MODE,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": result.get("prompt_version") or PROMPT_VERSION,
        "model_name": result.get("model_name") or model,
        "period": result.get("period") or evidence_pack.get("period"),
        "evidence_hash": result.get("evidence_hash") or evidence_pack.get("evidence_hash"),
        "input_snapshot": json.dumps(evidence_pack, ensure_ascii=False, default=str),
        "output_payload": json.dumps(result, ensure_ascii=False, default=str),
        "analysis_trace": json.dumps(analysis_trace, ensure_ascii=False, default=str),
        "provenance": json.dumps(build_provenance(result), ensure_ascii=False, default=str),
        "confidence": confidence,
        "source_raw_article_ids": source_raw_article_ids,
        "source_signal_ids": source_signal_ids,
        "peer_ids": [peer_id],
    }

    update_result = db.execute(
        text(
            """
            UPDATE peer_llm_analysis_snapshots
            SET
                scope = 'company',
                reference_peer_id = 'sk_ax',
                schema_version = :schema_version,
                status = 'active',
                evidence_hash = :evidence_hash,
                input_snapshot = CAST(:input_snapshot AS jsonb),
                output_payload = CAST(:output_payload AS jsonb),
                analysis_trace = CAST(:analysis_trace AS jsonb),
                provenance = CAST(:provenance AS jsonb),
                confidence = :confidence,
                source_raw_article_ids = :source_raw_article_ids,
                source_signal_ids = :source_signal_ids,
                source_metric_ids = CAST('{}' AS BIGINT[]),
                peer_ids = :peer_ids,
                generated_at = NOW(),
                expires_at = NOW() + INTERVAL '120 days',
                updated_at = NOW()
            WHERE analysis_type = :analysis_type
              AND scope = 'company'
              AND peer_id = :peer_id
              AND comparison_mode = :comparison_mode
              AND prompt_version = :prompt_version
              AND COALESCE(model_name, '') = COALESCE(:model_name, '')
              AND output_payload->>'period' = :period
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
                generated_at,
                expires_at
            )
            VALUES (
                :analysis_type,
                'company',
                :peer_id,
                'sk_ax',
                :comparison_mode,
                :schema_version,
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
                CAST('{}' AS BIGINT[]),
                :peer_ids,
                NOW(),
                NOW() + INTERVAL '120 days'
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
                expires_at = EXCLUDED.expires_at,
                updated_at = NOW()
            """
        ),
        params,
    )


def collect_evidence_ids_from_result(result: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("business_keyword", "technology_keyword"):
        item = result.get(key) if isinstance(result.get(key), dict) else {}
        for ref in item.get("evidence_refs") or []:
            if isinstance(ref, str) and ref.strip():
                refs.add(ref.strip())
    for item in result.get("analysis_trace") or []:
        if isinstance(item, dict):
            for ref in item.get("evidence_refs") or []:
                if isinstance(ref, str) and ref.strip():
                    refs.add(ref.strip())
    return refs


def extract_numeric_ids(evidence_refs: set[str], prefix: str) -> set[int]:
    ids: set[int] = set()
    for ref in evidence_refs:
        if not ref.startswith(prefix):
            continue
        raw_id = ref.removeprefix(prefix)
        if raw_id.isdigit():
            ids.add(int(raw_id))
    return ids


def collect_raw_article_ids(
    evidence_pack: dict[str, Any], source_signal_ids: list[int]
) -> list[int]:
    signal_ids = set(source_signal_ids)
    article_ids: set[int] = set()
    for item in evidence_pack.get("business_signals") or []:
        signal_id = item.get("signal_id")
        if signal_id in signal_ids and isinstance(item.get("raw_article_id"), int):
            article_ids.add(int(item["raw_article_id"]))
    return sorted(article_ids)


def average_confidence(result: dict[str, Any]) -> float | None:
    values = []
    for key in ("business_keyword", "technology_keyword"):
        item = result.get(key) if isinstance(result.get(key), dict) else {}
        confidence = safe_float(item.get("confidence"))
        if confidence is not None:
            values.append(confidence)
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def build_basis(result: dict[str, Any], evidence_pack: dict[str, Any]) -> str:
    confidence = average_confidence(result)
    article_count = len(
        {
            item.get("raw_article_id")
            for item in evidence_pack.get("business_signals") or []
            if item.get("raw_article_id")
        }
    )
    return (
        f"LLM grounded selection · signals {len(evidence_pack.get('business_signals') or [])} "
        f"· articles {article_count}"
        + (f" · confidence {confidence:.2f}" if confidence is not None else "")
    )


def build_provenance(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": "generate_peer_overview_keywords",
        "prompt_version": result.get("prompt_version") or PROMPT_VERSION,
        "model_name": result.get("model_name"),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def parse_json_response(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response root must be an object")
    return parsed


def evidence_hash(pack: dict[str, Any]) -> str:
    stable = json.dumps(pack, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]


def clean_label(value: Any) -> str | None:
    if value is None:
        return None
    text_value = re.sub(r"\s+", " ", str(value)).strip()
    if not text_value or text_value.lower() == "null":
        return None
    return text_value


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    text_value = re.sub(r"\s+", " ", str(value)).strip()
    if len(text_value) <= max_chars:
        return text_value
    cut = text_value.rfind(" ", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text_value[:cut].rstrip() + "..."


def estimate_text_tokens(text_value: str, model: str) -> int:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text_value))
    except Exception:
        return max(1, len(text_value) // 3)


def iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def print_or_write_json(payload: dict[str, Any], output: str | None) -> None:
    text_payload = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if output:
        Path(output).write_text(text_payload + "\n", encoding="utf-8")
        log.info("Wrote output to %s", output)
    else:
        print(text_payload)


if __name__ == "__main__":
    main()
