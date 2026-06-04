"""W5-2 CardEvaluatorSidecar — LLM-as-Judge 4 score 평가.

5분 주기 CronJob (`axis-cron-card-evaluator`) 에서 호출. cluster 처리 critical
path 와 무관하게 백그라운드에서 카드의 implication 품질을 평가하여
`card_news.evaluation_payload['llm_judge']` JSONB 에 저장한다.

Guards:
- `card_schema_version = 'v2'` (P5-DATA-2 — v1 카드 191건 자동 skip).
- `NOT (evaluation_payload ? 'llm_judge')` (idempotent).
- daily LLM 비용 soft cap (`EVAL_DAILY_BUDGET_USD`, default $5/일).
- Batch limit (`EVAL_BATCH_LIMIT`, default 20).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

log = logging.getLogger("evaluate_recent_cards")

_DEFAULT_BATCH_LIMIT = 20
_DEFAULT_LOOKBACK_HOURS = 24
_DEFAULT_DAILY_BUDGET_USD = 5.0
_GPT_4O_MINI_PER_CARD_USD = 0.05


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    batch_limit = int(os.environ.get("EVAL_BATCH_LIMIT", _DEFAULT_BATCH_LIMIT))
    lookback_hours = int(os.environ.get("EVAL_LOOKBACK_HOURS", _DEFAULT_LOOKBACK_HOURS))
    daily_budget = float(os.environ.get("EVAL_DAILY_BUDGET_USD", _DEFAULT_DAILY_BUDGET_USD))

    spent_today = _spend_today_usd()
    remaining_budget = max(0.0, daily_budget - spent_today)
    if remaining_budget <= 0:
        log.warning("daily budget exhausted | spent=$%.2f cap=$%.2f", spent_today, daily_budget)
        return
    effective_batch_limit = min(batch_limit, int(remaining_budget / _GPT_4O_MINI_PER_CARD_USD))
    if effective_batch_limit <= 0:
        log.info(
            "budget remaining $%.2f below per-card $%.2f → skip cycle",
            remaining_budget,
            _GPT_4O_MINI_PER_CARD_USD,
        )
        return

    rows = _select_unjudged_cards(limit=effective_batch_limit, hours=lookback_hours)
    log.info(
        "sidecar batch | candidates=%d limit=%d remaining_budget=$%.2f",
        len(rows),
        effective_batch_limit,
        remaining_budget,
    )
    for row in rows:
        try:
            judgment = _llm_judge_card(row)
        except Exception as exc:  # noqa: BLE001
            log.exception("judge failed | card=%s error=%s", row.get("id"), exc)
            continue
        _update_card_evaluation(card_id=row["id"], judgment=judgment)


# ─────────────────────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────────────────────


def _select_unjudged_cards(*, limit: int, hours: int) -> list[dict[str, Any]]:
    from src.db.postgres import SessionLocal

    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT cn.id,
                       cn.implication,
                       cn.sources,
                       COALESCE(cn.evaluation_payload, '{}'::jsonb) AS evaluation_payload,
                       cn.company,
                       cn.peer_company_id,
                       cn.primary_keyword_category,
                       cn.created_at,
                       COALESCE(
                         cn.evidence_payload->'financial_refs',
                         '{}'::jsonb
                       ) AS evidence_financial_refs,
                       COALESCE(
                         cn.evidence_payload->'source_links',
                         '[]'::jsonb
                       ) AS evidence_source_links,
                       COALESCE(
                         cn.evidence_payload->'mbb_refs',
                         '[]'::jsonb
                       ) AS evidence_mbb_refs
                  FROM card_news cn
                 WHERE cn.card_schema_version = 'v2'
                   AND NOT (COALESCE(cn.evaluation_payload, '{}'::jsonb) ? 'llm_judge')
                   AND cn.created_at >= NOW() - (:hours || ' hours')::interval
                 ORDER BY cn.created_at ASC
                 LIMIT :limit
                """
            ),
            {"hours": int(hours), "limit": int(limit)},
        ).fetchall()
    return [dict(row._mapping) for row in rows]


def _update_card_evaluation(*, card_id: str, judgment: dict[str, Any]) -> None:
    from src.db.postgres import SessionLocal

    with SessionLocal() as db:
        db.execute(
            text(
                """
                UPDATE card_news
                   SET evaluation_payload =
                       COALESCE(evaluation_payload, '{}'::jsonb)
                       || jsonb_build_object('llm_judge', CAST(:judgment AS jsonb))
                 WHERE id = :card_id
                """
            ),
            {
                "card_id": card_id,
                "judgment": json.dumps(judgment, ensure_ascii=False),
            },
        )
        db.commit()


def _spend_today_usd() -> float:
    from src.db.postgres import SessionLocal

    try:
        with SessionLocal() as db:
            exists = db.execute(text("SELECT to_regclass('public.card_news')")).scalar()
            if exists is None:
                return 0.0
            row = db.execute(
                text(
                    """
                    SELECT COUNT(*) AS n
                      FROM card_news
                     WHERE card_schema_version = 'v2'
                       AND (evaluation_payload ? 'llm_judge')
                       AND created_at::date = CURRENT_DATE
                    """
                )
            ).fetchone()
    except Exception as exc:  # noqa: BLE001 — pre-V33 fallback.
        log.debug("spend_today fallback | error=%s", exc)
        return 0.0
    if row is None:
        return 0.0
    n = int(row._mapping.get("n") or 0)
    return n * _GPT_4O_MINI_PER_CARD_USD


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────


def _llm_judge_card(row: dict[str, Any]) -> dict[str, Any]:
    from src.evaluators.llm_judge_prompts import (
        JUDGE_MODEL,
        JUDGE_PROMPT_VERSION,
        JUDGE_SYSTEM_PROMPT,
        SCORE_PROMPTS,
    )

    implication = _ensure_dict(row.get("implication"))
    skax = implication.get("skax_implication") or {}
    peer = implication.get("peer_implication") or {}
    # evidence_payload 우선순위: evidence_chain 테이블의 source_links/financial_refs/mbb_refs
    # > card_news.sources (raw fallback).
    evidence_payload = {
        "source_links": _ensure_list(row.get("evidence_source_links"))
        or _ensure_list(row.get("sources")),
        "financial_refs": _ensure_dict(row.get("evidence_financial_refs")),
        "mbb_refs": _ensure_list(row.get("evidence_mbb_refs")),
    }
    fact_basis: list[Any] = []
    integrated = implication.get("integrated_issue") or {}
    if isinstance(integrated, dict):
        fact_basis = integrated.get("fact_basis", [])

    cluster_meta = {
        "company": row.get("company"),
        "peer_company_id": row.get("peer_company_id"),
        "sector": row.get("primary_keyword_category"),
    }

    inputs = {
        "faithfulness": {
            "implication_json": _json_dumps(implication),
            "evidence_json": _json_dumps(evidence_payload),
            "fact_basis_json": _json_dumps(fact_basis),
        },
        "specificity_llm": {
            "implication_json": _json_dumps(implication),
            "cluster_meta_json": _json_dumps(cluster_meta),
        },
        "actionability_llm": {
            "skax_json": _json_dumps(skax),
        },
        "peer_relevance": {
            "peer_json": _json_dumps(peer),
            "skax_json": _json_dumps(skax),
        },
    }

    judgment: dict[str, Any] = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "evaluator_model_version": JUDGE_MODEL,
        "evaluator_prompt_version": JUDGE_PROMPT_VERSION,
    }
    llm = _make_judge_llm()
    for metric_name, prompt_template in SCORE_PROMPTS.items():
        user_prompt = prompt_template.format(**inputs[metric_name])
        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            parsed = _parse_judge_json(content)
            judgment[metric_name] = float(parsed.get("score") or 0.0)
            judgment.setdefault("reasoning", {})
            judgment["reasoning"][metric_name] = str(parsed.get("reasoning") or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM judge metric=%s failed | %s", metric_name, exc)
            judgment[metric_name] = 0.0
    return judgment


def _make_judge_llm() -> Any:
    from langchain_openai import ChatOpenAI

    from src.evaluators.llm_judge_prompts import JUDGE_MODEL

    return ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        max_completion_tokens=256,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _parse_judge_json(text_value: str) -> dict[str, Any]:
    text_value = (text_value or "").strip()
    if not text_value:
        return {}
    if text_value.startswith("```"):
        parts = text_value.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.startswith("json"):
                body = body[4:]
            text_value = body.strip()
    try:
        result = json.loads(text_value)
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _ensure_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


if __name__ == "__main__":
    main()
