"""W1-W5 통합 e2e — cluster postgres + 실 OpenAI 호출 1 cluster."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

# 1) env (DATABASE_URL 은 caller 가 export. OPENAI_API_KEY 도 caller 가 export.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _dumps(value: Any, *, limit: int | None = None) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(value)
    if limit and len(text) > limit:
        return text[:limit] + " … (truncated)"
    return text


def main(cluster_id: int) -> None:
    from src.agents.issue_integration_agent import (
        analysis_input_bundle_from_articles,
    )
    from src.db.article_store import get_articles_by_ids
    from src.pipeline.analysis_flow_graph import run_supervisor
    from src.pipeline.analysis_pipeline import (
        build_classification_from_articles,
        list_cluster_article_ids,
    )

    article_ids = list_cluster_article_ids(cluster_id)
    print(f"\n=== cluster {cluster_id} | article_ids={article_ids}")
    if not article_ids:
        print("no articles → exit")
        return

    articles = get_articles_by_ids(article_ids)
    print(f"loaded {len(articles)} articles")
    rep = next((a for a in articles if a.get("is_representative")), articles[0])
    rep_id = int(rep["id"])
    print(f"representative id={rep_id} title={rep.get('title', '')[:80]}")

    classification = build_classification_from_articles(
        cluster_id=cluster_id,
        representative_id=rep_id,
        articles=articles,
    )
    print("classification:", _dumps(classification, limit=400))

    bundle = analysis_input_bundle_from_articles(
        cluster_id=cluster_id,
        representative_id=rep_id,
        articles=articles,
        cluster_article_ids=article_ids,
        classification=classification,
    )
    print(f"bundle id={bundle.bundle_id} companies={bundle.companies} sectors={bundle.sectors}")

    t0 = time.perf_counter()
    state = run_supervisor(input_bundle=bundle, classification=classification)
    elapsed = time.perf_counter() - t0
    print(f"\n=== supervisor done | elapsed={elapsed:.2f}s")

    validation = state.get("validation")
    impl = state.get("implication") or {}
    metrics = validation.metrics if validation and validation.metrics else None
    ctx = state.get("analysis_context")

    print("\n--- AnalysisContext layers ---")
    if ctx:
        print(f"available_layer_count = {ctx.available_layer_count()}")
        print(f"timeline entries      = {len(ctx.peer_event_timeline_recent)}")
        print(f"sector_pulse rows     = {len(ctx.sector_pulse_recent)}")
        print(f"financial trends      = {len(ctx.financial_trend)}")
        print(f"event_chain cands     = {len(ctx.event_chain_candidates)}")
        print(f"similar cards (RAG)   = {len(ctx.similar_cards_rag)}")
        print(f"token_budget_used     ≈ {ctx.token_budget_used}")

    print("\n--- IntegratedIssue (요약) ---")
    integ = state.get("integrated_issue") or {}
    print(f"main_company = {integ.get('main_company')}")
    print(f"is_valid_summary = {integ.get('is_valid_summary')}")
    print(f"main_issue = {integ.get('main_issue', '')[:120]}")
    print("integrated_text[:200] =", str(integ.get("integrated_text", ""))[:200])

    print("\n--- AnalysisResult ---")
    analysis = state.get("analysis") or {}
    print(f"is_valid_analysis = {analysis.get('is_valid_analysis')}")
    print(f"analysis_summary = {analysis.get('analysis_summary', '')[:160]}")
    impact_level = analysis.get("impact_level")
    risk_or_opportunity = analysis.get("risk_or_opportunity")
    print(f"impact_level = {impact_level} | risk_or_opportunity = {risk_or_opportunity}")
    print(f"market_signal = {analysis.get('market_signal', '')[:160]}")
    print(f"strategic_meaning = {analysis.get('strategic_meaning')}")
    print(f"confidence = {analysis.get('confidence')}")

    print("\n--- ImplicationResult (v4/v5) ---")
    print(f"is_valid_implication = {impl.get('is_valid_implication')}")
    print(f"prompt_version = {(impl.get('provenance') or {}).get('prompt_version')}")
    print(f"used_context_layers = {(impl.get('provenance') or {}).get('used_context_layers')}")
    peer = impl.get("peer_implication") or {}
    skax = impl.get("skax_implication") or {}
    print(f"peer_implication.company = {peer.get('company_id')} ({peer.get('company_name_ko')})")
    print(f"peer_meaning = {(peer.get('peer_meaning') or '')[:200]}")
    print(f"capability_change = {(peer.get('capability_change') or '')[:200]}")
    print(f"\nskax_implication.why_important = {(skax.get('why_important') or '')[:200]}")
    print(f"skax_implication.potential_impact = {(skax.get('potential_impact') or '')[:200]}")
    print(f"opportunities = {skax.get('opportunities')}")
    print(f"threats = {skax.get('threats')}")
    print(f"recommended_actions = {skax.get('recommended_actions')}")
    print(f"business_line_mapping = {skax.get('business_line_mapping')}")
    print(f"confidence = {impl.get('confidence')} | evidence_label = {impl.get('evidence_label')}")
    print(f"follow_up_questions = {impl.get('follow_up_questions')}")
    print(f"watch_points = {impl.get('watch_points')}")

    print("\n--- ValidationReport (W2-3 + W5-1 metrics) ---")
    if validation:
        print(f"pass = {validation.passed} | sc_score = {validation.sc_score}")
        print(f"integrated_issue_valid = {validation.integrated_issue_valid}")
        print(f"analysis_valid = {validation.analysis_valid}")
        print(f"implication_valid = {validation.implication_valid}")
        print(f"numeric_violations = {[v.to_dict() for v in validation.numeric_violations]}")
        print(f"certainty_warnings = {validation.certainty_warnings}")
        print(f"evidence_chain_warnings = {validation.evidence_chain_warnings}")
    if metrics:
        print("\n  W5-1 rule-based metrics:")
        print(f"    context_hit_ratio   = {metrics.context_hit_ratio}")
        print(f"    evidence_claim_ratio = {metrics.evidence_claim_ratio}")
        print(f"    specificity_score   = {metrics.specificity_score}")
        print(f"    actionability_score = {metrics.actionability_score}")
        print(f"    regression_drift    = {metrics.regression_drift}")

    print("\n--- card_news (DB insert) ---")
    print(f"saved card_news_id = {state.get('card_news_id')}")
    print(f"human_review_flags = {state.get('human_review_flags')}")
    print(f"errors = {state.get('errors')}")


if __name__ == "__main__":
    cluster_id = int(sys.argv[1]) if len(sys.argv) > 1 else 40813
    main(cluster_id)
