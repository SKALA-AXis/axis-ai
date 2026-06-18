# 작성일: 2026-06-16
# 작성자: 박지원
# 변경이력:
#   2026-06-16 박지원 — 카드뉴스 통합/요약/시사점 생성 품질 개선 작업의 일부로 작성
"""Backfill industry trend cards up to integration + summary only.

This intentionally does not run StrategicInsightAgent / ImplicationAgent.
Use it when the DB needs card_news rows with integrated issue evidence and
summary lines, while insight/action fields remain empty for a later pass.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_card_news import _load_targets  # noqa: E402
from src.agents.integration_agent import (  # noqa: E402
    IntegrationAgent,
    analysis_input_bundle_from_articles,
)
from src.composers.card_news_composer import CardNewsComposer  # noqa: E402
from src.config.env_loader import load_profile  # noqa: E402
from src.db.article_store import (  # noqa: E402
    get_articles_by_ids,
    save_card_news,
    sync_card_sources_for_cluster,
)
from src.db.integrated_issues import save_integrated_issue  # noqa: E402
from src.db.postgres import reconfigure_from_env  # noqa: E402
from src.preprocessing.preprocessing import PreprocessingService  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("summary_only_card_news")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill card_news with integrated issue + summary only"
    )
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    parser.add_argument("--published-since", required=True)
    parser.add_argument("--published-until", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--cluster-ids", default=None)
    parser.add_argument("--industry-trend-only", action="store_true")
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    reconfigure_from_env()
    targets = _load_targets(
        published_since=args.published_since,
        published_until=args.published_until,
        limit=max(1, args.limit),
        include_existing=args.include_existing,
        only_card_schema_version=None,
        only_existing_active=False,
        only_needs_refresh=False,
        only_deleted_needs_refresh=False,
        cluster_ids=_parse_cluster_ids(args.cluster_ids),
        include_industry_trend=True,
        industry_trend_only=args.industry_trend_only,
    )
    log.info(
        "summary-only 대상 | profile=%s since=%s until=%s targets=%d include_existing=%s",
        profile,
        args.published_since,
        args.published_until,
        len(targets),
        args.include_existing,
    )
    if args.dry_run:
        for target in targets:
            print(
                f"cluster_id={target['cluster_id']} "
                f"representative_id={target['representative_id']} "
                f"articles={len(target['article_ids'])}"
            )
        return

    service = PreprocessingService()
    integrator = IntegrationAgent()
    composer = CardNewsComposer()
    created = 0
    skipped = 0
    errors = 0

    for index, target in enumerate(targets, start=1):
        cluster_id = int(target["cluster_id"])
        article_ids = [int(value) for value in target["article_ids"]]
        representative_id = int(target["representative_id"])
        try:
            classified = service.classify_clusters(
                representative_ids=[representative_id],
                cluster_map={cluster_id: article_ids},
                requested_companies=[],
            )
            if not classified:
                skipped += 1
                log.info("skip | cluster_id=%s reason=no_classification", cluster_id)
                continue

            articles = get_articles_by_ids(article_ids)
            if not articles:
                skipped += 1
                log.info("skip | cluster_id=%s reason=no_articles", cluster_id)
                continue

            classification = _industry_classification(classified[0])
            input_bundle = analysis_input_bundle_from_articles(
                cluster_id=cluster_id,
                representative_id=representative_id,
                articles=articles,
                cluster_article_ids=article_ids,
                classification=classification,
            )
            integrated_issue = integrator.integrate_input_bundle(input_bundle)
            issue_id = save_integrated_issue(integrated_issue, input_bundle=input_bundle)
            if issue_id:
                integrated_issue["integrated_issue_id"] = issue_id

            card = composer.generate_from_cluster(
                cluster_id=cluster_id,
                representative_id=representative_id,
                company="industry_trend",
                classification=classification,
                cluster_article_ids=article_ids,
                peer_id="industry_trend",
                summary=integrated_issue,
            )
            if not card:
                skipped += 1
                log.info("skip | cluster_id=%s reason=no_card", cluster_id)
                continue

            _make_summary_only_card(
                card,
                integrated_issue=integrated_issue,
                input_bundle=input_bundle.to_dict(),
                issue_id=issue_id,
            )
            card_id = save_card_news(card)
            if not card_id:
                skipped += 1
                log.info("skip | cluster_id=%s reason=save_failed", cluster_id)
                continue
            sync_card_sources_for_cluster(cluster_id)
            created += 1
            log.info(
                "saved summary-only | %d/%d card_id=%s cluster_id=%s issue_id=%s",
                index,
                len(targets),
                card_id,
                cluster_id,
                issue_id or "",
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.exception("error | cluster_id=%s error=%s", cluster_id, exc)

    log.info("summary-only 완료 | created=%d skipped=%d errors=%d", created, skipped, errors)
    print(f"created={created} skipped={skipped} errors={errors} targets={len(targets)}")


def _parse_cluster_ids(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _industry_classification(classification: dict[str, Any]) -> dict[str, Any]:
    result = dict(classification)
    result["company"] = ["industry_trend"]
    result["companies"] = ["industry_trend"]
    result.setdefault("sector", "it_trend")
    result.setdefault("sectors", [result["sector"]])
    return result


def _make_summary_only_card(
    card: dict[str, Any],
    *,
    integrated_issue: dict[str, Any],
    input_bundle: dict[str, Any],
    issue_id: str | None,
) -> None:
    card["company"] = "industry_trend"
    card["peer_id"] = "industry_trend"
    card["peer_company_id"] = "industry_trend"
    card["integrated_issue_id"] = issue_id
    card["card_schema_version"] = "v2"
    card["validation"] = {"pass": True, "sc_score": 1.0}
    card["validation_pass"] = True
    card["validation_sc_score"] = 1.0
    card["summary_only"] = True
    card["analysis_package"] = {
        "integrated_issue_id": issue_id,
        "integrated_issue": integrated_issue,
        "summary": integrated_issue,
        "input_bundle": input_bundle,
        "analysis": {},
        "implication": {},
        "summary_only": True,
    }
    card["evidence_payload"] = {
        "integrated_issue_id": issue_id,
        "analysis_package": card["analysis_package"],
        "summary_only": True,
    }
    card["implication"] = {
        "summary_only": True,
        "sector": card.get("sector", "it_trend"),
        "sectors": card.get("sectors", ["it_trend"]),
        "exposure_score": card.get("exposure_score", 0.0),
        "exposure_band": card.get("exposure_band", "low"),
        "signals": card.get("signals", {}),
        "evidence_chain": card.get("evidence_chain", {}),
    }
    card.pop("frontend_implication", None)
    card.pop("display_sections", None)
    card.pop("slides", None)
    card.pop("implication_result", None)


if __name__ == "__main__":
    main()
