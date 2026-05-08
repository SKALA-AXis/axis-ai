"""프론트엔드 CardNewsItem 호환 카드뉴스 생성 에이전트.

PeerNewsSummaryAgent와 PeerNewsAnalysisAgent의 결과를 받아
axis-frontend/src/features/card-news/model/cardNews.ts 타입에 맞는 dict를 만든다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.config.companies import company_name_ko
from src.config.global_companies import global_company_name_ko
from src.db.article_store import get_articles_by_ids

_DEFAULT_COVER_IMAGE_URL = "/png.png"
_DEFAULT_COVER_IMAGE_ALT = "카드뉴스 대표 이미지"
_PROMPT_VERSION = "card-news-v1.0"

_FRONTEND_PEER_IDS = {
    "samsung_sds",
    "lg_cns",
    "hyundai_autoever",
    "posco_dx",
}

_FRONTEND_SECTOR_IDS = {
    "security",
    "ax",
    "infra",
    "biz_area",
    "other",
}

_FRONTEND_EVENT_TYPES = {
    "partnership",
    "ma",
    "personnel",
    "tech",
    "regulation",
    "new_biz",
    "contract",
}


class CardNewsAgent:
    """카드뉴스 화면/ API schema에 맞는 CardNewsItem dict를 생성한다."""

    def generate(
        self,
        summary: dict[str, Any],
        analysis: dict[str, Any] | None = None,
        classification: dict[str, Any] | None = None,
        articles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """요약·분석 결과를 CardNewsItem 호환 dict로 변환한다."""
        analysis = analysis or {}
        classification = classification or {}
        articles = articles if articles is not None else _load_source_articles(summary)

        cluster_id = _optional_int(summary.get("cluster_id"))
        peer_id = _normalize_peer_id(summary.get("main_company"))
        sector = _normalize_sector(classification.get("sector"))
        event_type = _normalize_event_type(classification.get("event_type"))
        exposure_band = _normalize_exposure_band(classification.get("exposure_band"))
        exposure_score = _optional_float(classification.get("exposure_score"))
        trust_score = _trust_score(articles)
        source_article_ids = _source_article_ids(summary, articles)
        created_at = _now_iso()
        published_date = _published_date(articles, created_at)

        title = _first_non_empty(
            summary.get("headline"),
            summary.get("one_line_summary"),
            analysis.get("analysis_summary"),
            "피어사 주요 뉴스",
        )
        summary_lines = _summary_lines(summary)
        strategic_meaning = _string_list(analysis.get("strategic_meaning"))
        insights = strategic_meaning or _string_list(summary.get("fact_summary"))
        detail_points = strategic_meaning or summary_lines
        sources = _sources(articles)
        first_source = sources[0] if sources else {}
        media_assets = _media_assets(articles)
        cover_image = media_assets[0]["url"] if media_assets else _DEFAULT_COVER_IMAGE_URL
        cover_image_alt = media_assets[0]["alt"] if media_assets else _DEFAULT_COVER_IMAGE_ALT

        card = {
            "id": _card_id(cluster_id, created_at),
            "category": sector,
            "date": _display_date(published_date),
            "title": title,
            "coverImageUrl": cover_image,
            "coverImageAlt": cover_image_alt,
            "summary": summary_lines,
            "articlePages": _article_pages(summary, analysis),
            "insights": insights,
            "source": str(first_source.get("source_name") or "AXIS"),
            "sourceUrl": str(first_source.get("url") or ""),
            "detailTitle": _first_non_empty(
                analysis.get("analysis_summary"),
                summary.get("main_event"),
                "피어사 뉴스 분석",
            ),
            "detailDescription": _first_non_empty(
                analysis.get("market_signal"),
                analysis.get("impact_reason"),
                summary.get("one_line_summary"),
            ),
            "detailPoints": detail_points,
            "actionItems": _action_items(analysis),
            "mediaAssets": media_assets,
            "peer_id": peer_id,
            "cluster_id": cluster_id,
            "subtitle": _subtitle(analysis, classification),
            "category_label": sector.upper() if sector == "ax" else sector,
            "published_date": published_date,
            "summary_lines": summary_lines,
            "event_type": event_type,
            "sector": sector,
            "exposure_band": exposure_band,
            "exposure_score": exposure_score,
            "trust_score": trust_score,
            "implication": _implication(analysis),
            "sources": sources,
            "source_count": len(sources),
            "evidence_chain": _evidence_chain(
                summary=summary,
                analysis=analysis,
                sources=sources,
                source_article_ids=source_article_ids,
                cluster_id=cluster_id,
                created_at=created_at,
            ),
            "financial_context": None,
            "slides": _slides(title, summary_lines, insights, sources, media_assets),
            "display": _display_meta(sector, cover_image),
            "validation_pass": _validation_pass(summary, analysis),
            "is_human_reviewed": False,
            "is_bookmarked": False,
            "bookmark_count": 0,
            "share_count": 0,
            "created_at": created_at,
        }
        return card


def _load_source_articles(summary: dict[str, Any]) -> list[dict[str, Any]]:
    ids = [article_id for article_id in _string_list(summary.get("source_article_ids"))]
    numeric_ids = [_optional_int(article_id) for article_id in ids]
    article_ids = [article_id for article_id in numeric_ids if article_id is not None]
    return get_articles_by_ids(article_ids) if article_ids else []


def _card_id(cluster_id: int | None, created_at: str) -> str:
    date_key = created_at[:10].replace("-", "")
    suffix = f"{cluster_id:04d}" if cluster_id is not None else "0000"
    return f"CN-{date_key}-{suffix}"


def _summary_lines(summary: dict[str, Any]) -> list[str]:
    lines = _string_list(summary.get("fact_summary"))[:3]
    if lines:
        return lines

    one_line = str(summary.get("one_line_summary") or "").strip()
    return [one_line] if one_line else []


def _article_pages(summary: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    pages = [
        {
            "title": "사실 요약",
            "paragraphs": _summary_lines(summary),
        },
    ]
    analysis_lines = _string_list(analysis.get("strategic_meaning"))
    if analysis_lines:
        pages.append(
            {
                "title": "의미 분석",
                "paragraphs": analysis_lines,
            }
        )
    return pages


def _action_items(analysis: dict[str, Any]) -> list[str]:
    market_signal = str(analysis.get("market_signal") or "").strip()
    impact_reason = str(analysis.get("impact_reason") or "").strip()
    return [item for item in (market_signal, impact_reason) if item]


def _implication(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "why_important": str(analysis.get("analysis_summary") or "").strip(),
        "potential_impact": str(analysis.get("impact_reason") or "").strip(),
        "follow_up_questions": [],
        "suggested_actions": [],
        "confidence": _optional_float(analysis.get("confidence")),
    }


def _sources(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, article in enumerate(articles, start=1):
        title = str(article.get("title") or "").strip()
        url = str(article.get("url") or "").strip()
        if not title and not url:
            continue

        sources.append(
            {
                "index": index,
                "title": title,
                "url": url,
                "archive_url": article.get("archive_url"),
                "source_name": str(article.get("source_name") or article.get("publisher") or ""),
                "published_at": _string_or_none(article.get("published_at")),
                "credibility_grade": _credibility_grade(article.get("credibility_score")),
                "credibility_score": _optional_float(article.get("credibility_score")),
                "link_status": "ok",
            }
        )
    return sources


def _media_assets(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    seen: set[str] = set()

    for article in articles:
        article_id = _optional_int(article.get("id"))
        title = str(article.get("title") or "").strip()
        for url in _article_image_urls(article):
            if url in seen:
                continue
            seen.add(url)
            assets.append(
                {
                    "id": f"img-{article_id or len(assets) + 1}-{len(assets) + 1}",
                    "type": "image",
                    "url": url,
                    "alt": title or "뉴스 본문 이미지",
                }
            )

    return assets


def _article_image_urls(article: dict[str, Any]) -> list[str]:
    metadata = _metadata(article)
    candidates: list[str] = []

    for key in (
        "image_url",
        "thumbnail_url",
        "thumbnail",
        "og_image",
        "main_image",
        "image",
    ):
        value = article.get(key) or metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for key in ("image_urls", "images", "media_assets", "visual_images"):
        candidates.extend(_image_urls_from_value(article.get(key)))
        candidates.extend(_image_urls_from_value(metadata.get(key)))

    return _dedupe_keep_order([url for url in candidates if _is_image_reference(url)])


def _image_urls_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        urls: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
            elif isinstance(item, dict):
                urls.extend(
                    _string_list(
                        item.get("url")
                        or item.get("image_url")
                        or item.get("image_path")
                        or item.get("path")
                    )
                )
        return urls
    if isinstance(value, dict):
        return _string_list(
            value.get("url")
            or value.get("image_url")
            or value.get("image_path")
            or value.get("path")
        )
    return []


def _evidence_chain(
    summary: dict[str, Any],
    analysis: dict[str, Any],
    sources: list[dict[str, Any]],
    source_article_ids: list[int],
    cluster_id: int | None,
    created_at: str,
) -> dict[str, Any]:
    return {
        "source_links": [
            {
                "title": source.get("title"),
                "source_name": source.get("source_name"),
                "url": source.get("url"),
                "credibility_score": source.get("credibility_score"),
            }
            for source in sources
        ],
        "provenance": {
            "raw_article_ids": source_article_ids,
            "cluster_id": cluster_id,
            "llm_model": analysis.get("model") or summary.get("model"),
            "prompt_version": _PROMPT_VERSION,
            "evidence_version": "v1.0",
            "run_at": created_at,
        },
        "financial_refs": [],
        "mbb_refs": [],
        "evidence_version": "v1.0",
        "pass": _validation_pass(summary, analysis),
        "missing": [],
    }


def _slides(
    title: str,
    summary_lines: list[str],
    insights: list[str],
    sources: list[dict[str, Any]],
    media_assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cover_image = media_assets[0]["url"] if media_assets else _DEFAULT_COVER_IMAGE_URL
    cover_image_alt = media_assets[0]["alt"] if media_assets else _DEFAULT_COVER_IMAGE_ALT
    second_image = media_assets[1]["url"] if len(media_assets) > 1 else None
    second_image_alt = media_assets[1]["alt"] if len(media_assets) > 1 else None
    slides = [
        {
            "order": 1,
            "title": title,
            "body": "\n".join(summary_lines[:3]) or None,
            "image_url": cover_image,
            "image_alt": cover_image_alt,
            "evidence_source_indexes": _source_indexes(sources),
            "layout_type": "summary",
        }
    ]
    if insights:
        slides.append(
            {
                "order": 2,
                "title": "의미 분석",
                "body": "\n".join(insights[:3]),
                "image_url": second_image,
                "image_alt": second_image_alt,
                "evidence_source_indexes": _source_indexes(sources),
                "layout_type": "implication",
            }
        )
    return slides


def _display_meta(sector: str, background_asset_url: str) -> dict[str, Any]:
    return {
        "home_carousel": True,
        "carousel_order": None,
        "slide_count": None,
        "visual_style": "editorial",
        "background_asset_url": background_asset_url,
        "accent_color": _accent_color(sector),
    }


def _source_indexes(sources: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index in (_optional_int(source.get("index")) for source in sources)
        if index is not None
    ]


def _source_article_ids(summary: dict[str, Any], articles: list[dict[str, Any]]) -> list[int]:
    raw_ids = _string_list(summary.get("source_article_ids"))
    ids = [_optional_int(raw_id) for raw_id in raw_ids]
    article_ids = [article_id for article_id in ids if article_id is not None]
    if article_ids:
        return article_ids

    fallback_ids = [_optional_int(article.get("id")) for article in articles]
    return [article_id for article_id in fallback_ids if article_id is not None]


def _published_date(articles: list[dict[str, Any]], created_at: str) -> str:
    for article in articles:
        value = _string_or_none(article.get("published_at"))
        if value:
            return value[:10]
    return created_at[:10]


def _display_date(published_date: str) -> str:
    try:
        return datetime.fromisoformat(published_date).strftime("%Y.%m.%d")
    except ValueError:
        return published_date.replace("-", ".")


def _subtitle(analysis: dict[str, Any], classification: dict[str, Any]) -> str:
    impact = str(analysis.get("impact_level") or "").strip()
    event_type = _normalize_event_type(classification.get("event_type"))
    return impact.upper() if impact else event_type


def _trust_score(articles: list[dict[str, Any]]) -> float | None:
    scores = [
        score
        for score in (_optional_float(article.get("credibility_score")) for article in articles)
        if score is not None
    ]
    return max(scores) if scores else None


def _credibility_grade(value: Any) -> str:
    score = _optional_float(value)
    if score is None:
        return "Unverified"
    if score >= 0.8:
        return "High"
    if score >= 0.6:
        return "Medium"
    return "Low"


def _validation_pass(summary: dict[str, Any], analysis: dict[str, Any]) -> bool:
    summary_valid = bool(summary.get("is_valid_summary", True))
    analysis_valid = bool(analysis.get("is_valid_analysis", True))
    return summary_valid and analysis_valid


def _normalize_peer_id(value: Any) -> str | None:
    peer_id = str(value or "").strip()
    return peer_id if peer_id in _FRONTEND_PEER_IDS else None


def _normalize_sector(value: Any) -> str:
    sector = str(value or "").strip()
    if sector == "deal":
        return "biz_area"
    return sector if sector in _FRONTEND_SECTOR_IDS else "other"


def _normalize_event_type(value: Any) -> str:
    event_type = str(value or "").strip()
    if event_type == "tech_release":
        return "tech"
    if event_type == "financial":
        return "new_biz"
    if event_type == "expansion":
        return "new_biz"
    if event_type == "company":
        return "new_biz"
    return event_type if event_type in _FRONTEND_EVENT_TYPES else "tech"


def _normalize_exposure_band(value: Any) -> str:
    band = str(value or "").strip()
    return band if band in {"high", "medium", "low"} else "medium"


def _accent_color(sector: str) -> str:
    return {
        "ax": "coral",
        "security": "blue",
        "infra": "green",
        "biz_area": "orange",
        "other": "gray",
    }.get(sector, "gray")


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _metadata(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("metadata") or article.get("extra") or {}
    if isinstance(metadata, str):
        try:
            import json

            parsed = json.loads(metadata)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _is_image_reference(value: str) -> bool:
    lower = value.lower()
    if lower.startswith(("http://", "https://", "/", "file://")):
        return True
    return lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".avif"))


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def peer_company_label(company_id: str | None) -> str:
    """프론트 표시용 회사명을 반환한다."""
    if not company_id:
        return ""
    local_name = company_name_ko(company_id)
    return local_name if local_name != company_id else global_company_name_ko(company_id)
