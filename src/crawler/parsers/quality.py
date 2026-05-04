"""공통 크롤링 품질 KPI 평가."""

from __future__ import annotations

from collections import Counter
from typing import Any
from urllib.parse import urlparse

REQUIRED_FIELDS = ("url", "title", "source_name", "peer_id", "published_at")


def attach_quality(
    articles: list[Any],
    peer_aliases: dict[str, list[str]] | None = None,
) -> None:
    """RawArticle의 metadata 또는 extra에 공통 KPI 스키마를 채운다."""
    url_counts = Counter(article.url for article in articles if article.url)
    for article in articles:
        _quality_container(article)["quality"] = evaluate_quality(
            article,
            peer_aliases=peer_aliases or {},
            is_duplicate=url_counts.get(article.url, 0) > 1,
        )


def evaluate_quality(
    article: Any,
    peer_aliases: dict[str, list[str]] | None = None,
    is_duplicate: bool = False,
) -> dict[str, Any]:
    missing_fields = _missing_fields(article)
    quality_issues: list[str] = []

    peer_matched = _peer_matched(article, peer_aliases or {})
    required_fields_complete = not missing_fields
    content_extracted = _content_extracted(article)
    url_valid = _url_valid(article)

    if not peer_matched:
        quality_issues.append("peer_not_found_in_title_or_content")
    if not content_extracted:
        quality_issues.append("content_too_short")
    if not url_valid:
        quality_issues.append("url_invalid_or_inaccessible")
    if is_duplicate:
        quality_issues.append("duplicate_url")

    crawl_success = _crawl_success(article)
    is_usable = (
        crawl_success
        and peer_matched
        and required_fields_complete
        and content_extracted
        and url_valid
        and not is_duplicate
    )

    return {
        "crawl_success": crawl_success,
        "is_usable": is_usable,
        "peer_matched": peer_matched,
        "required_fields_complete": required_fields_complete,
        "content_extracted": content_extracted,
        "url_valid": url_valid,
        "is_duplicate": is_duplicate,
        "missing_fields": missing_fields,
        "quality_issues": quality_issues,
    }


def _missing_fields(article: Any) -> list[str]:
    missing = []
    for field in REQUIRED_FIELDS:
        if not getattr(article, field):
            missing.append(field)
    return missing


def _peer_matched(article: Any, peer_aliases: dict[str, list[str]]) -> bool:
    if not article.peer_id:
        return False
    aliases = [article.peer_id, *peer_aliases.get(article.peer_id, [])]
    haystack = f"{article.title} {article.content}".lower().replace(" ", "")
    return any(alias.lower().replace(" ", "") in haystack for alias in aliases)


def _content_extracted(article: Any) -> bool:
    return len(article.content or "") >= 80


def _url_valid(article: Any) -> bool:
    parsed = urlparse(article.url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    link_check = _quality_container(article).get("link_check", {})
    if link_check.get("status") == "failed":
        return False
    return True


def _crawl_success(article: Any) -> bool:
    link_check = _quality_container(article).get("link_check", {})
    if link_check.get("status") == "failed":
        return False
    return bool(article.url and article.title)


def _quality_container(article: Any) -> dict[str, Any]:
    if hasattr(article, "metadata"):
        return article.metadata
    if hasattr(article, "extra"):
        return article.extra
    return {}
