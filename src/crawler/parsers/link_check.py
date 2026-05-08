"""수집 URL 접근성 점검."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

REJECT_STATUS_CODES = {401, 403, 404, 408, 410, 429, 451}
_HTML_TYPES = ("text/html", "application/xhtml")
_DOCUMENT_TYPES = (
    "application/pdf",
    "application/octet-stream",
    "application/vnd.openxmlformats",
    "application/msword",
    "application/vnd.ms-",
)

# API 기반 소스는 URL 접근성 체크 불필요
# - content를 이미 API 응답에서 가져왔음
# - URL 자체가 브라우저 세션 없이 접근 불가인 경우가 많음 (DART 등)
_SKIP_LINK_CHECK_SOURCE_TYPES = {"api", "dart", "ir", "search_trend", "trend_report"}

# URL 접근성 체크를 스킵할 도메인
# DART 뷰어는 브라우저 세션 없이 항상 403 반환
_SKIP_LINK_CHECK_DOMAINS = {
    "dart.fss.or.kr",
    "opendart.fss.or.kr",
}


class LinkChecker:
    """URL이 비어 있거나 접근 불가능한 수집물을 저장 전에 걸러낸다."""

    def __init__(self, timeout: float = 6.0, concurrency: int = 20) -> None:
        self.timeout = timeout
        self.concurrency = concurrency

    async def filter_accessible(
        self,
        articles: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        valid: list[Any] = []
        rejected: list[Any] = []
        skipped: list[Any] = []
        semaphore = asyncio.Semaphore(self.concurrency)

        # LinkChecker 스킵 대상 먼저 분리
        to_check: list[Any] = []
        for article in articles:
            if _should_skip_link_check(article):
                _set_link_check(article, status="skipped", error="api_source")
                skipped.append(article)
            else:
                to_check.append(article)

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0"},
        ) as client:

            async def check(article: Any) -> tuple[Any, bool]:
                async with semaphore:
                    return article, await self._is_accessible(client, article)

            results = await asyncio.gather(*(check(article) for article in to_check))

        for article, ok in results:
            if ok:
                valid.append(article)
            else:
                rejected.append(article)

        # 스킵된 항목은 valid로 합산
        valid.extend(skipped)

        log.info(
            "LinkChecker | total=%d valid=%d rejected=%d skipped(api)=%d",
            len(articles),
            len(valid),
            len(rejected),
            len(skipped),
        )
        return valid, rejected

    async def _is_accessible(self, client: httpx.AsyncClient, article: Any) -> bool:
        if not article.url:
            _set_link_check(article, status="failed", error="empty_url")
            return False

        if not _has_http_scheme(article.url):
            _set_link_check(article, status="failed", error="invalid_scheme")
            return False

        try:
            response = await client.head(article.url)
            if response.status_code in (405, 501) or response.status_code >= 400:
                response = await client.get(article.url)
        except httpx.HTTPError as exc:
            _set_link_check(
                article,
                status="failed",
                error=f"request_error:{exc.__class__.__name__}",
            )
            return False

        content_type = response.headers.get("content-type", "").split(";")[0].lower()
        _set_link_check(
            article,
            status="success",
            status_code=response.status_code,
            final_url=str(response.url),
            content_type=content_type,
        )

        if response.status_code in REJECT_STATUS_CODES:
            _article_container(article)["link_check"]["status"] = "failed"
            _article_container(article)["link_check"]["error"] = (
                f"blocked_status:{response.status_code}"
            )
            return False

        if not 200 <= response.status_code <= 399:
            _article_container(article)["link_check"]["status"] = "failed"
            _article_container(article)["link_check"]["error"] = (
                f"invalid_status:{response.status_code}"
            )
            return False

        if content_type and not _is_allowed_content_type(content_type):
            _article_container(article)["link_check"]["status"] = "failed"
            _article_container(article)["link_check"]["error"] = (
                f"unexpected_content_type:{content_type}"
            )
            return False

        return True


def _should_skip_link_check(article: Any) -> bool:
    """API 기반 소스이거나 접근 불가 도메인이면 LinkChecker 스킵."""
    # source_type 기반 스킵
    source_type = getattr(article, "source_type", None)
    if source_type in _SKIP_LINK_CHECK_SOURCE_TYPES:
        return True

    # extra/metadata 안의 source_type도 확인
    container = _article_container(article)
    if container.get("source_type") in _SKIP_LINK_CHECK_SOURCE_TYPES:
        return True

    # 도메인 기반 스킵
    url = getattr(article, "url", "") or ""
    domain = urlparse(url).netloc
    if domain in _SKIP_LINK_CHECK_DOMAINS:
        return True

    return False


def _has_http_scheme(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def _is_allowed_content_type(content_type: str) -> bool:
    return any(content_type.startswith(prefix) for prefix in (*_HTML_TYPES, *_DOCUMENT_TYPES))


def _set_link_check(
    article: Any,
    status: str,
    error: str | None = None,
    status_code: int | None = None,
    final_url: str | None = None,
    content_type: str | None = None,
) -> None:
    _article_container(article)["link_check"] = {
        "status": status,
        "error": error,
        "status_code": status_code,
        "final_url": final_url,
        "content_type": content_type,
    }


def _article_container(article: Any) -> dict[str, Any]:
    if hasattr(article, "metadata"):
        return article.metadata
    if hasattr(article, "extra"):
        return article.extra
    return {}
