"""수집 URL 접근성 점검."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx

from src.crawler.base import RawArticle

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


class LinkChecker:
    """URL이 비어 있거나 접근 불가능한 수집물을 저장 전에 걸러낸다."""

    def __init__(self, timeout: float = 6.0, concurrency: int = 20) -> None:
        self.timeout = timeout
        self.concurrency = concurrency

    async def filter_accessible(
        self,
        articles: list[RawArticle],
    ) -> tuple[list[RawArticle], list[RawArticle]]:
        valid: list[RawArticle] = []
        rejected: list[RawArticle] = []
        semaphore = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0"},
        ) as client:

            async def check(article: RawArticle) -> tuple[RawArticle, bool]:
                async with semaphore:
                    return article, await self._is_accessible(client, article)

            results = await asyncio.gather(*(check(article) for article in articles))

        for article, ok in results:
            if ok:
                valid.append(article)
            else:
                rejected.append(article)

        log.info(
            "LinkChecker | total=%d valid=%d rejected=%d",
            len(articles),
            len(valid),
            len(rejected),
        )
        return valid, rejected

    async def _is_accessible(self, client: httpx.AsyncClient, article: RawArticle) -> bool:
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
            article.metadata["link_check"]["status"] = "failed"
            article.metadata["link_check"]["error"] = f"blocked_status:{response.status_code}"
            return False

        if not 200 <= response.status_code <= 399:
            article.metadata["link_check"]["status"] = "failed"
            article.metadata["link_check"]["error"] = f"invalid_status:{response.status_code}"
            return False

        if content_type and not _is_allowed_content_type(content_type):
            article.metadata["link_check"]["status"] = "failed"
            article.metadata["link_check"]["error"] = f"unexpected_content_type:{content_type}"
            return False

        return True


def _has_http_scheme(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def _is_allowed_content_type(content_type: str) -> bool:
    return any(
        content_type.startswith(prefix)
        for prefix in (*_HTML_TYPES, *_DOCUMENT_TYPES)
    )


def _set_link_check(
    article: RawArticle,
    status: str,
    error: str | None = None,
    status_code: int | None = None,
    final_url: str | None = None,
    content_type: str | None = None,
) -> None:
    article.metadata["link_check"] = {
        "status": status,
        "error": error,
        "status_code": status_code,
        "final_url": final_url,
        "content_type": content_type,
    }
