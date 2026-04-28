"""Playwright 공유 브라우저 세션 — 스텔스 모드 + Markdown 추출."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page, async_playwright

log = logging.getLogger(__name__)

# 매체별 본문 CSS 선택자 (readability 전에 우선 시도)
_MEDIA_SELECTORS: dict[str, str] = {
    "hankyung.com": "div.article-body",
    "consensus.hankyung.com": "div.article-body",
    "zdnet.co.kr": "div#article_body",
    "etnews.com": "div#article_txt",
    "mk.co.kr": "div#article_body",
    "inews24.com": "div#article_txt",
    "bloter.net": "div.article__content",
}

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

_STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
window.chrome = { runtime: {} };
"""


class PlaywrightClient:
    """Playwright 브라우저 인스턴스 공유 클라이언트. 오버헤드 최소화."""

    _browser: Optional[Browser] = None

    @classmethod
    async def get_browser(cls) -> Browser:
        if cls._browser is None or not cls._browser.is_connected():
            playwright = await async_playwright().start()
            cls._browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ],
            )
        return cls._browser

    @asynccontextmanager
    async def new_page(self, stealth: bool = True) -> AsyncGenerator[Page, None]:
        """스텔스 모드 페이지 컨텍스트 매니저."""
        browser = await self.get_browser()
        context = await browser.new_context(
            user_agent=_USER_AGENT,
            viewport={"width": 1280, "height": 800},
            locale="ko-KR",
        )
        if stealth:
            await context.add_init_script(_STEALTH_SCRIPT)
        page = await context.new_page()
        try:
            yield page
        finally:
            await context.close()

    async def fetch_html(self, url: str, wait_selector: Optional[str] = None) -> str:
        """URL → 렌더링된 raw HTML 반환 (파싱은 호출자가 담당)."""
        try:
            async with self.new_page() as page:
                await page.goto(url, wait_until="networkidle", timeout=15_000)
                if wait_selector:
                    await page.wait_for_selector(wait_selector, state="attached", timeout=10_000)
                return await page.content()
        except Exception as e:
            log.warning("Playwright fetch_html 실패 | url=%s error=%s", url, e)
            return ""

    async def fetch_markdown(self, url: str, wait_selector: Optional[str] = None) -> str:
        """URL → 본문 텍스트 추출 (Playwright 렌더링 후 BeautifulSoup/readability 파싱)."""
        try:
            async with self.new_page() as page:
                await page.goto(url, wait_until="networkidle", timeout=15_000)
                if wait_selector:
                    await page.wait_for_selector(wait_selector, state="attached", timeout=10_000)
                html = await page.content()
            return self._html_to_text(html, url)
        except Exception as e:
            log.warning("Playwright fetch 실패 | url=%s error=%s", url, e)
            return ""

    def _html_to_text(self, html: str, url: str) -> str:
        """매체별 선택자 우선 → readability 폴백으로 본문 추출."""
        domain = urlparse(url).netloc.replace("www.", "")
        soup = BeautifulSoup(html, "html.parser")

        sel = _MEDIA_SELECTORS.get(domain)
        if sel:
            node = soup.select_one(sel)
            if node:
                return node.get_text(separator="\n", strip=True)

        # readability 폴백
        try:
            from readability import Document

            doc = Document(html)
            return BeautifulSoup(doc.summary(), "html.parser").get_text(separator="\n", strip=True)
        except Exception:
            body = soup.find("body")
            return body.get_text(separator=" ", strip=True)[:3000] if body else ""
