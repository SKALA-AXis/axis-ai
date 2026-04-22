"""한경 컨센서스 애널리스트 리포트 크롤러 (Tier 2) — Playwright, Track B."""

import logging
from typing import Optional

from src.crawler.base import SOURCE_CREDIBILITY, DailyLimitGuard, RawArticle
from src.crawler.playwright_client import PlaywrightClient

log = logging.getLogger(__name__)

BASE_URL = "https://consensus.hankyung.com/analysis/list"
_PEERS = ["삼성SDS", "LG CNS"]


class HankyungConsensusCrawler:
    """한경 컨센서스 애널리스트 리포트 수집 — React SPA이므로 Playwright 필수."""

    def __init__(self, limit_guard: Optional[DailyLimitGuard] = None) -> None:
        self.pw = PlaywrightClient()
        self.limit_guard = limit_guard or DailyLimitGuard()

    async def crawl(self) -> list[RawArticle]:
        if not self.limit_guard.allow("consensus"):
            return []
        articles: list[RawArticle] = []
        for peer in _PEERS:
            try:
                articles.extend(await self._fetch_peer(peer))
            except Exception as e:
                log.error("한경 컨센서스 크롤링 실패 | peer=%s error=%s", peer, e)
        return articles

    async def _fetch_peer(self, peer_name: str) -> list[RawArticle]:
        async with self.pw.new_page() as page:
            await page.goto(BASE_URL, wait_until="networkidle", timeout=15_000)
            await page.fill("input[placeholder*='검색']", peer_name)
            await page.keyboard.press("Enter")
            await page.wait_for_selector(".analysis-list-item", timeout=8_000)

            items = await page.query_selector_all(".analysis-list-item")
            articles = []

            for item in items[:20]:
                try:
                    title_el = await item.query_selector(".title")
                    firm_el = await item.query_selector(".company")
                    date_el = await item.query_selector(".date")
                    opinion_el = await item.query_selector(".opinion")
                    price_el = await item.query_selector(".target-price")
                    link_el = await item.query_selector("a")

                    title = await title_el.inner_text() if title_el else ""
                    firm = await firm_el.inner_text() if firm_el else ""
                    date_str = await date_el.inner_text() if date_el else ""
                    opinion = await opinion_el.inner_text() if opinion_el else ""
                    price = await price_el.inner_text() if price_el else ""
                    href = await link_el.get_attribute("href") if link_el else BASE_URL

                    if not href or not href.startswith("http"):
                        href = f"https://consensus.hankyung.com{href}"

                    articles.append(
                        RawArticle(
                            url=href,
                            title=f"[{firm.strip()}] {peer_name} {title.strip()}",
                            content=f"투자의견: {opinion} / 목표주가: {price}",
                            source_tier=2,
                            source_name="hankyung_consensus",
                            credibility_score=SOURCE_CREDIBILITY["hankyung_consensus"],
                            peer_id=_resolve_peer(peer_name),
                            metadata={
                                "type": "analyst_report",
                                "securities_firm": firm.strip(),
                                "opinion": opinion,
                                "target_price": price,
                                "report_date": date_str.strip(),
                                "importance_hint": "high" if price else "normal",
                            },
                        )
                    )
                except Exception as e:
                    log.warning("컨센서스 항목 파싱 실패 | error=%s", e)

        return articles


def _resolve_peer(name: str) -> str:
    return "samsung_sds" if "삼성" in name else "lg_cns"
