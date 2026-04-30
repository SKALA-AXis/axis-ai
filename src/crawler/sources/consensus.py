"""한경 컨센서스 애널리스트 리포트 크롤러 (Tier 2) — Playwright, Track B."""

import logging
from typing import Optional

from src.crawler.base import DailyLimitGuard, RawArticle
from src.crawler.playwright_client import PlaywrightClient

log = logging.getLogger(__name__)

BASE_URL = "https://consensus.hankyung.com/analysis/list"
_PEERS = ["삼성SDS", "LG CNS", "현대오토에버", "포스코DX"]
_PEER_KEY_HINTS: list[tuple[str, str]] = [
    ("삼성", "samsung_sds"),
    ("LG CNS", "lg_cns"),
    ("엘지", "lg_cns"),
    ("오토에버", "hyundai_autoever"),
    ("포스코", "posco_dx"),
]


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

            # 검색 입력 필드 — 셀렉터 여러 개 시도 후 실패 시 로그만 남김
            search_input = None
            for sel in (
                "input[placeholder*='검색']",
                "input[type='search']",
                "input.search-input",
                "input[name*='keyword']",
                "input[id*='search']",
            ):
                try:
                    await page.wait_for_selector(sel, timeout=2_000)
                    search_input = sel
                    break
                except Exception:
                    continue

            if not search_input:
                log.warning(
                    "한경 컨센서스 검색 입력 필드 탐지 실패 | peer=%s — 목록 전체를 조회합니다",
                    peer_name,
                )
            else:
                try:
                    await page.fill(search_input, peer_name)
                    await page.keyboard.press("Enter")
                except Exception as e:
                    log.warning("한경 컨센서스 검색 입력 실패 | peer=%s error=%s", peer_name, e)

            # 결과 리스트 — 셀렉터 여러 개 시도
            list_selector = None
            for sel in (
                ".analysis-list-item",
                "table tbody tr",
                "ul.list-item li",
                ".list-item",
            ):
                try:
                    await page.wait_for_selector(sel, timeout=4_000)
                    list_selector = sel
                    break
                except Exception:
                    continue

            if not list_selector:
                log.warning("한경 컨센서스 결과 리스트 탐지 실패 | peer=%s", peer_name)
                return []

            items = await page.query_selector_all(list_selector)
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
                            source_name="hankyung_consensus",
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

            log.info(
                "한경 컨센서스 수집 | peer=%s list_sel=%s items=%d extracted=%d",
                peer_name,
                list_selector,
                len(items),
                len(articles),
            )

        return articles


def _resolve_peer(name: str) -> Optional[str]:
    for hint, peer_id in _PEER_KEY_HINTS:
        if hint in name:
            return peer_id
    return None
