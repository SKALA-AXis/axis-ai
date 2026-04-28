"""DART 공시 크롤러"""

import io
import logging
import os
import zipfile
from datetime import datetime, timedelta
from xml.etree import ElementTree

import httpx

from src.crawler.article_filter import strip_html
from src.crawler.base_crawler import BaseCrawler, RawArticle

log = logging.getLogger(__name__)

CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
DART_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DEFAULT_LOOKBACK_DAYS = 30


class DartCrawler(BaseCrawler):
    def __init__(
        self,
        peer_id: str,
        corp_code: str | None = None,
        corp_names: list[str] | None = None,
    ):
        super().__init__(peer_id)
        env_key = f"DART_CORP_CODE_{peer_id.upper()}"
        self.corp_code = corp_code or os.getenv(env_key, "")
        self.corp_names = corp_names or []
        self.api_key = os.getenv("DART_API_KEY", "")
        self.lookback_days = int(os.getenv("DART_LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS)))

    async def crawl(self) -> list[RawArticle]:
        if not self.api_key:
            log.warning("DART_API_KEY 미설정. 크롤링 스킵.")
            return []

        corp_codes = [self.corp_code] if self.corp_code else await self._resolve_corp_codes()
        if not corp_codes:
            log.warning(
                "DART corp_code를 찾지 못해 스킵 | peer_id=%s names=%s",
                self.peer_id,
                self.corp_names,
            )
            return []

        articles: list[RawArticle] = []
        for corp_code in corp_codes:
            articles.extend(await self._fetch_disclosures(corp_code=corp_code))
        return articles

    async def _resolve_corp_codes(self) -> list[str]:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(CORP_CODE_URL, params={"crtfc_key": self.api_key})
            resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
            xml_data = archive.read("CORPCODE.xml")

        root = ElementTree.fromstring(xml_data)
        corp_codes = []
        normalized_names = {_normalize_name(name) for name in self.corp_names}
        for item in root.findall("list"):
            corp_name = item.findtext("corp_name", default="")
            if _normalize_name(corp_name) in normalized_names:
                corp_codes.append(item.findtext("corp_code", default=""))
        return [corp_code for corp_code in corp_codes if corp_code]

    async def _fetch_disclosures(self, corp_code: str) -> list[RawArticle]:
        end = datetime.now()
        begin = end - timedelta(days=self.lookback_days)
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bgn_de": begin.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "page_count": 100,
            "sort": "date",
            "sort_mth": "desc",
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(DART_LIST_URL, params=params)
            resp.raise_for_status()

        payload = resp.json()
        if payload.get("status") not in ("000", "013"):
            log.warning(
                "DART 조회 실패 | peer_id=%s status=%s message=%s",
                self.peer_id,
                payload.get("status"),
                payload.get("message"),
            )
            return []

        return [
            RawArticle(
                url=f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item['rcept_no']}",
                title=strip_html(item.get("report_nm", "")),
                content=strip_html(_disclosure_content(item)),
                published_at=datetime.strptime(item["rcept_dt"], "%Y%m%d").astimezone(),
                source_name="dart",
                peer_id=self.peer_id,
            )
            for item in payload.get("list", [])
        ]


def _normalize_name(name: str) -> str:
    return name.replace(" ", "").replace("㈜", "").replace("(주)", "").lower()


def _disclosure_content(item: dict) -> str:
    return f"{item.get('corp_name', '')} {item.get('flr_nm', '')} {item.get('rm', '')}".strip()
