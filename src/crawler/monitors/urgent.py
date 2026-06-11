"""Track A — 실시간 긴급 감지 상시 실행 모니터.

docker-compose에서 별도 컨테이너(urgent-monitor)로 실행됩니다.
  python -m src.crawler.monitors.urgent

파이프라인:
  RSS/DART 폴링 → FastFilter + 경량 분류기 → LLM 검증 (urgent만) → Slack
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import feedparser
import httpx

from src.crawler.base import SOURCE_CREDIBILITY, DailyLimitGuard, RawArticle
from src.crawler.fast_filter import FastFilter, FilterResult

log = logging.getLogger(__name__)

# ── 상시 감시 소스 ────────────────────────────────────────────────
_RSS_SOURCES: dict[str, str] = {
    "google_news_sds": "https://news.google.com/rss/search?q=삼성SDS&hl=ko&gl=KR&ceid=KR:ko",
    "google_news_lgc": "https://news.google.com/rss/search?q=LG+CNS&hl=ko&gl=KR&ceid=KR:ko",
    "hankyung_it": "https://www.hankyung.com/feed/it",
    "etnews": "https://www.etnews.com/rss/",
}
_PAGE_MONITORS: dict[str, str] = {
    "samsung_sds_newsroom": "https://news.samsung.com/kr/category/press-release/",
    "lg_cns_newsroom": "https://www.lgcns.com/blog/",
}

RSS_INTERVAL = 60  # 초
PAGE_INTERVAL = 120  # 초
PEER_KEYWORDS = ["삼성SDS", "Samsung SDS", "LG CNS", "엘지씨엔에스"]

# LLM 검증은 urgent(score≥60)만 호출 — 하루 ~5~10건 예상
LLM_VERIFY_TIMEOUT = 15  # 초

_VERIFY_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 기사가 SK AX 관점에서 즉시 Slack 긴급 알림을 보낼 만큼 중요한지 판단하세요.

[기사 제목]
{title}

[기사 내용]
{content}

[출처]
{source_name} (신뢰도: {credibility_score})

판단 기준:
- 삼성SDS 또는 LG CNS의 M&A, 대형 수주, 전략적 파트너십, 경영진 교체
- SK AX 주요 사업(에이전틱AI·제조AX·MSP)에 직접적 위협 또는 기회

다음 JSON 형식으로만 응답하세요:
{{
  "confirmed": true/false,
  "reason": "판단 근거 1~2줄",
  "sk_ax_impact": "SK AX 관점 시사점 1줄 (confirmed=true일 때만)"
}}"""


@dataclass
class VerifyResult:
    confirmed: bool
    reason: str
    sk_ax_impact: str


class UrgentVerifier:
    """FastFilter urgent 판정 기사를 GPT-4o로 2차 검증."""

    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI  # 지연 임포트 — transformers/torch 체인 회피 (import ~분 단위)
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=300)

    async def verify(self, article: RawArticle) -> VerifyResult:
        """LLM에 검증 요청. 타임아웃·오류 시 confirmed=True로 폴백 (놓치지 않는 방향)."""
        prompt = _VERIFY_PROMPT.format(
            title=article.title,
            content=article.content[:800],
            source_name=article.source_name,
            credibility_score=article.credibility_score,
        )
        try:
            response = await asyncio.wait_for(
                self._llm.ainvoke(prompt),
                timeout=LLM_VERIFY_TIMEOUT,
            )
            return _parse_verify_response(response.content)
        except asyncio.TimeoutError:
            log.warning("LLM 검증 타임아웃 — confirmed=True 폴백 | title=%s", article.title)
            return VerifyResult(confirmed=True, reason="LLM 타임아웃 (폴백)", sk_ax_impact="")
        except Exception as e:
            log.error("LLM 검증 오류 — confirmed=True 폴백 | error=%s", e)
            return VerifyResult(confirmed=True, reason=f"LLM 오류: {e}", sk_ax_impact="")


def _parse_verify_response(content: str) -> VerifyResult:
    try:
        # 코드블록 제거 후 JSON 파싱
        cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        data = json.loads(cleaned)
        return VerifyResult(
            confirmed=bool(data.get("confirmed", True)),
            reason=data.get("reason", ""),
            sk_ax_impact=data.get("sk_ax_impact", ""),
        )
    except Exception:
        log.warning("LLM 응답 JSON 파싱 실패 — confirmed=True 폴백 | content=%s", content[:100])
        return VerifyResult(confirmed=True, reason="파싱 실패 (폴백)", sk_ax_impact="")


class UrgentMonitor:
    """모든 실시간 소스를 비동기로 병렬 감시.

    FastFilter → 경량 분류기 → LLM 검증 (urgent만) → Slack
    """

    def __init__(self) -> None:
        self.fast_filter = FastFilter()
        self.verifier = UrgentVerifier()
        self.limit_guard = DailyLimitGuard()
        self._rss_seen: dict[str, set[str]] = {k: set() for k in _RSS_SOURCES}
        self._page_hashes: dict[str, str] = {}
        self._slack_url = os.getenv("SLACK_WEBHOOK_URL", "")

    async def run(self) -> None:
        log.info("Track A UrgentMonitor 시작")
        await asyncio.gather(
            self._poll_all_rss(),
            self._monitor_all_pages(),
            self._poll_dart_rss(),
        )

    # ── 소스 폴링 ─────────────────────────────────────────────────

    async def _poll_all_rss(self) -> None:
        while True:
            for source_name, feed_url in _RSS_SOURCES.items():
                try:
                    articles = await self._fetch_rss(source_name, feed_url)
                    await self._handle_articles(articles)
                except Exception as e:
                    log.error("RSS 폴링 오류 | source=%s error=%s", source_name, e)
            await asyncio.sleep(RSS_INTERVAL)

    async def _poll_dart_rss(self) -> None:
        """DART 공시 RSS 폴링 (실시간 대안 — WebSocket 미지원)."""
        dart_url = "https://opendart.fss.or.kr/api/rss.xml"
        seen: set[str] = set()
        while True:
            try:
                feed = feedparser.parse(dart_url)
                for entry in feed.entries:
                    url = entry.get("link", "")
                    title = entry.get("title", "")
                    if url in seen or not any(kw in title for kw in PEER_KEYWORDS):
                        continue
                    seen.add(url)
                    await self._handle_articles(
                        [
                            RawArticle(
                                url=url,
                                title=title,
                                content=entry.get("summary", title),
                                source_tier=1,
                                source_name="dart",
                                credibility_score=SOURCE_CREDIBILITY["dart"],
                                peer_id=_guess_peer(title),
                            )
                        ]
                    )
            except Exception as e:
                log.error("DART RSS 폴링 오류 | error=%s", e)
            await asyncio.sleep(RSS_INTERVAL)

    async def _monitor_all_pages(self) -> None:
        while True:
            for source_name, url in _PAGE_MONITORS.items():
                try:
                    changed = await self._check_page_changed(source_name, url)
                    if changed:
                        log.info("페이지 변경 감지 | source=%s", source_name)
                except Exception as e:
                    log.error("페이지 모니터 오류 | source=%s error=%s", source_name, e)
            await asyncio.sleep(PAGE_INTERVAL)

    # ── 파이프라인 핵심 ───────────────────────────────────────────

    async def _handle_articles(self, articles: list[RawArticle]) -> None:
        if not articles:
            return
        results = self.fast_filter.filter_and_score(articles)
        # urgent만 LLM 검증, notable은 DB 저장 (Track B에서 처리)
        for r in results:
            if r.importance_hint == "urgent":
                await self._verify_and_alert(r)
            else:
                log.debug("notable/reference 감지 | title=%s", r.article.title)

    async def _verify_and_alert(self, result: FilterResult) -> None:
        article = result.article
        log.info("LLM 검증 시작 | score=%d title=%s", result.score, article.title)

        verification = await self.verifier.verify(article)

        if verification.confirmed:
            log.warning("🚨 URGENT 확정 | title=%s", article.title)
            await self._send_slack(article, result.score, verification)
        else:
            log.info(
                "LLM 검증 불통과 — 알림 스킵 | title=%s reason=%s",
                article.title,
                verification.reason,
            )

    # ── Slack 알림 ────────────────────────────────────────────────

    async def _send_slack(
        self, article: RawArticle, score: int, verification: VerifyResult
    ) -> None:
        if not self._slack_url:
            return
        blocks = _build_slack_blocks(article, score, verification)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(self._slack_url, json={"blocks": blocks})
        except Exception as e:
            log.error("Slack 알림 전송 실패 | error=%s", e)

    # ── 유틸 ──────────────────────────────────────────────────────

    async def _fetch_rss(self, source_name: str, feed_url: str) -> list[RawArticle]:
        feed = feedparser.parse(feed_url)
        articles = []
        seen = self._rss_seen[source_name]
        for entry in feed.entries:
            url = entry.get("link", "")
            title = entry.get("title", "")
            if url in seen or not any(kw.lower() in title.lower() for kw in PEER_KEYWORDS):
                continue
            seen.add(url)
            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    content=entry.get("summary", ""),
                    source_tier=2,
                    source_name=source_name,
                    credibility_score=SOURCE_CREDIBILITY.get(source_name, 0.65),
                    peer_id=_guess_peer(title),
                )
            )
        return articles

    async def _check_page_changed(self, source_name: str, url: str) -> bool:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={"User-Agent": "AXIS-Monitor/1.0"})
        h = hashlib.md5(resp.text.encode()).hexdigest()
        prev = self._page_hashes.get(source_name)
        self._page_hashes[source_name] = h
        return prev is not None and prev != h


# ── 헬퍼 ──────────────────────────────────────────────────────────


def _guess_peer(title: str) -> Optional[str]:
    if "삼성SDS" in title or "Samsung SDS" in title:
        return "samsung_sds"
    if "LG CNS" in title or "엘지씨엔에스" in title:
        return "lg_cns"
    return None


def _build_slack_blocks(article: RawArticle, score: int, v: VerifyResult) -> list[dict]:
    """Slack Block Kit 메시지 구성."""
    peer_label = {"samsung_sds": "삼성SDS", "lg_cns": "LG CNS"}.get(
        article.peer_id or "", "알 수 없음"
    )
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🚨 긴급 신호 감지 — {peer_label}"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{article.title}*"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*출처*\n{article.source_name}"},
                {"type": "mrkdwn", "text": f"*중요도 점수*\n{score}점"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*판단 근거*\n{v.reason}"},
        },
    ]
    if v.sk_ax_impact:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"✨ *SK AX 시사점 (AI 초안)*\n{v.sk_ax_impact}",
                },
            }
        )
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "원문 보기"},
                    "url": article.url,
                    "style": "primary",
                }
            ],
        }
    )
    return blocks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(UrgentMonitor().run())
