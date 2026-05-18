"""이슈 카드 생성 에이전트 — 뉴스 요약 결과 기반 카드 생성."""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.db.article_store import _generate_card_id, get_articles_by_ids

log = logging.getLogger(__name__)

_llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_completion_tokens=1024)
_PROMPT_VERSION = "ic-v3.0"  # evidence_agent.PROMPT_VERSION 과 통일

_ISSUE_CARD_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 기사들을 종합하여 이슈 카드를 작성해주세요.
여러 기사가 있을 경우 교차 검증하여 가장 신뢰도 높은 사실만 포함하세요.

## 기사 정보
{articles_text}

## 작성 규칙
- 제목: 핵심 사실을 담은 한 문장 (40자 이내)
- 3줄 요약: 반드시 3줄, 각 줄은 "1.", "2.", "3."으로 시작
- 출처 목록: 사용한 기사의 title, source_name, url 포함

## 이벤트 타입 (하나만 선택, 가장 두드러진 성격 기준)
- partnership: 기업간 협력·MOU·공동사업·파운드리 제공계약
- ma: 인수·합병·지분 인수·투자 유치
- personnel: 채용·인사·임원 선임·조직 개편
- tech: 신기술·신제품·플랫폼 출시·기술 실증
- regulation: 법규·가이드라인·정부 정책
- new_biz: 신사업 진출·수주·실적 발표·시장 확장

복수 해당 시 본문이 가장 크게 다루는 측면을 선택.
애매하면 tech 대신 personnel/new_biz/partnership 우선.

다음 JSON 형식으로만 응답하세요 (추가 텍스트 금지):
{{
  "title": "이슈 제목",
  "summary_lines": ["1. ...", "2. ...", "3. ..."],
  "event_type": "tech",
  "sources": [
    {{"index": 1, "title": "...", "source_name": "...", "url": "..."}}
  ]
}}"""


class IssueCardAgent:
    """뉴스 요약 결과를 우선 사용하고, 실패 시 기존 GPT 카드 생성으로 fallback한다."""

    def generate(
        self,
        cluster_id: int,
        representative_id: int,
        company: str,
        classification: dict[str, Any],
        cluster_article_ids: list[int] | None = None,
        peer_id: str = "",
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """이슈 카드를 생성한다.

        Args:
            cluster_id: 클러스터 ID.
            representative_id: 대표 기사 ID (가장 신뢰도 높은 기사).
            company: 회사 ID.
            classification: ClassificationAgent 결과.
            cluster_article_ids: 클러스터 내 전체 기사 ID 목록 (없으면 대표 기사만 사용).
            summary: PeerNewsSummaryAgent가 만든 클러스터 사실 요약.

        Returns:
            card_news dict (저장 전 validation 없는 상태) — DB 의 card_news 테이블 row 와 1:1.
        """
        company = company or peer_id

        # 카드 요약/출처 모두 클러스터 전체 기사를 기준으로 한다.
        article_ids = _build_cluster_fetch_ids(representative_id, cluster_article_ids)
        articles = _order_articles(get_articles_by_ids(article_ids), article_ids)
        if not articles:
            return {}

        summary_card = _card_from_summary(
            summary=summary or {},
            articles=articles,
            company=company,
            cluster_id=cluster_id,
            representative_id=representative_id,
            classification=classification,
        )
        if summary_card:
            log.info(
                "이슈카드 생성 완료(summary) | id=%s sector=%s band=%s event=%s sources=%d",
                summary_card["id"],
                summary_card["sector"],
                summary_card["exposure_band"],
                summary_card["event_type"],
                len(articles),
            )
            return summary_card

        articles_text = _format_articles(articles)
        prompt = _ISSUE_CARD_PROMPT.replace("{articles_text}", articles_text)

        try:
            from src.observability import tracing_config

            response = _llm.invoke(
                prompt,
                config=tracing_config(
                    agent="IssueCardAgent",
                    prompt_version=_PROMPT_VERSION,
                    company=company,
                    cluster_id=cluster_id,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            card_data = _parse_json(content)

            card = {
                "id": _generate_card_id(company),
                "company": company,
                "cluster_id": cluster_id,
                "representative_id": representative_id,
                "title": card_data.get("title", articles[0]["title"][:100]),
                "summary_lines": card_data.get("summary_lines", []),
                "event_type": card_data.get("event_type", classification.get("event_type", "tech")),
                # v3: 분류 결과의 sector·exposure 정보를 카드에 그대로 전파
                "sector": classification.get("sector", "other"),
                "sectors": classification.get("sectors", ["other"]),
                "exposure_score": classification.get("exposure_score", 0.0),
                "exposure_band": classification.get("exposure_band", "low"),
                "signals": classification.get("signals", {}),
                # 등급 자체는 v3에서 폐기되었으나, DB 컬럼 호환을 위해 노출도 밴드를 저장
                "importance": classification.get("importance", "low"),
                "importance_score": classification.get("importance_score", 0.0),
                "sources": _default_sources(articles),
            }

            log.info(
                "이슈카드 생성 완료 | id=%s sector=%s band=%s event=%s sources=%d",
                card["id"],
                card["sector"],
                card["exposure_band"],
                card["event_type"],
                len(articles),
            )
            return card

        except Exception as e:
            log.error("이슈카드 생성 실패 | cluster=%d error=%s", cluster_id, e)
            return {}


def _build_cluster_fetch_ids(
    representative_id: int, cluster_article_ids: list[int] | None
) -> list[int]:
    """대표 기사를 앞에 두고 클러스터 전체 ID 목록을 만든다."""
    if not cluster_article_ids:
        return [representative_id]
    others = [aid for aid in cluster_article_ids if aid != representative_id]
    return [representative_id, *others]


def _order_articles(
    articles: list[dict[str, Any]],
    ordered_ids: list[int],
) -> list[dict[str, Any]]:
    """DB 조회 결과를 대표기사 우선 순서로 되돌린다."""
    order = {article_id: index for index, article_id in enumerate(ordered_ids)}
    return sorted(articles, key=lambda article: order.get(int(article.get("id") or 0), len(order)))


def _card_from_summary(
    *,
    summary: dict[str, Any],
    articles: list[dict[str, Any]],
    company: str,
    cluster_id: int,
    representative_id: int,
    classification: dict[str, Any],
) -> dict[str, Any] | None:
    if not summary.get("is_valid_summary"):
        return None

    effective_company = _first_non_empty(summary.get("main_company"), company)
    summary_lines = _numbered_summary_lines(summary.get("fact_summary"))
    if not summary_lines:
        return None

    title = _first_non_empty(
        summary.get("headline"),
        summary.get("one_line_summary"),
        classification.get("title"),
        articles[0].get("title"),
    )

    return {
        "id": _generate_card_id(effective_company),
        "company": effective_company,
        "cluster_id": cluster_id,
        "representative_id": representative_id,
        "title": title[:100],
        "summary_lines": summary_lines,
        "event_type": classification.get("event_type", "tech"),
        "sector": classification.get("sector", "other"),
        "sectors": classification.get("sectors", ["other"]),
        "exposure_score": classification.get("exposure_score", 0.0),
        "exposure_band": classification.get("exposure_band", "low"),
        "signals": classification.get("signals", {}),
        "importance": classification.get("importance", "low"),
        "importance_score": classification.get("importance_score", 0.0),
        "sources": _default_sources(articles),
        "news_summary": summary,
    }


def _numbered_summary_lines(value: Any) -> list[str]:
    lines = [str(item).strip() for item in _list_value(value) if str(item).strip()]
    numbered: list[str] = []
    for index, line in enumerate(lines[:3], start=1):
        prefix = f"{index}."
        numbered.append(line if line.startswith(prefix) else f"{prefix} {line}")
    return numbered


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return "피어사 주요 뉴스"


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(
            f"[{i}] 제목: {a['title']}\n"
            f"    출처: {a['source_name']} | URL: {a['url']}\n"
            f"    내용: {' '.join((a.get('content') or '').split())}"
        )
    return "\n\n".join(lines)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _default_sources(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": i + 1,
            "title": a["title"],
            "source_name": a["source_name"],
            "url": a["url"],
        }
        for i, a in enumerate(articles)
    ]
