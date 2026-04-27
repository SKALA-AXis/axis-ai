"""검증 체인 에이전트 (v3 §3.2, §5.6).

모든 이슈카드에 4가지 검증 정보를 자동 첨부한다.

  1. source_links     원문 URL + 출처명 (백업 텍스트는 본문 일부)
  2. provenance       어떤 raw_articles, LLM 모델·프롬프트 버전, 실행 시각
  3. financial_refs   재무 숫자 출처 — FinancialLinkerAgent PoC가 stub JSON으로 채움
  4. mbb_refs         MBB·커니 자료 자동 매칭 (W5에 채움 — 현재 빈 리스트)

검증 첨부율 100%가 v3 §8 KPI. 누락 시 human_review=True 플래그 부착.

기존 SC 검증(ValidationAgent)은 _deprecated/로 이관되었음.
"""

import logging
from datetime import datetime
from typing import Any

from src.agents.financial_linker_agent import FinancialLinkerAgent

log = logging.getLogger(__name__)

EVIDENCE_VERSION = "v3.0"
PROMPT_VERSION = "ic-v3.0"
LLM_MODEL = "gpt-4o"


class EvidenceAgent:
    """4종 검증 정보를 카드에 첨부하고 누락 시 human review 플래그를 부착."""

    def __init__(self, financial_linker: FinancialLinkerAgent | None = None) -> None:
        self.financial_linker = financial_linker or FinancialLinkerAgent()

    def attach(
        self,
        card: dict[str, Any],
        cluster_article_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """카드에 evidence_chain dict를 붙이고 (pass, reason)을 반환한다.

        Args:
            card: IssueCardAgent 결과 dict.
            cluster_article_ids: 클러스터 전체 raw_articles ID 목록.

        Returns:
            {pass, reason, missing} — pass=False 시 human_review 필요.
        """
        sources = card.get("sources", []) or []
        source_links = [
            {
                "title": s.get("title", ""),
                "source_name": s.get("source_name", ""),
                "url": s.get("url", ""),
                "credibility_score": s.get("credibility_score", 0.0),
            }
            for s in sources
            if s.get("url")
        ]

        raw_article_ids = list(cluster_article_ids or [])
        provenance: dict[str, Any] = {
            "raw_article_ids": raw_article_ids,
            "cluster_id": card.get("cluster_id"),
            "llm_model": LLM_MODEL,
            "prompt_version": PROMPT_VERSION,
            "evidence_version": EVIDENCE_VERSION,
            "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        # 재무 연결 (PoC) — stub JSON 기반. peer_financials 테이블 마이그레이션 후 DB로 교체.
        link_result = self.financial_linker.link(card)
        financial_refs: list[dict[str, Any]] = link_result.get("financial_refs", [])
        # W5 MBB·커니 크롤러가 채울 자리. 현재는 빈 컨테이너.
        mbb_refs: list[dict[str, Any]] = []

        evidence_chain = {
            "source_links": source_links,
            "provenance": provenance,
            "financial_refs": financial_refs,
            "mbb_refs": mbb_refs,
            "financial_link": {
                "linked": link_result.get("linked", False),
                "reason": link_result.get("reason"),
                "segment": link_result.get("segment"),
                "highlights": link_result.get("highlights", []),
                "headcount_delta": link_result.get("headcount_delta"),
            },
        }

        missing: list[str] = []
        if not source_links:
            missing.append("source_links")
        if not provenance.get("raw_article_ids"):
            missing.append("provenance.raw_article_ids")
        # financial_refs / mbb_refs는 W5 이전엔 누락 허용 (warning만)

        passed = len(missing) == 0
        reason = "ok" if passed else f"missing: {', '.join(missing)}"

        card["evidence_chain"] = evidence_chain
        # 카드 dict에 검증 결과 자체도 표면화 — DB 저장 시 implication JSONB로 직렬화
        card["validation"] = {
            "pass": passed,
            "sc_score": 1.0 if passed else 0.0,
            "reason": reason,
            "type": "evidence_attachment",
        }

        if not passed:
            log.warning(
                "검증 정보 누락 — human review 플래그 | card_id=%s missing=%s",
                card.get("id"),
                missing,
            )
        else:
            log.info(
                "검증 체인 첨부 완료 | card_id=%s sources=%d cluster=%d",
                card.get("id"),
                len(source_links),
                len(raw_article_ids),
            )

        return {"pass": passed, "reason": reason, "missing": missing}


__all__ = ["EvidenceAgent", "EVIDENCE_VERSION", "PROMPT_VERSION"]
