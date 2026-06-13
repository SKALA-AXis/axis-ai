"""Qdrant 선례·유사 카드 검색 어댑터.

AnalysisContextBuilder / ContextPackAssembler 가 생성자 인자 ``qdrant_search`` 로
주입받는 어댑터의 실제 구현. 두 빌더는 어댑터가 없거나(미주입) 예외를 던지면
DB-only fallback(peer + event_type + 시간차) 으로 내려간다 — 그 계약을 그대로 따른다.

- find_precedents: axis_main 에서 임베딩 유사 + (회사 ∈ peers) + (published_at 이
  min_days_since 일 이상 과거) 인 카드를 찾는다. event_type 이 달라도 의미가 비슷한
  선례를 잡는 것이 DB fallback 대비 핵심 가치.
- search_by_bundle: Layer 3 RAG (similar_cards_rag) — 시간차 조건 없이 유사 카드 검색.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.analysis.models import AnalysisInputBundle, PrecedentCandidate, RetrievedCard

log = logging.getLogger(__name__)

_QUERY_TEXT_MAX_CHARS = 512
_SECONDS_PER_DAY = 86_400


class PrecedentSearchEmptyError(LookupError):
    """Qdrant 검색이 0건일 때 호출부의 DB fallback 을 유도하기 위한 신호.

    빌더의 계약상 어댑터가 빈 목록을 '반환'하면 fallback 없이 해당 레이어가
    비어 버린다. 예외를 던지면 빌더가 debug 로그 후 DB fallback 을 수행한다.
    """


class QdrantPrecedentSearch:
    """상태 없는(stateless) 어댑터 — 생성 비용 0, Qdrant 연결은 호출 시점에만."""

    def find_precedents(
        self,
        *,
        bundle: AnalysisInputBundle,
        peers: list[str],
        min_days_since: int,
        top_k: int,
    ) -> list[PrecedentCandidate]:
        """현재 이슈와 의미가 유사한 과거 선례 카드를 찾는다.

        Args:
            bundle: 분석 대상 입력 묶음 (쿼리 텍스트 소스).
            peers: 후보를 한정할 회사 id 목록.
            min_days_since: 최소 시간차(일) — 같은 사건 재매칭 방지.
            top_k: 반환 상한.

        Returns:
            PrecedentCandidate 목록 (Qdrant RRF 점수를 cosine 필드에 담는다).

        Raises:
            PrecedentSearchEmptyError: 유효 결과 0건 — 호출부 DB fallback 신호.
        """
        query = _bundle_query_text(bundle)
        if not query:
            raise PrecedentSearchEmptyError("bundle 에 쿼리 텍스트 없음")

        from src.rag.hybrid_search import hybrid_search

        now_ts = int(time.time())
        cutoff_ts = now_ts - int(min_days_since) * _SECONDS_PER_DAY
        hits = hybrid_search(
            query,
            top_k=max(top_k * 2, top_k),  # 자기 클러스터·card_id 누락분 제외 여유분
            peer_ids=peers or None,
            published_before_ts=cutoff_ts,
        )

        out: list[PrecedentCandidate] = []
        for hit in hits:
            if _is_same_cluster(hit, bundle):
                continue
            card_id = str(hit.get("card_news_id") or "").strip()
            if not card_id:
                continue
            out.append(
                PrecedentCandidate(
                    card_id=card_id,
                    company_id=str(hit.get("company") or ""),
                    event_type=str(hit.get("event_type") or ""),
                    days_since=_days_since_ts(now_ts=now_ts, published_at=hit.get("published_at")),
                    cosine=float(hit.get("score") or 0.0),
                    headline=str(hit.get("title") or ""),
                )
            )
            if len(out) >= top_k:
                break
        if not out:
            raise PrecedentSearchEmptyError("qdrant 선례 0건 — DB fallback")
        return out

    def search_by_bundle(self, bundle: AnalysisInputBundle, *, top_k: int) -> list[RetrievedCard]:
        """Layer 3 RAG — 현재 이슈와 유사한 카드를 시간 제약 없이 검색한다."""
        query = _bundle_query_text(bundle)
        if not query:
            return []

        from src.rag.hybrid_search import hybrid_search

        hits = hybrid_search(query, top_k=max(top_k * 2, top_k))
        out: list[RetrievedCard] = []
        for hit in hits:
            if _is_same_cluster(hit, bundle):
                continue
            card_id = str(hit.get("card_news_id") or "").strip()
            if not card_id:
                continue
            out.append(
                RetrievedCard(
                    card_id=card_id,
                    cosine=float(hit.get("score") or 0.0),
                    headline=str(hit.get("title") or ""),
                    sector=str(hit.get("sector") or ""),
                    company_id=str(hit.get("company") or "") or None,
                )
            )
            if len(out) >= top_k:
                break
        return out


def _bundle_query_text(bundle: AnalysisInputBundle) -> str:
    """bundle 에서 임베딩 쿼리 텍스트를 추출한다.

    우선순위: evidence_snippets → facts → items(title/summary) → companies+sectors+event_type.
    뉴스 경로의 items 는 id 만 담는 경우가 있어 텍스트 필드를 다단계로 탐색한다.
    """
    chunks: list[str] = []

    def _collect(records: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in keys:
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
                    break
            if len(chunks) >= 5:
                return

    _collect(bundle.evidence_snippets or [], ("text", "snippet", "content", "statement"))
    if not chunks:
        _collect(bundle.facts or [], ("statement", "fact", "text", "content"))
    if not chunks:
        _collect(bundle.items or [], ("title", "summary", "headline"))
    if not chunks:
        keywords = [*(bundle.companies or []), *(bundle.sectors or [])]
        if bundle.event_type:
            keywords.append(bundle.event_type)
        chunks = [token for token in keywords if token]

    return " ".join(chunks)[:_QUERY_TEXT_MAX_CHARS].strip()


def _is_same_cluster(hit: dict[str, Any], bundle: AnalysisInputBundle) -> bool:
    if bundle.cluster_id is None:
        return False
    hit_cluster = hit.get("cluster_id")
    if hit_cluster is None:
        return False
    return str(hit_cluster) == str(bundle.cluster_id)


def _days_since_ts(*, now_ts: int, published_at: Any) -> int:
    try:
        ts = int(published_at)
    except (TypeError, ValueError):
        return 0
    if ts <= 0:
        return 0
    return max(0, (now_ts - ts) // _SECONDS_PER_DAY)
