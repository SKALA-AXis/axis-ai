# 작성일: 2026-04-28
# 작성자: 최종민
# 변경이력:
#   2026-04-28 최종민 — Qdrant Cloud 전환과 함께 vector_index 노드 도입
#   2026-05-06 박지원 — 크롤러 구현·전처리 agent 수정에 맞춰 변경
#   2026-05-18 심유정 — 카드뉴스 agent 플로우 정비
"""카드뉴스 대표 기사를 BGE-M3로 임베딩해 Qdrant axis_main 컬렉션에 삽입.

카드 validation 통과분만 인덱싱한다.
페이로드는 §2.5 규약대로 메타데이터만 — 원문 본문 저장 금지.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from qdrant_client import models as qm

from src.db.article_store import get_articles_by_ids, update_classification
from src.db.qdrant_client import (
    COLLECTION_MAIN,
    ensure_collections,
    get_qdrant_client,
)
from src.rag.embedder import embed_text

log = logging.getLogger(__name__)


_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-00000000a715")  # AXIS 고정 네임스페이스


def _vector_id(rdb_id: int) -> str:
    """raw_articles.id 기반 결정적 UUID — 재인덱싱 시 같은 포인트 갱신."""
    return str(uuid.uuid5(_NAMESPACE, str(rdb_id)))


def _published_at_ts(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, datetime):
        return int(value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp())
    return 0


def index_card(card: dict[str, Any]) -> Optional[str]:
    """카드 1건을 Qdrant axis_main에 upsert. 반환값은 vector_id (실패 시 None)."""
    rep_id = card.get("representative_id")
    if not rep_id:
        log.warning("vector index 스킵 — representative_id 없음 | card=%s", card.get("id"))
        return None

    rep_articles = get_articles_by_ids([rep_id])
    if not rep_articles:
        log.warning("vector index 스킵 — 대표 기사 조회 실패 | rep_id=%s", rep_id)
        return None
    rep = rep_articles[0]

    # 임베딩 텍스트: 제목 + 3줄 요약 (본문은 저장 안 함)
    summary = " ".join(card.get("summary_lines", []))
    embed_input = f"{card.get('title', '')}\n{summary}".strip()
    if not embed_input:
        return None

    vec = embed_text(embed_input, mode="both")
    point_id = _vector_id(rep_id)

    sparse = vec["sparse"]
    sparse_vec = qm.SparseVector(
        indices=[int(i) for i in sparse["indices"]],
        values=[float(v) for v in sparse["values"]],
    )

    payload = {
        "rdb_id": rep_id,
        "card_news_id": card.get("id"),
        "company": card.get("company") or card.get("peer_id"),
        "event_type": card.get("event_type"),
        "sector": card.get("sector"),
        "exposure_score": card.get("exposure_score", 0.0),
        "exposure_band": card.get("exposure_band"),
        "published_at": _published_at_ts(rep.get("published_at")),
        "cluster_id": card.get("cluster_id"),
        "source_name": rep.get("source_name"),
        "title": card.get("title", "")[:500],
        "summary": summary[:1000],
    }

    client = get_qdrant_client()
    ensure_collections(client)
    client.upsert(
        collection_name=COLLECTION_MAIN,
        points=[
            qm.PointStruct(
                id=point_id,
                vector={"dense": vec["dense"], "sparse": sparse_vec},
                payload=payload,
            )
        ],
    )

    update_classification(
        article_id=rep_id,
        importance=card.get("importance", "low"),
        importance_score=card.get("importance_score", 0.0),
        qdrant_vector_id=point_id,
    )

    log.info(
        "Qdrant 인덱싱 완료 | card=%s rep=%d point=%s sector=%s band=%s",
        card.get("id"),
        rep_id,
        point_id,
        card.get("sector"),
        card.get("exposure_band"),
    )
    return point_id
