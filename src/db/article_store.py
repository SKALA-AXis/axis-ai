"""raw_articles / issue_cards 테이블 저장·조회·업데이트 레이어."""

import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text

from src.crawler.base import RawArticle
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

_INSERT_SQL = text("""
    INSERT INTO raw_articles (
        peer_id, source_tier, source_name, title, content, url,
        published_at, collected_at, credibility_score, processing_status, metadata
    ) VALUES (
        :peer_id, :source_tier, :source_name, :title, :content, :url,
        :published_at, :collected_at, :credibility_score, 'RAW', CAST(:metadata AS jsonb)
    )
    ON CONFLICT (url) DO NOTHING
    RETURNING id
""")


def save_articles(articles: list[RawArticle]) -> int:
    """RawArticle 목록을 raw_articles 테이블에 저장. 중복 URL은 스킵.

    Returns:
        실제 삽입된 건수
    """
    if not articles:
        return 0

    inserted = 0
    with SessionLocal() as db:
        for article in articles:
            if not _is_valid(article):
                continue
            try:
                result = db.execute(
                    _INSERT_SQL,
                    {
                        "peer_id": article.peer_id,
                        "source_tier": article.source_tier,
                        "source_name": article.source_name,
                        "title": article.title[:500],
                        "content": article.content[:10_000] if article.content else "",
                        "url": article.url,
                        "published_at": article.published_at or article.collected_at,
                        "collected_at": article.collected_at,
                        "credibility_score": article.credibility_score,
                        "metadata": _metadata_json(article),
                    },
                )
                if result.fetchone():
                    inserted += 1
            except Exception as e:
                log.error("raw_articles 저장 실패 | url=%s error=%s", article.url, e)
                db.rollback()
        db.commit()

    log.info(
        "raw_articles 저장 완료 | total=%d inserted=%d skipped=%d",
        len(articles),
        inserted,
        len(articles) - inserted,
    )
    return inserted


# ──────────────────────────────────────────────────────────────
# 조회
# ──────────────────────────────────────────────────────────────

def get_articles_by_ids(ids: list[int]) -> list[dict[str, Any]]:
    """raw_articles 테이블에서 ID 목록으로 기사를 조회한다."""
    if not ids:
        return []
    with SessionLocal() as db:
        rows = db.execute(
            text("""
                SELECT id, peer_id, title, content, url,
                       credibility_score, source_name, published_at, metadata
                FROM raw_articles
                WHERE id = ANY(:ids)
                ORDER BY credibility_score DESC
            """),
            {"ids": ids},
        ).fetchall()
    return [dict(row._mapping) for row in rows]


# ──────────────────────────────────────────────────────────────
# 업데이트 — 파이프라인 각 단계에서 호출
# ──────────────────────────────────────────────────────────────

def update_cluster(
    article_id: int,
    cluster_id: int,
    is_representative: bool,
) -> None:
    """cluster_id, is_representative, processing_status를 업데이트한다."""
    status = "RAW" if is_representative else "CLUSTERED_DUPE"
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET cluster_id = :cluster_id,
                    is_representative = :is_rep,
                    processing_status = :status
                WHERE id = :id
            """),
            {
                "cluster_id": cluster_id,
                "is_rep": is_representative,
                "status": status,
                "id": article_id,
            },
        )
        db.commit()


def update_classification(
    article_id: int,
    importance: str,
    importance_score: float,
    qdrant_vector_id: Optional[str] = None,
) -> None:
    """중요도 분류 결과를 raw_articles에 반영한다."""
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET importance_level = :importance,
                    importance_score = :score,
                    processing_status = 'CLASSIFIED',
                    qdrant_vector_id = CAST(:qdrant_id AS uuid)
                WHERE id = :id
            """),
            {
                "importance": importance,
                "score": importance_score,
                "qdrant_id": qdrant_vector_id,
                "id": article_id,
            },
        )
        db.commit()


# ──────────────────────────────────────────────────────────────
# 이슈카드 저장
# ──────────────────────────────────────────────────────────────

_INSERT_ISSUE_CARD = text("""
    INSERT INTO issue_cards (
        id, peer_id, cluster_id, title, summary_lines,
        event_type, importance, importance_score,
        implication, sources, validation_pass, validation_sc_score
    ) VALUES (
        :id, :peer_id, :cluster_id, :title, :summary_lines,
        :event_type, :importance, :importance_score,
        CAST(:implication AS jsonb), CAST(:sources AS jsonb),
        :validation_pass, :validation_sc_score
    )
    ON CONFLICT (id) DO UPDATE SET
        implication = CAST(:implication AS jsonb),
        validation_pass = :validation_pass,
        validation_sc_score = :validation_sc_score
    RETURNING id
""")


def save_issue_card(card: dict[str, Any]) -> Optional[str]:
    """이슈 카드를 issue_cards 테이블에 저장한다.

    Returns:
        저장된 issue card ID, 실패 시 None
    """
    try:
        with SessionLocal() as db:
            result = db.execute(
                _INSERT_ISSUE_CARD,
                {
                    "id": card["id"],
                    "peer_id": card["peer_id"],
                    "cluster_id": card.get("cluster_id"),
                    "title": card["title"][:500],
                    "summary_lines": card.get("summary_lines", []),
                    "event_type": card.get("event_type", "tech"),
                    "importance": card.get("importance", "reference"),
                    "importance_score": card.get("importance_score", 0.0),
                    "implication": json.dumps(card.get("implication", {}), ensure_ascii=False),
                    "sources": json.dumps(card.get("sources", []), ensure_ascii=False),
                    "validation_pass": card.get("validation", {}).get("pass", False),
                    "validation_sc_score": card.get("validation", {}).get("sc_score", 0.0),
                },
            )
            row = result.fetchone()
            db.commit()
            if row:
                log.info("이슈카드 저장 완료 | id=%s", card["id"])
                return card["id"]
    except Exception as e:
        log.error("이슈카드 저장 실패 | id=%s error=%s", card.get("id"), e)
    return None


def _generate_card_id(peer_id: str) -> str:
    """IC-YYYYMMDD-NNN 형식의 이슈카드 ID를 생성한다."""
    date_str = datetime.now().strftime("%Y%m%d")
    with SessionLocal() as db:
        row = db.execute(
            text("SELECT COUNT(*) FROM issue_cards WHERE id LIKE :prefix"),
            {"prefix": f"IC-{date_str}-%"},
        ).fetchone()
        seq = (row[0] if row else 0) + 1
    return f"IC-{date_str}-{seq:03d}"


# ──────────────────────────────────────────────────────────────

def _is_valid(article: RawArticle) -> bool:
    """Gate 1: 최소 품질 필터."""
    if not article.url or not article.title:
        return False
    if not article.peer_id:
        return False
    if len(article.content or "") < 10 and len(article.title) < 5:
        return False
    return True


def _metadata_json(article: RawArticle) -> str:
    import json

    meta = dict(article.metadata)
    meta["url_hash"] = article.url_hash
    return json.dumps(meta, ensure_ascii=False)
