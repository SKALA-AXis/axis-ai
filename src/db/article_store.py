"""raw_articles 테이블 저장 레이어."""

import logging

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
