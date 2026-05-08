"""Gate 2 소스 타입 기반 신뢰도 보정 에이전트.

이 단계에서는 source_type 기준으로 기본 credibility_score와 credibility_grade를 채운다.
신뢰도 점수만으로 기사를 탈락시키지는 않는다.
수집 실패, URL 없음, 제목 없음처럼 명백히 사용할 수 없는 데이터만 제외한다.
"""

import logging

from sqlalchemy import text

from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

DEFAULT_CREDIBILITY_SCORE = 0.50

SOURCE_TYPE_CREDIBILITY: dict[str, float] = {
    "dart": 1.00,
    "official": 0.90,
    "ir": 1.00,
    "securities_report": 0.80,
    "trend_report": 0.70,
    "news": 0.70,
    "market_data": 0.70,
    "job": 0.60,
    "search_trend": 0.55,
    "social": 0.40,
}

_ALLOWED_SOURCE_TYPES = set(SOURCE_TYPE_CREDIBILITY.keys())

_GRADE_MAP: list[tuple[float, str]] = [
    (0.85, "High"),
    (0.60, "Medium"),
    (0.40, "Low"),
    (0.00, "Unverified"),
]


class CredibilityAgent:
    """raw_articles의 source_type 기준 신뢰도 점수를 보정한다."""

    def filter(self, raw_article_ids: list[int]) -> tuple[list[int], list[int]]:
        """source_type 기준으로 credibility_score를 채우고 유효하지 않은 데이터만 제외한다.

        Args:
            raw_article_ids: 처리할 raw_articles ID 목록.

        Returns:
            (credible_ids, skipped_ids) 튜플.
        """
        if not raw_article_ids:
            return [], []

        credible_ids: list[int] = []
        skipped_ids: list[int] = []

        with SessionLocal() as db:
            rows = db.execute(
                text("""
                    SELECT
                        id,
                        url,
                        title,
                        source_type,
                        crawl_status,
                        credibility_score
                    FROM raw_articles
                    WHERE id = ANY(:ids)
                """),
                {"ids": raw_article_ids},
            ).fetchall()

            for row in rows:
                valid, reason = _validate_row(row)

                if not valid:
                    db.execute(
                        text("""
                            UPDATE raw_articles
                            SET processing_status = 'SKIPPED_INVALID_SOURCE',
                                error_message = :error_message
                            WHERE id = :id
                        """),
                        {
                            "error_message": reason,
                            "id": row.id,
                        },
                    )

                    skipped_ids.append(row.id)
                    log.debug("Gate 2 제외 | id=%d reason=%s", row.id, reason)
                    continue

                score = row.credibility_score

                if score is None:
                    score = compute_credibility_score(row.source_type)

                score = float(score)
                grade = _to_grade(score)

                db.execute(
                    text("""
                        UPDATE raw_articles
                        SET credibility_score = :score,
                            credibility_grade = :grade
                        WHERE id = :id
                    """),
                    {
                        "score": score,
                        "grade": grade,
                        "id": row.id,
                    },
                )

                credible_ids.append(row.id)

            db.commit()

        log.info(
            "Gate 2 신뢰도 보정 완료 | total=%d credible=%d skipped=%d",
            len(raw_article_ids),
            len(credible_ids),
            len(skipped_ids),
        )

        return credible_ids, skipped_ids


def compute_credibility_score(source_type: str | None) -> float:
    """source_type 기준으로 기본 신뢰도 점수를 반환한다."""
    if not source_type:
        return DEFAULT_CREDIBILITY_SCORE

    key = source_type.strip().lower()
    return SOURCE_TYPE_CREDIBILITY.get(key, DEFAULT_CREDIBILITY_SCORE)


def analyze_credibility_article(article: dict) -> tuple[dict, bool, str | None]:
    """JSON article에 Gate 2 신뢰도 결과를 붙인다.

    DB 기반 `CredibilityAgent.filter()`와 같은 validate/score 규칙을 로컬
    runner에서 재사용하기 위한 함수다.
    """

    valid, reason = _validate_mapping(article)
    item = dict(article)

    if not valid:
        item["processing_status"] = "SKIPPED_INVALID_SOURCE"
        item["skip_reason"] = reason
        return item, False, reason

    score = item.get("credibility_score")
    if score is None:
        score = compute_credibility_score(item.get("source_type"))

    score = float(score)
    item["credibility_score"] = score
    item["credibility_grade"] = _to_grade(score)
    return item, True, None


def _validate_row(row) -> tuple[bool, str | None]:
    if row.crawl_status == "failed":
        return False, "crawl_failed"

    if not row.url:
        return False, "empty_url"

    if not row.title:
        return False, "empty_title"

    if row.source_type and row.source_type not in _ALLOWED_SOURCE_TYPES:
        return False, f"invalid_source_type:{row.source_type}"

    return True, None


def _validate_mapping(article: dict) -> tuple[bool, str | None]:
    if article.get("crawl_status") == "failed":
        return False, "crawl_failed"

    if not article.get("url"):
        return False, "empty_url"

    if not article.get("title"):
        return False, "empty_title"

    source_type = article.get("source_type")
    if source_type and source_type not in _ALLOWED_SOURCE_TYPES:
        return False, f"invalid_source_type:{source_type}"

    return True, None


def _to_grade(score: float) -> str:
    for threshold, grade in _GRADE_MAP:
        if score >= threshold:
            return grade

    return "Unverified"
