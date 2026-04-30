"""Gate 2 신뢰도 분류 에이전트.

크롤러는 신뢰도를 계산하지 않는다. credibility_score가 아직 비어 있는
기사들은 이 단계에서 탈락시키지 않고 통과시킨다. 실제 점수 계산 로직은
후속 CredibilityAgent 구현에서 source_name/url/metadata 기반으로 채운다.
"""

import logging

from sqlalchemy import text

from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

# Gate 2 기준: 계산된 credibility_score < 0.5 → SKIPPED_CREDIBILITY
CREDIBILITY_THRESHOLD = 0.5

# credibility_score → grade 매핑 (schema.sql credibility_grade 컬럼)
_GRADE_MAP: list[tuple[float, str]] = [
    (0.85, "High"),
    (0.60, "Medium"),
    (0.40, "Low"),
    (0.00, "Unverified"),
]


class CredibilityAgent:
    """raw_articles의 credibility_score로 Gate 2 신뢰도 필터 적용."""

    def filter(self, raw_article_ids: list[int]) -> tuple[list[int], list[int]]:
        """credibility_score < CREDIBILITY_THRESHOLD 기사를 탈락시킨다.

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
                text("SELECT id, credibility_score FROM raw_articles WHERE id = ANY(:ids)"),
                {"ids": raw_article_ids},
            ).fetchall()

            for row in rows:
                if row.credibility_score is None:
                    credible_ids.append(row.id)
                    log.debug("Gate 2 보류/통과 | id=%d score=None", row.id)
                    continue

                score: float = row.credibility_score
                grade = _to_grade(score)
                passed = score >= CREDIBILITY_THRESHOLD

                db.execute(
                    text("""
                        UPDATE raw_articles
                        SET credibility_grade = :grade,
                            processing_status = CASE
                                WHEN :passed THEN processing_status
                                ELSE 'SKIPPED_CREDIBILITY'
                            END
                        WHERE id = :id
                    """),
                    {"grade": grade, "passed": passed, "id": row.id},
                )

                if passed:
                    credible_ids.append(row.id)
                else:
                    skipped_ids.append(row.id)
                    log.debug("Gate 2 탈락 | id=%d score=%.2f grade=%s", row.id, score, grade)

            db.commit()

        log.info(
            "Gate 2 신뢰도 필터 완료 | total=%d credible=%d skipped=%d",
            len(raw_article_ids),
            len(credible_ids),
            len(skipped_ids),
        )
        return credible_ids, skipped_ids


def _to_grade(score: float) -> str:
    for threshold, grade in _GRADE_MAP:
        if score >= threshold:
            return grade
    return "Unverified"
