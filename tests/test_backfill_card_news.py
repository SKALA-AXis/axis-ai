# 작성일: 2026-06-15
# 작성자: 박지원
# 변경이력:
#   2026-06-15 박지원 — 카드뉴스 인사이트 업데이트 보호 수정에 대한 테스트 추가
from __future__ import annotations

from scripts.backfill_card_news import _has_direct_frontend_ready


def test_has_direct_frontend_ready_accepts_llm_direct_blocks() -> None:
    card = {
        "implication": {
            "frontend_ready": {
                "source": "llm_direct",
                "key_implication": {
                    "sentence": "시장 변화가 구체화되고 있다.",
                    "evidence_sentence": "제품명과 적용 범위가 함께 제시됐다.",
                },
                "suggested_action": {
                    "sentence": "SK AX는 적용 범위가 넓어지는 흐름을 사업 방향에 반영해야 한다.",
                    "evidence_sentence": "업무 기능과 운영 범위가 함께 제시됐기 때문이다.",
                },
            }
        }
    }

    assert _has_direct_frontend_ready(card) is True


def test_has_direct_frontend_ready_rejects_structured_fallback() -> None:
    card = {
        "implication": {
            "frontend_ready": {
                "source": "structured_implication_fallback",
                "key_implication": {
                    "sentence": "피어 신호를 추적할 필요가 있습니다.",
                    "evidence_sentence": "기사 제목.",
                },
                "suggested_action": {
                    "sentence": "기회/위협 여부를 후속 데이터와 재평가해야 합니다.",
                    "evidence_sentence": "기회/위협 여부를 후속 데이터와 재평가해야 합니다.",
                },
            }
        }
    }

    assert _has_direct_frontend_ready(card) is False


def test_has_direct_frontend_ready_rejects_missing_frontend_ready() -> None:
    assert _has_direct_frontend_ready({"implication": {"frontend": {}}}) is False
