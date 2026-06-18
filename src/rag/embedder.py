# 작성일: 2026-04-21
# 작성자: 최종민
# 변경이력:
#   2026-04-21 최종민 — axis-ai 베이스라인으로 임베더 도입, ruff 포맷 적용
#   2026-04-28 박지원 — 크롤러 로직 개선 작업 반영 및 해당 변경 revert
"""BGE-M3 임베딩 — Dense + Sparse 원샷 생성.

메모리 spike 방어 정책:
  · 모델은 startup 시 preload (axis-ai/src/api/router.py lifespan) 해서 cold-start spike 제거.
  · encode 시 max_length 를 EMBED_MAX_LENGTH (default 2048) 로 명시 → BGE-M3 default
    8192 대비 attention matrix O(len²) 메모리 16x 감소.
  · 실측 (raw_articles 200건 sample): 2048 token 으로 85.5% 통과, 평균 손실 183 token.
    dedup 의 의미 시그널은 article 앞부분 (제목 + 리드 + 본문 상위) 에 집중되어 있어
    후반부 truncate 영향 미미.
  · 환경변수로 override 가능: EMBED_MAX_LENGTH.
"""

import logging
import os

log = logging.getLogger(__name__)

_model = None
_model_loaded = False

EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "2048"))


def get_embedder():
    """BGE-M3 모델 싱글톤 로더 (첫 호출 시 모델 로드)."""
    global _model, _model_loaded
    if _model is None:
        try:
            from FlagEmbedding import BGEM3FlagModel

            log.info("BGE-M3 모델 로딩 중... (max_length=%d)", EMBED_MAX_LENGTH)
            _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            _model_loaded = True
            log.info("BGE-M3 모델 로딩 완료")
        except Exception as e:
            log.error("BGE-M3 모델 로딩 실패: %s", e)
            raise
    return _model


def is_loaded() -> bool:
    """모델이 메모리에 올라와 있는지 — /health 응답용."""
    return _model_loaded


def preload_embedder() -> bool:
    """서버 startup 시 호출 — lazy load 로 인한 첫 cycle 의 메모리 spike 를 제거.

    실패해도 서버 startup 자체를 막진 않는다 (첫 호출 시 다시 시도).
    """
    try:
        get_embedder()
        return True
    except Exception as e:
        log.warning("BGE-M3 preload 실패 (첫 호출 시 재시도): %s", e)
        return False


def embed_text(text: str, mode: str = "both") -> dict:
    """BGE-M3로 텍스트를 임베딩한다.

    Args:
        text: 임베딩할 텍스트.
        mode: 'dense' | 'sparse' | 'both'. 기본값 'both'.

    Returns:
        {'dense': list, 'sparse': dict} 형태의 벡터.

    Raises:
        RuntimeError: 모델 로딩 실패 시.
    """
    model = get_embedder()
    result = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        max_length=EMBED_MAX_LENGTH,
    )
    output = {}
    if mode in ("dense", "both"):
        output["dense"] = result["dense_vecs"][0].tolist()
    if mode in ("sparse", "both"):
        lexical = result["lexical_weights"][0]
        output["sparse"] = {
            "indices": list(lexical.keys()),
            "values": list(lexical.values()),
        }
    return output
