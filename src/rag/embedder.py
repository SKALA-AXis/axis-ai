"""BGE-M3 임베딩 — Dense + Sparse 원샷 생성"""
import logging

log = logging.getLogger(__name__)

_model = None


def get_embedder():
    """BGE-M3 모델 싱글톤 로더 (첫 호출 시 모델 로드)"""
    global _model
    if _model is None:
        try:
            from FlagEmbedding import BGEM3FlagModel

            log.info("BGE-M3 모델 로딩 중...")
            _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            log.info("BGE-M3 모델 로딩 완료")
        except Exception as e:
            log.error("BGE-M3 모델 로딩 실패: %s", e)
            raise
    return _model


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
    result = model.encode([text], return_dense=True, return_sparse=True)
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
