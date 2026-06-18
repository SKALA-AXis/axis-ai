# 작성일: 2026-04-21
# 작성자: 최종민
# 변경이력:
#   2026-04-21 최종민 — axis-ai 베이스라인 작성, Supabase
#   2026-05-26 박지원 — allaround 수정
#   2026-06-08 박진 — 챗봇 에이전트 및 assistant RAG 추가
import os
from urllib.parse import urlparse, urlunparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

COLLECTION_MAIN = "axis_main"
COLLECTION_DOCUMENTS = "axis_documents"
DENSE_DIM = 1024  # BGE-M3 dense dimension


def get_qdrant_url() -> str:
    """Return a REST/client URL, preserving QDRANT_PORT when host is URL-like."""
    if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
        parsed = urlparse(QDRANT_HOST)
        if parsed.port is not None:
            return QDRANT_HOST
        netloc = parsed.hostname or parsed.netloc
        if parsed.username or parsed.password:
            auth = parsed.username or ""
            if parsed.password:
                auth = f"{auth}:{parsed.password}"
            netloc = f"{auth}@{netloc}"
        return urlunparse(parsed._replace(netloc=f"{netloc}:{QDRANT_PORT}"))
    return f"http://{QDRANT_HOST}:{QDRANT_PORT}"


def get_qdrant_client() -> QdrantClient:
    # Qdrant Cloud는 url=https://....cloud.qdrant.io + API key. 로컬은 host/port (API key 미사용).
    if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
        return QdrantClient(url=get_qdrant_url(), api_key=QDRANT_API_KEY)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collections(client: QdrantClient) -> None:
    """axis_main, axis_documents 컬렉션이 없으면 생성"""
    existing = {c.name for c in client.get_collections().collections}

    for name in [COLLECTION_MAIN, COLLECTION_DOCUMENTS]:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams()),
                },
            )
