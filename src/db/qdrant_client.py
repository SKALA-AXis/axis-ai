import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

COLLECTION_MAIN = "axis_main"
COLLECTION_HISTORY = "axis_history"
COLLECTION_DOCUMENTS = "axis_documents"
DENSE_DIM = 1024  # BGE-M3 dense dimension


def get_qdrant_client() -> QdrantClient:
    # Qdrant Cloud는 url=https://....cloud.qdrant.io + API key. 로컬은 host/port (API key 미사용).
    if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
        return QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collections(client: QdrantClient) -> None:
    """axis_main, axis_history, axis_documents 컬렉션이 없으면 생성"""
    existing = {c.name for c in client.get_collections().collections}

    for name in [COLLECTION_MAIN, COLLECTION_HISTORY, COLLECTION_DOCUMENTS]:
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
