"""Compatibility helpers for Qdrant servers that predate query_points.

The cluster currently runs Qdrant 1.9.x while local dependencies may install a
newer qdrant-client. Newer client-side ``query_points`` requests hit endpoints
that the older server does not expose, so RAG search needs a REST fallback.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.db.qdrant_client import QDRANT_API_KEY, get_qdrant_url

RRF_K = 60
DEFAULT_TIMEOUT_SECONDS = 20.0


def legacy_rrf_search(
    *,
    collection_name: str,
    dense_vector: list[float],
    sparse_vector: dict[str, Any],
    limit: int,
    prefetch_limit: int,
    query_filter: Any | None = None,
) -> list[dict[str, Any]]:
    """Run dense+sparse searches through the Qdrant 1.9 REST API and RRF-merge."""
    dense_hits = _search_named_vector(
        collection_name=collection_name,
        vector_name="dense",
        vector=dense_vector,
        limit=prefetch_limit,
        query_filter=query_filter,
    )
    sparse_hits = _search_named_vector(
        collection_name=collection_name,
        vector_name="sparse",
        vector={
            "indices": [int(index) for index in sparse_vector.get("indices", [])],
            "values": [float(value) for value in sparse_vector.get("values", [])],
        },
        limit=prefetch_limit,
        query_filter=query_filter,
    )
    return _rrf_merge(dense_hits, sparse_hits, limit=limit)


def delete_by_filter(
    *,
    collection_name: str,
    query_filter: Any,
    wait: bool = True,
) -> dict[str, Any]:
    """Delete points through the REST API using a Qdrant filter selector."""
    filter_payload = _filter_to_json(query_filter)
    if not filter_payload:
        raise ValueError("query_filter must not be empty")

    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    url = f"{get_qdrant_url().rstrip('/')}/collections/{collection_name}/points/delete"
    response = httpx.post(
        url,
        params={"wait": str(wait).lower()},
        json={"filter": filter_payload},
        headers=headers,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"result": payload}


def _search_named_vector(
    *,
    collection_name: str,
    vector_name: str,
    vector: Any,
    limit: int,
    query_filter: Any | None,
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "vector": {"name": vector_name, "vector": vector},
        "limit": int(limit),
        "with_payload": True,
        "with_vector": False,
    }
    filter_payload = _filter_to_json(query_filter)
    if filter_payload:
        body["filter"] = filter_payload

    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    url = f"{get_qdrant_url().rstrip('/')}/collections/{collection_name}/points/search"
    response = httpx.post(
        url,
        json=body,
        headers=headers,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json().get("result") or []
    return [item for item in result if isinstance(item, dict)]


def _filter_to_json(query_filter: Any | None) -> dict[str, Any] | None:
    if query_filter is None:
        return None
    if isinstance(query_filter, dict):
        return query_filter
    if hasattr(query_filter, "model_dump"):
        return query_filter.model_dump(mode="json", exclude_none=True)
    if hasattr(query_filter, "dict"):
        return query_filter.dict(exclude_none=True)
    return None


def _rrf_merge(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            point_id = str(hit.get("id") or "")
            if not point_id:
                continue
            entry = merged.setdefault(
                point_id,
                {
                    "id": point_id,
                    "payload": hit.get("payload") or {},
                    "score": 0.0,
                },
            )
            entry["score"] = float(entry["score"]) + 1.0 / (RRF_K + rank)
            if not entry.get("payload") and hit.get("payload"):
                entry["payload"] = hit["payload"]

    return sorted(merged.values(), key=lambda item: float(item.get("score") or 0), reverse=True)[
        :limit
    ]


__all__ = ["delete_by_filter", "legacy_rrf_search"]
