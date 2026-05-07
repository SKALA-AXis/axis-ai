"""단독 크롤러 실행 결과를 로컬 JSON으로 저장한다."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "crawler_results"


def save_crawler_results(
    articles: list[Any],
    source_name: str,
    output_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> Path:
    """크롤러 결과를 JSON 배열 파일로 저장한다."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_path = output_path / filename

    payload = [_to_dict(article) for article in articles]
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result_path


def _to_dict(article: Any) -> dict[str, Any]:
    if hasattr(article, "to_common_dict"):
        return article.to_common_dict()

    extra = dict(getattr(article, "extra", {}))
    metadata = dict(getattr(article, "metadata", {}))

    if metadata and not extra:
        extra = metadata

    published_at = getattr(article, "published_at", None)
    collected_at = getattr(article, "collected_at", None)

    data = {
        "id": getattr(article, "id", None),
        "source_type": getattr(article, "source_type", "news"),
        "source_name": getattr(article, "source_name", None),
        "publisher": getattr(article, "publisher", None),
        "title": getattr(article, "title", ""),
        "content": getattr(article, "content", None),
        "url": getattr(article, "url", ""),
        "url_hash": getattr(article, "url_hash", ""),
        "published_at": published_at.isoformat() if published_at else None,
        "collected_at": collected_at.isoformat() if collected_at else None,
        "company": list(getattr(article, "company", []) or []),
        "language": getattr(article, "language", "ko"),
        "content_type": getattr(article, "content_type", "html"),
        "crawl_status": getattr(article, "crawl_status", "success"),
        "error_message": getattr(article, "error_message", None),
        "source_tier": getattr(article, "source_tier", None),
        "credibility_score": getattr(article, "credibility_score", None),
        "extra": extra,
    }

    peer_id = getattr(article, "peer_id", None)
    if peer_id and not data["company"]:
        data["company"] = [peer_id]

    return data
