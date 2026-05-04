"""단독 크롤러 실행 결과를 로컬 JSONL로 저장한다."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.crawler.parsers.quality import attach_quality

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "crawler_results"


def save_crawler_results(
    articles: list[Any],
    source_name: str,
    peer_aliases: dict[str, list[str]] | None = None,
    output_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> Path:
    """품질 KPI를 붙인 뒤 JSONL 파일로 저장한다."""
    attach_quality(articles, peer_aliases or {})

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    result_path = output_path / filename

    with result_path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(_to_dict(article), ensure_ascii=False) + "\n")
    return result_path


def _to_dict(article: Any) -> dict[str, Any]:
    if hasattr(article, "to_common_dict"):
        data = article.to_common_dict()
        quality = data.get("extra", {}).get("quality", {})
    else:
        metadata = dict(getattr(article, "metadata", {}))
        quality = metadata.get("quality", {})
        data = {
            "url": article.url,
            "title": article.title,
            "content": article.content,
            "source_name": article.source_name,
            "peer_id": article.peer_id,
            "published_at": article.published_at.isoformat()
            if article.published_at
            else None,
            "collected_at": article.collected_at.isoformat(),
            "metadata": metadata,
        }
    data["quality"] = quality
    return data
