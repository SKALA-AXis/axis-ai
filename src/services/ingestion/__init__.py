"""Ingestion/preprocessing support services.

이 패키지는 추후 개발 placeholder를 포함한다. 현재 운영 코드는 기존
`src/crawler`, `src/parsers`, `src/preprocessing`, `src/rag` 경로를 사용한다.
"""

from __future__ import annotations

from src.services.ingestion.classification_service import ClassificationService
from src.services.ingestion.crawler_service import CrawlerJob, CrawlerService
from src.services.ingestion.dedup_service import DedupService
from src.services.ingestion.embed_index_service import EmbedIndexService
from src.services.ingestion.evidence_builder import EvidenceBuilder, EvidenceService
from src.services.ingestion.financial_linker_service import FinancialLinkerService
from src.services.ingestion.parser_service import ParserService
from src.services.ingestion.relevance_service import RelevanceService

__all__ = [
    "ClassificationService",
    "CrawlerJob",
    "CrawlerService",
    "DedupService",
    "EmbedIndexService",
    "EvidenceBuilder",
    "EvidenceService",
    "FinancialLinkerService",
    "ParserService",
    "RelevanceService",
]
