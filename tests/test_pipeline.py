"""파이프라인 단위 테스트"""
import pytest

from src.pipeline.ingestion_graph import IngestionState


def test_ingestion_state_structure():
    state: IngestionState = {
        "peer_ids": ["samsung_sds"],
        "trigger_type": "scheduled",
        "raw_article_ids": [],
        "credible_ids": [],
        "cluster_map": {},
        "representative_ids": [],
        "classified_clusters": [],
        "issue_cards": [],
        "implications": [],
        "validation_results": [],
        "errors": [],
        "human_review_flags": [],
    }
    assert state["peer_ids"] == ["samsung_sds"]
