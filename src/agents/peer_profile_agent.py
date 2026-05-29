"""PeerProfileAgent — orchestrates peer profile snapshot generation."""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from src.db.postgres import SessionLocal
from src.services.profile_evidence_selector import ProfileEvidenceSelector
from src.services.profile_input_builder import ProfileInputBuilder
from src.services.profile_snapshot_summarizer import (
    PROFILE_SNAPSHOT_SCHEMA_VERSION,
    ProfileSnapshotSummarizer,
)
from src.services.profile_snapshot_validator import ProfileSnapshotValidator

log = logging.getLogger(__name__)


class PeerProfileAgent:
    """Build and optionally persist ``peer_companies.profile_snapshot``."""

    def __init__(
        self,
        *,
        input_builder: ProfileInputBuilder | None = None,
        evidence_selector: ProfileEvidenceSelector | None = None,
        summarizer: ProfileSnapshotSummarizer | None = None,
        validator: ProfileSnapshotValidator | None = None,
        session_factory: Any = SessionLocal,
    ) -> None:
        self.input_builder = input_builder or ProfileInputBuilder(session_factory=session_factory)
        self.evidence_selector = evidence_selector or ProfileEvidenceSelector()
        self.summarizer = summarizer or ProfileSnapshotSummarizer()
        self.validator = validator or ProfileSnapshotValidator()
        self._session_factory = session_factory

    def build_snapshot(self, peer_id: str) -> dict[str, Any]:
        input_pack = self.input_builder.build(peer_id)
        selected_pack = self.evidence_selector.select(input_pack)
        snapshot = self.summarizer.summarize(selected_pack)
        validation = self.validator.validate(snapshot, selected_pack)
        if not validation["is_valid"]:
            log.warning(
                "profile snapshot validation failed | peer_id=%s errors=%s",
                peer_id,
                validation["errors"],
            )
        return snapshot

    def build_and_save(
        self,
        peer_id: str,
        *,
        version: str = PROFILE_SNAPSHOT_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        snapshot = self.build_snapshot(peer_id)
        self.save_snapshot(peer_id=peer_id, snapshot=snapshot, version=version)
        return snapshot

    def save_snapshot(
        self,
        *,
        peer_id: str,
        snapshot: dict[str, Any],
        version: str = PROFILE_SNAPSHOT_SCHEMA_VERSION,
    ) -> None:
        with self._session_factory() as db:
            db.execute(
                text(
                    """
                    UPDATE peer_companies
                       SET profile_snapshot = CAST(:snapshot AS jsonb),
                           profile_snapshot_version = :version,
                           profile_snapshot_generated_at = NOW()
                     WHERE id = :peer_id
                    """
                ),
                {
                    "peer_id": peer_id,
                    "snapshot": json.dumps(snapshot, ensure_ascii=False, default=str),
                    "version": version,
                },
            )
            db.commit()


__all__ = ["PeerProfileAgent"]
