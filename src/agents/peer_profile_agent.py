"""PeerProfileAgent — orchestrates peer profile snapshot generation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from sqlalchemy import text

from src.db.postgres import SessionLocal
from src.llm import LLMSpec, build_chat_llm
from src.services.profile_evidence_selector import ProfileEvidenceSelector
from src.services.profile_input_builder import ProfileInputBuilder
from src.services.profile_snapshot_summarizer import (
    PROFILE_SNAPSHOT_SCHEMA_VERSION,
    ProfileSnapshotSummarizer,
)
from src.services.profile_snapshot_validator import ProfileSnapshotValidator

log = logging.getLogger(__name__)

_PROFILE_LLM_MODEL = os.getenv("PROFILE_AGENT_LLM_MODEL", "gpt-4o-mini")
_PROFILE_LLM_TEMPERATURE = float(os.getenv("PROFILE_AGENT_LLM_TEMPERATURE", "0.1"))
_PROFILE_LLM_MAX_COMPLETION_TOKENS = int(
    os.getenv("PROFILE_AGENT_LLM_MAX_COMPLETION_TOKENS", "3000")
)


class PeerProfileAgent:
    """Build and optionally persist ``peer_companies.profile_snapshot``."""

    def __init__(
        self,
        *,
        input_builder: ProfileInputBuilder | None = None,
        evidence_selector: ProfileEvidenceSelector | None = None,
        summarizer: ProfileSnapshotSummarizer | None = None,
        validator: ProfileSnapshotValidator | None = None,
        llm: Any | None = None,
        session_factory: Any = SessionLocal,
    ) -> None:
        self.input_builder = input_builder or ProfileInputBuilder(session_factory=session_factory)
        self.evidence_selector = evidence_selector or ProfileEvidenceSelector()
        self.summarizer = summarizer or ProfileSnapshotSummarizer()
        self.validator = validator or ProfileSnapshotValidator()
        self._llm = llm
        self._session_factory = session_factory

    def build_snapshot(self, peer_id: str, *, use_llm: bool = False) -> dict[str, Any]:
        input_pack = self.input_builder.build(peer_id)
        selected_pack = self.evidence_selector.select(input_pack)
        summarizer = self._summarizer_for(use_llm=use_llm)
        snapshot = summarizer.summarize(selected_pack)
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
        use_llm: bool = False,
        version: str = PROFILE_SNAPSHOT_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        snapshot = self.build_snapshot(peer_id, use_llm=use_llm)
        self.save_snapshot(peer_id=peer_id, snapshot=snapshot, version=version)
        return snapshot

    def _summarizer_for(self, *, use_llm: bool) -> ProfileSnapshotSummarizer:
        if not use_llm:
            return self.summarizer

        if getattr(self.summarizer, "_llm", None) is not None:
            return self.summarizer

        return ProfileSnapshotSummarizer(llm=self._get_llm())

    def _get_llm(self) -> Any:
        if self._llm is None:
            # env 로 모델 지정 가능 → gpt-5 라도 reasoning_effort 미전달(기존 동작) 위해 None.
            self._llm = build_chat_llm(
                LLMSpec(
                    model=_PROFILE_LLM_MODEL,
                    temperature=_PROFILE_LLM_TEMPERATURE,
                    max_tokens=_PROFILE_LLM_MAX_COMPLETION_TOKENS,
                    reasoning_effort=None,
                )
            )
        return self._llm

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
