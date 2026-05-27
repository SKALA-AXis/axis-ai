"""Source profile inference for issue integration.

source_type 단일 매핑이 아니라 source_type, content_type, source_name, metadata,
parser section 신호를 함께 점수화해 source family와 scope를 추정한다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.analysis.models import AnalysisInputBundle
from src.services.issue_integration.lexicon import FAMILY_INDICATORS, SCOPE_INDICATORS


@dataclass(frozen=True, slots=True)
class SourceProfile:
    source_family: str
    source_type: str
    source_types: list[str]
    content_types: list[str]
    scope_type: str
    source_family_scores: dict[str, float] = field(default_factory=dict)
    scope_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_source_profile(input_bundle: AnalysisInputBundle) -> SourceProfile:
    source_types = _dedupe(
        [
            input_bundle.source_type,
            *[
                str(item.get("source_type") or "")
                for item in input_bundle.items
                if isinstance(item, dict)
            ],
        ]
    )
    content_types = _dedupe(
        [
            str(item.get("content_type") or "")
            for item in input_bundle.items
            if isinstance(item, dict)
        ]
    )
    corpus = _profile_corpus(input_bundle, source_types=source_types, content_types=content_types)
    family_scores = _score_labels(corpus, FAMILY_INDICATORS)
    scope_scores = _score_labels(corpus, SCOPE_INDICATORS)

    item_families = {
        _best_label(_score_labels(_profile_corpus_for_item(item), FAMILY_INDICATORS))
        for item in input_bundle.items
    }
    item_families.discard("unknown")
    if len(item_families) > 1:
        family = "mixed"
    else:
        family = _best_label(family_scores)
    if family == "unknown" and len(source_types) > 1:
        family = "mixed"

    scope = _best_label(scope_scores)
    if input_bundle.companies:
        scope = "peer_company"
    elif family == "trend":
        scope = "industry"
    elif family == "research":
        scope = "market" if scope == "unknown" else scope

    return SourceProfile(
        source_family=family,
        source_type=source_types[0] if source_types else input_bundle.source_type or "unknown",
        source_types=source_types,
        content_types=content_types,
        scope_type=scope,
        source_family_scores=family_scores,
        scope_scores=scope_scores,
    )


def _profile_corpus(
    input_bundle: AnalysisInputBundle,
    *,
    source_types: list[str],
    content_types: list[str],
) -> str:
    parts: list[str] = [*source_types, *content_types, *input_bundle.sectors]
    for item in input_bundle.items:
        if not isinstance(item, dict):
            continue
        parts.extend(
            [
                str(item.get("source_name") or ""),
                str(item.get("publisher") or ""),
                str(item.get("title") or ""),
                _json_text(item.get("metadata")),
                _parser_result_text(item),
            ]
        )
    return " ".join(part.lower() for part in parts if part)


def _profile_corpus_for_item(item: dict[str, Any]) -> str:
    return " ".join(
        part.lower()
        for part in (
            str(item.get("source_type") or ""),
            str(item.get("content_type") or ""),
            str(item.get("source_name") or ""),
            str(item.get("publisher") or ""),
            str(item.get("title") or ""),
            _json_text(item.get("metadata")),
            _parser_result_text(item),
        )
        if part
    )


def _score_labels(corpus: str, indicators: dict[str, tuple[str, ...]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, tokens in indicators.items():
        score = 0.0
        for token in tokens:
            normalized = token.lower()
            if normalized and normalized in corpus:
                score += 1.0
        scores[label] = score
    return scores


def _best_label(scores: dict[str, float]) -> str:
    non_zero = {label: score for label, score in scores.items() if score > 0}
    if not non_zero:
        return "unknown"
    best_score = max(non_zero.values())
    winners = sorted(label for label, score in non_zero.items() if score == best_score)
    return winners[0] if len(winners) == 1 else "mixed"


def _parser_result_text(item: dict[str, Any]) -> str:
    parser_result = item.get("parser_result")
    if not isinstance(parser_result, dict):
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            parser_result = metadata.get("parser_result")
    if not isinstance(parser_result, dict):
        return ""
    parts: list[str] = []
    sections = parser_result.get("sections")
    if isinstance(sections, dict):
        parts.extend(str(key) for key in sections)
    for key in ("document_type", "section_key", "report_type"):
        if parser_result.get(key):
            parts.append(str(parser_result[key]))
    return " ".join(parts)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not value:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["SourceProfile", "infer_source_profile"]
