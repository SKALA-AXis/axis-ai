"""W4-0 — data hygiene 모듈 4종 단위 테스트."""

from __future__ import annotations

import pytest

from src.agents.context._data_quality_checks import (
    confidence_floor_for_density,
    is_card_provenance_traceable,
    is_signal_well_grouped,
    signal_density_label,
)
from src.services.cluster_identity import stable_card_join_key
from src.services.metric_canonical import canonicalize_metric
from src.services.peer_id_aliases import (
    PEER_ID_ALIASES,
    expand_peer_aliases,
    is_known_peer,
    normalize_to_canonical_id,
)


# ─────────────────────────────────────────────────────────────────────────────
# peer_id_aliases
# ─────────────────────────────────────────────────────────────────────────────


def test_expand_peer_aliases_returns_at_least_three_for_known_peers():
    for peer_id in PEER_ID_ALIASES:
        aliases = expand_peer_aliases(peer_id)
        assert peer_id in aliases
        assert len(aliases) >= 3, f"{peer_id} aliases too few: {aliases}"


def test_expand_unknown_peer_id_returns_self_only():
    aliases = expand_peer_aliases("unknown_peer")
    assert aliases == ["unknown_peer"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("samsung_sds", "samsung_sds"),
        ("삼성SDS", "samsung_sds"),
        ("삼성 SDS", "samsung_sds"),
        ("Samsung SDS", "samsung_sds"),
        ("LG CNS", "lg_cns"),
        ("엘지씨엔에스", "lg_cns"),
        ("POSCO DX", "posco_dx"),
        ("포스코ICT", "posco_dx"),
        ("현대오토에버", "hyundai_autoever"),
        ("SK AX", "sk_ax"),
        ("Unknown Co.", None),
        ("", None),
    ],
)
def test_normalize_to_canonical_id(raw, expected):
    assert normalize_to_canonical_id(raw) == expected


def test_is_known_peer_for_canonical_and_unknown():
    assert is_known_peer("samsung_sds") is True
    assert is_known_peer("unknown_peer") is False
    assert is_known_peer(None) is False


# ─────────────────────────────────────────────────────────────────────────────
# metric_canonical
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("net_income", "net_income"),
        ("당기순이익", "net_income"),
        ("순이익", "net_income"),
        ("매출", "revenue_total"),
        ("매출액", "revenue_total"),
        ("revenue", "revenue_total"),
        ("영업이익", "operating_profit"),
        ("operating_margin", "operating_margin"),
        ("매출총이익", "gross_profit"),
        ("임직원수", "employees"),
        ("unknown_metric", "unknown_metric"),
        ("", ""),
    ],
)
def test_canonicalize_metric(raw, expected):
    assert canonicalize_metric(raw) == expected


# ─────────────────────────────────────────────────────────────────────────────
# cluster_identity
# ─────────────────────────────────────────────────────────────────────────────


def test_stable_card_join_key_with_source_articles():
    card = {
        "id": "CN-20260520-0001",
        "source_raw_article_ids": [1, 2, 2, 3],
        "company": "samsung_sds",
        "event_type": "partnership",
        "created_at": "2026-05-20T03:00:00+09:00",
    }
    identity = stable_card_join_key(card)
    assert identity.card_id == "CN-20260520-0001"
    assert identity.source_raw_article_ids == (1, 2, 3)
    assert identity.peer_event_date_key == (
        "samsung_sds",
        "partnership",
        "2026-05-20",
    )
    assert identity.has_raw_articles is True
    assert identity.has_peer_event_date is True


def test_stable_card_join_key_without_articles_or_event():
    card = {"id": "x"}
    identity = stable_card_join_key(card)
    assert identity.card_id == "x"
    assert identity.source_raw_article_ids == ()
    assert identity.peer_event_date_key is None


# ─────────────────────────────────────────────────────────────────────────────
# _data_quality_checks
# ─────────────────────────────────────────────────────────────────────────────


def test_is_signal_well_grouped_year_only():
    assert is_signal_well_grouped({"period_year": 2025}) is True
    assert is_signal_well_grouped({"period_year": None}) is False
    assert is_signal_well_grouped({}) is False


def test_is_card_provenance_traceable_paths():
    assert is_card_provenance_traceable({"source_raw_article_ids": [1]}) is True
    assert is_card_provenance_traceable({"sources": [{"url": "https://x"}]}) is True
    assert (
        is_card_provenance_traceable({"evidence_payload": {"source_links": [{"url": "https://y"}]}})
        is True
    )
    assert is_card_provenance_traceable({}) is False


def test_signal_density_label_thresholds():
    assert signal_density_label(signal_count_4q=10, metric_count_4q=5) == "sparse"
    assert signal_density_label(signal_count_4q=200, metric_count_4q=50) == "moderate"
    assert signal_density_label(signal_count_4q=500, metric_count_4q=200) == "rich"


def test_confidence_floor_for_density():
    assert confidence_floor_for_density("sparse") == 0.6
    assert confidence_floor_for_density("moderate") == 0.8
    assert confidence_floor_for_density("rich") == 1.0
