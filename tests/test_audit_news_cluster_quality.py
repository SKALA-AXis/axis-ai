# 작성일: 2026-06-09
# 작성자: 박지원
# 변경이력:
#   2026-06-09 박지원 — 뉴스 전처리/클러스터링 품질 개선 및 하드코딩 룰 제거(#119)에 대한 테스트 추가
from scripts.audit_news_cluster_quality import (
    _cluster_metrics,
    _group_candidate_payload,
    _merge_candidate_payload,
    _rate,
    _risk_level,
)
from scripts.postprocess_singleton_clusters import (
    Cluster,
    GroupMergeCandidate,
    MergeCandidate,
    SourceCluster,
)


def test_cluster_metrics_counts_singleton_and_small_rates() -> None:
    clusters = [
        Cluster(
            cluster_id=10,
            article_count=1,
            article_ids=[10],
            titles=["LG CNS, 클로드 도입"],
            latest_event_at=None,
        ),
        Cluster(
            cluster_id=20,
            article_count=3,
            article_ids=[20, 21, 22],
            titles=["플래티어, 현대오토에버 계약"],
            latest_event_at=None,
        ),
        Cluster(
            cluster_id=30,
            article_count=8,
            article_ids=[30],
            titles=["대형 클러스터"],
            latest_event_at=None,
        ),
    ]

    metrics = _cluster_metrics(clusters, max_source_size=3)

    assert metrics == {
        "cluster_count": 3,
        "article_count": 12,
        "singleton_count": 1,
        "small_count": 2,
        "singleton_rate": 1 / 3,
        "small_rate": 2 / 3,
    }


def test_risk_level_prefers_concrete_quality_signals_over_rates() -> None:
    metrics = {
        "cluster_count": 12,
        "singleton_rate": 0.2,
        "small_rate": 0.3,
    }

    assert (
        _risk_level(
            metrics=metrics,
            merge_candidate_count=0,
            group_candidate_count=0,
            stale_card_count=1,
            warn_singleton_rate=0.45,
            warn_small_rate=0.65,
        )
        == "HIGH"
    )
    assert (
        _risk_level(
            metrics=metrics,
            merge_candidate_count=1,
            group_candidate_count=0,
            stale_card_count=0,
            warn_singleton_rate=0.45,
            warn_small_rate=0.65,
        )
        == "MEDIUM"
    )


def test_risk_level_warns_on_high_small_cluster_rate_only_with_enough_clusters() -> None:
    assert (
        _risk_level(
            metrics={"cluster_count": 9, "singleton_rate": 0.8, "small_rate": 0.9},
            merge_candidate_count=0,
            group_candidate_count=0,
            stale_card_count=0,
            warn_singleton_rate=0.45,
            warn_small_rate=0.65,
        )
        == "LOW"
    )
    assert (
        _risk_level(
            metrics={"cluster_count": 10, "singleton_rate": 0.8, "small_rate": 0.9},
            merge_candidate_count=0,
            group_candidate_count=0,
            stale_card_count=0,
            warn_singleton_rate=0.45,
            warn_small_rate=0.65,
        )
        == "MEDIUM"
    )


def test_candidate_payloads_are_report_friendly() -> None:
    source = SourceCluster(
        cluster_id=100,
        article_ids=[100, 101],
        titles=["플래티어, 현대오토에버 계약 체결"],
        title="플래티어, 현대오토에버 계약 체결",
        article_count=2,
        event_at=None,
    )
    target = Cluster(
        cluster_id=200,
        article_count=5,
        article_ids=[200],
        titles=["플래티어, 현대오토에버와 24억 계약"],
        latest_event_at=None,
    )
    merge = MergeCandidate(
        source=source,
        target=target,
        event_key="generic:contract_event",
        shared_tokens={"platier", "hyundai_autoever"},
        score=0.67891,
    )
    group = GroupMergeCandidate(
        target=source,
        sources=[source],
        event_key="generic:contract_event",
        shared_tokens={"platier"},
        score=0.51234,
    )

    assert _merge_candidate_payload(merge)["score"] == 0.679
    assert _merge_candidate_payload(merge)["article_ids"] == [100, 101]
    assert _group_candidate_payload(group)["score"] == 0.512
    assert _group_candidate_payload(group)["from_clusters"] == [100]


def test_rate_handles_empty_total() -> None:
    assert _rate(3, 0) == 0.0
