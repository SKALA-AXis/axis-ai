from scripts.postprocess_singleton_clusters import (
    Cluster,
    GroupMergeCandidate,
    MergeCandidate,
    SourceCluster,
    _cluster_relation,
    _filter_group_candidates_after_target_merges,
    _find_candidates,
    _is_stock_noise,
    _strong_event_keys,
)


def test_lg_cns_anthropic_claude_titles_share_strong_event_key() -> None:
    assert "lg_cns_anthropic_claude" in _strong_event_keys(
        "LG CNS, 앤트로픽 '클로드 엔터프라이즈' 도입"
    )
    assert "lg_cns_anthropic_claude" in _strong_event_keys(
        "LG CNS, 클로드 도입... 그룹사 단계적 확대"
    )


def test_strong_event_key_merges_even_with_single_shared_token() -> None:
    relation = _cluster_relation(
        ["LGCNS, ‘클로드’ 도입"],
        ["LG CNS, 앤트로픽 '클로드 엔터프라이즈' 도입…그룹 차원 AX 확대 나서"],
        target_size=61,
    )

    assert relation is not None
    assert relation[0] == "lg_cns_anthropic_claude"


def test_platier_hyundai_autoever_contract_is_not_stock_noise() -> None:
    title = "[특징주] 플래티어, 현대오토에버와 24억 규모 추가 공급계약 체결"

    assert "platier_hyundai_autoever_contract" in _strong_event_keys(title)
    assert _is_stock_noise(title) is False


def test_small_lg_cns_anthropic_cluster_merges_to_large_cluster() -> None:
    source = SourceCluster(
        cluster_id=46655,
        article_ids=[46655, 46641],
        titles=[
            "[Tech & Now] LG CNS, 앤트로픽 '클로드' 품고 AX 사업 확대",
            "LG CNS, 전사에 앤트로픽 '클로드' 깐다...그룹·외부 AX도 가속",
        ],
        title="[Tech & Now] LG CNS, 앤트로픽 '클로드' 품고 AX 사업 확대",
        article_count=2,
        event_at=None,
    )
    target = Cluster(
        cluster_id=46656,
        article_count=47,
        article_ids=[46656, 46654],
        titles=[
            "LG CNS, 앤트로픽 ‘클로드’ 도입 계약 체결...그룹 차원 AX 박차",
            "LG CNS, 클로드 엔터프라이즈 도입 계약...그룹 전 계열사 적용",
        ],
        latest_event_at=None,
    )

    candidates = _find_candidates(
        sources=[source],
        targets=[target],
        max_time_gap_hours=72,
        min_score=0.45,
    )

    assert len(candidates) == 1
    assert candidates[0].source.cluster_id == 46655
    assert candidates[0].target.cluster_id == 46656


def test_group_merge_skips_sources_already_merged_to_large_target() -> None:
    source = SourceCluster(
        cluster_id=46655,
        article_ids=[46655],
        titles=["LG CNS, 앤트로픽 '클로드' 도입"],
        title="LG CNS, 앤트로픽 '클로드' 도입",
        article_count=1,
        event_at=None,
    )
    other_source = SourceCluster(
        cluster_id=46641,
        article_ids=[46641],
        titles=["LG CNS, 클로드 도입 계약"],
        title="LG CNS, 클로드 도입 계약",
        article_count=1,
        event_at=None,
    )
    target = Cluster(
        cluster_id=46656,
        article_count=47,
        article_ids=[46656],
        titles=["LG CNS, 앤트로픽 클로드 엔터프라이즈 도입 계약"],
        latest_event_at=None,
    )
    merge_candidate = MergeCandidate(
        source=source,
        target=target,
        event_key="lg_cns_anthropic_claude",
        shared_tokens={"anthropic", "claude"},
        score=0.7,
    )
    group_candidate = GroupMergeCandidate(
        target=source,
        sources=[source, other_source],
        event_key="generic:anthropic_claude",
        shared_tokens={"anthropic", "claude"},
        score=0.6,
    )

    assert (
        _filter_group_candidates_after_target_merges(
            [group_candidate],
            [merge_candidate],
        )
        == []
    )
