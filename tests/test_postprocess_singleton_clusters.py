import scripts.postprocess_singleton_clusters as postprocess
from scripts.postprocess_singleton_clusters import (
    Cluster,
    GroupMergeCandidate,
    MergeCandidate,
    SourceCluster,
    _cluster_relation,
    _filter_group_candidates_after_target_merges,
    _find_candidates,
    _has_title_anchor_overlap,
    _is_list_like,
    _is_stock_noise,
)


def test_same_company_product_event_merges_without_specific_event_key() -> None:
    relation = _cluster_relation(
        ["LGCNS, ‘클로드’ 도입"],
        ["LG CNS, 앤트로픽 '클로드 엔터프라이즈' 도입…그룹 차원 AX 확대 나서"],
        target_size=61,
    )

    assert relation is not None
    assert relation[0].startswith("generic:")


def test_title_anchor_overlap_rejects_different_same_company_ai_events() -> None:
    assert (
        _has_title_anchor_overlap(
            ["삼성SDS, 토큰증권 시장 정조준…AI·클라우드와 시너지 낼까"],
            [
                "[특징주] 삼성SDS, 정부 '온AI' 모바일 업무환경 지원에 강세",
                "삼성SDS, AI 협업 도구 '브리티웍스' 온AI 공식 솔루션 선정",
            ],
        )
        is False
    )


def test_title_anchor_overlap_accepts_same_event_surface_variants() -> None:
    assert (
        _has_title_anchor_overlap(
            ["LG CNS, 한전 AI 전환 밑그림 그린다"],
            ["LG CNS, 한국전력 차세대 영업배전시스템 ISP 사업 수주"],
        )
        is True
    )


def test_company_name_alone_does_not_create_generic_merge() -> None:
    relation = _cluster_relation(
        ['"현대오토에버, 그룹 AI 투자 확대 수혜 기대…목표가↑"-IBK'],
        ['[특징주] 플래티어, 현대오토에버와 24억 규모 추가 공급계약 체결'],
        target_size=6,
    )

    assert relation is None


def test_contract_stock_article_is_not_stock_noise_without_event_hardcoding() -> None:
    title = "[특징주] 플래티어, 현대오토에버와 24억 규모 추가 공급계약 체결"

    assert _is_stock_noise(title) is False


def test_multi_company_roundup_title_is_list_like_noise() -> None:
    assert _is_list_like("[#시큐리티 포커스] 유락 '디파스 프로 맥' 출시·삼성SDS 'AI 클라우드 ...")
    assert _is_list_like("[전자·IT 레이더] 삼성SDS·한컴·카페24, 보안·AI·커머스 핵심 사업")


def test_small_lg_cns_anthropic_cluster_merges_to_large_cluster_with_llm(monkeypatch) -> None:
    monkeypatch.setattr(postprocess, "_title_llm_same_event", lambda *args, **kwargs: True)
    source = SourceCluster(
        cluster_id=46655,
        article_ids=[46655, 46641],
        titles=[
            "LG CNS, 앤트로픽 '클로드' 품고 AX 사업 확대",
            "LG CNS, 전사에 앤트로픽 '클로드' 깐다...그룹·외부 AX도 가속",
        ],
        snippets=[],
        title="LG CNS, 앤트로픽 '클로드' 품고 AX 사업 확대",
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
        snippets=[],
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
        snippets=[],
        title="LG CNS, 앤트로픽 '클로드' 도입",
        article_count=1,
        event_at=None,
    )
    other_source = SourceCluster(
        cluster_id=46641,
        article_ids=[46641],
        titles=["LG CNS, 클로드 도입 계약"],
        snippets=[],
        title="LG CNS, 클로드 도입 계약",
        article_count=1,
        event_at=None,
    )
    target = Cluster(
        cluster_id=46656,
        article_count=47,
        article_ids=[46656],
        titles=["LG CNS, 앤트로픽 클로드 엔터프라이즈 도입 계약"],
        snippets=[],
        latest_event_at=None,
    )
    merge_candidate = MergeCandidate(
        source=source,
        target=target,
        event_key="generic:anthropic_claude",
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
