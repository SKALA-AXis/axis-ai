"""파이프라인 1회 실행 스크립트 — v3 결과를 콘솔에 출력."""

import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")

from src.agents.sector_keywords import sector_name_ko  # noqa: E402
from src.pipeline.ingestion_graph import ingestion_graph  # noqa: E402

_BAND_MARK = {"high": "■■■", "medium": "■■ ", "low": "■  "}


def main() -> None:
    peer_ids = ["samsung_sds", "lg_cns"]
    log.info("파이프라인 시작 | peer_ids=%s", peer_ids)

    initial_state = {
        "peer_ids": peer_ids,
        "trigger_type": "manual",
        "raw_article_ids": [],
        "credible_ids": [],
        "cluster_map": {},
        "representative_ids": [],
        "classified_clusters": [],
        "issue_cards": [],
        "evidence_results": [],
        "indexed_vector_ids": [],
        "errors": [],
        "human_review_flags": [],
    }

    result = ingestion_graph.invoke(initial_state)

    cards = result.get("issue_cards", [])
    evidence = result.get("evidence_results", [])
    pass_count = sum(1 for r in evidence if r.get("pass"))

    print("\n" + "=" * 78)
    print("📊 파이프라인 v3 실행 결과")
    print("=" * 78)
    print(f"  RAW 기사:        {len(result.get('raw_article_ids', []))}건")
    print(f"  신뢰도 통과:     {len(result.get('credible_ids', []))}건")
    print(f"  클러스터:        {len(result.get('cluster_map', {}))}개")
    print(f"  대표 기사:       {len(result.get('representative_ids', []))}건")
    print(f"  이슈카드:        {len(cards)}건")
    print(f"  검증 첨부 통과:  {pass_count}/{len(evidence)}건")
    print(f"  Human 검토 필요: {len(result.get('human_review_flags', []))}건")
    if result.get("errors"):
        print(f"  오류:            {result['errors']}")

    # 섹터 분포
    if cards:
        print("\n  ── 섹터 분포 ──")
        sector_counts: dict[str, int] = {}
        for c in cards:
            sector_counts[c.get("sector", "other")] = (
                sector_counts.get(c.get("sector", "other"), 0) + 1
            )
        for sid, n in sorted(sector_counts.items(), key=lambda x: -x[1]):
            print(f"    {sector_name_ko(sid):8s}  {n}건")

        print("\n  ── 노출도 밴드 ──")
        band_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for c in cards:
            band = c.get("exposure_band", "low")
            band_counts[band] = band_counts.get(band, 0) + 1
        for band in ("high", "medium", "low"):
            print(f"    {_BAND_MARK[band]} {band:6s} {band_counts[band]}건")

    print("\n" + "=" * 78)
    print("📋 이슈카드 상세")
    print("=" * 78)

    for i, card in enumerate(cards, 1):
        sector = card.get("sector", "other")
        band = card.get("exposure_band", "low")
        signals = card.get("signals", {}) or {}
        val = card.get("validation", {}) or {}
        chain = card.get("evidence_chain", {}) or {}

        evidence_mark = "✅" if val.get("pass") else "⚠️"

        print(f"\n[{i}] {_BAND_MARK[band]} {sector_name_ko(sector)} | {card.get('peer_id', '?')}")
        print(f"     ID: {card.get('id', '?')}  |  Event: {card.get('event_type', '?')}")
        print(f"     제목: {card.get('title', '')}")
        print(
            f"     노출도: {card.get('exposure_score', 0):.2f} ({band})"
            f"  |  cluster={signals.get('cluster_size', 0)}"
            f"  peer={signals.get('peer_mention_count', 0)}"
            f"  cred={signals.get('credibility_max', 0):.2f}"
            f"  tier1={signals.get('tier1_count', 0)}"
        )
        print(f"     검증 첨부: {evidence_mark} {val.get('reason', '')}")

        print("     ─── 3줄 요약 ───")
        for line in card.get("summary_lines", []):
            print(f"       {line}")

        # 검증 체인 4종 요약
        prov = chain.get("provenance", {}) or {}
        flink = chain.get("financial_link", {}) or {}
        seg = (flink.get("segment") or {}) if flink else {}
        print("     ─── 검증 체인 ───")
        print(f"       원문: {len(chain.get('source_links', []))}개")
        print(
            f"       Provenance: cluster={prov.get('cluster_id', '?')}"
            f" raw_ids={len(prov.get('raw_article_ids', []))}"
            f" model={prov.get('llm_model', '?')}"
            f" prompt={prov.get('prompt_version', '?')}"
        )
        fr = chain.get("financial_refs", [])
        seg_ko = seg.get("name_ko") if seg else "-"
        print(
            f"       Financial refs: {len(fr)}개"
            f" (segment={seg_ko}, linked={flink.get('linked', False)})"
        )
        for fr_item in fr[:3]:
            print(f"         · {fr_item.get('narrative', '')}")
        for hl in (flink.get("highlights") or [])[:3]:
            print(f"         ▸ {hl}")
        print(f"       MBB refs: {len(chain.get('mbb_refs', []))}개 (W5 채움)")

        print("     ─── 출처 (상위 2) ───")
        for s in card.get("sources", [])[:2]:
            title = s.get("title", "")[:60]
            print(f"       [{s.get('index','')}] {s.get('source_name','')} — {title}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
