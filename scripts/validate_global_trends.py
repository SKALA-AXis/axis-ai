# 작성일: 2026-05-26
# 작성자: 최종민
# 변경이력:
#   2026-05-26 최종민 — 글로벌 IT 트렌드 5-phase ITTrendAgent e2e 검증 스크립트 작성
#   2026-06-01 박지원 — peer profile 스냅샷 파이프라인 추가에 따른 반영
"""Validation simulation for ITTrendAgent end-to-end pipeline.

axis-ai pod 안에서 실행:
    kubectl exec <pod> -- python3 /tmp/validate_global_trends.py

설계서: axis-ai/design/30-analysis/global-trends.md §12 S8.

실행 흐름:
1. fetch_global_trend_inputs(30) — raw fetch 결과 카운트
2. ITTrendAgent.generate() — 5-phase 풀 호출 (LLM 3 회, 실제 upsert)
3. global_industry_trends 의 새 row 확인
4. fetch_latest_trend_context(7) — TrendContext shape + cache 동작 확인
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime

from sqlalchemy import text

from src.agents.it_trend_agent import ITTrendAgent, ITTrendInput
from src.db.article_store import (
    fetch_global_trend_inputs,
    fetch_latest_trend_context,
    invalidate_trend_context_cache,
)
from src.db.postgres import SessionLocal


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(f"▶ {title}")
    print("=" * 80)


def main() -> int:
    started_at = datetime.utcnow()

    section("Step 1 — fetch_global_trend_inputs(window_days=30)")
    rows = fetch_global_trend_inputs(window_days=30)
    print(f"  total rows fetched: {len(rows)}")
    # source breakdown
    by_source: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for r in rows:
        by_source[r.get("source_name") or "?"] = by_source.get(r.get("source_name") or "?", 0) + 1
        by_type[r.get("source_type") or "?"] = by_type.get(r.get("source_type") or "?", 0) + 1
    print(f"  by source_name (normalized): {json.dumps(by_source, ensure_ascii=False)}")
    print(f"  by source_type (normalized): {json.dumps(by_type, ensure_ascii=False)}")
    if not rows:
        print("  ❌ fetch result empty — abort.")
        return 1

    section("Step 2 — ITTrendAgent.generate() (full 5-phase, LLM 3 calls, real upsert)")
    print(f"  started_at_utc = {started_at.isoformat()}")
    trend_input = ITTrendInput(
        trend_items=[],
        period=None,
        source_groups=[],
        previous_trend_context=None,
        reference_issue_results=[],
        metadata={
            "window_days": 30,
            "include_peer_alignment": True,
            "max_trend_count": 5,
            "min_mention_count": 3,
        },
    )

    t0 = time.time()
    try:
        result = ITTrendAgent().generate(trend_input)
    except Exception as e:
        print(f"  ❌ ITTrendAgent.generate() raised: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 2
    elapsed = time.time() - t0
    print(f"  ✅ generate() elapsed = {elapsed:.1f}s")

    print(f"  analysis_id        = {result.get('analysis_id')}")
    print(f"  persisted_row_count= {result.get('persisted_row_count')}")
    print(f"  warning            = {result.get('warning')}")
    print(f"  confidence         = {result.get('confidence')}")
    print(f"  snapshots          = {len(result.get('snapshots') or [])} companies")
    snapshots = result.get("snapshots") or []
    for s in snapshots:
        print(
            f"    - {s['company_id']:15s} card_count={s['card_count']:3d} "
            f"top_themes={s['top_themes'][:3]}"
        )
    detections = result.get("trend_detections") or []
    print(f"  trend_detections   = {len(detections)}")
    for d in detections:
        print(
            f"    - {d['theme']:20s} mention={d['mention_count']:3d} "
            f"intensity={d['intensity']:8s} leading={d.get('leading_companies', [])}"
        )

    alignment = result.get("peer_alignment") or {}
    print(f"  peer_alignment     = {len(alignment)} keywords × peers")
    for kw, peers in list(alignment.items())[:3]:
        print(f"    keyword='{kw}':")
        for p in peers:
            print(
                f"      - {p['peer_id']:18s} type={p['alignment_type']:9s} "
                f"score={p['alignment_score']:.2f} peer_n={p['peer_mention_count']:2d} "
                f"note='{(p.get('strategic_note') or '')[:60]}'"
            )

    impact = result.get("impact_matrix") or []
    print(f"  impact_matrix      = {len(impact)} cells")
    for c in impact[:5]:
        print(
            f"    - trend='{c['trend_theme']}' line={c['sk_ax_line']:18s} "
            f"dir={c['direction']:8s} mag={c['magnitude']:6s} "
            f"channel='{c.get('channel', '')[:50]}'"
        )

    forecasts = result.get("forecasts") or []
    print(f"  forecasts          = {len(forecasts)}")
    for f in forecasts:
        print(
            f"    - horizon={f['horizon']:3s} scenario={f['scenario']:11s} "
            f"risk={f.get('risk_level')} narrative='{(f.get('narrative') or '')[:60]}'"
        )

    print(f"  final_one_liner    = {result.get('final_one_liner')}")
    print(f"  sk_ax_implication  = {result.get('sk_ax_implication')}")

    rows_built = result.get("rows") or []
    print(f"  persistence rows   = {len(rows_built)}")
    for r in rows_built[:3]:
        sa_id = r.get("source_analysis_id") or ""
        print(
            f"    - source_analysis_id='{sa_id}' (len={len(sa_id)}, hard cap 100), "
            f"keyword='{r.get('keyword')}' "
            f"impact_score={r.get('impact_score')} "
            f"related_peers={r.get('related_peer_ids')}"
        )

    section("Step 3 — DB verification (global_industry_trends 새 row 직접 SELECT)")
    today = started_at.date()
    with SessionLocal() as db:
        db_rows = (
            db.execute(
                text(
                    """
                    SELECT trend_date, region, industry, keyword, keyword_category,
                           mention_count, impact_score, confidence, source_analysis_id,
                           LENGTH(source_analysis_id) AS sa_id_len,
                           array_length(related_peer_ids, 1) AS aligned_peers,
                           array_length(related_card_ids, 1) AS evidence_cards,
                           updated_at
                    FROM global_industry_trends
                    WHERE trend_date = :today AND region = 'global'
                    ORDER BY impact_score DESC NULLS LAST, mention_count DESC
                    """
                ),
                {"today": today},
            )
            .mappings()
            .all()
        )
    print(f"  DB rows for trend_date={today}, region='global': {len(db_rows)}")
    for r in db_rows:
        print(
            f"    - keyword='{r['keyword']:18s}' cat={r['keyword_category']:10s} "
            f"mention={r['mention_count']:3d} impact={float(r['impact_score'] or 0):.1f} "
            f"conf={float(r['confidence'] or 0):.2f} "
            f"sa_id_len={r['sa_id_len']:3d} aligned_peers={r['aligned_peers']} "
            f"cards={r['evidence_cards']}"
        )
    if not db_rows:
        print("  ⚠ DB 에 row 가 안 들어감 — 위 persistence/warning 확인 필요")

    section("Step 4 — fetch_latest_trend_context(7) + TTL cache 동작")
    invalidate_trend_context_cache()
    t1 = time.time()
    ctx = fetch_latest_trend_context(within_days=7)
    cold = time.time() - t1
    t2 = time.time()
    ctx2 = fetch_latest_trend_context(within_days=7)
    warm = time.time() - t2
    print(f"  cold call elapsed  = {cold * 1000:.1f}ms")
    print(f"  warm call elapsed  = {warm * 1000:.1f}ms (cache hit 이면 cold 보다 훨씬 빨라야 함)")
    print(f"  trend_summary      = {ctx.get('trend_summary', '')[:100]}")
    print(f"  signals (count={len(ctx.get('signals', []))}): ")
    for sig in (ctx.get("signals") or [])[:5]:
        print(
            f"    - signal='{sig['signal']:20s}' intensity={sig.get('intensity')} "
            f"leading={sig.get('leading_companies')}"
        )
    print(f"  cache_consistent   = {ctx == ctx2}")

    section("✅ Validation complete")
    pipeline_status = "PASS" if result.get("persisted_row_count", 0) > 0 else "FAIL"
    print(
        f"  결과: 5-phase pipeline {pipeline_status}, "
        f"persisted={result.get('persisted_row_count')} rows, "
        f"trend_context cache={'OK' if ctx == ctx2 else 'INCONSISTENT'}"
    )
    return 0 if result.get("persisted_row_count", 0) > 0 else 3


if __name__ == "__main__":
    sys.exit(main())
