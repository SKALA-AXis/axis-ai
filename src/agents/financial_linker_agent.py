"""뉴스↔재무 연결 에이전트 (v3 §5.2 차별화 핵심) — PoC.

뉴스 카드를 받아 Peer사 재무 시계열에서 관련 사업부·변화량을 찾아 첨부한다.
시사점(implication)을 대체하는 "숫자로 설명되는 팩트"를 생성하는 것이 목적.

PoC 단계 데이터 소스: data/peer_financials/{peer_id}.json (stub).
W4 후반: peer_financials 테이블이 axis-backend Flyway로 생성되면 DB 조회로 교체.

핵심 로직:
  1. 카드의 sector + event_type + 제목·요약을 segment 키워드와 매칭
  2. 가장 일치도 높은 segment 1개 선정 (없으면 total로 폴백)
  3. 직전 분기 대비 변화량(QoQ) 및 1년 전 대비 변화량(YoY) 계산
  4. AI 매출 비중·헤드카운트 변화 등 부가 지표 첨부
  5. financial_refs 형식으로 출력 — DART 공시번호, IR 페이지 포함
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_FINANCIAL_DIR = Path(__file__).resolve().parents[2] / "data" / "peer_financials"
_SK_AX_PATH = Path(__file__).resolve().parents[2] / "data" / "sk_ax_financials.json"


@lru_cache(maxsize=8)
def _load_financials(peer_id: str) -> dict[str, Any] | None:
    """{peer_id}.json을 한 번만 로드하고 캐시한다."""
    path = _FINANCIAL_DIR / f"{peer_id}.json"
    if not path.exists():
        log.warning("재무 stub 파일 없음 | peer_id=%s path=%s", peer_id, path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("재무 stub 파싱 실패 | peer_id=%s error=%s", peer_id, e)
        return None


@lru_cache(maxsize=1)
def _load_sk_ax() -> dict[str, Any] | None:
    """SK AX 자체 재무 데이터 로드 — 비교 기준."""
    if not _SK_AX_PATH.exists():
        log.warning("SK AX 재무 파일 없음 | path=%s", _SK_AX_PATH)
        return None
    try:
        return json.loads(_SK_AX_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("SK AX 재무 파싱 실패 | error=%s", e)
        return None


def _match_segment(text: str, segments: dict[str, dict[str, Any]]) -> str | None:
    """텍스트에서 가장 많이 매칭된 segment ID를 반환. 동률은 정의 순서 우선."""
    if not text or not segments:
        return None
    text_lower = text.lower()
    best_id: str | None = None
    best_hits = 0
    for seg_id, meta in segments.items():
        hits = sum(1 for kw in meta.get("keywords", []) if kw.lower() in text_lower)
        if hits > best_hits:
            best_hits = hits
            best_id = seg_id
    return best_id if best_hits > 0 else None


def _delta_pct(curr: float, prev: float) -> float | None:
    if prev is None or prev == 0:
        return None
    return round((curr - prev) / prev * 100, 1)


def _format_krwbn(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 10000:
        return f"{value / 10000:.2f}조원"
    return f"{value:,.0f}억원"


def _compute_vs_sk_ax(
    peer_name: str,
    peer_latest: dict[str, Any],
    peer_prev_y: dict[str, Any] | None,
    peer_headcount: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Peer 최신 분기 vs SK AX 동기 비교 — 결정적 4개 지표.

    1) 전체 매출 격차 (배수·절대)
    2) 매출 YoY 성장률 격차 (%p)
    3) AI 매출 비중 격차 (%p)
    4) AI 엔지니어 수 격차 (배수·절대)
    """
    sk = _load_sk_ax()
    if not sk:
        return None

    sk_quarterly = sk.get("quarterly", [])
    if len(sk_quarterly) < 2:
        return None

    target_period = peer_latest.get("period")
    sk_match = next((q for q in sk_quarterly if q.get("period") == target_period), None)
    if not sk_match:
        sk_match = sk_quarterly[-1]
    sk_prev_y = (
        next((q for q in sk_quarterly if q.get("period") == (peer_prev_y or {}).get("period")), None)
        if peer_prev_y else None
    )
    if not sk_prev_y and len(sk_quarterly) >= 5:
        idx = sk_quarterly.index(sk_match)
        sk_prev_y = sk_quarterly[idx - 4] if idx >= 4 else sk_quarterly[0]

    metrics: list[dict[str, Any]] = []

    # ── 1) 매출 절대 격차 ──────────────────────────────────
    peer_rev = peer_latest.get("revenue_total_krwbn")
    sk_rev = sk_match.get("revenue_total_krwbn")
    if peer_rev and sk_rev:
        ratio = round(peer_rev / sk_rev, 2)
        gap = peer_rev - sk_rev
        metrics.append({
            "key": "revenue_total",
            "metric_ko": "전체 매출",
            "period": target_period,
            "sk_ax_value_krwbn": sk_rev,
            "peer_value_krwbn": peer_rev,
            "ratio_peer_over_skax": ratio,
            "gap_krwbn": gap,
            "narrative": (
                f"{target_period} 매출: SK AX {_format_krwbn(sk_rev)} vs "
                f"{peer_name} {_format_krwbn(peer_rev)} "
                f"(Peer {ratio}x · 격차 {_format_krwbn(abs(gap))})"
            ),
        })

    # ── 2) 매출 YoY 격차 (성장 속도) ─────────────────────────
    peer_yoy = _delta_pct(peer_rev, (peer_prev_y or {}).get("revenue_total_krwbn")) if peer_prev_y else None
    sk_yoy = _delta_pct(sk_rev, (sk_prev_y or {}).get("revenue_total_krwbn")) if sk_prev_y else None
    if peer_yoy is not None and sk_yoy is not None:
        gap_pp = round(peer_yoy - sk_yoy, 1)
        metrics.append({
            "key": "revenue_yoy_pct",
            "metric_ko": "매출 YoY 성장률",
            "sk_ax_pct": sk_yoy,
            "peer_pct": peer_yoy,
            "gap_pp": gap_pp,
            "narrative": (
                f"매출 YoY: SK AX {sk_yoy:+.1f}% vs {peer_name} {peer_yoy:+.1f}% "
                f"({'Peer 우위' if gap_pp > 0 else 'SK AX 우위'} {abs(gap_pp):+.1f}%p)"
            ),
        })

    # ── 3) AI 매출 비중 격차 ─────────────────────────────────
    peer_ai = peer_latest.get("ai_revenue_share_pct")
    sk_ai = sk_match.get("ai_revenue_share_pct")
    if peer_ai is not None and sk_ai is not None:
        gap_pp = round(peer_ai - sk_ai, 1)
        metrics.append({
            "key": "ai_share_pct",
            "metric_ko": "AI 매출 비중",
            "sk_ax_pct": sk_ai,
            "peer_pct": peer_ai,
            "gap_pp": gap_pp,
            "narrative": (
                f"AI 매출 비중: SK AX {sk_ai:.1f}% vs {peer_name} {peer_ai:.1f}% "
                f"({'Peer 우위' if gap_pp > 0 else 'SK AX 우위'} {abs(gap_pp):+.1f}%p)"
            ),
        })

    # ── 4) AI 엔지니어 격차 ──────────────────────────────────
    sk_hc = sk.get("headcount") or []
    if peer_headcount and sk_hc:
        peer_ai_eng = (peer_headcount[-1] or {}).get("ai_engineers_est")
        sk_ai_eng = (sk_hc[-1] or {}).get("ai_engineers_est")
        if peer_ai_eng and sk_ai_eng:
            ratio = round(peer_ai_eng / sk_ai_eng, 2)
            metrics.append({
                "key": "ai_engineers",
                "metric_ko": "AI 엔지니어",
                "sk_ax_count": sk_ai_eng,
                "peer_count": peer_ai_eng,
                "ratio_peer_over_skax": ratio,
                "gap_count": peer_ai_eng - sk_ai_eng,
                "narrative": (
                    f"AI 엔지니어: SK AX {sk_ai_eng}명 vs {peer_name} {peer_ai_eng}명 "
                    f"(Peer {ratio}x · 격차 {peer_ai_eng - sk_ai_eng:+}명)"
                ),
            })

    if not metrics:
        return None

    # 한 줄 요약 — 우위 방향성·핵심 격차 중심
    summary_bits: list[str] = []
    for m in metrics:
        if m["key"] == "revenue_total":
            summary_bits.append(f"매출 {m['ratio_peer_over_skax']}x")
        elif m["key"] == "revenue_yoy_pct":
            summary_bits.append(f"YoY {m['gap_pp']:+.1f}%p")
        elif m["key"] == "ai_share_pct":
            summary_bits.append(f"AI 비중 {m['gap_pp']:+.1f}%p")
        elif m["key"] == "ai_engineers":
            summary_bits.append(f"AI 인력 {m['ratio_peer_over_skax']}x")
    summary_one_liner = f"vs SK AX ({target_period}): " + " · ".join(summary_bits)

    return {
        "sk_ax_period": sk_match.get("period"),
        "peer_period": target_period,
        "sk_ax_data_source": sk.get("_source", "unknown"),
        "metrics": metrics,
        "summary_one_liner": summary_one_liner,
    }


class FinancialLinkerAgent:
    """카드 sector·event_type을 재무 segment에 연결하고 변화량을 계산한다."""

    def link(self, card: dict[str, Any]) -> dict[str, Any]:
        """카드에 재무 컨텍스트를 첨부하기 위한 dict를 반환한다.

        Returns:
            {
              "linked": bool,
              "reason": str,
              "segment": {"id", "name_ko", "match_keyword_hits"} | None,
              "financial_refs": [
                {"period", "metric", "value_krwbn", "delta_pct_qoq", "delta_pct_yoy",
                 "dart_rcept_no", "ir_page", "narrative"}
              ],
              "highlights": ["...", ...],   # 1단 한 줄 요약용
              "headcount_delta": {...} | None,
            }
        """
        peer_id = card.get("peer_id")
        if not peer_id:
            return self._empty("peer_id 없음")

        data = _load_financials(peer_id)
        if not data:
            return self._empty(f"재무 데이터 없음 (peer_id={peer_id})")

        quarterly = data.get("quarterly", [])
        if len(quarterly) < 2:
            return self._empty("분기 데이터 2개 미만")

        segments = data.get("segments", {})
        title = card.get("title", "") or ""
        summary = " ".join(card.get("summary_lines", []) or [])
        text = f"{title} {summary}"
        segment_id = _match_segment(text, segments)

        latest = quarterly[-1]
        prev_q = quarterly[-2]
        prev_y = quarterly[-5] if len(quarterly) >= 5 else None

        refs: list[dict[str, Any]] = []
        highlights: list[str] = []

        # ── 1) 전체 매출 변화 (항상 포함) ─────────────────────
        rev_curr = latest.get("revenue_total_krwbn")
        rev_prev = prev_q.get("revenue_total_krwbn")
        rev_yoy = prev_y.get("revenue_total_krwbn") if prev_y else None
        qoq = _delta_pct(rev_curr, rev_prev)
        yoy = _delta_pct(rev_curr, rev_yoy) if rev_yoy else None
        refs.append({
            "period": latest["period"],
            "metric": "revenue_total_krwbn",
            "metric_ko": "전체 매출",
            "value_krwbn": rev_curr,
            "delta_pct_qoq": qoq,
            "delta_pct_yoy": yoy,
            "dart_rcept_no": latest.get("dart_rcept_no"),
            "ir_page": latest.get("ir_page"),
            "narrative": (
                f"{latest['period']} 매출 {_format_krwbn(rev_curr)}"
                + (f" (QoQ {qoq:+.1f}%)" if qoq is not None else "")
                + (f" / YoY {yoy:+.1f}%" if yoy is not None else "")
            ),
        })
        if yoy is not None and abs(yoy) >= 5.0:
            highlights.append(f"매출 YoY {yoy:+.1f}%")

        # ── 2) Segment 매출 (매칭된 경우만) ───────────────────
        seg_obj = None
        if segment_id:
            seg_meta = segments.get(segment_id, {})
            seg_obj = {
                "id": segment_id,
                "name_ko": seg_meta.get("name_ko", segment_id),
                "match_keyword_hits": sum(
                    1 for kw in seg_meta.get("keywords", []) if kw.lower() in text.lower()
                ),
            }
            seg_curr = (latest.get("segment_revenue_krwbn") or {}).get(segment_id)
            seg_prev = (prev_q.get("segment_revenue_krwbn") or {}).get(segment_id)
            seg_yoy_v = (prev_y.get("segment_revenue_krwbn") or {}).get(segment_id) if prev_y else None
            if seg_curr is not None:
                seg_qoq = _delta_pct(seg_curr, seg_prev) if seg_prev else None
                seg_yoy = _delta_pct(seg_curr, seg_yoy_v) if seg_yoy_v else None
                refs.append({
                    "period": latest["period"],
                    "metric": f"segment_revenue.{segment_id}",
                    "metric_ko": f"{seg_obj['name_ko']} 매출",
                    "value_krwbn": seg_curr,
                    "delta_pct_qoq": seg_qoq,
                    "delta_pct_yoy": seg_yoy,
                    "dart_rcept_no": latest.get("dart_rcept_no"),
                    "ir_page": latest.get("ir_page"),
                    "narrative": (
                        f"{seg_obj['name_ko']} 매출 {_format_krwbn(seg_curr)}"
                        + (f" (QoQ {seg_qoq:+.1f}%)" if seg_qoq is not None else "")
                        + (f" / YoY {seg_yoy:+.1f}%" if seg_yoy is not None else "")
                    ),
                })
                if seg_yoy is not None and abs(seg_yoy) >= 10.0:
                    highlights.append(f"{seg_obj['name_ko']} YoY {seg_yoy:+.1f}%")

        # ── 3) AI 매출 비중 (sector=ai_tech 또는 키워드 매칭 시) ──
        sector = card.get("sector", "")
        ai_curr = latest.get("ai_revenue_share_pct")
        ai_yoy = prev_y.get("ai_revenue_share_pct") if prev_y else None
        if ai_curr is not None and (sector == "ai_tech" or segment_id in {"ai", "ai_dx"}):
            refs.append({
                "period": latest["period"],
                "metric": "ai_revenue_share_pct",
                "metric_ko": "AI 매출 비중",
                "value_krwbn": None,
                "value_pct": ai_curr,
                "delta_pct_yoy": round(ai_curr - ai_yoy, 1) if ai_yoy else None,
                "dart_rcept_no": latest.get("dart_rcept_no"),
                "ir_page": latest.get("ir_page"),
                "narrative": (
                    f"AI 매출 비중 {ai_curr:.1f}%"
                    + (f" (1년 전 {ai_yoy:.1f}% → +{ai_curr - ai_yoy:.1f}%p)" if ai_yoy else "")
                ),
            })
            if ai_yoy is not None:
                highlights.append(f"AI 비중 {ai_yoy:.1f}%→{ai_curr:.1f}%")

        # ── 4) 헤드카운트 변화 (있을 때만) ────────────────────
        headcount = data.get("headcount", [])
        head_delta = None
        if len(headcount) >= 2:
            hc_curr = headcount[-1]
            hc_base = headcount[0]
            head_delta = {
                "from_period": hc_base["period"],
                "to_period": hc_curr["period"],
                "total_delta": hc_curr["total"] - hc_base["total"],
                "ai_engineers_delta": (
                    hc_curr.get("ai_engineers_est", 0) - hc_base.get("ai_engineers_est", 0)
                ),
            }
            if head_delta["ai_engineers_delta"] >= 100:
                highlights.append(
                    f"AI 인력 +{head_delta['ai_engineers_delta']}명 ({hc_base['period']}→{hc_curr['period']})"
                )

        # ── 5) vs SK AX 비교 (결정적 4지표) ─────────────────────
        vs_sk_ax = _compute_vs_sk_ax(
            peer_name=data.get("company_name_ko", peer_id),
            peer_latest=latest,
            peer_prev_y=prev_y,
            peer_headcount=headcount,
        )
        if vs_sk_ax:
            for m in vs_sk_ax["metrics"]:
                if m["key"] == "ai_share_pct":
                    highlights.append(
                        f"vs SK AX: AI 비중 {m['sk_ax_pct']:.1f}%→{m['peer_pct']:.1f}% "
                        f"(격차 {m['gap_pp']:+.1f}%p)"
                    )
                elif m["key"] == "ai_engineers":
                    highlights.append(
                        f"vs SK AX: AI 인력 {m['sk_ax_count']}→{m['peer_count']}명 "
                        f"(Peer {m['ratio_peer_over_skax']}x)"
                    )

        log.info(
            "재무 연결 완료 | peer=%s segment=%s refs=%d highlights=%d vs_sk_ax=%s",
            peer_id, segment_id, len(refs), len(highlights),
            "yes" if vs_sk_ax else "no",
        )

        return {
            "linked": True,
            "reason": "ok",
            "segment": seg_obj,
            "financial_refs": refs,
            "highlights": highlights,
            "headcount_delta": head_delta,
            "vs_sk_ax": vs_sk_ax,
            "_data_source": data.get("_source", "unknown"),
        }

    @staticmethod
    def _empty(reason: str) -> dict[str, Any]:
        return {
            "linked": False,
            "reason": reason,
            "segment": None,
            "financial_refs": [],
            "highlights": [],
            "headcount_delta": None,
        }


__all__ = ["FinancialLinkerAgent"]
