"""IR/공시 PDF 파싱 에이전트 (v3 §5.1 차별화 핵심) — PoC 스켈레톤.

목적:
  IR 분기·연간 PDF에서 매출·영업이익·사업부 매출 비중·헤드카운트를 추출.
  추출 결과는 peer_financials 테이블(JSONB) 적재 — FinancialLinkerAgent의 입력.

PoC 단계 (W4):
  - 텍스트 레이어 추출 + 정규식 기반 핵심 지표 후보 추출
  - 표 영역은 좌표 기반 line/word 그룹핑으로 raw rows만 구성
  - LLM(GPT-4o)으로 raw rows → 구조화 JSON (선택)

W5에서 확장:
  - 차트 OCR/표 추출 정밀화 (pdfplumber 또는 camelot 검토)
  - DART 공시번호 자동 매칭 (filename 또는 metadata)
  - 4개 Peer사 분기 IR 자동 적재 파이프라인
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# 핵심 재무 지표 정규식 — 한국 IR 자료 관용 표현 기준
_REVENUE_PATTERNS = [
    re.compile(r"(?:매출액?|Revenue|총매출)\s*[:：]?\s*([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)"),
    re.compile(r"([\d,]+(?:\.\d+)?)\s*(억원|조원)\s*(?:매출|sales)", re.IGNORECASE),
]
_OPERATING_PROFIT_PATTERNS = [
    re.compile(
        r"(?:영업이익|Operating\s*Profit|OP)\s*[:：]?\s*([\d,]+(?:\.\d+)?)\s*(억원|조원|억|조)"
    ),
]
_PERIOD_PATTERNS = [
    re.compile(r"(20\d{2})\s*년?\s*([1-4])\s*분기"),
    re.compile(r"(20\d{2})Q([1-4])"),
    re.compile(r"FY\s*(20\d{2})\s*([1-4])Q"),
]


def _normalize_amount_krwbn(value: str, unit: str) -> float:
    """문자열 금액·단위를 억원(krwbn) 기준으로 정규화."""
    n = float(value.replace(",", ""))
    if unit in ("조원", "조"):
        return n * 10_000  # 1조 = 10000억
    return n  # 억원·억


def _extract_period(text: str) -> str | None:
    for p in _PERIOD_PATTERNS:
        m = p.search(text)
        if m:
            return f"{m.group(1)}Q{m.group(2)}"
    return None


def _first_amount(text: str, patterns: list[re.Pattern[str]]) -> float | None:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return _normalize_amount_krwbn(m.group(1), m.group(2))
    return None


class IRParserAgent:
    """단일 IR PDF에서 핵심 재무 지표 후보를 추출. (PoC)"""

    def parse(
        self,
        pdf_path: str | Path,
        *,
        peer_id: str | None = None,
        max_pages: int = 30,
    ) -> dict[str, Any]:
        """PDF를 파싱하여 후보 재무 지표를 반환.

        Returns:
            {
              "ok": bool,
              "peer_id": str | None,
              "page_count": int,
              "period": str | None,           # ex: "2026Q1"
              "revenue_total_krwbn": float | None,
              "operating_profit_krwbn": float | None,
              "candidates": [{"page", "type", "value_krwbn", "raw"}],
              "raw_text_pages": list[str],    # 디버깅·LLM 후처리용
              "warnings": list[str],
            }
        """
        path = Path(pdf_path)
        warnings: list[str] = []
        if not path.exists():
            return {"ok": False, "reason": f"파일 없음: {path}", "warnings": []}

        try:
            import pymupdf  # type: ignore
        except ImportError:
            return {"ok": False, "reason": "pymupdf 미설치 — `uv add pymupdf`", "warnings": []}

        try:
            doc = pymupdf.open(path)
        except Exception as e:
            return {"ok": False, "reason": f"PDF 열기 실패: {e}", "warnings": []}

        page_count = doc.page_count
        if page_count > max_pages:
            warnings.append(f"페이지 수 {page_count} > max_pages {max_pages}, 앞부분만 분석")

        raw_pages: list[str] = []
        candidates: list[dict[str, Any]] = []
        period: str | None = None
        revenue_total: float | None = None
        op_profit: float | None = None

        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            try:
                text = page.get_text("text") or ""
            except Exception as e:
                warnings.append(f"page {i + 1} text 추출 실패: {e}")
                continue
            raw_pages.append(text)

            if period is None:
                period = _extract_period(text)

            rev = _first_amount(text, _REVENUE_PATTERNS)
            if rev is not None and revenue_total is None:
                revenue_total = rev
                candidates.append({"page": i + 1, "type": "revenue_total", "value_krwbn": rev})

            op = _first_amount(text, _OPERATING_PROFIT_PATTERNS)
            if op is not None and op_profit is None:
                op_profit = op
                candidates.append({"page": i + 1, "type": "operating_profit", "value_krwbn": op})

        doc.close()

        if revenue_total is None:
            warnings.append("revenue_total 추출 실패 — LLM 후처리 권장")
        if op_profit is None:
            warnings.append("operating_profit 추출 실패 — LLM 후처리 권장")

        log.info(
            "IR PDF 파싱 완료 | path=%s pages=%d period=%s rev=%s op=%s",
            path.name,
            page_count,
            period,
            revenue_total,
            op_profit,
        )

        return {
            "ok": True,
            "peer_id": peer_id,
            "page_count": page_count,
            "period": period,
            "revenue_total_krwbn": revenue_total,
            "operating_profit_krwbn": op_profit,
            "candidates": candidates,
            "raw_text_pages": raw_pages,
            "warnings": warnings,
        }


__all__ = ["IRParserAgent"]
