# 작성일: 2026-06-11
# 작성자: 안가은
# 변경이력:
#   2026-06-11 안가은 — peer 키워드·SWOT 프리뷰 파이프라인 추가 및 ruff 포맷·lint 예외 처리, 대시보드 키워드 트렌드 파이프라인 갱신
"""Sector-based Naver DataLab keyword runner.

This runner keeps the existing keyword crawler helpers untouched and executes a flow where:
1. Naver DataLab keywordGroups are built from the project's sector taxonomy;
2. active sectors are read from DB first, with config fallback;
3. spike cause candidates are inferred from existing DB evidence, not web search;
4. all DataLab results are saved to raw_articles;
5. only selected sector groups are marked for the home screen.

It does not call an LLM.
"""

# ruff: noqa: E402, E501

from __future__ import annotations

import sys as _sys

_SCRIPT_DIR = __file__.rsplit("/", 1)[0]
_REMOVED_IMPORT_PATHS = []
for _path in ("", _SCRIPT_DIR):
    while _path in _sys.path:
        _sys.path.remove(_path)
        _REMOVED_IMPORT_PATHS.append(_path)
try:
    import keyword as _stdlib_keyword  # noqa: F401
finally:
    for _path in reversed(_REMOVED_IMPORT_PATHS):
        _sys.path.insert(0, _path)

import argparse
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.config.sectors import SECTOR_KEYWORDS  # noqa: E402
from src.crawler.base import CrawlRunContext  # noqa: E402
from src.crawler.sources.keyword import (  # noqa: E402
    NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP,
    build_company_keyword_groups,
    collect_naver_datalab_trends,
    datalab_rows_to_articles,
    dedupe_texts,
    mark_relative_peak_candidates,
    save_json,
)
from src.db.article_store import save_articles  # noqa: E402
from src.db.postgres import SessionLocal  # noqa: E402

log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
DEFAULT_HOME_LIMIT = 4
DEFAULT_SECTOR_IDS = ("ax", "security", "infra", "deal")
DEFAULT_FIXED_HOME_KEYWORDS = ("AX", "사이버보안", "인프라", "수주")
DEFAULT_MAX_DRIVER_KEYWORDS = 6
DEFAULT_SPIKE_DELTA_THRESHOLD = 50.0
DB_STATEMENT_TIMEOUT_MS = 5000
SECTOR_GROUP_DISPLAY_NAMES = {
    "ax": "AX",
    "security": "사이버보안",
    "infra": "인프라",
    "deal": "수주",
}
SECTOR_NAME_ALIASES = {
    "ax": "ax",
    "ai": "ax",
    "ai_tech": "ax",
    "security": "security",
    "secure": "security",
    "보안": "security",
    "infra": "infra",
    "infrastructure": "infra",
    "인프라": "infra",
    "deal": "deal",
    "sales": "deal",
    "contract": "deal",
    "수주": "deal",
    "other": "other",
    "general": "other",
}

STOPWORDS = {
    "기자",
    "뉴스",
    "기업",
    "사업",
    "서비스",
    "시장",
    "기술",
    "산업",
    "분야",
    "올해",
    "최근",
    "관련",
    "기반",
    "확대",
    "추진",
    "제공",
    "지원",
    "강화",
    "도입",
    "개발",
    "운영",
    "플랫폼",
    "솔루션",
    "디지털",
    "전환",
    "데이터",
    "인공지능",
}

ALLOWLIST_HINTS = (
    "AI",
    "LLM",
    "클라우드",
    "보안",
    "데이터",
    "센터",
    "에이전트",
    "팩토리",
    "스마트",
    "자동화",
    "DX",
    "AX",
    "RAG",
    "GPU",
    "로봇",
    "양자",
    "제로트러스트",
)

TOKEN_PATTERN = re.compile(
    r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9+.#/&-]*(?:\s+[가-힣A-Za-z0-9][가-힣A-Za-z0-9+.#/&-]*){0,2}"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DB/설정 섹터 기반 keywordGroups로 Naver DataLab 상대 검색지수를 조회합니다."
    )
    parser.add_argument("--days", type=int, default=14, help="DB 섹터/원인 후보를 찾을 기간")
    parser.add_argument("--lookback-days", type=int, default=14, help="DataLab 조회 기간")
    parser.add_argument("--min-mentions", type=int, default=5, help="자동 추가 최소 등장 횟수")
    parser.add_argument(
        "--min-sources", type=int, default=2, help="자동 추가 최소 source_name 개수"
    )
    parser.add_argument("--candidate-limit", type=int, default=15, help="자동 추가 후보 최대 개수")
    parser.add_argument(
        "--sectors",
        default="db",
        help=(
            "DataLab에 조회할 섹터 ID. 기본 db=최근 DB에서 활성 섹터만 사용. "
            "예: ax,security,infra,deal 또는 all"
        ),
    )
    parser.add_argument(
        "--include-company",
        action="store_true",
        help="디버깅용: 기존 회사명 alias 그룹도 DataLab 조회에 포함합니다. 기본 수집은 섹터만 사용합니다.",
    )
    parser.add_argument(
        "--include-auto-discovered",
        action="store_true",
        help="섹터 그룹 외에 최근 raw_articles 기반 자동발굴 후보 그룹을 추가합니다. 기본은 섹터만 수집합니다.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과 JSON 경로. 생략하면 src/crawler/crawler_results 아래에 저장합니다.",
    )
    parser.add_argument(
        "--dry-keywords",
        action="store_true",
        help="DataLab API 호출 없이 최종 keywordGroups만 출력합니다.",
    )
    parser.add_argument(
        "--print-all-ratios",
        action="store_true",
        help="DataLab에서 받은 기간별 검색 지수 ratio를 전부 출력합니다.",
    )
    parser.add_argument(
        "--no-print-ratios",
        action="store_true",
        help="DataLab 검색 지수 화면 출력을 생략하고 JSON 저장만 합니다.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="raw_articles DB 저장을 건너뜁니다.",
    )
    parser.add_argument(
        "--home-limit",
        type=int,
        default=DEFAULT_HOME_LIMIT,
        help="홈 화면에 노출할 급등 키워드 개수. 기본 4.",
    )
    parser.add_argument(
        "--fixed-home-keywords",
        default=",".join(DEFAULT_FIXED_HOME_KEYWORDS),
        help="홈 화면에서 감시할 고정 groupName 목록. 쉼표로 구분합니다.",
    )
    parser.add_argument(
        "--spike-delta-threshold",
        type=float,
        default=DEFAULT_SPIKE_DELTA_THRESHOLD,
        help="급등 후보로 볼 최소 전일 대비 검색지수 상승폭(pt). 기본 50.",
    )
    parser.add_argument(
        "--cause-window-days",
        type=int,
        default=1,
        help="급등일 직전부터 당일까지 원인 후보를 탐색할 기간",
    )
    parser.add_argument(
        "--cause-baseline-days", type=int, default=14, help="급등 전 baseline 기사량 비교 기간"
    )
    parser.add_argument(
        "--max-cause-evidence", type=int, default=5, help="급등 원인 후보 근거 최대 출력 개수"
    )
    parser.add_argument(
        "--max-driver-keywords",
        type=int,
        default=DEFAULT_MAX_DRIVER_KEYWORDS,
        help="급등 원인 분석에서 세부 키워드별로 추가 분석할 최대 키워드 수",
    )
    parser.add_argument(
        "--db-statement-timeout-ms",
        type=int,
        default=DB_STATEMENT_TIMEOUT_MS,
        help="급등 원인 후보 DB 조회 1건당 statement_timeout(ms). 0이면 비활성화합니다.",
    )
    parser.add_argument(
        "--skip-cause-analysis",
        action="store_true",
        help="DB 기반 급등 원인 후보 분석을 건너뜁니다.",
    )
    return parser.parse_args(argv)


def configured_sector_ids() -> list[str]:
    return [sector_id for sector_id in DEFAULT_SECTOR_IDS if sector_id in SECTOR_KEYWORDS]


def normalize_sector_id(value: Any) -> str | None:
    text_value = str(value or "").strip()
    if not text_value:
        return None

    lowered = text_value.lower()
    if lowered in SECTOR_KEYWORDS:
        return lowered
    if lowered in SECTOR_NAME_ALIASES:
        return SECTOR_NAME_ALIASES[lowered]

    for sector_id, info in SECTOR_KEYWORDS.items():
        if text_value == info.get("name_ko"):
            return sector_id

    return None


def resolve_sector_ids(sectors_arg: str, *, days: int) -> list[str]:
    requested = [item.strip() for item in sectors_arg.split(",") if item.strip()]

    if not requested or requested == ["db"]:
        return fetch_active_sector_ids(days=days)

    if len(requested) == 1 and requested[0].lower() == "all":
        return configured_sector_ids()

    sector_ids: list[str] = []
    for item in requested:
        sector_id = normalize_sector_id(item)
        if not sector_id or sector_id == "other":
            continue
        if sector_id not in sector_ids:
            sector_ids.append(sector_id)

    return sector_ids or configured_sector_ids()


def fetch_active_sector_ids(*, days: int) -> list[str]:
    cutoff = datetime.now(KST) - timedelta(days=days)
    candidates: list[str] = []

    candidates.extend(
        _query_scalar_values(
            """
        SELECT DISTINCT primary_keyword_category AS sector
        FROM card_news
        WHERE created_at >= :cutoff
          AND primary_keyword_category IS NOT NULL
          AND primary_keyword_category <> ''
        """,
            {"cutoff": cutoff},
        )
    )
    candidates.extend(
        _query_scalar_values(
            """
        SELECT DISTINCT unnest(sectors) AS sector
        FROM integrated_issues
        WHERE created_at >= :cutoff
          AND is_current = TRUE
          AND sectors IS NOT NULL
        """,
            {"cutoff": cutoff},
        )
    )
    candidates.extend(
        _query_scalar_values(
            """
        SELECT DISTINCT jsonb_array_elements_text(matched_sectors) AS sector
        FROM raw_articles
        WHERE collected_at >= :cutoff
          AND jsonb_typeof(matched_sectors) = 'array'
          AND jsonb_array_length(matched_sectors) > 0
        """,
            {"cutoff": cutoff},
        )
    )
    candidates.extend(
        _query_scalar_values(
            """
        SELECT DISTINCT sector
        FROM sector_pulse
        WHERE week_start >= CAST(:cutoff AS date)
          AND sector IS NOT NULL
          AND sector <> ''
        """,
            {"cutoff": cutoff},
        )
    )

    sector_ids: list[str] = []
    for candidate in candidates:
        sector_id = normalize_sector_id(candidate)
        if not sector_id or sector_id == "other":
            continue
        if sector_id not in sector_ids and sector_id in SECTOR_KEYWORDS:
            sector_ids.append(sector_id)

    return [sector_id for sector_id in configured_sector_ids() if sector_id in sector_ids]


def _query_scalar_values(sql: str, params: dict[str, Any]) -> list[str]:
    try:
        with SessionLocal() as db:
            result = db.execute(text(sql), params)
            return [str(row[0]) for row in result if row[0] is not None]
    except Exception as exc:
        log.debug("섹터 후보 DB 조회 skip: %s", exc)
        return []


def build_sector_keyword_groups(sector_ids: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    for sector_id in sector_ids:
        if sector_id == "other":
            continue
        sector = SECTOR_KEYWORDS.get(sector_id)
        if not sector:
            continue

        group_name = SECTOR_GROUP_DISPLAY_NAMES.get(sector_id) or str(
            sector.get("name_ko") or sector_id
        )
        keywords = dedupe_texts(sector.get("keywords", []))[:NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP]
        if not keywords:
            continue

        groups.append(
            {
                "groupName": group_name,
                "keywords": keywords,
                "metadata": {
                    "mode": "sector_taxonomy",
                    "sector_id": sector_id,
                    "sector_name_ko": group_name,
                    "source": "src.config.sectors.SECTOR_KEYWORDS",
                },
            }
        )

    return groups


def discover_candidate_keywords(
    *,
    days: int,
    min_mentions: int,
    min_sources: int,
    limit: int,
    existing_keywords: set[str],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    rows = _fetch_recent_article_texts(days)
    if not rows:
        return []

    counts: Counter[str] = Counter()
    sources_by_keyword: dict[str, set[str]] = defaultdict(set)
    examples_by_keyword: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        source_name = str(row.get("source_name") or "unknown")
        title = str(row.get("title") or "")
        content = str(row.get("content") or "")
        text_value = f"{title}\n{content[:3000]}"

        for candidate in extract_keyword_candidates(text_value):
            normalized = normalize_keyword(candidate)
            if not is_candidate_keyword(normalized, existing_keywords):
                continue

            counts[normalized] += 1
            sources_by_keyword[normalized].add(source_name)

            if len(examples_by_keyword[normalized]) < 3 and title:
                examples_by_keyword[normalized].append(title[:160])

    ranked: list[dict[str, Any]] = []
    for keyword, mention_count in counts.items():
        source_count = len(sources_by_keyword[keyword])
        if mention_count < min_mentions or source_count < min_sources:
            continue

        ranked.append(
            {
                "keyword": keyword,
                "mention_count": mention_count,
                "source_count": source_count,
                "score": mention_count + source_count * 2,
                "sources": sorted(sources_by_keyword[keyword]),
                "examples": examples_by_keyword[keyword],
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["keyword"]))
    return ranked[:limit]


def _fetch_recent_article_texts(days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(KST) - timedelta(days=days)
    query = text(
        """
        SELECT source_name, title, content
        FROM raw_articles
        WHERE collected_at >= :cutoff
          AND source_name <> 'naver_datalab'
          AND (
              title IS NOT NULL
              OR content IS NOT NULL
          )
        ORDER BY collected_at DESC
        LIMIT 1000
        """
    )

    try:
        with SessionLocal() as db:
            result = db.execute(query, {"cutoff": cutoff})
            return [dict(row._mapping) for row in result]
    except Exception as exc:
        log.warning("DB에서 신규 키워드 후보를 읽지 못했습니다: %s", exc)
        return []


def extract_keyword_candidates(text_value: str) -> list[str]:
    candidates: list[str] = []
    for match in TOKEN_PATTERN.finditer(text_value):
        raw = match.group(0)
        for candidate in expand_candidate(raw):
            candidates.append(candidate)
    return candidates


def expand_candidate(value: str) -> list[str]:
    value = normalize_keyword(value)
    if not value:
        return []

    parts = value.split()
    if len(parts) <= 1:
        return [value]

    expanded = [value]
    expanded.extend(part for part in parts if len(part) >= 2)
    return expanded


def normalize_keyword(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = value.strip(".,;:!?()[]{}<>\"'“”‘’")
    return value


def is_candidate_keyword(keyword: str, existing_keywords: set[str]) -> bool:
    if not keyword:
        return False
    if keyword in existing_keywords:
        return False
    if keyword in STOPWORDS:
        return False
    if len(keyword) < 2 or len(keyword) > 30:
        return False
    if keyword.isdigit():
        return False
    if not any(hint.lower() in keyword.lower() for hint in ALLOWLIST_HINTS):
        return False
    return True


def build_keyword_groups(
    *,
    sector_ids: list[str],
    include_company: bool,
    discovered_keywords: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    if include_company:
        groups.extend(build_company_keyword_groups())

    groups.extend(build_sector_keyword_groups(sector_ids))

    if discovered_keywords:
        groups.append(
            {
                "groupName": "자동발굴 후보",
                "keywords": [
                    item["keyword"]
                    for item in discovered_keywords[:NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP]
                ],
                "metadata": {
                    "mode": "auto_discovered",
                    "candidates": discovered_keywords,
                },
            }
        )

    return normalize_keyword_groups(groups)


def normalize_keyword_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for group in groups:
        group_name = str(group.get("groupName") or "").strip()
        keywords = dedupe_texts(group.get("keywords") or [])[:NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP]
        if not group_name or not keywords:
            continue

        normalized.append(
            {
                **group,
                "groupName": group_name,
                "keywords": keywords,
            }
        )
    return normalized


def existing_keyword_set(*, sector_ids: list[str], include_company: bool) -> set[str]:
    groups = build_sector_keyword_groups(sector_ids)
    if include_company:
        groups.extend(build_company_keyword_groups())

    keywords: set[str] = set()
    for group in groups:
        keywords.update(str(keyword).strip() for keyword in group.get("keywords", []) if keyword)
    return keywords


def print_keyword_groups(groups: list[dict[str, Any]]) -> None:
    print("\nFinal DataLab keywordGroups")
    print("=" * 80)
    for group in groups:
        print(f"- {group['groupName']}: {', '.join(group['keywords'])}")


def print_search_index_rows(rows: list[dict[str, Any]], *, print_all: bool = False) -> None:
    if not rows:
        print("\nNaver DataLab search index: no rows")
        return

    if print_all:
        selected_rows = sorted(
            rows,
            key=lambda row: (str(row.get("group_name") or ""), str(row.get("period") or "")),
        )
        title = "Naver DataLab search index ratios"
    else:
        selected_rows = latest_row_per_group(rows)
        title = "Naver DataLab latest search index ratios"

    print(f"\n{title}")
    print("=" * 80)
    print(f"{'group':24s} {'period':12s} {'ratio':>8s} peak")
    print("-" * 80)
    for row in selected_rows:
        group_name = str(row.get("group_name") or "")
        period = str(row.get("period") or "")
        ratio = row.get("ratio")
        peak = "Y" if row.get("is_peak_candidate") else ""
        print(f"{group_name[:24]:24s} {period:12s} {format_ratio(ratio):>8s} {peak}")

    if not print_all:
        print("\n전체 기간별 ratio까지 보려면 --print-all-ratios 옵션을 붙이세요.")


def select_home_rows(
    rows: list[dict[str, Any]],
    *,
    fixed_keywords: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    latest_rows = latest_row_per_group(rows)
    rising_rows = [row for row in latest_rows if is_rising_spike(row)]
    latest_by_group = {str(row.get("group_name") or ""): row for row in rising_rows}

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for group_name in fixed_keywords:
        row = latest_by_group.get(group_name)
        if row and group_name not in seen:
            selected.append({**row, "home_reason": "fixed", "home_rank": len(selected) + 1})
            seen.add(group_name)
        if len(selected) >= limit:
            return selected

    return selected


def is_rising_spike(row: dict[str, Any]) -> bool:
    if not row.get("is_peak_candidate"):
        return False
    peak = row.get("peak") if isinstance(row.get("peak"), dict) else {}
    delta = parse_float(peak.get("delta"))
    return delta is not None and delta > 0


def annotate_home_display(
    rows: list[dict[str, Any]],
    home_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    home_by_key = {
        (str(row.get("group_name") or ""), str(row.get("period") or "")): row for row in home_rows
    }
    annotated: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("group_name") or ""), str(row.get("period") or ""))
        home_row = home_by_key.get(key)
        annotated.append(
            {
                **row,
                "home_display": home_row is not None,
                "home_reason": home_row.get("home_reason") if home_row else None,
                "home_rank": home_row.get("home_rank") if home_row else None,
            }
        )
    return annotated


def annotate_group_metadata(
    rows: list[dict[str, Any]], groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups_by_name = {str(group.get("groupName") or ""): group for group in groups}
    annotated: list[dict[str, Any]] = []

    for row in rows:
        group = groups_by_name.get(str(row.get("group_name") or ""), {})
        metadata = dict(group.get("metadata") or {})
        annotated.append(
            {
                **row,
                "keywords": row.get("keywords") or group.get("keywords") or [],
                "keyword_group_metadata": metadata,
                "sector_id": metadata.get("sector_id"),
                "sector_name_ko": metadata.get("sector_name_ko"),
            }
        )

    return annotated


def analyze_spike_causes(
    rows: list[dict[str, Any]],
    *,
    window_days: int,
    baseline_days: int,
    max_evidence: int,
    max_driver_keywords: int,
) -> list[dict[str, Any]]:
    analyzed: list[dict[str, Any]] = []

    for row in rows:
        if not is_rising_spike(row):
            analyzed.append(row)
            continue

        analyzed.append(
            {
                **row,
                "cause_analysis": build_cause_analysis(
                    row,
                    window_days=window_days,
                    baseline_days=baseline_days,
                    max_evidence=max_evidence,
                    max_driver_keywords=max_driver_keywords,
                ),
            }
        )

    return analyzed


def build_cause_analysis(
    row: dict[str, Any],
    *,
    window_days: int,
    baseline_days: int,
    max_evidence: int,
    max_driver_keywords: int,
) -> dict[str, Any]:
    period_date = parse_period_date(str(row.get("period") or ""))
    keywords = cause_keywords(row)

    if period_date is None or not keywords:
        return {
            "status": "skipped",
            "reason_summary": "급등일 또는 키워드 그룹 정보가 부족해 DB 기반 원인 후보를 만들 수 없습니다.",
            "analysis_sources": [],
            "is_web_search_used": False,
        }

    start_date = period_date - timedelta(days=max(window_days, 0))
    end_date = period_date
    baseline_start = start_date - timedelta(days=max(baseline_days, 1))
    baseline_end = start_date - timedelta(days=1)
    sector_id = str(row.get("sector_id") or "").strip()
    sector_name = str(row.get("sector_name_ko") or row.get("group_name") or "").strip()
    keyword_drivers = build_keyword_drivers(
        row=row,
        start_date=start_date,
        end_date=end_date,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        baseline_days=baseline_days,
        max_evidence=max_evidence,
        max_driver_keywords=max_driver_keywords,
    )

    raw_window_count = count_raw_article_mentions(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
    )
    raw_baseline_count = count_raw_article_mentions(
        keywords=keywords,
        start_date=baseline_start,
        end_date=baseline_end,
    )
    raw_evidence = query_raw_article_evidence(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        max_evidence=max_evidence,
    )
    card_evidence = query_card_news_evidence(
        keywords=keywords,
        sector_id=sector_id,
        sector_name=sector_name,
        start_date=start_date,
        end_date=end_date,
        max_evidence=max_evidence,
    )
    issue_evidence = query_integrated_issue_evidence(
        keywords=keywords,
        sector_id=sector_id,
        sector_name=sector_name,
        start_date=start_date,
        end_date=end_date,
        max_evidence=max_evidence,
    )
    signal_evidence = query_business_signal_evidence(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        max_evidence=max_evidence,
    )

    baseline_daily_avg = raw_baseline_count / max(baseline_days, 1)
    window_daily_avg = raw_window_count / max((end_date - start_date).days + 1, 1)
    evidence_count = (
        len(raw_evidence) + len(card_evidence) + len(issue_evidence) + len(signal_evidence)
    )
    driver_evidence_count = sum(
        int(driver.get("evidence_count") or 0) for driver in keyword_drivers
    )

    status = "candidate" if evidence_count or driver_evidence_count else "insufficient_db_evidence"
    summary = build_cause_summary(
        row=row,
        raw_window_count=raw_window_count,
        raw_baseline_count=raw_baseline_count,
        window_daily_avg=window_daily_avg,
        baseline_daily_avg=baseline_daily_avg,
        card_count=len(card_evidence),
        issue_count=len(issue_evidence),
        signal_count=len(signal_evidence),
        evidence_count=evidence_count,
        keyword_drivers=keyword_drivers,
    )

    return {
        "status": status,
        "reason_summary": summary,
        "is_web_search_used": False,
        "method": (
            "DB-only 분석: 급등일 직전부터 당일까지 raw_articles 키워드 언급량을 직전 baseline과 비교하고, "
            "같은 기간 card_news/integrated_issues/raw_article_business_signals 근거를 매칭했습니다."
        ),
        "period": str(row.get("period") or ""),
        "window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "baseline_start_date": baseline_start.isoformat(),
            "baseline_end_date": baseline_end.isoformat(),
        },
        "signals": {
            "raw_article_window_count": raw_window_count,
            "raw_article_baseline_count": raw_baseline_count,
            "raw_article_window_daily_avg": round(window_daily_avg, 3),
            "raw_article_baseline_daily_avg": round(baseline_daily_avg, 3),
            "card_news_count": len(card_evidence),
            "integrated_issue_count": len(issue_evidence),
            "business_signal_count": len(signal_evidence),
            "keyword_driver_evidence_count": driver_evidence_count,
        },
        "keyword_drivers": keyword_drivers,
        "analysis_sources": [
            source
            for source, items in (
                ("raw_articles", raw_evidence),
                ("card_news", card_evidence),
                ("integrated_issues", issue_evidence),
                ("raw_article_business_signals", signal_evidence),
            )
            if items
        ],
        "evidence": {
            "raw_articles": raw_evidence,
            "card_news": card_evidence,
            "integrated_issues": issue_evidence,
            "raw_article_business_signals": signal_evidence,
        },
    }


def build_cause_summary(
    *,
    row: dict[str, Any],
    raw_window_count: int,
    raw_baseline_count: int,
    window_daily_avg: float,
    baseline_daily_avg: float,
    card_count: int,
    issue_count: int,
    signal_count: int,
    evidence_count: int,
    keyword_drivers: list[dict[str, Any]] | None = None,
) -> str:
    group_name = str(row.get("group_name") or "키워드")
    delta = (row.get("peak") or {}).get("delta") if isinstance(row.get("peak"), dict) else None
    delta_text = f" 전일 대비 +{format_ratio(delta)}pt" if delta is not None else ""
    driver_text = format_keyword_driver_summary(keyword_drivers or [])

    if evidence_count <= 0 and not driver_text:
        return (
            f"{group_name} 검색지수{delta_text} 급등은 감지됐지만, 같은 기간 내부 DB에서 "
            "직접 연결할 원문/카드/통합이슈 근거가 충분하지 않습니다."
        )

    pieces: list[str] = []
    if raw_window_count > 0:
        if baseline_daily_avg > 0:
            multiple = window_daily_avg / baseline_daily_avg
            pieces.append(
                f"원문 언급 {raw_window_count}건(직전 baseline {raw_baseline_count}건, 일평균 {multiple:.1f}배)"
            )
        else:
            pieces.append(f"원문 언급 {raw_window_count}건")
    if card_count:
        pieces.append(f"카드뉴스 {card_count}건")
    if issue_count:
        pieces.append(f"통합이슈 {issue_count}건")
    if signal_count:
        pieces.append(f"비즈니스 시그널 {signal_count}건")

    if not pieces and driver_text:
        return f"{group_name} 검색지수{delta_text} 급등 시점의 세부 키워드 후보: {driver_text}."

    if driver_text:
        return (
            f"{group_name} 검색지수{delta_text} 급등 시점에 "
            + ", ".join(pieces)
            + f"이 확인됩니다. 세부 키워드 후보: {driver_text}."
        )

    return f"{group_name} 검색지수{delta_text} 급등 시점에 " + ", ".join(pieces) + "이 확인됩니다."


def build_keyword_drivers(
    *,
    row: dict[str, Any],
    start_date: date,
    end_date: date,
    baseline_start: date,
    baseline_end: date,
    baseline_days: int,
    max_evidence: int,
    max_driver_keywords: int,
) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    window_days = max((end_date - start_date).days + 1, 1)
    evidence_limit = max(1, min(max_evidence, 2))

    for keyword in driver_keywords(row, max_keywords=max_driver_keywords):
        raw_window_count = count_raw_article_mentions(
            keywords=[keyword],
            start_date=start_date,
            end_date=end_date,
        )
        raw_baseline_count = count_raw_article_mentions(
            keywords=[keyword],
            start_date=baseline_start,
            end_date=baseline_end,
        )
        raw_evidence = query_raw_article_evidence(
            keywords=[keyword],
            start_date=start_date,
            end_date=end_date,
            max_evidence=evidence_limit,
        )
        card_evidence = query_card_news_evidence(
            keywords=[keyword],
            sector_id="",
            sector_name="",
            start_date=start_date,
            end_date=end_date,
            max_evidence=evidence_limit,
            include_sector_match=False,
        )
        issue_evidence = query_integrated_issue_evidence(
            keywords=[keyword],
            sector_id="",
            sector_name="",
            start_date=start_date,
            end_date=end_date,
            max_evidence=evidence_limit,
            include_sector_match=False,
        )
        signal_evidence = query_business_signal_evidence(
            keywords=[keyword],
            start_date=start_date,
            end_date=end_date,
            max_evidence=evidence_limit,
        )

        baseline_daily_avg = raw_baseline_count / max(baseline_days, 1)
        window_daily_avg = raw_window_count / window_days
        raw_lift = (
            round(window_daily_avg / baseline_daily_avg, 3) if baseline_daily_avg > 0 else None
        )
        evidence_count = (
            len(raw_evidence) + len(card_evidence) + len(issue_evidence) + len(signal_evidence)
        )
        expected_window_count = baseline_daily_avg * window_days
        raw_delta = raw_window_count - expected_window_count
        score = raw_delta + evidence_count * 3

        if raw_window_count <= 0 and evidence_count <= 0:
            continue

        drivers.append(
            {
                "keyword": keyword,
                "score": round(score, 3),
                "raw_article_window_count": raw_window_count,
                "raw_article_baseline_count": raw_baseline_count,
                "raw_article_window_daily_avg": round(window_daily_avg, 3),
                "raw_article_baseline_daily_avg": round(baseline_daily_avg, 3),
                "raw_article_lift": raw_lift,
                "evidence_count": evidence_count,
                "analysis_sources": [
                    source
                    for source, items in (
                        ("raw_articles", raw_evidence),
                        ("card_news", card_evidence),
                        ("integrated_issues", issue_evidence),
                        ("raw_article_business_signals", signal_evidence),
                    )
                    if items
                ],
                "evidence": {
                    "raw_articles": raw_evidence,
                    "card_news": card_evidence,
                    "integrated_issues": issue_evidence,
                    "raw_article_business_signals": signal_evidence,
                },
            }
        )

    drivers.sort(
        key=lambda item: (
            -float(item.get("score") or 0),
            -int(item.get("evidence_count") or 0),
            str(item.get("keyword") or ""),
        )
    )
    return drivers[: max(3, min(max_evidence, 5))]


def format_keyword_driver_summary(drivers: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for driver in drivers[:3]:
        keyword = str(driver.get("keyword") or "")
        if not keyword:
            continue
        window_count = int(driver.get("raw_article_window_count") or 0)
        lift = driver.get("raw_article_lift")
        evidence_count = int(driver.get("evidence_count") or 0)
        if lift:
            pieces.append(f"{keyword}(원문 {window_count}건, baseline 대비 {float(lift):.1f}배)")
        elif evidence_count:
            pieces.append(f"{keyword}(근거 {evidence_count}건)")
        else:
            pieces.append(f"{keyword}(원문 {window_count}건)")
    return ", ".join(pieces)


def cause_keywords(row: dict[str, Any]) -> list[str]:
    values = [str(row.get("group_name") or "")]
    values.extend(str(keyword) for keyword in row.get("keywords") or [])
    return dedupe_texts([value for value in values if value])[:10]


def driver_keywords(row: dict[str, Any], *, max_keywords: int) -> list[str]:
    group_name = str(row.get("group_name") or "").strip()
    values = [str(keyword) for keyword in row.get("keywords") or []]
    if group_name:
        values.insert(0, group_name)
    limit = max(1, min(max_keywords, NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP))
    return dedupe_texts([value for value in values if value])[:limit]


def parse_period_date(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def count_raw_article_mentions(
    *,
    keywords: list[str],
    start_date: date,
    end_date: date,
) -> int:
    match_sql, params = keyword_match_clause(["title", "content"], keywords, "raw_count_kw")
    if not match_sql:
        return 0

    sql = f"""
        SELECT COUNT(*) AS count
        FROM raw_articles
        WHERE source_name <> 'naver_datalab'
          AND COALESCE(published_at::date, collected_at::date) BETWEEN :start_date AND :end_date
          AND ({match_sql})
    """
    params.update({"start_date": start_date, "end_date": end_date})

    try:
        with SessionLocal() as db:
            apply_statement_timeout(db)
            return int(db.execute(text(sql), params).scalar() or 0)
    except Exception as exc:
        log.debug("raw_articles 언급량 조회 skip: %s", exc)
        return 0


def query_raw_article_evidence(
    *,
    keywords: list[str],
    start_date: date,
    end_date: date,
    max_evidence: int,
) -> list[dict[str, Any]]:
    match_sql, params = keyword_match_clause(["title", "content"], keywords, "raw_kw")
    if not match_sql:
        return []

    sql = f"""
        SELECT id, source_name, title, url, published_at, collected_at,
               LEFT(COALESCE(content, ''), 220) AS snippet
        FROM raw_articles
        WHERE source_name <> 'naver_datalab'
          AND COALESCE(published_at::date, collected_at::date) BETWEEN :start_date AND :end_date
          AND ({match_sql})
        ORDER BY COALESCE(published_at, collected_at) DESC NULLS LAST, id DESC
        LIMIT :limit
    """
    params.update({"start_date": start_date, "end_date": end_date, "limit": max_evidence})
    return query_dicts(sql, params)


def query_card_news_evidence(
    *,
    keywords: list[str],
    sector_id: str,
    sector_name: str,
    start_date: date,
    end_date: date,
    max_evidence: int,
    include_sector_match: bool = True,
) -> list[dict[str, Any]]:
    match_sql, params = keyword_match_clause(
        ["title", "array_to_string(summary_lines, ' ')"], keywords, "card_kw"
    )
    if not match_sql:
        return []
    sector_checks = [
        "LOWER(COALESCE(primary_keyword_category, '')) = :sector_id",
        "LOWER(COALESCE(primary_keyword_category, '')) = :sector_name",
        "LOWER(COALESCE(primary_keyword_category, '')) = :group_name",
    ]
    filters = (
        f"({' OR '.join(sector_checks)} OR ({match_sql}))"
        if include_sector_match
        else f"({match_sql})"
    )
    sql = f"""
        SELECT id, title, event_type, primary_keyword_category AS sector,
               importance_score, created_at
        FROM card_news
        WHERE created_at::date BETWEEN :start_date AND :end_date
          AND {filters}
        ORDER BY importance_score DESC NULLS LAST, created_at DESC
        LIMIT :limit
    """
    params.update(
        {
            "sector_id": sector_id.lower(),
            "sector_name": sector_name.lower(),
            "group_name": sector_name.lower(),
            "start_date": start_date,
            "end_date": end_date,
            "limit": max_evidence,
        }
    )
    return query_dicts(sql, params)


def query_integrated_issue_evidence(
    *,
    keywords: list[str],
    sector_id: str,
    sector_name: str,
    start_date: date,
    end_date: date,
    max_evidence: int,
    include_sector_match: bool = True,
) -> list[dict[str, Any]]:
    match_sql, params = keyword_match_clause(
        ["headline", "one_line_summary", "content_summary"], keywords, "issue_kw"
    )
    if not match_sql:
        return []
    filters = (
        f":sector_id = ANY(sectors) OR :sector_name = ANY(sectors) OR ({match_sql})"
        if include_sector_match
        else f"({match_sql})"
    )
    sql = f"""
        SELECT id::text AS id, headline, one_line_summary, event_type,
               sectors, confidence, created_at
        FROM integrated_issues
        WHERE created_at::date BETWEEN :start_date AND :end_date
          AND is_current = TRUE
          AND ({filters})
        ORDER BY confidence DESC NULLS LAST, created_at DESC
        LIMIT :limit
    """
    params.update(
        {
            "sector_id": sector_id,
            "sector_name": sector_name,
            "start_date": start_date,
            "end_date": end_date,
            "limit": max_evidence,
        }
    )
    return query_dicts(sql, params)


def query_business_signal_evidence(
    *,
    keywords: list[str],
    start_date: date,
    end_date: date,
    max_evidence: int,
) -> list[dict[str, Any]]:
    match_sql, params = keyword_match_clause(
        ["business_area", "signal_type", "summary", "evidence_text"],
        keywords,
        "signal_kw",
    )
    if not match_sql:
        return []

    sql = f"""
        SELECT id, raw_article_id, peer_id, business_area, signal_type,
               sentiment, summary, evidence_text, confidence, created_at
        FROM raw_article_business_signals
        WHERE created_at::date BETWEEN :start_date AND :end_date
          AND ({match_sql})
        ORDER BY confidence DESC NULLS LAST, created_at DESC
        LIMIT :limit
    """
    params.update({"start_date": start_date, "end_date": end_date, "limit": max_evidence})
    return query_dicts(sql, params)


def keyword_match_clause(
    columns: list[str], keywords: list[str], prefix: str
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    for index, keyword in enumerate(dedupe_texts(keywords)[:10]):
        if not keyword:
            continue
        param_name = f"{prefix}_{index}"
        params[param_name] = f"%{keyword}%"
        clauses.extend(f"COALESCE({column}, '') ILIKE :{param_name}" for column in columns)

    return " OR ".join(clauses), params


def query_dicts(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        with SessionLocal() as db:
            apply_statement_timeout(db)
            result = db.execute(text(sql), params)
            return [to_json_safe(dict(row._mapping)) for row in result]
    except Exception as exc:
        log.debug("급등 원인 후보 DB 조회 skip: %s", exc)
        return []


def apply_statement_timeout(db: Any) -> None:
    if DB_STATEMENT_TIMEOUT_MS <= 0:
        return
    try:
        db.execute(
            text("SELECT set_config('statement_timeout', :timeout_ms, true)"),
            {"timeout_ms": f"{DB_STATEMENT_TIMEOUT_MS}ms"},
        )
    except Exception as exc:
        log.debug("statement_timeout 설정 skip: %s", exc)


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def print_cause_analysis(rows: list[dict[str, Any]]) -> None:
    cause_rows = [row for row in rows if row.get("cause_analysis")]
    if not cause_rows:
        print("\nRising spike cause analysis: no +threshold rows")
        return

    print("\nRising spike cause analysis")
    print("=" * 80)
    for row in cause_rows:
        analysis = row.get("cause_analysis") or {}
        group_name = str(row.get("group_name") or "")
        period = str(row.get("period") or "")
        print(f"- {group_name} / {period} / {analysis.get('status')}")
        print(f"  {analysis.get('reason_summary')}")
        sources = ", ".join(analysis.get("analysis_sources") or [])
        print(f"  sources: {sources or 'none'}")
        drivers = analysis.get("keyword_drivers") or []
        if drivers:
            driver_text = format_keyword_driver_summary(drivers)
            print(f"  detail keywords: {driver_text or 'none'}")


def print_home_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("\nHome keyword selection: no +threshold rows")
        return

    print("\nHome keyword selection (+threshold rising spikes only)")
    print("=" * 80)
    print(f"{'rank':>4s} {'group':24s} {'period':12s} {'ratio':>8s} {'reason':12s} peak")
    print("-" * 80)
    for row in rows:
        rank = str(row.get("home_rank") or "")
        group_name = str(row.get("group_name") or "")
        period = str(row.get("period") or "")
        ratio = row.get("ratio")
        reason = str(row.get("home_reason") or "")
        peak = "Y" if row.get("is_peak_candidate") else ""
        print(
            f"{rank:>4s} {group_name[:24]:24s} {period:12s} {format_ratio(ratio):>8s} {reason:12s} {peak}"
        )


def latest_row_per_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_name = str(row.get("group_name") or "")
        period = str(row.get("period") or "")
        if not group_name:
            continue

        current = latest.get(group_name)
        if current is None or period > str(current.get("period") or ""):
            latest[group_name] = row

    return [latest[key] for key in sorted(latest)]


def format_ratio(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value or "")


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def save_rows_to_db(
    rows: list[dict[str, Any]],
    *,
    crawl_run_id: str | None = None,
) -> int:
    articles = datalab_rows_to_articles(rows, peer_id="keyword_trend")
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        run_context = (
            CrawlRunContext(
                collection_mode="realtime",
                crawl_run_id=crawl_run_id,
                source_name="naver_datalab",
            )
            if crawl_run_id
            else None
        )
        inserted = save_articles(articles, run_context=run_context)
    except Exception as exc:
        log.error("DataLab raw_articles 저장 전 DB 연결 확인 실패: %s", exc)
        return 0

    log.info("DataLab raw_articles 저장 완료: inserted=%s total=%s", inserted, len(articles))
    return inserted


def default_output_path() -> Path:
    output_dir = PROJECT_ROOT / "src" / "crawler" / "crawler_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    date_text = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    return output_dir / f"naver_datalab_custom_{date_text}.json"


def run(
    argv: list[str] | None = None,
    *,
    crawl_run_id: str | None = None,
) -> int:
    global DB_STATEMENT_TIMEOUT_MS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = parse_args(argv)
    DB_STATEMENT_TIMEOUT_MS = max(0, args.db_statement_timeout_ms)

    sector_ids = resolve_sector_ids(args.sectors, days=args.days)
    if not sector_ids:
        raise SystemExit(
            "최근 DB에서 활성 섹터를 찾지 못했습니다. 목업/기본 섹터를 자동 주입하지 않습니다. "
            "수동 실행이 필요하면 --sectors ax,security,infra,deal 또는 --sectors all 을 명시하세요."
        )
    existing_keywords = existing_keyword_set(
        sector_ids=sector_ids,
        include_company=args.include_company,
    )
    discovered = (
        discover_candidate_keywords(
            days=args.days,
            min_mentions=args.min_mentions,
            min_sources=args.min_sources,
            limit=args.candidate_limit,
            existing_keywords=existing_keywords,
        )
        if args.include_auto_discovered
        else []
    )
    groups = build_keyword_groups(
        sector_ids=sector_ids,
        include_company=args.include_company,
        discovered_keywords=discovered,
    )

    print(f"\nSelected sectors: {', '.join(sector_ids)}")
    print_keyword_groups(groups)
    if discovered:
        print("\nAuto-discovered candidates")
        print("=" * 80)
        for item in discovered:
            print(
                f"- {item['keyword']} "
                f"(mentions={item['mention_count']}, sources={item['source_count']})"
            )

    if args.dry_keywords:
        return 0

    raw_rows = collect_naver_datalab_trends(
        keyword_groups=groups,
        lookback_days=args.lookback_days,
    )
    raw_rows = annotate_group_metadata(raw_rows, groups)
    marked_rows = mark_relative_peak_candidates(
        raw_rows,
        min_ratio=0.0,
        min_delta=max(0.0, args.spike_delta_threshold),
    )
    if not args.skip_cause_analysis:
        marked_rows = analyze_spike_causes(
            marked_rows,
            window_days=args.cause_window_days,
            baseline_days=args.cause_baseline_days,
            max_evidence=args.max_cause_evidence,
            max_driver_keywords=args.max_driver_keywords,
        )
    fixed_home_keywords = parse_csv_values(args.fixed_home_keywords)
    home_rows = select_home_rows(
        marked_rows,
        fixed_keywords=fixed_home_keywords,
        limit=args.home_limit,
    )
    marked_rows = annotate_home_display(marked_rows, home_rows)

    if not args.no_print_ratios:
        print_search_index_rows(marked_rows, print_all=args.print_all_ratios)
    print_home_rows(home_rows)
    if not args.skip_cause_analysis:
        print_cause_analysis(marked_rows)

    marked_rows = to_json_safe(marked_rows)
    output_path = Path(args.output) if args.output else default_output_path()
    save_json(marked_rows, output_path)
    inserted = 0
    if not args.skip_db:
        inserted = save_rows_to_db(marked_rows, crawl_run_id=crawl_run_id)
    log.info(
        "custom DataLab 실행 완료: rows=%s inserted=%s output=%s",
        len(marked_rows),
        inserted,
        output_path,
    )
    return inserted


def run_scheduled(*, crawl_run_id: str | None = None) -> int:
    return run(
        [
            "--sectors",
            os.getenv("AXIS_DATALAB_SECTORS", "ax,security,infra,deal"),
            "--lookback-days",
            os.getenv("AXIS_DATALAB_LOOKBACK_DAYS", "14"),
            "--days",
            os.getenv("AXIS_DATALAB_SECTOR_DAYS", "14"),
            "--home-limit",
            os.getenv("AXIS_DATALAB_HOME_LIMIT", str(DEFAULT_HOME_LIMIT)),
            "--spike-delta-threshold",
            os.getenv(
                "AXIS_DATALAB_SPIKE_DELTA_THRESHOLD",
                str(DEFAULT_SPIKE_DELTA_THRESHOLD),
            ),
            "--fixed-home-keywords",
            os.getenv(
                "AXIS_DATALAB_FIXED_HOME_KEYWORDS",
                ",".join(DEFAULT_FIXED_HOME_KEYWORDS),
            ),
            "--no-print-ratios",
        ],
        crawl_run_id=crawl_run_id,
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
