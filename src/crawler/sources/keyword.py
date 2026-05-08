import json
import logging
import os
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.crawler.base import RawArticle
from src.crawler.base_crawler import BaseCrawler

log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

API_URL = "https://openapi.naver.com/v1/datalab/search"
SOURCE_NAME = "naver_datalab"
CRAWL_TYPE = "trend_signal"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "crawler_results"
DEFAULT_CHART_DIR = DEFAULT_OUTPUT_DIR / "charts"

DEFAULT_TIMEOUT = 10
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 1.5

DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_TIME_UNIT = "date"

NAVER_DATALAB_MAX_GROUPS = 5
NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP = 20

DEFAULT_PEAK_MIN_RATIO = 70.0
DEFAULT_PEAK_MIN_DELTA = 25.0
DEFAULT_PEAK_MAX_GAP_DAYS = 3

SECTOR_KEYWORD_GROUPS: list[dict[str, Any]] = [
    {
        "groupName": "보안",
        "keywords": [
            "보안",
            "정보보안",
            "사이버보안",
            "해킹",
            "랜섬웨어",
            "정보보호",
        ],
    },
    {
        "groupName": "인프라",
        "keywords": [
            "클라우드",
            "데이터센터",
            "IDC",
            "서버",
            "네트워크",
            "IT 인프라",
        ],
    },
    {
        "groupName": "AX(제조)",
        "keywords": [
            "스마트팩토리",
            "제조 AI",
            "공장 자동화",
            "제조 디지털 전환",
            "제조 DX",
            "산업 AI",
        ],
    },
    {
        "groupName": "수주",
        "keywords": [
            "수주",
            "계약",
            "계약 체결",
            "MOU",
            "업무협약",
            "파트너십",
        ],
    },
]


def load_naver_credentials() -> tuple[str, str]:
    load_dotenv(override=True)

    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET 환경변수가 설정되지 않았습니다.")

    client_id = client_id.strip().strip('"').strip("'")
    client_secret = client_secret.strip().strip('"').strip("'")

    return client_id, client_secret


def dedupe_texts(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        text = str(value).strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        result.append(text)

    return result


def build_sector_keyword_groups(
    sector_keyword_groups: list[dict[str, Any]] = SECTOR_KEYWORD_GROUPS,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    for sector_group in sector_keyword_groups:
        sector_name = str(sector_group.get("groupName") or "").strip()
        sector_keywords = sector_group.get("keywords") or []

        if not sector_name or not isinstance(sector_keywords, list):
            continue

        keywords = dedupe_texts(sector_keywords)[:NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP]

        if not keywords:
            continue

        groups.append(
            {
                "groupName": sector_name,
                "keywords": keywords,
                "metadata": {
                    "sector": sector_name,
                    "mode": "sector_only",
                },
            }
        )

    return groups


def build_default_keyword_groups() -> list[dict[str, Any]]:
    return build_sector_keyword_groups()


def chunk_keyword_groups(
    keyword_groups: list[dict[str, Any]],
    chunk_size: int = NAVER_DATALAB_MAX_GROUPS,
) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")

    return [
        keyword_groups[index : index + chunk_size]
        for index in range(0, len(keyword_groups), chunk_size)
    ]


def get_default_date_range(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[str, str]:
    end_date = datetime.now(KST).date()
    start_date = end_date - timedelta(days=lookback_days)

    return start_date.isoformat(), end_date.isoformat()


def build_payload(
    start_date: str,
    end_date: str,
    time_unit: str,
    keyword_groups: list[dict[str, Any]],
    device: str | None = None,
    gender: str | None = None,
    ages: list[str] | None = None,
) -> dict[str, Any]:
    api_keyword_groups = [
        {
            "groupName": str(group["groupName"]),
            "keywords": dedupe_texts(group.get("keywords", []))[
                :NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP
            ],
        }
        for group in keyword_groups
    ]

    payload: dict[str, Any] = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": api_keyword_groups,
    }

    if device:
        payload["device"] = device

    if gender:
        payload["gender"] = gender

    if ages:
        payload["ages"] = ages

    return payload


def request_datalab_api(
    payload: dict[str, Any],
    retry_count: int = DEFAULT_RETRY_COUNT,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    client_id, client_secret = load_naver_credentials()

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None

    for attempt in range(1, retry_count + 1):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 400:
                log.error("Naver DataLab 요청 형식 오류: %s", response.text)
                response.raise_for_status()

            if response.status_code == 401:
                log.error(
                    "Naver DataLab 인증 실패: Client ID/Secret 값을 확인하세요. response=%s",
                    response.text,
                )
                response.raise_for_status()

            if response.status_code == 403:
                log.error(
                    "Naver DataLab API 권한 오류: 데이터랩 API 추가 여부를 확인하세요. response=%s",
                    response.text,
                )
                response.raise_for_status()

            if response.status_code == 429:
                log.warning("Naver DataLab 호출 한도 초과 가능성: %s", response.text)

            if response.status_code >= 500:
                log.warning("Naver DataLab 서버 오류: %s", response.text)

            response.raise_for_status()

        except requests.Timeout as exc:
            last_error = exc
            log.warning("Naver DataLab 요청 timeout: attempt=%s", attempt)

        except requests.RequestException as exc:
            last_error = exc
            log.warning("Naver DataLab 요청 실패: attempt=%s, error=%s", attempt, exc)

        except ValueError as exc:
            last_error = exc
            log.error("Naver DataLab 응답 JSON 파싱 실패: %s", exc)
            break

        if attempt < retry_count:
            time.sleep(DEFAULT_RETRY_DELAY * attempt)

    raise RuntimeError(f"Naver DataLab API 요청 실패: {last_error}")


def normalize_datalab_response(
    response: dict[str, Any],
    collected_at: str,
    request_chunk_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    start_date = response.get("startDate")
    end_date = response.get("endDate")
    time_unit = response.get("timeUnit")

    for result in response.get("results", []):
        group_name = result.get("title")
        keywords = result.get("keywords", [])

        for item in result.get("data", []):
            rows.append(
                {
                    "source": SOURCE_NAME,
                    "type": CRAWL_TYPE,
                    "collected_at": collected_at,
                    "start_date": start_date,
                    "end_date": end_date,
                    "time_unit": time_unit,
                    "group_name": group_name,
                    "keywords": keywords,
                    "period": item.get("period"),
                    "ratio": item.get("ratio"),
                    "is_peak_candidate": False,
                    "peak": None,
                    "metadata": {
                        "api": "naver_datalab_search",
                        "request_chunk_index": request_chunk_index,
                        "is_relative_ratio": True,
                        "is_actual_search_count": False,
                        "ratio_description": (
                            "같은 API 요청 결과 안에서 최대 검색량을 100으로 둔 상대 검색지수"
                        ),
                        "usage_note": (
                            "실제 검색 건수나 전체 검색량 대비 점유율로 해석하지 않는다."
                        ),
                    },
                }
            )

    rows.sort(key=lambda row: (row.get("group_name") or "", row.get("period") or ""))

    return rows


def collect_naver_datalab_trends(
    keyword_groups: list[dict[str, Any]] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    time_unit: str = DEFAULT_TIME_UNIT,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    device: str | None = None,
    gender: str | None = None,
    ages: list[str] | None = None,
) -> list[dict[str, Any]]:
    if keyword_groups is None:
        keyword_groups = build_default_keyword_groups()

    if not keyword_groups:
        log.warning("수집할 keywordGroups가 없습니다.")
        return []

    if not start_date or not end_date:
        default_start_date, default_end_date = get_default_date_range(lookback_days)
        start_date = start_date or default_start_date
        end_date = end_date or default_end_date

    log.info("최종 keywordGroups=%s", [group["groupName"] for group in keyword_groups])
    log.info(
        "Naver DataLab 조회 조건: start_date=%s, end_date=%s, time_unit=%s",
        start_date,
        end_date,
        time_unit,
    )

    collected_at = datetime.now(KST).isoformat()
    normalized_results: list[dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunk_keyword_groups(keyword_groups), start=1):
        payload = build_payload(
            start_date=start_date,
            end_date=end_date,
            time_unit=time_unit,
            keyword_groups=chunk,
            device=device,
            gender=gender,
            ages=ages,
        )

        log.info(
            "Naver DataLab 수집 시작: chunk=%s, groups=%s",
            chunk_index,
            [group["groupName"] for group in chunk],
        )

        response = request_datalab_api(payload)
        rows = normalize_datalab_response(
            response=response,
            collected_at=collected_at,
            request_chunk_index=chunk_index,
        )
        normalized_results.extend(rows)

        log.info(
            "Naver DataLab 수집 완료: chunk=%s, rows=%s",
            chunk_index,
            len(rows),
        )

    normalized_results.sort(
        key=lambda row: (
            row.get("group_name") or "",
            row.get("period") or "",
        )
    )

    return normalized_results


def detect_relative_peak_candidates(
    rows: list[dict[str, Any]],
    min_ratio: float = DEFAULT_PEAK_MIN_RATIO,
    min_delta: float = DEFAULT_PEAK_MIN_DELTA,
    max_gap_days: int = DEFAULT_PEAK_MAX_GAP_DAYS,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        group_name = row.get("group_name")

        if not group_name:
            continue

        grouped.setdefault(str(group_name), []).append(row)

    candidates: list[dict[str, Any]] = []

    for group_name, group_rows in grouped.items():
        sorted_rows = sorted(group_rows, key=lambda row: row.get("period") or "")

        previous_row: dict[str, Any] | None = None

        for row in sorted_rows:
            current_ratio = parse_float(row.get("ratio"))

            if current_ratio is None:
                continue

            if previous_row is None:
                previous_row = row
                continue

            previous_ratio = parse_float(previous_row.get("ratio"))

            if previous_ratio is None:
                previous_row = row
                continue

            current_period = str(row.get("period") or "")
            previous_period = str(previous_row.get("period") or "")
            gap_days = date_gap_days(current_period, previous_period)
            delta = current_ratio - previous_ratio

            is_valid_gap = gap_days is not None and 1 <= gap_days <= max_gap_days
            is_peak_candidate = is_valid_gap and current_ratio >= min_ratio and delta >= min_delta

            if is_peak_candidate:
                candidates.append(
                    {
                        "source": SOURCE_NAME,
                        "type": "relative_trend_peak_candidate",
                        "group_name": group_name,
                        "period": current_period,
                        "ratio": current_ratio,
                        "previous_period": previous_period,
                        "previous_ratio": previous_ratio,
                        "delta": delta,
                        "gap_days": gap_days,
                        "keywords": row.get("keywords", []),
                        "start_date": row.get("start_date"),
                        "end_date": row.get("end_date"),
                        "time_unit": row.get("time_unit"),
                        "collected_at": row.get("collected_at"),
                        "metadata": {
                            "detection_rule": (
                                "원본 DataLab 응답 기준, "
                                "current_ratio >= min_ratio, "
                                "delta >= min_delta, "
                                "gap_days <= max_gap_days"
                            ),
                            "min_ratio": min_ratio,
                            "min_delta": min_delta,
                            "max_gap_days": max_gap_days,
                            "is_actual_search_count": False,
                            "usage_note": (
                                "실제 검색량 급증 확정이 아니라 원인 탐색 후보일로만 사용한다."
                            ),
                        },
                    }
                )

            previous_row = row

    candidates.sort(
        key=lambda row: (
            row.get("period") or "",
            row.get("group_name") or "",
        )
    )

    return candidates


def mark_relative_peak_candidates(
    rows: list[dict[str, Any]],
    min_ratio: float = DEFAULT_PEAK_MIN_RATIO,
    min_delta: float = DEFAULT_PEAK_MIN_DELTA,
    max_gap_days: int = DEFAULT_PEAK_MAX_GAP_DAYS,
) -> list[dict[str, Any]]:
    marked_rows = [
        {
            **row,
            "is_peak_candidate": False,
            "peak": None,
        }
        for row in rows
    ]

    row_index: dict[tuple[str, str], dict[str, Any]] = {
        (str(row.get("group_name") or ""), str(row.get("period") or "")): row for row in marked_rows
    }

    candidates = detect_relative_peak_candidates(
        rows,
        min_ratio=min_ratio,
        min_delta=min_delta,
        max_gap_days=max_gap_days,
    )

    for candidate in candidates:
        key = (
            str(candidate.get("group_name") or ""),
            str(candidate.get("period") or ""),
        )

        target = row_index.get(key)

        if not target:
            continue

        target["is_peak_candidate"] = True
        target["peak"] = {
            "previous_period": candidate.get("previous_period"),
            "previous_ratio": candidate.get("previous_ratio"),
            "delta": candidate.get("delta"),
            "gap_days": candidate.get("gap_days"),
            "rule": (
                f"ratio >= {min_ratio} and delta >= {min_delta} and gap_days <= {max_gap_days}"
            ),
            "note": "실제 검색 건수가 아니라 조회 기간 내 상대 검색지수 기준의 피크 후보입니다.",
        }

    return marked_rows


def parse_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def date_gap_days(current_period: str, previous_period: str) -> int | None:
    try:
        current = datetime.strptime(current_period, "%Y-%m-%d").date()
        previous = datetime.strptime(previous_period, "%Y-%m-%d").date()
    except ValueError:
        return None

    return (current - previous).days


def parse_period(period: str) -> datetime | None:
    if not period:
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(period, fmt).replace(tzinfo=KST)
        except ValueError:
            continue

    return None


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def parse_chart_date(period: Any) -> datetime | None:
    if not period:
        return None

    try:
        return datetime.strptime(str(period), "%Y-%m-%d")
    except ValueError:
        return None


def moving_average(
    values: list[float],
    window_size: int = 3,
) -> list[float]:
    if window_size <= 1:
        return values

    smoothed: list[float] = []

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        smoothed.append(sum(window) / len(window))

    return smoothed


def datalab_rows_to_articles(
    rows: list[dict[str, Any]],
    peer_id: str = "all",
) -> list[RawArticle]:
    articles: list[RawArticle] = []

    for row in rows:
        group_name = str(row.get("group_name") or peer_id)
        period = str(row.get("period") or "")
        ratio = row.get("ratio")
        keywords = row.get("keywords") or []
        collected_at = parse_datetime(row.get("collected_at")) or datetime.now(KST)
        published_at = parse_period(period)

        articles.append(
            RawArticle(
                url=API_URL,
                title=f"Naver DataLab 상대 검색지수 - {group_name} ({period})",
                content=(
                    f"{group_name} 상대 검색지수: {period} 기준 ratio {ratio}. "
                    f"키워드: {', '.join(map(str, keywords))}"
                ),
                published_at=published_at,
                source_name=SOURCE_NAME,
                peer_id=peer_id,
                source_type="search_trend",
                collected_at=collected_at,
                publisher="Naver DataLab",
                company=[group_name],
                content_type="api",
                extra={
                    **row,
                    "crawl_type": CRAWL_TYPE,
                    "is_actual_search_count": False,
                },
            )
        )

    return articles


class KeywordCrawler(BaseCrawler):
    """Naver DataLab 섹터별 상대 검색지수를 RawArticle 형태로 수집한다."""

    def __init__(
        self,
        peer_id: str = "all",
        keywords: list[str] | dict[str, list[str]] | None = None,
        keyword_groups: list[dict[str, Any]] | None = None,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        time_unit: str = DEFAULT_TIME_UNIT,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        super().__init__(peer_id)
        self.keywords = keywords
        self.keyword_groups = keyword_groups
        self.lookback_days = lookback_days
        self.time_unit = time_unit
        self.start_date = start_date
        self.end_date = end_date

    async def crawl(self) -> list[RawArticle]:
        keyword_groups = self.keyword_groups or self._build_keyword_groups()
        rows = collect_naver_datalab_trends(
            keyword_groups=keyword_groups,
            start_date=self.start_date,
            end_date=self.end_date,
            time_unit=self.time_unit,
            lookback_days=self.lookback_days,
        )
        marked_rows = mark_relative_peak_candidates(rows)
        return datalab_rows_to_articles(marked_rows, peer_id=self.peer_id)

    def _build_keyword_groups(self) -> list[dict[str, Any]]:
        if self.keyword_groups:
            return self.keyword_groups

        if isinstance(self.keywords, list):
            return [
                {
                    "groupName": str(self.peer_id or "custom"),
                    "keywords": dedupe_texts(self.keywords)[:NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP],
                    "metadata": {
                        "mode": "custom_keywords",
                    },
                }
            ]

        if isinstance(self.keywords, dict):
            custom_groups: list[dict[str, Any]] = []

            for group_name, keywords in self.keywords.items():
                custom_groups.append(
                    {
                        "groupName": str(group_name),
                        "keywords": dedupe_texts(keywords)[:NAVER_DATALAB_MAX_KEYWORDS_PER_GROUP],
                        "metadata": {
                            "mode": "custom_keyword_groups",
                        },
                    }
                )

            return custom_groups

        return build_default_keyword_groups()


def save_json(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log.info("JSON 저장 완료: path=%s, rows=%s", output_path, len(rows))

    return output_path


def build_output_path(prefix: str) -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_text = datetime.now(KST).strftime("%Y%m%d")
    return DEFAULT_OUTPUT_DIR / f"{prefix}_{date_text}.json"


def configure_korean_font() -> None:
    try:
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
    except ImportError:
        return

    system_name = platform.system()

    if system_name == "Darwin":
        font_candidates = ["AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    elif system_name == "Windows":
        font_candidates = ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
    else:
        font_candidates = [
            "NanumGothic",
            "Noto Sans CJK KR",
            "Noto Sans CJK",
            "DejaVu Sans",
        ]

    installed_fonts = {font.name for font in fm.fontManager.ttflist}

    for font_name in font_candidates:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


def save_trend_chart(
    rows: list[dict[str, Any]],
    output_path: str | Path | None = None,
    smoothing_window: int = 3,
) -> Path | None:
    if not rows:
        log.warning("시각화할 Naver DataLab 결과가 없습니다.")
        return None

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib이 설치되어 있지 않아 차트를 저장하지 않습니다.")
        return None

    configure_korean_font()

    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        group_name = row.get("group_name")

        if not group_name:
            continue

        grouped.setdefault(str(group_name), []).append(row)

    if output_path is None:
        DEFAULT_CHART_DIR.mkdir(parents=True, exist_ok=True)
        date_text = datetime.now(KST).strftime("%Y%m%d")
        output_path = DEFAULT_CHART_DIR / f"naver_datalab_trend_{date_text}.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 8))

    for group_name, group_rows in grouped.items():
        parsed_rows: list[tuple[datetime, float]] = []

        for row in group_rows:
            period_date = parse_chart_date(row.get("period"))
            ratio = parse_float(row.get("ratio"))

            if period_date is None or ratio is None:
                continue

            parsed_rows.append((period_date, ratio))

        parsed_rows.sort(key=lambda item: item[0])

        if not parsed_rows:
            continue

        dates = [item[0] for item in parsed_rows]
        ratios = [item[1] for item in parsed_rows]
        smoothed_ratios = moving_average(ratios, window_size=smoothing_window)

        plt.scatter(
            dates,
            ratios,
            alpha=0.35,
            s=25,
        )

        plt.plot(
            dates,
            smoothed_ratios,
            linewidth=2,
            label=f"{group_name} {smoothing_window}d MA",
        )

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

    plt.title("Naver DataLab Sector Relative Search Index")
    plt.xlabel("Period")
    plt.ylabel("Relative Search Index Ratio")
    plt.xticks(rotation=45, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    log.info("Naver DataLab 스무딩 차트 저장 완료: path=%s", output_path)

    return output_path


def run_keyword_crawler(
    save_chart: bool = False,
) -> list[dict[str, Any]]:
    raw_rows = collect_naver_datalab_trends()
    marked_rows = mark_relative_peak_candidates(raw_rows)

    save_json(
        marked_rows,
        build_output_path("naver_datalab"),
    )

    if save_chart:
        save_trend_chart(marked_rows)

    return marked_rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    rows = run_keyword_crawler(save_chart=True)
    log.info("keyword_crawler 실행 완료: total_rows=%s", len(rows))


if __name__ == "__main__":
    main()
