"""KRX stock latest-price crawler.

Design notes for safe, reproducible collection:
- Target analysis: monitored Korean listed stocks are resolved from
  src.config.companies. The company registry owns Naver item codes and aliases;
  this crawler owns only the market-price collection logic.
- Access strategy: latest KRX trading data is fetched with FinanceDataReader.
- Extraction and parsing: the latest trading row is mapped as
  date|open|high|low|close|volume|change_pct, then normalized into the existing
  stock_price_ohlcv_v1 payload under the repository's common crawler envelope.
- Exception handling: provider/import/data failures are returned as failed
  common-envelope documents so the DB save path can record the failure without
  breaking the whole crawler batch.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import math
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx

try:
    from src.config.companies import COMPANIES, NAVER_ITEM_CODES
    from src.config.company_tiers import company_tier_map
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config.companies import COMPANIES, NAVER_ITEM_CODES
    from src.config.company_tiers import company_tier_map

log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
SCHEMA_NAME = "stock_price_ohlcv_v1"
SOURCE_NAME = "stock_market"
SOURCE_TYPE = "market_data"
CONTENT_TYPE = "api"
DEFAULT_LOOKBACK_DAYS = int(os.getenv("STOCK_LOOKBACK_DAYS", "10"))
DEFAULT_TARGET_KEYS = tuple(NAVER_ITEM_CODES.keys())

MANUAL_CODE_MAP = {
    "SK": "034730",
    "삼성SDS": "018260",
    "LG CNS": "064400",
    "포스코DX": "022100",
    "sk_ax": "034730",
    "samsung_sds": "018260",
    "lg_cns": "064400",
    "posco_dx": "022100",
}

NAVER_CHART_URL = "https://fchart.stock.naver.com/sise.nhn"
NAVER_REALTIME_URL = "https://polling.finance.naver.com/api/realtime"
NAVER_ITEM_REFERER = "https://finance.naver.com/item/main.naver?code={ticker}"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/xml,text/xml,application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}

MissingPolicy = Literal["drop", "previous_close"]


@dataclass(frozen=True)
class StockTarget:
    """Supported target metadata."""

    key: str
    ticker: str
    company_name: str
    exchange: str
    aliases: tuple[str, ...]


@dataclass
class StockOHLCVRecord:
    """Validated downstream-ready OHLCV row."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    change_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "change_pct": self.change_pct,
        }


@dataclass
class StockCrawlResult:
    """Common crawler envelope plus a stock-specific OHLCV payload."""

    ticker: str
    source: str
    title: str
    content: str | None
    company: list[str]
    data: list[StockOHLCVRecord]
    peer_id: str | None = None
    publisher: str = "FinanceDataReader / KRX"
    source_type: str = SOURCE_TYPE
    source_name: str = SOURCE_NAME
    content_type: str = CONTENT_TYPE
    crawl_status: str = "success"
    error_message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=lambda: datetime.now(KST))
    id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def url(self) -> str:
        return self.source

    @property
    def url_hash(self) -> str:
        return hashlib.md5(self.url.encode()).hexdigest()

    def to_common_dict(self) -> dict[str, Any]:
        """Return the strict JSON structure expected by downstream consumers."""

        return {
            "id": self.id,
            "schema_name": SCHEMA_NAME,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "publisher": self.publisher,
            "title": self.title,
            "content": self.content,
            "ticker": self.ticker,
            "source": self.source,
            "url": self.url,
            "url_hash": self.url_hash,
            "published_at": None,
            "collected_at": self.collected_at.isoformat(),
            "company": self.company,
            "company_tier": company_tier_map(self.company),
            "peer_id": self.peer_id,
            "language": "ko",
            "content_type": self.content_type,
            "crawl_status": self.crawl_status,
            "error_message": self.error_message,
            "extra": _public_extra(self.extra),
            "data": [row.to_dict() for row in self.data],
        }


class StockCrawlerError(Exception):
    """Base stock crawler error."""


class DataIntegrityError(StockCrawlerError):
    """Raised when fetched data cannot pass integrity checks."""


class RowValidationError(DataIntegrityError):
    """Raised for one invalid OHLCV row."""


def _public_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """Remove crawler-internal and explanatory fields from serialized output."""

    public: dict[str, Any] = {}

    for key in ("target", "requested_window", "coverage", "validation"):
        value = extra.get(key)
        if value:
            public[key] = value

    missing_policy = extra.get("missing_policy")
    if missing_policy and missing_policy != "drop":
        public["missing_policy"] = missing_policy

    realtime_quote = extra.get("realtime_quote")
    if realtime_quote is not None:
        public["realtime_quote"] = realtime_quote

    latest_price = extra.get("latest_price")
    if latest_price is not None:
        public["latest_price"] = latest_price

    realtime_error = extra.get("realtime_error")
    if realtime_error:
        public["realtime_error"] = realtime_error

    return public


def _normalize_lookup(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.strip().lower())


def _dedupe_strings(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []

    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)

    return tuple(deduped)


def _target_from_company_config(
    company_id: str,
    config: dict[str, Any],
    *,
    item_code: str | None = None,
) -> StockTarget | None:
    ticker = item_code or config.get("naver_item_code") or NAVER_ITEM_CODES.get(company_id)
    if not ticker:
        return None

    company_name = str(config.get("naver_item_name_ko") or config.get("name_ko") or company_id)
    aliases = _dedupe_strings(
        [
            company_id,
            str(ticker),
            company_name,
            *[str(alias) for alias in config.get("aliases", [])],
        ]
    )
    return StockTarget(
        key=company_id,
        ticker=str(ticker),
        company_name=company_name,
        exchange=str(config.get("exchange") or "KRX"),
        aliases=aliases,
    )


def _target_from_company_id(
    company_id: str,
    *,
    item_code: str | None = None,
) -> StockTarget | None:
    config = COMPANIES.get(company_id)
    if config is None:
        return None
    return _target_from_company_config(company_id, dict(config), item_code=item_code)


def _manual_ticker_target(
    ticker: str,
    *,
    key: str | None = None,
    company_name: str | None = None,
) -> StockTarget:
    target_key = key or f"krx_{ticker}"
    target_name = company_name or ticker
    return StockTarget(
        key=target_key,
        ticker=ticker,
        company_name=target_name,
        exchange="KRX",
        aliases=_dedupe_strings([target_key, ticker, target_name]),
    )


def _build_config_targets() -> tuple[StockTarget, ...]:
    targets: list[StockTarget] = []
    for company_id, config in COMPANIES.items():
        target = _target_from_company_config(company_id, dict(config))
        if target:
            targets.append(target)
    return tuple(targets)


TARGETS: tuple[StockTarget, ...] = _build_config_targets()

TARGET_BY_KEY = {target.key: target for target in TARGETS}
TARGET_BY_ALIAS = {
    alias: target
    for target in TARGETS
    for alias in {
        _normalize_lookup(value) for value in (target.key, target.ticker, *target.aliases)
    }
}

_KRX_LISTING_CACHE: Any | None = None


def _require_finance_datareader() -> Any:
    try:
        return importlib.import_module("FinanceDataReader")
    except ModuleNotFoundError as exc:
        raise StockCrawlerError(
            "FinanceDataReader가 설치되어 있지 않습니다. "
            "uv add finance-datareader pandas 또는 "
            "pip install finance-datareader pandas 후 다시 실행하세요."
        ) from exc


def _krx_listing() -> Any:
    global _KRX_LISTING_CACHE
    if _KRX_LISTING_CACHE is None:
        _KRX_LISTING_CACHE = _require_finance_datareader().StockListing("KRX")
    return _KRX_LISTING_CACHE


def _find_symbol(company_name: str, fallback_ticker: str | None = None) -> str | None:
    """Resolve a KRX symbol by exact/normalized company name, then manual map."""

    manual = MANUAL_CODE_MAP.get(company_name)
    if manual:
        return manual

    listing = _krx_listing()
    exact = listing[listing["Name"] == company_name]
    if not exact.empty:
        return str(exact.iloc[0]["Code"]).zfill(6)

    normalized = company_name.replace(" ", "")
    candidates = listing[
        listing["Name"].str.replace(" ", "", regex=False).str.contains(normalized, na=False)
    ]
    if not candidates.empty:
        return str(candidates.iloc[0]["Code"]).zfill(6)

    manual = MANUAL_CODE_MAP.get(_normalize_lookup(company_name))
    if manual:
        return manual

    return fallback_ticker


def _fetch_latest_price(
    *,
    target: StockTarget,
    start_date: date,
    end_date: date,
) -> tuple[StockOHLCVRecord, dict[str, Any]]:
    """Fetch the latest trading row through FinanceDataReader."""

    reader = _require_finance_datareader()
    symbol = _find_symbol(target.company_name, fallback_ticker=target.ticker)
    if not symbol:
        raise DataIntegrityError(
            f"{target.company_name} 종목코드 미확인 (비상장 또는 명칭 불일치 가능)"
        )

    effective_start = min(start_date, end_date - timedelta(days=10))
    df = reader.DataReader(symbol, effective_start, end_date)
    if df.empty:
        raise DataIntegrityError(f"{symbol} 데이터가 비어 있습니다.")

    latest = df.tail(1).copy()
    row = latest.iloc[0]
    trade_date = latest.index[-1]
    trade_date_str = trade_date.strftime("%Y-%m-%d")
    change_pct = _row_change_pct(df)

    record = StockOHLCVRecord(
        date=trade_date_str,
        open=float(row.get("Open", row.get("Close", 0))),
        high=float(row.get("High", row.get("Close", 0))),
        low=float(row.get("Low", row.get("Close", 0))),
        close=float(row["Close"]),
        volume=int(row.get("Volume", 0) or 0),
        change_pct=change_pct,
    )
    validation = {
        "provider": "FinanceDataReader",
        "resolved_symbol": symbol,
        "requested_company_name": target.company_name,
        "requested_ticker": target.ticker,
        "status": "normal",
        "dropped_rows": 0,
    }
    return record, validation


def _row_change_pct(df: Any) -> float | None:
    latest_row = df.tail(1).iloc[0]
    if "Change" in df.columns and not _is_null_value(latest_row.get("Change")):
        return float(latest_row["Change"] * 100)

    close = df["Close"].dropna() if "Close" in df.columns else []
    if len(close) >= 2:
        previous_close = float(close.iloc[-2])
        latest_close = float(close.iloc[-1])
        if previous_close:
            return ((latest_close - previous_close) / previous_close) * 100
    return None


def _is_null_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _latest_price_payload(target: StockTarget, record: StockOHLCVRecord) -> dict[str, Any]:
    return {
        "company_name": target.company_name,
        "ticker": target.ticker,
        "trade_date": record.date,
        "close": record.close,
        "change_pct": record.change_pct,
        "volume": record.volume,
        "status": "정상",
    }


class StockCrawler:
    """Collect latest KRX stock price data with FinanceDataReader."""

    def __init__(
        self,
        targets: list[StockTarget] | str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        *,
        peer_id: str | None = None,
        item_code: str | None = None,
        lookback_days: int | None = None,
        include_realtime: bool = True,
        missing_policy: MissingPolicy = "drop",
        request_delay: float = 0.35,
        timeout: float = 10.0,
        max_count: int = 20_000,
    ) -> None:
        if isinstance(targets, str):
            peer_id = peer_id or targets
            targets = None

        self.peer_id = peer_id
        self.targets = self._resolve_init_targets(
            targets=targets,
            peer_id=peer_id,
            item_code=item_code,
        )
        effective_lookback_days = lookback_days or DEFAULT_LOOKBACK_DAYS
        self.start_date = start_date or (
            datetime.now(KST).date() - timedelta(days=effective_lookback_days)
        )
        self.end_date = end_date or datetime.now(KST).date()
        self.include_realtime = include_realtime
        self.missing_policy = missing_policy
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_count = max_count

        if self.start_date > self.end_date:
            raise ValueError("start_date must be earlier than or equal to end_date")
        if self.missing_policy not in ("drop", "previous_close"):
            raise ValueError("missing_policy must be 'drop' or 'previous_close'")

    def _resolve_init_targets(
        self,
        *,
        targets: list[StockTarget] | None,
        peer_id: str | None,
        item_code: str | None,
    ) -> list[StockTarget]:
        """Match run_local_crawler_once's peer-based construction style."""

        if targets is not None:
            return targets

        if peer_id:
            config_target = _target_from_company_id(peer_id, item_code=item_code)
            if config_target:
                return [config_target]

            if item_code:
                return [_manual_ticker_target(item_code, key=peer_id, company_name=peer_id)]

            return resolve_targets([peer_id])

        if item_code:
            return [_manual_ticker_target(item_code)]

        return resolve_targets(None)

    async def crawl(self) -> list[StockCrawlResult]:
        """Fetch all configured targets, returning one common-envelope document each."""

        results: list[StockCrawlResult] = []
        for index, target in enumerate(self.targets):
            if index > 0:
                await asyncio.sleep(self.request_delay + random.uniform(0, 0.15))

            try:
                results.append(await asyncio.to_thread(self._crawl_one, target))
            except Exception as exc:
                log.exception("stock crawl failed | ticker=%s", target.ticker)
                results.append(self._failed_result(target, str(exc)))

        return results

    def _crawl_one(self, target: StockTarget) -> StockCrawlResult:
        source_url = NAVER_ITEM_REFERER.format(ticker=target.ticker)
        latest, validation = _fetch_latest_price(
            target=target,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        data = [latest]
        latest_price = _latest_price_payload(target, latest)

        extra = {
            "target": {
                "company_name": target.company_name,
                "exchange": target.exchange,
            },
            "requested_window": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
            },
            "coverage": {
                "first_date": latest.date,
                "last_date": latest.date,
                "row_count": len(data),
            },
            "source_type": "api",
            "content_type": CONTENT_TYPE,
            "missing_policy": self.missing_policy,
            "validation": validation,
            "latest_price": latest_price,
        }
        if self.include_realtime:
            extra["realtime_quote"] = latest_price

        return StockCrawlResult(
            ticker=target.ticker,
            source=source_url,
            title=(f"{target.company_name}({target.ticker}) 최신 주가 {latest.date}"),
            content=(f"{target.company_name} latest KRX price collected with FinanceDataReader."),
            company=[target.key],
            peer_id=target.key,
            data=data,
            extra=extra,
        )

    async def _fetch_realtime_quote(
        self,
        client: httpx.AsyncClient,
        target: StockTarget,
    ) -> dict[str, Any]:
        response = await self._get_with_retry(
            client,
            self._realtime_url(target),
            headers={
                "Accept": "application/json,text/plain,*/*",
                "Referer": NAVER_ITEM_REFERER.format(ticker=target.ticker),
            },
        )
        payload = _decode_json_response(response)
        if payload.get("resultCode") != "success":
            raise StockCrawlerError(f"realtime resultCode={payload.get('resultCode')}")

        item = _first_realtime_item(payload)
        if not item:
            raise StockCrawlerError("realtime quote not found")

        result_time = payload.get("result", {}).get("time")
        return {
            "ticker": target.ticker,
            "company_name": target.company_name,
            "quote_time": _epoch_ms_to_iso(result_time),
            "market_status": item.get("ms"),
            "current_price": _safe_float(item.get("nv")),
            "previous_close": _safe_float(item.get("pcv") or item.get("sv")),
            "change": _safe_float(item.get("cv")),
            "change_percent": _safe_float(item.get("cr")),
            "open": _safe_float(item.get("ov")),
            "high": _safe_float(item.get("hv")),
            "low": _safe_float(item.get("lv")),
            "volume": _safe_int(item.get("aq")),
            "trading_value": _safe_float(item.get("aa")),
            "nxt_over_market": _normalize_nxt_quote(item.get("nxtOverMarketPriceInfo")),
            "source": self._realtime_url(target),
        }

    async def _get_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        backoff = (0.3, 0.8, 1.6)
        last_error: Exception | None = None

        for attempt in range(len(backoff) + 1):
            try:
                response = await client.get(url, headers=headers)
                if response.status_code in (403, 429):
                    raise StockCrawlerError(f"blocked or rate-limited: HTTP {response.status_code}")
                response.raise_for_status()
                if not response.content:
                    raise StockCrawlerError("empty response")
                return response
            except Exception as exc:
                last_error = exc
                if attempt >= len(backoff):
                    break
                await asyncio.sleep(backoff[attempt] + random.uniform(0, 0.2))

        raise StockCrawlerError(f"GET failed after retries: {url} ({last_error})")

    def _failed_result(self, target: StockTarget, error_message: str) -> StockCrawlResult:
        source_url = NAVER_ITEM_REFERER.format(ticker=target.ticker)
        return StockCrawlResult(
            ticker=target.ticker,
            source=source_url,
            title=(f"{target.company_name}({target.ticker}) 최신 주가 조회 실패"),
            content=None,
            company=[target.key],
            peer_id=target.key,
            data=[],
            crawl_status="failed",
            error_message=error_message,
            extra={
                "target": {
                    "company_name": target.company_name,
                    "exchange": target.exchange,
                },
                "requested_window": {
                    "start_date": self.start_date.isoformat(),
                    "end_date": self.end_date.isoformat(),
                },
                "source_type": "api",
                "content_type": CONTENT_TYPE,
                "missing_policy": self.missing_policy,
                "validation": {
                    "valid_rows": 0,
                    "dropped_rows": 0,
                    "duplicate_rows_removed": 0,
                },
            },
        )

    def _chart_url(self, target: StockTarget) -> str:
        count = self._history_count()
        return f"{NAVER_CHART_URL}?symbol={target.ticker}&timeframe=day&count={count}&requestType=0"

    def _realtime_url(self, target: StockTarget) -> str:
        query = f"SERVICE_ITEM:{target.ticker}|SERVICE_RECENT_ITEM:{target.ticker}"
        return f"{NAVER_REALTIME_URL}?query={query}"

    def _history_count(self) -> int:
        today = datetime.now(KST).date()
        earliest_needed = min(self.start_date, today)
        calendar_days = max((today - earliest_needed).days, 1)
        count = calendar_days + 90
        return max(1, min(count, self.max_count))


def _parse_naver_chart_xml(content: bytes) -> list[list[str]]:
    text = content.decode("euc-kr", errors="replace")
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text, count=1)
    try:
        root = ET.fromstring(text)
    except (ET.ParseError, ValueError) as exc:
        raise StockCrawlerError(f"invalid chart XML: {exc}") from exc

    rows: list[list[str]] = []
    for item in root.findall("./chartdata/item"):
        raw = item.attrib.get("data", "")
        parts = raw.split("|")
        if len(parts) != 6:
            raise DataIntegrityError(f"unexpected OHLCV field count: {raw!r}")
        rows.append(parts)
    return rows


def _validate_and_clean_rows(
    raw_rows: list[list[str]],
    start_date: date,
    end_date: date,
    missing_policy: MissingPolicy,
) -> tuple[list[StockOHLCVRecord], dict[str, Any]]:
    valid: list[StockOHLCVRecord] = []
    dropped_samples: list[dict[str, Any]] = []
    dropped_count = 0
    previous_close: float | None = None

    for raw in raw_rows:
        try:
            row_date = _parse_yyyymmdd(raw[0])
            if row_date < start_date or row_date > end_date:
                continue

            record = _row_to_record(raw, previous_close, missing_policy)
            _assert_sanity(record)
            valid.append(record)
            previous_close = record.close
        except RowValidationError as exc:
            dropped_count += 1
            if len(dropped_samples) < 20:
                dropped_samples.append({"raw": raw, "reason": str(exc)})

    deduped, duplicate_count = _dedupe_records(valid)
    validation = {
        "valid_rows": len(deduped),
        "dropped_rows": dropped_count,
        "duplicate_rows_removed": duplicate_count,
    }
    if dropped_samples:
        validation["dropped_row_samples"] = dropped_samples
    return deduped, validation


def _row_to_record(
    raw: list[str],
    previous_close: float | None,
    missing_policy: MissingPolicy,
) -> StockOHLCVRecord:
    row_date = _parse_yyyymmdd(raw[0])
    price_names = ("open", "high", "low", "close")
    parsed_prices: dict[str, float | None] = {}

    for name, value in zip(price_names, raw[1:5], strict=True):
        parsed_prices[name] = _parse_optional_float(value)

    volume = _parse_optional_int(raw[5])
    missing = [name for name, value in parsed_prices.items() if value is None]
    if volume is None:
        missing.append("volume")

    if missing:
        if missing_policy == "previous_close" and previous_close is not None:
            for name in price_names:
                if parsed_prices[name] is None:
                    parsed_prices[name] = previous_close
            if volume is None:
                volume = 0
        else:
            raise RowValidationError(f"missing_or_null_values={missing}")

    return StockOHLCVRecord(
        date=row_date.isoformat(),
        open=float(parsed_prices["open"]),
        high=float(parsed_prices["high"]),
        low=float(parsed_prices["low"]),
        close=float(parsed_prices["close"]),
        volume=int(volume),
    )


def _assert_sanity(record: StockOHLCVRecord) -> None:
    prices = (record.open, record.high, record.low, record.close)
    if not all(isinstance(value, float) and math.isfinite(value) for value in prices):
        raise RowValidationError("price_type_check_failed")
    if not isinstance(record.volume, int):
        raise RowValidationError("volume_type_check_failed")
    if record.high < record.low:
        raise RowValidationError("sanity_failed: high < low")
    if record.high < record.open or record.high < record.close:
        raise RowValidationError("sanity_failed: high lower than open/close")
    if record.low > record.open or record.low > record.close:
        raise RowValidationError("sanity_failed: low greater than open/close")
    if record.volume < 0:
        raise RowValidationError("sanity_failed: negative volume")


def _dedupe_records(records: list[StockOHLCVRecord]) -> tuple[list[StockOHLCVRecord], int]:
    by_date: dict[str, StockOHLCVRecord] = {}
    duplicate_count = 0
    for record in records:
        if record.date in by_date:
            duplicate_count += 1
        by_date[record.date] = record

    return [by_date[key] for key in sorted(by_date)], duplicate_count


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "null", "none", "-"}:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise RowValidationError(f"invalid_float={value!r}") from exc
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_optional_int(value: Any) -> int | None:
    parsed = _parse_optional_float(value)
    if parsed is None:
        return None
    if parsed < 0:
        raise RowValidationError(f"invalid_negative_int={value!r}")
    if not parsed.is_integer():
        raise RowValidationError(f"invalid_integer={value!r}")
    return int(parsed)


def _parse_yyyymmdd(value: str) -> date:
    text = str(value).strip()
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:
        raise RowValidationError(f"invalid_date={value!r}") from exc


def _parse_cli_date(value: str) -> date:
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("date must be YYYY-MM-DD or YYYYMMDD")


def _build_date_warnings(
    start_date: date,
    end_date: date,
    data: list[StockOHLCVRecord],
) -> list[str]:
    warnings: list[str] = []
    today = datetime.now(KST).date()
    if end_date > today:
        warnings.append(
            f"end_date is in the future; available data only through {today.isoformat()}"
        )
    if data:
        first = data[0].date
        last = data[-1].date
        if first > start_date.isoformat():
            warnings.append(f"first available trading date is {first}, after requested start_date")
        if last < min(end_date, today).isoformat():
            warnings.append(f"last available trading date is {last}, before requested end_date")
    return warnings


def _first_realtime_item(payload: dict[str, Any]) -> dict[str, Any] | None:
    for area in payload.get("result", {}).get("areas", []):
        if area.get("name") == "SERVICE_ITEM" and area.get("datas"):
            return area["datas"][0]
    for area in payload.get("result", {}).get("areas", []):
        if area.get("datas"):
            return area["datas"][0]
    return None


def _epoch_ms_to_iso(value: Any) -> str | None:
    try:
        if value is None:
            return None
        return (
            datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc).astimezone(KST).isoformat()
        )
    except (TypeError, ValueError, OSError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        parsed = _parse_optional_float(value)
    except RowValidationError:
        return None
    return parsed


def _safe_int(value: Any) -> int | None:
    try:
        parsed = _parse_optional_int(value)
    except RowValidationError:
        return None
    return parsed


def _normalize_nxt_quote(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "trading_session_type": value.get("tradingSessionType"),
        "market_status": value.get("overMarketStatus"),
        "price": _safe_float(_strip_number_text(value.get("overPrice"))),
        "open": _safe_float(_strip_number_text(value.get("openPrice"))),
        "high": _safe_float(_strip_number_text(value.get("highPrice"))),
        "low": _safe_float(_strip_number_text(value.get("lowPrice"))),
        "change": _safe_float(_strip_number_text(value.get("compareToPreviousClosePrice"))),
        "change_percent": _safe_float(_strip_number_text(value.get("fluctuationsRatio"))),
        "local_traded_at": value.get("localTradedAt"),
        "volume": _safe_int(_strip_number_text(value.get("accumulatedTradingVolume"))),
    }


def _strip_number_text(value: Any) -> Any:
    if value is None:
        return None
    return re.sub(r"[^0-9.\-]", "", str(value))


def _realtime_from_last_ohlcv(record: StockOHLCVRecord) -> dict[str, Any]:
    return {
        "ticker": None,
        "company_name": None,
        "quote_time": None,
        "market_status": "FALLBACK_FROM_DAILY_OHLCV",
        "current_price": record.close,
        "previous_close": None,
        "change": None,
        "change_percent": None,
        "open": record.open,
        "high": record.high,
        "low": record.low,
        "volume": record.volume,
        "trading_value": None,
        "nxt_over_market": None,
        "source": "historical_last_row",
    }


def resolve_targets(values: list[str] | None) -> list[StockTarget]:
    if not values:
        return [TARGET_BY_KEY[key] for key in DEFAULT_TARGET_KEYS if key in TARGET_BY_KEY]

    resolved: list[StockTarget] = []
    for value in values:
        key = _normalize_lookup(value)
        target = TARGET_BY_ALIAS.get(key)
        if target:
            resolved.append(target)
            continue

        if re.fullmatch(r"\d{6}", value.strip()):
            ticker = value.strip()
            resolved.append(_manual_ticker_target(ticker))
            continue

        supported = ", ".join(f"{target.company_name}({target.ticker})" for target in TARGETS)
        raise ValueError(f"unsupported ticker/company '{value}'. Supported targets: {supported}")

    return _dedupe_targets(resolved)


def _dedupe_targets(targets: list[StockTarget]) -> list[StockTarget]:
    seen: set[str] = set()
    deduped: list[StockTarget] = []
    for target in targets:
        if target.ticker in seen:
            continue
        seen.add(target.ticker)
        deduped.append(target)
    return deduped


def _decode_json_response(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except UnicodeDecodeError:
        payload = json.loads(response.content.decode("euc-kr", errors="replace"))
    except json.JSONDecodeError:
        payload = json.loads(response.content.decode("euc-kr", errors="replace"))

    if not isinstance(payload, dict):
        raise StockCrawlerError("JSON response root is not an object")
    return payload


def serialize_results(results: list[StockCrawlResult]) -> Any:
    payload = [result.to_common_dict() for result in results]
    return payload[0] if len(payload) == 1 else payload


async def run_stock_crawler(
    *,
    tickers: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    include_realtime: bool = True,
    missing_policy: MissingPolicy = "drop",
    request_delay: float = 0.35,
) -> list[StockCrawlResult]:
    crawler = StockCrawler(
        targets=resolve_targets(tickers),
        start_date=start_date,
        end_date=end_date,
        include_realtime=include_realtime,
        missing_policy=missing_policy,
        request_delay=request_delay,
    )
    return await crawler.crawl()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect KRX stock OHLCV as JSON.")
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help=(
            "Ticker, company id, or company alias. Repeatable. Defaults to every "
            "company with naver_item_code in src/config/companies.py."
        ),
    )
    parser.add_argument("--start-date", type=_parse_cli_date, required=True)
    parser.add_argument("--end-date", type=_parse_cli_date, required=True)
    parser.add_argument(
        "--missing-policy",
        choices=("drop", "previous_close"),
        default="drop",
        help="How to handle rows with null/NaN fields.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Skip latest quote polling and collect historical OHLCV only.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Seconds to wait between ticker requests.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-count", type=int, default=20_000)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    crawler = StockCrawler(
        targets=resolve_targets(args.tickers),
        start_date=args.start_date,
        end_date=args.end_date,
        include_realtime=not args.no_realtime,
        missing_policy=args.missing_policy,
        request_delay=args.delay,
        timeout=args.timeout,
        max_count=args.max_count,
    )
    results = await crawler.crawl()
    payload = serialize_results(results)
    output = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)

    return 0 if all(result.crawl_status == "success" for result in results) else 1


def main() -> int:
    args = build_arg_parser().parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
