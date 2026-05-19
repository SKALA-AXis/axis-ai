from datetime import datetime

from src.crawler.base import RawArticle
from src.db.article_store import _market_price_ohlcv_rows


def test_market_price_ohlcv_rows_from_stock_payload() -> None:
    article = RawArticle(
        url="https://finance.naver.com/item/main.naver?code=018260#ohlcv:2026-05-01:2026-05-02",
        title="삼성SDS(018260) 최신 주가 2026-05-02",
        content="삼성SDS latest KRX price collected with FinanceDataReader.",
        source_name="stock_market",
        source_type="market_data",
        content_type="api",
        publisher="FinanceDataReader / KRX",
        company=["samsung_sds"],
        peer_id="samsung_sds",
        collected_at=datetime(2026, 5, 2, 15, 40),
        extra={
            "ticker": "018260",
            "data": [
                {
                    "date": "2026-05-02",
                    "open": 160000,
                    "high": 165000,
                    "low": 159000,
                    "close": 164500,
                    "volume": 123456,
                    "change_pct": 1.23,
                }
            ],
        },
    )

    rows = _market_price_ohlcv_rows(100, article)

    assert rows[0]["raw_article_id"] == 100
    assert rows[0]["peer_id"] == "samsung_sds"
    assert rows[0]["ticker"] == "018260"
    assert rows[0]["trade_date"].isoformat() == "2026-05-02"
    assert rows[0]["close"] == 164500
    assert rows[0]["volume"] == 123456
