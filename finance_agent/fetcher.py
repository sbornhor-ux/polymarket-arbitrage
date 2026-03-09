"""
Fetches financial time series from Polygon.io via the polygon-api-client SDK.

Fetches the broadest date range needed for all swing windows plus the
trailing baseline, then returns a single DataFrame per ticker.
Callers slice the window they need.

Setup:
    export POLYGON_API_KEY=$(python -c "import json; print(json.load(open('lab_08/secrets.json'))['polygon'])")
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import pandas as pd
from polygon import RESTClient

from polymarket_agent.models import SwingEvent


# ---------------------------------------------------------------------------
# Ticker mapping: yfinance public IDs → Polygon tickers (+ ETF fallbacks)
# ---------------------------------------------------------------------------

# Map from the public series ID (yfinance convention, used in DEFAULT_SERIES)
# to (primary_polygon_ticker, fallback_etf_ticker | None)
POLYGON_TICKER_MAP: dict[str, tuple[str, str | None]] = {
    "^GSPC":    ("I:SPX",    "SPY"),
    "^TNX":     ("I:TNX",    "IEF"),
    "^VIX":     ("I:VIX",    "VIXY"),
    "BTC-USD":  ("X:BTCUSD", None),
    "DX-Y.NYB": ("C:DXY",    "UUP"),
    "CL=F":     ("USO",      None),   # ETF primary — futures not in Polygon
    "GC=F":     ("GLD",      None),   # ETF primary — futures not in Polygon
}

# Polygon granularity thresholds
_MINUTE_MAX_DAYS = 7
_INTRADAY_MAX_DAYS = 729


def _get_client() -> RESTClient:
    api_key = os.environ.get("POLYGON_API_KEY", "")
    return RESTClient(api_key=api_key)


def _fetch_aggs(
    client: RESTClient,
    polygon_ticker: str,
    timespan: str,
    from_: str,
    to: str,
    multiplier: int = 1,
) -> list:
    """Return list of Agg objects, or empty list on failure."""
    try:
        return list(client.get_aggs(
            polygon_ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_,
            to=to,
            limit=50000,
        ))
    except Exception:
        return []


def _aggs_to_df(aggs: list) -> pd.DataFrame | None:
    """Convert a list of Polygon Agg objects to a normalised DataFrame."""
    if not aggs:
        return None
    records = [
        {
            "ts": agg.timestamp,
            "close": agg.close,
            "high": agg.high,
            "low": agg.low,
            "volume": agg.volume or 0.0,
        }
        for agg in aggs
    ]
    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns="ts")[["close", "high", "low", "volume"]]
    df = df.sort_index().dropna(subset=["close"])
    return df if not df.empty else None


def fetch_series(
    ticker: str,
    swing_events: list[SwingEvent],
    trailing_days: int = 60,
) -> pd.DataFrame | None:
    """
    Download a financial series covering all swing windows + trailing baseline.

    Args:
        ticker: Public series identifier (e.g. '^GSPC'). Translated internally
                to a Polygon ticker via POLYGON_TICKER_MAP.
        swing_events: List of SwingEvent objects; used to determine date range.
        trailing_days: Extra days before earliest swing for z-score baseline.

    Returns:
        DataFrame with DatetimeIndex (UTC) and columns:
            close, high, low, volume
        or None if the fetch fails.
    """
    if not swing_events:
        return None

    # Determine the overall date range
    starts = [pd.Timestamp(e.window_start_utc) for e in swing_events]
    ends = [pd.Timestamp(e.window_end_utc) for e in swing_events]

    earliest = min(starts) - pd.Timedelta(days=trailing_days)
    latest = max(ends) + pd.Timedelta(hours=1)

    now = pd.Timestamp.now("UTC")
    days_back = (now - earliest).days
    if days_back <= _MINUTE_MAX_DAYS:
        # Use 15-minute bars for recent data (within 7 days) — matches
        # the 15-minute Polymarket probability series resolution.
        timespan = "minute"
        bar_multiplier = 15
    elif days_back <= _INTRADAY_MAX_DAYS:
        timespan = "hour"
        bar_multiplier = 1
    else:
        timespan = "day"
        bar_multiplier = 1

    from_str = earliest.strftime("%Y-%m-%d")
    to_str = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Resolve Polygon ticker(s)
    primary, fallback = POLYGON_TICKER_MAP.get(ticker, (ticker, None))

    client = _get_client()

    aggs = _fetch_aggs(client, primary, timespan, from_str, to_str, multiplier=bar_multiplier)
    df = _aggs_to_df(aggs)

    if df is None and fallback:
        print(
            f"[finance_agent.fetcher] {primary} returned no data for {ticker}; "
            f"retrying with fallback {fallback}"
        )
        aggs = _fetch_aggs(client, fallback, timespan, from_str, to_str, multiplier=bar_multiplier)
        df = _aggs_to_df(aggs)

    return df


def fetch_ticker_info(ticker: str) -> dict:
    """
    Fetch static metadata for a ticker via Polygon ticker details.

    Returns a dict with:
        market_cap (float | None)
    """
    primary, _ = POLYGON_TICKER_MAP.get(ticker, (ticker, None))
    try:
        client = _get_client()
        details = client.get_ticker_details(primary)
        return {"market_cap": getattr(details, "market_cap", None)}
    except Exception:
        return {"market_cap": None}
