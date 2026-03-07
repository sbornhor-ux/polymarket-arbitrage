"""
Data Loader
-----------
Fetches and parses input data from:
  - Polymarket scan JSON  (from R2 or local)
  - Polymarket CSV export (hourly prices from production CSV)

load_polymarket_scan()  — JSON scan snapshot (no time series)
load_polymarket_csv()   — production CSV with 49-point hourly price series

build_aligned_series()  — aligns one Polymarket record with one financial record;
                          uses real hourly prices when available, falls back to the
                          financial-data proxy when only a scan snapshot exists.

Financial market data loading is handled by tools/financial_adapter.py.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import POLYMARKET_SCAN_URL
from models.schemas import (
    FinancialMarketRecord,
    PolymarketRecord,
)

logger = logging.getLogger("trend_analyst.data_loader")

# ── Column name constants ─────────────────────────────────────────────────────

# t_minus columns from oldest → newest (t-24h … t-1h)
_T_MINUS_COLS = [f"price_t_minus_{i:02d}" for i in range(24, 0, -1)]
# t_plus columns from nearest → furthest (t+1h … t+24h)
_T_PLUS_COLS  = [f"price_t_plus_{i:02d}"  for i in range(1, 25)]

_MIN_ALIGNED_PTS = 5   # minimum overlapping observations for a useful analysis


# ══════════════════════════════════════════════════════════════════════════════
# POLYMARKET LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_polymarket_scan(
    source: Optional[str] = None,
) -> tuple[dict, list[PolymarketRecord]]:
    """
    Load the Polymarket scan JSON (snapshot format, no hourly time series).

    Args:
        source: URL or local filepath.  Defaults to POLYMARKET_SCAN_URL.

    Returns:
        (summary_dict, list_of_PolymarketRecord)
        PolymarketRecord.poly_prices will be empty — use load_polymarket_csv()
        when you need real hourly prices for statistical comparison.
    """
    source = source or POLYMARKET_SCAN_URL

    if source.startswith("http"):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    else:
        with open(source, "r") as f:
            data = json.load(f)

    summary = data.get("summary", {})
    markets = []

    for m in data.get("markets", []):
        markets.append(PolymarketRecord(
            market_id=str(m["market_id"]),
            question=m.get("question", ""),
            description=m.get("description", ""),
            category=m.get("category", "uncategorized"),
            end_date=m.get("end_date"),
            current_price=m.get("current_price", 0),
            current_volume=m.get("current_volume", 0),
            current_liquidity=m.get("current_liquidity", 0),
            odds_swing_pct=m.get("odds_swing_pct", 0),
            volume_surge_pct=m.get("volume_surge_pct", 0),
            composite_score=m.get("composite_score", 0),
            passed_initial_filter=m.get("passed_initial_filter", False),
            velocity_flag=m.get("velocity_flag", False),
            velocity_score=m.get("velocity_score", 0),
            velocity_detail=m.get("velocity_detail", ""),
            liquidity_shift_flag=m.get("liquidity_shift_flag", False),
            liquidity_shift_score=m.get("liquidity_shift_score", 0),
            liquidity_shift_detail=m.get("liquidity_shift_detail", ""),
            time_decay_urgency_flag=m.get("time_decay_urgency_flag", False),
            time_decay_score=m.get("time_decay_score", 0),
            time_decay_detail=m.get("time_decay_detail", ""),
            spread_flag=m.get("spread_flag", False),
            spread_score=m.get("spread_score", 0),
            spread_detail=m.get("spread_detail", ""),
            volume_weighted_flag=m.get("volume_weighted_flag", False),
            volume_weighted_score=m.get("volume_weighted_score", 0),
            volume_weighted_detail=m.get("volume_weighted_detail", ""),
            deep_analysis_score=m.get("deep_analysis_score", 0),
            flags_triggered=m.get("flags_triggered", []),
            # poly_dates / poly_prices / poly_volumes left empty (snapshot only)
        ))

    return summary, markets


def load_polymarket_csv(
    csv_path: str,
) -> tuple[dict, list[PolymarketRecord]]:
    """
    Load Polymarket data from the production CSV export.

    The CSV is expected to have one row per market with the following columns:
      market_id, question, category, end_date, current_price, current_as_of,
      volume, liquidity, volume24hr, one_month_price_change, spread, …
      price_t_minus_24 … price_t_minus_01  (hourly prices, 24h before snapshot)
      price_t_plus_01  … price_t_plus_24   (hourly prices, 24h after  snapshot)

    Missing price cells (empty / NaN) are forward-filled then back-filled so
    that Polymarket's "last traded price" semantic is preserved.

    Args:
        csv_path: Local path to the Polymarket CSV file.

    Returns:
        (summary_dict, list_of_PolymarketRecord)
        Each record has poly_dates and poly_prices populated with up to 49
        UTC-anchored hourly observations.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Only use price columns that are actually present
    t_minus_present = [c for c in _T_MINUS_COLS if c in df.columns]
    t_plus_present  = [c for c in _T_PLUS_COLS  if c in df.columns]

    markets: list[PolymarketRecord] = []

    def _safe_float(row, key: str, default: float = 0.0) -> float:
        try:
            v = row.get(key)
            return float(v) if v is not None and pd.notna(v) else default
        except Exception:
            return default

    for _, row in df.iterrows():
        # ── Parse anchor timestamp ────────────────────────────────────────
        current_ts: Optional[pd.Timestamp] = None
        raw_ts = row.get("current_as_of", "")
        if raw_ts and str(raw_ts) not in ("", "nan"):
            try:
                current_ts = pd.Timestamp(raw_ts, tz="UTC")
            except Exception:
                pass

        # ── Build ordered price + date lists ─────────────────────────────
        # Time order: t-24h … t-1h, t=0, t+1h … t+24h
        raw_prices: list[float] = []
        raw_dates:  list[str]   = []

        for col in t_minus_present:
            hours_back = int(col.split("_")[-1])   # e.g. 24 from "price_t_minus_24"
            val = row.get(col)
            raw_prices.append(float(val) if pd.notna(val) and val != "" else float("nan"))
            ts_label = (
                (current_ts - pd.Timedelta(hours=hours_back)).isoformat()
                if current_ts is not None else col
            )
            raw_dates.append(ts_label)

        # t = 0 (snapshot price)
        cur_val = row.get("current_price")
        raw_prices.append(float(cur_val) if pd.notna(cur_val) and cur_val != "" else float("nan"))
        raw_dates.append(current_ts.isoformat() if current_ts is not None else "current")

        for col in t_plus_present:
            hours_fwd = int(col.split("_")[-1])
            val = row.get(col)
            raw_prices.append(float(val) if pd.notna(val) and val != "" else float("nan"))
            ts_label = (
                (current_ts + pd.Timedelta(hours=hours_fwd)).isoformat()
                if current_ts is not None else col
            )
            raw_dates.append(ts_label)

        # Forward-fill then back-fill NaNs (last-traded-price semantic)
        price_s = pd.Series(raw_prices, index=raw_dates, dtype=float).ffill().bfill()
        price_s = price_s.dropna()

        poly_dates  = price_s.index.tolist()
        poly_prices = price_s.values.tolist()

        # ── Odds swing: range of the hourly series ────────────────────────
        if len(poly_prices) >= 2:
            odds_swing = round(abs(max(poly_prices) - min(poly_prices)) * 100, 4)
        else:
            try:
                odds_swing = round(abs(float(row.get("one_month_price_change") or 0)) * 100, 4)
            except Exception:
                odds_swing = 0.0

        markets.append(PolymarketRecord(
            market_id=str(row["market_id"]),
            question=str(row.get("question", "")),
            description="",
            category=str(row.get("category", "uncategorized")),
            end_date=str(row.get("end_date", "")),
            current_price=_safe_float(row, "current_price"),
            current_volume=_safe_float(row, "volume24hr") or _safe_float(row, "volume"),
            current_liquidity=_safe_float(row, "liquidity"),
            odds_swing_pct=odds_swing,
            volume_surge_pct=0.0,    # not in CSV
            composite_score=0.5,     # not computed in CSV; neutral default
            passed_initial_filter=True,
            # Hourly time series
            poly_dates=poly_dates,
            poly_prices=poly_prices,
            poly_volumes=[],
            # t=0 anchor timestamp (used by realtime_analyst to locate the spike point)
            current_as_of=current_ts.isoformat() if current_ts is not None else "",
        ))

    summary = {
        "scan_timestamp": markets[0].current_as_of if markets else "",
        "source": csv_path,
        "total_markets": len(markets),
    }

    logger.info("Loaded %d markets from CSV (%s)", len(markets), csv_path)
    return summary, markets


# ══════════════════════════════════════════════════════════════════════════════
# ALIGNMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _align_series(
    poly_dates: list[str],
    poly_prices: list[float],
    fin_dates: list[str],
    fin_prices: list[float],
    fin_volumes: list[float],
) -> Optional[dict]:
    """
    Align Polymarket hourly prices with financial series by UTC timestamp.

    Both series are converted to UTC, floored to the nearest hour, then
    inner-joined.  Duplicate timestamps (if any) keep the last observation.
    Log-returns are computed from aligned prices for both series.

    Returns None when fewer than _MIN_ALIGNED_PTS overlap observations exist.
    """
    try:
        poly_ts = pd.to_datetime(poly_dates, utc=True, errors="coerce")
        fin_ts  = pd.to_datetime(fin_dates,  utc=True, errors="coerce")
    except Exception as exc:
        logger.debug("Timestamp parse failed during alignment: %s", exc)
        return None

    # Floor to hour so ±30-min clock drift still matches
    poly_idx = poly_ts.floor("h")
    fin_idx  = fin_ts.floor("h")

    poly_df = (
        pd.DataFrame({"price": poly_prices}, index=poly_idx)
        .dropna()
        .loc[lambda df: ~df.index.duplicated(keep="last")]
        .sort_index()
    )

    fin_has_vol = len(fin_volumes) == len(fin_prices)
    fin_df = (
        pd.DataFrame({
            "price":  fin_prices,
            "volume": fin_volumes if fin_has_vol else [0.0] * len(fin_prices),
        }, index=fin_idx)
        .dropna(subset=["price"])
        .loc[lambda df: ~df.index.duplicated(keep="last")]
        .sort_index()
    )

    merged = poly_df.join(fin_df, how="inner", lsuffix="_poly", rsuffix="_fin")
    merged = merged.sort_index()

    if len(merged) < _MIN_ALIGNED_PTS:
        logger.debug(
            "Only %d overlapping timestamps found (need %d) — skipping real alignment",
            len(merged), _MIN_ALIGNED_PTS,
        )
        return None

    poly_p = merged["price_poly"].to_numpy(dtype=np.float64)
    fin_p  = merged["price_fin"].to_numpy(dtype=np.float64)
    fin_v  = merged["volume"].to_numpy(dtype=np.float64)

    # Log-returns (N-1 length, consistent with the rest of the toolkit)
    poly_r = np.diff(np.log(np.clip(poly_p, 1e-10, None)))
    fin_r  = np.diff(np.log(np.clip(fin_p,  1e-10, None)))

    logger.debug("Aligned %d observations for statistical tests", len(merged))
    return {
        "dates":        merged.index.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
        "poly_prices":  poly_p,
        "poly_returns": poly_r,
        "poly_volumes": np.zeros(len(poly_p)),
        "fin_prices":   fin_p,
        "fin_returns":  fin_r,
        "fin_volumes":  fin_v,
        "has_poly_timeseries": True,
    }


def build_aligned_series(
    poly_market: PolymarketRecord,
    fin_record: FinancialMarketRecord,
) -> Optional[dict]:
    """
    Build aligned numpy arrays for a single Polymarket–Equity pair.

    Strategy
    --------
    1. **Real series** (CSV mode): if ``poly_market.poly_prices`` is populated,
       align the hourly Polymarket price series with the financial time series
       by UTC timestamp.  Returns ``has_poly_timeseries = True``.

    2. **Snapshot fallback** (scan-JSON mode): if no Polymarket time series is
       available, the financial series is used as a proxy for both sides.
       Statistical results in this mode have no real informational content —
       the caveats field in PairAnalysis will flag this.
       Returns ``has_poly_timeseries = False``.

    Returns None if the financial record has no data.
    """
    if not fin_record.dates or not fin_record.prices:
        return None

    # ── Case 1: Real Polymarket hourly time series ────────────────────────
    if poly_market.poly_prices:
        aligned = _align_series(
            poly_market.poly_dates,
            poly_market.poly_prices,
            fin_record.dates,
            fin_record.prices,
            fin_record.volumes or [],
        )
        if aligned is not None:
            aligned.update({
                "poly_price":       poly_market.current_price,
                "poly_volume":      poly_market.current_volume,
                "poly_odds_swing":  poly_market.odds_swing_pct,
                "poly_composite":   poly_market.composite_score,
            })
            return aligned

        logger.warning(
            "Real Polymarket series available for %s but timestamp alignment failed "
            "(fin series has %d pts); falling back to proxy mode.",
            poly_market.market_id, len(fin_record.dates),
        )

    # ── Case 2: Snapshot only — use fin data as both-side proxy ──────────
    fin_prices_arr = np.array(fin_record.prices, dtype=np.float64)
    fin_returns_arr = (
        np.array(fin_record.returns, dtype=np.float64)
        if fin_record.returns
        else np.diff(np.log(fin_prices_arr + 1e-10))
    )
    fin_volumes_arr = (
        np.array(fin_record.volumes, dtype=np.float64)
        if fin_record.volumes
        else np.zeros(len(fin_record.prices))
    )

    return {
        "dates":            np.array(fin_record.dates),
        "fin_prices":       fin_prices_arr,
        "fin_returns":      fin_returns_arr,
        "fin_volumes":      fin_volumes_arr,
        "poly_prices":      fin_prices_arr,   # proxy
        "poly_returns":     fin_returns_arr,  # proxy
        "poly_volumes":     fin_volumes_arr,  # proxy
        "poly_price":       poly_market.current_price,
        "poly_volume":      poly_market.current_volume,
        "poly_odds_swing":  poly_market.odds_swing_pct,
        "poly_composite":   poly_market.composite_score,
        "has_poly_timeseries": False,
    }

# ============================================================================
# FORMAT D: POLYMARKET DATA FROM FINANCIAL PIPELINE JSON
# ============================================================================

def load_polymarket_from_financial_json(
    source: str,
) -> tuple[dict, list]:
    """
    Load PolymarketRecord objects directly from a Format D financial pipeline JSON.

    This allows the Trend Analyst to run with only a financial pipeline JSON file
    (no separate Polymarket CSV or scan JSON required) when the financial pipeline
    has produced a Format D output that embeds Polymarket probability series.

    The Polymarket probability series is sourced from:
      1. probability_series (5-min resolution, resampled to hourly)  -- preferred
      2. time_series.mid_price (hourly)                              -- fallback

    Args:
        source: Local filepath or HTTP/HTTPS URL to the financial pipeline JSON.

    Returns:
        (summary_dict, list[PolymarketRecord])

    Raises:
        ValueError: If the JSON is not Format D.
    """
    from tools.financial_adapter import load_polymarket_from_format_d
    return load_polymarket_from_format_d(source)
