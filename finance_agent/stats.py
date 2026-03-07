"""
Computes pre/post move statistics, trailing z-scores, and lead-lag
correlations for a financial series over a given swing window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from polymarket_agent.models import SwingEvent
from finance_agent.models import (
    FinanceSeriesWindowStats,
    FinanceTimeSeries,
    SERIES_LABELS,
    SERIES_SECTORS,
    DataQuality,
)
from finance_agent.fetcher import fetch_ticker_info


def compute_window_stats(
    ticker: str,
    swing: SwingEvent,
    series_df: pd.DataFrame,
    trailing_days: int = 60,
    market_id: str = "",
    benchmark_df: pd.DataFrame | None = None,
    benchmark_ticker: str = "^GSPC",
) -> FinanceSeriesWindowStats | None:
    """
    Compute stats for (ticker, swing_event) pair.

    Args:
        ticker: yfinance ticker string.
        swing: SwingEvent with window_start_utc and window_end_utc.
        series_df: DataFrame with 'close' column (and optionally 'high', 'low',
                   'volume') and UTC DatetimeIndex (from fetcher.fetch_series).
        trailing_days: Days before window_start used for z-score baseline.
        market_id: Polymarket market_id (join key for downstream consumers).
        benchmark_df: Optional benchmark series DataFrame (same format as series_df).
        benchmark_ticker: Ticker symbol of the benchmark.

    Returns:
        FinanceSeriesWindowStats or None if data is insufficient.
    """
    window_start = pd.Timestamp(swing.window_start_utc)
    window_end = pd.Timestamp(swing.window_end_utc)
    baseline_start = window_start - pd.Timedelta(days=trailing_days)

    # Slice window data (full row — all available columns)
    window_mask = (series_df.index >= window_start) & (series_df.index <= window_end)
    window_full = series_df.loc[window_mask]
    window_data = window_full["close"] if not window_full.empty else pd.Series(dtype=float)

    if window_data.empty:
        return FinanceSeriesWindowStats(
            swing_id=swing.swing_id,
            series_id=ticker,
            series_label=SERIES_LABELS.get(ticker, ticker),
            polymarket_id=market_id,
            sector=SERIES_SECTORS.get(ticker, ""),
            window=swing.window,
            window_start_utc=swing.window_start_utc,
            window_end_utc=swing.window_end_utc,
            benchmark_ticker=benchmark_ticker,
            data_quality=DataQuality(
                missing_pct=1.0,
                notes=["No data in window — market may have been closed"],
            ),
        )

    level_pre = float(window_data.iloc[0])
    level_post = float(window_data.iloc[-1])
    move = level_post - level_pre
    move_pct = ((level_post / level_pre) - 1.0) * 100.0 if level_pre != 0 else None

    # Trailing baseline for z-score
    baseline_data = series_df.loc[
        (series_df.index >= baseline_start) & (series_df.index < window_start),
        "close",
    ]
    z_score = _compute_z_score(move, baseline_data)

    # Lead-lag correlation vs Polymarket Δp
    corr, best_lag_hours = _compute_lead_lag(swing, window_data, series_df)

    # Data quality
    n_window = len(window_data)
    n_expected = max(
        1,
        int((window_end - window_start).total_seconds() / 3600),  # hourly buckets
    )
    missing_pct = max(0.0, (n_expected - n_window) / n_expected)

    # Build FinanceTimeSeries from window slice
    ts = _build_time_series(window_full)

    # Benchmark returns over same window
    bm_returns: list[float] = []
    if benchmark_df is not None and not benchmark_df.empty:
        bm_window = benchmark_df.loc[window_mask, "close"] if window_mask.any() else pd.Series(dtype=float)
        if not bm_window.empty:
            bm_returns = bm_window.pct_change().fillna(0).round(6).tolist()

    # Market cap (live call — gracefully returns None on failure)
    ticker_info = fetch_ticker_info(ticker)

    return FinanceSeriesWindowStats(
        swing_id=swing.swing_id,
        series_id=ticker,
        series_label=SERIES_LABELS.get(ticker, ticker),
        polymarket_id=market_id,
        sector=SERIES_SECTORS.get(ticker, ""),
        market_cap=ticker_info.get("market_cap"),
        window=swing.window,
        window_start_utc=swing.window_start_utc,
        window_end_utc=swing.window_end_utc,
        level_pre=round(level_pre, 4),
        level_post=round(level_post, 4),
        move_pre_to_post=round(move, 4),
        move_pct=round(move_pct, 4) if move_pct is not None else None,
        units=_infer_units(ticker),
        time_series=ts,
        z_score_vs_trailing=round(z_score, 3) if z_score is not None else None,
        trailing_days=trailing_days,
        corr_with_prob_change=round(corr, 3) if corr is not None else None,
        best_lag_hours=round(best_lag_hours, 1) if best_lag_hours is not None else None,
        benchmark_ticker=benchmark_ticker,
        benchmark_returns=bm_returns,
        data_quality=DataQuality(missing_pct=round(missing_pct, 3)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_time_series(window_full: pd.DataFrame) -> FinanceTimeSeries:
    """Build a FinanceTimeSeries from a window-sliced DataFrame."""
    if window_full.empty:
        return FinanceTimeSeries()

    close = window_full["close"]
    dates = [ts.isoformat() for ts in window_full.index]
    prices = [round(float(v), 4) for v in close.tolist()]
    returns = [round(float(v), 6) for v in close.pct_change().fillna(0).tolist()]

    volumes: list[float] = []
    if "volume" in window_full.columns:
        volumes = [round(float(v), 2) for v in window_full["volume"].fillna(0).tolist()]

    intraday_high: list[float] = []
    if "high" in window_full.columns:
        intraday_high = [round(float(v), 4) for v in window_full["high"].tolist()]

    intraday_low: list[float] = []
    if "low" in window_full.columns:
        intraday_low = [round(float(v), 4) for v in window_full["low"].tolist()]

    return FinanceTimeSeries(
        dates=dates,
        prices=prices,
        returns=returns,
        volumes=volumes,
        intraday_high=intraday_high,
        intraday_low=intraday_low,
        # bid_ask_spread always [] — unavailable from yfinance historical data
    )


def _compute_z_score(move: float, baseline: pd.Series) -> float | None:
    """Z-score of |move| vs distribution of |hourly moves| in baseline."""
    if len(baseline) < 5:
        return None
    returns = baseline.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return None
    abs_returns = returns.abs()
    abs_move = abs(move / baseline.iloc[-1]) if baseline.iloc[-1] != 0 else abs(move)
    return float((abs_move - abs_returns.mean()) / abs_returns.std())


def _compute_lead_lag(
    swing: SwingEvent,
    window_data: pd.Series,
    full_series: pd.DataFrame,
    max_lag_hours: int = 24,
) -> tuple[float | None, float | None]:
    """
    Estimate Pearson correlation and the best lag between series returns
    and the Polymarket probability swing.

    Returns (correlation_at_best_lag, best_lag_hours).
    Negative lag = series leads Polymarket.
    """
    if len(window_data) < 3:
        return None, None

    # Build a simple linear proxy for Polymarket Δp over the window
    n = len(window_data)
    poly_proxy = np.linspace(swing.p_pre, swing.p_post, n)
    poly_diff = np.diff(poly_proxy)

    series_returns = window_data.pct_change().dropna().values

    if len(series_returns) < 2:
        return None, None

    # Align lengths
    min_len = min(len(poly_diff), len(series_returns))
    poly_diff = poly_diff[:min_len]
    series_returns = series_returns[:min_len]

    # Try lags from -max_lag to +max_lag (in index steps, not hours)
    # Approximate: each step ≈ 1h (hourly data)
    max_lag_steps = min(max_lag_hours, min_len - 2)
    best_corr = 0.0
    best_lag = 0.0

    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag == 0:
            s = series_returns
            p = poly_diff
        elif lag > 0:
            s = series_returns[lag:]
            p = poly_diff[:-lag]
        else:  # lag < 0
            s = series_returns[:lag]
            p = poly_diff[-lag:]

        if len(s) < 3:
            continue
        try:
            corr, _ = pearsonr(s, p)
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = float(lag)
        except Exception:
            continue

    if best_corr == 0.0:
        return None, None

    return float(best_corr), best_lag


def _infer_units(ticker: str) -> str:
    """Return human-readable units label for a ticker."""
    rate_tickers = {"^TNX", "^IRX", "^FVX", "^TYX"}
    if ticker in rate_tickers:
        return "pct_pts"
    if ticker == "^VIX":
        return "vix_pts"
    if ticker.endswith("=F"):
        return "usd"
    return "native"
