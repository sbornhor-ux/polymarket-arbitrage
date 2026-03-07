"""
Tests for finance_agent.stats

Uses synthetic DataFrames to avoid any network calls.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import timezone
from unittest.mock import patch

from polymarket_agent.models import SwingEvent, DetectionRule
from finance_agent.stats import compute_window_stats, _compute_z_score, _infer_units
from finance_agent.models import DataQuality


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_swing(
    swing_id: str = "S1",
    window: str = "6H",
    p_pre: float = 0.40,
    p_post: float = 0.55,
    start: str = "2026-01-10T08:00:00+00:00",
    end: str = "2026-01-10T14:00:00+00:00",
) -> SwingEvent:
    return SwingEvent(
        swing_id=swing_id,
        window=window,
        window_start_utc=start,
        window_end_utc=end,
        p_pre=p_pre,
        p_post=p_post,
        delta_p=p_post - p_pre,
        abs_delta_p=abs(p_post - p_pre),
        z_score_vs_trailing=2.5,
        detection_rule=DetectionRule(),
    )


def _make_series(
    start: str = "2026-01-01T00:00:00",
    periods: int = 24 * 70,  # ~70 days of hourly data
    freq: str = "1h",
    base: float = 4500.0,
    noise_scale: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic hourly OHLCV series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    closes = base + np.cumsum(rng.normal(0, noise_scale, periods))
    highs = closes + rng.uniform(0, noise_scale / 2, periods)
    lows = closes - rng.uniform(0, noise_scale / 2, periods)
    volumes = rng.uniform(1e6, 1e8, periods)
    return pd.DataFrame(
        {"close": closes, "high": highs, "low": lows, "volume": volumes},
        index=idx,
    )


_MOCK_TICKER_INFO = {"market_cap": 1_234_567_890.0}


# ---------------------------------------------------------------------------
# Tests: compute_window_stats — existing contract (must not break)
# ---------------------------------------------------------------------------

def test_returns_stats_object():
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    assert result is not None
    assert result.swing_id == "S1"
    assert result.series_id == "^GSPC"
    assert result.series_label == "S&P 500 (via SPY)"


def test_level_pre_post_populated():
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    assert result.level_pre is not None
    assert result.level_post is not None


def test_move_equals_post_minus_pre():
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    expected = result.level_post - result.level_pre
    assert result.move_pre_to_post == pytest.approx(expected, abs=1e-3)


def test_move_pct_sign_matches_move():
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    if result.move_pre_to_post > 0:
        assert result.move_pct > 0
    elif result.move_pre_to_post < 0:
        assert result.move_pct < 0


def test_empty_window_returns_missing_quality():
    """If no data exists in the window, data_quality.missing_pct == 1.0."""
    swing = _make_swing(
        start="2020-01-01T00:00:00+00:00",
        end="2020-01-01T06:00:00+00:00",
    )
    df = _make_series(start="2026-01-01T00:00:00")  # far future → no overlap
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    assert result is not None
    assert result.data_quality.missing_pct == 1.0


def test_z_score_is_float_or_none():
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    assert result.z_score_vs_trailing is None or isinstance(result.z_score_vs_trailing, float)


def test_corr_bounded():
    """Pearson correlation must be in [-1, 1]."""
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    if result.corr_with_prob_change is not None:
        assert -1.0 <= result.corr_with_prob_change <= 1.0


# ---------------------------------------------------------------------------
# Tests: compute_window_stats — new fields
# ---------------------------------------------------------------------------

def test_polymarket_id_passed_through():
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df, market_id="0xabc")
    assert result.polymarket_id == "0xabc"


def test_sector_populated():
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    assert result.sector == "Equities"


def test_market_cap_from_ticker_info():
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value={"market_cap": 5e12}):
        result = compute_window_stats("^GSPC", swing, df)
    assert result.market_cap == pytest.approx(5e12)


def test_time_series_populated():
    """time_series.dates should have the same length as the window slice."""
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    assert len(result.time_series.dates) > 0
    assert len(result.time_series.prices) == len(result.time_series.dates)
    assert len(result.time_series.returns) == len(result.time_series.dates)


def test_time_series_intraday_high_low():
    """time_series.intraday_high/low should be populated from high/low columns."""
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    assert len(result.time_series.intraday_high) == len(result.time_series.dates)
    assert len(result.time_series.intraday_low) == len(result.time_series.dates)


def test_time_series_volumes():
    """time_series.volumes should be populated when volume column is present."""
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    assert len(result.time_series.volumes) == len(result.time_series.dates)


def test_bid_ask_spread_always_empty():
    """bid_ask_spread should always be empty (unavailable from yfinance historical)."""
    swing = _make_swing()
    df = _make_series()
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df)
    assert result.time_series.bid_ask_spread == []


def test_benchmark_returns_populated():
    """benchmark_returns should be populated when benchmark_df is provided."""
    swing = _make_swing()
    df = _make_series()
    bm_df = _make_series(seed=99)
    with patch("finance_agent.stats.fetch_ticker_info", return_value=_MOCK_TICKER_INFO):
        result = compute_window_stats("^GSPC", swing, df, benchmark_df=bm_df)
    assert len(result.benchmark_returns) > 0


# ---------------------------------------------------------------------------
# Tests: _compute_z_score
# ---------------------------------------------------------------------------

def test_z_score_positive_for_large_move():
    baseline = pd.Series(np.random.default_rng(0).normal(4500, 1.0, 100))
    big_move = 50.0  # much larger than std
    z = _compute_z_score(big_move, baseline)
    assert z is not None
    assert z > 0


def test_z_score_none_for_short_baseline():
    baseline = pd.Series([4500.0, 4501.0])  # only 2 points
    z = _compute_z_score(10.0, baseline)
    assert z is None


# ---------------------------------------------------------------------------
# Tests: _infer_units
# ---------------------------------------------------------------------------

def test_units_rate_tickers():
    assert _infer_units("^TNX") == "pct_pts"
    assert _infer_units("^IRX") == "pct_pts"


def test_units_vix():
    assert _infer_units("^VIX") == "vix_pts"


def test_units_futures():
    assert _infer_units("CL=F") == "usd"
    assert _infer_units("GC=F") == "usd"


def test_units_default():
    assert _infer_units("^GSPC") == "native"
    assert _infer_units("BTC-USD") == "native"
