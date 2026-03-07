"""
Tests for finance_agent.fetcher

Network calls are mocked via unittest.mock to keep tests fast and offline.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from datetime import timezone

from polymarket_agent.models import SwingEvent, DetectionRule
from finance_agent.fetcher import fetch_series


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_swing(
    start: str = "2026-01-10T08:00:00+00:00",
    end: str = "2026-01-10T14:00:00+00:00",
) -> SwingEvent:
    return SwingEvent(
        swing_id="S1",
        window="6H",
        window_start_utc=start,
        window_end_utc=end,
        p_pre=0.4,
        p_post=0.55,
        delta_p=0.15,
        abs_delta_p=0.15,
        z_score_vs_trailing=2.5,
        detection_rule=DetectionRule(),
    )


def _make_aggs(n_periods: int = 500) -> list:
    """Return a list of Agg-like objects with OHLCV + timestamp fields."""
    base_ts = int(pd.Timestamp("2025-10-01", tz="UTC").timestamp() * 1000)
    hour_ms = 3_600_000
    rng = np.random.default_rng(0)
    closes = 4500 + np.cumsum(rng.normal(0, 5, n_periods))
    highs = closes + rng.uniform(0, 10, n_periods)
    lows = closes - rng.uniform(0, 10, n_periods)
    volumes = rng.uniform(1e6, 1e8, n_periods)
    return [
        SimpleNamespace(
            timestamp=base_ts + i * hour_ms,
            open=closes[i],
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=volumes[i],
        )
        for i in range(n_periods)
    ]


def _make_mock_client(aggs: list | None = None) -> MagicMock:
    """Return a mock RESTClient whose get_aggs() returns the given agg list."""
    client = MagicMock()
    client.get_aggs.return_value = aggs if aggs is not None else _make_aggs()
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fetch_returns_dataframe():
    swing = _make_swing()
    mock_client = _make_mock_client()

    with patch("finance_agent.fetcher.RESTClient", return_value=mock_client):
        result = fetch_series("^GSPC", [swing], trailing_days=60)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "close" in result.columns


def test_close_column_normalised():
    """Output should have a lowercase 'close' column."""
    swing = _make_swing()
    mock_client = _make_mock_client()

    with patch("finance_agent.fetcher.RESTClient", return_value=mock_client):
        result = fetch_series("^GSPC", [swing], trailing_days=60)

    assert "close" in result.columns


def test_index_is_utc():
    swing = _make_swing()
    mock_client = _make_mock_client()

    with patch("finance_agent.fetcher.RESTClient", return_value=mock_client):
        result = fetch_series("^GSPC", [swing], trailing_days=60)

    assert result.index.tzinfo is not None
    assert str(result.index.tzinfo) in {"UTC", "utc"}


def test_returns_none_on_empty_download():
    swing = _make_swing()
    mock_client = _make_mock_client(aggs=[])

    with patch("finance_agent.fetcher.RESTClient", return_value=mock_client):
        result = fetch_series("^GSPC", [swing], trailing_days=60)

    assert result is None


def test_returns_none_on_exception():
    swing = _make_swing()
    mock_client = MagicMock()
    mock_client.get_aggs.side_effect = Exception("network error")

    with patch("finance_agent.fetcher.RESTClient", return_value=mock_client):
        result = fetch_series("^GSPC", [swing], trailing_days=60)

    assert result is None


def test_empty_swing_list_returns_none():
    result = fetch_series("^GSPC", [], trailing_days=60)
    assert result is None


def test_index_sorted():
    swing = _make_swing()
    aggs = _make_aggs()
    # Shuffle the mock data
    import random
    random.seed(42)
    random.shuffle(aggs)
    mock_client = _make_mock_client(aggs=aggs)

    with patch("finance_agent.fetcher.RESTClient", return_value=mock_client):
        result = fetch_series("^GSPC", [swing], trailing_days=60)

    assert result.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Tests — new High/Low/Volume columns
# ---------------------------------------------------------------------------

def test_high_low_volume_columns():
    """Fetcher should return high, low, and volume columns alongside close."""
    swing = _make_swing()
    mock_client = _make_mock_client()

    with patch("finance_agent.fetcher.RESTClient", return_value=mock_client):
        result = fetch_series("^GSPC", [swing], trailing_days=60)

    assert "high" in result.columns
    assert "low" in result.columns
    assert "volume" in result.columns


def test_high_gte_close():
    """high should always be >= close."""
    swing = _make_swing()
    mock_client = _make_mock_client()

    with patch("finance_agent.fetcher.RESTClient", return_value=mock_client):
        result = fetch_series("^GSPC", [swing], trailing_days=60)

    assert (result["high"] >= result["close"]).all()


def test_low_lte_close():
    """low should always be <= close."""
    swing = _make_swing()
    mock_client = _make_mock_client()

    with patch("finance_agent.fetcher.RESTClient", return_value=mock_client):
        result = fetch_series("^GSPC", [swing], trailing_days=60)

    assert (result["low"] <= result["close"]).all()
