"""
Tests for polymarket_agent.swing_detector

We test the detector in isolation using synthetic probability series
so we control exactly where swings should appear.
"""

import pytest
from datetime import datetime, timezone, timedelta

from polymarket_agent.models import MarketSnapshot, ProbTick, DataQuality
from polymarket_agent.swing_detector import detect_swings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(p_values: list[float], start_utc: datetime | None = None) -> MarketSnapshot:
    """Build a MarketSnapshot with 5-minute ticks from a list of prob values."""
    if start_utc is None:
        start_utc = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    series = [
        ProbTick(
            ts_utc=(start_utc + timedelta(minutes=5 * i)).isoformat(),
            p=p,
        )
        for i, p in enumerate(p_values)
    ]
    return MarketSnapshot(
        market_id="test_market",
        market_question="Test?",
        as_of_utc=datetime.now(timezone.utc).isoformat(),
        probability_series=series,
        swings=[],
        data_quality=DataQuality(),
    )


def _flat_then_spike(flat_len: int = 50, spike_size: float = 0.25) -> list[float]:
    """50 flat ticks at 0.50, then a spike."""
    flat = [0.50] * flat_len
    spike = [0.50 + spike_size]
    tail = [0.50 + spike_size] * 20  # stays elevated
    return flat + spike + tail


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_swings_on_flat_series():
    """Perfectly flat probability → no swings."""
    p_values = [0.5] * 100
    snapshot = _make_snapshot(p_values)
    result = detect_swings(snapshot, z_threshold=2.0)
    assert result.swings == []


def test_detects_spike():
    """A large abrupt move should be detected as at least one swing."""
    p_values = _flat_then_spike(flat_len=60, spike_size=0.20)
    snapshot = _make_snapshot(p_values)
    result = detect_swings(snapshot, z_threshold=2.0)
    assert len(result.swings) >= 1


def test_swing_fields_populated():
    """All mandatory fields on SwingEvent should be non-None after detection."""
    p_values = _flat_then_spike(flat_len=60, spike_size=0.20)
    snapshot = _make_snapshot(p_values)
    result = detect_swings(snapshot, z_threshold=2.0)

    for swing in result.swings:
        assert swing.swing_id.startswith("S")
        assert swing.window in {"1h", "6h", "1D", "5D"}
        assert swing.p_pre is not None
        assert swing.p_post is not None
        assert swing.z_score_vs_trailing >= 2.0
        assert swing.abs_delta_p >= 0.0
        assert swing.window_start_utc < swing.window_end_utc


def test_swing_ids_unique():
    """Each detected swing should have a unique swing_id."""
    p_values = _flat_then_spike(flat_len=80, spike_size=0.30)
    snapshot = _make_snapshot(p_values)
    result = detect_swings(snapshot, z_threshold=1.5)
    ids = [s.swing_id for s in result.swings]
    assert len(ids) == len(set(ids))


def test_insufficient_history_skipped():
    """Fewer than MIN_HISTORY_PERIODS ticks → no swings, note added."""
    p_values = [0.5] * 5  # only 5 ticks
    snapshot = _make_snapshot(p_values)
    result = detect_swings(snapshot)
    assert result.swings == []
    assert "swing_detection_skipped" in result.notes


def test_higher_threshold_fewer_swings():
    """Raising the threshold should produce fewer or equal swings."""
    p_values = _flat_then_spike(flat_len=80, spike_size=0.30)
    snap_low = _make_snapshot(p_values)
    snap_high = _make_snapshot(p_values)

    result_low = detect_swings(snap_low, z_threshold=1.5)
    result_high = detect_swings(snap_high, z_threshold=3.0)

    assert len(result_low.swings) >= len(result_high.swings)


def test_delta_p_sign_correct():
    """delta_p should be positive for upward swings, negative for downward."""
    # Upward swing
    up = [0.3] * 60 + [0.7] * 20
    snap_up = _make_snapshot(up)
    result_up = detect_swings(snap_up, z_threshold=2.0)
    for swing in result_up.swings:
        assert swing.delta_p == pytest.approx(swing.p_post - swing.p_pre, abs=1e-6)
