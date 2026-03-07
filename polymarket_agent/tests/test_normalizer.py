"""
Tests for polymarket_agent.normalizer
"""

import pytest
from polymarket_agent.normalizer import normalize_market


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw(n_ticks: int = 10, missing: int = 0, extra_meta: dict | None = None) -> dict:
    """Build a minimal raw API response dict."""
    ticks = [{"t": 1_700_000_000 + i * 60, "p": 0.4 + i * 0.01} for i in range(n_ticks)]
    # Inject bad ticks to simulate missing data
    for i in range(missing):
        ticks[i] = {"bad_key": "junk"}
    metadata = {
        "condition_id": "0xabc123",
        "question": "Will X happen?",
        "market_slug": "will-x-happen",
        "tokens": [{"token_id": "tok1", "outcome": "Yes", "price": 0.5}],
    }
    if extra_meta:
        metadata.update(extra_meta)
    return {
        "metadata": metadata,
        "history": ticks,
    }


# ---------------------------------------------------------------------------
# Tests — existing contract (must not break)
# ---------------------------------------------------------------------------

def test_normalize_market_basic():
    raw = _make_raw(n_ticks=10)
    snapshot = normalize_market(raw)

    assert snapshot.market_id == "0xabc123"
    assert snapshot.market_question == "Will X happen?"
    assert snapshot.market_slug == "will-x-happen"
    assert len(snapshot.probability_series) == 10
    assert snapshot.swings == []


def test_prob_series_values():
    raw = _make_raw(n_ticks=5)
    snapshot = normalize_market(raw)

    probs = [t.p for t in snapshot.probability_series]
    # p values should be 0.40, 0.41, 0.42, 0.43, 0.44
    assert probs[0] == pytest.approx(0.40, abs=1e-6)
    assert probs[-1] == pytest.approx(0.44, abs=1e-6)


def test_prob_series_timestamps_are_iso():
    raw = _make_raw(n_ticks=3)
    snapshot = normalize_market(raw)
    for tick in snapshot.probability_series:
        # Should parse without error
        from datetime import datetime
        dt = datetime.fromisoformat(tick.ts_utc)
        assert dt is not None


def test_missing_ticks_tracked():
    raw = _make_raw(n_ticks=20, missing=4)
    snapshot = normalize_market(raw)

    # 4 out of 20 ticks are malformed → 20% missing
    assert snapshot.data_quality.missing_pct == pytest.approx(0.20, abs=1e-6)
    assert len(snapshot.probability_series) == 16


def test_empty_history():
    raw = _make_raw(n_ticks=0)
    snapshot = normalize_market(raw)
    assert snapshot.probability_series == []
    assert snapshot.data_quality.missing_pct == 0.0


def test_missing_metadata_fields_use_defaults():
    raw = {
        "metadata": {
            "tokens": [{"token_id": "t1", "outcome": "Yes", "price": 0.5}],
        },
        "history": [{"t": 1_700_000_000, "p": 0.5}],
    }
    snapshot = normalize_market(raw)
    assert snapshot.market_id == "unknown"
    assert snapshot.market_question == ""
    assert snapshot.market_slug == ""


# ---------------------------------------------------------------------------
# Tests — new time_series parallel lists
# ---------------------------------------------------------------------------

def test_time_series_parallel_lists():
    """time_series.dates and mid_price must mirror probability_series."""
    raw = _make_raw(n_ticks=5)
    snapshot = normalize_market(raw)

    assert len(snapshot.time_series.dates) == len(snapshot.probability_series)
    assert len(snapshot.time_series.mid_price) == len(snapshot.probability_series)

    for tick, date, mid in zip(
        snapshot.probability_series,
        snapshot.time_series.dates,
        snapshot.time_series.mid_price,
    ):
        assert tick.ts_utc == date
        assert tick.p == pytest.approx(mid, abs=1e-6)


def test_volume_unavailable_noted():
    """volume/bid/ask/n_trades/open_interest should be empty lists with a note."""
    raw = _make_raw(n_ticks=5)
    snapshot = normalize_market(raw)

    assert snapshot.time_series.volume == []
    assert snapshot.time_series.bid == []
    assert snapshot.time_series.ask == []
    assert snapshot.time_series.n_trades == []
    assert snapshot.time_series.open_interest == []

    # The data_quality or notes should mention the unavailability
    note_text = " ".join(snapshot.data_quality.notes)
    assert "unavailable" in note_text


# ---------------------------------------------------------------------------
# Tests — new metadata fields
# ---------------------------------------------------------------------------

def test_category_passed_through():
    raw = _make_raw(extra_meta={"category": "Politics"})
    snapshot = normalize_market(raw)
    assert snapshot.category == "Politics"


def test_end_date_passed_through():
    raw = _make_raw(extra_meta={"end_date": "2026-11-03"})
    snapshot = normalize_market(raw)
    assert snapshot.end_date == "2026-11-03"


def test_resolution_criteria_from_description():
    raw = _make_raw(extra_meta={"description": "Resolves YES if candidate wins."})
    snapshot = normalize_market(raw)
    assert snapshot.resolution_criteria == "Resolves YES if candidate wins."


def test_enrichment_fields_default_to_none():
    raw = _make_raw()
    snapshot = normalize_market(raw)
    assert snapshot.current_liquidity is None
    assert snapshot.odds_swing_pct is None
    assert snapshot.velocity_score is None
    assert snapshot.deep_analysis_score is None
    assert snapshot.related_market_ids == []
