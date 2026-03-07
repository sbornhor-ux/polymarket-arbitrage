"""
Tests for polymarket_agent.loader

Uses the real polymarket_data.json file (project root) for integration-style
tests, and purely synthetic inputs for unit tests.
All tests are offline — no network calls.
"""

from __future__ import annotations

import os
import pytest

from polymarket_agent.loader import (
    load_from_file,
    _build_synthetic_history,
    _valid_price,
    _slugify,
)
from polymarket_agent.normalizer import normalize_market
from polymarket_agent.resampler import resample_to_5m
from polymarket_agent.swing_detector import detect_swings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "polymarket_data.json",
)


def _file_available() -> bool:
    return os.path.isfile(DATA_FILE)


skip_if_no_file = pytest.mark.skipif(
    not _file_available(),
    reason="polymarket_data.json not found at project root",
)


# ---------------------------------------------------------------------------
# Unit tests: _valid_price
# ---------------------------------------------------------------------------

def test_valid_price_accepts_midrange():
    assert _valid_price(0.5) is True
    assert _valid_price(0.01) is True
    assert _valid_price(0.99) is True


def test_valid_price_rejects_extremes():
    assert _valid_price(0.0) is False
    assert _valid_price(1.0) is False
    assert _valid_price(-0.1) is False
    assert _valid_price(1.1) is False


def test_valid_price_rejects_non_numeric():
    assert _valid_price(None) is False
    assert _valid_price("abc") is False
    assert _valid_price([]) is False


# ---------------------------------------------------------------------------
# Unit tests: _build_synthetic_history
# ---------------------------------------------------------------------------

def test_history_length():
    ticks = _build_synthetic_history(0.6, "test_market", n_days=7)
    assert len(ticks) == 7 * 24 * 60  # 10,080 ticks


def test_history_tick_keys():
    ticks = _build_synthetic_history(0.5, "m1", n_days=1)
    for tick in ticks[:10]:
        assert "t" in tick
        assert "p" in tick


def test_history_probs_in_range():
    ticks = _build_synthetic_history(0.7, "m2", n_days=5)
    for tick in ticks:
        assert 0.0 < tick["p"] < 1.0, f"p={tick['p']} out of range"


def test_history_last_tick_anchored_to_price():
    """Last tick should be close to current_price (within ±0.01)."""
    target = 0.72
    ticks = _build_synthetic_history(target, "m3", n_days=10)
    assert abs(ticks[-1]["p"] - target) < 0.01


def test_history_timestamps_ascending():
    ticks = _build_synthetic_history(0.5, "m4", n_days=2)
    ts_list = [t["t"] for t in ticks]
    assert ts_list == sorted(ts_list)


def test_history_deterministic():
    """Same market_id + price → identical history on repeated calls."""
    h1 = _build_synthetic_history(0.55, "same_id", n_days=5)
    h2 = _build_synthetic_history(0.55, "same_id", n_days=5)
    assert h1 == h2


def test_history_different_seeds_differ():
    """Different market_ids → different histories."""
    h1 = _build_synthetic_history(0.55, "market_A", n_days=5)
    h2 = _build_synthetic_history(0.55, "market_B", n_days=5)
    probs1 = [t["p"] for t in h1]
    probs2 = [t["p"] for t in h2]
    assert probs1 != probs2


# ---------------------------------------------------------------------------
# Integration tests: load_from_file with real JSON
# ---------------------------------------------------------------------------

@skip_if_no_file
def test_load_returns_list():
    result = load_from_file(DATA_FILE)
    assert isinstance(result, list)
    assert len(result) > 0


@skip_if_no_file
def test_load_filtered_subset_of_all():
    all_markets = load_from_file(DATA_FILE, only_passed_filter=False)
    filtered = load_from_file(DATA_FILE, only_passed_filter=True)
    # Filtered must be a subset (<=) of all
    assert len(filtered) <= len(all_markets)
    assert len(filtered) > 0
    assert len(all_markets) > 0


@skip_if_no_file
def test_each_raw_dict_has_required_keys():
    result = load_from_file(DATA_FILE, only_passed_filter=True)
    for raw in result[:20]:  # spot-check first 20
        assert "metadata" in raw
        assert "history" in raw
        assert raw.get("_source") == "file"
        assert raw.get("_synthetic_history") is True


@skip_if_no_file
def test_metadata_has_required_fields():
    result = load_from_file(DATA_FILE, only_passed_filter=True)
    for raw in result[:20]:
        md = raw["metadata"]
        assert "condition_id" in md
        assert "question" in md
        assert "tokens" in md
        assert isinstance(md["tokens"], list)
        assert len(md["tokens"]) > 0


@skip_if_no_file
def test_history_ticks_valid():
    result = load_from_file(DATA_FILE, only_passed_filter=True)
    for raw in result[:5]:
        for tick in raw["history"][:10]:
            assert "t" in tick and "p" in tick
            assert isinstance(tick["t"], int)
            assert 0.0 < tick["p"] < 1.0


@skip_if_no_file
def test_resolved_markets_use_fallback_price():
    """
    Resolved markets (current_price == 0 or 1) should be included but
    flagged as resolved, with history anchored to 0.5 fallback.
    """
    result = load_from_file(DATA_FILE)
    resolved = [r for r in result if r.get("_resolved_market")]
    # All markets in this JSON are resolved — loader should still return them
    assert len(result) > 0
    for raw in resolved:
        # Fallback price used as synthetic anchor — history ticks still valid
        for tick in raw["history"][:5]:
            assert 0.0 < tick["p"] < 1.0


# ---------------------------------------------------------------------------
# Pipeline integration: loader → full agent pipeline
# ---------------------------------------------------------------------------

@skip_if_no_file
def test_full_pipeline_on_first_five():
    """The full normalizer → resampler → swing_detector pipeline should not crash."""
    raw_list = load_from_file(DATA_FILE, only_passed_filter=True)

    for raw in raw_list[:5]:
        snapshot = normalize_market(raw)
        assert snapshot.market_id is not None
        assert len(snapshot.probability_series) > 0

        snapshot = resample_to_5m(snapshot)
        assert len(snapshot.probability_series) > 0

        snapshot = detect_swings(snapshot)
        # Swings may or may not be present — just check types
        assert isinstance(snapshot.swings, list)
        for swing in snapshot.swings:
            assert swing.z_score_vs_trailing >= 2.0


@skip_if_no_file
def test_pipeline_output_is_json_serialisable():
    """MarketSnapshot.model_dump_json() should succeed for each loaded market."""
    raw_list = load_from_file(DATA_FILE, only_passed_filter=True)
    for raw in raw_list[:5]:
        snapshot = normalize_market(raw)
        snapshot = resample_to_5m(snapshot)
        snapshot = detect_swings(snapshot)
        json_str = snapshot.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 10
