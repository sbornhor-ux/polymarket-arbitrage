"""
polymarket_agent
================
Public interface for the Polymarket Data Interpreter.

Usage (from orchestrator):
    from polymarket_agent import fetch_and_interpret
    snapshot = fetch_and_interpret(market_id="...", lookback_days=30)
    print(snapshot.model_dump_json(indent=2))
"""

from polymarket_agent.client import fetch_market_raw
from polymarket_agent.normalizer import normalize_market
from polymarket_agent.resampler import resample_to_5m
from polymarket_agent.swing_detector import detect_swings
from polymarket_agent.models import MarketSnapshot


def fetch_and_interpret(market_id: str, lookback_days: int = 30) -> MarketSnapshot:
    """
    Full pipeline: fetch → normalize → resample → detect swings.

    Args:
        market_id: Polymarket CLOB market condition ID (slug or hex ID).
        lookback_days: How many days of history to request for z-score baseline.

    Returns:
        MarketSnapshot with detected SwingEvents and linked series slots ready
        for the finance_agent to populate.
    """
    raw = fetch_market_raw(market_id, lookback_days=lookback_days)
    snapshot = normalize_market(raw)
    snapshot = resample_to_5m(snapshot)
    snapshot = detect_swings(snapshot)
    return snapshot


__all__ = ["fetch_and_interpret", "MarketSnapshot"]
