"""
Polymarket CLOB REST API client.

Public endpoints used (no auth required):
  - GET /markets/{condition_id}          → market metadata + current prices
  - GET /prices-history?market={token_id}&interval=1m&startTs=...&endTs=...
        → 1-minute probability ticks

Docs: https://docs.polymarket.com/#get-market
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

_BASE = "https://clob.polymarket.com"
_TIMEOUT = 15.0  # seconds


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_market_raw(market_id: str, lookback_days: int = 30) -> dict[str, Any]:
    """
    Fetch raw market metadata + price history from the Polymarket CLOB API.

    Args:
        market_id: Condition ID (hex string) or market slug.
        lookback_days: Number of days of price history to retrieve.

    Returns:
        Dict with keys:
            "metadata"  → raw /markets/{id} response
            "history"   → list of {t: unix_ts, p: float} ticks (1m cadence)

    Raises:
        httpx.HTTPStatusError: on non-2xx responses.
        ValueError: if market not found or token ID unavailable.
    """
    with httpx.Client(base_url=_BASE, timeout=_TIMEOUT) as client:
        # 1. Fetch market metadata
        meta_resp = client.get(f"/markets/{market_id}")
        meta_resp.raise_for_status()
        metadata: dict = meta_resp.json()

        # 2. Resolve the YES token ID (first outcome token)
        tokens = metadata.get("tokens", [])
        if not tokens:
            raise ValueError(
                f"No tokens found for market '{market_id}'. "
                "Check that the condition ID is correct."
            )
        # Polymarket tokens: [{token_id, outcome, price}, ...]
        # Pick the "Yes" token; fall back to first token if no "Yes" label.
        yes_token = next(
            (t for t in tokens if t.get("outcome", "").lower() == "yes"),
            tokens[0],
        )
        token_id: str = yes_token["token_id"]

        # 3. Fetch price history
        end_ts = int(_utc_now().timestamp())
        start_ts = int((_utc_now() - timedelta(days=lookback_days)).timestamp())

        history_resp = client.get(
            "/prices-history",
            params={
                "market": token_id,
                "interval": "1m",
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": 1,
            },
        )
        history_resp.raise_for_status()
        history_data = history_resp.json()

        # The CLOB API returns {"history": [{t: ..., p: ...}, ...]}
        ticks: list[dict] = history_data.get("history", [])

    return {
        "metadata": metadata,
        "history": ticks,
    }
