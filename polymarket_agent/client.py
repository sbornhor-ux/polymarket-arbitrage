"""
Polymarket CLOB REST API client.

Public endpoints used (no auth required):
  - GET /markets/{condition_id}          → market metadata + current prices
  - GET /prices-history?market={token_id}&interval=1m&startTs=...&endTs=...
        → 1-minute probability ticks

Docs: https://docs.polymarket.com/#get-market
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

_BASE = "https://clob.polymarket.com"
_TIMEOUT = 15.0  # seconds


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_price_history(
    token_id: str,
    lookback_hours: int = 72,
    end_utc: datetime | None = None,
) -> list[dict]:
    """
    Fetch price history for a YES token directly from the CLOB API.

    Args:
        token_id: The Polymarket YES token ID (decimal string).
        lookback_hours: How many hours back from end_utc to fetch (default 72).
        end_utc: End of the fetch window (default: now). Pass first_seen to
                 get the 72 hours leading up to when the swing was detected.

    Returns:
        List of {"t": unix_ts, "p": float} dicts ordered oldest-first.
        Returns empty list on any error.
    """
    end = end_utc if end_utc is not None else _utc_now()
    end_ts = int(end.timestamp())
    start_ts = int((end - timedelta(hours=lookback_hours)).timestamp())
    try:
        with httpx.Client(base_url=_BASE, timeout=_TIMEOUT) as client:
            resp = client.get(
                "/prices-history",
                params={
                    "market": token_id,
                    "interval": "1m",
                    "startTs": start_ts,
                    "endTs": end_ts,
                    "fidelity": 15,
                },
            )
            resp.raise_for_status()
            return resp.json().get("history", [])
    except Exception:
        return []


def extract_yes_token_id(clob_token_ids_json: str) -> str | None:
    """
    Parse the clob_token_ids JSON string and return the YES token ID.

    Args:
        clob_token_ids_json: JSON string like '["0xabc...", "0xdef..."]'
                             where index 0 is YES and index 1 is NO.

    Returns:
        YES token ID string, or None if unparseable.
    """
    try:
        ids = json.loads(clob_token_ids_json)
        if ids and isinstance(ids, list):
            return str(ids[0])
    except Exception:
        pass
    return None


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
