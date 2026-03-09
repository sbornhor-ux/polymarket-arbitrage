"""
polymarket_agent.loader
=======================
Two loaders — both return the {metadata, history} dict format that
normalizer.normalize_market() expects.

load_from_file(path)  — reads polymarket_data.json.  History is SYNTHETIC
                        (seeded random walk anchored to current_price) because
                        the JSON only contains a point-in-time snapshot.

load_from_csv(path)   — reads polymarket_prod_*.csv files that contain real
                        hourly price columns (price_t_minus_24 … price_t_plus_24).
                        History is REAL (no random walk).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_from_file(
    path: str,
    only_passed_filter: bool = False,
    n_days: int = 30,
) -> list[dict[str, Any]]:
    """
    Load polymarket_data.json and convert each market entry into a raw dict
    compatible with normalizer.normalize_market().

    Args:
        path: Absolute or relative path to polymarket_data.json.
        only_passed_filter: If True, return only markets where
                            passed_initial_filter == True (reduces set from
                            ~1123 to ~264).
        n_days: Length of synthetic history to generate (days).

    Returns:
        List of raw dicts, each with keys:
            "metadata"  → condition_id, question, market_slug, tokens
            "history"   → list of {t, p} ticks (1-minute cadence, synthetic)
            "_source"   → "file"  (signals downstream that history is synthetic)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    markets: list[dict] = data.get("markets", [])

    if only_passed_filter:
        markets = [m for m in markets if m.get("passed_initial_filter", False)]

    # Keep all markets that have a question; use fallback price for resolved ones.
    markets = [m for m in markets if m.get("question")]

    return [_to_raw(m, n_days=n_days) for m in markets]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _valid_price(price: Any) -> bool:
    """Return True if price is a usable mid-range probability value."""
    try:
        p = float(price)
        return 0.0 < p < 1.0
    except (TypeError, ValueError):
        return False


def _resolve_price(market: dict) -> tuple[float, bool]:
    """
    Return (price_to_use, is_synthetic_anchor).

    If current_price is a valid probability we use it directly.
    For resolved markets (price == 0 or == 1) we fall back to 0.5 and
    flag the history as fully synthetic.
    """
    raw = market.get("current_price", None)
    if _valid_price(raw):
        return float(raw), False
    # Resolved or missing — use neutral 0.5 as walk anchor
    return 0.5, True


def _to_raw(market: dict, n_days: int = 30) -> dict[str, Any]:
    """Convert a single polymarket_data.json market entry into a raw dict."""
    market_id = str(market.get("market_id", "unknown"))
    question = market.get("question", "")
    current_price, is_resolved = _resolve_price(market)

    # Build synthetic {t, p} history
    history = _build_synthetic_history(
        current_price=current_price,
        market_id=market_id,
        n_days=n_days,
    )

    # Build a minimal metadata dict matching what client.fetch_market_raw() returns
    metadata: dict[str, Any] = {
        "condition_id": market_id,
        "question": question,
        "market_slug": _slugify(question),
        # Provide a minimal tokens list so normalizer.normalize_market() works
        # (it doesn't use tokens — only client.py uses them to resolve token_id)
        "tokens": [
            {"token_id": f"{market_id}_yes", "outcome": "Yes", "price": current_price}
        ],
        # Pass through useful scoring fields for downstream consumers
        "composite_score": market.get("composite_score"),
        "odds_swing_pct": market.get("odds_swing_pct"),
        "flags_triggered": market.get("flags_triggered", []),
        "category": market.get("category", ""),
        # Kevin required fields
        "end_date": market.get("end_date"),
        "description": market.get("description", ""),   # used as resolution_criteria
        "current_liquidity": market.get("current_liquidity"),
        "velocity_score": market.get("velocity_score"),
        "deep_analysis_score": market.get("deep_analysis_score"),
    }

    return {
        "metadata": metadata,
        "history": history,
        "_source": "file",             # marks this as file-loaded, not live API
        "_synthetic_history": True,
        "_resolved_market": is_resolved,  # True when current_price was 0/1
    }


def _build_synthetic_history(
    current_price: float,
    market_id: str,
    n_days: int = 30,
) -> list[dict[str, float | int]]:
    """
    Generate a synthetic 1-minute probability tick series using a seeded random walk.

    The walk ends at current_price (anchored), so the most recent tick always
    matches the real snapshot value.

    Seed is derived from market_id for full reproducibility:
    same market_id → same history every time.

    Args:
        current_price: The real current probability (from JSON).
        market_id: Used to seed the RNG (deterministic).
        n_days: History length in days.

    Returns:
        List of {"t": unix_ts_int, "p": float} dicts, oldest first.
    """
    seed = abs(hash(market_id)) % (2**31)
    rng = np.random.default_rng(seed)

    n_ticks = n_days * 24 * 60  # 1-minute cadence

    # Step noise: std ≈ 0.002 per tick (realistic Polymarket 1m noise)
    step_std = 0.002
    steps = rng.normal(0.0, step_std, n_ticks)

    # Build raw walk, then drift-correct so last value == current_price
    raw_walk = np.cumsum(steps)
    drift = current_price - (0.5 + raw_walk[-1])  # target: start near 0.5
    corrected = 0.5 + raw_walk + drift * np.linspace(0, 1, n_ticks)

    # Clip to valid probability range
    probs = np.clip(corrected, 0.001, 0.999)

    # Build timestamps: n_days ago → now (1-minute steps)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = now_ts - n_days * 24 * 3600
    timestamps = [start_ts + i * 60 for i in range(n_ticks)]

    return [
        {"t": int(ts), "p": float(round(p, 4))}
        for ts, p in zip(timestamps, probs)
    ]


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert a market question to a URL-safe slug."""
    import re
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug[:max_len]


# ---------------------------------------------------------------------------
# CSV loader (real hourly price data)
# ---------------------------------------------------------------------------

# Ordered list of hourly price column offsets relative to current_as_of (t=0).
# Each entry is (column_name, hour_offset) where hour_offset is negative for past.
_PRICE_COLS: list[tuple[str, int]] = (
    [(f"price_t_minus_{i:02d}", -i) for i in range(24, 0, -1)]
    + [(f"price_t_plus_{i:02d}", +i) for i in range(1, 25)]
)


def load_from_csv(path: str) -> list[dict[str, Any]]:
    """
    Load a polymarket_prod_*.csv file and convert each row into a raw dict
    compatible with normalizer.normalize_market().

    Unlike load_from_file(), the price history here is REAL — each row
    contains up to 49 hourly price observations (price_t_minus_24 …
    price_t_plus_24) anchored to current_as_of.  NaN entries are skipped.

    Args:
        path: Path to the CSV file (e.g. 'AI Financial Folder/polymarket_prod_2026-03-01.csv').

    Returns:
        List of raw dicts with keys "metadata", "history", "_source" = "csv".
        Rows with no valid price observations are excluded.
    """
    import pandas as pd

    df = pd.read_csv(path)
    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw = _csv_row_to_raw(row)
        if raw is not None:
            results.append(raw)
    return results


def _csv_row_to_raw(row: Any) -> dict[str, Any] | None:
    """Convert a single CSV row to a normalize_market()-compatible raw dict."""
    import pandas as pd

    # Reference timestamp: current_as_of (UTC ISO string)
    current_as_of = row.get("current_as_of")
    if pd.isna(current_as_of):
        ref_ts = int(datetime.now(timezone.utc).timestamp())
    else:
        ref_ts = int(datetime.fromisoformat(str(current_as_of)).timestamp())

    # Build history from real hourly price columns
    history: list[dict[str, float | int]] = []

    # Add the snapshot price at t=0
    current_price_raw = row.get("current_price")
    if not pd.isna(current_price_raw):
        p0 = float(current_price_raw)
        if 0.0 <= p0 <= 1.0:
            history.append({"t": ref_ts, "p": round(p0, 4)})

    for col, hour_offset in _PRICE_COLS:
        val = row.get(col)
        if pd.isna(val):
            continue
        p = float(val)
        if not (0.0 <= p <= 1.0):
            continue
        ts = ref_ts + hour_offset * 3600
        history.append({"t": ts, "p": round(p, 4)})

    # Sort chronologically and deduplicate by timestamp
    history.sort(key=lambda x: x["t"])
    seen: set[int] = set()
    history = [h for h in history if not (h["t"] in seen or seen.add(h["t"]))]  # type: ignore[func-returns-value]

    if not history:
        return None

    market_id_raw = row.get("market_id")
    market_id = str(int(market_id_raw)) if not pd.isna(market_id_raw) else "unknown"
    question = str(row.get("question", ""))
    category = str(row.get("category", ""))

    # Anchor price for tokens list (use current_price if valid, else midpoint)
    anchor = float(current_price_raw) if not pd.isna(current_price_raw) else 0.5

    liquidity_raw = row.get("liquidity")
    volume_raw = row.get("volume")

    clob_token_ids_raw = row.get("clob_token_ids", "")
    clob_token_ids = "" if pd.isna(clob_token_ids_raw) else str(clob_token_ids_raw)

    first_seen_raw = row.get("first_seen", "")
    first_seen = "" if pd.isna(first_seen_raw) else str(first_seen_raw)

    metadata: dict[str, Any] = {
        "condition_id": market_id,
        "question": question,
        "market_slug": _slugify(question),
        "tokens": [{"token_id": f"{market_id}_yes", "outcome": "Yes", "price": anchor}],
        "category": category,
        "end_date": None if pd.isna(row.get("end_date", float("nan"))) else row["end_date"],
        "current_liquidity": None if pd.isna(liquidity_raw) else float(liquidity_raw),
        "odds_swing_pct": None if pd.isna(row.get("one_month_price_change", float("nan"))) else float(row["one_month_price_change"]),
        "clob_token_ids": clob_token_ids,
        "first_seen": first_seen,
    }

    return {
        "metadata": metadata,
        "history": history,
        "_source": "csv",
        "_synthetic_history": False,
    }
