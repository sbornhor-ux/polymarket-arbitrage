"""
Normalizes raw Polymarket API responses into MarketSnapshot objects.

Keeps raw tick data intact (resampler handles cadence reduction).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from polymarket_agent.models import MarketSnapshot, ProbTick, PolymarketTimeSeries, DataQuality
from polymarket_agent.relevance_screener import score_financial_relevance


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unix_to_iso(ts: int | float) -> str:
    """Convert a Unix timestamp (seconds) to UTC ISO-8601 string."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def normalize_market(raw: dict[str, Any]) -> MarketSnapshot:
    """
    Convert raw fetch_market_raw() output into a MarketSnapshot.

    The snapshot at this stage contains the full 1-minute tick series.
    resampler.py will downsample to 5-minute cadence.

    Args:
        raw: Dict with "metadata" and "history" keys (from client.py).

    Returns:
        MarketSnapshot with probability_series and time_series populated, no swings yet.
    """
    metadata: dict = raw["metadata"]
    ticks: list[dict] = raw["history"]

    # --- Probability series ---
    prob_series: list[ProbTick] = []
    missing = 0
    for tick in ticks:
        try:
            prob_series.append(
                ProbTick(
                    ts_utc=_unix_to_iso(tick["t"]),
                    p=float(tick["p"]),
                )
            )
        except (KeyError, ValueError):
            missing += 1

    missing_pct = missing / len(ticks) if ticks else 0.0
    dq_notes: list[str] = []
    if missing_pct > 0.05:
        dq_notes.append(f"Warning: {missing_pct:.1%} of ticks dropped during normalization")

    # --- Parallel flat-list time_series (Kevin required format) ---
    # dates/mid_price mirror probability_series.
    # volume/bid/ask/n_trades/open_interest unavailable from CLOB /prices-history.
    time_series = PolymarketTimeSeries(
        dates=[t.ts_utc for t in prob_series],
        mid_price=[t.p for t in prob_series],
    )
    dq_notes.append(
        "time_series.volume/bid/ask/n_trades/open_interest unavailable from CLOB /prices-history endpoint"
    )

    # --- New metadata fields (Kevin required) ---
    category = metadata.get("category", "")
    end_date = metadata.get("end_date_iso") or metadata.get("end_date") or None
    resolution_criteria = metadata.get("description") or metadata.get("resolution_criteria") or None

    # Nice-to-have enrichment (present in pre-scan data; None for live API fetches)
    current_liquidity = metadata.get("current_liquidity")
    odds_swing_pct = metadata.get("odds_swing_pct")
    velocity_score = metadata.get("velocity_score")
    deep_analysis_score = metadata.get("deep_analysis_score")
    related_market_ids = metadata.get("related_market_ids") or []

    question = metadata.get("question", "")
    rel_score, rel_rationale = score_financial_relevance(question, category)

    return MarketSnapshot(
        market_id=metadata.get("condition_id", metadata.get("id", "unknown")),
        market_question=question,
        market_slug=metadata.get("market_slug", ""),
        as_of_utc=_utc_iso(),
        category=category,
        end_date=end_date,
        resolution_criteria=resolution_criteria,
        probability_series=prob_series,
        time_series=time_series,
        swings=[],
        current_liquidity=current_liquidity,
        odds_swing_pct=odds_swing_pct,
        velocity_score=velocity_score,
        deep_analysis_score=deep_analysis_score,
        related_market_ids=related_market_ids,
        financial_relevance_score=rel_score,
        financial_relevance_rationale=rel_rationale,
        notes={
            "polymarket_source": "clob.polymarket.com",
            "resample_cadence": "5T",  # will be set by resampler
            "assumptions": ["p = last observed probability in each 5m bucket"],
            "missing_data": dq_notes,
        },
        data_quality=DataQuality(missing_pct=missing_pct, notes=dq_notes),
    )
