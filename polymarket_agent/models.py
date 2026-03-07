"""
Pydantic models for the Polymarket Data Interpreter.

These are the canonical JSON contracts consumed by downstream agents.
All timestamps are UTC ISO-8601 strings for JSON serialisability.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------

class ProbTick(BaseModel):
    """A single probability observation."""
    ts_utc: str = Field(..., description="ISO-8601 UTC timestamp")
    p: float = Field(..., ge=0.0, le=1.0, description="Implied probability [0, 1]")


class PolymarketTimeSeries(BaseModel):
    """
    Parallel flat-list time series for a Polymarket market.

    volume, bid, ask, n_trades, open_interest are always empty lists when
    fetched via the CLOB /prices-history endpoint (which only returns {t, p}).
    They may be populated if a richer data source (e.g. Gamma API) is used.
    """
    dates: list[str] = Field(default_factory=list, description="ISO-8601 UTC timestamps")
    mid_price: list[float] = Field(default_factory=list, description="Implied probability [0, 1]")
    volume: list[float] = Field(default_factory=list, description="Daily USD volume (unavailable from CLOB /prices-history)")
    bid: list[float] = Field(default_factory=list, description="Bid price (unavailable from CLOB /prices-history)")
    ask: list[float] = Field(default_factory=list, description="Ask price (unavailable from CLOB /prices-history)")
    n_trades: list[int] = Field(default_factory=list, description="Trade count (unavailable from CLOB /prices-history)")
    open_interest: list[float] = Field(default_factory=list, description="Open interest in USD (unavailable from CLOB /prices-history)")


class DetectionRule(BaseModel):
    """Parameters used for swing detection — makes results reproducible."""
    method: Literal["rolling_z"] = "rolling_z"
    trailing_lookback: str = Field("30D", description="Pandas offset alias, e.g. '30D'")
    threshold: float = Field(2.0, description="Z-score threshold to declare a swing")


# ---------------------------------------------------------------------------
# Swing event (output of swing_detector)
# ---------------------------------------------------------------------------

class SwingEvent(BaseModel):
    """
    A detected probability swing over a fixed time window.
    Keyed by (market_id, swing_id) for joining with finance stats.
    """
    swing_id: str = Field(..., description="Unique ID within this market, e.g. 'S1'")
    window: str = Field(..., description="Window label, e.g. '6H', '1D'")
    window_start_utc: str
    window_end_utc: str
    p_pre: float = Field(..., ge=0.0, le=1.0, description="Prob at window open")
    p_post: float = Field(..., ge=0.0, le=1.0, description="Prob at window close")
    delta_p: float = Field(..., description="p_post - p_pre (signed)")
    abs_delta_p: float = Field(..., description="|delta_p|")
    z_score_vs_trailing: float = Field(
        ..., description="Z-score of abs_delta_p vs trailing distribution"
    )
    detection_rule: DetectionRule = Field(default_factory=DetectionRule)


# ---------------------------------------------------------------------------
# Top-level market snapshot
# ---------------------------------------------------------------------------

class DataQuality(BaseModel):
    missing_pct: float = 0.0
    notes: list[str] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    """
    Normalised, resampled Polymarket market with detected swings.
    This is the output contract of polymarket_agent.fetch_and_interpret().
    """
    market_id: str
    market_question: str
    market_slug: str = ""
    as_of_utc: str = Field(..., description="Snapshot creation timestamp (UTC)")

    # Market metadata (Kevin required)
    category: str = ""
    end_date: str | None = None
    resolution_criteria: str | None = None

    # 5-minute resampled series (output of resampler) — kept for backward compat
    probability_series: list[ProbTick] = Field(default_factory=list)

    # Parallel flat-list time series (Kevin required format).
    # dates/mid_price mirror probability_series.
    # volume/bid/ask/n_trades/open_interest are empty when using the CLOB
    # /prices-history endpoint — see data_quality.notes for explanation.
    time_series: PolymarketTimeSeries = Field(default_factory=PolymarketTimeSeries)

    # Detected swing events (output of swing_detector)
    swings: list[SwingEvent] = Field(default_factory=list)

    # Nice-to-have enrichment (populated from pre-scan data where available)
    current_liquidity: float | None = None
    odds_swing_pct: float | None = None
    velocity_score: float | None = None
    deep_analysis_score: float | None = None
    related_market_ids: list[str] = Field(default_factory=list)

    # Financial relevance screening
    financial_relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence [0, 1] that this event is relevant to financial markets",
    )
    financial_relevance_rationale: str | None = Field(
        default=None,
        description="Human-readable explanation of the relevance score",
    )

    # Provenance / audit
    notes: dict = Field(
        default_factory=lambda: {
            "polymarket_source": "clob.polymarket.com",
            "resample_cadence": "5T",
            "assumptions": ["p = last observed probability in each 5m bucket"],
            "missing_data": [],
        }
    )
    data_quality: DataQuality = Field(default_factory=DataQuality)
