"""
Pydantic models for the Financial Market Workflow.

FinanceSeriesWindowStats is the canonical output contract consumed by
Kevin's Trend Analyst and Michael's Synthesiser.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM Instrument Selection models
# ---------------------------------------------------------------------------

class InstrumentPrediction(BaseModel):
    """LLM-predicted instrument for a Polymarket market."""
    ticker: str = Field(..., description="Exchange ticker, e.g. 'AAPL', '^GSPC', 'SPY'")
    company_name: str = Field("", description="Full company/index name")
    instrument_type: str = Field("stock", description="'stock', 'etf', or 'index'")
    predicted_direction: str = Field(
        ..., description="'up' or 'down' — asset direction when Polymarket YES rises"
    )
    confidence: int = Field(..., description="LLM confidence 1–100")
    rationale: str = Field("", description="One-sentence explanation")


class InstrumentSelection(BaseModel):
    """LLM-selected instruments for a single Polymarket market."""
    market_id: str
    market_question: str
    instruments: list[InstrumentPrediction] = Field(default_factory=list)
    selection_rationale: str = Field("", description="Overall selection rationale")


# ---------------------------------------------------------------------------
# Default series (yfinance tickers)
# ---------------------------------------------------------------------------

DEFAULT_SERIES: list[str] = [
    "^TNX",      # US 10-Year Treasury yield
    "^GSPC",     # S&P 500
    "^VIX",      # CBOE Volatility Index
    "BTC-USD",   # Bitcoin
    "DX-Y.NYB",  # US Dollar Index (DXY)
    "CL=F",      # WTI Crude Oil (front-month futures)
    "GC=F",      # Gold (front-month futures)
]

# Human-readable labels for display / narrative agents
SERIES_LABELS: dict[str, str] = {
    "^TNX":     "US 10Y Treasury Yield (via IEF)",
    "^GSPC":    "S&P 500 (via SPY)",
    "^VIX":     "VIX (via VIXY)",
    "BTC-USD":  "Bitcoin (USD)",
    "DX-Y.NYB": "US Dollar Index (via UUP)",
    "CL=F":     "WTI Crude Oil (via USO)",
    "GC=F":     "Gold (via GLD)",
}

# Sector labels for sector rotation analysis (Kevin required)
SERIES_SECTORS: dict[str, str] = {
    "^TNX":     "Fixed Income",
    "^GSPC":    "Equities",
    "^VIX":     "Volatility",
    "BTC-USD":  "Crypto",
    "DX-Y.NYB": "FX",
    "CL=F":     "Commodities",
    "GC=F":     "Commodities",
}


# ---------------------------------------------------------------------------
# Time series sub-model (Kevin required)
# ---------------------------------------------------------------------------

class FinanceTimeSeries(BaseModel):
    """
    Parallel flat-list time series for a financial instrument over a swing window.

    bid_ask_spread is always an empty list — yfinance historical data does not
    include bid/ask quotes. intraday_high/low are populated when the fetcher
    returns High/Low columns (requires yfinance download with auto_adjust=True).
    """
    dates: list[str] = Field(default_factory=list, description="ISO-8601 UTC timestamps")
    prices: list[float] = Field(default_factory=list, description="Close prices")
    returns: list[float] = Field(default_factory=list, description="Period-over-period returns")
    volumes: list[float] = Field(default_factory=list, description="Trading volumes")
    intraday_high: list[float] = Field(default_factory=list, description="Intraday high prices")
    intraday_low: list[float] = Field(default_factory=list, description="Intraday low prices")
    bid_ask_spread: list[float] = Field(
        default_factory=list,
        description="Bid-ask spread (unavailable from yfinance historical data)",
    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class DataQuality(BaseModel):
    missing_pct: float = 0.0
    notes: list[str] = Field(default_factory=list)


class FinanceSeriesWindowStats(BaseModel):
    """
    Statistics for a single financial series over a single swing window.

    Keyed upstream by (swing_id, series_id) for joining with Polymarket data.
    """
    # Identity
    swing_id: str = Field(..., description="Matches SwingEvent.swing_id")
    series_id: str = Field(..., description="yfinance ticker, e.g. '^GSPC'")
    series_label: str = Field("", description="Human-readable name")
    polymarket_id: str = Field("", description="Join key back to MarketSnapshot.market_id")

    # Sector / market metadata (Kevin required)
    sector: str = Field("", description="Sector label, e.g. 'Equities'")
    market_cap: float | None = Field(None, description="Market cap in USD from yf.Ticker.info")

    # Window definition (mirrors the SwingEvent window for auditability)
    window: str = Field(..., description="Window label, e.g. '6H', '1D'")
    window_start_utc: str
    window_end_utc: str

    # Core stats
    level_pre: float | None = Field(None, description="Series level at window_start")
    level_post: float | None = Field(None, description="Series level at window_end")
    move_pre_to_post: float | None = Field(
        None, description="Absolute change: level_post - level_pre"
    )
    move_pct: float | None = Field(
        None, description="Percentage change: (post/pre - 1) × 100"
    )
    units: str = Field("native", description="Units of the series (e.g. 'pct_pts', 'index')")

    # Full time series over the swing window (Kevin required)
    time_series: FinanceTimeSeries = Field(default_factory=FinanceTimeSeries)

    # Z-score vs trailing baseline
    z_score_vs_trailing: float | None = Field(
        None,
        description="Z-score of |move_pre_to_post| vs trailing distribution",
    )
    trailing_days: int = Field(60, description="Days used for z-score baseline")

    # Lead-lag correlation with Polymarket swing
    corr_with_prob_change: float | None = Field(
        None,
        description="Pearson correlation between series returns and Δp in window",
    )
    best_lag_hours: float | None = Field(
        None,
        description="Lag (hours) at which cross-correlation is maximised; "
                    "negative = series leads Polymarket",
    )

    # Benchmark comparison (Kevin required)
    benchmark_ticker: str = Field("^GSPC", description="Benchmark ticker for excess return computation")
    benchmark_returns: list[float] = Field(
        default_factory=list,
        description="Benchmark period returns over the same window (market-adjusted returns isolate idiosyncratic component)",
    )

    data_quality: DataQuality = Field(default_factory=DataQuality)

    # LLM instrument selection metadata
    llm_confidence: int | None = Field(
        None, description="LLM confidence score 1–100 for this ticker's relevance"
    )
    llm_predicted_direction: str | None = Field(
        None, description="'with' or 'against' — alignment when Polymarket YES rises"
    )
    llm_company_name: str | None = Field(None, description="LLM-provided company/index name")
    llm_rationale: str | None = Field(None, description="LLM explanation for why this instrument was selected")
