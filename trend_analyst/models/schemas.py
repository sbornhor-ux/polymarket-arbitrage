"""
Data Models (v0.4.0)
---------------------
Pydantic-style dataclasses for type safety and serialization.

Defines the contract between:
  - Upstream workflows     → Trend Analyst     (PolymarketRecord, FinancialMarketRecord)
  - Trend Analyst          → Synthesizer       (PairAnalysis, TrendAnalystOutput)
  - Historical Analyst     → Synthesizer       (HistoricalPairResult, HistoricalAnalystOutput)
  - Realtime Analyst       → Synthesizer       (RealtimePairResult, RealtimeAnalystOutput)

ACTIVE TEST RESULT CLASSES (produced by the current 6-test pipeline):
  CorrelationResult, GrangerResult, LeadLagResult,
  EventStudyResult, SpikeEventStudyResult, VolatilitySpilloverResult,
  DivergenceSignalResult

NEW RESULT CLASSES (v0.4.0 — dual-mode architecture):
  CorrelationBreakdownResult — pre/post spike correlation comparison (realtime mode)
  HistoricalPairResult       — timing finding per pair (historical mode)
  HistoricalAnalystOutput    — top-level output for historical_analyst.py
  RealtimePairResult         — info gap finding per pair (realtime mode)
  RealtimeAnalystOutput      — top-level output for realtime_analyst.py

DEPRECATED RESULT CLASSES (not produced by current code; kept for JSON backward compat):
  RollingCorrelationResult, DTWResult, VolumeAnalysisResult, CointegrationResult
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# INPUT MODELS (from upstream workflows)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolymarketRecord:
    """
    Single market from the Polymarket scan JSON or production CSV.

    Core fields are populated from both sources.  The deep-analysis score fields
    (velocity, liquidity_shift, etc.) come from the scan JSON only.  The hourly
    time series fields (poly_dates, poly_prices) come from the production CSV.
    """
    market_id: str
    question: str
    description: str
    category: str
    end_date: Optional[str]
    current_price: float        # Polymarket probability [0, 1]
    current_volume: float       # 24-hour volume in USD
    current_liquidity: float    # total liquidity in USD
    odds_swing_pct: float       # range of hourly prices * 100 (percentage points)
    volume_surge_pct: float     # volume surge vs baseline
    composite_score: float      # upstream composite signal score
    passed_initial_filter: bool

    # ── Deep analysis fields (from scan JSON; default False/0.0 when absent) ──
    velocity_flag: bool = False
    velocity_score: float = 0.0
    velocity_detail: str = ""
    liquidity_shift_flag: bool = False
    liquidity_shift_score: float = 0.0
    liquidity_shift_detail: str = ""
    time_decay_urgency_flag: bool = False
    time_decay_score: float = 0.0
    time_decay_detail: str = ""
    spread_flag: bool = False
    spread_score: float = 0.0
    spread_detail: str = ""
    volume_weighted_flag: bool = False
    volume_weighted_score: float = 0.0
    volume_weighted_detail: str = ""
    deep_analysis_score: float = 0.0
    flags_triggered: list = field(default_factory=list)

    # ── Hourly time series (populated from CSV; empty when only scan JSON available) ──
    # poly_dates: ISO-8601 UTC strings, one per hour
    # poly_prices: Polymarket probabilities [0, 1] aligned to poly_dates
    # poly_volumes: hourly Polymarket volume (often unavailable; left empty)
    # current_as_of: ISO-8601 UTC string for t=0 (the snapshot anchor point, index 24)
    poly_dates: list[str] = field(default_factory=list)
    poly_prices: list[float] = field(default_factory=list)
    poly_volumes: list[float] = field(default_factory=list)
    current_as_of: str = ""     # t=0 timestamp; poly_prices[24] corresponds to this point


@dataclass
class FinancialMarketRecord:
    """
    Financial market data for a recommended ticker.

    Produced by tools/financial_adapter.py from the Financial Market Workflow JSON.
    All time series fields are index-aligned (dates[i] ↔ prices[i] ↔ volumes[i]).
    returns is length len(prices)-1: returns[i] = log(prices[i+1] / prices[i]).
    """
    polymarket_id: str          # links this record to a PolymarketRecord
    ticker: str                 # e.g. "SPY", "CL=F", "BTC-USD"
    ticker_name: str            # human-readable name, e.g. "S&P 500 ETF"
    recommendation_reasoning: str  # why this ticker was paired with the market

    # Time series — all lists, aligned by index
    dates: list[str] = field(default_factory=list)     # ISO-8601 UTC timestamps
    prices: list[float] = field(default_factory=list)  # closing prices
    returns: list[float] = field(default_factory=list) # log-returns (length N-1)
    volumes: list[float] = field(default_factory=list) # trading volume


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVE TEST RESULT CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CorrelationResult:
    """
    Pearson and Spearman correlation between Polymarket absolute probability
    changes (dP) and financial log-returns over the aligned window.
    """
    pearson_r: float        # Pearson correlation coefficient [-1, 1]
    pearson_p: float        # two-tailed p-value
    spearman_r: float       # Spearman rank correlation [-1, 1]
    spearman_p: float       # two-tailed p-value
    n_observations: int     # number of hourly intervals used
    interpretation: str     # human-readable summary


@dataclass
class GrangerResult:
    """
    Bidirectional Granger causality F-test results.
    Tests whether past Polymarket dP predicts future financial returns, and
    vice versa.  The best lag is the lag that minimises the p-value.
    """
    poly_causes_fin: bool       # True if poly dP Granger-causes fin returns (p < alpha)
    fin_causes_poly: bool       # True if fin returns Granger-cause poly dP (p < alpha)
    poly_to_fin_best_lag: int   # lag (hours) with lowest p for poly→fin direction
    poly_to_fin_p_value: float
    fin_to_poly_best_lag: int
    fin_to_poly_p_value: float
    interpretation: str


@dataclass
class LeadLagResult:
    """
    Lead-lag cross-correlation function (CCF) over lags -10h to +10h.

    Convention: positive peak_lag means Polymarket LEADS financial market by
    that many hours — the trading alpha signal.  Negative peak_lag means the
    financial market already reacted before Polymarket moved.
    """
    peak_lag: int               # lag (hours) with strongest absolute CCF value
    peak_correlation: float     # correlation at peak_lag
    peak_p_value: float         # approximate p-value for peak correlation
    ccf_values: dict            # {lag_int: correlation_float} for all lags tested
    interpretation: str


@dataclass
class EventStudyResult:
    """
    Event study: financial market response after large Polymarket probability moves.

    Events are defined by |dP| > threshold_pp (e.g. 3 percentage points per hour).
    avg_response and avg_response_t_stats keys are forward window labels like "1h", "3h".
    """
    n_events: int                       # number of qualifying Polymarket events
    threshold_pct: float                # dP threshold that defined an event (pp)
    avg_response: dict                  # {window_label: mean_cumulative_fin_return}
    avg_response_t_stats: dict          # {window_label: t-statistic}
    direction_consistency: float        # fraction of events where fin moved same direction as dP
    interpretation: str


@dataclass
class SpikeEventStudyResult:
    """
    Price discovery direction around Polymarket probability spikes.

    For each qualifying spike (|dP| >= spike_threshold_pp per hour), measures
    cumulative financial returns in the preceding and following window_hours.

    discovery_direction values:
      "polymarket_leads"  — fin moved AFTER the spike (information was NOT yet priced in)
      "financial_leads"   — fin moved BEFORE the spike (information was already priced in)
      "contemporaneous"   — both pre and post windows show directional agreement
      "no_signal"         — no consistent directional pattern found
    """
    n_spikes: int                       # number of qualifying spikes analyzed
    spike_threshold_pp: float           # |dP| threshold that defined a spike (pp)
    window_hours: int                   # hours examined before and after each spike
    pre_spike_avg_return: float         # mean cumulative fin return in pre-spike window
    post_spike_avg_return: float        # mean cumulative fin return in post-spike window
    pre_spike_t_stat: float             # t-statistic for pre_spike_avg_return vs zero
    post_spike_t_stat: float            # t-statistic for post_spike_avg_return vs zero
    directional_pre_agreement: float    # fraction of spikes where pre-window fin moved same way
    directional_post_agreement: float   # fraction of spikes where post-window fin moved same way
    discovery_direction: str            # see docstring above
    interpretation: str


@dataclass
class VolatilitySpilloverResult:
    """
    Volatility spillover: does Polymarket realized volatility predict financial
    realized volatility?  Both series use a rolling window standard deviation of
    absolute probability changes (dP) and log-returns respectively.
    """
    vol_correlation: float          # Pearson correlation of rolling vol series
    vol_correlation_p: float        # p-value for vol_correlation
    poly_vol_predicts_fin: bool     # True if lagged poly vol Granger-predicts fin vol
    poly_vol_lead_p_value: float    # p-value for the Granger-like lead F-test
    interpretation: str


@dataclass
class DivergenceSignalResult:
    """
    Current divergence signal: compares what the financial market "should" have
    moved (based on historical OLS β against Polymarket dP) with what it actually
    moved over a recent lookback window.

    A large positive divergence_pct signals financial market UNDERREACTION — the
    instrument has not yet moved as much as the historical β would imply given the
    recent Polymarket probability change.  A large negative divergence_pct signals
    OVERREACTION (financial moved more than implied).

    signal_direction values:
      "underreaction"  — financial significantly lagged the implied move (z > threshold)
      "overreaction"   — financial moved more than implied (z < -threshold)
      "aligned"        — divergence within noise (|z| < threshold)
      "no_signal"      — insufficient data, flat Polymarket series, or OLS failed
    """
    lookback_hours: int             # recent window examined (hours)
    poly_move_pp: float             # cumulative Polymarket move over lookback (pp)
    current_probability: float      # current Polymarket probability [0, 1]
    predicted_fin_return_pct: float # OLS-implied financial return over lookback (%)
    actual_fin_return_pct: float    # actual financial log-return over lookback (%)
    divergence_pct: float           # predicted - actual (%; positive = underreaction)
    divergence_z: float             # divergence normalised by scaled historical residual std
    regression_beta: float          # fitted β: fin log-return per unit Polymarket dP
    regression_r_squared: float     # R² of the historical OLS fit
    signal_direction: str           # see docstring above
    interpretation: str


# ══════════════════════════════════════════════════════════════════════════════
# DEPRECATED TEST RESULT CLASSES
# (not produced by the current pipeline; retained so old JSON output files can
#  still be deserialized without errors by tools that use asdict/from_dict)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RollingCorrelationResult:
    """
    DEPRECATED — rolling correlation provided little additional signal beyond
    CCF and Granger on 48-hour hourly windows; excluded from the active suite.
    """
    window_size: int
    mean_correlation: float
    std_correlation: float
    min_correlation: float
    max_correlation: float
    current_correlation: float  # most recent window
    trend: str                  # "strengthening" | "weakening" | "stable"


@dataclass
class DTWResult:
    """
    DEPRECATED — Dynamic Time Warping shape similarity is not meaningful when
    comparing bounded [0,1] probability data to unbounded financial prices.
    """
    dtw_distance: float
    normalized_distance: float
    optimal_path_length: int
    interpretation: str


@dataclass
class VolumeAnalysisResult:
    """
    DEPRECATED — Polymarket dollar volume is not comparable to exchange trading
    volume; spike coincidence analysis was not informative for this pair type.
    """
    poly_spike_count: int
    fin_spike_count: int
    coincident_spikes: int
    coincidence_ratio: float    # coincident / max(poly_spikes, fin_spikes)
    volume_correlation: float
    interpretation: str


@dataclass
class CointegrationResult:
    """
    DEPRECATED — Engle-Granger cointegration requires long-run price level
    stationarity; not appropriate for 48-hour bounded probability series.
    """
    test_statistic: float
    p_value: float
    critical_values: dict       # {"1%": x, "5%": y, "10%": z}
    is_cointegrated: bool
    interpretation: str


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT MODELS (for the Synthesizer)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairAnalysis:
    """
    Complete analysis for one Polymarket-financial instrument pair.

    This is the primary output record consumed by the Synthesizer agent.
    Each PairAnalysis contains the results of all statistical tests that were
    selected and run for this specific pair, plus a composite score and
    LLM-generated narrative summary.
    """
    # ── Identifiers ──────────────────────────────────────────────────────────
    analysis_id: str            # short UUID for deduplication
    polymarket_id: str
    polymarket_question: str
    ticker: str
    ticker_name: str
    recommendation_reasoning: str

    # ── Metadata ─────────────────────────────────────────────────────────────
    analysis_timestamp: str     # ISO-8601 UTC when this analysis was run
    data_start_date: str        # first aligned timestamp
    data_end_date: str          # last aligned timestamp
    n_observations: int         # number of aligned hourly observations

    # ── Active test results (None if test was skipped / insufficient data) ───
    correlation: Optional[CorrelationResult] = None
    granger_causality: Optional[GrangerResult] = None
    lead_lag_ccf: Optional[LeadLagResult] = None
    event_study: Optional[EventStudyResult] = None
    spike_event_study: Optional[SpikeEventStudyResult] = None
    volatility_spillover: Optional[VolatilitySpilloverResult] = None
    divergence_signal: Optional[DivergenceSignalResult] = None

    # ── Deprecated test results (always None in current output; kept for compat) ──
    rolling_correlations: list[RollingCorrelationResult] = field(default_factory=list)
    dtw: Optional[DTWResult] = None
    volume_analysis: Optional[VolumeAnalysisResult] = None
    cointegration: Optional[CointegrationResult] = None

    # ── Composite score and narrative ────────────────────────────────────────
    overall_similarity_score: float = 0.0  # weighted composite [0, 1]
    confidence_level: str = "low"           # "low" | "medium" | "high"
    agent_summary: str = ""                 # mechanical or LLM-generated narrative
    key_findings: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class TrendAnalystOutput:
    """
    Top-level output payload: all pair analyses for a single scan run.
    This is the full document delivered to the Synthesizer agent.
    """
    scan_timestamp: str         # timestamp of the Polymarket scan used as input
    analysis_timestamp: str     # when the Trend Analyst ran
    analyst_version: str = "0.3.0"
    pairs: list[PairAnalysis] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.to_json())
        print(f"[TrendAnalyst] Output saved to {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
# DUAL-MODE OUTPUT MODELS (v0.4.1)
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared primitive ──────────────────────────────────────────────────────────

@dataclass
class CorrelationBreakdownResult:
    """
    Pre/post spike correlation comparison (used by both realtime and historical modes).

    Measures Pearson correlation in the pre-spike window vs the post-spike window.
    A significant drop (breakdown_detected=True) signals the two series decoupled
    after the Polymarket event.
    """
    pre_correlation: float      # Pearson r before the spike
    pre_p_value: float
    post_correlation: float     # Pearson r after the spike
    post_p_value: float
    correlation_change: float   # post - pre  (negative = breakdown)
    breakdown_detected: bool    # True when pre_r was strong AND post_r fell enough
    n_pre: int                  # change-observations in pre window
    n_post: int                 # change-observations in post window
    interpretation: str


# ── Historical mode (thesis verification) ────────────────────────────────────

@dataclass
class SpikeEvent:
    """
    Evidence record for a single Polymarket spike event in the historical backtest.

    Each SpikeEvent shows what Polymarket did and what the financial market did
    in the hours BEFORE and AFTER the spike — the raw evidence for/against
    the thesis that Polymarket led financial markets.
    """
    spike_timestamp: str        # ISO-8601 UTC timestamp of the spike hour
    poly_move_pp: float         # Polymarket absolute move at spike (percentage points)

    # Financial market behavior around the spike
    pre_window_hours: int       # how many hours before the spike we measured
    post_window_hours: int      # how many hours after the spike we measured
    pre_fin_return_pct: float   # cumulative fin return in pre window (%)
    post_fin_return_pct: float  # cumulative fin return in post window (%)

    # Thesis assessment for this individual event
    poly_led: bool              # True = fin moved more AFTER spike than before (Poly led)
    thesis_support: str         # "supports" | "contradicts" | "inconclusive"
    explanation: str            # human-readable explanation of this event


@dataclass
class HistoricalPairResult:
    """
    Thesis-verification result for one Polymarket-financial pair.

    Answers: "Did Polymarket systematically LEAD the financial market around
    spike events in the historical data?"

    Contains:
      - A binary thesis verdict per pair (CONFIRMED / INCONCLUSIVE / NOT_CONFIRMED)
      - Event-by-event evidence (one SpikeEvent per qualifying spike)
      - Supporting statistics from CCF and Granger tests
    """
    analysis_id: str
    polymarket_id: str
    polymarket_question: str
    ticker: str
    ticker_name: str
    n_observations: int

    # ── Thesis verdict ────────────────────────────────────────────────────────
    thesis_verdict: str         # "CONFIRMED" | "INCONCLUSIVE" | "NOT_CONFIRMED"
    n_supporting_events: int    # spike events where Polymarket led
    n_total_events: int         # total qualifying spike events examined
    support_rate: float         # n_supporting / n_total  [0, 1]

    # ── Event-by-event evidence ───────────────────────────────────────────────
    spike_events: list[SpikeEvent] = field(default_factory=list)

    # ── Supporting statistical tests ──────────────────────────────────────────
    lead_direction: str = "neutral"     # "polymarket_leads" | "financial_leads" | "neutral"
    lead_lag_hours: int = 0             # CCF peak lag (positive = Polymarket leads)
    lead_lag_peak_r: float = 0.0
    lead_lag_p_value: float = 1.0

    granger_poly_causes_fin: bool = False
    granger_poly_best_lag: int = 0
    granger_poly_p_value: float = 1.0
    granger_fin_causes_poly: bool = False
    granger_fin_best_lag: int = 0
    granger_fin_p_value: float = 1.0

    timing_confidence: str = "low"      # "high" | "medium" | "low"
    analysis_timestamp: str = ""
    summary: str = ""
    caveats: list[str] = field(default_factory=list)

    # Raw test objects for downstream consumers
    raw_lead_lag: Optional[LeadLagResult] = None
    raw_granger: Optional[GrangerResult] = None
    raw_spike_event: Optional[SpikeEventStudyResult] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HistoricalAnalystOutput:
    """
    Top-level output of historical_analyst.py.

    Contains the thesis verdict: "Does Polymarket systematically lead financial
    markets around spike events?"

    Includes both an aggregate verdict across all pairs and per-pair evidence.
    """
    analysis_type: str = "historical_thesis_verification"
    analysis_timestamp: str = ""
    analyst_version: str = "1.0.0"

    # ── Aggregate thesis verdict ──────────────────────────────────────────────
    overall_thesis_verdict: str = "INCONCLUSIVE"
    # "CONFIRMED"     — majority of pairs show Polymarket leading with statistical significance
    # "INCONCLUSIVE"  — mixed evidence or insufficient data
    # "NOT_CONFIRMED" — evidence does not support Polymarket leading

    overall_support_rate: float = 0.0   # fraction of spike events that support thesis
    n_pairs_confirmed: int = 0          # pairs where thesis_verdict == CONFIRMED
    n_pairs_total: int = 0

    pairs: list[HistoricalPairResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.to_json())
        print(f"[HistoricalAnalyst] Output saved to {filepath}")


# ── Realtime mode (alert system) ──────────────────────────────────────────────

@dataclass
class RealtimePairResult:
    """
    Alert record for one Polymarket-financial pair in realtime monitoring mode.

    Answers: "Should I be alerted about an information gap between this
    Polymarket market and its financial instrument RIGHT NOW?"

    Two independent triggers can fire an alert:
      Trigger 1 — UNPRICED_MOVE:
        Polymarket moved significantly and the financial market has NOT caught
        up to what the historical β would imply.  This is the most actionable
        signal: there is a measurable gap between where the financial market
        IS and where the Polymarket signal implies it SHOULD be.

      Trigger 2 — CORRELATION_BREAKDOWN:
        The two series were moving together before the Polymarket spike (t=0)
        but have stopped moving together after it.  This suggests the financial
        market is ignoring new information that Polymarket has already priced in.

    alert_status is "ALERT" when one or both triggers fire; "NO_ALERT" otherwise.
    """
    analysis_id: str
    polymarket_id: str
    polymarket_question: str
    ticker: str
    ticker_name: str

    # ── Alert decision ────────────────────────────────────────────────────────
    alert_status: str           # "ALERT" | "NO_ALERT"
    alert_triggers: list[str]   # which conditions fired: "UNPRICED_MOVE", "CORRELATION_BREAKDOWN"
    alert_message: str          # clear, actionable one-paragraph alert text

    # ── Trigger 1 detail: unpriced Polymarket move ───────────────────────────
    unpriced_move_detected: bool
    poly_move_pp: float         # how much Polymarket moved in the recent window
    implied_fin_move_pct: float # what the financial market "should" have done (β × poly_move)
    actual_fin_move_pct: float  # what it actually did
    move_gap_pct: float         # implied - actual (positive = underreaction)
    divergence_z: float         # z-score of the gap
    divergence_direction: str   # "underreaction" | "overreaction" | "aligned" | "no_signal"

    # ── Trigger 2 detail: correlation breakdown ───────────────────────────────
    breakdown_detected: bool
    pre_spike_correlation: float    # Pearson r before t=0
    post_spike_correlation: float   # Pearson r after t=0
    correlation_change: float       # post - pre

    # ── Severity score ────────────────────────────────────────────────────────
    alert_score: float          # 0.0–1.0 combined severity (useful for ranking)
    n_pre_obs: int
    n_post_obs: int

    analysis_timestamp: str
    caveats: list[str] = field(default_factory=list)

    # Raw signal objects for downstream consumers
    raw_divergence: Optional[DivergenceSignalResult] = None
    raw_correlation_breakdown: Optional[CorrelationBreakdownResult] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RealtimeAnalystOutput:
    """
    Top-level output of realtime_analyst.py.

    Contains the current alert state across all monitored pairs.
    Pairs with alert_status == "ALERT" are sorted first.
    """
    analysis_type: str = "realtime_alert"
    analysis_timestamp: str = ""
    analyst_version: str = "1.0.0"

    # ── Alert summary ─────────────────────────────────────────────────────────
    n_alerts: int = 0           # number of pairs currently triggering alerts
    n_pairs_monitored: int = 0

    pairs: list[RealtimePairResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.to_json())
        print(f"[RealtimeAnalyst] Output saved to {filepath}")
