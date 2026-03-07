"""
Trend Analyst Configuration  (v0.3.0)
---------------------------------------
Central config for data sources, statistical parameters, and output settings.

ACTIVE TESTS (6): correlation, granger, lead_lag_ccf, event_study,
                  spike_event_study, volatility_spillover
DEPRECATED TESTS (4, kept for backward compat): rolling_correlation, dtw,
                  volume, cointegration
"""

# ── Data Sources ──────────────────────────────────────────────────────────────

# Public URL for the Polymarket scan snapshot (no hourly time series)
POLYMARKET_SCAN_URL = (
    "https://pub-758436d0743b4cee966caace6c186999.r2.dev/latest_scan.json"
)

# Cloudflare R2 bucket used for both input (financial data) and output (analysis)
R2_BUCKET_URL = (
    "https://3164db02212ba104d3623df6c4a26a97.r2.cloudflarestorage.com/polymarket-data"
)


# ── Statistical Test Parameters ───────────────────────────────────────────────
#
# Each active test reads its own sub-dict from STATS_CONFIG.
# Deprecated test configs are kept so the deprecated functions in
# statistical_toolkit.py still work without errors.

STATS_CONFIG = {

    # ── ACTIVE TESTS ─────────────────────────────────────────────────────────

    # Pearson / Spearman correlation (poly dP vs fin log-returns)
    "correlation": {
        "min_observations": 20,
        "significance_level": 0.05,
    },

    # Granger causality (bidirectional, OLS F-test)
    "granger": {
        "max_lag": 5,               # test lags 1h through max_lag h
        "significance_level": 0.05,
    },

    # Lead-lag cross-correlation function (CCF)
    # Positive lag = Polymarket leads financial market by that many hours
    "lead_lag_ccf": {
        "max_lag": 10,              # test lags -max_lag to +max_lag (hours)
        "significance_level": 0.05,
    },

    # Event study: financial response after large Polymarket probability moves
    # Events defined by |dP| > threshold_pp per hour (not log-return z-scores).
    # This is more appropriate for bounded probability data.
    "event_study": {
        "threshold_pp": 3.0,        # |dP| > 3 percentage points per hour = event
        "forward_windows": [1, 3, 6, 12],  # hours forward to measure fin response
        "min_events": 3,            # skip if fewer than this many events found
    },

    # Spike event study: pre/post window analysis to determine price discovery direction.
    # Key question: did financial markets price in the event BEFORE or AFTER Polymarket?
    "spike_event_study": {
        "spike_threshold_pp": 3.0,  # |dP| > 3pp = qualifying spike
        "window_hours": 6,          # look +/-6 hours around each spike
        "min_spikes": 2,            # need at least 2 spikes with full margins on both sides
    },

    # Volatility spillover: does Polymarket rolling vol predict financial rolling vol?
    "volatility_spillover": {
        "rolling_window": 5,        # rolling window size (hours) for realized vol
        "min_observations": 15,     # minimum obs before computing rolling vol
    },

    # Current divergence signal: fits OLS on historical poly_dP → fin_returns,
    # then compares the predicted vs actual financial move over the most recent window.
    # Positive divergence = financial underreaction (hasn't moved as much as β implies).
    "divergence_signal": {
        "lookback_hours": 6,        # recent window to examine for divergence
        "min_train_obs": 20,        # minimum training observations for OLS fit
        "significance_z": 1.5,     # |z| threshold to classify as under/overreaction
    },

    # ── DEPRECATED TESTS (kept so deprecated functions still run without KeyError) ──

    # Rolling correlation — excluded: noisy on short 48h hourly windows; CCF is superior.
    "rolling_correlation": {
        "windows": [5, 10, 21],     # window sizes (hours) — originally days, repurposed
        "min_periods": 5,
    },

    # Dynamic Time Warping — excluded: shape similarity on bounded [0,1] vs unbounded prices
    # is not meaningful; the series exist in fundamentally different value domains.
    "dtw": {
        "window_size": 10,          # Sakoe-Chiba band constraint
        "normalize": True,
    },

    # Volume coincidence — excluded: Polymarket USD volume is not comparable to
    # exchange trading volume; coincidence analysis was not informative.
    "volume": {
        "spike_threshold_std": 2.0,     # volume spike = >2 std above mean
        "coincidence_window_days": 1,
    },

    # Engle-Granger cointegration — excluded: requires long-run price level
    # stationarity; not appropriate for 48-hour bounded probability series.
    "cointegration": {
        "significance_level": 0.05,
        "max_lag": 5,
        "trend": "c",               # constant trend in ADF regression
    },
}


# Ordered list of active statistical tests.
# Used by the LLM planner (anthropic_client.py) and static fallback (trend_analyst.py).
# Add/remove here and both places update automatically.
ACTIVE_TESTS = [
    "correlation", "granger", "lead_lag_ccf",
    "event_study", "spike_event_study", "volatility_spillover",
]

# ── Historical Mode Settings (historical_analyst.py) ─────────────────────────
#
# Controls how the historical timing-detection script interprets its results.
# Three tests vote on direction; confidence is determined by agreement count.

HISTORICAL_CONFIG = {
    # CCF: minimum |r| at peak lag to treat the lead as meaningful
    "lead_lag_r_min": 0.3,
    # Minimum number of hours of lead to report (lags of 0 = contemporaneous, skipped)
    "lead_lag_hours_min": 1,
    # Number of tests that must agree on the same direction for "high" confidence
    "high_confidence_votes": 3,    # all 3: CCF, Granger, SpikeEventStudy
    "medium_confidence_votes": 2,  # 2 out of 3
    # Thesis verdict rules
    "min_events_for_verdict": 3,   # need at least this many spike events to give a verdict
    "confirm_support_rate":  0.60, # ≥ 60% of events support thesis → CONFIRMED
    "reject_support_rate":   0.40, # < 40% of events support thesis → NOT_CONFIRMED
}


# ── Realtime Mode Settings (realtime_analyst.py) ──────────────────────────────
#
# Controls correlation-breakdown detection and the combined info-gap score.

REALTIME_CONFIG = {
    # Correlation breakdown thresholds
    "breakdown_pre_r_min":  0.4,    # pre-spike r must be at least this to qualify
    "breakdown_post_r_max": 0.2,    # post-spike r must fall at or below this
    "breakdown_delta_min":  0.3,    # |correlation_change| must be at least this
    # Divergence z-score threshold to flag underreaction/overreaction (same as trend_analyst)
    "divergence_z_threshold": 1.5,
    # Weights for combining the two signals into a single info_gap_score
    "info_gap_score_weights": {
        "correlation_breakdown": 0.5,
        "divergence":            0.5,
    },
}


# ── Output Settings ───────────────────────────────────────────────────────────

OUTPUT_DIR = "data/output"
OUTPUT_FORMAT = "json"


# ── Agent Settings ────────────────────────────────────────────────────────────

# System persona injected into LLM enrichment calls (OpenAI GPT-4o).
AGENT_PERSONA = (
    "You are an expert quantitative analyst specializing in cross-market "
    "statistical analysis. You compare prediction market movements with "
    "traditional financial market data to identify statistically significant "
    "parallel movements. You are rigorous, transparent about confidence levels, "
    "and always distinguish between correlation and causation."
)


# ── Interpretation Thresholds ─────────────────────────────────────────────────
#
# Used by statistical_toolkit.py to convert numeric results into plain-English
# interpretations.  Deprecated thresholds are kept so the deprecated test
# functions still produce interpretations without KeyError.

INTERPRETATION = {
    # Pearson/Spearman correlation strength cutoffs
    "correlation_strong":   0.7,
    "correlation_moderate": 0.4,
    "correlation_weak":     0.2,

    # Granger causality
    "granger_significant":  0.05,

    # Lead-lag CCF — peak correlation considered "strong"
    "ccf_strong":           0.5,

    # Event study — direction consistency thresholds
    "event_consistency_high":     0.7,   # 70%+ same-direction events = strong
    "event_consistency_moderate": 0.5,

    # Volatility spillover
    "vol_spillover_significant":  0.05,

    # ── Deprecated thresholds (used only by deprecated test functions) ────────
    "cointegration_significant":  0.05,  # Engle-Granger (deprecated)
    "dtw_similar_percentile":     25,    # DTW lower = more similar (deprecated)
    "volume_coincidence_high":    0.6,   # 60%+ volume spike overlap (deprecated)
}
