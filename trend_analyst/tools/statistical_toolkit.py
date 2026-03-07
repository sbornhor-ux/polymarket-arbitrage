"""
Statistical Toolkit
-------------------
Implements statistical tests for comparing Polymarket probability series with
financial market time series.

ACTIVE TEST SUITE (6 tests, redesigned for bounded probability data):
  1. Pearson / Spearman correlation      — poly absolute changes vs fin log-returns
  2. Granger causality (bidirectional)   — directional information flow
  3. Lead-lag cross-correlation (CCF)    — which market leads/lags the other
  4. Event study                         — absolute pp spike events → fin response
  5. Spike event study (NEW)             — pre/post window price discovery direction
  6. Volatility spillover                — poly vol regime predicts fin vol regime

DEPRECATED (kept for backward compatibility, not included in default weights):
  7. Rolling correlation
  8. Dynamic Time Warping (DTW)
  9. Volume spike coincidence
  10. Engle-Granger cointegration

Design note: All tests receive Polymarket absolute price changes (dP, in
probability units) rather than log-returns, since Polymarket prices are
bounded [0, 1] with flat periods and binary jumps — log-returns are
ill-defined at those boundaries and produce NaN/inf for flat segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.linalg import lstsq

from models.schemas import (
    CorrelationResult,
    RollingCorrelationResult,
    GrangerResult,
    DTWResult,
    VolumeAnalysisResult,
    CointegrationResult,
    LeadLagResult,
    EventStudyResult,
    SpikeEventStudyResult,
    VolatilitySpilloverResult,
    DivergenceSignalResult,
)
from config import STATS_CONFIG, INTERPRETATION


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    OLS via scipy lstsq.  X must already include an intercept column.
    Used by: compute_granger_causality, compute_volatility_spillover,
             compute_cointegration (deprecated).
    """
    coef, _, _, _ = lstsq(X, y)
    return y - X @ coef


def _adf_test(series: np.ndarray, max_lag: int = 5) -> tuple[float, float]:
    """
    Augmented Dickey-Fuller test (manual implementation, no statsmodels dependency).
    Returns (test_statistic, approximate_p_value).

    NOTE: Only used by the DEPRECATED compute_cointegration().
    Uses MacKinnon (1994) approximate critical values for the p-value mapping.
    """
    y = np.diff(series)
    n = len(y)
    lag = min(max_lag, n // 3)

    y_lag = series[lag:-1]
    dy = y[lag:]
    T = len(dy)

    X_cols = [np.ones(T), y_lag[:T]]
    for i in range(1, lag + 1):
        X_cols.append(y[lag - i: lag - i + T])
    X = np.column_stack(X_cols)

    coef, _, _, _ = lstsq(X, dy[:T])
    resid = dy[:T] - X @ coef
    se = np.sqrt(np.sum(resid ** 2) / (T - len(coef)))

    XtX_inv = np.linalg.pinv(X.T @ X)
    se_gamma = se * np.sqrt(XtX_inv[1, 1])
    t_stat = coef[1] / se_gamma if se_gamma > 1e-12 else 0.0

    # Approximate p-value using MacKinnon critical values
    if t_stat < -3.43:
        p_approx = 0.005
    elif t_stat < -2.86:
        p_approx = 0.03
    elif t_stat < -2.57:
        p_approx = 0.07
    elif t_stat < -1.94:
        p_approx = 0.15
    else:
        p_approx = 0.5 + min(0.49, t_stat * 0.1)

    return t_stat, max(0.0, min(1.0, p_approx))


# ══════════════════════════════════════════════════════════════════════════════
# 1. PEARSON / SPEARMAN CORRELATION
#    Input: poly_changes (absolute dP), fin_returns (log-returns)
#    Both length N-1 and index-aligned (same time interval i→i+1).
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlation(
    poly_changes: np.ndarray,
    fin_returns: np.ndarray,
) -> CorrelationResult:
    """
    Pearson and Spearman correlation between Polymarket absolute probability
    changes (dP) and financial log-returns.  Both arrays must be the same
    length and represent the same set of hourly intervals.
    """
    cfg = STATS_CONFIG["correlation"]
    n = len(poly_changes)

    if n < cfg["min_observations"]:
        return CorrelationResult(
            pearson_r=0.0, pearson_p=1.0, spearman_r=0.0, spearman_p=1.0,
            n_observations=n,
            interpretation=f"Insufficient data: {n} obs (need {cfg['min_observations']})",
        )

    # Guard against zero-variance (flat Polymarket price — all changes are zero)
    if np.std(poly_changes) < 1e-10:
        return CorrelationResult(
            pearson_r=0.0, pearson_p=1.0, spearman_r=0.0, spearman_p=1.0,
            n_observations=n,
            interpretation="Polymarket price flat over aligned window — correlation undefined",
        )

    pearson_r, pearson_p = sp_stats.pearsonr(poly_changes, fin_returns)
    spearman_r, spearman_p = sp_stats.spearmanr(poly_changes, fin_returns)

    abs_r = abs(pearson_r)
    if abs_r >= INTERPRETATION["correlation_strong"]:
        strength = "strong"
    elif abs_r >= INTERPRETATION["correlation_moderate"]:
        strength = "moderate"
    elif abs_r >= INTERPRETATION["correlation_weak"]:
        strength = "weak"
    else:
        strength = "negligible"

    direction = "positive" if pearson_r > 0 else "negative"
    sig = "statistically significant" if pearson_p < cfg["significance_level"] else "not statistically significant"

    interpretation = (
        f"{strength.capitalize()} {direction} correlation between Polymarket dP and "
        f"financial returns (r={pearson_r:.3f}, p={pearson_p:.4f}), {sig}. "
        f"Spearman rho={spearman_r:.3f} (p={spearman_p:.4f})."
    )

    return CorrelationResult(
        pearson_r=round(float(pearson_r), 6),
        pearson_p=round(float(pearson_p), 6),
        spearman_r=round(float(spearman_r), 6),
        spearman_p=round(float(spearman_p), 6),
        n_observations=n,
        interpretation=interpretation,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRANGER CAUSALITY (OLS + F-test)
#    Tests whether past poly_changes help predict fin_returns (and vice versa).
# ══════════════════════════════════════════════════════════════════════════════

def compute_granger_causality(
    poly_changes: np.ndarray,
    fin_returns: np.ndarray,
) -> GrangerResult:
    """
    Bidirectional Granger causality via OLS F-test.

    Tests two directions separately:
      poly_changes → fin_returns:  does past dP help predict future fin returns?
      fin_returns → poly_changes:  does past fin return help predict future dP?

    The best lag for each direction is the one that minimises the p-value
    across lags 1..max_lag.  Both arrays must be the same length (N-1).
    """
    cfg = STATS_CONFIG["granger"]
    max_lag = cfg["max_lag"]
    alpha = cfg["significance_level"]
    n = len(poly_changes)

    if n < max_lag * 3 + 10:
        return GrangerResult(
            poly_causes_fin=False, fin_causes_poly=False,
            poly_to_fin_best_lag=0, poly_to_fin_p_value=1.0,
            fin_to_poly_best_lag=0, fin_to_poly_p_value=1.0,
            interpretation=f"Insufficient data for Granger test: {n} obs",
        )

    def _granger_ftest(x, y, lag):
        T = len(y) - lag
        if T < lag * 2 + 5:
            return 1.0
        y_dep = y[lag:]
        cols_r = [np.ones(T)]
        for i in range(1, lag + 1):
            cols_r.append(y[lag - i: lag - i + T])
        X_r = np.column_stack(cols_r)
        cols_u = list(cols_r)
        for i in range(1, lag + 1):
            cols_u.append(x[lag - i: lag - i + T])
        X_u = np.column_stack(cols_u)
        try:
            resid_r = _ols_residuals(y_dep[:T], X_r)
            resid_u = _ols_residuals(y_dep[:T], X_u)
            rss_r = np.sum(resid_r ** 2)
            rss_u = np.sum(resid_u ** 2)
            p = lag
            df_u = T - X_u.shape[1]
            if df_u <= 0 or rss_u < 1e-15:
                return 1.0
            f_stat = ((rss_r - rss_u) / p) / (rss_u / df_u)
            return float(1.0 - sp_stats.f.cdf(f_stat, p, df_u))
        except Exception:
            return 1.0

    def _run(x, y):
        best_lag, best_p = 1, 1.0
        for lag in range(1, max_lag + 1):
            p = _granger_ftest(x, y, lag)
            if p < best_p:
                best_p, best_lag = p, lag
        return best_lag, best_p

    p2f_lag, p2f_p = _run(poly_changes, fin_returns)
    f2p_lag, f2p_p = _run(fin_returns, poly_changes)
    poly_causes = p2f_p < alpha
    fin_causes = f2p_p < alpha

    parts = []
    if poly_causes:
        parts.append(f"Polymarket dP Granger-causes financial returns at lag {p2f_lag}h (p={p2f_p:.4f})")
    if fin_causes:
        parts.append(f"Financial returns Granger-cause Polymarket dP at lag {f2p_lag}h (p={f2p_p:.4f})")
    if not poly_causes and not fin_causes:
        parts.append("No significant Granger-causal relationship in either direction")
    if poly_causes and fin_causes:
        parts.append("Bidirectional causality — possible feedback loop or common driver")

    return GrangerResult(
        poly_causes_fin=poly_causes, fin_causes_poly=fin_causes,
        poly_to_fin_best_lag=p2f_lag, poly_to_fin_p_value=round(p2f_p, 6),
        fin_to_poly_best_lag=f2p_lag, fin_to_poly_p_value=round(f2p_p, 6),
        interpretation=". ".join(parts) + ".",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. LEAD-LAG CROSS-CORRELATION FUNCTION (CCF)
#    Convention: positive lag k means poly_changes[t] correlates with
#    fin_returns[t+k] — Polymarket LEADS by k hours.
# ══════════════════════════════════════════════════════════════════════════════

def compute_lead_lag_ccf(
    poly_changes: np.ndarray,
    fin_returns: np.ndarray,
) -> LeadLagResult:
    cfg = STATS_CONFIG["lead_lag_ccf"]
    max_lag = cfg["max_lag"]
    alpha = cfg["significance_level"]
    n = len(poly_changes)

    if n < max_lag + 10:
        return LeadLagResult(
            peak_lag=0, peak_correlation=0.0, peak_p_value=1.0,
            ccf_values={}, interpretation=f"Insufficient data for CCF: {n} obs",
        )

    ccf_values = {}
    for k in range(-max_lag, max_lag + 1):
        if k > 0:
            x = poly_changes[:n - k]
            y = fin_returns[k:]
        elif k < 0:
            x = poly_changes[-k:]
            y = fin_returns[:n + k]
        else:
            x = poly_changes
            y = fin_returns
        if len(x) < 10 or np.std(x) < 1e-10:
            continue
        r, _ = sp_stats.pearsonr(x, y)
        ccf_values[k] = round(float(r), 6)

    if not ccf_values:
        return LeadLagResult(
            peak_lag=0, peak_correlation=0.0, peak_p_value=1.0,
            ccf_values={}, interpretation="Could not compute CCF",
        )

    peak_lag = max(ccf_values, key=lambda k: abs(ccf_values[k]))
    peak_r = ccf_values[peak_lag]

    n_eff = n - abs(peak_lag)
    if abs(peak_r) < 1.0 - 1e-10 and n_eff > 2:
        t_stat = peak_r * np.sqrt((n_eff - 2) / (1 - peak_r**2))
        peak_p = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), n_eff - 2)))
    else:
        peak_p = 0.0 if abs(peak_r) > 1.0 - 1e-10 else 1.0

    if peak_p < alpha and abs(peak_r) >= INTERPRETATION.get("ccf_strong", 0.5):
        if peak_lag > 0:
            interp = (f"Strong signal: Polymarket leads financial market by {peak_lag}h "
                      f"(r={peak_r:.3f}, p={peak_p:.4f})")
        elif peak_lag < 0:
            interp = (f"Strong signal: financial market leads Polymarket by {-peak_lag}h "
                      f"(r={peak_r:.3f}, p={peak_p:.4f})")
        else:
            interp = f"Strongest correlation is contemporaneous (r={peak_r:.3f}, p={peak_p:.4f})"
    elif peak_p < alpha:
        if peak_lag > 0:
            interp = (f"Moderate signal: Polymarket leads financial market by {peak_lag}h "
                      f"(r={peak_r:.3f}, p={peak_p:.4f})")
        elif peak_lag < 0:
            interp = (f"Moderate signal: financial market leads Polymarket by {-peak_lag}h "
                      f"(r={peak_r:.3f}, p={peak_p:.4f})")
        else:
            interp = f"Contemporaneous correlation dominates (r={peak_r:.3f}, p={peak_p:.4f})"
    else:
        interp = f"No significant lead-lag relationship detected (peak at lag {peak_lag}h, p={peak_p:.4f})"

    return LeadLagResult(
        peak_lag=peak_lag,
        peak_correlation=round(float(peak_r), 6),
        peak_p_value=round(float(peak_p), 6),
        ccf_values=ccf_values,
        interpretation=interp,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVENT STUDY (recalibrated)
#    Events: |dP| > threshold_pp (absolute probability change in pp).
#    Measures: cumulative fin returns over forward windows (hours).
# ══════════════════════════════════════════════════════════════════════════════

def compute_event_study(
    poly_prices: np.ndarray,
    fin_returns: np.ndarray,
) -> EventStudyResult:
    """
    Identify Polymarket spike events as absolute probability changes above a
    threshold (e.g., |dP| > 3pp per hour) and measure the cumulative financial
    return over subsequent hourly windows.

    poly_prices: raw probability levels, length N
    fin_returns: financial log-returns, length N-1 (interval i → i+1)
    """
    cfg = STATS_CONFIG["event_study"]
    threshold_pp = cfg.get("threshold_pp", 3.0)
    windows = cfg["forward_windows"]
    max_window = max(windows)
    min_events = cfg["min_events"]

    poly_changes_pp = np.diff(poly_prices) * 100   # length N-1, in percentage points
    n_changes = len(poly_changes_pp)
    n_ret = len(fin_returns)                        # should equal N-1

    if n_changes < max_window + 5:
        return EventStudyResult(
            n_events=0, threshold_pct=threshold_pp,
            avg_response={}, avg_response_t_stats={},
            direction_consistency=0.0,
            interpretation=f"Insufficient data for event study: {n_changes} intervals",
        )

    # Events at index i in poly_changes_pp: |dP_i| > threshold_pp
    # Contemporaneous fin return is fin_returns[i]; forward returns are fin_returns[i+1..i+w]
    event_indices = np.where(np.abs(poly_changes_pp) >= threshold_pp)[0]
    # Exclude events where we can't measure all forward windows
    event_indices = event_indices[event_indices + max_window < n_ret]

    if len(event_indices) < min_events:
        return EventStudyResult(
            n_events=len(event_indices), threshold_pct=threshold_pp,
            avg_response={}, avg_response_t_stats={},
            direction_consistency=0.0,
            interpretation=f"Only {len(event_indices)} events >{threshold_pp:.0f}pp (need {min_events}+)",
        )

    avg_response = {}
    avg_response_t = {}

    for w in windows:
        cum_returns = []
        for i in event_indices:
            # Cumulative fin return over the w hours following the event
            end = min(int(i) + 1 + w, n_ret)
            cum_r = float(np.sum(fin_returns[int(i) + 1: end]))
            cum_returns.append(cum_r)
        if not cum_returns:
            continue
        arr = np.array(cum_returns)
        mean_r = float(arr.mean())
        std_r = float(arr.std(ddof=1)) if len(arr) > 1 else 1e-10
        t_stat = mean_r / (std_r / np.sqrt(len(arr))) if std_r > 1e-10 else 0.0
        avg_response[f"{w}h"] = round(mean_r, 6)
        avg_response_t[f"{w}h"] = round(float(t_stat), 4)

    # Direction consistency: does fin move same direction as poly event?
    direction_matches = 0
    for i in event_indices:
        poly_dir = np.sign(poly_changes_pp[int(i)])
        fwd = min(3, n_ret - int(i) - 1)
        if fwd > 0:
            fin_cum = np.sum(fin_returns[int(i) + 1: int(i) + 1 + fwd])
            if np.sign(fin_cum) == poly_dir:
                direction_matches += 1
    dir_consistency = direction_matches / len(event_indices)

    n_events = len(event_indices)
    any_significant = any(abs(t) > 1.96 for t in avg_response_t.values())
    if dir_consistency >= INTERPRETATION.get("event_consistency_high", 0.7) and any_significant:
        interp = (f"Strong event response: after >{threshold_pp:.0f}pp Polymarket moves, "
                  f"financials respond same direction {dir_consistency:.0%} of the time "
                  f"({n_events} events)")
    elif dir_consistency >= INTERPRETATION.get("event_consistency_moderate", 0.5):
        interp = (f"Moderate event response: {dir_consistency:.0%} direction consistency "
                  f"across {n_events} events (threshold: {threshold_pp:.0f}pp)")
    else:
        interp = (f"Weak/no event response: {dir_consistency:.0%} direction consistency "
                  f"across {n_events} events")

    return EventStudyResult(
        n_events=n_events,
        threshold_pct=threshold_pp,
        avg_response=avg_response,
        avg_response_t_stats=avg_response_t,
        direction_consistency=round(float(dir_consistency), 4),
        interpretation=interp,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. SPIKE EVENT STUDY (NEW) — price discovery direction
#    For each large Polymarket spike, measures cumulative fin returns in the
#    preceding and following window_hours to determine price discovery direction.
# ══════════════════════════════════════════════════════════════════════════════

def compute_spike_event_study(
    poly_prices: np.ndarray,
    fin_returns: np.ndarray,
) -> SpikeEventStudyResult:
    """
    Pre/post window analysis around Polymarket probability spikes to determine
    price discovery direction:

      "polymarket_leads"  — fin moved AFTER the spike (info NOT priced in)
      "financial_leads"   — fin moved BEFORE the spike (info already priced in)
      "contemporaneous"   — both pre and post show directional agreement
      "no_signal"         — no consistent pattern

    poly_prices: raw probability levels, length N
    fin_returns: financial log-returns, length N-1
    """
    cfg = STATS_CONFIG.get("spike_event_study", {})
    threshold_pp = float(cfg.get("spike_threshold_pp", 3.0))
    window_h = int(cfg.get("window_hours", 6))
    min_spikes = int(cfg.get("min_spikes", 2))

    poly_changes_pp = np.diff(poly_prices) * 100   # length N-1
    n_ret = len(fin_returns)                        # should equal N-1

    # Find spikes (qualifying as large absolute probability moves)
    spike_mask = np.abs(poly_changes_pp) >= threshold_pp
    spike_indices = np.where(spike_mask)[0]  # indices into poly_changes_pp / fin_returns

    # Keep only spikes with enough pre AND post data for full windows
    valid = (spike_indices >= window_h) & (spike_indices + window_h < n_ret)
    spike_indices = spike_indices[valid]
    spike_changes = poly_changes_pp[spike_indices]

    if len(spike_indices) < min_spikes:
        return SpikeEventStudyResult(
            n_spikes=len(spike_indices),
            spike_threshold_pp=threshold_pp,
            window_hours=window_h,
            pre_spike_avg_return=0.0,
            post_spike_avg_return=0.0,
            pre_spike_t_stat=0.0,
            post_spike_t_stat=0.0,
            directional_pre_agreement=0.0,
            directional_post_agreement=0.0,
            discovery_direction="no_signal",
            interpretation=(
                f"Only {len(spike_indices)} qualifying spikes >{threshold_pp:.0f}pp "
                f"(need {min_spikes}+ with >{window_h}h margin on each side)"
            ),
        )

    pre_returns = []
    post_returns = []
    directional_pre = 0
    directional_post = 0

    for idx, delta in zip(spike_indices, spike_changes):
        spike_dir = np.sign(delta)
        # Pre-window: fin returns in the window_h hours BEFORE the spike
        pre_cum = float(np.sum(fin_returns[int(idx) - window_h: int(idx)]))
        # Post-window: fin returns in the window_h hours AFTER the spike
        post_cum = float(np.sum(fin_returns[int(idx) + 1: int(idx) + 1 + window_h]))
        pre_returns.append(pre_cum)
        post_returns.append(post_cum)
        if np.sign(pre_cum) == spike_dir:
            directional_pre += 1
        if np.sign(post_cum) == spike_dir:
            directional_post += 1

    pre_arr = np.array(pre_returns)
    post_arr = np.array(post_returns)
    n_spikes = len(pre_arr)

    pre_mean = float(pre_arr.mean())
    post_mean = float(post_arr.mean())
    pre_std = float(pre_arr.std(ddof=1)) if n_spikes > 1 else 1e-10
    post_std = float(post_arr.std(ddof=1)) if n_spikes > 1 else 1e-10

    pre_t = pre_mean / (pre_std / np.sqrt(n_spikes)) if pre_std > 1e-10 else 0.0
    post_t = post_mean / (post_std / np.sqrt(n_spikes)) if post_std > 1e-10 else 0.0

    dir_pre = directional_pre / n_spikes
    dir_post = directional_post / n_spikes

    # Determine price discovery direction
    # A threshold of 0.65 requires majority agreement; 0.5 is random
    pre_dominant = dir_pre >= 0.65
    post_dominant = dir_post >= 0.65

    if post_dominant and not pre_dominant:
        direction = "polymarket_leads"
    elif pre_dominant and not post_dominant:
        direction = "financial_leads"
    elif pre_dominant and post_dominant:
        direction = "contemporaneous"
    else:
        direction = "no_signal"

    if direction == "polymarket_leads":
        interp = (
            f"Polymarket LEADS the financial market ({n_spikes} spikes >{threshold_pp:.0f}pp): "
            f"fin moves same direction as spike {dir_post:.0%} post-spike vs "
            f"{dir_pre:.0%} pre-spike. Information was NOT priced in before the spike."
        )
    elif direction == "financial_leads":
        interp = (
            f"Financial market LEADS Polymarket ({n_spikes} spikes >{threshold_pp:.0f}pp): "
            f"fin moved same direction {dir_pre:.0%} pre-spike vs "
            f"{dir_post:.0%} post-spike. Information MAY HAVE BEEN priced in already."
        )
    elif direction == "contemporaneous":
        interp = (
            f"Markets move together around spikes ({n_spikes} events, >{threshold_pp:.0f}pp): "
            f"pre-agreement {dir_pre:.0%}, post-agreement {dir_post:.0%}. "
            f"Cannot determine price discovery direction."
        )
    else:
        interp = (
            f"No clear price discovery pattern ({n_spikes} spikes >{threshold_pp:.0f}pp): "
            f"pre-agreement {dir_pre:.0%}, post-agreement {dir_post:.0%}."
        )

    return SpikeEventStudyResult(
        n_spikes=n_spikes,
        spike_threshold_pp=threshold_pp,
        window_hours=window_h,
        pre_spike_avg_return=round(pre_mean, 6),
        post_spike_avg_return=round(post_mean, 6),
        pre_spike_t_stat=round(float(pre_t), 4),
        post_spike_t_stat=round(float(post_t), 4),
        directional_pre_agreement=round(dir_pre, 4),
        directional_post_agreement=round(dir_post, 4),
        discovery_direction=direction,
        interpretation=interp,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. VOLATILITY SPILLOVER
#    Does rolling poly volatility (std of |dP|) predict rolling fin volatility?
# ══════════════════════════════════════════════════════════════════════════════

def compute_volatility_spillover(
    poly_changes: np.ndarray,
    fin_returns: np.ndarray,
) -> VolatilitySpilloverResult:
    """
    Rolling realized volatility for both series, correlate them, and test
    whether poly vol predicts fin vol (Granger-like F-test on vol series).

    poly_changes: absolute probability changes (dP), length N-1
    fin_returns:  financial log-returns, length N-1
    """
    cfg = STATS_CONFIG["volatility_spillover"]
    window = cfg["rolling_window"]
    min_obs = cfg["min_observations"]
    n = len(poly_changes)

    if n < min_obs:
        return VolatilitySpilloverResult(
            vol_correlation=0.0, vol_correlation_p=1.0,
            poly_vol_predicts_fin=False, poly_vol_lead_p_value=1.0,
            interpretation=f"Insufficient data for volatility analysis: {n} obs",
        )

    # Rolling realized vol
    poly_vol = pd.Series(poly_changes).rolling(window=window, min_periods=window).std().dropna().values
    fin_vol = pd.Series(fin_returns).rolling(window=window, min_periods=window).std().dropna().values

    min_len = min(len(poly_vol), len(fin_vol))
    if min_len < 10:
        return VolatilitySpilloverResult(
            vol_correlation=0.0, vol_correlation_p=1.0,
            poly_vol_predicts_fin=False, poly_vol_lead_p_value=1.0,
            interpretation="Insufficient rolling vol observations after windowing",
        )
    poly_vol = poly_vol[:min_len]
    fin_vol = fin_vol[:min_len]

    if np.std(poly_vol) < 1e-10 or np.std(fin_vol) < 1e-10:
        return VolatilitySpilloverResult(
            vol_correlation=0.0, vol_correlation_p=1.0,
            poly_vol_predicts_fin=False, poly_vol_lead_p_value=1.0,
            interpretation="Constant volatility series - correlation undefined",
        )
    vol_r, vol_p = sp_stats.pearsonr(poly_vol, fin_vol)

    # Granger-like 1-lag F-test: does lagged poly_vol predict fin_vol?
    T = len(fin_vol) - 1
    if T < 10:
        lead_p = 1.0
    else:
        y = fin_vol[1:]
        X_r = np.column_stack([np.ones(T), fin_vol[:T]])
        X_u = np.column_stack([np.ones(T), fin_vol[:T], poly_vol[:T]])
        try:
            resid_r = _ols_residuals(y, X_r)
            resid_u = _ols_residuals(y, X_u)
            rss_r = np.sum(resid_r**2)
            rss_u = np.sum(resid_u**2)
            df_u = T - X_u.shape[1]
            if df_u > 0 and rss_u > 1e-15:
                f_stat = ((rss_r - rss_u) / 1) / (rss_u / df_u)
                lead_p = float(1.0 - sp_stats.f.cdf(f_stat, 1, df_u))
            else:
                lead_p = 1.0
        except Exception:
            lead_p = 1.0

    vol_predicts = lead_p < INTERPRETATION.get("vol_spillover_significant", 0.05)

    parts = []
    if abs(vol_r) >= 0.5 and vol_p < 0.05:
        parts.append(f"Strong volatility co-movement (r={vol_r:.3f}, p={vol_p:.4f})")
    elif abs(vol_r) >= 0.3 and vol_p < 0.10:
        parts.append(f"Moderate volatility co-movement (r={vol_r:.3f}, p={vol_p:.4f})")
    else:
        parts.append(f"Weak volatility co-movement (r={vol_r:.3f}, p={vol_p:.4f})")

    if vol_predicts:
        parts.append(f"Polymarket volatility Granger-predicts financial volatility (p={lead_p:.4f})")
    else:
        parts.append("No significant volatility spillover from Polymarket to financial market")

    return VolatilitySpilloverResult(
        vol_correlation=round(float(vol_r), 6),
        vol_correlation_p=round(float(vol_p), 6),
        poly_vol_predicts_fin=vol_predicts,
        poly_vol_lead_p_value=round(float(lead_p), 6),
        interpretation=". ".join(parts) + ".",
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEPRECATED TESTS (kept for backward compatibility; not in default weights)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_correlation(
    poly_changes: np.ndarray,
    fin_returns: np.ndarray,
) -> list[RollingCorrelationResult]:
    """
    DEPRECATED — rolling correlation on short hourly series provides little
    additional signal beyond CCF and Granger; excluded from default test suite.
    Kept for backward compatibility.
    """
    cfg = STATS_CONFIG["rolling_correlation"]
    results = []
    poly_s = pd.Series(poly_changes)
    fin_s = pd.Series(fin_returns)

    for window in cfg["windows"]:
        if len(poly_changes) < window + cfg["min_periods"]:
            continue
        rolling = poly_s.rolling(window=window, min_periods=cfg["min_periods"]).corr(fin_s)
        rolling_clean = rolling.dropna()
        if len(rolling_clean) < 3:
            continue

        mid = len(rolling_clean) // 2
        diff = rolling_clean.iloc[mid:].mean() - rolling_clean.iloc[:mid].mean()
        trend = "strengthening" if diff > 0.1 else ("weakening" if diff < -0.1 else "stable")

        results.append(RollingCorrelationResult(
            window_size=window,
            mean_correlation=round(float(rolling_clean.mean()), 6),
            std_correlation=round(float(rolling_clean.std()), 6),
            min_correlation=round(float(rolling_clean.min()), 6),
            max_correlation=round(float(rolling_clean.max()), 6),
            current_correlation=round(float(rolling_clean.iloc[-1]), 6),
            trend=trend,
        ))
    return results


def _dtw_distance(s1, s2, window=10):
    """
    Dynamic Time Warping distance with Sakoe-Chiba band constraint.
    NOTE: Only used by the DEPRECATED compute_dtw().
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1],
            )
    i, j, path_length = n, m, 0
    while i > 0 or j > 0:
        path_length += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            opts = [(dtw_matrix[i-1,j-1],i-1,j-1),(dtw_matrix[i-1,j],i-1,j),(dtw_matrix[i,j-1],i,j-1)]
            _, i, j = min(opts, key=lambda x: x[0])
    return dtw_matrix[n, m], path_length


def compute_dtw(poly_prices, fin_prices):
    """DEPRECATED — shape similarity on bounded probability data is not meaningful."""
    cfg = STATS_CONFIG["dtw"]
    def _norm(x):
        mn, mx = x.min(), x.max()
        return np.zeros_like(x) if mx - mn < 1e-10 else (x - mn) / (mx - mn)

    pn = _norm(poly_prices.astype(np.float64))
    fn = _norm(fin_prices.astype(np.float64))
    dist, path_len = _dtw_distance(pn, fn, window=cfg["window_size"])
    nd = dist / path_len if path_len > 0 else dist

    if nd < 0.05:
        interp = "Very high shape similarity"
    elif nd < 0.15:
        interp = "Moderate shape similarity"
    elif nd < 0.30:
        interp = "Weak shape similarity"
    else:
        interp = "Low shape similarity"

    return DTWResult(
        dtw_distance=round(float(dist), 6),
        normalized_distance=round(float(nd), 6),
        optimal_path_length=path_len,
        interpretation=interp,
    )


def compute_volume_analysis(poly_volumes, fin_volumes):
    """DEPRECATED — Polymarket $ volume is not comparable to financial market volume."""
    cfg = STATS_CONFIG["volume"]
    def _spikes(v):
        if len(v) < 5 or np.std(v) < 1e-10:
            return np.zeros(len(v), dtype=bool)
        return (v - np.mean(v)) / np.std(v) > cfg["spike_threshold_std"]

    ps, fs = _spikes(poly_volumes), _spikes(fin_volumes)
    window = cfg["coincidence_window_days"]
    coincident = 0
    for i in range(len(ps)):
        if ps[i]:
            lo, hi = max(0, i - window), min(len(fs), i + window + 1)
            if fs[lo:hi].any():
                coincident += 1

    pc, fc = int(ps.sum()), int(fs.sum())
    ratio = coincident / max(pc, fc, 1)
    vol_corr = float(sp_stats.pearsonr(poly_volumes, fin_volumes)[0]) if len(poly_volumes) > 5 else 0.0

    return VolumeAnalysisResult(
        poly_spike_count=pc, fin_spike_count=fc, coincident_spikes=coincident,
        coincidence_ratio=round(float(ratio), 4),
        volume_correlation=round(vol_corr, 6),
        interpretation=f"Volume spike coincidence: {ratio:.0%} (deprecated test)",
    )


def compute_cointegration(poly_prices, fin_prices):
    """DEPRECATED — requires long-run stationarity; not appropriate for 48h bounded data."""
    cfg = STATS_CONFIG["cointegration"]
    if len(poly_prices) < 20:
        return CointegrationResult(
            test_statistic=0.0, p_value=1.0, critical_values={},
            is_cointegrated=False, interpretation="Insufficient data",
        )
    try:
        X = np.column_stack([np.ones(len(fin_prices)), fin_prices])
        residuals = _ols_residuals(poly_prices, X)
        t_stat, p_value = _adf_test(residuals, max_lag=cfg["max_lag"])
        crit_dict = {"1%": -3.90, "5%": -3.34, "10%": -3.05}
        is_coint = t_stat < crit_dict["5%"]

        interp = (
            f"{'Cointegrated' if is_coint else 'Not cointegrated'} "
            f"(ADF={t_stat:.3f}) — deprecated test"
        )
        return CointegrationResult(
            test_statistic=round(float(t_stat), 6), p_value=round(float(p_value), 6),
            critical_values=crit_dict, is_cointegrated=is_coint, interpretation=interp,
        )
    except Exception as e:
        return CointegrationResult(
            test_statistic=0.0, p_value=1.0, critical_values={},
            is_cointegrated=False, interpretation=f"Cointegration test failed: {e}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. CURRENT DIVERGENCE SIGNAL
#    Fits OLS: fin_return = α + β * poly_dP  on historical (training) data.
#    Applies the fitted β to the most recent `lookback_hours` Polymarket moves
#    to get a predicted financial return, then compares to the actual return.
#    Positive divergence = financial market has NOT yet moved as much as implied
#    (underreaction); negative = it moved more than implied (overreaction).
# ══════════════════════════════════════════════════════════════════════════════

def compute_divergence_signal(
    poly_prices: np.ndarray,
    fin_returns: np.ndarray,
) -> DivergenceSignalResult:
    """
    Detect whether the financial market has over- or under-reacted to a recent
    Polymarket probability shift compared to what the historical relationship implies.

    Algorithm:
      1. Compute poly_changes = diff(poly_prices)   — length N-1, aligned with fin_returns
      2. Split at N-1-lookback_hours:
           training window  = first (N-1-lookback_hours) observations
           recent window    = last  lookback_hours observations
      3. Fit OLS on training window: fin_return = α + β * poly_change
      4. Predict recent financial return: Σ(α + β * poly_change_i) for recent i
      5. Divergence = predicted - actual  (positive = financial underreaction)
      6. Z-score = divergence / (resid_std * √lookback_hours)

    poly_prices: raw probability levels [0, 1], length N
    fin_returns: financial log-returns,           length N-1
    """
    cfg = STATS_CONFIG.get("divergence_signal", {})
    lookback = int(cfg.get("lookback_hours", 6))
    min_train = int(cfg.get("min_train_obs", 20))
    sig_z = float(cfg.get("significance_z", 1.5))

    poly_changes = np.diff(poly_prices)    # length N-1
    n = len(fin_returns)                   # should equal N-1

    current_prob = float(poly_prices[-1])

    def _no_signal(reason: str) -> DivergenceSignalResult:
        return DivergenceSignalResult(
            lookback_hours=lookback,
            poly_move_pp=0.0,
            current_probability=current_prob,
            predicted_fin_return_pct=0.0,
            actual_fin_return_pct=0.0,
            divergence_pct=0.0,
            divergence_z=0.0,
            regression_beta=0.0,
            regression_r_squared=0.0,
            signal_direction="no_signal",
            interpretation=reason,
        )

    if n < min_train + lookback:
        return _no_signal(
            f"Insufficient data for divergence signal: {n} obs "
            f"(need {min_train + lookback}+ = {min_train} train + {lookback} recent)"
        )

    split = n - lookback
    train_poly = poly_changes[:split]
    train_fin  = fin_returns[:split]
    recent_poly = poly_changes[split:]
    recent_fin  = fin_returns[split:]

    # Guard: flat Polymarket in training window makes β undefined
    if np.std(train_poly) < 1e-10:
        return _no_signal("Polymarket price flat over training window — β undefined")

    # OLS fit: fin_return = α + β * poly_change
    X_train = np.column_stack([np.ones(len(train_poly)), train_poly])
    try:
        coef, _, _, _ = lstsq(X_train, train_fin)
    except Exception as e:
        return _no_signal(f"OLS fit failed: {e}")

    alpha, beta = float(coef[0]), float(coef[1])

    # R² on training data
    fitted   = X_train @ coef
    ss_res   = float(np.sum((train_fin - fitted) ** 2))
    ss_tot   = float(np.sum((train_fin - np.mean(train_fin)) ** 2))
    r_sq     = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # Historical residual std (basis for z-score)
    resid_std = float(np.std(train_fin - fitted, ddof=1))
    if resid_std < 1e-12:
        return _no_signal("Zero residual variance in training window")

    # Predicted and actual recent financial returns
    predicted_recent = float(sum(alpha + beta * dp for dp in recent_poly))
    actual_recent    = float(np.sum(recent_fin))
    divergence       = predicted_recent - actual_recent   # positive = underreaction

    # Z-score: sum of lookback_hours independent residuals scales std by √lookback
    z = divergence / (resid_std * np.sqrt(lookback))

    # Cumulative Polymarket move over recent window (in percentage points)
    poly_move_pp = float(np.sum(recent_poly)) * 100

    # Classify direction
    if abs(z) < sig_z:
        direction = "aligned"
    elif divergence > 0:
        direction = "underreaction"
    else:
        direction = "overreaction"

    # Human-readable interpretation
    prob_pct   = current_prob * 100
    pred_pct   = predicted_recent * 100
    actual_pct = actual_recent * 100
    gap_pct    = divergence * 100

    if direction == "underreaction":
        interp = (
            f"Financial market UNDERREACTION detected: Polymarket moved {poly_move_pp:+.1f}pp "
            f"over the last {lookback}h (now {prob_pct:.0f}%); historical β={beta:.4f} implies "
            f"a {pred_pct:+.2f}% financial move, but actual was {actual_pct:+.2f}% — "
            f"gap of {gap_pct:+.2f}% (z={z:.2f}). Financial market may not yet have priced in "
            f"the Polymarket signal."
        )
    elif direction == "overreaction":
        interp = (
            f"Financial market OVERREACTION detected: Polymarket moved {poly_move_pp:+.1f}pp "
            f"over the last {lookback}h (now {prob_pct:.0f}%); historical β={beta:.4f} implies "
            f"a {pred_pct:+.2f}% financial move, but actual was {actual_pct:+.2f}% — "
            f"financial moved MORE than expected (gap {gap_pct:+.2f}%, z={z:.2f})."
        )
    else:
        interp = (
            f"Financial market aligned with Polymarket signal: Polymarket moved "
            f"{poly_move_pp:+.1f}pp over last {lookback}h (now {prob_pct:.0f}%); "
            f"implied {pred_pct:+.2f}%, actual {actual_pct:+.2f}% (gap {gap_pct:+.2f}%, z={z:.2f})."
        )

    return DivergenceSignalResult(
        lookback_hours=lookback,
        poly_move_pp=round(poly_move_pp, 2),
        current_probability=round(current_prob, 4),
        predicted_fin_return_pct=round(pred_pct, 4),
        actual_fin_return_pct=round(actual_pct, 4),
        divergence_pct=round(gap_pct, 4),
        divergence_z=round(z, 4),
        regression_beta=round(beta, 6),
        regression_r_squared=round(r_sq, 4),
        signal_direction=direction,
        interpretation=interp,
    )


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORER
# ══════════════════════════════════════════════════════════════════════════════

# Active test weights (must sum to 1.0)
_DEFAULT_WEIGHTS = {
    "correlation":         0.10,
    "granger":             0.25,
    "lead_lag_ccf":        0.25,
    "event_study":         0.15,
    "spike_event_study":   0.15,
    "volatility_spillover": 0.10,
}


def compute_composite_score(
    correlation=None, granger=None,
    lead_lag_ccf=None, event_study=None, vol_spillover=None,
    spike_event_study=None,
    weights: dict | None = None,
):
    """
    Compute weighted composite similarity/co-movement score from test results.

    Active tests: correlation, granger, lead_lag_ccf, event_study,
                  spike_event_study, volatility_spillover.
    Deprecated tests (rolling, dtw_result, volume, cointegration) are accepted
    for backward compatibility but excluded from default weights.

    Pass `weights` to override defaults; it will be renormalized to 1.0.
    """
    scores = {}

    if correlation is not None:
        if correlation.n_observations >= 5 and abs(correlation.pearson_r) < 1.0:
            scores["correlation"] = min(abs(correlation.pearson_r), 1.0)

    if granger is not None:
        if granger.poly_causes_fin or granger.fin_causes_poly:
            scores["granger"] = 1.0
        elif min(granger.poly_to_fin_p_value, granger.fin_to_poly_p_value) < 0.10:
            scores["granger"] = 0.5
        else:
            scores["granger"] = 0.0

    if lead_lag_ccf is not None and lead_lag_ccf.ccf_values:
        raw = min(abs(lead_lag_ccf.peak_correlation), 1.0)
        scores["lead_lag_ccf"] = raw if lead_lag_ccf.peak_p_value < 0.05 else raw * 0.5
    elif lead_lag_ccf is not None:
        scores["lead_lag_ccf"] = 0.0

    if event_study is not None and event_study.n_events > 0:
        dc = event_study.direction_consistency
        any_sig = any(abs(t) > 1.96 for t in event_study.avg_response_t_stats.values())
        if dc >= 0.7 and any_sig:
            scores["event_study"] = 1.0
        elif dc >= 0.5:
            scores["event_study"] = 0.5
        else:
            scores["event_study"] = max(0, dc)
    elif event_study is not None:
        scores["event_study"] = 0.0

    if spike_event_study is not None and spike_event_study.n_spikes > 0:
        # Score based on strength of directionality (max of pre/post agreement)
        directional_strength = max(
            spike_event_study.directional_pre_agreement,
            spike_event_study.directional_post_agreement,
        )
        if spike_event_study.discovery_direction == "no_signal":
            scores["spike_event_study"] = 0.0
        else:
            # Scale: 0.5 agreement = 0, 1.0 agreement = 1.0
            scores["spike_event_study"] = max(0.0, (directional_strength - 0.5) * 2)
    elif spike_event_study is not None:
        scores["spike_event_study"] = 0.0

    if vol_spillover is not None:
        raw = min(abs(vol_spillover.vol_correlation), 1.0)
        if vol_spillover.poly_vol_predicts_fin:
            scores["volatility_spillover"] = min(raw + 0.3, 1.0)
        elif vol_spillover.vol_correlation_p < 0.10:
            scores["volatility_spillover"] = raw
        else:
            scores["volatility_spillover"] = raw * 0.5

    if not scores:
        return 0.0, "low"

    # ── Resolve weights ────────────────────────────────────────────────────
    base_w = weights if weights is not None else _DEFAULT_WEIGHTS
    active_w = {k: float(base_w.get(k, 0.0)) for k in scores}
    total = sum(active_w.values())
    if total > 0:
        active_w = {k: v / total for k, v in active_w.items()}
    else:
        equal = 1.0 / len(scores)
        active_w = {k: equal for k in scores}

    composite = sum(scores[k] * active_w.get(k, 0) for k in scores)

    # ── Confidence: scale by fraction of active tests that are significant
    n_tests = len(scores)
    n_sig = sum(1 for v in scores.values() if v > 0.5)
    sig_ratio = n_sig / n_tests
    if composite >= 0.6 and sig_ratio >= 0.5:
        confidence = "high"
    elif composite >= 0.35 and sig_ratio >= 0.30:
        confidence = "medium"
    else:
        confidence = "low"

    return round(float(composite), 4), confidence


# ══════════════════════════════════════════════════════════════════════════════
# REALTIME MODE: CORRELATION BREAKDOWN
#   Splits the 49-point series at t=0 (split_index) and measures whether
#   Pearson correlation dropped significantly between the pre- and post-spike
#   windows.  A large negative correlation_change signals decoupling.
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlation_breakdown(
    poly_prices: np.ndarray,
    fin_returns: np.ndarray,
    split_index: int,
) -> "CorrelationBreakdownResult":
    """
    Measure whether the correlation between Polymarket and financial markets
    dropped after a spike event (t=0 = split_index in the aligned arrays).

    pre window:  poly_prices[0 : split_index+1] → np.diff → poly_changes
                 fin_returns[0 : split_index]
    post window: poly_prices[split_index :]      → np.diff → poly_changes
                 fin_returns[split_index :]

    Breakdown is detected when:
      - pre_r  ≥  REALTIME_CONFIG["breakdown_pre_r_min"]   (was co-moving)
      - post_r ≤  REALTIME_CONFIG["breakdown_post_r_max"]  (decoupled)
      - |correlation_change| ≥ REALTIME_CONFIG["breakdown_delta_min"]

    Returns CorrelationBreakdownResult (imported from models.schemas via lazy import
    to avoid circular dependency — statistical_toolkit imports schemas via models.schemas).
    """
    from models.schemas import CorrelationBreakdownResult
    from config import REALTIME_CONFIG

    pre_r_min   = float(REALTIME_CONFIG.get("breakdown_pre_r_min",  0.4))
    post_r_max  = float(REALTIME_CONFIG.get("breakdown_post_r_max", 0.2))
    delta_min   = float(REALTIME_CONFIG.get("breakdown_delta_min",  0.3))

    def _safe_pearsonr(x, y):
        if len(x) < 4 or len(y) < 4 or len(x) != len(y):
            return 0.0, 1.0
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0, 1.0
        try:
            r, p = sp_stats.pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            return 0.0, 1.0

    # Pre-spike window
    pre_poly_changes = np.diff(poly_prices[:split_index + 1])
    pre_fin = fin_returns[:split_index]
    min_pre = min(len(pre_poly_changes), len(pre_fin))
    pre_poly_changes = pre_poly_changes[:min_pre]
    pre_fin = pre_fin[:min_pre]
    pre_r, pre_p = _safe_pearsonr(pre_poly_changes, pre_fin)

    # Post-spike window
    post_poly_changes = np.diff(poly_prices[split_index:])
    post_fin = fin_returns[split_index:]
    min_post = min(len(post_poly_changes), len(post_fin))
    post_poly_changes = post_poly_changes[:min_post]
    post_fin = post_fin[:min_post]
    post_r, post_p = _safe_pearsonr(post_poly_changes, post_fin)

    correlation_change = post_r - pre_r
    breakdown_detected = (
        pre_r >= pre_r_min
        and post_r <= post_r_max
        and abs(correlation_change) >= delta_min
    )

    if breakdown_detected:
        interp = (
            f"Correlation BREAKDOWN detected: pre-spike r={pre_r:.3f} collapsed to "
            f"post-spike r={post_r:.3f} (Δ={correlation_change:+.3f}). "
            f"The two series decoupled after the Polymarket event."
        )
    elif pre_r >= pre_r_min:
        interp = (
            f"Pre-spike correlation was strong (r={pre_r:.3f}) but did not collapse "
            f"significantly post-spike (r={post_r:.3f}, Δ={correlation_change:+.3f})."
        )
    else:
        interp = (
            f"Weak pre-spike correlation (r={pre_r:.3f}); breakdown detection requires "
            f"pre_r ≥ {pre_r_min:.2f} to be meaningful. Post-spike r={post_r:.3f}."
        )

    return CorrelationBreakdownResult(
        pre_correlation=round(float(pre_r), 6),
        pre_p_value=round(float(pre_p), 6),
        post_correlation=round(float(post_r), 6),
        post_p_value=round(float(post_p), 6),
        correlation_change=round(float(correlation_change), 6),
        breakdown_detected=breakdown_detected,
        n_pre=int(min_pre),
        n_post=int(min_post),
        interpretation=interp,
    )
