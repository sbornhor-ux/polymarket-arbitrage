"""
Detects significant probability swings in a resampled Polymarket series.

Method: rolling z-score of |Δp| over a configurable trailing window.
A swing is declared when z > threshold.

Windows checked: 1H, 6H, 1D, 5D
For each window, we find the single worst (highest z-score) period
that exceeds the threshold.
"""

from __future__ import annotations

from datetime import timezone
from typing import NamedTuple

import pandas as pd
import numpy as np

from polymarket_agent.models import MarketSnapshot, SwingEvent, DetectionRule


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_WINDOWS = ["1h", "6h", "1D", "5D"]
_TRAILING_LOOKBACK = "30D"
_Z_THRESHOLD = 2.0
_MIN_HISTORY_PERIODS = 20  # need at least this many 5m ticks for a stable z-score


class _WindowResult(NamedTuple):
    window_label: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    p_pre: float
    p_post: float
    delta_p: float
    abs_delta_p: float
    z_score: float


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_swings(
    snapshot: MarketSnapshot,
    trailing_lookback: str = _TRAILING_LOOKBACK,
    z_threshold: float = _Z_THRESHOLD,
) -> MarketSnapshot:
    """
    Detect swing events in snapshot.probability_series.

    Appends SwingEvent objects to snapshot.swings and returns snapshot.

    Args:
        snapshot: MarketSnapshot with a 5m resampled probability_series.
        trailing_lookback: Pandas offset for rolling z-score window (e.g. '30D').
        z_threshold: Minimum z-score to declare a swing.

    Returns:
        snapshot (mutated in place, returned for chaining).
    """
    if len(snapshot.probability_series) < _MIN_HISTORY_PERIODS:
        snapshot.notes["swing_detection_skipped"] = (
            f"Fewer than {_MIN_HISTORY_PERIODS} ticks — not enough history"
        )
        return snapshot

    df = _build_series(snapshot)
    rule = DetectionRule(
        trailing_lookback=trailing_lookback,
        threshold=z_threshold,
    )

    # Compute rolling z-score baseline for |Δp| over trailing window
    lookback_periods = _offset_to_5m_periods(trailing_lookback)
    rolling_mean = df["abs_dp"].rolling(lookback_periods, min_periods=10).mean()
    rolling_std = df["abs_dp"].rolling(lookback_periods, min_periods=10).std()

    df["z_score"] = (df["abs_dp"] - rolling_mean) / rolling_std.replace(0, np.nan)

    swings: list[SwingEvent] = []
    swing_counter = 0

    for window_label in _WINDOWS:
        result = _best_swing_in_window(df, window_label, z_threshold)
        if result is None:
            continue

        swing_counter += 1
        swings.append(
            SwingEvent(
                swing_id=f"S{swing_counter}",
                window=result.window_label,
                window_start_utc=result.window_start.isoformat(),
                window_end_utc=result.window_end.isoformat(),
                p_pre=round(result.p_pre, 4),
                p_post=round(result.p_post, 4),
                delta_p=round(result.delta_p, 4),
                abs_delta_p=round(result.abs_delta_p, 4),
                z_score_vs_trailing=round(result.z_score, 3),
                detection_rule=rule,
            )
        )

    snapshot.swings = swings
    return snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_series(snapshot: MarketSnapshot) -> pd.DataFrame:
    """Convert probability_series list into a DataFrame with Δp column."""
    rows = [{"ts_utc": t.ts_utc, "p": t.p} for t in snapshot.probability_series]
    df = pd.DataFrame(rows)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.set_index("ts_utc").sort_index()
    df["dp"] = df["p"].diff()
    df["abs_dp"] = df["dp"].abs()
    return df.dropna(subset=["dp"])


def _offset_to_5m_periods(offset: str) -> int:
    """Convert a Pandas offset alias (e.g. '30D') to number of 5-minute periods."""
    # Use a reference timestamp to materialise calendar-based offsets (Day, etc.)
    ref = pd.Timestamp("2000-01-01", tz="UTC")
    end = ref + pd.tseries.frequencies.to_offset(offset)
    seconds = (end - ref).total_seconds()
    return max(10, int(seconds / 300))


def _best_swing_in_window(
    df: pd.DataFrame,
    window_label: str,
    z_threshold: float,
) -> _WindowResult | None:
    """
    Find the rolling sub-window of length `window_label` ending at each tick
    with the highest z-score. Return the worst one if it exceeds the threshold.
    """
    n_periods = _offset_to_5m_periods(window_label)

    best: _WindowResult | None = None
    best_z = z_threshold  # only accept if >= threshold

    timestamps = df.index.tolist()
    p_values = df["p"].tolist()
    z_values = df["z_score"].tolist()

    for i in range(n_periods, len(timestamps)):
        window_start = timestamps[i - n_periods]
        window_end = timestamps[i]
        p_pre = p_values[i - n_periods]
        p_post = p_values[i]
        delta_p = p_post - p_pre
        abs_delta_p = abs(delta_p)

        # Use the max z_score within this window as the window's signal
        window_z = max(
            (z for z in z_values[i - n_periods : i + 1] if not np.isnan(z)),
            default=float("nan"),
        )
        if np.isnan(window_z):
            continue

        if window_z >= best_z:
            best_z = window_z
            best = _WindowResult(
                window_label=window_label,
                window_start=window_start,
                window_end=window_end,
                p_pre=p_pre,
                p_post=p_post,
                delta_p=delta_p,
                abs_delta_p=abs_delta_p,
                z_score=window_z,
            )

    return best
