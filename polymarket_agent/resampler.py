"""
Resamples a 1-minute probability tick series to 5-minute cadence.

Strategy: last observed probability in each 5-minute bucket (not mean),
because Polymarket probabilities are discrete state observations, not flows.
Forward-fill gaps up to 15 minutes; mark longer gaps in data_quality.
"""

from __future__ import annotations

import pandas as pd

from polymarket_agent.models import MarketSnapshot, ProbTick


_RESAMPLE_RULE = "15min"
_MAX_FFILL_PERIODS = 2  # 2 × 15m = 30m forward-fill limit


def resample_to_5m(snapshot: MarketSnapshot) -> MarketSnapshot:
    """
    Replace snapshot.probability_series with a 5-minute resampled version.

    Args:
        snapshot: MarketSnapshot with raw (typically 1m) probability_series.

    Returns:
        New MarketSnapshot with downsampled probability_series.
        Mutates snapshot in-place and returns it for chaining.
    """
    if not snapshot.probability_series:
        return snapshot

    # Build DataFrame from tick list
    df = pd.DataFrame(
        [{"ts_utc": t.ts_utc, "p": t.p} for t in snapshot.probability_series]
    )
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.set_index("ts_utc").sort_index()

    # Resample: last value in each 5m bucket, then forward-fill short gaps
    resampled = (
        df["p"]
        .resample(_RESAMPLE_RULE)
        .last()
        .ffill(limit=_MAX_FFILL_PERIODS)
        .dropna()
    )

    # Track any gaps introduced by resampling
    n_raw = len(df)
    n_resampled = len(resampled)
    n_expected = max(1, (df.index[-1] - df.index[0]) // pd.Timedelta("5min") + 1)
    gap_pct = max(0.0, (n_expected - n_resampled) / n_expected)

    new_series = [
        ProbTick(ts_utc=ts.isoformat(), p=float(p))
        for ts, p in resampled.items()
    ]

    # Update notes
    notes = snapshot.notes.copy()
    notes["resample_cadence"] = _RESAMPLE_RULE
    notes["resample_input_ticks"] = n_raw
    notes["resample_output_ticks"] = n_resampled

    dq = snapshot.data_quality.model_copy()
    if gap_pct > 0.05:
        dq.notes.append(
            f"Warning: {gap_pct:.1%} of 5m buckets missing after resample"
        )
        dq.missing_pct = max(dq.missing_pct, gap_pct)

    snapshot.probability_series = new_series
    snapshot.notes = notes
    snapshot.data_quality = dq
    return snapshot
