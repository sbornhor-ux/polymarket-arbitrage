"""
finance_agent
=============
Public interface for the Financial Market Workflow.

Usage (from orchestrator):
    from polymarket_agent import fetch_and_interpret
    from finance_agent import get_window_stats

    snapshot = fetch_and_interpret(market_id="...")
    stats = get_window_stats(snapshot.swings, market_id=snapshot.market_id)

    for s in stats:
        print(s.model_dump_json(indent=2))
"""

from finance_agent.fetcher import fetch_series
from finance_agent.stats import compute_window_stats
from finance_agent.models import FinanceSeriesWindowStats, DEFAULT_SERIES
from polymarket_agent.models import SwingEvent


def get_window_stats(
    swing_events: list[SwingEvent],
    series: list[str] | None = None,
    trailing_days: int = 30,
    market_id: str = "",
    benchmark_ticker: str = "^GSPC",
) -> list[FinanceSeriesWindowStats]:
    """
    For each (swing_event × financial_series) pair, compute pre/post move,
    trailing z-score, lead-lag correlation, and full time series.

    Args:
        swing_events: List of SwingEvent objects from polymarket_agent.
        series: yfinance tickers to analyse. Defaults to DEFAULT_SERIES.
        trailing_days: Number of days used for z-score baseline.
        market_id: Polymarket market_id for join key in output records.
        benchmark_ticker: Ticker used for benchmark excess returns (default '^GSPC').

    Returns:
        Flat list of FinanceSeriesWindowStats, one per (swing × series) pair.
    """
    if not swing_events:
        return []

    tickers = series or DEFAULT_SERIES

    # Fetch benchmark series once (for excess return computation)
    benchmark_df = None
    if benchmark_ticker not in tickers:
        benchmark_df = fetch_series(benchmark_ticker, swing_events, trailing_days=trailing_days)

    all_stats: list[FinanceSeriesWindowStats] = []

    for ticker in tickers:
        df = fetch_series(ticker, swing_events, trailing_days=trailing_days)
        if df is None or df.empty:
            continue

        # If this ticker IS the benchmark, reuse its df rather than a separate fetch
        bm_df = df if ticker == benchmark_ticker else benchmark_df

        for swing in swing_events:
            stat = compute_window_stats(
                ticker,
                swing,
                df,
                trailing_days=trailing_days,
                market_id=market_id,
                benchmark_df=bm_df,
                benchmark_ticker=benchmark_ticker,
            )
            if stat is not None:
                all_stats.append(stat)

    return all_stats


__all__ = ["get_window_stats", "FinanceSeriesWindowStats", "DEFAULT_SERIES"]
