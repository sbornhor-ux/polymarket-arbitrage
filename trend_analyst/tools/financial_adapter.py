"""
Financial Market Workflow Adapter
----------------------------------
Converts the Financial Market Workflow JSON output into the list of
FinancialMarketRecord objects that the Trend Analyst expects.

The adapter auto-detects three JSON formats so the Trend Analyst can consume
whatever the Financial Market Workflow saves to R2 or disk without changes to
either side's schema.

──────────────────────────────────────────────────────────────────────────────
SUPPORTED INPUT FORMATS
──────────────────────────────────────────────────────────────────────────────

Format A  ·  Native "pairs" schema (passthrough)
    Already produced by load_financial_data() — adapter is a no-op.

    {
      "pairs": [
        {
          "polymarket_id": "...",
          "ticker": "SPY",
          "ticker_name": "S&P 500 ETF",
          "recommendation_reasoning": "...",
          "time_series": {
            "dates":   ["2026-03-01T00:00:00Z", ...],
            "prices":  [450.0, ...],
            "returns": [0.01, ...],    // optional
            "volumes": [1000000, ...]  // optional
          }
        }
      ]
    }

Format B  ·  Market-centric "markets/tickers" schema
    Each market object has a list of financial ticker series.

    {
      "generated_at": "...",
      "version": "1.0",          // optional
      "markets": [
        {
          "polymarket_id": "...",    // also accepted: "market_id"
          "question": "...",
          "recommendation_reasoning": "...",   // optional
          "tickers": [               // also accepted: "series", "financial_series"
            {
              "ticker": "^TNX",
              "name": "US 10Y Treasury Yield",  // also accepted: "ticker_name"
              "reasoning": "...",               // optional
              "dates": [...],                   // also accepted: "timestamps"
              "prices": [...],                  // also accepted: "close", "adj_close"
              "volumes": [...],                 // optional
              "returns": [...]                  // optional; computed if absent
            }
          ]
        }
      ]
    }

Format C  ·  Flat "results" schema
    A top-level list (or "results" key) of flat records, one per ticker pair.

    {
      "results": [
        {
          "polymarket_id": "...",
          "ticker": "GLD",
          "ticker_name": "SPDR Gold Shares",
          "reasoning": "...",
          "dates": [...],
          "prices": [...],
          "volumes": [...],
          "returns": [...]
        }
      ]
    }

──────────────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────────────

    from tools.financial_adapter import load_financial_workflow_json

    records = load_financial_workflow_json("path/to/financial_output.json")
    # → list[FinancialMarketRecord], ready for build_aligned_series()

    # Or with a URL:
    records = load_financial_workflow_json(
        "https://3164db02212ba104d3623df6c4a26a97.r2.cloudflarestorage.com/"
        "polymarket-data/financial_output.json"
    )
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests

# Allow import from both tools/ and project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.schemas import FinancialMarketRecord

logger = logging.getLogger("trend_analyst.financial_adapter")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_financial_workflow_json(source: str) -> list[FinancialMarketRecord]:
    """
    Load and adapt a Financial Market Workflow JSON file into a list of
    FinancialMarketRecord objects.

    Auto-detects Format A / B / C (see module docstring).  Raises ValueError
    if the format cannot be recognised.

    Args:
        source: Local filepath OR HTTP/HTTPS URL.

    Returns:
        list[FinancialMarketRecord] ready for build_aligned_series().
    """
    data = _fetch(source)
    fmt  = _detect_format(data)
    logger.info("Financial workflow JSON detected as Format %s (%s)", fmt, source)

    if fmt == "A":
        return _parse_format_a(data)
    if fmt == "B":
        return _parse_format_b(data)
    if fmt == "C":
        return _parse_format_c(data)
    if fmt == "D":
        return _parse_format_d(data)

    raise ValueError(
        f"Unrecognised Financial Market Workflow JSON format in '{source}'. "
        "Expected a top-level 'pairs', 'markets', or 'results' key. "
        "See tools/financial_adapter.py docstring for supported schemas."
    )


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_format(data: Any) -> str:
    """Return 'A', 'B', 'C', 'D', or raise ValueError."""
    if isinstance(data, list):
        return "C"                          # bare list
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object or array")

    if "pairs" in data:
        return "A"
    # Format D: markets list + finance_stats swing-level financial records
    if "markets" in data and "finance_stats" in data:
        return "D"
    if "markets" in data:
        return "B"
    if "results" in data:
        return "C"
    # Last-ditch: a dict that looks like a single flat record
    if "polymarket_id" in data and "ticker" in data:
        return "C"

    raise ValueError(
        "Could not detect format: no 'pairs', 'markets', 'finance_stats', or 'results' key found"
    )


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_format_a(data: dict) -> list[FinancialMarketRecord]:
    """Native 'pairs' schema — minimal transformation."""
    records = []
    for p in data.get("pairs", []):
        ts = p.get("time_series", {})
        rec = FinancialMarketRecord(
            polymarket_id=str(p["polymarket_id"]),
            ticker=str(p["ticker"]),
            ticker_name=p.get("ticker_name", p["ticker"]),
            recommendation_reasoning=p.get("recommendation_reasoning", ""),
            dates=ts.get("dates", []),
            prices=ts.get("prices", []),
            returns=ts.get("returns", []),
            volumes=ts.get("volumes", []),
        )
        _maybe_compute_returns(rec)
        records.append(rec)
    logger.debug("Format A: parsed %d records", len(records))
    return records


def _parse_format_b(data: dict) -> list[FinancialMarketRecord]:
    """Market-centric 'markets/tickers' schema."""
    records = []
    for mkt in data.get("markets", []):
        poly_id = str(
            mkt.get("polymarket_id") or mkt.get("market_id", "")
        )
        mkt_reasoning = mkt.get("recommendation_reasoning", "")

        # Accept 'tickers', 'series', or 'financial_series'
        ticker_list = (
            mkt.get("tickers")
            or mkt.get("series")
            or mkt.get("financial_series")
            or []
        )

        for t in ticker_list:
            ticker  = str(t.get("ticker", ""))
            name    = t.get("name") or t.get("ticker_name") or ticker
            reason  = t.get("reasoning") or t.get("recommendation_reasoning") or mkt_reasoning
            dates   = t.get("dates") or t.get("timestamps") or []
            prices  = t.get("prices") or t.get("close") or t.get("adj_close") or []
            volumes = t.get("volumes") or t.get("volume") or []
            returns = t.get("returns") or []

            if not ticker or not prices:
                logger.debug("Skipping empty ticker entry in market %s", poly_id)
                continue

            rec = FinancialMarketRecord(
                polymarket_id=poly_id,
                ticker=ticker,
                ticker_name=str(name),
                recommendation_reasoning=str(reason),
                dates=dates,
                prices=prices,
                returns=returns,
                volumes=volumes if isinstance(volumes, list) else [],
            )
            _maybe_compute_returns(rec)
            records.append(rec)

    logger.debug("Format B: parsed %d records", len(records))
    return records


def _parse_format_c(data: Any) -> list[FinancialMarketRecord]:
    """Flat 'results' list schema (also handles bare JSON array)."""
    rows: list[dict] = []
    if isinstance(data, list):
        rows = data
    else:
        rows = data.get("results", [])
        # Wrap a single flat object
        if isinstance(rows, dict):
            rows = [rows]

    records = []
    for r in rows:
        poly_id = str(r.get("polymarket_id") or r.get("market_id", ""))
        ticker  = str(r.get("ticker", ""))
        if not ticker:
            continue

        dates   = r.get("dates") or r.get("timestamps") or []
        prices  = r.get("prices") or r.get("close") or r.get("adj_close") or []
        volumes = r.get("volumes") or r.get("volume") or []
        returns = r.get("returns") or []
        reason  = r.get("reasoning") or r.get("recommendation_reasoning") or ""
        name    = r.get("ticker_name") or r.get("name") or ticker

        if not prices:
            continue

        rec = FinancialMarketRecord(
            polymarket_id=poly_id,
            ticker=ticker,
            ticker_name=str(name),
            recommendation_reasoning=str(reason),
            dates=dates,
            prices=prices,
            returns=returns,
            volumes=volumes if isinstance(volumes, list) else [],
        )
        _maybe_compute_returns(rec)
        records.append(rec)

    logger.debug("Format C: parsed %d records", len(records))
    return records




def _parse_format_d(data: dict) -> list[FinancialMarketRecord]:
    """
    Format D: markets + finance_stats schema.

    markets[]       -- Polymarket market metadata with financial_relevance_score
                       and financial_relevance_rationale per market.
    finance_stats[] -- per-swing financial records keyed by (polymarket_id, series_id)
                       with time_series {dates, prices, returns, volumes}.

    Multiple swings for the same (polymarket_id, series_id) are merged and
    sorted chronologically. Markets with financial_relevance_score < 0.4 are
    skipped as not financially relevant.
    """
    from collections import defaultdict

    # Build per-market metadata for reasoning + relevance filtering
    market_meta: dict[str, dict] = {}
    for mkt in data.get("markets", []):
        poly_id = str(mkt.get("market_id") or "")
        if poly_id:
            score = float(mkt.get("financial_relevance_score") or 0.0)
            rationale = str(mkt.get("financial_relevance_rationale") or "")
            market_meta[poly_id] = {"score": score, "rationale": rationale}

    # Group swing records by (polymarket_id, series_id)
    groups: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "label": "", "dates": [], "prices": [], "returns": [], "volumes": [],
    })

    skipped_low_relevance = 0
    for item in data.get("finance_stats", []):
        poly_id   = str(item.get("polymarket_id") or "")
        series_id = str(item.get("series_id") or "")
        if not poly_id or not series_id:
            continue

        meta = market_meta.get(poly_id, {})
        if meta.get("score", 1.0) < 0.4:
            skipped_low_relevance += 1
            continue

        ts     = item.get("time_series") or {}
        dates  = ts.get("dates")   or []
        prices = ts.get("prices")  or []
        rets   = ts.get("returns") or []
        vols   = ts.get("volumes") or []

        if not dates or not prices:
            continue

        key = (poly_id, series_id)
        g   = groups[key]
        if not g["label"]:
            g["label"] = str(item.get("series_label") or series_id)

        n = len(dates)
        g["dates"].extend(dates)
        g["prices"].extend(prices[:n])
        g["returns"].extend(rets[:n] if rets else [None] * n)
        g["volumes"].extend(vols[:n] if vols else [None] * n)

    if skipped_low_relevance:
        logger.debug("Format D: skipped %d finance_stats rows (relevance < 0.4)",
                     skipped_low_relevance)

    records = []
    for (poly_id, series_id), g in groups.items():
        combined = sorted(
            zip(g["dates"], g["prices"], g["returns"], g["volumes"]),
            key=lambda x: x[0],
        )
        seen: set[str] = set()
        uniq = []
        for row in combined:
            if row[0] not in seen:
                seen.add(row[0])
                uniq.append(row)

        if not uniq:
            continue

        dates_out   = [r[0] for r in uniq]
        prices_out  = [r[1] for r in uniq if r[1] is not None]
        returns_out = [r[2] for r in uniq if r[2] is not None]
        volumes_out = [r[3] for r in uniq if r[3] is not None]

        meta = market_meta.get(poly_id, {})
        reasoning = ("relevance={:.2f}: {}".format(
            meta.get("score", 0.0), meta.get("rationale", "")
        )).rstrip(": ")

        rec = FinancialMarketRecord(
            polymarket_id=poly_id,
            ticker=series_id,
            ticker_name=g["label"],
            recommendation_reasoning=reasoning,
            dates=dates_out,
            prices=prices_out,
            returns=returns_out,
            volumes=volumes_out,
        )
        _maybe_compute_returns(rec)
        records.append(rec)

    logger.debug("Format D: parsed %d records from finance_stats", len(records))
    return records


def load_polymarket_from_format_d(source: str):
    """
    Extract PolymarketRecord objects from a Format D financial workflow JSON.

    Uses probability_series (5-min, resampled to hourly) when present,
    otherwise falls back to time_series.mid_price (already hourly).

    Args:
        source: Local filepath or HTTP/HTTPS URL to the financial pipeline JSON.

    Returns:
        (summary_dict, list[PolymarketRecord])
    """
    import pandas as pd
    from models.schemas import PolymarketRecord

    data = _fetch(source)
    if not ("markets" in data and "finance_stats" in data):
        raise ValueError(
            "{!r} is not a Format D financial JSON (missing markets/finance_stats)".format(source)
        )

    markets_out = []
    as_of_times = []

    for mkt in data.get("markets", []):
        poly_id = str(mkt.get("market_id") or "")
        if not poly_id:
            continue

        prob_series = mkt.get("probability_series") or []
        ts_data     = mkt.get("time_series") or {}

        poly_dates:  list[str]   = []
        poly_prices: list[float] = []

        if prob_series:
            df = pd.DataFrame(prob_series)
            df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
            df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
            hourly = df["p"].resample("1h").last().ffill().dropna()
            poly_dates  = hourly.index.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
            poly_prices = hourly.values.tolist()
        elif ts_data.get("dates") and ts_data.get("mid_price"):
            poly_dates  = [str(d) for d in ts_data["dates"]]
            poly_prices = [float(p) for p in ts_data["mid_price"]]

        as_of_utc = str(mkt.get("as_of_utc") or "")
        if as_of_utc:
            as_of_times.append(as_of_utc)

        def _sf(key, default=0.0, _mkt=mkt):
            try:
                v = _mkt.get(key)
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        odds_swing = _sf("odds_swing_pct")
        if abs(odds_swing) <= 1.0:
            odds_swing = abs(odds_swing) * 100

        rec = PolymarketRecord(
            market_id=poly_id,
            question=str(mkt.get("market_question") or mkt.get("question") or ""),
            description="",
            category=str(mkt.get("category") or "uncategorized"),
            end_date=str(mkt.get("end_date") or ""),
            current_price=poly_prices[-1] if poly_prices else _sf("current_price"),
            current_volume=_sf("current_volume"),
            current_liquidity=_sf("current_liquidity"),
            odds_swing_pct=odds_swing,
            volume_surge_pct=_sf("velocity_score"),
            composite_score=_sf("deep_analysis_score", default=0.5),
            passed_initial_filter=True,
            deep_analysis_score=_sf("deep_analysis_score"),
            poly_dates=poly_dates,
            poly_prices=poly_prices,
            poly_volumes=[],
            current_as_of=as_of_utc,
        )
        markets_out.append(rec)

    scan_ts = max(as_of_times) if as_of_times else ""
    summary = {
        "scan_timestamp": scan_ts,
        "source": source,
        "total_markets": len(markets_out),
    }

    logger.debug("Format D: extracted %d PolymarketRecords from markets", len(markets_out))
    return summary, markets_out


# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(source: str) -> Any:
    """Load JSON from a URL or local filepath."""
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        return resp.json()
    with open(source, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _maybe_compute_returns(rec: FinancialMarketRecord) -> None:
    """
    Compute log-returns in-place if the record has prices but no returns.
    This ensures alignment with the log-return convention used in build_aligned_series().
    """
    if rec.returns or not rec.prices:
        return
    prices = np.array(rec.prices, dtype=np.float64)
    rec.returns = np.diff(np.log(np.clip(prices, 1e-10, None))).tolist()
