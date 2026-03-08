"""
instrument_selector.py
======================
LLM-driven financial instrument selection for Polymarket markets.

For each market question with detected swings, calls OpenAI to select up to 3
financial instruments (ideally 2 US stocks + 1 index/ETF), predict the directional
relationship with the Polymarket outcome, and assign a confidence score.

All selected tickers are validated against Polygon.io before returning.
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import OpenAI

from finance_agent.fetcher import _get_client as _get_polygon_client, POLYGON_TICKER_MAP
from finance_agent.models import InstrumentPrediction, InstrumentSelection
from polymarket_agent.models import SwingEvent

log = logging.getLogger(__name__)

_OPENAI_MODEL = "gpt-4o-mini"

# Tickers that are known-valid index/macro instruments (no Polygon equity lookup needed)
_KNOWN_VALID: set[str] = {
    "^GSPC", "^TNX", "^VIX", "^DJI", "^IXIC", "^RUT",
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "GDX",
    "USO", "UUP", "TLT", "IEF", "HYG", "LQD",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLRE", "XLP", "XLY", "XLB",
    "DX-Y.NYB", "CL=F", "GC=F",
    "VIXY",
}

_SYSTEM_PROMPT = """\
You are a financial analyst specializing in identifying which publicly traded \
US equities and ETFs are most directly affected by specific geopolitical, \
macro-economic, or corporate events.

Given a Polymarket prediction market question, identify up to 3 financial instruments \
that would be most meaningfully impacted if the market resolves YES.

Selection priority:
1. First pick 1–2 specific US-listed stocks (by their exact ticker symbol) of companies \
   directly named or clearly most exposed to this event.
2. Then add 1 broad market index or sector ETF (e.g., SPY, QQQ, XLF, XLE) that captures \
   the sector/macro impact.

Rules:
- Use exact US exchange ticker symbols (e.g., AAPL, MSFT, JPM, XOM, SPY, QQQ, ^GSPC).
- Do NOT use crypto tickers (BTC-USD, ETH-USD, etc.).
- Do NOT hallucinate tickers — only use real, actively traded US securities.
- predicted_direction is "up" if the asset price RISES when Polymarket YES probability rises, \
  "down" if it FALLS.
- confidence is 1–100: how confident are you this instrument moves with the outcome?

Return ONLY valid JSON (no markdown fences) in this exact structure:
{
  "instruments": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "instrument_type": "stock",
      "predicted_direction": "up",
      "confidence": 85,
      "rationale": "Apple is directly mentioned and would benefit from..."
    }
  ],
  "selection_rationale": "Overall reason for these selections."
}
"""


def _call_openai(question: str) -> dict | None:
    """Call OpenAI and return parsed JSON dict, or None on failure."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.warning("[instrument_selector] OPENAI_API_KEY not set — skipping LLM selection")
        return None

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Market question: {question}"},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        log.warning(f"[instrument_selector] OpenAI call failed: {e}")
        return None


def _validate_ticker_polygon(ticker: str) -> bool:
    """
    Return True if the ticker is valid and fetchable on Polygon.

    For known macro symbols, skip the API call.
    For stock tickers, attempt get_ticker_details().
    """
    if ticker in _KNOWN_VALID:
        return True
    # Map through POLYGON_TICKER_MAP first
    polygon_ticker, _ = POLYGON_TICKER_MAP.get(ticker, (ticker, None))
    try:
        client = _get_polygon_client()
        client.get_ticker_details(polygon_ticker)
        return True
    except Exception:
        return False


def select_instruments(
    market_question: str,
    swings: list[SwingEvent],
    market_id: str = "",
) -> InstrumentSelection:
    """
    Select up to 3 financial instruments for a Polymarket market via LLM.

    Args:
        market_question: The Polymarket question text.
        swings: Detected swing events (used to determine whether to run — if empty, skip).
        market_id: Polymarket market ID for the output record.

    Returns:
        InstrumentSelection with validated tickers and LLM predictions.
        Falls back to an empty selection if LLM/Polygon unavailable.
    """
    if not swings:
        return InstrumentSelection(
            market_id=market_id,
            market_question=market_question,
            instruments=[],
            selection_rationale="No swings detected — skipped instrument selection.",
        )

    result = _call_openai(market_question)
    if not result:
        return InstrumentSelection(
            market_id=market_id,
            market_question=market_question,
            instruments=[],
            selection_rationale="LLM unavailable.",
        )

    raw_instruments = result.get("instruments", [])
    selection_rationale = result.get("selection_rationale", "")

    validated: list[InstrumentPrediction] = []
    seen_tickers: set[str] = set()

    for item in raw_instruments[:5]:  # process up to 5 candidates
        ticker = str(item.get("ticker", "")).strip().upper()
        if not ticker or ticker in seen_tickers:
            continue

        # Basic format check: letters, digits, ^, -, =, .
        if not re.match(r'^[\w\^\-\=\.]+$', ticker):
            log.debug(f"[instrument_selector] Skipping malformed ticker: {ticker!r}")
            continue

        # Validate against Polygon
        if not _validate_ticker_polygon(ticker):
            log.warning(f"[instrument_selector] Ticker {ticker!r} failed Polygon validation — dropped")
            continue

        direction = str(item.get("predicted_direction", "up")).lower()
        if direction not in ("up", "down"):
            direction = "up"

        confidence = int(item.get("confidence", 50))
        confidence = max(1, min(100, confidence))

        validated.append(InstrumentPrediction(
            ticker=ticker,
            company_name=str(item.get("company_name", "")),
            instrument_type=str(item.get("instrument_type", "stock")),
            predicted_direction=direction,
            confidence=confidence,
            rationale=str(item.get("rationale", "")),
        ))
        seen_tickers.add(ticker)

        if len(validated) >= 3:
            break

    log.info(
        f"[instrument_selector] {market_question[:60]!r} -> "
        f"{[p.ticker for p in validated]} "
        f"(confidences: {[p.confidence for p in validated]})"
    )

    return InstrumentSelection(
        market_id=market_id,
        market_question=market_question,
        instruments=validated,
        selection_rationale=selection_rationale,
    )
