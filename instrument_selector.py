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
You are a senior equity analyst specializing in event-driven trading. Your job is to identify \
which US-listed stocks and ETFs are most meaningfully affected by specific prediction market outcomes, \
and to be brutally honest about how confident you actually are in each connection.

Given a Polymarket prediction market question, select up to 3 financial instruments that a \
rational hedge fund manager would actually consider trading based on this market's outcome.

SELECTION PRIORITY:
1. If a specific company is named in the question, pick that company's stock first (highest confidence).
2. Pick 1–2 other stocks/ETFs of companies with clear, direct economic exposure to the outcome.
3. Add 1 sector ETF only if a meaningful sector rotation would follow the outcome.
4. If no instrument has a plausible direct connection, still return your best guess but assign \
   very low confidence (5–25) and say so explicitly in the rationale.

ALWAYS return 2–3 instruments unless the market is a pure celebrity/entertainment question with \
absolutely no financial connection. One instrument is rarely enough — look harder for the 2nd and 3rd.

SPECIFIC GUIDANCE BY MARKET TYPE:
- Federal Reserve / central bank appointments (Fed Chair, FOMC, monetary policy): ALWAYS include \
  XLF (financial sector ETF) AND at least one major bank stock (JPM, BAC, GS, or C). Optionally \
  add TLT or IEF if the appointee has known views on interest rates.
- Treasury Secretary / fiscal policy appointments: Include XLF, SPY, and TLT (government bonds).
- Military / geopolitical events (Iran, NATO, strikes): LMT, NOC, or RTX (defense); XOM or CVX \
  (oil) if the region is an oil producer; IEF or GLD as safe-haven.
- Political election markets (US candidates): SPY is acceptable as a macro proxy, but also include \
  a sector ETF that the candidate's platform most directly affects (e.g. clean energy for Democrats \
  → ICLN; deregulation → XLF; healthcare policy → XLV).

CONFIDENCE CALIBRATION — use the FULL range 1–100, not just 60–90:
- 90–100: Company is literally named in the question, OR the outcome directly determines this \
  instrument's price (e.g. Fed rate decision → TLT/IEF directly priced by rate levels).
- 70–89: Strong direct exposure — the company's core revenue is clearly and materially affected \
  (e.g. Strait of Hormuz closure → XOM, CVX lose access to key supply routes).
- 50–69: Solid sector-level exposure — the outcome affects a whole sector this stock dominates \
  (e.g. Iran military escalation → defense sector ETF XLI).
- 30–49: Indirect or macro exposure — second-order effects; a plausible but non-obvious link \
  (e.g. US political appointment → broad financials XLF with no named company).
- 10–29: Tenuous connection — you can construct a story but most traders would not hedge here \
  (e.g. a geopolitical event → a tech stock only loosely exposed to the region).
- 1–9: Little to no rational financial connection. You are essentially guessing. Assign these \
  scores for celebrity, sports, entertainment, or political questions with no clear market impact \
  (e.g. "Will Tom Brady win the Republican nomination?" → any stock is a stretch; score 5–8 max).

CRITICAL RULES:
- Do NOT cluster scores in 60–80. Use the full range. A score of 15 or 85 is often more \
  accurate than 65.
- Ask yourself: "Would a quant fund actually delta-hedge a Polymarket position with this \
  instrument?" If clearly not, confidence must be below 30.
- Use exact US exchange ticker symbols (AAPL, JPM, XOM, SPY, XLF). No crypto tickers.
- Do NOT hallucinate tickers — only use real, actively traded US securities.
- alignment is "with" if the instrument price RISES when Polymarket YES probability rises; \
  "against" if it FALLS when YES probability rises.
- Write a specific, honest rationale. If confidence is low, explicitly state why the connection \
  is weak or speculative.

Return ONLY valid JSON (no markdown fences) in this exact structure:
{
  "instruments": [
    {
      "ticker": "JPM",
      "company_name": "JPMorgan Chase & Co.",
      "instrument_type": "stock",
      "alignment": "with",
      "confidence": 35,
      "rationale": "A dovish Fed nominee could compress net interest margins for banks; \
JPM has some exposure but no specific company is named in the question and the link \
is indirect — most rate sensitivity is already priced in."
    }
  ],
  "selection_rationale": "Overall reasoning including any caveats about weak connections."
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

        # Accept both old "predicted_direction" (up/down) and new "alignment" (with/against)
        direction = str(item.get("alignment") or item.get("predicted_direction", "with")).lower()
        if direction in ("up",):
            direction = "with"
        elif direction in ("down",):
            direction = "against"
        if direction not in ("with", "against"):
            direction = "with"

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
