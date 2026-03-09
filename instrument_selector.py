"""
instrument_selector.py
======================
LLM-driven financial instrument selection for Polymarket markets.

Two-step approach:
  Step 1 (gpt-4o-mini): Brainstorm 7 candidate instruments broadly.
  Step 2 (gpt-4o):      Critically evaluate the 7 candidates, select the best 2–3,
                         and write a 3-sentence justification for each based on the
                         company's business model and specific exposure to the outcome.

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

_MODEL_STEP1 = "gpt-4o-mini"
_MODEL_STEP2 = "gpt-4o-mini"

# Tickers that are known-valid index/macro instruments (no Polygon equity lookup needed)
_KNOWN_VALID: set[str] = {
    "^GSPC", "^TNX", "^VIX", "^DJI", "^IXIC", "^RUT",
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "GDX",
    "USO", "UUP", "TLT", "IEF", "HYG", "LQD",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLRE", "XLP", "XLY", "XLB",
    "DX-Y.NYB", "CL=F", "GC=F",
    "VIXY",
}

# ---------------------------------------------------------------------------
# Step 1 prompt — broad brainstorm, 7 candidates
# ---------------------------------------------------------------------------

_STEP1_SYSTEM_PROMPT = """\
You are a senior equity analyst brainstorming financial instruments that could be correlated \
with a Polymarket prediction market outcome.

Your task: generate exactly 7 candidate US-listed stocks or ETFs that a hedge fund manager \
might consider trading based on the market question's outcome. Cast a wide net — include both \
obvious first-order plays and less obvious second-order candidates.

COVERAGE RULES:
- Always include at least one sector ETF relevant to the market theme.
- If a specific company is named in the question, include its stock.
- Include a mix: individual stocks, sector ETFs, and (if relevant) macro ETFs like TLT, GLD, USO.
- For Fed/central bank questions: include XLF and a major bank stock (JPM, BAC, GS, or C).
- For geopolitical/military questions: include defense (LMT, NOC, RTX) and energy (XOM, CVX).
- For political appointment questions: think broadly — financials, bonds, affected sectors.
- Do NOT pre-screen quality — the goal is breadth, not precision. Even weak candidates are useful.

Use exact US exchange ticker symbols only. No crypto. No hallucinated tickers.

Return ONLY valid JSON (no markdown fences):
{
  "candidates": [
    {"ticker": "JPM", "company_name": "JPMorgan Chase & Co.", "instrument_type": "stock", "one_line": "Large bank directly affected by interest rate policy changes."},
    ...7 total...
  ]
}
"""

# ---------------------------------------------------------------------------
# Step 2 prompt — critical evaluation, select best 2–3 with full justification
# ---------------------------------------------------------------------------

_STEP2_SYSTEM_PROMPT = """\
You are a senior equity analyst critically evaluating a shortlist of financial instruments \
for relevance to a specific Polymarket prediction market outcome.

You will receive a market question and 7 candidate instruments. Your job is to:
1. Reason carefully about each company's business model, revenue sources, and specific \
   exposure to the market outcome — drawing on what you know about each company.
2. Select the 2–3 instruments with the strongest, most direct connection to the outcome.
3. Write a detailed 3-sentence rationale for each selected instrument covering:
   - Sentence 1: What the company does and its core business/revenue model.
   - Sentence 2: Exactly how and why this market outcome specifically affects this company.
   - Sentence 3: An honest assessment of the connection strength and any caveats.
4. Assign alignment and a calibrated confidence score.

CONFIDENCE CALIBRATION — use the FULL range 1–100:
- 90–100: Company is literally named in the question, OR outcome directly determines price.
- 70–89: Strong direct exposure — company's core revenue clearly and materially affected.
- 50–69: Solid sector-level exposure — outcome affects the whole sector this instrument tracks.
- 30–49: Indirect or macro exposure — second-order effects; plausible but non-obvious.
- 10–29: Tenuous — you can construct a story but most traders would not hedge here.
- 1–9: Little to no rational financial connection; essentially guessing.

CRITICAL RULES:
- Use the FULL range. Do NOT cluster scores in 60–80.
- Ask: "Would a quant fund actually delta-hedge a Polymarket position with this instrument?"
- alignment is "with" if price RISES when YES probability rises; "against" if it FALLS.
- Reject weak candidates — it is better to return 2 strong instruments than 3 weak ones.
- Be specific in rationales. Generic phrases like "broadly exposed to macro conditions" are \
  not acceptable — explain the specific mechanism.

Return ONLY valid JSON (no markdown fences):
{
  "instruments": [
    {
      "ticker": "JPM",
      "company_name": "JPMorgan Chase & Co.",
      "instrument_type": "stock",
      "alignment": "with",
      "confidence": 72,
      "rationale": "JPMorgan Chase is the largest US bank by assets, generating the majority of its revenue from net interest income, investment banking fees, and trading operations across global capital markets. A hawkish Fed Chair appointment would sustain elevated interest rates, directly expanding JPM's net interest margin on its $3 trillion loan book and boosting returns on its massive fixed-income portfolio. While the bank has some offsetting exposure through slower loan demand at higher rates, the net effect historically favors JPM in rising-rate environments, making this a high-conviction play."
    }
  ],
  "selection_rationale": "Brief overall summary of why these instruments were chosen over the others."
}
"""


def _get_openai_client() -> OpenAI | None:
    """Return an OpenAI client, or None if API key is missing."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.warning("[instrument_selector] OPENAI_API_KEY not set — skipping LLM selection")
        return None
    return OpenAI(api_key=api_key)


def _parse_json_response(raw: str) -> dict | None:
    """Strip markdown fences and parse JSON, returning None on failure."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception as e:
        log.warning(f"[instrument_selector] JSON parse failed: {e}")
        return None


def _step1_generate_candidates(client: OpenAI, question: str) -> list[dict]:
    """
    Step 1: Brainstorm 7 candidate instruments broadly using gpt-4o-mini.

    Returns a list of candidate dicts with keys: ticker, company_name,
    instrument_type, one_line. Returns empty list on failure.
    """
    try:
        response = client.chat.completions.create(
            model=_MODEL_STEP1,
            messages=[
                {"role": "system", "content": _STEP1_SYSTEM_PROMPT},
                {"role": "user", "content": f"Market question: {question}"},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        result = _parse_json_response(response.choices[0].message.content)
        if not result:
            return []
        candidates = result.get("candidates", [])
        log.debug(
            f"[instrument_selector] Step 1 candidates: "
            f"{[c.get('ticker') for c in candidates]}"
        )
        return candidates
    except Exception as e:
        log.warning(f"[instrument_selector] Step 1 (brainstorm) failed: {e}")
        return []


def _step2_evaluate_candidates(
    client: OpenAI, question: str, candidates: list[dict]
) -> dict | None:
    """
    Step 2: Critically evaluate the 7 candidates and select the best 2–3 using gpt-4o.

    Returns the parsed JSON dict with 'instruments' and 'selection_rationale' keys,
    or None on failure.
    """
    candidates_text = json.dumps(candidates, indent=2)
    user_msg = (
        f"Market question: {question}\n\n"
        f"Candidate instruments to evaluate:\n{candidates_text}"
    )
    try:
        response = client.chat.completions.create(
            model=_MODEL_STEP2,
            messages=[
                {"role": "system", "content": _STEP2_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        return _parse_json_response(response.choices[0].message.content)
    except Exception as e:
        log.warning(f"[instrument_selector] Step 2 (evaluation) failed: {e}")
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
    Select up to 3 financial instruments for a Polymarket market via two-step LLM.

    Step 1 (gpt-4o-mini): Brainstorm 7 broad candidates.
    Step 2 (gpt-4o):      Evaluate candidates, select best 2–3 with 3-sentence justifications.

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

    client = _get_openai_client()
    if not client:
        return InstrumentSelection(
            market_id=market_id,
            market_question=market_question,
            instruments=[],
            selection_rationale="LLM unavailable.",
        )

    # Step 1: brainstorm 7 candidates
    candidates = _step1_generate_candidates(client, market_question)
    if not candidates:
        return InstrumentSelection(
            market_id=market_id,
            market_question=market_question,
            instruments=[],
            selection_rationale="Step 1 (candidate generation) failed.",
        )

    # Step 2: evaluate and select best 2–3 with full justifications
    result = _step2_evaluate_candidates(client, market_question, candidates)
    if not result:
        return InstrumentSelection(
            market_id=market_id,
            market_question=market_question,
            instruments=[],
            selection_rationale="Step 2 (candidate evaluation) failed.",
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
