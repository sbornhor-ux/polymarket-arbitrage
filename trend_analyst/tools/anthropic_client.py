"""
LLM helper using the OpenAI API.

Two entry points:
  - generate_agent_summary()   : enrich a completed pair analysis with narrative interpretation
  - select_tests_and_weights() : choose which statistical tests to run and their weights
                                 for a specific Polymarket-ticker pair (per-pair, pre-analysis)

Both fall back silently when OPENAI_API_KEY is not set or the call fails.
"""
from __future__ import annotations

import json
import os
import logging
from typing import Optional, Dict, Any

from config import ACTIVE_TESTS as _ALL_TESTS

logger = logging.getLogger("trend_analyst.llm")

# ── Shared constants ──────────────────────────────────────────────────────────

_SUMMARY_MODEL  = "gpt-4o-mini"
_PLANNER_MODEL  = "gpt-4o-mini"


_TEST_DESCRIPTIONS = """\
Available statistical tests. Each runs on hourly Polymarket probability changes (dP)
vs financial log-returns over the overlapping window:

  correlation
    Pearson + Spearman correlation between Polymarket dP and financial returns.
    Best for: liquid instruments with continuous two-way flow (equities, futures).
    Less useful when: Polymarket is flat most of the window (binary near-resolution).
    Minimum obs: 20.

  granger
    Granger causality F-test (bidirectional): does past Polymarket dP predict future
    financial returns, and vice versa?
    Best for: pairs where one market is expected to lead the other with a lag.
    Minimum obs: 25.

  lead_lag_ccf
    Cross-correlation at hourly lags -10h to +10h. Identifies the optimal lead/lag
    offset and whether Polymarket or the financial instrument leads.
    Best for: all pairs; especially powerful for geopolitical events vs commodities/FX.
    Minimum obs: 20.

  event_study
    Flags hours where |dP| > 3pp as events; measures avg financial return over 1/3/6/12h
    afterward. Tests whether Polymarket spikes systematically precede market moves.
    Best for: markets with multiple moderate spikes (not single binary resolution).
    Minimum obs: 3+ qualifying events.

  spike_event_study
    Compares pre-spike vs post-spike financial returns around each Polymarket spike.
    Directly answers: was the information already priced in BEFORE the Polymarket move?
    Best for: the core price-discovery question; high-weight for geopolitical/macro events.
    Minimum obs: 2+ spikes each with 6h margin on both sides.

  volatility_spillover
    Tests whether rolling Polymarket volatility regimes predict rolling financial volatility.
    Best for: crypto (BTC/ETH), volatile commodities, VIX-related pairs.
    Less useful for: slow-moving bond yields or equity ETFs during calm periods.
    Minimum obs: 15.
"""

# ── Instrument-type inference ─────────────────────────────────────────────────

def _infer_instrument_type(ticker: str) -> str:
    """
    Classify a ticker into a broad instrument type to guide the LLM planner.
    Returns one of: equity_etf, commodity_futures, crypto, bond_yield,
                    volatility_index, currency_futures, unknown.
    """
    t = ticker.upper()
    if t in ("^VIX", "VIX"):
        return "volatility_index"
    if t in ("^TNX", "^TYX", "^FVX", "^IRX"):
        return "bond_yield"
    if t.endswith("-USD") or t in ("BTC", "ETH", "SOL", "XRP"):
        return "crypto"
    if t in ("DX=F", "DX-Y.NYB", "6E=F", "6J=F", "6B=F", "6A=F", "6C=F"):
        return "currency_futures"
    if t.endswith("=F"):
        return "commodity_futures"
    return "equity_etf"


_INSTRUMENT_GUIDANCE = {
    "equity_etf": (
        "Equity ETFs (e.g. SPY, GLD) react to macro news with moderate lag. "
        "Granger and lead_lag_ccf are highly relevant. "
        "Event study is useful if there are multiple spikes. "
        "Volatility spillover is less informative unless the market is very active."
    ),
    "commodity_futures": (
        "Commodity futures (e.g. CL=F oil, GC=F gold) can react very quickly to geopolitical events. "
        "spike_event_study and lead_lag_ccf are the most important tests here — "
        "they directly answer whether futures moved before or after Polymarket. "
        "Granger causality is also highly relevant. "
        "Weight spike_event_study and lead_lag_ccf heavily."
    ),
    "crypto": (
        "Crypto (BTC, ETH) trades 24/7 and reacts to sentiment extremely fast. "
        "Volatility spillover is very relevant. "
        "Lead_lag_ccf and spike_event_study are key to detect if crypto priced in the event. "
        "Correlation can be noisy due to crypto's high baseline volatility unrelated to the event."
    ),
    "bond_yield": (
        "Bond yields (^TNX) move slowly and are primarily driven by Fed policy expectations. "
        "Granger and correlation are most informative. "
        "spike_event_study is useful for major policy events. "
        "Volatility spillover is less relevant for yields."
    ),
    "volatility_index": (
        "VIX responds to uncertainty shocks directly. "
        "Event study and spike_event_study are highly relevant. "
        "Volatility spillover can confirm regime correlation. "
        "Short observation windows are common — skip tests with high minimum obs if needed."
    ),
    "currency_futures": (
        "Currency futures react to geopolitical risk and monetary policy. "
        "Lead_lag_ccf and Granger are key. "
        "Spike_event_study is useful for binary policy events. "
        "Correlation is a good baseline."
    ),
    "unknown": (
        "Instrument type unknown. Use a balanced weighting across all tests "
        "that have sufficient observations."
    ),
}


def _get_client():
    """Return an OpenAI client, or None if the package/key is unavailable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        logger.debug("openai package not installed, skipping LLM calls")
        return None


# ── Agent summary ─────────────────────────────────────────────────────────────

def generate_agent_summary(
    pair_analysis: Dict[str, Any],
    persona: str,
    model: str = _SUMMARY_MODEL,
    max_tokens: int = 1024,
) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI to generate a nuanced agent summary for a completed pair analysis.

    Returns dict with keys: `agent_summary`, `key_findings`, `risk_flags`.
    Returns None if API key not configured or call fails.
    """
    client = _get_client()
    if client is None:
        return None

    schema_instruction = (
        "Respond ONLY with a JSON object (no markdown fences, no preamble) with these keys:\n"
        '  "agent_summary": a 2-3 sentence nuanced interpretation,\n'
        '  "key_findings": array of 3-5 specific findings connecting stats to the real-world event,\n'
        '  "risk_flags": array of up to 3 risks or anomalies the statistical tests might miss.\n'
    )

    user_prompt = (
        "You are given a JSON object with statistical test results for a "
        "Polymarket prediction market vs. equity pair.\n\n"
        "Interpret the results in context of the specific Polymarket question. "
        "Be explicit about confidence levels and distinguish correlation from causation.\n\n"
        f"{schema_instruction}\n"
        "PAIR_ANALYSIS_JSON:\n"
    )
    try:
        user_prompt += json.dumps(pair_analysis, indent=2, default=str)
    except Exception:
        user_prompt += str(pair_analysis)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": persona},
                {"role": "user",   "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        parsed = json.loads(text)
        return {
            "agent_summary": parsed.get("agent_summary", ""),
            "key_findings":  parsed.get("key_findings", []),
            "risk_flags":    parsed.get("risk_flags", []),
        }
    except json.JSONDecodeError:
        # Return raw text as summary if JSON parsing fails
        return {"agent_summary": text.strip(), "key_findings": [], "risk_flags": []}
    except Exception as e:
        logger.warning("LLM summary call failed: %s", e)
        return None


# ── Adaptive test planner ─────────────────────────────────────────────────────

def select_tests_and_weights(
    polymarket_question: str,
    polymarket_price: float,
    ticker: str,
    ticker_name: str,
    n_observations: int,
    category: str = "",
    model: str = _PLANNER_MODEL,
) -> dict | None:
    """
    Ask an LLM to choose which statistical tests to run and assign weights
    for a specific Polymarket–financial instrument pair.

    The LLM receives full context about the question, sector/category, instrument
    type, current probability level, and observation count, and returns per-test
    weights tailored to this specific pair.  Tests with weight 0 are excluded from
    the run entirely.

    Returns dict: {selected_tests: list, weights: dict (sum=1.0), reasoning: str}.
    Returns None on failure; caller falls back to default static weights.
    """
    client = _get_client()
    if client is None:
        return None

    instrument_type = _infer_instrument_type(ticker)
    instrument_guidance = _INSTRUMENT_GUIDANCE.get(instrument_type, _INSTRUMENT_GUIDANCE["unknown"])

    # Interpret probability level
    price_pct = polymarket_price * 100
    if price_pct < 5:
        prob_context = f"{price_pct:.1f}% — near-zero, market almost certainly resolves NO; price moves are likely noisy"
    elif price_pct < 20:
        prob_context = f"{price_pct:.1f}% — unlikely event; spikes may carry outsized information"
    elif price_pct < 45:
        prob_context = f"{price_pct:.1f}% — below-even probability; market is actively pricing risk"
    elif price_pct < 55:
        prob_context = f"{price_pct:.1f}% — highly uncertain (near 50/50); both sides are actively traded"
    elif price_pct < 80:
        prob_context = f"{price_pct:.1f}% — likely event; market tracking resolution path"
    elif price_pct < 95:
        prob_context = f"{price_pct:.1f}% — high probability; remaining uncertainty drives spikes"
    else:
        prob_context = f"{price_pct:.1f}% — near-certain resolution; price series is nearly flat, most tests will find little signal"

    # Observation count context
    if n_observations < 15:
        obs_context = f"{n_observations} hourly obs — very limited; skip tests with high minimum requirements"
    elif n_observations < 25:
        obs_context = f"{n_observations} hourly obs — moderate; skip Granger if fewer than 25"
    else:
        obs_context = f"{n_observations} hourly obs — sufficient for all tests"

    system_prompt = (
        "You are a quantitative analyst designing a bespoke statistical test plan for a "
        "prediction-market vs financial-instrument pair. Your goal is to surface genuine "
        "information-flow and price-discovery signals while ignoring tests that are noisy, "
        "underpowered, or irrelevant for this specific pair. "
        "You must assign a weight between 0.0 and 1.0 to every test in the list. "
        "A weight of 0.0 means the test will NOT be run. Weights for selected tests must sum to 1.0. "
        "Return only valid JSON with no extra commentary."
    )

    user_prompt = f"""\
Assign statistical test weights for the following Polymarket–financial instrument pair.

=== PAIR CONTEXT ===
Polymarket question : {polymarket_question}
Polymarket category : {category or "unknown"}
Current probability : {prob_context}
Financial instrument: {ticker} — {ticker_name}
Instrument type     : {instrument_type.replace("_", " ")}
Data available      : {obs_context}

=== INSTRUMENT-SPECIFIC GUIDANCE ===
{instrument_guidance}

=== AVAILABLE TESTS ===
{_TEST_DESCRIPTIONS}

=== YOUR TASK ===
Assign a weight (0.0–1.0) to each test below. Weights for non-zero tests must sum to 1.0.
Set weight to 0.0 to skip a test entirely (e.g. if insufficient observations, or irrelevant).

Consider:
- Which tests most directly answer "was this information already priced into {ticker}?"
- Which tests are underpowered given the observation count?
- Whether the probability level makes event-detection tests useful (near-flat prices = few events)
- The instrument type guidance above

Respond ONLY with a JSON object:
{{
  "weights": {{
    "correlation": <float 0.0-1.0>,
    "granger": <float 0.0-1.0>,
    "lead_lag_ccf": <float 0.0-1.0>,
    "event_study": <float 0.0-1.0>,
    "spike_event_study": <float 0.0-1.0>,
    "volatility_spillover": <float 0.0-1.0>
  }},
  "reasoning": "<one concise sentence explaining the weighting rationale>"
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=300,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content or ""
        parsed = json.loads(text)

        raw_weights = parsed.get("weights", {})
        reasoning = parsed.get("reasoning", "")

        # Build weight dict: only include valid test names, clamp to [0, 1]
        w_all = {t: max(0.0, float(raw_weights.get(t, 0.0))) for t in _ALL_TESTS}

        # Tests with non-zero weight are the selected tests
        selected = [t for t in _ALL_TESTS if w_all[t] > 0]
        if not selected:
            logger.warning("LLM planner assigned all-zero weights for %s x %s",
                           ticker, polymarket_question[:50])
            return None

        # Normalise selected weights to sum to 1.0
        total = sum(w_all[t] for t in selected)
        w_norm = {t: w_all[t] / total for t in selected}

        logger.info(
            "  LLM weights for %s x %s: %s",
            ticker,
            polymarket_question[:50],
            {t: round(v, 2) for t, v in w_norm.items()},
        )
        return {"selected_tests": selected, "weights": w_norm, "reasoning": reasoning}

    except json.JSONDecodeError as e:
        logger.warning("LLM planner JSON parse error (%s) for %s x %s",
                       e, ticker, polymarket_question[:50])
        return None
    except Exception as e:
        logger.warning("LLM planner call failed: %s", e)
        return None
