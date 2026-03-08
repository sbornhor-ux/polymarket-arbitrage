"""
polymarket_agent.relevance_screener
====================================
Rule-based financial relevance scorer for Polymarket markets.

Returns a score in [0.0, 1.0] indicating how directly an event is likely to
move financial instruments (rates, equities, FX, commodities, crypto).

Design: deterministic, no external calls → safe to use in unit tests without
mocking. Two inputs: the market question text and its category tag.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Action thresholds
# ---------------------------------------------------------------------------
# Tune these to control which markets proceed to finance_agent analysis.

HIGH_RELEVANCE_THRESHOLD   = 0.75  # always run finance analysis
MEDIUM_RELEVANCE_THRESHOLD = 0.40  # run only if a significant swing is also detected
# below MEDIUM → skip (sports, celebrity, etc.)


def classify_relevance(score: float) -> str:
    """
    Map a relevance score to a tier label.

    Returns one of: "high", "medium", "low"
    """
    if score >= HIGH_RELEVANCE_THRESHOLD:
        return "high"
    if score >= MEDIUM_RELEVANCE_THRESHOLD:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Category base scores
# ---------------------------------------------------------------------------

_CATEGORY_BASE: dict[str, float] = {
    "fed / monetary policy":      0.90,
    "earnings / corporate events": 0.82,
    "m&a / ipo":                  0.78,
    "financial markets":          0.65,
    "economics / macro":          0.70,
    "geopolitics":                0.65,
    "us politics":                0.50,
    "social / other":             0.15,
    "sports / excluded":          0.05,
}

_DEFAULT_BASE = 0.40  # uncategorized / unknown


# ---------------------------------------------------------------------------
# Keyword rules: (regex pattern, delta, label)
# Applied to the lowercased question text.
# Each matching rule contributes its delta once.
# ---------------------------------------------------------------------------

_BOOSTS: list[tuple[str, float, str]] = [
    # Fed / monetary policy
    (
        r"\b(federal reserve|fed chair|interest rate|rate hike|rate cut"
        r"|monetary policy|basis point|bps|inflation|cpi|pce|gdp)\b",
        0.30,
        "Fed/monetary keywords",
    ),
    # Tariff / trade policy
    (
        r"\b(tariffs?|trade war|trade deal|sanctions|export ban|import ban)\b",
        0.25,
        "tariff/trade keywords",
    ),
    # Earnings / corporate events
    (
        r"\b(earnings|eps|revenue|quarterly results|profit|guidance|dividend"
        r"|stock split|share repurchase|buyback|analyst upgrade|price target)\b",
        0.25,
        "earnings/corporate keywords",
    ),
    # M&A / IPO
    (
        r"\b(merger|acquisition|acquire|takeover|buyout|ipo|initial public offering"
        r"|go public|spac|spin.?off|divestiture|antitrust|tender offer)\b",
        0.25,
        "M&A/IPO keywords",
    ),
    # Energy / commodities
    (
        r"\b(oil|crude|opec|energy|natural gas|strait of hormuz|lng)\b",
        0.20,
        "energy/commodity keywords",
    ),
    # Currency / forex
    (
        r"\b(dollar|dxy|forex|currency|yuan|renminbi|ruble|yen|euro)\b",
        0.20,
        "currency/FX keywords",
    ),
    # Equity / credit / macro
    (
        r"\b(stock market|s&p|recession|debt ceiling|yield|treasury bond"
        r"|nasdaq|equity markets)\b",
        0.20,
        "equity/macro keywords",
    ),
    # Geopolitical risk with direct financial knock-on
    (
        r"\b(war|military strike|attack|invasion|coup|nuclear|regime)\b",
        0.15,
        "geopolitical risk keywords",
    ),
    # Iran specifically (oil supply / Strait of Hormuz impact); matches iran/irani/iranian
    (r"\birani(an)?\b|\biran\b", 0.15, "Iran (oil risk)"),
]

_PENALTIES: list[tuple[str, float, str]] = [
    # Sports — no financial market connection
    (
        r"\b(nba|nfl|mlb|nhl|fifa|world cup|super bowl|finals|playoff"
        r"|championship|cricket|t20|rugby|tennis|golf|soccer)\b",
        -0.30,
        "sports keywords",
    ),
    # Crypto — excluded from scope (pure crypto markets, not macro crypto)
    (
        r"\b(bitcoin|btc|ethereum|eth|solana|dogecoin|ripple|xrp|cardano"
        r"|nft|defi|altcoin|memecoin|stablecoin|blockchain wallet|airdrop)\b",
        -0.35,
        "crypto-specific keywords",
    ),
    # Social / celebrity / entertainment
    (
        r"\b(kardashian|tweet|tweets|retweet|jesus christ|aliens exist|pete hegseth ban"
        r"|reality tv|celebrity|award show|oscars|grammy|grammy award)\b",
        -0.30,
        "social/celebrity keywords",
    ),
    # Long-horizon 2028 political races — minimal near-term market impact
    (r"2028 (us )?presidential", -0.10, "2028 presidential (long horizon)"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_financial_relevance(
    question: str,
    category: str,
) -> tuple[float, str]:
    """
    Score a Polymarket market's relevance to financial markets.

    Args:
        question: The market question text (e.g. "Will the Fed cut rates?").
        category: The market category tag (e.g. "Fed / Monetary Policy").

    Returns:
        (score, rationale) where:
          - score ∈ [0.0, 1.0], rounded to 4 decimal places
          - rationale is a concise human-readable explanation of the score
    """
    q_lower = question.lower()
    cat_lower = category.lower().strip()

    base = _CATEGORY_BASE.get(cat_lower, _DEFAULT_BASE)
    factors: list[str] = [f"base={base:.2f} ({category or 'uncategorized'})"]

    delta = 0.0

    for pattern, boost, label in _BOOSTS:
        if re.search(pattern, q_lower):
            delta += boost
            factors.append(f"+{boost:.2f} {label}")

    for pattern, penalty, label in _PENALTIES:
        if re.search(pattern, q_lower):
            delta += penalty
            factors.append(f"{penalty:.2f} {label}")

    score = round(max(0.0, min(1.0, base + delta)), 4)
    rationale = "; ".join(factors)
    return score, rationale
