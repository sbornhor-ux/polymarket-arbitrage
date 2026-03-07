"""
Tests for polymarket_agent.relevance_screener
"""

import pytest
from polymarket_agent.relevance_screener import score_financial_relevance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score(question: str, category: str = "") -> float:
    s, _ = score_financial_relevance(question, category)
    return s


def rationale(question: str, category: str = "") -> str:
    _, r = score_financial_relevance(question, category)
    return r


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------

def test_returns_tuple_of_float_and_str():
    result = score_financial_relevance("Will something happen?", "US Politics")
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], str)


def test_score_within_bounds():
    for q, cat in [
        ("Will Uzbekistan win the FIFA World Cup?", "Sports / Excluded"),
        ("Will the Fed cut rates by 25bps?", "Fed / Monetary Policy"),
        ("US strikes Iran?", "Geopolitics"),
        ("Will Kim Kardashian win the 2028 US Presidential Election?", "US Politics"),
    ]:
        s = score(q, cat)
        assert 0.0 <= s <= 1.0, f"Score {s} out of bounds for: {q}"


def test_score_rounded_to_4dp():
    s = score("Will the Fed raise interest rates?", "Fed / Monetary Policy")
    assert s == round(s, 4)


# ---------------------------------------------------------------------------
# Category base scoring
# ---------------------------------------------------------------------------

def test_fed_monetary_high_base():
    s = score("Will there be no change in rates?", "Fed / Monetary Policy")
    assert s >= 0.85


def test_geopolitics_medium_base():
    s = score("Will a summit occur?", "Geopolitics")
    assert 0.50 <= s <= 0.85


def test_sports_excluded_very_low():
    s = score("Will the Indiana Pacers win the NBA Finals?", "Sports / Excluded")
    assert s <= 0.10


def test_unknown_category_default():
    s = score("Will something happen?", "")
    assert s == pytest.approx(0.40, abs=0.05)


# ---------------------------------------------------------------------------
# Keyword boosts
# ---------------------------------------------------------------------------

def test_boost_fed_keywords():
    base = score("Will a leader change?", "Geopolitics")
    boosted = score("Will the Fed chair change interest rates?", "Geopolitics")
    assert boosted > base


def test_boost_tariff_keywords():
    base = score("Will a deal be signed?", "US Politics")
    boosted = score("Will the US impose new tariffs on China?", "US Politics")
    assert boosted > base


def test_boost_iran_keyword():
    base = score("Will a country's regime fall?", "Geopolitics")
    boosted = score("Will Iran's regime fall?", "Geopolitics")
    assert boosted > base


def test_boost_energy_keywords():
    base = score("Will a country close a waterway?", "Geopolitics")
    boosted = score("Will Iran close the Strait of Hormuz?", "Geopolitics")
    # Both 'iran' and 'strait of hormuz' should boost
    assert boosted > base


def test_boost_crypto_keywords():
    base = score("Will a token reach $100k?", "")
    boosted = score("Will Bitcoin reach $100k?", "")
    assert boosted > base


# ---------------------------------------------------------------------------
# Keyword penalties
# ---------------------------------------------------------------------------

def test_penalty_sports_nba():
    s = score("Will the Dallas Mavericks win the NBA Finals?", "Sports / Excluded")
    assert s <= 0.05


def test_penalty_tweet_count():
    s = score("Will Elon Musk post 0-19 tweets this week?", "US Politics")
    # tweet penalty should drag it well below base US Politics score (0.50)
    assert s < 0.35


def test_penalty_kardashian():
    s = score("Will Kim Kardashian win the 2028 US Presidential Election?", "US Politics")
    assert s < 0.30


def test_penalty_2028_presidential_long_horizon():
    s_2028 = score("Will Andy Beshear win the 2028 US Presidential Election?", "US Politics")
    s_near = score("Will the Fed cut rates after the March 2026 meeting?", "Fed / Monetary Policy")
    assert s_near > s_2028


# ---------------------------------------------------------------------------
# Realistic market examples from prod data
# ---------------------------------------------------------------------------

def test_fed_rate_decision_high():
    s = score(
        "Will the Fed decrease interest rates by 25 bps after the March 2026 meeting?",
        "Fed / Monetary Policy",
    )
    assert s >= 0.90


def test_iran_strike_high():
    s = score("US strikes Iran by March 31, 2026?", "Geopolitics")
    assert s >= 0.80


def test_strait_of_hormuz_very_high():
    s = score("Will Iran close the Strait of Hormuz by March 31?", "Geopolitics")
    assert s >= 0.90


def test_khamenei_regime_relevant():
    s = score("Khamenei out as Supreme Leader of Iran by March 31?", "Geopolitics")
    assert s >= 0.75


def test_sports_world_cup_irrelevant():
    s = score("Will Uzbekistan win the 2026 FIFA World Cup?", "Sports / Excluded")
    assert s <= 0.05


def test_jesus_christ_low():
    s = score("Will Jesus Christ return before 2027?", "Social / Other")
    assert s <= 0.15


# ---------------------------------------------------------------------------
# Rationale contains useful info
# ---------------------------------------------------------------------------

def test_rationale_contains_base():
    _, r = score_financial_relevance("Will the Fed cut rates?", "Fed / Monetary Policy")
    assert "base=" in r


def test_rationale_mentions_boost_when_keyword_matches():
    _, r = score_financial_relevance("Will Bitcoin reach $100k?", "")
    assert "crypto" in r.lower()


def test_rationale_mentions_penalty_when_triggered():
    _, r = score_financial_relevance(
        "Will Elon Musk post 100 tweets?", "US Politics"
    )
    assert "social" in r.lower() or "celebrity" in r.lower() or "tweet" in r.lower()
