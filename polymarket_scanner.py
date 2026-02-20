"""
Polymarket Scanner - Simplified workflow for detecting significant market movements

This script:
1. Fetches markets from Polymarket API
2. Filters for finance-related (non-crypto) markets
3. Calculates composite scores for initial filtering
4. Performs deep analysis on top 10%
5. Outputs JSON + CSV for downstream AI agents and Excel review

No agentic/LLM components - pure data pipeline.
"""

import requests
import json
import csv
import math
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# =============================================================================
# CONFIGURABLE WEIGHTS AND THRESHOLDS
# =============================================================================

class ScannerConfig:
    """All configurable parameters in one place for easy tuning."""

    # --- Stage 1: Composite Score Weights (must sum to 1.0) ---
    WEIGHT_MARKET_VOLUME = 0.40    # Larger markets = more signal
    WEIGHT_ODDS_SWING = 0.35       # Core signal of prediction change
    WEIGHT_VOLUME_SURGE = 0.25    # Capital flow confirms conviction

    # --- Stage 1: Thresholds ---
    COMPOSITE_SCORE_CUTOFF = 0.3   # Minimum score to pass initial filter (0-1 scale)
    TOP_PERCENT_FOR_DEEP_ANALYSIS = 0.10  # Top 10% get deep analysis
    MIN_MARKET_VOLUME = 10000      # Minimum volume to consider ($)

    # --- Lookback ---
    LOOKBACK_HOURS = 24            # How far back to analyze

    # --- Finance Keywords (include) ---
    FINANCE_KEYWORDS = [
        'stock', 'stocks', 'share', 'shares', 'equity', 'equities',
        'market', 'markets', 'trading', 'trader', 'trade',
        'price', 'prices', 'valuation',
        'fed', 'federal reserve', 'interest rate', 'rates', 'fomc',
        'inflation', 'cpi', 'ppi', 'gdp', 'recession', 'economy', 'economic',
        'treasury', 'bond', 'bonds', 'yield', 'yields',
        'bank', 'banks', 'banking', 'financial', 'finance',
        'oil', 'gold', 'silver', 'commodity', 'commodities',
        'dollar', 'euro', 'yen', 'currency', 'forex', 'fx',
        'tariff', 'tariffs', 'trade war', 'sanctions',
        's&p', 'sp500', 'dow', 'nasdaq', 'russell',
        'earnings', 'revenue', 'profit', 'dividend',
        'ipo', 'merger', 'acquisition', 'buyout', 'bankruptcy',
        'sec', 'regulation', 'antitrust',
        'unemployment', 'jobs', 'payroll', 'labor',
        'housing', 'mortgage', 'real estate',
        'debt', 'deficit', 'budget', 'fiscal',
        'etf', 'index', 'indices', 'fund', 'funds',
        'bull', 'bear', 'rally', 'crash', 'correction',
        'volatility', 'vix',
    ]

    # --- Crypto Keywords (exclude) ---
    CRYPTO_EXCLUDE_KEYWORDS = [
        'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'token',
        'solana', 'sol', 'dogecoin', 'doge', 'ripple', 'xrp',
        'cardano', 'ada', 'polkadot', 'chainlink', 'avalanche',
        'polygon', 'matic', 'litecoin', 'ltc', 'shiba', 'pepe',
        'memecoin', 'meme coin', 'nft', 'defi', 'blockchain',
        'coinbase', 'binance', 'fdv', 'airdrop', 'stablecoin',
        'usdt', 'usdc', 'tether', 'altcoin', 'web3',
    ]

    # --- Deep Analysis Thresholds ---
    VELOCITY_ACCELERATION_THRESHOLD = 1.5   # Rate increase multiplier to flag
    LIQUIDITY_SHIFT_THRESHOLD = 0.20        # 20% liquidity change to flag
    TIME_DECAY_URGENCY_HOURS = 48           # Markets closing within X hours
    TIME_DECAY_ACTIVITY_MULTIPLIER = 1.5    # Activity must be X times normal
    SPREAD_CHANGE_THRESHOLD = 0.10          # 10% spread change to flag
    VOLUME_WEIGHT_RATIO_THRESHOLD = 2.0     # Price move must have 2x normal volume

    # --- API ---
    POLYMARKET_API_URL = "https://gamma-api.polymarket.com/markets"
    API_PAGE_SIZE = 500

    # --- Output ---
    OUTPUT_DIR = Path(".")

    @classmethod
    def get_weights_summary(cls) -> Dict:
        """Return current weight configuration as dict."""
        return {
            "composite_weights": {
                "market_volume": cls.WEIGHT_MARKET_VOLUME,
                "odds_swing": cls.WEIGHT_ODDS_SWING,
                "volume_surge": cls.WEIGHT_VOLUME_SURGE,
            },
            "thresholds": {
                "composite_score_cutoff": cls.COMPOSITE_SCORE_CUTOFF,
                "top_percent_for_deep_analysis": cls.TOP_PERCENT_FOR_DEEP_ANALYSIS,
                "min_market_volume": cls.MIN_MARKET_VOLUME,
                "lookback_hours": cls.LOOKBACK_HOURS,
            },
            "deep_analysis_thresholds": {
                "velocity_acceleration": cls.VELOCITY_ACCELERATION_THRESHOLD,
                "liquidity_shift_pct": cls.LIQUIDITY_SHIFT_THRESHOLD,
                "time_decay_urgency_hours": cls.TIME_DECAY_URGENCY_HOURS,
                "spread_change_pct": cls.SPREAD_CHANGE_THRESHOLD,
                "volume_weight_ratio": cls.VOLUME_WEIGHT_RATIO_THRESHOLD,
            }
        }


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketSnapshot:
    """A point-in-time snapshot of a market."""
    market_id: str
    timestamp: datetime
    price: float       # mid price (yes_price)
    volume: float
    liquidity: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    open_interest: Optional[float] = None
    n_trades: Optional[int] = None


@dataclass
class MarketData:
    """Enriched market data with computed metrics."""
    # Basic info
    market_id: str
    question: str
    description: str
    resolution_criteria: str
    category: str
    end_date: Optional[str]

    # Current state
    current_price: float
    current_volume: float
    current_liquidity: float

    # Historical data (for analysis)
    price_history: List[Tuple[datetime, float]] = field(default_factory=list)
    volume_history: List[Tuple[datetime, float]] = field(default_factory=list)
    liquidity_history: List[Tuple[datetime, float]] = field(default_factory=list)
    bid_history: List[Tuple[datetime, Optional[float]]] = field(default_factory=list)
    ask_history: List[Tuple[datetime, Optional[float]]] = field(default_factory=list)
    open_interest_history: List[Tuple[datetime, Optional[float]]] = field(default_factory=list)
    n_trades_history: List[Tuple[datetime, Optional[int]]] = field(default_factory=list)

    # Stage 1: Composite score components
    odds_swing_pct: float = 0.0
    volume_surge_pct: float = 0.0
    composite_score: float = 0.0
    passed_initial_filter: bool = False

    # Stage 2: Deep analysis flags
    velocity_flag: bool = False
    velocity_score: float = 0.0
    velocity_detail: str = ""

    liquidity_shift_flag: bool = False
    liquidity_shift_score: float = 0.0
    liquidity_shift_detail: str = ""

    time_decay_urgency_flag: bool = False
    time_decay_score: float = 0.0
    time_decay_detail: str = ""

    spread_flag: bool = False
    spread_score: float = 0.0
    spread_detail: str = ""

    volume_weighted_flag: bool = False
    volume_weighted_score: float = 0.0
    volume_weighted_detail: str = ""

    # Final
    deep_analysis_score: float = 0.0
    flags_triggered: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        # Build unified time_series from snapshot history
        # Use price_history as the alignment axis (all series share the same timestamps)
        dates = [ts.isoformat() for ts, _ in self.price_history]

        def _series(history):
            return [v for _, v in history] if history else []

        def _round_series(history, decimals=4):
            return [round(v, decimals) if v is not None else None for _, v in history] if history else []

        # Mid price = (bid + ask) / 2 if available, else fall back to yes_price
        mid_prices = []
        for i, (ts, yp) in enumerate(self.price_history):
            b = self.bid_history[i][1] if i < len(self.bid_history) else None
            a = self.ask_history[i][1] if i < len(self.ask_history) else None
            if b is not None and a is not None:
                mid_prices.append(round((b + a) / 2, 4))
            else:
                mid_prices.append(round(yp, 4))

        return {
            "market_id": self.market_id,
            "question": self.question,
            "category": self.category,
            "end_date": self.end_date,
            "resolution_criteria": self.resolution_criteria[:1000] if self.resolution_criteria else "",
            "current_price": self.current_price,
            "current_volume": self.current_volume,
            "current_liquidity": self.current_liquidity,
            "time_series": {
                "dates": dates,
                "mid_price": mid_prices,
                "volume": _round_series(self.volume_history, 2),
                "bid": _round_series(self.bid_history, 4),
                "ask": _round_series(self.ask_history, 4),
                "open_interest": _round_series(self.open_interest_history, 2),
                "n_trades": _series(self.n_trades_history),
            },
            "odds_swing_pct": round(self.odds_swing_pct, 4),
            "volume_surge_pct": round(self.volume_surge_pct, 4),
            "composite_score": round(self.composite_score, 4),
            "passed_initial_filter": self.passed_initial_filter,
            "velocity_flag": self.velocity_flag,
            "velocity_score": round(self.velocity_score, 4),
            "velocity_detail": self.velocity_detail,
            "liquidity_shift_flag": self.liquidity_shift_flag,
            "liquidity_shift_score": round(self.liquidity_shift_score, 4),
            "liquidity_shift_detail": self.liquidity_shift_detail,
            "time_decay_urgency_flag": self.time_decay_urgency_flag,
            "time_decay_score": round(self.time_decay_score, 4),
            "time_decay_detail": self.time_decay_detail,
            "spread_flag": self.spread_flag,
            "spread_score": round(self.spread_score, 4),
            "spread_detail": self.spread_detail,
            "volume_weighted_flag": self.volume_weighted_flag,
            "volume_weighted_score": round(self.volume_weighted_score, 4),
            "volume_weighted_detail": self.volume_weighted_detail,
            "deep_analysis_score": round(self.deep_analysis_score, 4),
            "flags_triggered": self.flags_triggered,
        }

    def to_csv_row(self) -> Dict:
        """Convert to flat dict for CSV output."""
        return {
            "market_id": self.market_id,
            "question": self.question[:200] if self.question else "",
            "category": self.category,
            "end_date": self.end_date or "",
            "current_price": round(self.current_price, 4),
            "current_volume": round(self.current_volume, 2),
            "current_liquidity": round(self.current_liquidity, 2),
            "odds_swing_pct": round(self.odds_swing_pct * 100, 2),
            "volume_surge_pct": round(self.volume_surge_pct * 100, 2),
            "composite_score": round(self.composite_score, 4),
            "passed_initial_filter": self.passed_initial_filter,
            "velocity_flag": self.velocity_flag,
            "velocity_score": round(self.velocity_score, 4),
            "liquidity_shift_flag": self.liquidity_shift_flag,
            "liquidity_shift_score": round(self.liquidity_shift_score, 4),
            "time_decay_urgency_flag": self.time_decay_urgency_flag,
            "time_decay_score": round(self.time_decay_score, 4),
            "spread_flag": self.spread_flag,
            "spread_score": round(self.spread_score, 4),
            "volume_weighted_flag": self.volume_weighted_flag,
            "volume_weighted_score": round(self.volume_weighted_score, 4),
            "deep_analysis_score": round(self.deep_analysis_score, 4),
            "flags_triggered": "; ".join(self.flags_triggered),
        }


# =============================================================================
# POLYMARKET API CLIENT
# =============================================================================

class PolymarketClient:
    """Fetch markets from Polymarket API."""

    def __init__(self, config: ScannerConfig = None):
        self.config = config or ScannerConfig()

    def fetch_all_markets(self, max_pages: int = None) -> List[Dict]:
        """Fetch all active markets from Polymarket API."""
        all_markets = []
        offset = 0
        page = 0

        while True:
            params = {
                "limit": self.config.API_PAGE_SIZE,
                "offset": offset,
                "active": "true",
            }

            try:
                response = requests.get(
                    self.config.POLYMARKET_API_URL,
                    params=params,
                    timeout=60
                )
                response.raise_for_status()
                markets = response.json()

                if not markets:
                    break

                all_markets.extend(markets)
                offset += len(markets)
                page += 1
                print(f"    Page {page}: fetched {len(markets)} markets (total: {len(all_markets)})")

                if len(markets) < self.config.API_PAGE_SIZE:
                    break

                if max_pages and page >= max_pages:
                    print(f"    Reached max pages limit ({max_pages})")
                    break

            except Exception as e:
                print(f"API error at offset {offset}: {e}")
                break

        return all_markets

    def is_finance_market(self, market: Dict) -> bool:
        """Check if market is finance-related and not crypto."""
        import re

        text = (
            market.get('question', '') + ' ' +
            (market.get('description') or '')
        ).lower()

        # First check: exclude crypto (word boundary matching to avoid false positives)
        # e.g., "eth" should not match "threatened"
        for kw in self.config.CRYPTO_EXCLUDE_KEYWORDS:
            # Use word boundaries for short keywords to avoid false positives
            if len(kw) <= 4:
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    return False
            else:
                if kw in text:
                    return False

        # Second check: must match finance keywords
        return any(kw in text for kw in self.config.FINANCE_KEYWORDS)

    def filter_finance_markets(self, markets: List[Dict]) -> List[Dict]:
        """Filter to only finance-related, non-crypto markets."""
        return [m for m in markets if self.is_finance_market(m)]


# =============================================================================
# DATABASE CLIENT (for historical data)
# =============================================================================

class SnapshotDB:
    """Interface to the existing SQLite database for historical snapshots."""

    def __init__(self, db_path: str = "prediction_markets.db"):
        self.db_path = db_path

    def get_market_history(
        self,
        market_id: str,
        lookback_hours: int
    ) -> List[MarketSnapshot]:
        """Get historical snapshots for a market within lookback window."""
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT market_id, timestamp, yes_price, volume, liquidity,
                       bid, ask, open_interest, n_trades
                FROM market_snapshots
                WHERE market_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (market_id, cutoff.isoformat()))

            rows = cursor.fetchall()
            conn.close()

            snapshots = []
            for row in rows:
                try:
                    snapshots.append(MarketSnapshot(
                        market_id=str(row[0]),
                        timestamp=datetime.fromisoformat(row[1]) if row[1] else datetime.utcnow(),
                        price=float(row[2]) if row[2] is not None else 0.0,
                        volume=float(row[3]) if row[3] is not None else 0.0,
                        liquidity=float(row[4]) if row[4] is not None else 0.0,
                        bid=float(row[5]) if row[5] is not None else None,
                        ask=float(row[6]) if row[6] is not None else None,
                        open_interest=float(row[7]) if row[7] is not None else None,
                        n_trades=int(row[8]) if row[8] is not None else None,
                    ))
                except (ValueError, TypeError):
                    continue

            return snapshots

        except Exception as e:
            print(f"DB error for market {market_id}: {e}")
            return []

    def get_all_market_ids_with_snapshots(self, lookback_hours: int = None) -> set:
        """Get all market IDs that have snapshots in the lookback window."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if lookback_hours:
                cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
                cursor.execute("""
                    SELECT DISTINCT market_id
                    FROM market_snapshots
                    WHERE timestamp >= ?
                """, (cutoff.isoformat(),))
            else:
                # Get all markets with any snapshots
                cursor.execute("SELECT DISTINCT market_id FROM market_snapshots")

            ids = {str(row[0]) for row in cursor.fetchall()}
            conn.close()
            return ids

        except Exception as e:
            print(f"DB error getting market IDs: {e}")
            return set()

    def get_markets_from_db(self, config: 'ScannerConfig') -> List[Dict]:
        """Load markets directly from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT market_id, question, description, resolution_criteria,
                       category, volume, liquidity, end_date
                FROM markets
            """)

            markets = []
            for row in cursor.fetchall():
                market = {
                    'id': row[0],
                    'question': row[1] or '',
                    'description': row[2] or '',
                    'resolutionCriteria': row[3] or '',
                    'category': row[4] or 'uncategorized',
                    'volume': row[5] or 0,
                    'liquidity': row[6] or 0,
                    'endDate': row[7],
                }
                markets.append(market)

            conn.close()
            return markets

        except Exception as e:
            print(f"DB error loading markets: {e}")
            return []

    def get_market_history_all(self, market_id: str) -> List[MarketSnapshot]:
        """Get ALL historical snapshots for a market (no time filter)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT market_id, timestamp, yes_price, volume, liquidity,
                       bid, ask, open_interest, n_trades
                FROM market_snapshots
                WHERE market_id = ?
                ORDER BY timestamp ASC
            """, (market_id,))

            rows = cursor.fetchall()
            conn.close()

            snapshots = []
            for row in rows:
                try:
                    snapshots.append(MarketSnapshot(
                        market_id=str(row[0]),
                        timestamp=datetime.fromisoformat(row[1]) if row[1] else datetime.utcnow(),
                        price=float(row[2]) if row[2] is not None else 0.0,
                        volume=float(row[3]) if row[3] is not None else 0.0,
                        liquidity=float(row[4]) if row[4] is not None else 0.0,
                        bid=float(row[5]) if row[5] is not None else None,
                        ask=float(row[6]) if row[6] is not None else None,
                        open_interest=float(row[7]) if row[7] is not None else None,
                        n_trades=int(row[8]) if row[8] is not None else None,
                    ))
                except (ValueError, TypeError):
                    continue

            return snapshots

        except Exception as e:
            print(f"DB error for market {market_id}: {e}")
            return []

    def _get_all_market_ids_with_snapshots_old(self, lookback_hours: int) -> set:
        """Get all market IDs that have snapshots in the lookback window."""
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT market_id
                FROM market_snapshots
                WHERE timestamp >= ?
            """, (cutoff.isoformat(),))

            ids = {str(row[0]) for row in cursor.fetchall()}
            conn.close()
            return ids

        except Exception as e:
            print(f"DB error getting market IDs: {e}")
            return set()


# =============================================================================
# STAGE 1: COMPOSITE SCORING
# =============================================================================

class CompositeScorer:
    """Calculate composite scores for initial filtering."""

    def __init__(self, config: ScannerConfig = None):
        self.config = config or ScannerConfig()

    def calculate_odds_swing(self, snapshots: List[MarketSnapshot]) -> float:
        """
        Calculate the maximum odds swing as a percentage.
        Returns value between 0 and 1+ (can exceed 1 for >100% swings).
        """
        if len(snapshots) < 2:
            return 0.0

        prices = [s.price for s in snapshots if s.price > 0]
        if len(prices) < 2:
            return 0.0

        min_price = min(prices)
        max_price = max(prices)

        if min_price == 0:
            return 0.0

        # Relative change from min to max
        swing = (max_price - min_price) / min_price
        return swing

    def calculate_volume_surge(self, snapshots: List[MarketSnapshot]) -> float:
        """
        Calculate volume surge as percentage increase.
        Returns value between 0 and 1+ (can exceed 1 for >100% surges).
        """
        if len(snapshots) < 2:
            return 0.0

        # Compare first and last volume
        first_vol = snapshots[0].volume
        last_vol = snapshots[-1].volume

        if first_vol <= 0:
            return 0.0

        surge = (last_vol - first_vol) / first_vol
        return max(0, surge)  # Only positive surges

    def normalize_volume_score(self, volume: float) -> float:
        """
        Normalize market volume to 0-1 scale using log scaling.
        $10k = ~0.2, $100k = ~0.4, $1M = ~0.6, $10M = ~0.8, $100M = ~1.0
        """
        if volume <= 0:
            return 0.0

        # Log scale: log10(10k)=4, log10(100M)=8
        # Map to 0-1 range
        log_vol = math.log10(max(volume, 1))
        normalized = (log_vol - 4) / 4  # 4 to 8 -> 0 to 1
        return max(0, min(1, normalized))

    def normalize_swing_score(self, swing_pct: float) -> float:
        """
        Normalize odds swing to 0-1 scale.
        10% swing = 0.33, 30% swing = 1.0, >30% capped at 1.0
        """
        # Cap at 30% for normalization
        return min(1.0, swing_pct / 0.30)

    def normalize_surge_score(self, surge_pct: float) -> float:
        """
        Normalize volume surge to 0-1 scale.
        25% surge = 0.5, 50% surge = 1.0, >50% capped at 1.0
        """
        return min(1.0, surge_pct / 0.50)

    def calculate_composite_score(
        self,
        volume: float,
        odds_swing_pct: float,
        volume_surge_pct: float
    ) -> float:
        """
        Calculate weighted composite score.
        Returns value between 0 and 1.
        """
        vol_score = self.normalize_volume_score(volume)
        swing_score = self.normalize_swing_score(odds_swing_pct)
        surge_score = self.normalize_surge_score(volume_surge_pct)

        composite = (
            self.config.WEIGHT_MARKET_VOLUME * vol_score +
            self.config.WEIGHT_ODDS_SWING * swing_score +
            self.config.WEIGHT_VOLUME_SURGE * surge_score
        )

        return composite

    def score_market(self, market_data: MarketData) -> MarketData:
        """Calculate composite score for a single market."""
        # Calculate component metrics
        if market_data.price_history:
            snapshots = [
                MarketSnapshot(
                    market_id=market_data.market_id,
                    timestamp=ts,
                    price=p,
                    volume=0,
                    liquidity=0
                )
                for ts, p in market_data.price_history
            ]
            market_data.odds_swing_pct = self.calculate_odds_swing(snapshots)

        if market_data.volume_history:
            snapshots = [
                MarketSnapshot(
                    market_id=market_data.market_id,
                    timestamp=ts,
                    price=0,
                    volume=v,
                    liquidity=0
                )
                for ts, v in market_data.volume_history
            ]
            market_data.volume_surge_pct = self.calculate_volume_surge(snapshots)

        # Calculate composite
        market_data.composite_score = self.calculate_composite_score(
            market_data.current_volume,
            market_data.odds_swing_pct,
            market_data.volume_surge_pct
        )

        # Check cutoff
        market_data.passed_initial_filter = (
            market_data.composite_score >= self.config.COMPOSITE_SCORE_CUTOFF
        )

        return market_data


# =============================================================================
# STAGE 2: DEEP ANALYSIS
# =============================================================================

class DeepAnalyzer:
    """Perform deep analysis on top markets."""

    def __init__(self, config: ScannerConfig = None):
        self.config = config or ScannerConfig()

    def analyze_velocity(self, market_data: MarketData) -> MarketData:
        """
        Detect rate of change acceleration.
        Compares velocity in recent period vs earlier period.
        """
        if len(market_data.price_history) < 4:
            market_data.velocity_detail = "Insufficient data points"
            return market_data

        prices = market_data.price_history
        mid = len(prices) // 2

        # Early period velocity
        early_prices = [p for _, p in prices[:mid]]
        if len(early_prices) >= 2 and early_prices[0] > 0:
            early_velocity = abs(early_prices[-1] - early_prices[0]) / early_prices[0]
        else:
            early_velocity = 0

        # Recent period velocity
        recent_prices = [p for _, p in prices[mid:]]
        if len(recent_prices) >= 2 and recent_prices[0] > 0:
            recent_velocity = abs(recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        else:
            recent_velocity = 0

        # Calculate acceleration
        if early_velocity > 0:
            acceleration = recent_velocity / early_velocity
        else:
            acceleration = recent_velocity * 10 if recent_velocity > 0 else 0

        market_data.velocity_score = min(1.0, acceleration / 3.0)  # Normalize
        market_data.velocity_flag = acceleration >= self.config.VELOCITY_ACCELERATION_THRESHOLD

        if market_data.velocity_flag:
            market_data.velocity_detail = f"Velocity accelerated {acceleration:.1f}x (early: {early_velocity:.2%}, recent: {recent_velocity:.2%})"
            market_data.flags_triggered.append("VELOCITY")
        else:
            market_data.velocity_detail = f"Velocity ratio: {acceleration:.1f}x"

        return market_data

    def analyze_liquidity_shift(self, market_data: MarketData) -> MarketData:
        """
        Detect significant liquidity changes (smart money indicator).
        """
        if len(market_data.liquidity_history) < 2:
            market_data.liquidity_shift_detail = "Insufficient data points"
            return market_data

        first_liq = market_data.liquidity_history[0][1]
        last_liq = market_data.liquidity_history[-1][1]

        if first_liq <= 0:
            market_data.liquidity_shift_detail = "No baseline liquidity"
            return market_data

        shift_pct = (last_liq - first_liq) / first_liq

        market_data.liquidity_shift_score = min(1.0, abs(shift_pct) / 0.5)  # Normalize
        market_data.liquidity_shift_flag = abs(shift_pct) >= self.config.LIQUIDITY_SHIFT_THRESHOLD

        direction = "increased" if shift_pct > 0 else "decreased"
        if market_data.liquidity_shift_flag:
            market_data.liquidity_shift_detail = f"Liquidity {direction} {abs(shift_pct):.1%} (${first_liq:,.0f} -> ${last_liq:,.0f})"
            market_data.flags_triggered.append("LIQUIDITY_SHIFT")
        else:
            market_data.liquidity_shift_detail = f"Liquidity {direction} {abs(shift_pct):.1%}"

        return market_data

    def analyze_time_decay_urgency(self, market_data: MarketData) -> MarketData:
        """
        Detect markets approaching close with unusual activity.
        """
        if not market_data.end_date:
            market_data.time_decay_detail = "No end date"
            return market_data

        try:
            end_dt = datetime.fromisoformat(market_data.end_date.replace('Z', '+00:00'))
            hours_to_close = (end_dt - datetime.now(end_dt.tzinfo)).total_seconds() / 3600
        except (ValueError, TypeError):
            market_data.time_decay_detail = "Invalid end date format"
            return market_data

        if hours_to_close <= 0:
            market_data.time_decay_detail = "Market already closed"
            return market_data

        if hours_to_close > self.config.TIME_DECAY_URGENCY_HOURS:
            market_data.time_decay_detail = f"{hours_to_close:.0f}h to close (not urgent)"
            return market_data

        # Check if activity is elevated
        if market_data.volume_surge_pct >= (self.config.TIME_DECAY_ACTIVITY_MULTIPLIER - 1):
            market_data.time_decay_urgency_flag = True
            market_data.time_decay_score = min(1.0, (1 / hours_to_close) * market_data.volume_surge_pct)
            market_data.time_decay_detail = f"URGENT: {hours_to_close:.0f}h to close with {market_data.volume_surge_pct:.0%} volume surge"
            market_data.flags_triggered.append("TIME_DECAY_URGENCY")
        else:
            market_data.time_decay_detail = f"{hours_to_close:.0f}h to close, normal activity"

        return market_data

    def analyze_spread(self, market_data: MarketData) -> MarketData:
        """
        Analyze price spread changes (using price volatility as proxy).
        Higher volatility suggests wider effective spreads.
        """
        if len(market_data.price_history) < 3:
            market_data.spread_detail = "Insufficient data points"
            return market_data

        prices = [p for _, p in market_data.price_history]

        # Calculate rolling volatility as spread proxy
        diffs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0

        # Compare early vs recent volatility
        mid = len(diffs) // 2
        early_vol = sum(diffs[:mid]) / max(1, mid)
        recent_vol = sum(diffs[mid:]) / max(1, len(diffs) - mid)

        if early_vol > 0:
            vol_change = (recent_vol - early_vol) / early_vol
        else:
            vol_change = 0

        market_data.spread_score = min(1.0, abs(vol_change) / 0.3)
        market_data.spread_flag = abs(vol_change) >= self.config.SPREAD_CHANGE_THRESHOLD

        direction = "widened" if vol_change > 0 else "narrowed"
        if market_data.spread_flag:
            market_data.spread_detail = f"Effective spread {direction} {abs(vol_change):.0%}"
            market_data.flags_triggered.append("SPREAD_CHANGE")
        else:
            market_data.spread_detail = f"Spread change: {vol_change:+.0%}"

        return market_data

    def analyze_volume_weighted_movement(self, market_data: MarketData) -> MarketData:
        """
        Check if price movements are backed by significant volume.
        High-volume price moves are more meaningful.
        """
        if len(market_data.price_history) < 2 or len(market_data.volume_history) < 2:
            market_data.volume_weighted_detail = "Insufficient data points"
            return market_data

        # Price change
        first_price = market_data.price_history[0][1]
        last_price = market_data.price_history[-1][1]

        if first_price <= 0:
            market_data.volume_weighted_detail = "Invalid price data"
            return market_data

        price_change = abs(last_price - first_price) / first_price

        # Volume relative to baseline
        if market_data.volume_surge_pct > 0:
            # If we have price movement AND high volume, flag it
            volume_weight_ratio = (1 + market_data.volume_surge_pct) * (1 + price_change * 10)

            market_data.volume_weighted_score = min(1.0, volume_weight_ratio / 3.0)
            market_data.volume_weighted_flag = volume_weight_ratio >= self.config.VOLUME_WEIGHT_RATIO_THRESHOLD

            if market_data.volume_weighted_flag:
                market_data.volume_weighted_detail = f"Price move ({price_change:.1%}) backed by {market_data.volume_surge_pct:.0%} volume surge (ratio: {volume_weight_ratio:.1f})"
                market_data.flags_triggered.append("VOLUME_WEIGHTED")
            else:
                market_data.volume_weighted_detail = f"Volume-weight ratio: {volume_weight_ratio:.1f}"
        else:
            market_data.volume_weighted_detail = "No volume surge to weight"

        return market_data

    def calculate_deep_analysis_score(self, market_data: MarketData) -> MarketData:
        """Combine all deep analysis scores."""
        # Weighted by priority: velocity > liquidity > time_decay > spread > volume_weighted
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        scores = [
            market_data.velocity_score,
            market_data.liquidity_shift_score,
            market_data.time_decay_score,
            market_data.spread_score,
            market_data.volume_weighted_score,
        ]

        market_data.deep_analysis_score = sum(w * s for w, s in zip(weights, scores))
        return market_data

    def analyze(self, market_data: MarketData) -> MarketData:
        """Run all deep analysis on a market."""
        market_data = self.analyze_velocity(market_data)
        market_data = self.analyze_liquidity_shift(market_data)
        market_data = self.analyze_time_decay_urgency(market_data)
        market_data = self.analyze_spread(market_data)
        market_data = self.analyze_volume_weighted_movement(market_data)
        market_data = self.calculate_deep_analysis_score(market_data)
        return market_data


# =============================================================================
# MAIN SCANNER
# =============================================================================

class PolymarketScanner:
    """Main scanner orchestrating the full workflow."""

    def __init__(self, config: ScannerConfig = None, db_path: str = "prediction_markets.db"):
        self.config = config or ScannerConfig()
        self.client = PolymarketClient(self.config)
        self.db = SnapshotDB(db_path)
        self.scorer = CompositeScorer(self.config)
        self.analyzer = DeepAnalyzer(self.config)

        self.scan_timestamp = datetime.utcnow().isoformat()
        self.stats = {
            "total_markets_fetched": 0,
            "finance_markets": 0,
            "markets_with_history": 0,
            "passed_initial_filter": 0,
            "deep_analyzed": 0,
            "flags_triggered": {},
        }

    def run(self, max_pages: int = None) -> List[MarketData]:
        """Run the full scanning workflow."""
        print(f"[{self.scan_timestamp}] Starting Polymarket scan...")

        # Step 1: Fetch markets
        print("  Fetching markets from API...")
        raw_markets = self.client.fetch_all_markets(max_pages=max_pages)
        self.stats["total_markets_fetched"] = len(raw_markets)
        print(f"    Fetched {len(raw_markets)} total markets")

        # Step 2: Filter for finance
        print("  Filtering for finance markets...")
        finance_markets = self.client.filter_finance_markets(raw_markets)
        self.stats["finance_markets"] = len(finance_markets)
        print(f"    Found {len(finance_markets)} finance markets")

        # Step 3: Get market IDs with snapshots
        market_ids_with_history = self.db.get_all_market_ids_with_snapshots(
            self.config.LOOKBACK_HOURS
        )

        # Step 4: Build MarketData objects with history
        print("  Loading historical data...")
        market_data_list = []

        for market in finance_markets:
            market_id = str(market.get('id', ''))

            # Get volume
            volume = 0
            try:
                volume = float(market.get('volume', 0) or market.get('volumeNum', 0) or 0)
            except (ValueError, TypeError):
                pass

            # Skip low volume
            if volume < self.config.MIN_MARKET_VOLUME:
                continue

            # Get current price
            price = 0
            try:
                price = float(market.get('outcomePrices', '[0]').strip('[]').split(',')[0])
            except (ValueError, TypeError, IndexError):
                try:
                    price = float(market.get('lastTradePrice', 0) or 0)
                except (ValueError, TypeError):
                    pass

            # Get liquidity
            liquidity = 0
            try:
                liquidity = float(market.get('liquidity', 0) or 0)
            except (ValueError, TypeError):
                pass

            resolution_criteria = (
                market.get('resolutionCriteria') or
                market.get('resolutionSource') or
                market.get('description') or ''
            )

            md = MarketData(
                market_id=market_id,
                question=market.get('question', ''),
                description=market.get('description', ''),
                resolution_criteria=resolution_criteria,
                category=market.get('category', 'uncategorized'),
                end_date=market.get('endDate'),
                current_price=price,
                current_volume=volume,
                current_liquidity=liquidity,
            )

            # Load history if available
            if market_id in market_ids_with_history:
                snapshots = self.db.get_market_history(
                    market_id,
                    self.config.LOOKBACK_HOURS
                )
                if snapshots:
                    md.price_history = [(s.timestamp, s.price) for s in snapshots]
                    md.volume_history = [(s.timestamp, s.volume) for s in snapshots]
                    md.liquidity_history = [(s.timestamp, s.liquidity) for s in snapshots]
                    md.bid_history = [(s.timestamp, s.bid) for s in snapshots]
                    md.ask_history = [(s.timestamp, s.ask) for s in snapshots]
                    md.open_interest_history = [(s.timestamp, s.open_interest) for s in snapshots]
                    md.n_trades_history = [(s.timestamp, s.n_trades) for s in snapshots]

            market_data_list.append(md)

        self.stats["markets_with_history"] = sum(
            1 for m in market_data_list if m.price_history
        )
        print(f"    {len(market_data_list)} markets above volume threshold")
        print(f"    {self.stats['markets_with_history']} have historical data")

        # Step 5: Calculate composite scores
        print("  Calculating composite scores...")
        for md in market_data_list:
            self.scorer.score_market(md)

        passed_filter = [m for m in market_data_list if m.passed_initial_filter]
        self.stats["passed_initial_filter"] = len(passed_filter)
        print(f"    {len(passed_filter)} passed initial filter (score >= {self.config.COMPOSITE_SCORE_CUTOFF})")

        # Step 6: Select top 10% for deep analysis
        passed_filter.sort(key=lambda m: m.composite_score, reverse=True)
        top_count = max(1, int(len(passed_filter) * self.config.TOP_PERCENT_FOR_DEEP_ANALYSIS))
        top_markets = passed_filter[:top_count]
        self.stats["deep_analyzed"] = len(top_markets)
        print(f"  Running deep analysis on top {len(top_markets)} markets...")

        # Step 7: Deep analysis
        for md in top_markets:
            self.analyzer.analyze(md)
            for flag in md.flags_triggered:
                self.stats["flags_triggered"][flag] = self.stats["flags_triggered"].get(flag, 0) + 1

        # Sort by deep analysis score
        top_markets.sort(key=lambda m: m.deep_analysis_score, reverse=True)

        print(f"  Done! {sum(len(m.flags_triggered) for m in top_markets)} total flags triggered")

        return top_markets

    def run_from_db(self) -> List[MarketData]:
        """Run scanning workflow using existing database data (no API calls)."""
        print(f"[{self.scan_timestamp}] Starting Polymarket scan (from database)...")

        # Step 1: Load markets from DB
        print("  Loading markets from database...")
        raw_markets = self.db.get_markets_from_db(self.config)
        self.stats["total_markets_fetched"] = len(raw_markets)
        print(f"    Loaded {len(raw_markets)} markets from DB")

        # Step 2: Filter for finance (non-crypto)
        print("  Filtering for finance markets...")
        finance_markets = self.client.filter_finance_markets(raw_markets)
        self.stats["finance_markets"] = len(finance_markets)
        print(f"    Found {len(finance_markets)} finance markets")

        # Step 3: Get all market IDs with snapshots
        market_ids_with_history = self.db.get_all_market_ids_with_snapshots()

        # Step 4: Build MarketData objects with ALL history
        print("  Loading historical data...")
        market_data_list = []

        for market in finance_markets:
            market_id = str(market.get('id', ''))

            # Skip if no snapshots
            if market_id not in market_ids_with_history:
                continue

            # Get volume from DB
            volume = 0
            try:
                volume = float(market.get('volume', 0) or 0)
            except (ValueError, TypeError):
                pass

            # Skip low volume
            if volume < self.config.MIN_MARKET_VOLUME:
                continue

            # Get liquidity
            liquidity = 0
            try:
                liquidity = float(market.get('liquidity', 0) or 0)
            except (ValueError, TypeError):
                pass

            # Load ALL snapshots for this market
            snapshots = self.db.get_market_history_all(market_id)
            if not snapshots or len(snapshots) < 2:
                continue

            # Use latest snapshot for current values
            latest = snapshots[-1]

            resolution_criteria = (
                market.get('resolutionCriteria') or
                market.get('resolutionSource') or
                market.get('description') or ''
            )

            md = MarketData(
                market_id=market_id,
                question=market.get('question', ''),
                description=market.get('description', ''),
                resolution_criteria=resolution_criteria,
                category=market.get('category', 'uncategorized'),
                end_date=market.get('endDate'),
                current_price=latest.price,
                current_volume=latest.volume,
                current_liquidity=latest.liquidity,
                price_history=[(s.timestamp, s.price) for s in snapshots],
                volume_history=[(s.timestamp, s.volume) for s in snapshots],
                liquidity_history=[(s.timestamp, s.liquidity) for s in snapshots],
                bid_history=[(s.timestamp, s.bid) for s in snapshots],
                ask_history=[(s.timestamp, s.ask) for s in snapshots],
                open_interest_history=[(s.timestamp, s.open_interest) for s in snapshots],
                n_trades_history=[(s.timestamp, s.n_trades) for s in snapshots],
            )

            market_data_list.append(md)

        self.stats["markets_with_history"] = len(market_data_list)
        print(f"    {len(market_data_list)} markets with historical data")

        # Step 5: Calculate composite scores
        print("  Calculating composite scores...")
        for md in market_data_list:
            self.scorer.score_market(md)

        passed_filter = [m for m in market_data_list if m.passed_initial_filter]
        self.stats["passed_initial_filter"] = len(passed_filter)
        print(f"    {len(passed_filter)} passed initial filter (score >= {self.config.COMPOSITE_SCORE_CUTOFF})")

        # Step 6: Select top 10% for deep analysis
        passed_filter.sort(key=lambda m: m.composite_score, reverse=True)
        top_count = max(1, int(len(passed_filter) * self.config.TOP_PERCENT_FOR_DEEP_ANALYSIS))
        top_markets = passed_filter[:top_count]
        self.stats["deep_analyzed"] = len(top_markets)
        print(f"  Running deep analysis on top {len(top_markets)} markets...")

        # Step 7: Deep analysis
        for md in top_markets:
            self.analyzer.analyze(md)
            for flag in md.flags_triggered:
                self.stats["flags_triggered"][flag] = self.stats["flags_triggered"].get(flag, 0) + 1

        # Sort by deep analysis score
        top_markets.sort(key=lambda m: m.deep_analysis_score, reverse=True)

        print(f"  Done! {sum(len(m.flags_triggered) for m in top_markets)} total flags triggered")

        return top_markets

    def get_summary(self) -> Dict:
        """Generate summary of the scan."""
        return {
            "scan_timestamp": self.scan_timestamp,
            "config": self.config.get_weights_summary(),
            "stats": self.stats,
        }

    def output_json(self, markets: List[MarketData], filepath: str = None) -> str:
        """Output results to JSON file."""
        if filepath is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = self.config.OUTPUT_DIR / f"scan_results_{timestamp}.json"

        output = {
            "summary": self.get_summary(),
            "markets": [m.to_dict() for m in markets],
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)

        return str(filepath)

    def output_csv(self, markets: List[MarketData], filepath: str = None) -> str:
        """Output results to CSV file."""
        if filepath is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = self.config.OUTPUT_DIR / f"scan_results_{timestamp}.csv"

        if not markets:
            return str(filepath)

        rows = [m.to_csv_row() for m in markets]
        fieldnames = list(rows[0].keys())

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return str(filepath)

    def print_summary(self, markets: List[MarketData]):
        """Print summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("POLYMARKET SCANNER - RESULTS SUMMARY")
        print("=" * 70)

        print(f"\nScan Timestamp: {summary['scan_timestamp']}")

        print("\n--- Configuration ---")
        weights = summary['config']['composite_weights']
        print(f"  Composite Weights:")
        print(f"    Market Volume: {weights['market_volume']:.0%}")
        print(f"    Odds Swing:    {weights['odds_swing']:.0%}")
        print(f"    Volume Surge:  {weights['volume_surge']:.0%}")

        thresholds = summary['config']['thresholds']
        print(f"  Thresholds:")
        print(f"    Score Cutoff:  {thresholds['composite_score_cutoff']}")
        print(f"    Top % Analyzed:{thresholds['top_percent_for_deep_analysis']:.0%}")
        print(f"    Min Volume:    ${thresholds['min_market_volume']:,}")
        print(f"    Lookback:      {thresholds['lookback_hours']} hours")

        print("\n--- Scan Statistics ---")
        stats = summary['stats']
        print(f"  Total Markets Fetched:  {stats['total_markets_fetched']:,}")
        print(f"  Finance Markets:        {stats['finance_markets']:,}")
        print(f"  With Historical Data:   {stats['markets_with_history']:,}")
        print(f"  Passed Initial Filter:  {stats['passed_initial_filter']:,}")
        print(f"  Deep Analyzed:          {stats['deep_analyzed']:,}")

        if stats['flags_triggered']:
            print("\n--- Flags Triggered ---")
            for flag, count in sorted(stats['flags_triggered'].items(), key=lambda x: -x[1]):
                print(f"  {flag}: {count}")

        if markets:
            print("\n--- Top 5 Markets by Deep Analysis Score ---")
            for i, m in enumerate(markets[:5], 1):
                flags = ", ".join(m.flags_triggered) if m.flags_triggered else "none"
                print(f"\n  {i}. {m.question[:70]}...")
                print(f"     Composite: {m.composite_score:.3f} | Deep: {m.deep_analysis_score:.3f}")
                print(f"     Volume: ${m.current_volume:,.0f} | Price: {m.current_price:.2%}")
                print(f"     Flags: {flags}")

        print("\n" + "=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Scanner - Detect significant market movements")
    parser.add_argument("--db", default="prediction_markets.db", help="Path to SQLite database")
    parser.add_argument("--output-dir", default=".", help="Output directory for results")
    parser.add_argument("--lookback", type=int, default=24, help="Lookback window in hours")
    parser.add_argument("--min-volume", type=int, default=10000, help="Minimum market volume ($)")
    parser.add_argument("--cutoff", type=float, default=0.3, help="Composite score cutoff (0-1)")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only")
    parser.add_argument("--csv-only", action="store_true", help="Output CSV only")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    parser.add_argument("--max-pages", type=int, default=None, help="Max API pages to fetch (for testing)")
    parser.add_argument("--use-db", action="store_true", help="Use existing database data instead of API (faster, uses all historical data)")

    args = parser.parse_args()

    # Configure
    config = ScannerConfig()
    config.OUTPUT_DIR = Path(args.output_dir)
    config.LOOKBACK_HOURS = args.lookback
    config.MIN_MARKET_VOLUME = args.min_volume
    config.COMPOSITE_SCORE_CUTOFF = args.cutoff

    # Run scanner
    scanner = PolymarketScanner(config=config, db_path=args.db)
    if args.use_db:
        results = scanner.run_from_db()
    else:
        results = scanner.run(max_pages=args.max_pages)

    # Output
    if not args.csv_only:
        json_path = scanner.output_json(results)
        if not args.quiet:
            print(f"\nJSON output: {json_path}")

    if not args.json_only:
        csv_path = scanner.output_csv(results)
        if not args.quiet:
            print(f"CSV output: {csv_path}")

    if not args.quiet:
        scanner.print_summary(results)


if __name__ == "__main__":
    main()
