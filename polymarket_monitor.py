"""
Polymarket Platform Monitor Agent

Responsibilities:
- Poll Polymarket API on schedule
- Handle platform-specific authentication/rate limiting
- Parse platform-specific data formats
- Detect and report API errors or downtime
- Maintain platform-specific state (last poll time, error counts)
- Output: Normalized market data to central database
"""

import time
import requests
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('polymarket_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarketStatus(Enum):
    """Market status enumeration"""
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


@dataclass
class NormalizedMarket:
    """Normalized market data structure for database storage"""
    # Platform identifiers
    platform: str
    market_id: str
    condition_id: str
    
    # Market metadata
    question: str
    description: str
    category: str
    status: str
    
    # Pricing data
    yes_price: Optional[float]
    no_price: Optional[float]
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    no_bid: Optional[float]
    no_ask: Optional[float]
    
    # Volume and liquidity
    volume: float
    liquidity: float
    
    # Timing
    start_date: str
    end_date: str
    last_updated: str
    
    # Additional metadata
    tags: str  # JSON string of tags
    outcomes: str  # JSON string of outcomes
    raw_data: str  # Full raw API response for reference


class RateLimiter:
    """Token bucket rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        
    def can_proceed(self) -> bool:
        """Check if a call can proceed without exceeding rate limit"""
        now = time.time()
        # Remove calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        
        return len(self.calls) < self.max_calls
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        while not self.can_proceed():
            logger.info("Rate limit reached, waiting...")
            time.sleep(1)
        
        self.calls.append(time.time())


class PolymarketMonitor:
    """
    Platform Monitor Agent for Polymarket
    """
    
    # API endpoints
    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"
    
    def __init__(
        self,
        db_path: str = "prediction_markets.db",
        poll_interval: int = 60,  # seconds
        max_calls_per_minute: int = 100
    ):
        """
        Initialize Polymarket Monitor
        
        Args:
            db_path: Path to SQLite database
            poll_interval: Time between polls in seconds
            max_calls_per_minute: Rate limit for API calls
        """
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.rate_limiter = RateLimiter(max_calls_per_minute, 60)
        
        # State tracking
        self.last_poll_time: Optional[datetime] = None
        self.error_count = 0
        self.consecutive_errors = 0
        self.total_markets_fetched = 0
        self.platform_status = "unknown"
        
        # Initialize database
        self._init_database()
        
        # Load state from database
        self._load_state()
        
        logger.info("Polymarket Monitor initialized")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Markets table - normalized market data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                market_id TEXT NOT NULL,
                condition_id TEXT,
                question TEXT NOT NULL,
                description TEXT,
                category TEXT,
                status TEXT,
                yes_price REAL,
                no_price REAL,
                yes_bid REAL,
                yes_ask REAL,
                no_bid REAL,
                no_ask REAL,
                volume REAL,
                liquidity REAL,
                start_date TEXT,
                end_date TEXT,
                last_updated TEXT,
                tags TEXT,
                outcomes TEXT,
                raw_data TEXT,
                UNIQUE(platform, market_id)
            )
        """)
        
        # State table - agent state tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                platform TEXT PRIMARY KEY,
                last_poll_time TEXT,
                error_count INTEGER,
                consecutive_errors INTEGER,
                total_markets_fetched INTEGER,
                platform_status TEXT
            )
        """)
        
        # Error log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                error_type TEXT,
                error_message TEXT,
                endpoint TEXT,
                response_code INTEGER
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_id ON markets(market_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON markets(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_updated ON markets(last_updated)")
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def _load_state(self):
        """Load agent state from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT last_poll_time, error_count, consecutive_errors, 
                   total_markets_fetched, platform_status
            FROM agent_state WHERE platform = 'polymarket'
        """)
        
        result = cursor.fetchone()
        if result:
            last_poll_str, self.error_count, self.consecutive_errors, \
                self.total_markets_fetched, self.platform_status = result
            
            if last_poll_str:
                self.last_poll_time = datetime.fromisoformat(last_poll_str)
        
        conn.close()
        logger.info(f"State loaded - Last poll: {self.last_poll_time}, Errors: {self.error_count}")
    
    def _save_state(self):
        """Save agent state to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO agent_state 
            (platform, last_poll_time, error_count, consecutive_errors, 
             total_markets_fetched, platform_status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'polymarket',
            self.last_poll_time.isoformat() if self.last_poll_time else None,
            self.error_count,
            self.consecutive_errors,
            self.total_markets_fetched,
            self.platform_status
        ))
        
        conn.commit()
        conn.close()
    
    def _log_error(self, error_type: str, error_message: str, 
                   endpoint: str = "", response_code: int = None):
        """Log error to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO error_log (timestamp, error_type, error_message, endpoint, response_code)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            error_type,
            error_message,
            endpoint,
            response_code
        ))
        
        conn.commit()
        conn.close()
        
        self.error_count += 1
        self.consecutive_errors += 1
        logger.error(f"{error_type}: {error_message}")
    
    def _api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make rate-limited API request with error handling
        
        Args:
            endpoint: Full API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response or None on error
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            
            if response.status_code == 200:
                self.consecutive_errors = 0  # Reset on success
                self.platform_status = "online"
                return response.json()
            
            elif response.status_code == 429:
                # Rate limited
                self._log_error(
                    "RateLimitError",
                    "API rate limit exceeded",
                    endpoint,
                    429
                )
                self.platform_status = "rate_limited"
                time.sleep(60)  # Back off for 1 minute
                return None
            
            else:
                self._log_error(
                    "HTTPError",
                    f"HTTP {response.status_code}: {response.text}",
                    endpoint,
                    response.status_code
                )
                self.platform_status = "error"
                return None
                
        except requests.exceptions.Timeout:
            self._log_error("TimeoutError", "Request timeout", endpoint)
            self.platform_status = "timeout"
            return None
            
        except requests.exceptions.ConnectionError as e:
            self._log_error("ConnectionError", str(e), endpoint)
            self.platform_status = "offline"
            return None
            
        except Exception as e:
            self._log_error("UnknownError", str(e), endpoint)
            self.platform_status = "error"
            return None
    
    def fetch_markets(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Fetch markets from Polymarket Gamma API
        
        Args:
            limit: Number of markets to fetch
            offset: Offset for pagination
            
        Returns:
            List of market data dictionaries
        """
        endpoint = f"{self.GAMMA_API_BASE}/markets"
        params = {
            'limit': limit,
            'offset': offset,
            'active': 'true',  # Only fetch active markets
            'closed': 'false'  # Exclude closed markets
        }
        
        logger.info(f"Fetching markets (limit={limit}, offset={offset})")
        response = self._api_request(endpoint, params)
        
        if response and isinstance(response, list):
            logger.info(f"Successfully fetched {len(response)} markets")
            return response
        
        return []
    
    def fetch_market_prices(self, token_id: str) -> Optional[Dict]:
        """
        Fetch current prices for a specific market
        
        Args:
            token_id: Token ID for the market
            
        Returns:
            Price data dictionary or None
        """
        endpoint = f"{self.CLOB_API_BASE}/price"
        params = {'token_id': token_id}
        
        return self._api_request(endpoint, params)
    
    def _normalize_market_data(self, raw_market: Dict) -> NormalizedMarket:
        """
        Convert raw Polymarket API response to normalized format
        
        Args:
            raw_market: Raw market data from API
            
        Returns:
            NormalizedMarket object
        """
        # Extract outcomes and prices from API response
        # API returns outcomes as JSON string e.g. '["Yes", "No"]'
        # and prices in outcomePrices e.g. '["0.65", "0.35"]'
        outcomes_raw = raw_market.get('outcomes', '[]')
        if isinstance(outcomes_raw, str):
            outcomes = json.loads(outcomes_raw)
        else:
            outcomes = outcomes_raw

        prices_raw = raw_market.get('outcomePrices', '[]')
        if isinstance(prices_raw, str):
            prices = json.loads(prices_raw)
        else:
            prices = prices_raw if prices_raw else []

        yes_price = None
        no_price = None

        for i, outcome in enumerate(outcomes):
            price = float(prices[i]) if i < len(prices) else 0
            if outcome == 'Yes':
                yes_price = price
            elif outcome == 'No':
                no_price = price
        
        # Determine status
        status = MarketStatus.ACTIVE.value
        if raw_market.get('closed'):
            status = MarketStatus.CLOSED.value
        elif raw_market.get('resolved'):
            status = MarketStatus.RESOLVED.value
        elif raw_market.get('archived'):
            status = MarketStatus.ARCHIVED.value
        
        return NormalizedMarket(
            platform='polymarket',
            market_id=raw_market.get('id', ''),
            condition_id=raw_market.get('conditionId', raw_market.get('condition_id', '')),
            question=raw_market.get('question', ''),
            description=raw_market.get('description', ''),
            category=raw_market.get('category', 'uncategorized'),
            status=status,
            yes_price=yes_price,
            no_price=no_price,
            yes_bid=None,  # Would need CLOB API for bid/ask
            yes_ask=None,
            no_bid=None,
            no_ask=None,
            volume=float(raw_market.get('volume', 0)),
            liquidity=float(raw_market.get('liquidity', 0)),
            start_date=raw_market.get('startDateIso', raw_market.get('start_date_iso', '')),
            end_date=raw_market.get('endDateIso', raw_market.get('end_date_iso', '')),
            last_updated=datetime.now().isoformat(),
            tags=json.dumps(raw_market.get('tags', [])),
            outcomes=json.dumps(outcomes_raw if isinstance(outcomes_raw, list) else outcomes),
            raw_data=json.dumps(raw_market)
        )
    
    def save_markets(self, markets: List[NormalizedMarket]):
        """
        Save normalized markets to database
        
        Args:
            markets: List of NormalizedMarket objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for market in markets:
            market_dict = asdict(market)
            
            cursor.execute("""
                INSERT OR REPLACE INTO markets (
                    platform, market_id, condition_id, question, description,
                    category, status, yes_price, no_price, yes_bid, yes_ask,
                    no_bid, no_ask, volume, liquidity, start_date, end_date,
                    last_updated, tags, outcomes, raw_data
                ) VALUES (
                    :platform, :market_id, :condition_id, :question, :description,
                    :category, :status, :yes_price, :no_price, :yes_bid, :yes_ask,
                    :no_bid, :no_ask, :volume, :liquidity, :start_date, :end_date,
                    :last_updated, :tags, :outcomes, :raw_data
                )
            """, market_dict)
        
        conn.commit()
        conn.close()
        
        self.total_markets_fetched += len(markets)
        logger.info(f"Saved {len(markets)} markets to database")
    
    # Finance-related keywords for filtering markets
    FINANCE_KEYWORDS = [
        'fed ', 'federal reserve', 'interest rate', 'inflation', 'cpi',
        'gdp', 'recession', 'stock', 's&p', 'nasdaq', 'dow jones',
        'treasury', 'bond', 'yield', 'forex', 'dollar', 'euro',
        'ipo', 'earnings', 'revenue', 'profit', 'market cap',
        'tariff', 'trade war', 'sanctions', 'debt ceiling',
        'unemployment', 'jobs report', 'nonfarm', 'payroll',
        'oil', 'gold', 'commodity', 'futures',
        'bank', 'sec ', 'regulation', 'monetary policy',
        'rate cut', 'rate hike', 'quantitative', 'stimulus',
        'default', 'credit', 'hedge fund', 'etf',
        'price', 'valuation', 'ipo', 'merger', 'acquisition',
        'budget', 'deficit', 'surplus', 'fiscal',
    ]

    CRYPTO_EXCLUDE_KEYWORDS = [
        'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'token',
        'solana', 'sol', 'dogecoin', 'doge', 'ripple', 'xrp',
        'cardano', 'ada', 'polkadot', 'chainlink', 'avalanche',
        'polygon', 'matic', 'litecoin', 'ltc', 'shiba', 'pepe',
        'memecoin', 'meme coin', 'nft', 'defi', 'blockchain',
        'coinbase', 'binance', 'fdv', 'airdrop',
    ]

    def _is_finance_market(self, market: Dict) -> bool:
        """Check if a market is finance-related and not crypto."""
        text = (market.get('question', '') + ' ' + market.get('description', '')).lower()
        if any(kw in text for kw in self.CRYPTO_EXCLUDE_KEYWORDS):
            return False
        return any(kw in text for kw in self.FINANCE_KEYWORDS)

    def poll_once(self) -> bool:
        """
        Execute one poll cycle

        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting poll cycle")

        try:
            # Fetch markets in batches
            all_markets = []
            offset = 0
            batch_size = 100

            while True:
                raw_markets = self.fetch_markets(limit=batch_size, offset=offset)

                if not raw_markets:
                    break

                # Filter to finance-related markets only
                finance_markets = [m for m in raw_markets if self._is_finance_market(m)]

                # Normalize market data
                normalized = [self._normalize_market_data(m) for m in finance_markets]
                all_markets.extend(normalized)
                
                # Check if we've fetched all markets
                if len(raw_markets) < batch_size:
                    break
                
                offset += batch_size
            
            if all_markets:
                self.save_markets(all_markets)
                self.last_poll_time = datetime.now()
                self._save_state()
                logger.info(f"Poll completed successfully - {len(all_markets)} markets processed")
                return True
            else:
                logger.warning("No markets fetched in poll cycle")
                return False
                
        except Exception as e:
            self._log_error("PollError", str(e))
            return False
    
    def run(self, max_iterations: int = None):
        """
        Run continuous polling loop
        
        Args:
            max_iterations: Maximum number of iterations (None for infinite)
        """
        logger.info(f"Starting continuous monitoring (poll interval: {self.poll_interval}s)")
        
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            success = self.poll_once()
            
            # Exponential backoff on consecutive errors
            if not success:
                backoff_time = min(300, self.poll_interval * (2 ** min(self.consecutive_errors, 5)))
                logger.warning(f"Poll failed, backing off for {backoff_time}s")
                time.sleep(backoff_time)
            else:
                time.sleep(self.poll_interval)
            
            iteration += 1
        
        logger.info("Monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status
        
        Returns:
            Status dictionary
        """
        return {
            'platform': 'polymarket',
            'last_poll_time': self.last_poll_time.isoformat() if self.last_poll_time else None,
            'platform_status': self.platform_status,
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'total_markets_fetched': self.total_markets_fetched,
            'uptime': (datetime.now() - self.last_poll_time).total_seconds() 
                      if self.last_poll_time else None
        }


if __name__ == "__main__":
    # Example usage
    monitor = PolymarketMonitor(
        db_path="prediction_markets.db",
        poll_interval=60,  # Poll every minute
        max_calls_per_minute=100
    )
    
    # Run for a limited time (e.g., 5 iterations for testing)
    # For production, use monitor.run() with no arguments
    monitor.run(max_iterations=5)
    
    # Print status
    status = monitor.get_status()
    print("\n=== Agent Status ===")
    for key, value in status.items():
        print(f"{key}: {value}")
