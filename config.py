"""
Configuration file for Polymarket Monitor Agent
"""

# Database settings
DATABASE_PATH = "prediction_markets.db"

# Polling settings
POLL_INTERVAL_SECONDS = 60  # How often to poll the API
MAX_CALLS_PER_MINUTE = 100  # Rate limit (Polymarket free tier)

# API endpoints (no changes needed, but here for reference)
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Retry and backoff settings
MAX_CONSECUTIVE_ERRORS = 5  # After this many errors, increase backoff
MAX_BACKOFF_SECONDS = 300   # Maximum backoff time (5 minutes)

# Batch settings
MARKETS_BATCH_SIZE = 100    # Number of markets to fetch per request

# Logging settings
LOG_FILE = "polymarket_monitor.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
