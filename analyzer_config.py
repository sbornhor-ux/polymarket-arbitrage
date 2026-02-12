"""
Configuration for Market Movement Analyzer
"""

# Email settings
ALERT_EMAIL = "sbornhor@uchicago.edu"
ALERT_TIME_CST = "16:00"  # 4 PM CST

# SMTP settings (for sending emails)
# Option 1: Use Gmail
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Option 2: Use other providers
# SMTP_SERVER = "smtp.outlook.com"  # For Outlook
# SMTP_PORT = 587

# Email credentials (IMPORTANT: Keep these secure!)
# Leave as None to save reports to files instead of sending emails
SENDER_EMAIL = None  # e.g., "your-email@gmail.com"
SENDER_PASSWORD = None  # e.g., "your-app-password"

# Note: For Gmail, you'll need to use an "App Password" not your regular password
# Generate one at: https://myaccount.google.com/apppasswords

# Analysis thresholds
ODDS_SWING_THRESHOLD = 0.30  # 30% change triggers alert
ODDS_SWING_WINDOW_HOURS = 6  # Look for swings within 6 hours

CAPITAL_FLOW_THRESHOLD_PCT = 0.50  # 50% volume increase triggers alert
CAPITAL_FLOW_WINDOW_HOURS = 6  # Look for flows within 6 hours
CAPITAL_FLOW_MIN_DOLLARS = 1000  # Minimum dollar amount to consider

# Minimum market volume to consider for any movement detection
# Markets below this volume are ignored as insignificant
MIN_MARKET_VOLUME = 4000  # $4,000 minimum volume

IGNORE_CLOSE_HOURS = 6  # Ignore movements within 6 hours of market close

# Snapshot settings
SNAPSHOT_INTERVAL_HOURS = 1  # Take snapshot every hour

# Database
DATABASE_PATH = "prediction_markets.db"

# Logging
LOG_FILE = "movement_analyzer.log"
LOG_LEVEL = "INFO"
