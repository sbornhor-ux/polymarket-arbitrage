"""
Market Matcher - Compares Polymarket prediction movements with real financial market data

Uses Alpha Vantage API to fetch actual market prices and compare them with
detected prediction market movements to identify aligned vs diverged signals.
"""

import sqlite3
import json
import re
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("requests library required. Install with: pip install requests")
    raise

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

ALPHA_VANTAGE_API_KEY = None  # Get a free key at https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
API_CALL_DELAY = 12  # Seconds between calls (free tier: 25 calls/day, 5 calls/minute)
DATABASE_PATH = "prediction_markets.db"

# --------------------------------------------------------------------------
# Keyword-to-ticker mapping
# --------------------------------------------------------------------------

KEYWORD_TICKER_MAP = [
    # Crypto
    (["bitcoin", "btc"], "BTC", "crypto"),
    (["ethereum", "eth", "ether"], "ETH", "crypto"),
    (["solana", "sol"], "SOL", "crypto"),
    (["dogecoin", "doge"], "DOGE", "crypto"),
    (["ripple", "xrp"], "XRP", "crypto"),
    (["cardano", "ada"], "ADA", "crypto"),
    (["crypto", "cryptocurrency"], "BTC", "crypto"),

    # Treasury / rates
    (["fed", "federal reserve", "rate cut", "rate hike", "interest rate",
      "fomc", "monetary policy", "basis points"], "TLT", "treasury"),
    (["treasury", "bond", "yield", "10-year", "10 year"], "TLT", "treasury"),

    # Indices / broad market
    (["s&p", "s&p 500", "sp500", "spy"], "SPY", "stock"),
    (["nasdaq", "qqq", "tech stocks"], "QQQ", "stock"),
    (["dow jones", "dow", "djia"], "DIA", "stock"),
    (["russell", "small cap"], "IWM", "stock"),
    (["stock market", "equities", "wall street"], "SPY", "stock"),

    # Volatility / political uncertainty
    (["trump", "election", "politics", "tariff", "trade war",
      "geopolitical", "sanctions"], "VIX", "volatility"),
    (["vix", "volatility", "fear index"], "VIX", "volatility"),

    # Commodities
    (["gold", "precious metal"], "GLD", "commodity"),
    (["silver"], "SLV", "commodity"),
    (["oil", "crude", "petroleum", "opec", "wti", "brent"], "USO", "commodity"),
    (["natural gas"], "UNG", "commodity"),

    # Forex
    (["dollar", "usd", "dxy"], "UUP", "stock"),
    (["euro", "eur/usd"], "FXE", "stock"),

    # Sectors
    (["bank", "banking", "financial sector"], "XLF", "stock"),
    (["tech", "technology sector"], "XLK", "stock"),
    (["energy", "energy sector"], "XLE", "stock"),
    (["real estate", "housing", "reit"], "VNQ", "stock"),
    (["healthcare", "pharma", "biotech"], "XLV", "stock"),
]

COMPANY_TICKER_MAP = {
    "tesla": "TSLA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "disney": "DIS",
    "boeing": "BA",
    "jpmorgan": "JPM",
    "goldman sachs": "GS",
    "berkshire": "BRK-B",
    "walmart": "WMT",
    "coinbase": "COIN",
    "robinhood": "HOOD",
    "palantir": "PLTR",
    "gamestop": "GME",
    "amc": "AMC",
}

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MarketMatcher")

# --------------------------------------------------------------------------
# MarketMatcher class
# --------------------------------------------------------------------------


class MarketMatcher:
    """
    Compares Polymarket prediction movements with real financial market data
    fetched from Alpha Vantage.
    """

    def __init__(self, db_path: str = DATABASE_PATH, api_key: Optional[str] = None):
        """
        Initialize MarketMatcher.

        Args:
            db_path: Path to the SQLite database.
            api_key: Alpha Vantage API key. Falls back to module-level constant.
        """
        self.db_path = db_path
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        self._last_api_call: float = 0.0
        logger.info("MarketMatcher initialized (db=%s)", db_path)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_comparisons_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movement_id INTEGER,
                question TEXT,
                summary TEXT,
                ticker TEXT,
                asset_type TEXT,
                rationale TEXT,
                prediction_direction TEXT,
                prediction_change_pct REAL,
                market_direction TEXT,
                market_change_pct REAL,
                aligned INTEGER,
                alignment_analysis TEXT,
                api_success INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Ticker identification
    # ------------------------------------------------------------------

    def generate_summary_and_search(
        self, movement: Dict
    ) -> Dict[str, str]:
        """
        Analyse a movement row and return a summary, ticker, asset type, and rationale.

        Args:
            movement: A dict-like row from detected_movements.

        Returns:
            Dict with keys: summary, ticker, asset_type, rationale.
        """
        question = (movement["question"] or "").lower()
        explanation = (movement["explanation"] or "").lower()
        combined = f"{question} {explanation}"

        ticker = None
        asset_type = None
        rationale = None

        # 1. Check company names first (more specific)
        for company, company_ticker in COMPANY_TICKER_MAP.items():
            if company in combined:
                ticker = company_ticker
                asset_type = "stock"
                rationale = f"Question mentions '{company}' which maps to {company_ticker}"
                break

        # 2. Check keyword patterns
        if ticker is None:
            for keywords, kw_ticker, kw_asset in KEYWORD_TICKER_MAP:
                for kw in keywords:
                    if kw in combined:
                        ticker = kw_ticker
                        asset_type = kw_asset
                        rationale = f"Keyword '{kw}' found, mapped to {kw_ticker} ({kw_asset})"
                        break
                if ticker is not None:
                    break

        # 3. Fallback to SPY (broad market proxy)
        if ticker is None:
            ticker = "SPY"
            asset_type = "stock"
            rationale = "No specific ticker identified; using SPY as broad market proxy"

        # Build 2-sentence summary
        direction = "increased" if (movement["price_change_pct"] or 0) > 0 else "decreased"
        pct = abs(movement["price_change_pct"] or 0) * 100
        summary = (
            f"The prediction market for \"{movement['question'][:80]}\" "
            f"{direction} by {pct:.1f}% over {movement.get('time_period_hours', '?')} hours. "
            f"This suggests shifting sentiment that may correlate with {ticker} price movement."
        )

        return {
            "summary": summary,
            "ticker": ticker,
            "asset_type": asset_type,
            "rationale": rationale,
        }

    # ------------------------------------------------------------------
    # Alpha Vantage API
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce minimum delay between API calls."""
        elapsed = time.time() - self._last_api_call
        if elapsed < API_CALL_DELAY:
            wait = API_CALL_DELAY - elapsed
            logger.info("Rate limiting: waiting %.1f seconds", wait)
            time.sleep(wait)
        self._last_api_call = time.time()

    def fetch_alpha_vantage_data(
        self,
        ticker: str,
        asset_type: str,
        start_date: str,
        end_date: str,
    ) -> Dict:
        """
        Fetch price data from Alpha Vantage for the given ticker and date range.

        Args:
            ticker: Symbol to look up (e.g. "SPY", "BTC").
            asset_type: One of "stock", "crypto", "forex", "treasury",
                        "commodity", "volatility".
            start_date: ISO date string for range start.
            end_date: ISO date string for range end.

        Returns:
            Dict with keys: start_price, end_price, change_pct, success, error.
        """
        if not self.api_key:
            logger.error(
                "Alpha Vantage API key not set. "
                "Get a free key at https://www.alphavantage.co/support/#api-key"
            )
            return {"success": False, "error": "API key not configured"}

        # Build request params based on asset type
        params: Dict[str, str] = {"apikey": self.api_key, "outputsize": "compact"}

        if asset_type == "crypto":
            params["function"] = "DIGITAL_CURRENCY_DAILY"
            params["symbol"] = ticker
            params["market"] = "USD"
            ts_key = "Time Series (Digital Currency Daily)"
            price_key = "4a. close (USD)"
        elif asset_type == "forex":
            params["function"] = "FX_DAILY"
            params["from_symbol"] = ticker[:3]
            params["to_symbol"] = ticker[3:] if len(ticker) > 3 else "USD"
            ts_key = "Time Series FX (Daily)"
            price_key = "4. close"
        else:
            # stock, treasury, commodity, volatility all use daily time series
            # (VIX, GLD, TLT, etc. are all ETFs/indices with daily data)
            params["function"] = "TIME_SERIES_DAILY"
            params["symbol"] = ticker
            ts_key = "Time Series (Daily)"
            price_key = "4. close"

        logger.info("Fetching Alpha Vantage data: ticker=%s asset_type=%s", ticker, asset_type)
        self._rate_limit()

        try:
            resp = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            logger.error("Network error fetching %s: %s", ticker, exc)
            return {"success": False, "error": f"Network error: {exc}"}

        # Handle HTTP errors
        if resp.status_code == 429:
            logger.warning("Rate limited (429). Waiting 60s and retrying once.")
            time.sleep(60)
            try:
                resp = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
            except requests.RequestException as exc:
                return {"success": False, "error": f"Retry network error: {exc}"}

        if resp.status_code != 200:
            logger.error("HTTP %d for %s", resp.status_code, ticker)
            return {"success": False, "error": f"HTTP {resp.status_code}"}

        data = resp.json()

        # Check for API error messages
        if "Error Message" in data:
            logger.warning("API error for %s: %s", ticker, data["Error Message"])
            return {"success": False, "error": data["Error Message"]}
        if "Note" in data:
            logger.warning("API note (likely rate limit): %s", data["Note"])
            return {"success": False, "error": data["Note"]}
        if "Information" in data:
            logger.warning("API info: %s", data["Information"])
            return {"success": False, "error": data["Information"]}

        time_series = data.get(ts_key)
        if not time_series:
            logger.warning("No time series data for %s (key=%s)", ticker, ts_key)
            return {"success": False, "error": "No time series data returned"}

        # Parse dates and find nearest available trading days
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "")).date()
            end_dt = datetime.fromisoformat(end_date.replace("Z", "")).date()
        except (ValueError, AttributeError):
            start_dt = datetime.now().date() - timedelta(days=7)
            end_dt = datetime.now().date()

        sorted_dates = sorted(time_series.keys())
        if not sorted_dates:
            return {"success": False, "error": "Empty time series"}

        # Find closest available dates
        start_price = self._find_nearest_price(sorted_dates, time_series, start_dt, price_key)
        end_price = self._find_nearest_price(sorted_dates, time_series, end_dt, price_key)

        if start_price is None or end_price is None:
            return {"success": False, "error": "Could not find price data for date range"}

        change_pct = (end_price - start_price) / start_price if start_price != 0 else 0

        logger.info(
            "%s: start=%.2f end=%.2f change=%.2f%%",
            ticker, start_price, end_price, change_pct * 100,
        )

        return {
            "start_price": start_price,
            "end_price": end_price,
            "change_pct": change_pct,
            "success": True,
            "error": None,
        }

    @staticmethod
    def _find_nearest_price(
        sorted_dates: List[str],
        time_series: Dict,
        target: "datetime.date",
        price_key: str,
    ) -> Optional[float]:
        """Return the closing price on or nearest before *target*."""
        target_str = target.isoformat()
        best_date = None
        for d in sorted_dates:
            if d <= target_str:
                best_date = d
        if best_date is None and sorted_dates:
            best_date = sorted_dates[0]
        if best_date is None:
            return None
        try:
            return float(time_series[best_date][price_key])
        except (KeyError, ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Comparison logic
    # ------------------------------------------------------------------

    def compare_markets(
        self, movement: Dict, ticker: str, asset_type: str
    ) -> Dict:
        """
        Compare a prediction market movement with real market price data.

        Args:
            movement: Row dict from detected_movements.
            ticker: Alpha Vantage symbol.
            asset_type: Asset classification string.

        Returns:
            Dict with prediction_direction, market_direction, aligned, etc.
        """
        start_date = movement.get("movement_start") or movement.get("detected_at", "")
        end_date = movement.get("movement_end") or movement.get("detected_at", "")

        price_data = self.fetch_alpha_vantage_data(ticker, asset_type, start_date, end_date)

        pred_change = movement.get("price_change_pct") or 0
        pred_direction = "bullish" if pred_change > 0 else "bearish"

        if not price_data["success"]:
            return {
                "prediction_direction": pred_direction,
                "prediction_change_pct": pred_change,
                "market_direction": "unknown",
                "market_change_pct": None,
                "aligned": None,
                "alignment_analysis": f"Could not fetch market data: {price_data.get('error')}",
                "api_success": False,
            }

        mkt_change = price_data["change_pct"]
        mkt_direction = "bullish" if mkt_change > 0 else "bearish"

        aligned = pred_direction == mkt_direction

        if aligned:
            analysis = (
                f"ALIGNED: Both prediction market ({pred_direction}, "
                f"{pred_change*100:+.1f}%) and {ticker} ({mkt_direction}, "
                f"{mkt_change*100:+.1f}%) moved in the same direction."
            )
        else:
            analysis = (
                f"DIVERGED: Prediction market is {pred_direction} "
                f"({pred_change*100:+.1f}%) but {ticker} is {mkt_direction} "
                f"({mkt_change*100:+.1f}%). Potential arbitrage signal."
            )

        return {
            "prediction_direction": pred_direction,
            "prediction_change_pct": pred_change,
            "market_direction": mkt_direction,
            "market_change_pct": mkt_change,
            "aligned": aligned,
            "alignment_analysis": analysis,
            "api_success": True,
        }

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_all_movements(self) -> int:
        """
        Process every row in detected_movements: identify a ticker, fetch
        market data, compare, and store results in market_comparisons.

        Returns:
            Number of successfully processed movements.
        """
        conn = self._get_connection()
        self._ensure_comparisons_table(conn)

        rows = conn.execute(
            "SELECT * FROM detected_movements ORDER BY detected_at DESC"
        ).fetchall()

        if not rows:
            logger.warning("No movements found in detected_movements table")
            conn.close()
            return 0

        logger.info("Processing %d movements", len(rows))
        success_count = 0

        for idx, row in enumerate(rows, 1):
            movement = dict(row)
            movement_id = movement.get("id")
            question = movement.get("question", "?")
            logger.info("[%d/%d] Processing: %s", idx, len(rows), question[:60])

            try:
                # Identify ticker
                match_info = self.generate_summary_and_search(movement)

                # Compare with real market
                comparison = self.compare_markets(
                    movement, match_info["ticker"], match_info["asset_type"]
                )

                # Persist
                conn.execute(
                    """
                    INSERT INTO market_comparisons (
                        movement_id, question, summary, ticker, asset_type,
                        rationale, prediction_direction, prediction_change_pct,
                        market_direction, market_change_pct, aligned,
                        alignment_analysis, api_success, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        movement_id,
                        question,
                        match_info["summary"],
                        match_info["ticker"],
                        match_info["asset_type"],
                        match_info["rationale"],
                        comparison["prediction_direction"],
                        comparison["prediction_change_pct"],
                        comparison["market_direction"],
                        comparison["market_change_pct"],
                        1 if comparison["aligned"] else (0 if comparison["aligned"] is not None else None),
                        comparison["alignment_analysis"],
                        1 if comparison["api_success"] else 0,
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
                success_count += 1
                logger.info(
                    "  -> %s %s (%s)",
                    match_info["ticker"],
                    "ALIGNED" if comparison.get("aligned") else "DIVERGED" if comparison.get("aligned") is not None else "NO DATA",
                    match_info["asset_type"],
                )

            except Exception as exc:
                logger.error("Error processing movement %s: %s", movement_id, exc)
                continue

        conn.close()
        logger.info("Processing complete: %d/%d succeeded", success_count, len(rows))
        return success_count

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """
        Generate an HTML report from the market_comparisons table.

        Returns:
            Path to the saved HTML report file.
        """
        conn = self._get_connection()
        self._ensure_comparisons_table(conn)
        rows = conn.execute(
            "SELECT * FROM market_comparisons ORDER BY created_at DESC"
        ).fetchall()
        conn.close()

        comparisons = [dict(r) for r in rows]
        total = len(comparisons)
        aligned = [c for c in comparisons if c["aligned"] == 1]
        diverged = [c for c in comparisons if c["aligned"] == 0]
        failed = [c for c in comparisons if not c["api_success"]]
        succeeded = [c for c in comparisons if c["api_success"]]

        aligned_count = len(aligned)
        diverged_count = len(diverged)
        failed_count = len(failed)
        success_rate = (len(succeeded) / total * 100) if total else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Market Comparison Report</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 20px;
        background: #f5f5f5;
        color: #333;
    }}
    h1 {{
        color: #1a1a2e;
        border-bottom: 3px solid #16213e;
        padding-bottom: 10px;
    }}
    h2 {{
        color: #16213e;
        margin-top: 30px;
    }}
    .stats {{
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        margin: 20px 0;
    }}
    .stat-box {{
        background: white;
        border-radius: 8px;
        padding: 15px 25px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        min-width: 140px;
    }}
    .stat-box .number {{
        font-size: 28px;
        font-weight: bold;
    }}
    .stat-box .label {{
        font-size: 13px;
        color: #666;
        margin-top: 4px;
    }}
    .green {{ color: #27ae60; }}
    .red {{ color: #e74c3c; }}
    .gray {{ color: #95a5a6; }}
    .blue {{ color: #2980b9; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }}
    th {{
        background: #1a1a2e;
        color: white;
        padding: 12px 10px;
        text-align: left;
        font-size: 13px;
    }}
    td {{
        padding: 10px;
        border-bottom: 1px solid #eee;
        font-size: 13px;
    }}
    tr:hover {{
        background: #f8f9fa;
    }}
    .aligned-row {{
        border-left: 4px solid #27ae60;
    }}
    .diverged-row {{
        border-left: 4px solid #e74c3c;
    }}
    .failed-row {{
        border-left: 4px solid #95a5a6;
    }}
    .tag {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
    }}
    .tag-aligned {{ background: #d4edda; color: #155724; }}
    .tag-diverged {{ background: #f8d7da; color: #721c24; }}
    .tag-failed {{ background: #e2e3e5; color: #383d41; }}
    .footer {{
        text-align: center;
        color: #999;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }}
</style>
</head>
<body>
<h1>Market Comparison Report</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div class="stats">
    <div class="stat-box">
        <div class="number blue">{total}</div>
        <div class="label">Total Comparisons</div>
    </div>
    <div class="stat-box">
        <div class="number green">{aligned_count}</div>
        <div class="label">Aligned</div>
    </div>
    <div class="stat-box">
        <div class="number red">{diverged_count}</div>
        <div class="label">Diverged</div>
    </div>
    <div class="stat-box">
        <div class="number gray">{failed_count}</div>
        <div class="label">API Failed</div>
    </div>
    <div class="stat-box">
        <div class="number blue">{success_rate:.0f}%</div>
        <div class="label">API Success Rate</div>
    </div>
</div>
"""

        # Section 1: Aligned
        html += "<h2 class='green'>Aligned Markets</h2>\n"
        if aligned:
            html += self._build_table(aligned, "aligned")
        else:
            html += "<p>No aligned markets found.</p>\n"

        # Section 2: Diverged (arbitrage opportunities)
        html += "<h2 class='red'>Diverged Markets (Potential Arbitrage Opportunities)</h2>\n"
        if diverged:
            html += self._build_table(diverged, "diverged")
        else:
            html += "<p>No diverged markets found.</p>\n"

        # Section 3: Failed
        html += "<h2 class='gray'>Failed API Calls</h2>\n"
        if failed:
            html += self._build_table(failed, "failed")
        else:
            html += "<p>All API calls succeeded.</p>\n"

        html += """
<div class="footer">
    Polymarket Market Matcher | Data from Alpha Vantage
</div>
</body>
</html>"""

        output_path = "market_comparison_report.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Report saved to %s", output_path)
        return output_path

    @staticmethod
    def _build_table(rows: List[Dict], row_class: str) -> str:
        """Build an HTML table for a set of comparison rows."""
        html = """<table>
<tr>
    <th>Question</th>
    <th>Summary</th>
    <th>Ticker</th>
    <th>Pred. Direction</th>
    <th>Pred. Change</th>
    <th>Mkt Direction</th>
    <th>Mkt Change</th>
    <th>Status</th>
</tr>
"""
        for r in rows:
            pred_pct = f"{(r['prediction_change_pct'] or 0)*100:+.1f}%" if r.get("prediction_change_pct") is not None else "N/A"
            mkt_pct = f"{(r['market_change_pct'] or 0)*100:+.1f}%" if r.get("market_change_pct") is not None else "N/A"
            question = (r.get("question") or "")[:60]
            summary = (r.get("summary") or "")[:100]

            if row_class == "aligned":
                tag = '<span class="tag tag-aligned">ALIGNED</span>'
            elif row_class == "diverged":
                tag = '<span class="tag tag-diverged">DIVERGED</span>'
            else:
                tag = '<span class="tag tag-failed">FAILED</span>'

            html += f"""<tr class="{row_class}-row">
    <td>{question}</td>
    <td>{summary}</td>
    <td><strong>{r.get('ticker', '?')}</strong></td>
    <td>{r.get('prediction_direction', '?')}</td>
    <td>{pred_pct}</td>
    <td>{r.get('market_direction', '?')}</td>
    <td>{mkt_pct}</td>
    <td>{tag}</td>
</tr>
"""
        html += "</table>\n"
        return html


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Market Matcher - Compare Polymarket movements with real market data"
    )
    parser.add_argument(
        "--mode",
        choices=["process", "report", "both"],
        default="both",
        help="Operation mode (default: both)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Alpha Vantage API key (overrides constant in file)",
    )

    args = parser.parse_args()

    # Resolve API key
    effective_key = args.api_key or ALPHA_VANTAGE_API_KEY

    if args.mode in ("process", "both") and not effective_key:
        print("\n" + "=" * 60)
        print("Alpha Vantage API key required.")
        print("Get a free key at: https://www.alphavantage.co/support/#api-key")
        print()
        print("Then either:")
        print("  1) Add it to the ALPHA_VANTAGE_API_KEY constant in this file")
        print("  2) Use --api-key YOUR_KEY argument")
        print("=" * 60 + "\n")
        exit(1)

    matcher = MarketMatcher(api_key=effective_key)

    if args.mode in ("process", "both"):
        print("\nProcessing movements...")
        count = matcher.process_all_movements()
        print(f"Successfully processed {count} movements.\n")

    if args.mode in ("report", "both"):
        print("Generating report...")
        path = matcher.generate_report()
        print(f"Report saved to: {path}")
        print("Open this file in a browser to view the formatted report.\n")
