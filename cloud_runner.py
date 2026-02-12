"""
Cloud Runner - Scheduled Polymarket scanning for Railway deployment

This script:
1. Runs on a schedule (default: hourly)
2. Fetches markets from Polymarket API
3. Stores snapshots in SQLite
4. Runs the scanner analysis
5. Uploads results to Cloudflare R2 (S3-compatible)

Environment Variables Required:
- R2_ACCOUNT_ID: Cloudflare account ID
- R2_ACCESS_KEY_ID: R2 access key
- R2_SECRET_ACCESS_KEY: R2 secret key
- R2_BUCKET_NAME: R2 bucket name (default: polymarket-data)

Optional:
- SCAN_INTERVAL_MINUTES: How often to run (default: 60)
- MIN_MARKET_VOLUME: Minimum volume filter (default: 10000)
"""

import os
import sys
import time
import json
import sqlite3
import logging
import gc
from datetime import datetime, timezone
from pathlib import Path

import requests
import schedule

# Memory-efficient settings
MAX_PAGES = int(os.environ.get('MAX_PAGES', 50))  # 50 pages = 25k markets max

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import scanner components
from polymarket_scanner import (
    ScannerConfig,
    PolymarketClient,
    PolymarketScanner,
)


# =============================================================================
# R2 STORAGE CLIENT
# =============================================================================

class R2Storage:
    """Upload files to Cloudflare R2 (S3-compatible)."""

    def __init__(self):
        self.account_id = os.environ.get('R2_ACCOUNT_ID')
        self.access_key = os.environ.get('R2_ACCESS_KEY_ID')
        self.secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
        self.bucket_name = os.environ.get('R2_BUCKET_NAME', 'polymarket-data')

        self.enabled = all([self.account_id, self.access_key, self.secret_key])

        if self.enabled:
            import boto3
            self.client = boto3.client(
                's3',
                endpoint_url=f'https://{self.account_id}.r2.cloudflarestorage.com',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name='auto',
            )
            logger.info(f"R2 storage enabled, bucket: {self.bucket_name}")
        else:
            logger.warning("R2 storage disabled - missing credentials")
            self.client = None

    def upload_file(self, local_path: str, remote_key: str, content_type: str = None):
        """Upload a file to R2."""
        if not self.enabled:
            logger.warning(f"R2 disabled, skipping upload: {remote_key}")
            return None

        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            self.client.upload_file(
                local_path,
                self.bucket_name,
                remote_key,
                ExtraArgs=extra_args
            )
            logger.info(f"Uploaded to R2: {remote_key}")
            return f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{remote_key}"
        except Exception as e:
            logger.error(f"R2 upload failed for {remote_key}: {e}")
            return None

    def upload_string(self, content: str, remote_key: str, content_type: str = 'application/json'):
        """Upload string content directly to R2."""
        if not self.enabled:
            return None

        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=remote_key,
                Body=content.encode('utf-8'),
                ContentType=content_type
            )
            logger.info(f"Uploaded to R2: {remote_key}")
            return remote_key
        except Exception as e:
            logger.error(f"R2 upload failed for {remote_key}: {e}")
            return None


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Manage SQLite database for storing market snapshots."""

    def __init__(self, db_path: str = 'prediction_markets.db'):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS markets (
                market_id TEXT PRIMARY KEY,
                question TEXT,
                description TEXT,
                category TEXT,
                volume REAL,
                liquidity REAL,
                end_date TEXT,
                last_updated TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                timestamp TEXT,
                yes_price REAL,
                volume REAL,
                liquidity REAL,
                status TEXT
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_snapshots_market_id
            ON market_snapshots(market_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
            ON market_snapshots(timestamp)
        ''')

        conn.commit()
        conn.close()
        logger.info("Database tables verified")

    def upsert_market(self, market: dict):
        """Insert or update a market."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO markets
            (market_id, question, description, category, volume, liquidity, end_date, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(market.get('id', '')),
            market.get('question', ''),
            market.get('description', ''),
            market.get('category', 'uncategorized'),
            float(market.get('volume', 0) or 0),
            float(market.get('liquidity', 0) or 0),
            market.get('endDate'),
            datetime.now(timezone.utc).isoformat()
        ))

        conn.commit()
        conn.close()

    def add_snapshot(self, market: dict):
        """Add a market snapshot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Parse price from outcomePrices
        price = 0.0
        try:
            prices_str = market.get('outcomePrices', '[0]')
            price = float(prices_str.strip('[]').split(',')[0])
        except (ValueError, TypeError, IndexError):
            pass

        cursor.execute('''
            INSERT INTO market_snapshots
            (market_id, timestamp, yes_price, volume, liquidity, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(market.get('id', '')),
            datetime.now(timezone.utc).isoformat(),
            price,
            float(market.get('volume', 0) or 0),
            float(market.get('liquidity', 0) or 0),
            'active' if market.get('active') else 'inactive'
        ))

        conn.commit()
        conn.close()

    def cleanup_old_snapshots(self, days: int = 7):
        """Remove snapshots older than X days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = datetime.now(timezone.utc).isoformat()[:10]  # Just date part
        cursor.execute('''
            DELETE FROM market_snapshots
            WHERE date(timestamp) < date(?, '-' || ? || ' days')
        ''', (cutoff, days))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old snapshots")


# =============================================================================
# MAIN RUNNER
# =============================================================================

class CloudRunner:
    """Main orchestrator for cloud-based scanning."""

    def __init__(self):
        self.config = ScannerConfig()
        self.config.MIN_MARKET_VOLUME = int(os.environ.get('MIN_MARKET_VOLUME', 10000))

        self.client = PolymarketClient(self.config)
        self.db = DatabaseManager()
        self.r2 = R2Storage()

        self.scan_interval = int(os.environ.get('SCAN_INTERVAL_MINUTES', 60))

    def fetch_and_store_markets(self):
        """Fetch markets from API and store in database (memory-efficient)."""
        logger.info(f"Fetching markets from Polymarket API (max {MAX_PAGES} pages)...")

        total_fetched = 0
        finance_count = 0
        offset = 0
        page = 0

        try:
            while page < MAX_PAGES:
                # Fetch one page at a time
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

                    total_fetched += len(markets)
                    page += 1

                    # Filter and store immediately (don't keep in memory)
                    for market in markets:
                        if self.client.is_finance_market(market):
                            # Check volume threshold
                            try:
                                volume = float(market.get('volume', 0) or 0)
                            except (ValueError, TypeError):
                                volume = 0

                            if volume >= self.config.MIN_MARKET_VOLUME:
                                self.db.upsert_market(market)
                                self.db.add_snapshot(market)
                                finance_count += 1

                    # Clear the markets list to free memory
                    del markets

                    if page % 10 == 0:
                        logger.info(f"  Page {page}: {total_fetched} fetched, {finance_count} finance markets stored")
                        gc.collect()  # Force garbage collection

                    offset += self.config.API_PAGE_SIZE

                except requests.RequestException as e:
                    logger.warning(f"API error at page {page}: {e}")
                    break

            logger.info(f"Fetched {total_fetched} total markets, stored {finance_count} finance markets")
            gc.collect()
            return finance_count

        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return 0

    def run_analysis(self):
        """Run the scanner analysis and generate outputs."""
        logger.info("Running scanner analysis...")

        try:
            scanner = PolymarketScanner(
                config=self.config,
                db_path=self.db.db_path
            )
            results = scanner.run_from_db()

            # Generate outputs
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            # JSON output
            json_data = {
                "summary": scanner.get_summary(),
                "markets": [m.to_dict() for m in results],
            }
            json_path = f"scan_results_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)

            # CSV output
            csv_path = f"scan_results_{timestamp}.csv"
            if results:
                import csv
                rows = [m.to_csv_row() for m in results]
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            # Also create "latest" versions
            with open('latest_scan.json', 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)

            if results:
                with open('latest_scan.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            logger.info(f"Analysis complete: {len(results)} flagged markets")

            return {
                'json_path': json_path,
                'csv_path': csv_path,
                'results_count': len(results),
                'summary': scanner.get_summary()
            }

        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            return None

    def upload_to_r2(self, analysis_result: dict):
        """Upload results to R2 storage."""
        if not self.r2.enabled:
            logger.info("R2 disabled, skipping upload")
            return

        # Upload latest files (overwrite)
        self.r2.upload_file('latest_scan.json', 'latest_scan.json', 'application/json')
        self.r2.upload_file('latest_scan.csv', 'latest_scan.csv', 'text/csv')

        # Upload timestamped files (archive)
        if analysis_result:
            self.r2.upload_file(
                analysis_result['json_path'],
                f"archive/{analysis_result['json_path']}",
                'application/json'
            )

        # Upload database
        self.r2.upload_file(
            self.db.db_path,
            'prediction_markets.db',
            'application/x-sqlite3'
        )

        logger.info("Uploaded results to R2")

    def run_once(self):
        """Run a single scan cycle."""
        logger.info("=" * 60)
        logger.info(f"Starting scan cycle at {datetime.now(timezone.utc).isoformat()}")
        logger.info("=" * 60)

        # Step 1: Fetch and store
        market_count = self.fetch_and_store_markets()
        gc.collect()  # Free memory after fetch

        if market_count == 0:
            logger.warning("No markets fetched, skipping analysis")
            return

        # Step 2: Run analysis
        result = self.run_analysis()
        gc.collect()  # Free memory after analysis

        # Step 3: Upload to R2
        if result:
            self.upload_to_r2(result)

        # Step 4: Cleanup old data
        self.db.cleanup_old_snapshots(days=7)

        # Final cleanup
        gc.collect()
        logger.info("Scan cycle complete")

    def run_scheduled(self):
        """Run on a schedule (for Railway cron or continuous running)."""
        logger.info(f"Starting scheduled runner (interval: {self.scan_interval} minutes)")

        # Run immediately on start
        self.run_once()

        # Schedule future runs
        schedule.every(self.scan_interval).minutes.do(self.run_once)

        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cloud Runner for Polymarket Scanner")
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, help='Scan interval in minutes')

    args = parser.parse_args()

    runner = CloudRunner()

    if args.interval:
        runner.scan_interval = args.interval

    if args.once:
        runner.run_once()
    else:
        runner.run_scheduled()


if __name__ == '__main__':
    main()
