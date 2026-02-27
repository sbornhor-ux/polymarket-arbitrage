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
from datetime import datetime, timedelta, timezone
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

    def download_file(self, remote_key: str, local_path: str) -> bool:
        """Download a file from R2. Returns True if successful."""
        if not self.enabled:
            return False
        try:
            self.client.download_file(self.bucket_name, remote_key, local_path)
            logger.info(f"Downloaded from R2: {remote_key} -> {local_path}")
            return True
        except Exception as e:
            logger.info(f"R2 download skipped ({remote_key}): {e}")
            return False

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
        """Create tables if they don't exist, and migrate new columns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS markets (
                market_id TEXT PRIMARY KEY,
                question TEXT,
                description TEXT,
                resolution_criteria TEXT,
                category TEXT,
                volume REAL,
                liquidity REAL,
                end_date TEXT,
                last_updated TEXT,
                first_seen TEXT,
                volume24hr REAL,
                volume1wk REAL,
                volume1mo REAL,
                one_month_price_change REAL,
                last_trade_price REAL,
                spread REAL,
                clob_token_ids TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_price_grid (
                market_id TEXT,
                hour_offset INTEGER,
                price REAL,
                fetched_at TEXT,
                PRIMARY KEY (market_id, hour_offset)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                timestamp TEXT,
                yes_price REAL,
                bid REAL,
                ask REAL,
                volume REAL,
                liquidity REAL,
                open_interest REAL,
                n_trades INTEGER,
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

        # Migrate existing databases — add new columns if missing
        migrations = [
            ("markets", "resolution_criteria", "TEXT"),
            ("markets", "first_seen", "TEXT"),
            ("markets", "volume24hr", "REAL"),
            ("markets", "volume1wk", "REAL"),
            ("markets", "volume1mo", "REAL"),
            ("markets", "one_month_price_change", "REAL"),
            ("markets", "last_trade_price", "REAL"),
            ("markets", "spread", "REAL"),
            ("markets", "clob_token_ids", "TEXT"),
            ("market_snapshots", "bid", "REAL"),
            ("market_snapshots", "ask", "REAL"),
            ("market_snapshots", "open_interest", "REAL"),
            ("market_snapshots", "n_trades", "INTEGER"),
        ]
        for table, col, col_type in migrations:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.commit()
        conn.close()
        logger.info("Database tables verified")

    def upsert_market(self, market: dict):
        """Insert or update a market. first_seen is only set on the first insert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        def _f(val):
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        market_id = str(market.get('id', ''))
        now_iso = datetime.now(timezone.utc).isoformat()

        resolution_criteria = (
            market.get('resolutionCriteria') or
            market.get('resolutionSource') or
            market.get('description') or ''
        )

        # Normalise clobTokenIds to a JSON string
        clob_raw = market.get('clobTokenIds', '[]')
        clob_token_ids = clob_raw if isinstance(clob_raw, str) else json.dumps(clob_raw)

        # INSERT IGNORE — only fires on first encounter; preserves first_seen
        cursor.execute('''
            INSERT OR IGNORE INTO markets
            (market_id, first_seen, last_updated)
            VALUES (?, ?, ?)
        ''', (market_id, now_iso, now_iso))

        # UPDATE all mutable fields.
        # first_seen uses COALESCE so existing NULL rows (pre-migration) get
        # stamped on next upsert while already-set values are preserved.
        cursor.execute('''
            UPDATE markets SET
                question=?, description=?, resolution_criteria=?, category=?,
                volume=?, liquidity=?, end_date=?,
                volume24hr=?, volume1wk=?, volume1mo=?,
                one_month_price_change=?, last_trade_price=?, spread=?,
                clob_token_ids=?, last_updated=?,
                first_seen=COALESCE(first_seen, ?)
            WHERE market_id=?
        ''', (
            market.get('question', ''),
            market.get('description', ''),
            resolution_criteria,
            market.get('category', 'uncategorized'),
            _f(market.get('volume')) or 0,
            _f(market.get('liquidity')) or 0,
            market.get('endDate'),
            _f(market.get('volume24hr')),
            _f(market.get('volume1wk')),
            _f(market.get('volume1mo')),
            _f(market.get('oneMonthPriceChange')),
            _f(market.get('lastTradePrice')),
            _f(market.get('spread')),
            clob_token_ids,
            now_iso,
            now_iso,  # COALESCE fallback for first_seen
            market_id,
        ))

        conn.commit()
        conn.close()

    def add_snapshot(self, market: dict):
        """Add a market snapshot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Parse mid price from outcomePrices (YES outcome).
        # The Gamma API returns a JSON string with quoted numbers:
        #   '["0.0005", "0.9995"]'
        # so we must parse as JSON rather than stripping brackets.
        yes_price = 0.0
        try:
            prices_raw = market.get('outcomePrices', '[0]')
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            yes_price = float(prices[0])
        except (ValueError, TypeError, IndexError, json.JSONDecodeError):
            pass

        # Bid / ask
        def _f(val):
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        bid = _f(market.get('bestBid'))
        ask = _f(market.get('bestAsk'))

        # If bid/ask missing, approximate from outcomePrices spread
        if bid is None and ask is None and yes_price > 0:
            bid = yes_price
            ask = yes_price

        # Trade count
        n_trades = None
        for key in ('tradesCount', 'numTrades', 'tradeCount'):
            raw = market.get(key)
            if raw is not None:
                try:
                    n_trades = int(raw)
                except (ValueError, TypeError):
                    pass
                break

        # Open interest
        open_interest = _f(market.get('openInterest'))

        cursor.execute('''
            INSERT INTO market_snapshots
            (market_id, timestamp, yes_price, bid, ask, volume, liquidity, open_interest, n_trades, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(market.get('id', '')),
            datetime.now(timezone.utc).isoformat(),
            yes_price,
            bid,
            ask,
            float(market.get('volume', 0) or 0),
            float(market.get('liquidity', 0) or 0),
            open_interest,
            n_trades,
            'active' if market.get('active') else 'inactive'
        ))

        conn.commit()
        conn.close()

    def get_first_seen(self, market_id: str):
        """Return the first_seen datetime for a market, or None."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT first_seen FROM markets WHERE market_id=?", (market_id,))
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                return datetime.fromisoformat(row[0])
        except Exception:
            pass
        return None

    def store_price_grid_point(self, market_id: str, hour_offset: int, price: float):
        """Upsert a single T±N price grid entry."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO market_price_grid
                (market_id, hour_offset, price, fetched_at)
                VALUES (?, ?, ?, ?)
            ''', (market_id, hour_offset, price, datetime.now(timezone.utc).isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to store price grid point: {e}")

    def get_price_grid(self, market_id: str) -> dict:
        """Return {hour_offset: price} dict for a market (-24..+24)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT hour_offset, price FROM market_price_grid WHERE market_id=?",
                (market_id,)
            )
            rows = cursor.fetchall()
            conn.close()
            return {row[0]: row[1] for row in rows}
        except Exception:
            return {}

    def cleanup_expired_markets(self):
        """Delete markets whose end_date has passed, plus their snapshots."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Remove snapshots for expired markets first (FK-style cleanup)
        cursor.execute("""
            DELETE FROM market_snapshots
            WHERE market_id IN (
                SELECT market_id FROM markets
                WHERE end_date IS NOT NULL
                  AND datetime(end_date) <= datetime('now')
            )
        """)
        snap_deleted = cursor.rowcount

        # Remove the expired market rows themselves
        cursor.execute("""
            DELETE FROM markets
            WHERE end_date IS NOT NULL
              AND datetime(end_date) <= datetime('now')
        """)
        mkt_deleted = cursor.rowcount

        conn.commit()
        conn.close()

        if mkt_deleted > 0:
            logger.info(f"Removed {mkt_deleted} expired markets and {snap_deleted} associated snapshots")

    def get_all_market_ids(self) -> set:
        """Return the set of market IDs already registered in the markets table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT market_id FROM markets")
        ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return ids

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

    DB_PATH = 'prediction_markets.db'

    def __init__(self):
        self.config = ScannerConfig()
        self.config.MIN_MARKET_VOLUME = int(os.environ.get('MIN_MARKET_VOLUME', 10000))

        self.client = PolymarketClient(self.config)
        self.r2 = R2Storage()

        # Restore DB from R2 on startup so history survives redeploys
        self._restore_db_from_r2()

        self.db = DatabaseManager(self.DB_PATH)
        self.scan_interval = int(os.environ.get('SCAN_INTERVAL_MINUTES', 60))

    def _restore_db_from_r2(self):
        """Download the database from R2 if it doesn't exist locally."""
        import os
        if os.path.exists(self.DB_PATH):
            size = os.path.getsize(self.DB_PATH)
            logger.info(f"Local database found ({size:,} bytes), skipping restore")
            return

        logger.info("No local database found — attempting restore from R2...")
        success = self.r2.download_file('prediction_markets.db', self.DB_PATH)
        if success:
            size = os.path.getsize(self.DB_PATH)
            logger.info(f"Database restored from R2 ({size:,} bytes)")
        else:
            logger.info("No R2 backup found — starting with fresh database")

    # -------------------------------------------------------------------------
    # CLOB helpers
    # -------------------------------------------------------------------------

    CLOB_BASE = "https://clob.polymarket.com"

    def _backfill_clob_prices(self, market: dict):
        """One-time CLOB backfill for a newly discovered market (T-24 to T-1).

        Fetches the past 24 hours of hourly prices from the CLOB API and maps
        each data point to its hour offset relative to first_seen.
        """
        try:
            clob_raw = market.get('clobTokenIds', '[]')
            clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
            if not clob_ids:
                return

            token_id = clob_ids[0]
            market_id = str(market.get('id', ''))

            first_seen_dt = self.db.get_first_seen(market_id)
            if not first_seen_dt:
                first_seen_dt = datetime.now(timezone.utc)

            start_ts = int((first_seen_dt - timedelta(hours=24)).timestamp())
            end_ts = int(first_seen_dt.timestamp())

            resp = requests.get(
                f"{self.CLOB_BASE}/prices-history",
                params={"market": token_id, "interval": "1h",
                        "startTs": start_ts, "endTs": end_ts, "fidelity": 60},
                timeout=15,
            )
            resp.raise_for_status()
            history = resp.json().get("history", [])

            stored = 0
            for point in history:
                ts = datetime.fromtimestamp(point["t"], tz=timezone.utc)
                hours_before = (first_seen_dt - ts).total_seconds() / 3600
                offset = -round(hours_before)  # negative: T-24 to T-1
                if -24 <= offset <= -1:
                    self.db.store_price_grid_point(market_id, offset, round(float(point["p"]), 4))
                    stored += 1

            logger.info(f"CLOB backfill: {stored} points stored for market {market_id}")

        except Exception as e:
            logger.warning(f"CLOB backfill failed for market {market.get('id')}: {e}")

    def _update_tplus_price(self, market: dict):
        """Store a T+ price entry if the market is within 24 h of first_seen.

        Called on every hourly scan so the live price is recorded at each
        offset T+1 through T+24.
        """
        try:
            market_id = str(market.get('id', ''))
            first_seen_dt = self.db.get_first_seen(market_id)
            if not first_seen_dt:
                return

            now = datetime.now(timezone.utc)
            hours_since = (now - first_seen_dt).total_seconds() / 3600
            offset = round(hours_since)
            if not (1 <= offset <= 24):
                return

            yes_price = 0.0
            try:
                prices_raw = market.get('outcomePrices', '[0]')
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                yes_price = float(prices[0])
            except (ValueError, TypeError, IndexError, json.JSONDecodeError):
                pass

            self.db.store_price_grid_point(market_id, offset, round(yes_price, 4))

        except Exception as e:
            logger.warning(f"T+ price update failed for market {market.get('id')}: {e}")

    def fetch_and_store_markets(self, discover_new: bool = True):
        """Fetch markets from API and store in database (memory-efficient).

        discover_new=True  — Full sweep: upsert new market metadata AND add
                             snapshots for all qualifying finance markets.
                             Run every 3 hours.
        discover_new=False — Snapshot-only: add snapshots only for markets
                             already registered in the DB. Faster and cheaper.
                             Run every hour.
        """
        # Always load existing IDs: in full-discovery mode we use them to
        # detect newly seen markets (for CLOB backfill); in snapshot-only mode
        # we use them to skip markets we haven't registered yet.
        existing_ids = self.db.get_all_market_ids()
        if discover_new:
            logger.info(f"Full discovery scan (max {MAX_PAGES} pages) ...")
        else:
            logger.info(f"Snapshot-only scan for {len(existing_ids)} existing markets ...")

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
                    "order": "volume24hr",
                    "ascending": "false",
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
                    now_iso = datetime.now(timezone.utc).isoformat()
                    for market in markets:
                        # Skip markets that have already closed
                        end_date = market.get('endDate') or market.get('end_date')
                        if end_date:
                            try:
                                if end_date.replace('Z', '+00:00') < now_iso:
                                    continue
                            except (TypeError, AttributeError):
                                pass

                        if self.client.is_finance_market(market):
                            try:
                                # Use 24-hour volume: only store markets actively trading today
                                volume = float(market.get('volume24hr', 0) or market.get('volume', 0) or 0)
                            except (ValueError, TypeError):
                                volume = 0

                            if volume >= self.config.MIN_MARKET_VOLUME:
                                market_id = str(market.get('id', ''))
                                is_new = market_id not in existing_ids

                                if discover_new:
                                    # Full scan: register market + snapshot
                                    market['category'] = self.client.classify_market(market)
                                    self.db.upsert_market(market)
                                    self.db.add_snapshot(market)
                                    finance_count += 1
                                    if is_new or not self.db.get_price_grid(market_id):
                                        # Backfill: new market, or existing market
                                        # with no price grid (e.g. pre-migration rows)
                                        self._backfill_clob_prices(market)
                                elif market_id in existing_ids:
                                    # Snapshot-only: skip unknown markets
                                    self.db.add_snapshot(market)
                                    finance_count += 1

                                # T+ price grid update for markets within 24 h
                                # of first_seen (both scan types)
                                if not is_new:
                                    self._update_tplus_price(market)

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

    def _flush_tplus_from_snapshots(self):
        """Update T+ price grid for all markets within their 24h discovery window.

        Reads the latest snapshot price directly from the DB rather than
        relying on the market appearing in the current API scan loop.
        This ensures hourly T+ entries are written even when a market doesn't
        surface through the finance/volume filters during a snapshot-only scan.
        """
        try:
            now = datetime.now(timezone.utc)
            cutoff = (now - timedelta(hours=24)).isoformat()

            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()

            # Markets with first_seen in the past 24h + their latest snapshot price
            cursor.execute('''
                SELECT m.market_id, m.first_seen, s.yes_price
                FROM markets m
                JOIN market_snapshots s ON s.market_id = m.market_id
                WHERE m.first_seen IS NOT NULL
                  AND m.first_seen >= ?
                  AND s.id = (
                      SELECT MAX(id) FROM market_snapshots
                      WHERE market_id = m.market_id
                  )
            ''', (cutoff,))

            rows = cursor.fetchall()
            conn.close()

            updated = 0
            for market_id, first_seen_str, yes_price in rows:
                try:
                    first_seen_dt = datetime.fromisoformat(first_seen_str)
                    hours_since = (now - first_seen_dt).total_seconds() / 3600
                    offset = round(hours_since)
                    if 1 <= offset <= 24 and yes_price is not None:
                        self.db.store_price_grid_point(
                            market_id, offset, round(float(yes_price), 4)
                        )
                        updated += 1
                except Exception as e:
                    logger.warning(f"T+ flush failed for market {market_id}: {e}")

            if updated:
                logger.info(f"T+ flush: wrote {updated} price grid point(s)")

        except Exception as e:
            logger.error(f"T+ flush error: {e}")

    def _backfill_tplus_from_clob(self):
        """Re-fetch T+ prices from CLOB for all markets whose T+ grid has only 0.0 values.

        The original add_snapshot() stored yes_price=0.0 due to a JSON parsing bug.
        This method detects stale 0.0-only T+ grids and replaces them with real CLOB history.
        Safe to call every run — skips markets whose T+ grid already has non-zero prices.
        """
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()

            # Find markets where every T+ entry is 0.0 (stale) but T- exists (properly set up)
            cursor.execute("""
                SELECT m.market_id, m.clob_token_ids, m.first_seen
                FROM markets m
                WHERE m.first_seen IS NOT NULL
                  AND m.clob_token_ids IS NOT NULL
                  AND m.clob_token_ids != '[]'
                  AND EXISTS (
                      SELECT 1 FROM market_price_grid g
                      WHERE g.market_id = m.market_id AND g.hour_offset > 0
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM market_price_grid g
                      WHERE g.market_id = m.market_id AND g.hour_offset > 0 AND g.price > 0
                  )
            """)
            stale = cursor.fetchall()
            conn.close()

            if not stale:
                return

            logger.info(f"T+ CLOB re-fetch: {len(stale)} market(s) with stale 0.0 T+ prices")

            for market_id, clob_raw, first_seen_str in stale:
                try:
                    clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
                    if not clob_ids:
                        continue
                    token_id = clob_ids[0]

                    first_seen_dt = datetime.fromisoformat(first_seen_str)
                    if first_seen_dt.tzinfo is None:
                        first_seen_dt = first_seen_dt.replace(tzinfo=timezone.utc)

                    start_ts = int(first_seen_dt.timestamp())
                    end_ts   = int((first_seen_dt + timedelta(hours=25)).timestamp())

                    resp = requests.get(
                        f"{self.CLOB_BASE}/prices-history",
                        params={"market": token_id, "interval": "1h",
                                "startTs": start_ts, "endTs": end_ts, "fidelity": 60},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    history = resp.json().get("history", [])

                    # Delete stale 0.0 T+ entries before replacing
                    conn = sqlite3.connect(self.db.db_path)
                    conn.execute(
                        "DELETE FROM market_price_grid WHERE market_id=? AND hour_offset > 0",
                        (market_id,)
                    )
                    stored = 0
                    for point in history:
                        pt_dt = datetime.fromtimestamp(point["t"], tz=timezone.utc)
                        hours_after = (pt_dt - first_seen_dt).total_seconds() / 3600
                        offset = round(hours_after)
                        if 1 <= offset <= 24:
                            conn.execute(
                                "INSERT OR REPLACE INTO market_price_grid "
                                "(market_id, hour_offset, price, fetched_at) VALUES (?,?,?,?)",
                                (market_id, offset, round(float(point["p"]), 4),
                                 datetime.now(timezone.utc).isoformat())
                            )
                            stored += 1
                    conn.commit()
                    conn.close()
                    logger.info(f"T+ re-fetch: {stored} pts for market {market_id}")

                except Exception as e:
                    logger.warning(f"T+ CLOB re-fetch failed for {market_id}: {e}")

        except Exception as e:
            logger.error(f"T+ CLOB re-fetch error: {e}")

    def _backfill_zombie_markets(self):
        """One-time backfill for markets that have snapshots but no first_seen or price grid.

        Handles pre-migration rows that were stored before the new pipeline.
        For each zombie market:
          1. Sets first_seen from its oldest snapshot timestamp.
          2. Resolves clobTokenIds via Gamma API if not stored.
          3. Calls CLOB API to populate T-minus price grid (T-24..T-1).
        Safe to call on every run — quickly no-ops once all markets are fixed.
        """
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()

            # Markets that have at least one snapshot but still have first_seen=NULL
            cursor.execute("""
                SELECT m.market_id, m.clob_token_ids, MIN(s.timestamp) AS oldest_snap
                FROM markets m
                JOIN market_snapshots s ON s.market_id = m.market_id
                WHERE m.first_seen IS NULL
                GROUP BY m.market_id
            """)
            zombies = cursor.fetchall()
            conn.close()

            if not zombies:
                return

            logger.info(f"Zombie backfill: fixing {len(zombies)} market(s) with missing first_seen")

            for market_id, clob_raw, oldest_snap in zombies:
                try:
                    # 1. Stamp first_seen with the oldest snapshot we have
                    conn = sqlite3.connect(self.db.db_path)
                    conn.execute(
                        "UPDATE markets SET first_seen=? WHERE market_id=? AND first_seen IS NULL",
                        (oldest_snap, market_id),
                    )
                    conn.commit()
                    conn.close()
                    logger.info(f"Zombie {market_id}: set first_seen={oldest_snap[:19]}")

                    # 2. Resolve clob_token_ids — stored value or Gamma API lookup
                    clob_ids = []
                    if clob_raw and clob_raw != '[]':
                        try:
                            clob_ids = json.loads(clob_raw)
                        except Exception:
                            pass

                    if not clob_ids:
                        try:
                            resp = requests.get(
                                self.config.POLYMARKET_API_URL,
                                params={"id": market_id, "limit": 1},
                                timeout=15,
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            if data:
                                raw = data[0].get('clobTokenIds', '[]')
                                clob_ids = json.loads(raw) if isinstance(raw, str) else raw
                                if clob_ids:
                                    conn = sqlite3.connect(self.db.db_path)
                                    conn.execute(
                                        "UPDATE markets SET clob_token_ids=? WHERE market_id=?",
                                        (json.dumps(clob_ids), market_id),
                                    )
                                    conn.commit()
                                    conn.close()
                                    logger.info(f"Zombie {market_id}: fetched clob_token_ids from Gamma API")
                        except Exception as e:
                            logger.warning(f"Gamma API lookup failed for zombie {market_id}: {e}")

                    if not clob_ids:
                        logger.warning(f"Zombie {market_id}: no CLOB token IDs found, skipping price backfill")
                        continue

                    # 3. CLOB price backfill anchored to first_seen (oldest snapshot)
                    self._backfill_clob_prices({'id': market_id, 'clobTokenIds': json.dumps(clob_ids)})

                    # 4. Retrospective T+ backfill from existing snapshots.
                    #    The T+ flush normally only covers first_seen within the last 24h,
                    #    but zombie first_seen values are older, so we scan snapshots directly.
                    try:
                        first_seen_dt = datetime.fromisoformat(oldest_snap)
                        tplus_end = first_seen_dt + timedelta(hours=24)
                        conn = sqlite3.connect(self.db.db_path)
                        snap_cur = conn.cursor()
                        snap_cur.execute("""
                            SELECT timestamp, yes_price FROM market_snapshots
                            WHERE market_id=? AND timestamp >= ? AND timestamp <= ?
                              AND yes_price IS NOT NULL
                            ORDER BY timestamp ASC
                        """, (market_id, oldest_snap, tplus_end.isoformat()))
                        snaps = snap_cur.fetchall()
                        conn.close()
                        tplus_stored = 0
                        for snap_ts, yes_price in snaps:
                            snap_dt = datetime.fromisoformat(snap_ts)
                            hours_since = (snap_dt - first_seen_dt).total_seconds() / 3600
                            offset = round(hours_since)
                            if 1 <= offset <= 24:
                                self.db.store_price_grid_point(
                                    market_id, offset, round(float(yes_price), 4)
                                )
                                tplus_stored += 1
                        if tplus_stored:
                            logger.info(f"Zombie {market_id}: retrospective T+ stored {tplus_stored} point(s)")
                    except Exception as e:
                        logger.warning(f"Retrospective T+ failed for zombie {market_id}: {e}")

                except Exception as e:
                    logger.warning(f"Zombie fix failed for market {market_id}: {e}")

            logger.info("Zombie backfill complete")

        except Exception as e:
            logger.error(f"Zombie backfill error: {e}")

    def run_once(self, discover_new: bool = True):
        """Run a single scan cycle.

        discover_new=True  — full discovery + snapshot (runs every 3 h).
        discover_new=False — snapshot-only for existing markets (runs every 1 h).
        """
        mode = "FULL DISCOVERY" if discover_new else "SNAPSHOT ONLY"
        logger.info("=" * 60)
        logger.info(f"Starting {mode} scan at {datetime.now(timezone.utc).isoformat()}")
        logger.info("=" * 60)

        # Step 0a: Fix pre-migration markets missing first_seen / price grid
        self._backfill_zombie_markets()
        # Step 0b: Re-fetch T+ prices from CLOB where yes_price was stored as 0.0
        self._backfill_tplus_from_clob()

        # Step 1: Fetch and store
        market_count = self.fetch_and_store_markets(discover_new=discover_new)
        gc.collect()  # Free memory after fetch

        if market_count == 0:
            logger.warning("No markets fetched, skipping analysis")
            return

        # Step 2: Update T+ price grid from latest snapshots (every scan cycle)
        self._flush_tplus_from_snapshots()

        # Step 3: Run analysis
        result = self.run_analysis()
        gc.collect()  # Free memory after analysis

        # Step 4: Upload to R2
        if result:
            self.upload_to_r2(result)

        # Step 5: Cleanup — remove expired markets and old snapshots
        self.db.cleanup_expired_markets()
        self.db.cleanup_old_snapshots(days=30)

        # Final cleanup
        gc.collect()
        logger.info("Scan cycle complete")

    def run_scheduled(self):
        """Run on a schedule.

        Snapshot-only scan: every SCAN_INTERVAL_MINUTES (default 60).
        Full discovery scan: every 3 hours (picks up new markets).
        Both run immediately on startup.
        """
        logger.info(
            f"Starting scheduled runner — snapshots every {self.scan_interval} min, "
            "full discovery every 3 h"
        )

        # Run a full discovery scan immediately on startup
        self.run_once(discover_new=True)

        # Hourly: snapshot-only (fast)
        schedule.every(self.scan_interval).minutes.do(self.run_once, discover_new=False)

        # Every 3 hours: full discovery (finds new markets)
        schedule.every(3).hours.do(self.run_once, discover_new=True)

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
