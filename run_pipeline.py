#!/usr/bin/env python3
"""
run_pipeline.py — Full local pipeline runner

Chains the complete Polymarket analysis pipeline:
  1. Download latest DB from R2
  2. Export CSV from DB
  3. Run Finance Agent (Polygon.io) → pipeline JSON
  4. Run Trend Analyst (OpenAI) → trend analysis JSON
  5. Run Synthesizer (OpenAI) → Markdown investment report
  6. Upload all outputs to R2

Usage:
    python run_pipeline.py                  # full run
    python run_pipeline.py --skip-download  # use existing local DB
    python run_pipeline.py --skip-finance   # skip finance/trend/synth (CSV export only)
    python run_pipeline.py --skip-upload    # don't push outputs to R2

Requirements:
    Copy .env.example to .env and fill in your API keys.
"""

import os
import sys
import json
import csv
import sqlite3
import logging
import argparse
import importlib
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('pipeline')

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
def _load_env(path: str = '.env') -> None:
    """Simple .env loader — no external dependency needed."""
    env_path = Path(path)
    if not env_path.exists():
        log.warning(f".env not found at {path} — falling back to existing environment variables")
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:  # don't override existing env
                os.environ[key] = val
    log.info(f"Loaded environment from {path}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

TODAY = datetime.now(timezone.utc).strftime('%Y-%m-%d')
DB_LOCAL     = ROOT / 'pipeline_latest.db'
CSV_PATH     = OUTPUT_DIR / f'polymarket_prod_{TODAY}.csv'
PIPELINE_PATH = OUTPUT_DIR / f'pipeline_{TODAY}.json'
TREND_PATH   = OUTPUT_DIR / f'trend_analysis_{TODAY}.json'
REPORT_PATH  = OUTPUT_DIR / f'investment_report_{TODAY}.md'


# ---------------------------------------------------------------------------
# Step 1 — Download latest DB from R2
# ---------------------------------------------------------------------------
def download_db() -> Path:
    log.info("Step 1 — Downloading latest DB from R2...")
    import boto3

    account_id = os.environ['R2_ACCOUNT_ID']
    access_key = os.environ['R2_ACCESS_KEY_ID']
    secret_key = os.environ['R2_SECRET_ACCESS_KEY']
    bucket     = os.environ.get('R2_BUCKET_NAME', 'polymarket-data')

    client = boto3.client(
        's3',
        endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='auto',
    )
    client.download_file(bucket, 'prediction_markets.db', str(DB_LOCAL))
    size = DB_LOCAL.stat().st_size
    log.info(f"  Downloaded {DB_LOCAL.name} ({size:,} bytes)")
    return DB_LOCAL


# ---------------------------------------------------------------------------
# Step 2 — Export CSV from DB
# ---------------------------------------------------------------------------
def export_csv(db_path: Path) -> Path:
    log.info("Step 2 — Exporting CSV from DB...")
    sys.path.insert(0, str(ROOT))
    import polymarket_scanner
    importlib.reload(polymarket_scanner)
    from polymarket_scanner import ScannerConfig, PolymarketClient

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    markets = conn.execute("""
        SELECT m.market_id, m.question, m.description, m.category, m.slug,
               m.end_date, m.first_seen,
               m.volume, m.liquidity, m.volume24hr, m.volume1wk, m.volume1mo,
               m.one_month_price_change, m.last_trade_price, m.spread,
               m.clob_token_ids
        FROM markets m
        ORDER BY m.volume24hr DESC NULLS LAST
    """).fetchall()

    grid_rows = conn.execute(
        "SELECT market_id, hour_offset, price FROM market_price_grid"
    ).fetchall()
    grid = {}
    for r in grid_rows:
        grid.setdefault(r[0], {})[r[1]] = r[2]

    snap_rows = conn.execute("""
        SELECT market_id, yes_price, timestamp FROM market_snapshots
        WHERE (market_id, timestamp) IN (
            SELECT market_id, MAX(timestamp) FROM market_snapshots GROUP BY market_id
        )
    """).fetchall()
    snaps = {r[0]: (r[1], r[2]) for r in snap_rows}
    conn.close()

    cfg = ScannerConfig()
    client = PolymarketClient(cfg)

    tminus = list(range(-24, 0))
    tplus  = list(range(1, 25))
    fieldnames = (
        ['market_id', 'question', 'category', 'market_slug', 'end_date', 'first_seen',
         'volume', 'liquidity', 'volume24hr', 'volume1wk', 'volume1mo',
         'one_month_price_change', 'last_trade_price', 'spread',
         'current_price', 'current_as_of', 'clob_token_ids'] +
        [f'price_t_minus_{abs(h):02d}' for h in tminus] +
        [f'price_t_plus_{h:02d}' for h in tplus]
    )

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in markets:
            mid = m['market_id']
            g   = grid.get(mid, {})
            cur_price, cur_ts = snaps.get(mid, (None, None))
            mdict = dict(m)
            is_finance = client.is_finance_market(mdict)
            category   = client.classify_market(mdict) if is_finance else 'Sports / Excluded'
            row = {
                'market_id': mid,
                'question': m['question'],
                'category': category,
                'market_slug': m['slug'] or '',
                'end_date': m['end_date'],
                'first_seen': m['first_seen'],
                'volume': m['volume'],
                'liquidity': m['liquidity'],
                'volume24hr': m['volume24hr'],
                'volume1wk': m['volume1wk'],
                'volume1mo': m['volume1mo'],
                'one_month_price_change': m['one_month_price_change'],
                'last_trade_price': m['last_trade_price'],
                'spread': m['spread'],
                'current_price': cur_price,
                'current_as_of': cur_ts,
                'clob_token_ids': m['clob_token_ids'] or '',
            }
            for h in tminus:
                row[f'price_t_minus_{abs(h):02d}'] = g.get(h)
            for h in tplus:
                row[f'price_t_plus_{h:02d}'] = g.get(h)
            writer.writerow(row)

    log.info(f"  Exported {len(markets)} markets → {CSV_PATH.name}")
    return CSV_PATH


# ---------------------------------------------------------------------------
# Step 3 — Finance Agent (Polygon.io)
# ---------------------------------------------------------------------------
def run_finance_agent(csv_path: Path) -> Path:
    log.info("Step 3 — Running Finance Agent (Polygon.io) + LLM Instrument Selection...")
    sys.path.insert(0, str(ROOT))

    from polymarket_agent.loader import load_from_csv
    from polymarket_agent.normalizer import normalize_market
    from polymarket_agent.resampler import resample_to_5m
    from polymarket_agent.swing_detector import detect_swings
    from polymarket_agent.relevance_screener import classify_relevance
    from polymarket_agent.client import fetch_price_history, extract_yes_token_id
    from finance_agent import get_window_stats
    from instrument_selector import select_instruments

    raw_markets = load_from_csv(str(csv_path))
    log.info(f"  Loaded {len(raw_markets)} markets from CSV")

    snapshots = []
    all_stats = []
    all_selections = []
    skipped = 0

    clob_fetched = 0
    clob_failed = 0

    for i, raw in enumerate(raw_markets):
        try:
            # Fetch 72-hour 15-minute price history from Polymarket CLOB API,
            # replacing the 24-hour hourly history stored in the DB/CSV.
            clob_token_ids_str = raw.get('metadata', {}).get('clob_token_ids', '') or ''
            yes_token_id = extract_yes_token_id(clob_token_ids_str)
            if yes_token_id:
                fresh_history = fetch_price_history(yes_token_id, lookback_hours=72)
                if fresh_history:
                    raw['history'] = fresh_history
                    clob_fetched += 1
                else:
                    clob_failed += 1
            else:
                clob_failed += 1

            snap = normalize_market(raw)
            snap = resample_to_5m(snap)
            snap = detect_swings(snap)

            rel_score = snap.financial_relevance_score or 0.0
            tier = classify_relevance(rel_score)

            # Only run finance agent on medium+ relevance markets with detected swings
            if tier == 'low' or not snap.swings:
                skipped += 1
                snapshots.append(snap)
                continue

            log.info(f"  [{i+1}/{len(raw_markets)}] {snap.market_question[:60]} "
                     f"(rel={rel_score:.2f}, swings={len(snap.swings)})")

            # Step 3a: LLM selects instruments for this market
            selection = select_instruments(
                market_question=snap.market_question,
                swings=snap.swings,
                market_id=snap.market_id,
            )
            all_selections.append(selection.model_dump())

            # Step 3b: Use LLM-selected tickers; fall back to DEFAULT_SERIES if none validated
            selected_tickers = [p.ticker for p in selection.instruments] or None
            if selected_tickers:
                log.info(f"    LLM selected: {selected_tickers}")
            else:
                log.info(f"    LLM returned no valid tickers — using default series")

            # Build a metadata lookup keyed by ticker
            llm_meta = {
                p.ticker: (p.confidence, p.predicted_direction, p.company_name, p.rationale)
                for p in selection.instruments
            }

            stats = get_window_stats(
                swing_events=snap.swings,
                market_id=snap.market_id,
                series=selected_tickers,
            )

            # Attach LLM metadata to each stat record
            for stat in stats:
                if stat.series_id in llm_meta:
                    conf, direction, name, rationale = llm_meta[stat.series_id]
                    stat.llm_confidence = conf
                    stat.llm_predicted_direction = direction
                    stat.llm_company_name = name
                    stat.llm_rationale = rationale

            all_stats.extend(stats)
            snapshots.append(snap)

        except Exception as e:
            log.warning(f"  Market {i+1} failed: {e}")
            skipped += 1
            continue

    log.info(f"  CLOB 72h history: {clob_fetched} fetched, {clob_failed} fell back to CSV data")
    log.info(f"  Processed {len(snapshots)} markets, {skipped} skipped (low relevance/no swings)")
    log.info(f"  Generated {len(all_stats)} finance data points across {len(all_selections)} selections")

    # Serialize to Format D JSON (compatible with Trend Analyst financial_adapter)
    output = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_csv': str(csv_path),
        'markets': [s.model_dump() for s in snapshots],
        'finance_stats': [s.model_dump() for s in all_stats],
        'instrument_selections': all_selections,
    }
    with open(PIPELINE_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"  Saved pipeline -> {PIPELINE_PATH.name}")
    return PIPELINE_PATH


# ---------------------------------------------------------------------------
# Step 4 — Trend Analyst (OpenAI)
# ---------------------------------------------------------------------------
def run_trend_analyst(csv_path: Path, pipeline_path: Path) -> Path:
    log.info("Step 4 — Running Trend Analyst (OpenAI)...")

    # Add trend_analyst/ to path so its internal imports work
    trend_dir = ROOT / 'trend_analyst'
    sys.path.insert(0, str(trend_dir))

    from trend_analyst import run_trend_analyst as _run

    result = _run(
        polymarket_csv=str(csv_path),
        financial_source=str(pipeline_path),
        output_path=str(TREND_PATH),
        upload_r2=False,
        use_llm=True,
    )

    if TREND_PATH.exists():
        log.info(f"  Saved trend analysis → {TREND_PATH.name}")
    else:
        raise RuntimeError("Trend analyst did not produce output file")

    return TREND_PATH


# ---------------------------------------------------------------------------
# Step 5 — Synthesizer (OpenAI → Markdown report)
# ---------------------------------------------------------------------------
def run_synthesizer(trend_path: Path) -> Path:
    log.info("Step 5 — Running Synthesizer (OpenAI)...")
    sys.path.insert(0, str(ROOT))

    from synthesizer import generate_report_from_json

    generate_report_from_json(
        json_file_path=str(trend_path),
        output_file=str(REPORT_PATH),
    )

    if REPORT_PATH.exists():
        log.info(f"  Saved report → {REPORT_PATH.name}")
    else:
        raise RuntimeError("Synthesizer did not produce output file")

    return REPORT_PATH


# ---------------------------------------------------------------------------
# Step 6 — Upload outputs to R2
# ---------------------------------------------------------------------------
def upload_outputs(*paths: Path) -> None:
    log.info("Step 6 — Uploading outputs to R2...")
    import boto3

    account_id = os.environ['R2_ACCOUNT_ID']
    access_key = os.environ['R2_ACCESS_KEY_ID']
    secret_key = os.environ['R2_SECRET_ACCESS_KEY']
    bucket     = os.environ.get('R2_BUCKET_NAME', 'polymarket-data')

    client = boto3.client(
        's3',
        endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='auto',
    )

    content_types = {
        '.csv':  'text/csv',
        '.json': 'application/json',
        '.md':   'text/markdown',
        '.db':   'application/octet-stream',
    }

    for path in paths:
        if not path.exists():
            log.warning(f"  Skipping missing file: {path.name}")
            continue
        ct = content_types.get(path.suffix, 'application/octet-stream')
        key = f'pipeline/{path.name}'
        client.upload_file(
            str(path), bucket, key,
            ExtraArgs={'ContentType': ct},
        )
        log.info(f"  Uploaded → {key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Run full Polymarket analysis pipeline')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip R2 download — use existing pipeline_latest.db')
    parser.add_argument('--skip-finance', action='store_true',
                        help='Skip finance agent, trend analyst, and synthesizer')
    parser.add_argument('--skip-upload', action='store_true',
                        help='Skip uploading outputs to R2')
    args = parser.parse_args()

    _load_env()

    t_start = datetime.now()
    log.info("=" * 60)
    log.info("  Polymarket Pipeline — " + TODAY)
    log.info("=" * 60)

    try:
        # Step 1: Download DB
        db_path = DB_LOCAL
        if not args.skip_download:
            db_path = download_db()
        else:
            log.info("Step 1 — Skipped (using existing local DB)")
            if not DB_LOCAL.exists():
                log.error(f"No local DB found at {DB_LOCAL}. Remove --skip-download.")
                sys.exit(1)

        # Step 2: Export CSV
        csv_path = export_csv(db_path)

        if args.skip_finance:
            log.info("Steps 3-5 — Skipped (--skip-finance)")
        else:
            # Step 3: Finance Agent
            pipeline_path = run_finance_agent(csv_path)

            # Step 4: Trend Analyst
            trend_path = run_trend_analyst(csv_path, pipeline_path)

            # Step 5: Synthesizer
            report_path = run_synthesizer(trend_path)

            # Step 6: Upload
            if not args.skip_upload:
                upload_outputs(csv_path, pipeline_path, trend_path, report_path)
            else:
                log.info("Step 6 — Skipped (--skip-upload)")

    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    elapsed = (datetime.now() - t_start).total_seconds()
    log.info("=" * 60)
    log.info(f"  Done in {elapsed:.0f}s  — outputs in output/")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
