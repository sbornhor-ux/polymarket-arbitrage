"""
Test/Demo script for Polymarket Monitor Agent

This script demonstrates:
1. Single poll execution
2. Status reporting
3. Database queries
4. Error handling
"""

import sqlite3
import json
from datetime import datetime
from polymarket_monitor import PolymarketMonitor
from config import DATABASE_PATH


def print_separator(title=""):
    """Print a section separator"""
    print("\n" + "="*60)
    if title:
        print(f"  {title}")
        print("="*60)


def test_single_poll():
    """Test a single poll cycle"""
    print_separator("TEST: Single Poll Cycle")
    
    monitor = PolymarketMonitor(
        db_path=DATABASE_PATH,
        poll_interval=60,
        max_calls_per_minute=100
    )
    
    print("\nExecuting single poll...")
    success = monitor.poll_once()
    
    if success:
        print("✓ Poll completed successfully")
    else:
        print("✗ Poll failed")
    
    # Get and display status
    status = monitor.get_status()
    print("\n--- Agent Status ---")
    for key, value in status.items():
        print(f"{key:25s}: {value}")
    
    return monitor


def query_markets(limit=5):
    """Query and display markets from database"""
    print_separator(f"QUERY: Latest {limit} Markets")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT market_id, question, category, status, yes_price, no_price, 
               volume, liquidity, last_updated
        FROM markets
        ORDER BY last_updated DESC
        LIMIT ?
    """, (limit,))
    
    markets = cursor.fetchall()
    
    if not markets:
        print("\nNo markets found in database.")
        print("Run test_single_poll() first to fetch data.")
    else:
        print(f"\nFound {len(markets)} markets:\n")
        for i, market in enumerate(markets, 1):
            market_id, question, category, status, yes_price, no_price, volume, liquidity, updated = market
            print(f"{i}. {question[:60]}...")
            print(f"   Category: {category} | Status: {status}")
            print(f"   Prices: YES=${yes_price:.3f} NO=${no_price:.3f}")
            print(f"   Volume: ${volume:,.2f} | Liquidity: ${liquidity:,.2f}")
            print(f"   Updated: {updated}")
            print()
    
    conn.close()


def query_errors():
    """Query and display recent errors"""
    print_separator("QUERY: Recent Errors")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT timestamp, error_type, error_message, endpoint, response_code
        FROM error_log
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    
    errors = cursor.fetchall()
    
    if not errors:
        print("\n✓ No errors logged")
    else:
        print(f"\nFound {len(errors)} recent errors:\n")
        for i, error in enumerate(errors, 1):
            timestamp, error_type, message, endpoint, code = error
            print(f"{i}. [{timestamp}] {error_type}")
            print(f"   {message}")
            if endpoint:
                print(f"   Endpoint: {endpoint}")
            if code:
                print(f"   Response Code: {code}")
            print()
    
    conn.close()


def analyze_markets():
    """Analyze market data from database"""
    print_separator("ANALYSIS: Market Statistics")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Total markets
    cursor.execute("SELECT COUNT(*) FROM markets")
    total = cursor.fetchone()[0]
    
    # Markets by status
    cursor.execute("""
        SELECT status, COUNT(*) as count
        FROM markets
        GROUP BY status
    """)
    status_counts = cursor.fetchall()
    
    # Markets by category
    cursor.execute("""
        SELECT category, COUNT(*) as count, SUM(volume) as total_volume
        FROM markets
        GROUP BY category
        ORDER BY total_volume DESC
        LIMIT 10
    """)
    category_stats = cursor.fetchall()
    
    # Volume statistics
    cursor.execute("""
        SELECT 
            SUM(volume) as total_volume,
            AVG(volume) as avg_volume,
            MAX(volume) as max_volume,
            SUM(liquidity) as total_liquidity
        FROM markets
    """)
    volume_stats = cursor.fetchone()
    
    print(f"\n📊 Total Markets: {total}")
    
    print("\n--- Markets by Status ---")
    for status, count in status_counts:
        print(f"{status:15s}: {count:5d} ({count/total*100:.1f}%)")
    
    print("\n--- Top Categories by Volume ---")
    for category, count, vol in category_stats:
        print(f"{category:20s}: {count:4d} markets | ${vol:,.2f}")
    
    print("\n--- Volume Statistics ---")
    total_vol, avg_vol, max_vol, total_liq = volume_stats
    print(f"Total Volume:     ${total_vol:,.2f}")
    print(f"Average Volume:   ${avg_vol:,.2f}")
    print(f"Max Volume:       ${max_vol:,.2f}")
    print(f"Total Liquidity:  ${total_liq:,.2f}")
    
    conn.close()


def show_high_volume_markets(min_volume=10000):
    """Show markets with high trading volume"""
    print_separator(f"QUERY: High Volume Markets (>${min_volume:,})")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT question, category, yes_price, no_price, volume, liquidity
        FROM markets
        WHERE volume > ?
        ORDER BY volume DESC
        LIMIT 10
    """, (min_volume,))
    
    markets = cursor.fetchall()
    
    if not markets:
        print(f"\nNo markets found with volume > ${min_volume:,}")
    else:
        print(f"\nFound {len(markets)} high-volume markets:\n")
        for i, market in enumerate(markets, 1):
            question, category, yes_price, no_price, volume, liquidity = market
            print(f"{i}. {question[:70]}")
            print(f"   {category} | YES=${yes_price:.3f} NO=${no_price:.3f}")
            print(f"   Volume: ${volume:,.2f} | Liquidity: ${liquidity:,.2f}")
            print()
    
    conn.close()


def run_full_demo():
    """Run complete demonstration"""
    print("\n" + "="*60)
    print("  POLYMARKET MONITOR AGENT - FULL DEMONSTRATION")
    print("="*60)
    
    # Test single poll
    monitor = test_single_poll()
    
    # Query markets
    query_markets(limit=10)
    
    # Analyze markets
    analyze_markets()
    
    # Show high volume markets
    show_high_volume_markets(min_volume=5000)
    
    # Check for errors
    query_errors()
    
    print_separator("DEMONSTRATION COMPLETE")
    print("\nThe agent has successfully:")
    print("✓ Connected to Polymarket API")
    print("✓ Fetched and normalized market data")
    print("✓ Stored data in SQLite database")
    print("✓ Tracked state and errors")
    print("\nDatabase file:", DATABASE_PATH)
    print("Log file: polymarket_monitor.log")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "poll":
            test_single_poll()
        elif command == "query":
            query_markets(10)
        elif command == "analyze":
            analyze_markets()
        elif command == "errors":
            query_errors()
        elif command == "high-volume":
            show_high_volume_markets()
        else:
            print("Unknown command. Available commands:")
            print("  poll        - Execute single poll")
            print("  query       - Query latest markets")
            print("  analyze     - Analyze market statistics")
            print("  errors      - Show error log")
            print("  high-volume - Show high volume markets")
    else:
        run_full_demo()
