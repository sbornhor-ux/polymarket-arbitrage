"""
Integration Script for Polymarket Monitoring System

This script demonstrates how the Platform Monitor Agent and Movement Analyzer
work together to detect and alert on significant market movements.
"""

import time
import threading
from datetime import datetime
from polymarket_monitor import PolymarketMonitor
from movement_analyzer import MovementAnalyzer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedMonitoringSystem:
    """
    Integrated monitoring system combining market data collection
    and movement analysis
    """
    
    def __init__(
        self,
        db_path: str = "prediction_markets.db",
        alert_email: str = "sbornhor@uchicago.edu",
        poll_interval: int = 60,
        snapshot_interval: int = 3600  # 1 hour
    ):
        """
        Initialize integrated system
        
        Args:
            db_path: Path to SQLite database
            alert_email: Email for alerts
            poll_interval: Seconds between market data polls
            snapshot_interval: Seconds between snapshots for analysis
        """
        self.db_path = db_path
        self.alert_email = alert_email
        self.poll_interval = poll_interval
        self.snapshot_interval = snapshot_interval
        
        # Initialize both agents
        self.monitor = PolymarketMonitor(
            db_path=db_path,
            poll_interval=poll_interval
        )
        
        self.analyzer = MovementAnalyzer(
            db_path=db_path,
            alert_email=alert_email
        )
        
        logger.info("Integrated monitoring system initialized")
    
    def run_monitor_loop(self):
        """Run the platform monitor in a loop"""
        logger.info("Starting market data monitor")
        self.monitor.run()
    
    def run_analyzer_loop(self):
        """Run the movement analyzer in a loop"""
        logger.info("Starting movement analyzer")
        self.analyzer.run_continuous()
    
    def run_integrated(self):
        """
        Run both agents in parallel threads
        """
        logger.info("="*60)
        logger.info("STARTING INTEGRATED POLYMARKET MONITORING SYSTEM")
        logger.info("="*60)
        logger.info("")
        logger.info("Components:")
        logger.info("  1. Platform Monitor - Polls Polymarket API every 60s")
        logger.info("  2. Movement Analyzer - Takes snapshots hourly")
        logger.info("  3. Daily Alert - Sends email at 4 PM CST")
        logger.info("")
        logger.info(f"Alert destination: {self.alert_email}")
        logger.info(f"Database: {self.db_path}")
        logger.info("="*60)
        
        # Create threads for parallel execution
        monitor_thread = threading.Thread(
            target=self.run_monitor_loop,
            name="MonitorThread",
            daemon=True
        )
        
        analyzer_thread = threading.Thread(
            target=self.run_analyzer_loop,
            name="AnalyzerThread",
            daemon=True
        )
        
        # Start both threads
        monitor_thread.start()
        time.sleep(2)  # Brief delay to let monitor initialize
        analyzer_thread.start()
        
        logger.info("\nBoth agents running. Press Ctrl+C to stop.")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(60)
                
                # Periodic status check
                if datetime.now().minute == 0:  # Every hour
                    status = self.monitor.get_status()
                    logger.info(f"Status Update - Markets fetched: {status['total_markets_fetched']}, "
                              f"Errors: {status['error_count']}")
        
        except KeyboardInterrupt:
            logger.info("\nShutting down monitoring system...")
            logger.info("Agents stopped.")


def quick_test():
    """
    Run a quick test of the integrated system
    """
    print("\n" + "="*60)
    print("QUICK TEST - Integrated Monitoring System")
    print("="*60 + "\n")
    
    # Initialize system
    system = IntegratedMonitoringSystem()
    
    # Step 1: Collect market data
    print("Step 1: Collecting market data from Polymarket...")
    system.monitor.poll_once()
    print("[OK] Market data collected\n")
    
    # Step 2: Take snapshot
    print("Step 2: Taking snapshot for movement analysis...")
    snapshot_count = system.analyzer.take_snapshot()
    print(f"[OK] Snapshot taken for {snapshot_count} markets\n")
    
    # Wait a bit and take another snapshot (simulating time passage)
    print("Waiting 5 seconds...")
    time.sleep(5)
    
    # Step 3: Collect data again
    print("\nStep 3: Collecting updated market data...")
    system.monitor.poll_once()
    print("[OK] Updated data collected\n")
    
    # Step 4: Take another snapshot
    print("Step 4: Taking second snapshot...")
    system.analyzer.take_snapshot()
    print("[OK] Second snapshot taken\n")
    
    # Step 5: Analyze movements
    print("Step 5: Analyzing market movements...")
    analysis = system.analyzer.analyze_last_24_hours()
    print(f"[OK] Analysis complete\n")
    
    # Step 6: Display results
    print("="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"Total Movements Detected: {analysis['total_movements']}")
    print(f"  - Odds Swings (>30% in <6hrs): {analysis['odds_swings']}")
    print(f"  - Capital Flow Surges: {analysis['capital_flows']}")
    print(f"\nSeverity Breakdown:")
    print(f"  - [HIGH] {analysis['high_severity']}")
    print(f"  - [MED]  {analysis['medium_severity']}")
    print(f"  - [LOW]  {analysis['low_severity']}")
    
    if analysis['movements']:
        print(f"\nTop 5 Movements:")
        for i, movement in enumerate(analysis['movements'][:5], 1):
            print(f"\n{i}. {movement.question[:60]}...")
            print(f"   Type: {movement.movement_type} | Severity: {movement.severity}")
            print(f"   {movement.explanation}")
    
    # Step 7: Generate report
    print("\n" + "="*60)
    print("Step 7: Generating email report...")
    html = system.analyzer.generate_email_report(analysis)
    
    filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"[OK] Report generated and saved to: {filename}")
    print("\nYou can open this file in a browser to see the formatted email.")
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated HTML report")
    print("2. Configure email settings in analyzer_config.py")
    print("3. Run full system with: python integrated_system.py --mode production")
    print("\n")


def production_mode():
    """
    Run in production mode with continuous monitoring
    """
    print("\n" + "="*60)
    print("PRODUCTION MODE")
    print("="*60)
    print("\nThis will run continuously until stopped with Ctrl+C")
    print("Features:")
    print("  - Polls Polymarket every 60 seconds")
    print("  - Takes snapshots every hour")
    print("  - Sends daily email alert at 4 PM CST")
    print("\nPress Ctrl+C at any time to stop.\n")
    
    input("Press Enter to start production monitoring...")
    
    system = IntegratedMonitoringSystem()
    system.run_integrated()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integrated Polymarket Monitoring System'
    )
    parser.add_argument(
        '--mode',
        choices=['test', 'production'],
        default='test',
        help='Operation mode (test for quick test, production for continuous)'
    )
    parser.add_argument(
        '--email',
        default='sbornhor@uchicago.edu',
        help='Alert email address'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        quick_test()
    else:
        production_mode()
