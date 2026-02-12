"""
Market Movement Analysis Agent

Monitors Polymarket data for significant market movements and anomalies:
- Large odds swings (>30% in <6 hours)
- Large capital inflows in short periods
- Unusual betting patterns
- Sends daily email alerts at 4 PM CST with analysis summary
"""

import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import pytz
import time
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('movement_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MarketMovement:
    """Data class for significant market movements"""
    market_id: str
    question: str
    category: str
    movement_type: str  # "odds_swing", "capital_flow", "large_bet"
    severity: str  # "high", "medium", "low"
    
    # Movement details
    old_price: float
    new_price: float
    price_change_pct: float
    time_period_hours: float
    
    # Volume details
    volume_change: float
    volume_change_pct: float
    
    # Timing
    detected_at: str
    movement_start: str
    movement_end: str
    hours_to_close: Optional[float]
    
    # Context
    current_volume: float
    current_liquidity: float
    explanation: str


class MovementAnalyzer:
    """
    Analyzes market data for significant movements and anomalies
    """
    
    def __init__(
        self,
        db_path: str = "prediction_markets.db",
        alert_email: str = "sbornhor@uchicago.edu",
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None
    ):
        """
        Initialize Movement Analyzer
        
        Args:
            db_path: Path to SQLite database
            alert_email: Email to send alerts to
            smtp_server: SMTP server for sending emails
            smtp_port: SMTP port
            sender_email: Email to send from (will prompt if None)
            sender_password: Email password (will prompt if None)
        """
        self.db_path = db_path
        self.alert_email = alert_email
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        
        # Analysis parameters
        self.odds_swing_threshold = 0.30  # 30% change
        self.odds_swing_window_hours = 6
        self.large_flow_threshold_pct = 0.50  # 50% volume increase
        self.large_flow_window_hours = 6
        self.ignore_close_hours = 6  # Ignore movements within 6 hours of close
        self.min_market_volume = 4000  # Minimum $4k market volume to consider
        
        # Initialize database for tracking
        self._init_tracking_database()
        
        logger.info("Movement Analyzer initialized")
    
    def _init_tracking_database(self):
        """Initialize database tables for movement tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historical snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                yes_price REAL,
                no_price REAL,
                volume REAL,
                liquidity REAL,
                status TEXT,
                UNIQUE(market_id, timestamp)
            )
        """)
        
        # Detected movements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detected_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                question TEXT,
                movement_type TEXT,
                severity TEXT,
                old_price REAL,
                new_price REAL,
                price_change_pct REAL,
                volume_change REAL,
                detected_at TEXT,
                movement_start TEXT,
                movement_end TEXT,
                explanation TEXT,
                alerted BOOLEAN DEFAULT 0
            )
        """)
        
        # Alert history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_date TEXT,
                num_movements INTEGER,
                email_sent BOOLEAN,
                email_content TEXT,
                sent_at TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_market ON market_snapshots(market_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_time ON market_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_movement_detected ON detected_movements(detected_at)")
        
        conn.commit()
        conn.close()
        logger.info("Tracking database initialized")
    
    def take_snapshot(self):
        """
        Take a snapshot of current market state for comparison
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all active markets
        cursor.execute("""
            SELECT market_id, yes_price, no_price, volume, liquidity, status
            FROM markets
            WHERE status IN ('active', 'closed')
        """)
        
        markets = cursor.fetchall()
        timestamp = datetime.now().isoformat()
        
        # Insert snapshots
        for market in markets:
            market_id, yes_price, no_price, volume, liquidity, status = market
            
            cursor.execute("""
                INSERT OR REPLACE INTO market_snapshots 
                (market_id, timestamp, yes_price, no_price, volume, liquidity, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (market_id, timestamp, yes_price, no_price, volume, liquidity, status))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Snapshot taken for {len(markets)} markets")
        return len(markets)
    
    def _get_hours_to_close(self, market_id: str) -> Optional[float]:
        """Get hours until market closes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT end_date FROM markets WHERE market_id = ?
        """, (market_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            try:
                end_date = datetime.fromisoformat(result[0].replace('Z', '+00:00'))
                hours_to_close = (end_date - datetime.now(pytz.UTC)).total_seconds() / 3600
                return hours_to_close
            except:
                return None
        return None
    
    def _should_ignore_movement(self, market_id: str) -> bool:
        """Check if movement should be ignored (too close to closing)"""
        hours_to_close = self._get_hours_to_close(market_id)
        
        if hours_to_close is None:
            return False  # If we can't determine, don't ignore
        
        if hours_to_close < self.ignore_close_hours:
            logger.debug(f"Ignoring movement for {market_id} - only {hours_to_close:.1f} hours to close")
            return True
        
        return False
    
    def detect_odds_swings(self, lookback_hours: float = 6) -> List[MarketMovement]:
        """
        Detect large odds swings in the specified time window
        
        Args:
            lookback_hours: Hours to look back for comparison
            
        Returns:
            List of MarketMovement objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        movements = []
        cutoff_time = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()
        
        # Get markets with snapshots in the lookback period
        cursor.execute("""
            SELECT DISTINCT market_id FROM market_snapshots
            WHERE timestamp >= ?
        """, (cutoff_time,))
        
        market_ids = [row[0] for row in cursor.fetchall()]
        
        for market_id in market_ids:
            # Skip if too close to closing
            if self._should_ignore_movement(market_id):
                continue
            
            # Get oldest and newest snapshots in window
            cursor.execute("""
                SELECT timestamp, yes_price, volume
                FROM market_snapshots
                WHERE market_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (market_id, cutoff_time))
            
            snapshots = cursor.fetchall()
            
            if len(snapshots) < 2:
                continue
            
            # Compare first and last snapshot
            old_time, old_price, old_volume = snapshots[0]
            new_time, new_price, new_volume = snapshots[-1]
            
            if old_price is None or new_price is None or old_price == 0:
                continue
            
            # Calculate price change
            price_change = abs(new_price - old_price)
            price_change_pct = price_change / old_price
            
            # Check if it exceeds threshold
            if price_change_pct >= self.odds_swing_threshold:
                # Get market details
                cursor.execute("""
                    SELECT question, category, volume, liquidity
                    FROM markets WHERE market_id = ?
                """, (market_id,))
                
                market_details = cursor.fetchone()
                if not market_details:
                    continue
                
                question, category, current_volume, current_liquidity = market_details

                # Skip markets with insufficient volume
                if (current_volume or 0) < self.min_market_volume:
                    continue

                # Calculate time period
                time_period = (datetime.fromisoformat(new_time) -
                             datetime.fromisoformat(old_time)).total_seconds() / 3600
                
                # Determine severity
                if price_change_pct >= 0.50:
                    severity = "high"
                elif price_change_pct >= 0.40:
                    severity = "medium"
                else:
                    severity = "low"
                
                volume_change = new_volume - old_volume if old_volume else 0
                volume_change_pct = volume_change / old_volume if old_volume and old_volume > 0 else 0
                
                hours_to_close = self._get_hours_to_close(market_id)
                
                explanation = (
                    f"Odds shifted from {old_price:.1%} to {new_price:.1%} "
                    f"({price_change_pct:+.1%}) in {time_period:.1f} hours"
                )
                
                movement = MarketMovement(
                    market_id=market_id,
                    question=question,
                    category=category,
                    movement_type="odds_swing",
                    severity=severity,
                    old_price=old_price,
                    new_price=new_price,
                    price_change_pct=price_change_pct,
                    time_period_hours=time_period,
                    volume_change=volume_change,
                    volume_change_pct=volume_change_pct,
                    detected_at=datetime.now().isoformat(),
                    movement_start=old_time,
                    movement_end=new_time,
                    hours_to_close=hours_to_close,
                    current_volume=current_volume,
                    current_liquidity=current_liquidity,
                    explanation=explanation
                )
                
                movements.append(movement)
        
        conn.close()
        logger.info(f"Detected {len(movements)} odds swings")
        return movements
    
    def detect_capital_flows(self, lookback_hours: float = 6) -> List[MarketMovement]:
        """
        Detect large capital inflows in short periods
        
        Args:
            lookback_hours: Hours to look back for comparison
            
        Returns:
            List of MarketMovement objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        movements = []
        cutoff_time = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()
        
        # Get markets with volume changes
        cursor.execute("""
            SELECT DISTINCT market_id FROM market_snapshots
            WHERE timestamp >= ?
        """, (cutoff_time,))
        
        market_ids = [row[0] for row in cursor.fetchall()]
        
        for market_id in market_ids:
            # Skip if too close to closing
            if self._should_ignore_movement(market_id):
                continue
            
            # Get volume progression
            cursor.execute("""
                SELECT timestamp, volume, yes_price
                FROM market_snapshots
                WHERE market_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (market_id, cutoff_time))
            
            snapshots = cursor.fetchall()
            
            if len(snapshots) < 2:
                continue
            
            old_time, old_volume, old_price = snapshots[0]
            new_time, new_volume, new_price = snapshots[-1]
            
            if old_volume is None or new_volume is None or old_volume == 0:
                continue
            
            # Calculate volume change
            volume_change = new_volume - old_volume
            volume_change_pct = volume_change / old_volume
            
            # Check if it exceeds threshold
            if volume_change_pct >= self.large_flow_threshold_pct and volume_change > 1000:
                # Get market details
                cursor.execute("""
                    SELECT question, category, volume, liquidity
                    FROM markets WHERE market_id = ?
                """, (market_id,))

                market_details = cursor.fetchone()
                if not market_details:
                    continue

                question, category, current_volume, current_liquidity = market_details

                # Skip markets with insufficient volume
                if (current_volume or 0) < self.min_market_volume:
                    continue

                time_period = (datetime.fromisoformat(new_time) -
                             datetime.fromisoformat(old_time)).total_seconds() / 3600
                
                # Determine severity
                if volume_change_pct >= 1.0:  # 100%+ increase
                    severity = "high"
                elif volume_change_pct >= 0.75:
                    severity = "medium"
                else:
                    severity = "low"
                
                price_change_pct = (new_price - old_price) / old_price if old_price and old_price > 0 else 0
                hours_to_close = self._get_hours_to_close(market_id)
                
                explanation = (
                    f"Volume surged ${volume_change:,.0f} ({volume_change_pct:+.1%}) "
                    f"in {time_period:.1f} hours. Current volume: ${current_volume:,.0f}"
                )
                
                movement = MarketMovement(
                    market_id=market_id,
                    question=question,
                    category=category,
                    movement_type="capital_flow",
                    severity=severity,
                    old_price=old_price or 0,
                    new_price=new_price or 0,
                    price_change_pct=price_change_pct,
                    time_period_hours=time_period,
                    volume_change=volume_change,
                    volume_change_pct=volume_change_pct,
                    detected_at=datetime.now().isoformat(),
                    movement_start=old_time,
                    movement_end=new_time,
                    hours_to_close=hours_to_close,
                    current_volume=current_volume,
                    current_liquidity=current_liquidity,
                    explanation=explanation
                )
                
                movements.append(movement)
        
        conn.close()
        logger.info(f"Detected {len(movements)} capital flow events")
        return movements
    
    def save_movements(self, movements: List[MarketMovement]):
        """Save detected movements to database"""
        if not movements:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for movement in movements:
            cursor.execute("""
                INSERT INTO detected_movements (
                    market_id, question, movement_type, severity,
                    old_price, new_price, price_change_pct, volume_change,
                    detected_at, movement_start, movement_end, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                movement.market_id,
                movement.question,
                movement.movement_type,
                movement.severity,
                movement.old_price,
                movement.new_price,
                movement.price_change_pct,
                movement.volume_change,
                movement.detected_at,
                movement.movement_start,
                movement.movement_end,
                movement.explanation
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(movements)} movements to database")
    
    def analyze_last_24_hours(self) -> Dict:
        """
        Analyze movements from the last 24 hours
        
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting 24-hour analysis")
        
        # Detect odds swings
        odds_swings = self.detect_odds_swings(lookback_hours=6)
        
        # Detect capital flows
        capital_flows = self.detect_capital_flows(lookback_hours=6)
        
        # Combine and deduplicate
        all_movements = odds_swings + capital_flows
        
        # Save to database
        self.save_movements(all_movements)
        
        # Sort by severity and magnitude
        all_movements.sort(
            key=lambda x: (
                {'high': 0, 'medium': 1, 'low': 2}[x.severity],
                -abs(x.price_change_pct)
            )
        )
        
        # Group by category
        by_category = {}
        for movement in all_movements:
            if movement.category not in by_category:
                by_category[movement.category] = []
            by_category[movement.category].append(movement)
        
        return {
            'total_movements': len(all_movements),
            'odds_swings': len(odds_swings),
            'capital_flows': len(capital_flows),
            'movements': all_movements,
            'by_category': by_category,
            'high_severity': len([m for m in all_movements if m.severity == 'high']),
            'medium_severity': len([m for m in all_movements if m.severity == 'medium']),
            'low_severity': len([m for m in all_movements if m.severity == 'low'])
        }
    
    def generate_email_report(self, analysis: Dict) -> str:
        """
        Generate HTML email report from analysis
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            HTML email content
        """
        movements = analysis['movements']
        
        if not movements:
            return """
            <html>
            <body>
                <h2>Polymarket Movement Alert - No Significant Activity</h2>
                <p>No significant market movements detected in the last 24 hours.</p>
                <p>Analysis completed at: {}</p>
            </body>
            </html>
            """.format(datetime.now().strftime('%Y-%m-%d %I:%M %p CST'))
        
        # Build HTML report
        report_date = datetime.now().strftime('%Y-%m-%d %I:%M %p CST')
        total = analysis['total_movements']
        swings = analysis['odds_swings']
        flows = analysis['capital_flows']
        high = analysis['high_severity']
        med = analysis['medium_severity']
        low = analysis['low_severity']

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h2 {{ color: #2c3e50; }}
                .summary {{ background-color: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .movement {{
                    border: 1px solid #bdc3c7;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                    background-color: #ffffff;
                }}
                .high {{ border-left: 5px solid #e74c3c; }}
                .medium {{ border-left: 5px solid #f39c12; }}
                .low {{ border-left: 5px solid #3498db; }}
                .market-question {{ font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
                .details {{ color: #7f8c8d; font-size: 14px; }}
                .metric {{ display: inline-block; margin-right: 20px; }}
                .category {{ background-color: #3498db; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h2>Polymarket Movement Alert - Daily Summary</h2>
            <p><strong>Report Date:</strong> {report_date}</p>

            <div class="summary">
                <h3>Summary</h3>
                <p><strong>Total Significant Movements:</strong> {total}</p>
                <p><strong>Odds Swings (>30% in &lt;6hrs):</strong> {swings}</p>
                <p><strong>Capital Flow Surges (>50% in &lt;6hrs):</strong> {flows}</p>
                <p><strong>Severity Breakdown:</strong> {high} High | {med} Medium | {low} Low</p>
            </div>

            <h3>Detected Movements</h3>
        """
        
        # Add each movement
        for i, movement in enumerate(movements[:20], 1):  # Limit to top 20
            severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🔵'}[movement.severity]
            
            html += f"""
            <div class="movement {movement.severity}">
                <div class="market-question">
                    {severity_icon} {i}. {movement.question}
                </div>
                <div class="details">
                    <span class="category">{movement.category}</span>
                    <br><br>
                    <div class="metric"><strong>Type:</strong> {movement.movement_type.replace('_', ' ').title()}</div>
                    <div class="metric"><strong>Severity:</strong> {movement.severity.upper()}</div>
                    <br>
                    <div class="metric"><strong>Price Change:</strong> {movement.old_price:.1%} → {movement.new_price:.1%} ({movement.price_change_pct:+.1%})</div>
                    <div class="metric"><strong>Time Period:</strong> {movement.time_period_hours:.1f} hours</div>
                    <br>
                    <div class="metric"><strong>Volume Change:</strong> ${movement.volume_change:,.0f} ({movement.volume_change_pct:+.1%})</div>
                    <div class="metric"><strong>Current Volume:</strong> ${movement.current_volume:,.0f}</div>
                    <br>
                    <p><em>{movement.explanation}</em></p>
            """
            
            if movement.hours_to_close is not None:
                html += f'<p><strong>Time to Close:</strong> {movement.hours_to_close:.1f} hours</p>'
            
            html += """
                </div>
            </div>
            """
        
        if len(movements) > 20:
            html += f"<p><em>... and {len(movements) - 20} more movements</em></p>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def send_email_alert(self, subject: str, html_content: str) -> bool:
        """
        Send email alert
        
        Args:
            subject: Email subject
            html_content: HTML content for email body
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email or "polymarket-alerts@example.com"
            msg['To'] = self.alert_email
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email (if credentials provided)
            if self.sender_email and self.sender_password:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                server.quit()
                
                logger.info(f"Email alert sent to {self.alert_email}")
                return True
            else:
                # Save to file instead
                filename = f"email_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                with open(filename, 'w') as f:
                    f.write(html_content)
                logger.info(f"Email credentials not provided - saved to {filename}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_daily_alert(self):
        """Generate and send daily alert email"""
        logger.info("Generating daily alert")
        
        # Run analysis
        analysis = self.analyze_last_24_hours()
        
        # Generate email
        html_content = self.generate_email_report(analysis)
        
        subject = f"Polymarket Alert: {analysis['total_movements']} Significant Movements Detected"
        
        if analysis['total_movements'] == 0:
            subject = "Polymarket Alert: No Significant Movements"
        
        # Send email
        success = self.send_email_alert(subject, html_content)
        
        # Log alert
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alert_history (
                alert_date, num_movements, email_sent, email_content, sent_at
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().date().isoformat(),
            analysis['total_movements'],
            success,
            html_content,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return success
    
    def run_continuous(self):
        """
        Run continuous monitoring with scheduled alerts
        Takes snapshots every hour and sends daily alert at 4 PM CST
        """
        logger.info("Starting continuous monitoring")
        
        # Define CST timezone
        cst = pytz.timezone('America/Chicago')
        
        # Schedule snapshot every hour
        schedule.every().hour.do(self.take_snapshot)
        
        # Schedule daily alert at 4 PM CST
        schedule.every().day.at("16:00").do(self.send_daily_alert)
        
        # Take initial snapshot
        self.take_snapshot()
        
        logger.info("Scheduled: Snapshots every hour, Daily alert at 4 PM CST")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Movement Analysis Agent')
    parser.add_argument('--mode', choices=['analyze', 'continuous', 'test'], 
                       default='test', help='Operation mode')
    parser.add_argument('--email', default='sbornhor@uchicago.edu', 
                       help='Alert email address')
    parser.add_argument('--sender-email', help='Sender email address (optional)')
    parser.add_argument('--sender-password', help='Sender email password (optional)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MovementAnalyzer(
        alert_email=args.email,
        sender_email=args.sender_email,
        sender_password=args.sender_password
    )
    
    if args.mode == 'test':
        # Test mode - run analysis and save report
        print("Running test analysis...")
        analyzer.take_snapshot()
        analysis = analyzer.analyze_last_24_hours()
        
        print(f"\n{'='*60}")
        print(f"Analysis Results:")
        print(f"{'='*60}")
        print(f"Total Movements: {analysis['total_movements']}")
        print(f"Odds Swings: {analysis['odds_swings']}")
        print(f"Capital Flows: {analysis['capital_flows']}")
        print(f"High Severity: {analysis['high_severity']}")
        print(f"Medium Severity: {analysis['medium_severity']}")
        print(f"Low Severity: {analysis['low_severity']}")
        
        # Generate and save report
        html = analyzer.generate_email_report(analysis)
        filename = f"movement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w') as f:
            f.write(html)
        print(f"\nReport saved to: {filename}")
        
    elif args.mode == 'analyze':
        # One-time analysis and alert
        print("Running analysis and sending alert...")
        analyzer.send_daily_alert()
        
    elif args.mode == 'continuous':
        # Continuous monitoring
        print("Starting continuous monitoring...")
        print("Snapshots: Every hour")
        print("Daily alerts: 4 PM CST")
        analyzer.run_continuous()
