# Complete Setup Guide - Polymarket Monitoring System

This guide will walk you through setting up your prediction market arbitrage agent from scratch.

## Table of Contents
1. [Choose Your Setup Method](#choose-your-setup-method)
2. [Method A: Claude Code (Recommended)](#method-a-claude-code-recommended)
3. [Method B: Local Computer Setup](#method-b-local-computer-setup)
4. [First Run & Testing](#first-run--testing)
5. [Understanding the System](#understanding-the-system)
6. [Troubleshooting](#troubleshooting)

---

## Choose Your Setup Method

### Method A: Claude Code (Recommended) ⭐
**Best for:** Quick setup, no local Python installation needed, easy to manage

**Pros:**
- No need to install Python locally
- Built-in terminal and code editor
- Easy file management
- Can run 24/7 (if you keep it open)

**Cons:**
- Requires Claude Pro subscription
- Needs to stay open for continuous monitoring

### Method B: Local Computer
**Best for:** Long-term production use, running 24/7

**Pros:**
- Can run in background
- Complete control
- Can use task scheduler for automatic startup

**Cons:**
- Need to install Python
- More setup steps
- Computer must stay on for 24/7 operation

---

## Method A: Claude Code (Recommended)

### Step 1: Open Claude Code

1. Go to claude.ai
2. Click on the "Code" icon in the sidebar (or press `Cmd/Ctrl + K`)
3. Claude Code will open with a terminal

### Step 2: Create Project Directory

In the Claude Code terminal, type:

```bash
# Create and enter project directory
mkdir polymarket-monitor
cd polymarket-monitor
```

### Step 3: Download Project Files

You have all the files already! Here's how to get them into Claude Code:

**Option 1: Copy from this conversation**
1. Download all the files I shared (they're in the outputs)
2. In Claude Code, click "Upload files"
3. Upload all .py and .md files

**Option 2: Create files manually**
For each file, in Claude Code terminal:

```bash
# Create the file
touch polymarket_monitor.py

# Then edit it (Claude Code will open the editor)
# Copy the content from the file I provided
```

The files you need:
- `polymarket_monitor.py`
- `movement_analyzer.py`
- `integrated_system.py`
- `config.py`
- `analyzer_config.py`
- `test_monitor.py`
- `requirements.txt`

### Step 4: Install Dependencies

In Claude Code terminal:

```bash
# Install required packages
pip install requests pytz schedule
```

If you get a "pip not found" error:
```bash
python3 -m pip install requests pytz schedule
```

### Step 5: Verify Installation

```bash
# Check Python is installed
python --version

# Should show something like: Python 3.9.x or higher
```

### Step 6: Run First Test

```bash
# Run the integrated test
python integrated_system.py --mode test
```

**What you should see:**
```
==============================================================
QUICK TEST - Integrated Monitoring System
==============================================================

Step 1: Collecting market data from Polymarket...
✓ Market data collected

Step 2: Taking snapshot for movement analysis...
✓ Snapshot taken for 150 markets

...
```

### Step 7: Check the Results

After the test completes, you'll see:
- `prediction_markets.db` - Database with market data
- `test_report_YYYYMMDD_HHMMSS.html` - Generated report

In Claude Code:
```bash
# List files to see the report
ls -la

# The report will be named something like:
# test_report_20260131_143022.html
```

### Step 8: View the Report

1. In Claude Code, click on the HTML file
2. Right-click and select "Download"
3. Open it in your browser

You should see a professional HTML report with market movements!

### Step 9: Configure Email (Optional)

If you want actual emails sent to sbornhor@uchicago.edu:

1. Open `analyzer_config.py` in Claude Code
2. Follow the email setup guide (EMAIL_SETUP_GUIDE.md)
3. Add your Gmail and App Password:

```python
SENDER_EMAIL = "your-email@gmail.com"
SENDER_PASSWORD = "your-16-char-app-password"
```

### Step 10: Run Production Mode

```bash
# Start the full monitoring system
python integrated_system.py --mode production
```

**This will:**
- Poll Polymarket every 60 seconds
- Take snapshots every hour
- Send daily email at 4 PM CST

To stop: Press `Ctrl+C`

---

## Method B: Local Computer Setup

### Prerequisites

You'll need:
- Python 3.8 or higher
- pip (Python package manager)
- A text editor (VS Code recommended, or any editor you like)

### Step 1: Check if Python is Installed

**Mac/Linux:**
```bash
python3 --version
```

**Windows:**
```bash
python --version
```

Should show Python 3.8 or higher. If not installed:
- **Mac:** Install from python.org or use `brew install python3`
- **Windows:** Download from python.org
- **Linux:** `sudo apt install python3 python3-pip`

### Step 2: Create Project Directory

**Mac/Linux:**
```bash
mkdir ~/polymarket-monitor
cd ~/polymarket-monitor
```

**Windows (PowerShell):**
```bash
mkdir C:\Users\YourName\polymarket-monitor
cd C:\Users\YourName\polymarket-monitor
```

### Step 3: Save Project Files

1. Download all files from this conversation
2. Put them in your project directory
3. You should have:
   - polymarket_monitor.py
   - movement_analyzer.py
   - integrated_system.py
   - config.py
   - analyzer_config.py
   - test_monitor.py
   - requirements.txt

### Step 4: Create Virtual Environment (Recommended)

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run First Test

**Mac/Linux:**
```bash
python3 integrated_system.py --mode test
```

**Windows:**
```bash
python integrated_system.py --mode test
```

### Step 7: Check Results

Look for:
- `prediction_markets.db` - Created
- `test_report_*.html` - Generated report
- No errors in terminal

### Step 8: View Report

Open the HTML file in your browser:

**Mac:**
```bash
open test_report_*.html
```

**Windows:**
```bash
start test_report_*.html
```

**Linux:**
```bash
xdg-open test_report_*.html
```

### Step 9: Run Production Mode

```bash
python integrated_system.py --mode production
```

### Step 10: Keep It Running 24/7

**Option 1: Keep terminal open**
- Simple but requires terminal to stay open

**Option 2: Use screen/tmux (Mac/Linux)**
```bash
# Install screen
sudo apt install screen  # Linux
brew install screen      # Mac

# Start screen session
screen -S polymarket

# Run the system
python integrated_system.py --mode production

# Detach: Press Ctrl+A then D
# Reattach later: screen -r polymarket
```

**Option 3: Windows Task Scheduler**
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: At startup
4. Action: Start a program
5. Program: `C:\Users\YourName\polymarket-monitor\venv\Scripts\python.exe`
6. Arguments: `integrated_system.py --mode production`
7. Start in: `C:\Users\YourName\polymarket-monitor`

**Option 4: macOS launchd**
Create file `~/Library/LaunchAgents/com.polymarket.monitor.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.polymarket.monitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/Users/YourName/polymarket-monitor/integrated_system.py</string>
        <string>--mode</string>
        <string>production</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.polymarket.monitor.plist
```

---

## First Run & Testing

### What to Expect on First Run

1. **First 5 minutes:**
   - System connects to Polymarket
   - Downloads current market data
   - Creates database
   - Takes first snapshot

2. **First hour:**
   - Continues polling every 60 seconds
   - Takes second snapshot after 1 hour
   - Now can detect movements!

3. **First day:**
   - At 4 PM CST, generates first daily report
   - If email configured: sends email
   - If not: saves HTML file

### Test Without Waiting

Want to test the movement detection immediately?

```bash
# Run this to see what the system does
python test_monitor.py

# This will:
# - Fetch current market data
# - Display statistics
# - Show high-volume markets
```

### Manual Analysis

To run analysis manually without waiting for 4 PM:

```bash
python movement_analyzer.py --mode analyze
```

---

## Understanding the System

### File Structure

```
polymarket-monitor/
├── polymarket_monitor.py      # Agent 1: Collects market data
├── movement_analyzer.py        # Agent 2: Analyzes movements
├── integrated_system.py        # Runs both together
├── config.py                   # Configuration for Agent 1
├── analyzer_config.py          # Configuration for Agent 2
├── test_monitor.py             # Testing utilities
├── requirements.txt            # Python dependencies
├── prediction_markets.db       # Database (created automatically)
├── *.log                       # Log files (created automatically)
└── *.html                      # Reports (created automatically)
```

### How the Agents Work Together

```
┌─────────────────────────────────────────────────┐
│         Integrated System                       │
├─────────────────────────────────────────────────┤
│                                                 │
│  Thread 1: Platform Monitor                     │
│  ┌───────────────────────────────────────────┐ │
│  │ Every 60 seconds:                         │ │
│  │ 1. Fetch markets from Polymarket          │ │
│  │ 2. Normalize data                         │ │
│  │ 3. Save to database                       │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  Thread 2: Movement Analyzer                    │
│  ┌───────────────────────────────────────────┐ │
│  │ Every hour:                               │ │
│  │ 1. Take snapshot of current state         │ │
│  │ 2. Compare with previous snapshots        │ │
│  │ 3. Detect significant movements           │ │
│  │                                           │ │
│  │ Every day at 4 PM CST:                    │ │
│  │ 4. Generate 24-hour summary               │ │
│  │ 5. Send email to sbornhor@uchicago.edu   │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
└─────────────────────────────────────────────────┘
```

### What Each File Does

**polymarket_monitor.py** - Platform Monitor Agent
- Connects to Polymarket API
- Fetches market data every 60 seconds
- Normalizes data to standard format
- Saves to SQLite database
- Handles errors and rate limiting

**movement_analyzer.py** - Movement Analysis Agent
- Takes hourly snapshots
- Compares snapshots to detect movements
- Identifies odds swings and capital flows
- Generates HTML email reports
- Sends daily alerts at 4 PM CST

**integrated_system.py** - Integration Layer
- Runs both agents in parallel
- Coordinates their activities
- Provides unified interface
- Handles logging and status

**config.py** - Monitor Configuration
- Polling frequency
- Rate limits
- Database path

**analyzer_config.py** - Analyzer Configuration
- Alert email
- Detection thresholds
- SMTP settings
- Alert schedule

### Database Tables

The system creates these tables automatically:

1. **markets** - Current state of all markets
2. **market_snapshots** - Historical snapshots for comparison
3. **detected_movements** - All significant movements found
4. **agent_state** - System state (last poll time, errors)
5. **error_log** - All errors encountered
6. **alert_history** - Record of sent alerts

---

## Troubleshooting

### "Module not found" error

```bash
# Make sure you installed dependencies
pip install -r requirements.txt

# Or install individually
pip install requests pytz schedule
```

### "Permission denied" error

**Mac/Linux:**
```bash
# Make scripts executable
chmod +x *.py
```

**Windows:**
- Run PowerShell as Administrator

### "Database is locked" error

- Only run one instance at a time
- Close any SQLite browser tools
- Restart the system

### No markets being fetched

**Check internet connection:**
```bash
# Test API connectivity
curl https://gamma-api.polymarket.com/markets?limit=1
```

**Check logs:**
```bash
# View monitor log
tail -f polymarket_monitor.log

# Look for errors
```

### Email not sending

1. Check `movement_analyzer.log` for errors
2. Verify email settings in `analyzer_config.py`
3. Try test mode first: `python movement_analyzer.py --mode test`
4. Reports are saved as HTML files even if email fails

### System stops running

**Check logs:**
```bash
tail -100 polymarket_monitor.log
tail -100 movement_analyzer.log
```

**Common causes:**
- Network interruption
- API rate limit hit
- System went to sleep (disable sleep for 24/7 operation)

### Want to reset everything?

```bash
# Delete database to start fresh
rm prediction_markets.db

# Delete logs
rm *.log

# Run test again
python integrated_system.py --mode test
```

---

## Quick Command Reference

```bash
# Test the system
python integrated_system.py --mode test

# Run production monitoring
python integrated_system.py --mode production

# One-time analysis
python movement_analyzer.py --mode analyze

# Test platform monitor only
python test_monitor.py

# Check what's in database
sqlite3 prediction_markets.db "SELECT COUNT(*) FROM markets"

# View recent movements
sqlite3 prediction_markets.db "SELECT * FROM detected_movements ORDER BY detected_at DESC LIMIT 5"

# Check logs
tail -f polymarket_monitor.log
tail -f movement_analyzer.log
```

---

## Next Steps After Setup

1. **Let it run for a few hours** to collect data
2. **Check the database** to see markets being tracked
3. **Wait for 4 PM CST** to receive first daily email
4. **Review the email** to understand the format
5. **Adjust thresholds** in `analyzer_config.py` if needed
6. **Add more platforms** when ready (Kalshi, Manifold, etc.)

---

## Getting Help

If you run into issues:

1. **Check the logs first** - They show what's happening
2. **Read the error message carefully** - Often tells you exactly what's wrong
3. **Try test mode** - Isolate if it's setup or runtime issue
4. **Review this guide** - Make sure you didn't skip a step
5. **Check the README files** - They have detailed troubleshooting sections

---

## For Your MBA Project

**What to Document:**
- Initial setup process (screenshots help!)
- First week of data collection
- Analysis of detected movements
- Accuracy of predictions vs actual outcomes
- Insights gained from monitoring
- Future enhancements planned

**Presentation Tips:**
- Show a live demo of the HTML reports
- Explain the technical architecture
- Discuss the business applications
- Highlight the automation aspects
- Show real movement examples

Good luck with your project! The system is ready to start monitoring Polymarket 24/7.
