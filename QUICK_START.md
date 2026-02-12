# Quick Start Checklist

Use this checklist to get your system running in minutes!

## ✅ Pre-Flight Checklist

### Step 1: Choose Your Environment
- [ ] **Using Claude Code?** → Continue below
- [ ] **Using local computer?** → See SETUP_GUIDE.md, Method B

---

## 🚀 Claude Code Setup (5 minutes)

### Step 2: Get the Files
- [ ] Open Claude Code (Cmd/Ctrl + K in claude.ai)
- [ ] Create new folder: `mkdir polymarket-monitor && cd polymarket-monitor`
- [ ] Download all .py and .txt files from this conversation
- [ ] Upload them to Claude Code using "Upload files" button

**Files you need (9 total):**
- [ ] polymarket_monitor.py
- [ ] movement_analyzer.py  
- [ ] integrated_system.py
- [ ] config.py
- [ ] analyzer_config.py
- [ ] test_monitor.py
- [ ] requirements.txt
- [ ] README.md
- [ ] MOVEMENT_ANALYZER_README.md

### Step 3: Install Dependencies
```bash
pip install requests pytz schedule
```
- [ ] Command completed without errors

### Step 4: Run Your First Test
```bash
python integrated_system.py --mode test
```

**You should see:**
- [ ] "Collecting market data from Polymarket..." 
- [ ] "Snapshot taken for XXX markets"
- [ ] "Analysis complete"
- [ ] "Report saved to: test_report_YYYYMMDD_HHMMSS.html"

### Step 5: View Your First Report
- [ ] Find the HTML file in Claude Code file browser
- [ ] Download it
- [ ] Open in your web browser
- [ ] See formatted report with market movements

**If you see a nice HTML report with market data, you're ready! 🎉**

---

## 🔧 Optional: Email Setup

Want actual emails sent to sbornhor@uchicago.edu?

### Step 6: Set Up Gmail App Password
- [ ] Go to https://myaccount.google.com/security
- [ ] Enable 2-Factor Authentication
- [ ] Go to https://myaccount.google.com/apppasswords
- [ ] Create app password for "Mail"
- [ ] Copy the 16-character password

### Step 7: Configure Email
In Claude Code, edit `analyzer_config.py`:

```python
SENDER_EMAIL = "your-email@gmail.com"
SENDER_PASSWORD = "your-16-char-app-password"
```

- [ ] Saved analyzer_config.py with your credentials

### Step 8: Test Email
```bash
python movement_analyzer.py --mode test
```

- [ ] Check sbornhor@uchicago.edu inbox
- [ ] Email received (or check spam folder)

**If no email setup:** Reports are saved as HTML files - totally fine for testing!

---

## 🏃 Production Mode

### Step 9: Start Continuous Monitoring
```bash
python integrated_system.py --mode production
```

**You should see:**
- [ ] "Starting market data monitor"
- [ ] "Starting movement analyzer"
- [ ] "Both agents running. Press Ctrl+C to stop."

### Step 10: Verify It's Working

**After 5 minutes:**
```bash
# In a new terminal tab
sqlite3 prediction_markets.db "SELECT COUNT(*) FROM markets"
```
- [ ] Shows a number > 0

**After 1 hour:**
```bash
sqlite3 prediction_markets.db "SELECT COUNT(*) FROM market_snapshots"
```
- [ ] Shows snapshots being taken

**At 4 PM CST:**
- [ ] Email arrives at sbornhor@uchicago.edu (or HTML file created)
- [ ] Contains summary of past 24 hours

---

## 📊 What Happens Now?

Your system is now:
- ✅ Polling Polymarket every 60 seconds
- ✅ Taking snapshots every hour
- ✅ Analyzing for significant movements
- ✅ Sending daily email at 4 PM CST

**Keep Claude Code open** for continuous monitoring, or set up to run 24/7 on a server.

---

## 🆘 Quick Troubleshooting

### "Module not found"
```bash
pip install requests pytz schedule
```

### "No markets fetched"
- Check internet connection
- Try: `curl https://gamma-api.polymarket.com/markets?limit=1`

### "Email not working"
- Don't worry! Reports save as HTML files
- Open the .html file in your browser
- Email is optional for testing

### "Database locked"
- Only run one instance at a time
- Restart: Delete `prediction_markets.db` and run test again

### Want to start over?
```bash
rm prediction_markets.db *.log *.html
python integrated_system.py --mode test
```

---

## 📚 Learning More

- **SETUP_GUIDE.md** - Detailed setup instructions
- **README.md** - Platform Monitor documentation  
- **MOVEMENT_ANALYZER_README.md** - Analyzer documentation
- **EMAIL_SETUP_GUIDE.md** - Email configuration help

---

## 🎓 For Your MBA Project

### Data to Collect:
- [ ] Daily email reports (save them!)
- [ ] Screenshots of significant movements
- [ ] Database queries showing trends
- [ ] Examples of odds swings detected
- [ ] Capital flow patterns identified

### Questions to Explore:
- Do prediction markets react faster than traditional news?
- What events cause the biggest movements?
- Are certain categories more volatile?
- Can we predict movements before they happen?
- How accurate are the crowd predictions?

---

## ✨ Success Criteria

You know it's working when:
- ✅ Database file grows over time
- ✅ Logs show regular polling activity  
- ✅ Snapshots accumulate each hour
- ✅ Daily email arrives at 4 PM CST
- ✅ Reports show actual market movements

**Congratulations! Your prediction market monitoring system is live! 🚀**

---

## 🔜 Next Steps

After the system runs for a day:
1. Review the movements detected
2. Adjust thresholds if too sensitive/not sensitive enough
3. Start thinking about arbitrage opportunities
4. Plan for adding more platforms (Kalshi, Manifold)
5. Document insights for your MBA project

**Questions?** Check the detailed guides or review the logs for what's happening.
