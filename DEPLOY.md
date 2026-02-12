# Deployment Guide: Polymarket Scanner on Railway + Cloudflare R2

## Overview

This guide deploys a scheduled Polymarket scanner that:
- Runs every hour (configurable)
- Fetches finance markets from Polymarket API
- Stores historical snapshots in SQLite
- Runs movement detection analysis
- Uploads results to Cloudflare R2 for your team to access

**Estimated cost: ~$5/month** (Railway Hobby plan)

---

## Step 1: Set Up Cloudflare R2 (Free Storage)

### 1.1 Create Cloudflare Account
1. Go to https://dash.cloudflare.com/sign-up
2. Create a free account (no credit card required for R2 free tier)

### 1.2 Create R2 Bucket
1. In Cloudflare dashboard, click **R2** in the left sidebar
2. Click **Create bucket**
3. Name it: `polymarket-data`
4. Click **Create bucket**

### 1.3 Create R2 API Token
1. In R2, click **Manage R2 API Tokens**
2. Click **Create API token**
3. Give it a name: `polymarket-scanner`
4. Permissions: **Object Read & Write**
5. Specify bucket: `polymarket-data`
6. Click **Create API Token**
7. **SAVE THESE VALUES** (you won't see them again):
   - Access Key ID
   - Secret Access Key

### 1.4 Get Your Account ID
1. In Cloudflare dashboard, your Account ID is in the right sidebar
2. Or go to any domain > Overview > right sidebar shows Account ID
3. **SAVE THIS VALUE**

### 1.5 Make Bucket Public (for team access)
1. Go to R2 > your bucket > **Settings**
2. Under **Public access**, click **Allow Access**
3. Note your public URL: `https://pub-XXXXX.r2.dev`

---

## Step 2: Set Up Railway

### 2.1 Create Railway Account
1. Go to https://railway.app
2. Sign up with GitHub (recommended) or email
3. Select **Hobby Plan** ($5/month)

### 2.2 Create New Project
1. Click **New Project**
2. Select **Deploy from GitHub repo**
3. Connect your GitHub account if not already
4. Select your repository (you'll push code in Step 3)

### 2.3 Configure Environment Variables
In your Railway project:
1. Click on your service
2. Go to **Variables** tab
3. Add these variables:

```
R2_ACCOUNT_ID=your_cloudflare_account_id
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=polymarket-data
SCAN_INTERVAL_MINUTES=60
MIN_MARKET_VOLUME=10000
```

---

## Step 3: Push Code to GitHub

### 3.1 Create GitHub Repository
1. Go to https://github.com/new
2. Name: `polymarket-scanner` (or whatever you prefer)
3. Make it **Private** (recommended)
4. Don't initialize with README

### 3.2 Push Your Code
Open terminal in your project folder and run:

```bash
cd "c:\Users\sambo\Downloads\Polymarket Arb"

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Polymarket scanner"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/polymarket-scanner.git

# Push
git push -u origin main
```

### 3.3 Deploy on Railway
1. Back in Railway, your project should auto-detect the new code
2. Railway will automatically build and deploy
3. Check **Deployments** tab for build logs
4. Check **Logs** tab for runtime logs

---

## Step 4: Verify Deployment

### 4.1 Check Railway Logs
1. In Railway, go to your service > **Logs**
2. You should see output like:
   ```
   Starting scheduled runner (interval: 60 minutes)
   Starting scan cycle at 2026-02-09T19:00:00
   Fetching markets from Polymarket API...
   Fetched 25000 total markets
   Filtered to 5000 finance markets
   ...
   Uploaded results to R2
   Scan cycle complete
   ```

### 4.2 Check R2 Files
1. Go to Cloudflare R2 > your bucket
2. You should see:
   - `latest_scan.json`
   - `latest_scan.csv`
   - `prediction_markets.db`
   - `archive/scan_results_TIMESTAMP.json`

### 4.3 Access Your Data
Your team can access data via public URLs:

**For AI Agents (JSON):**
```
https://pub-XXXXX.r2.dev/latest_scan.json
```

**For Excel Users (CSV):**
```
https://pub-XXXXX.r2.dev/latest_scan.csv
```

**For Direct DB Access (SQLite):**
```
https://pub-XXXXX.r2.dev/prediction_markets.db
```

---

## Step 5: Configure Your Team's Access

### Option A: Direct URL Access (Simplest)
Share the public R2 URLs with your team:
- JSON: `https://pub-XXXXX.r2.dev/latest_scan.json`
- CSV: `https://pub-XXXXX.r2.dev/latest_scan.csv`

### Option B: Python Client Example
```python
import requests
import json

# Fetch latest results
url = "https://pub-XXXXX.r2.dev/latest_scan.json"
response = requests.get(url)
data = response.json()

# Access summary
print(f"Last scan: {data['summary']['scan_timestamp']}")
print(f"Flagged markets: {data['summary']['stats']['deep_analyzed']}")

# Access markets
for market in data['markets'][:5]:
    print(f"- {market['question']}")
    print(f"  Score: {market['composite_score']}, Flags: {market['flags_triggered']}")
```

### Option C: Download DB for Local Analysis
```python
import requests
import sqlite3

# Download DB
url = "https://pub-XXXXX.r2.dev/prediction_markets.db"
response = requests.get(url)
with open('local_markets.db', 'wb') as f:
    f.write(response.content)

# Query locally
conn = sqlite3.connect('local_markets.db')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM market_snapshots")
print(f"Total snapshots: {cursor.fetchone()[0]}")
```

---

## Configuration Options

### Change Scan Interval
In Railway Variables, set:
```
SCAN_INTERVAL_MINUTES=30  # Run every 30 minutes
```

### Change Volume Threshold
```
MIN_MARKET_VOLUME=50000  # Only markets with $50k+ volume
```

### Adjust Analysis Weights
Edit `polymarket_scanner.py` in the `ScannerConfig` class:
```python
WEIGHT_MARKET_VOLUME = 0.40
WEIGHT_ODDS_SWING = 0.35
WEIGHT_VOLUME_SURGE = 0.25
```

Push changes to GitHub and Railway will auto-redeploy.

---

## Troubleshooting

### "R2 disabled - missing credentials"
- Check all 4 R2 environment variables are set correctly in Railway

### "API error at offset X"
- Polymarket API may be slow/rate limited
- The scanner will retry on next cycle

### Build fails on Railway
- Check `requirements.txt` has all dependencies
- Check Railway build logs for specific errors

### No data in R2
- Check Railway logs for upload errors
- Verify R2 API token has write permissions

---

## Cost Summary

| Service | Cost |
|---------|------|
| Railway Hobby | $5/month |
| Cloudflare R2 | Free (10GB storage, 10M reads) |
| **Total** | **~$5/month** |

---

## Files Reference

| File | Purpose |
|------|---------|
| `cloud_runner.py` | Main scheduled runner |
| `polymarket_scanner.py` | Analysis logic |
| `requirements.txt` | Python dependencies |
| `Procfile` | Railway start command |
| `railway.json` | Railway config |
| `.gitignore` | Files to exclude from git |
