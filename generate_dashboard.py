#!/usr/bin/env python3
"""
generate_dashboard.py — Build a self-contained HTML dashboard from pipeline outputs.

Reads the latest trend_analysis JSON, CSV, and investment report MD from
the output/ directory, then writes output/dashboard.html.

Usage:
    python generate_dashboard.py
    python generate_dashboard.py --open     # auto-open in browser
"""
import json
import csv
import argparse
import webbrowser
import urllib.request
import time
from pathlib import Path
from datetime import datetime, date

ROOT       = Path(__file__).parent
OUTPUT_DIR = ROOT / 'output'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_latest(pattern: str) -> Path | None:
    files = sorted(OUTPUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_data(trend_path: Path, csv_path: Path, report_path: Path | None):
    # CSV → market metadata keyed by market_id
    meta = {}
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mid = str(row.get('market_id', ''))
            def _f(k):
                v = row.get(k)
                try:    return float(v) if v else None
                except: return None
            meta[mid] = {
                'slug':     row.get('market_slug', ''),
                'category': row.get('category', ''),
                'price':    _f('current_price'),
                'vol24hr':  _f('volume24hr'),
                'end_date': row.get('end_date', ''),
            }

    # Trend analysis JSON
    with open(trend_path, encoding='utf-8') as f:
        trend = json.load(f)

    markets: dict = {}
    for p in trend.get('pairs', []):
        mid = str(p['polymarket_id'])
        if mid not in markets:
            m = meta.get(mid, {})
            markets[mid] = {
                'id':       mid,
                'question': p['polymarket_question'],
                'category': m.get('category', ''),
                'slug':     m.get('slug', ''),
                'price':    m.get('price'),
                'vol24hr':  m.get('vol24hr'),
                'end_date': m.get('end_date', ''),
                'pairs':    [],
            }
        corr = p.get('correlation') or {}
        ses  = p.get('spike_event_study') or {}
        markets[mid]['pairs'].append({
            'ticker':      p['ticker'],
            'ticker_name': p['ticker_name'],
            'score':       p.get('overall_similarity_score') or 0.0,
            'confidence':  p.get('confidence_level', 'low'),
            'summary':     p.get('agent_summary', ''),
            'findings':    p.get('key_findings', []),
            'caveats':     p.get('caveats', []),
            'discovery':   ses.get('discovery_direction', ''),
            'pearson_r':   corr.get('pearson_r'),
            'n_obs':       p.get('n_observations', 0),
        })

    for m in markets.values():
        m['pairs'].sort(key=lambda p: p['score'], reverse=True)
        b = m['pairs'][0] if m['pairs'] else {}
        m['best_ticker']      = b.get('ticker', '')
        m['best_ticker_name'] = b.get('ticker_name', '')
        m['best_score']       = b.get('score', 0.0)
        m['best_confidence']  = b.get('confidence', 'low')

    market_list = sorted(
        markets.values(),
        key=lambda m: (m['best_score'], m['vol24hr'] or 0),
        reverse=True,
    )

    # Investment report markdown
    report_md  = ''
    report_mid = ''
    if report_path and report_path.exists():
        report_md = report_path.read_text(encoding='utf-8')
        for line in report_md.splitlines():
            if line.startswith('# '):
                title = line[2:].strip().lower()
                for m in market_list:
                    if title[:40] in m['question'].lower() or m['question'].lower()[:40] in title:
                        report_mid = m['id']
                        break
                break

    # Filter out markets that have already closed
    today = date.today().isoformat()
    market_list = [m for m in market_list if not m['end_date'] or m['end_date'][:10] >= today]

    return market_list, report_md, report_mid, trend.get('analysis_timestamp', '')


def fetch_slugs_from_gamma(market_list: list) -> None:
    """Fetch missing slugs from Polymarket Gamma API (in-place update)."""
    missing = [m for m in market_list if not m.get('slug')]
    if not missing:
        return
    print(f'  Fetching Polymarket slugs for {len(missing)} markets...')
    fetched = 0
    for m in missing:
        try:
            url = f'https://gamma-api.polymarket.com/markets/{m["id"]}'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=6) as r:
                data = json.loads(r.read())
                if isinstance(data, list):
                    data = data[0] if data else {}
                slug = data.get('slug', '') or data.get('groupSlug', '')
                if slug:
                    m['slug'] = slug
                    fetched += 1
        except Exception:
            pass
        time.sleep(0.05)
    print(f'  Got slugs for {fetched}/{len(missing)} markets')


# ---------------------------------------------------------------------------
# HTML template  (placeholders replaced at runtime — no f-string)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polymarket Intelligence Dashboard</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0d1117; color: #c9d1d9;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}

/* Header */
.header {
  background: #161b22; border-bottom: 1px solid #30363d;
  padding: 10px 20px; display: flex; align-items: center; gap: 16px; flex-shrink: 0;
}
.header h1 { font-size: 15px; font-weight: 600; color: #e6edf3; }
.header-stat { font-size: 12px; color: #7d8590; background: #21262d; padding: 2px 8px; border-radius: 10px; }
.header-ts   { font-size: 11px; color: #7d8590; margin-left: auto; }

/* Layout */
.main { display: flex; flex: 1; overflow: hidden; }

/* Feed */
.feed { width: 360px; flex-shrink: 0; overflow-y: auto; border-right: 1px solid #30363d; }
.feed::-webkit-scrollbar { width: 4px; }
.feed::-webkit-scrollbar-track { background: #0d1117; }
.feed::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }

/* Market card */
.market-card {
  padding: 13px 16px; border-bottom: 1px solid #21262d;
  cursor: pointer; transition: background 0.1s; position: relative;
}
.market-card:hover { background: #161b22; }
.market-card.active { background: #1c2128; }
.market-card.active::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0;
  width: 3px; background: #388bfd;
}
.card-question {
  font-size: 13px; color: #e6edf3; line-height: 1.4; margin-bottom: 8px;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.card-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; align-items: center; }
.card-stats { display: flex; gap: 8px; align-items: center; }
.card-price { font-size: 15px; font-weight: 700; color: #58a6ff; }
.card-sep   { color: #30363d; }
.card-ticker { font-size: 11px; font-family: monospace; color: #7d8590; }
.card-score-wrap { margin-left: auto; text-align: right; }
.card-score-num   { font-size: 13px; font-weight: 600; color: #e6edf3; }
.card-score-label { font-size: 10px; color: #7d8590; }

/* Badges */
.badge {
  display: inline-block; padding: 2px 7px; border-radius: 12px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.3px; white-space: nowrap;
}
.conf-high   { background: #1a4731; color: #3fb950; }
.conf-medium { background: #3d2b00; color: #d29922; }
.conf-low    { background: #3d0f0f; color: #f85149; }
.cat-fed      { background: #0c2d4d; color: #58a6ff; }
.cat-politics { background: #2b1b56; color: #bc8cff; }
.cat-geo      { background: #3b1f0a; color: #e3b341; }
.cat-macro    { background: #0b2e2a; color: #39d353; }
.cat-fin      { background: #102010; color: #3fb950; }
.cat-social   { background: #1c2128; color: #8b949e; }

/* Detail panel */
.detail { flex: 1; overflow-y: auto; }
.detail::-webkit-scrollbar { width: 4px; }
.detail::-webkit-scrollbar-track { background: #0d1117; }
.detail::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }

.detail-header {
  background: #161b22; border-bottom: 1px solid #30363d;
  padding: 16px 24px; position: sticky; top: 0; z-index: 10;
}
.detail-question { font-size: 17px; font-weight: 600; color: #e6edf3; line-height: 1.4; margin-bottom: 10px; }
.detail-meta-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
.price-badge { font-size: 18px; font-weight: 700; color: #58a6ff; }
.meta-chip   { font-size: 11px; color: #7d8590; background: #21262d; padding: 2px 8px; border-radius: 12px; }
.links-bar   { display: flex; gap: 10px; flex-wrap: wrap; }
.ext-link {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 14px; border-radius: 6px; font-size: 12px; font-weight: 500;
  text-decoration: none; transition: opacity 0.15s;
}
.ext-link:hover { opacity: 0.8; }
.poly-link  { background: #0048ff18; color: #58a6ff; border: 1px solid #388bfd40; }
.yahoo-link { background: #56029418; color: #bc8cff; border: 1px solid #8957e540; }

.detail-body { padding: 20px 24px; }

/* Sections */
.section { margin-bottom: 24px; }
.section-title {
  font-size: 11px; font-weight: 600; color: #7d8590;
  text-transform: uppercase; letter-spacing: 0.8px;
  margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid #21262d;
}

/* Summary card */
.summary-card {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px;
}
.summary-conf-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.agent-summary { font-size: 13px; line-height: 1.6; color: #c9d1d9; margin-bottom: 12px; }
.findings-list { list-style: none; display: flex; flex-direction: column; gap: 6px; }
.findings-list li {
  font-size: 12px; color: #c9d1d9; padding-left: 16px; position: relative; line-height: 1.5;
}
.findings-list li::before { content: "→"; position: absolute; left: 0; color: #58a6ff; }
.caveats-details summary { font-size: 11px; color: #7d8590; cursor: pointer; margin-top: 10px; }
.caveats-list { margin-top: 8px; list-style: none; display: flex; flex-direction: column; gap: 4px; }
.caveats-list li { font-size: 11px; color: #7d8590; padding-left: 12px; position: relative; }
.caveats-list li::before { content: "!"; position: absolute; left: 0; color: #d29922; }

/* Pairs table */
.pairs-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.pairs-table th {
  text-align: left; padding: 6px 10px; color: #7d8590;
  border-bottom: 1px solid #30363d; font-weight: 500; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.pairs-table td { padding: 8px 10px; border-bottom: 1px solid #21262d; vertical-align: middle; }
.pairs-table tr:hover td { background: #161b22; }
.pairs-table code { font-family: monospace; font-size: 12px; color: #e6edf3; }
.score-bar {
  display: inline-flex; align-items: center; gap: 6px;
}
.score-track {
  width: 60px; height: 5px; background: #21262d; border-radius: 3px; flex-shrink: 0;
}
.score-fill { height: 100%; background: #388bfd; border-radius: 3px; }
.table-link { color: #58a6ff; text-decoration: none; font-size: 11px; }
.table-link:hover { text-decoration: underline; }
.disc-poly    { color: #58a6ff; }
.disc-fin     { color: #bc8cff; }
.disc-contemp { color: #3fb950; }
.disc-none    { color: #7d8590; }

/* Report */
.report-body {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 20px 24px; font-size: 13px; line-height: 1.7; color: #c9d1d9;
}
.report-body h1 { font-size: 18px; color: #e6edf3; margin: 0 0 16px; }
.report-body h2 { font-size: 15px; color: #e6edf3; margin: 20px 0 10px; }
.report-body h3 { font-size: 13px; color: #e6edf3; margin: 16px 0 8px; }
.report-body p  { margin: 0 0 10px; }
.report-body ul, .report-body ol { margin: 0 0 10px 20px; }
.report-body li { margin-bottom: 4px; }
.report-body table { border-collapse: collapse; width: 100%; margin: 12px 0; }
.report-body th { background: #21262d; padding: 6px 10px; text-align: left; color: #e6edf3; }
.report-body td { padding: 6px 10px; border-bottom: 1px solid #21262d; }
.report-body a  { color: #58a6ff; }
.report-body strong { color: #e6edf3; }

/* Arbitrage signal banner */
.signal-banner {
  border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;
  display: flex; align-items: flex-start; gap: 12px;
}
.signal-banner.sig-poly  { background: #0c2d4d; border: 1px solid #388bfd40; }
.signal-banner.sig-fin   { background: #2b1b56; border: 1px solid #8957e540; }
.signal-banner.sig-both  { background: #0b2e2a; border: 1px solid #39d35340; }
.signal-banner.sig-none  { background: #1c2128; border: 1px solid #30363d; }
.signal-icon  { font-size: 22px; flex-shrink: 0; }
.signal-label { font-size: 13px; font-weight: 700; margin-bottom: 3px; }
.signal-desc  { font-size: 12px; color: #8b949e; line-height: 1.4; }
.signal-banner.sig-poly .signal-label  { color: #58a6ff; }
.signal-banner.sig-fin .signal-label   { color: #bc8cff; }
.signal-banner.sig-both .signal-label  { color: #3fb950; }
.signal-banner.sig-none .signal-label  { color: #7d8590; }

/* Empty state */
.empty-state {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  height: 100%; color: #7d8590; gap: 8px;
}
.empty-state-title { font-size: 16px; color: #8b949e; }
</style>
</head>
<body>

<div class="header">
  <h1>Polymarket Intelligence</h1>
  <span class="header-stat">__N_MARKETS__ markets</span>
  <span class="header-stat">__N_PAIRS__ pairs</span>
  <span class="header-ts">Analysis: __TS__</span>
</div>

<div class="main">
  <div class="feed" id="feed"></div>
  <div class="detail" id="detail">
    <div class="empty-state">
      <div class="empty-state-title">Select a market</div>
      <div style="font-size:13px;">Choose a market from the feed on the left</div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<script>
const MARKETS   = __MARKETS__;
const REPORT_MD = __REPORT__;
const REPORT_MID = __REPORT_MID__;

function esc(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function catClass(cat) {
  if (!cat) return "social";
  const c = cat.toLowerCase();
  if (c.includes("fed") || c.includes("monetary")) return "fed";
  if (c.includes("politics") || c.includes("political")) return "politics";
  if (c.includes("geo")) return "geo";
  if (c.includes("econ") || c.includes("macro")) return "macro";
  if (c.includes("financial market")) return "fin";
  return "social";
}

function fmtNum(n) {
  if (n == null) return "—";
  if (n >= 1e6) return "$" + (n/1e6).toFixed(1) + "M";
  if (n >= 1e3) return "$" + (n/1e3).toFixed(1) + "K";
  return "$" + n.toFixed(0);
}

function fmtDate(d) {
  if (!d) return "";
  try { return new Date(d).toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"}); }
  catch(e) { return d; }
}

function discLabel(d) {
  if (!d || d === "no_signal") return { cls:"disc-none", label:"—" };
  if (d === "polymarket_leads")  return { cls:"disc-poly",    label:"Poly leads" };
  if (d === "finance_leads")     return { cls:"disc-fin",     label:"Fin leads"  };
  if (d === "contemporaneous")   return { cls:"disc-contemp", label:"Simultaneous" };
  return { cls:"disc-none", label: d };
}

function signalBanner(discovery, ticker) {
  const t = esc(ticker || "financial markets");
  if (discovery === "polymarket_leads") return `
    <div class="signal-banner sig-poly">
      <div class="signal-icon">&#x1F4C8;</div>
      <div>
        <div class="signal-label">Polymarket leads — potential opportunity</div>
        <div class="signal-desc">Polymarket priced this information <strong>before</strong> ${t} reacted.
        If the financial market hasn't fully caught up, there may be a tradeable edge.</div>
      </div>
    </div>`;
  if (discovery === "finance_leads") return `
    <div class="signal-banner sig-fin">
      <div class="signal-icon">&#x1F3DB;</div>
      <div>
        <div class="signal-label">Financial markets lead — information already priced in</div>
        <div class="signal-desc">${t} moved <strong>before</strong> Polymarket. Traditional markets
        priced this event first — Polymarket odds may lag reality.</div>
      </div>
    </div>`;
  if (discovery === "contemporaneous") return `
    <div class="signal-banner sig-both">
      <div class="signal-icon">&#x21C4;</div>
      <div>
        <div class="signal-label">Simultaneous movement — markets in sync</div>
        <div class="signal-desc">Polymarket and ${t} moved together at the same time.
        Strong co-movement but no clear information advantage on either side.</div>
      </div>
    </div>`;
  return `
    <div class="signal-banner sig-none">
      <div class="signal-icon">&#x2014;</div>
      <div>
        <div class="signal-label">No clear price discovery signal</div>
        <div class="signal-desc">Insufficient data or no significant relationship detected
        between this market and ${t}.</div>
      </div>
    </div>`;
}

function yahooUrl(ticker) {
  return "https://finance.yahoo.com/quote/" + ticker.replace("^", "%5E");
}

/* ── Render feed ── */
function renderFeed() {
  document.getElementById("feed").innerHTML = MARKETS.map(m => `
    <div class="market-card" id="card-${esc(m.id)}" onclick="selectMarket('${esc(m.id)}')">
      <div class="card-question">${esc(m.question)}</div>
      <div class="card-badges">
        <span class="badge cat-${catClass(m.category)}">${esc(m.category || "Unknown")}</span>
        <span class="badge conf-${m.best_confidence}">${m.best_confidence.toUpperCase()}</span>
      </div>
      <div class="card-stats">
        <span class="card-price">${m.price != null ? (m.price*100).toFixed(0)+"%" : "—"}</span>
        <span class="card-sep">·</span>
        <span class="card-ticker">${esc(m.best_ticker)}</span>
        <span class="card-score-wrap">
          <div class="card-score-num">${((m.best_score||0)*100).toFixed(0)}</div>
          <div class="card-score-label">score</div>
        </span>
      </div>
    </div>
  `).join("");
}

/* ── Select market ── */
function selectMarket(id) {
  document.querySelectorAll(".market-card").forEach(c => c.classList.remove("active"));
  const card = document.getElementById("card-" + id);
  if (card) { card.classList.add("active"); card.scrollIntoView({block:"nearest"}); }
  const m = MARKETS.find(m => m.id === id);
  if (m) renderDetail(m);
}

/* ── Render detail ── */
function renderDetail(m) {
  const slug     = m.slug || "";
  const polyUrl  = slug ? "https://polymarket.com/event/" + slug : "https://polymarket.com";
  const topTicker = m.best_ticker || "";
  const best     = m.pairs && m.pairs[0];
  const hasReport = (m.id === REPORT_MID && REPORT_MD);

  let html = `
    <div class="detail-header">
      <div class="detail-question">${esc(m.question)}</div>
      <div class="detail-meta-row">
        <span class="badge cat-${catClass(m.category)}">${esc(m.category || "Unknown")}</span>
        ${m.price != null ? `<span class="price-badge">${(m.price*100).toFixed(1)}% YES</span>` : ""}
        ${m.end_date ? `<span class="meta-chip">Closes ${fmtDate(m.end_date)}</span>` : ""}
        ${m.vol24hr  ? `<span class="meta-chip">24h vol ${fmtNum(m.vol24hr)}</span>` : ""}
      </div>
      <div class="links-bar">
        <a class="ext-link poly-link" href="${esc(polyUrl)}" target="_blank">View on Polymarket</a>
        ${topTicker ? `<a class="ext-link yahoo-link" href="${esc(yahooUrl(topTicker))}" target="_blank">${esc(topTicker)} on Yahoo Finance</a>` : ""}
      </div>
    </div>
    <div class="detail-body">
      __SIGNAL_BANNER__
  `;

  /* Best pair summary */
  if (best) {
    const d = discLabel(best.discovery);
    html += `
      <div class="section">
        <div class="section-title">Best Match — ${esc(best.ticker_name)}</div>
        <div class="summary-card">
          <div class="summary-conf-row">
            <span class="badge conf-${best.confidence}">${best.confidence.toUpperCase()} confidence</span>
            <span style="font-size:13px;color:#8b949e;">
              Similarity: <strong style="color:#e6edf3;">${((best.score||0)*100).toFixed(1)}</strong>/100
              &nbsp;·&nbsp; ${d.label !== "—" ? `<span class="${d.cls}">${d.label}</span>` : "No signal"}
              &nbsp;·&nbsp; ${best.n_obs} obs
            </span>
          </div>
          ${best.summary ? `<p class="agent-summary">${esc(best.summary)}</p>` : ""}
          ${best.findings && best.findings.length ? `
            <ul class="findings-list">
              ${best.findings.map(f=>`<li>${esc(f)}</li>`).join("")}
            </ul>
          ` : ""}
          ${best.caveats && best.caveats.length ? `
            <details class="caveats-details">
              <summary>${best.caveats.length} caveat${best.caveats.length>1?"s":""}</summary>
              <ul class="caveats-list">
                ${best.caveats.map(c=>`<li>${esc(c)}</li>`).join("")}
              </ul>
            </details>
          ` : ""}
        </div>
      </div>
    `;
  }

  /* All pairs table (if more than 1 pair) */
  if (m.pairs && m.pairs.length > 1) {
    html += `
      <div class="section">
        <div class="section-title">All Financial Pairs (${m.pairs.length})</div>
        <table class="pairs-table">
          <thead><tr>
            <th>Ticker</th><th>Instrument</th><th>Score</th>
            <th>Confidence</th><th>Discovery</th><th>Obs</th><th></th>
          </tr></thead>
          <tbody>
            ${m.pairs.map(p => {
              const sc = ((p.score||0)*100).toFixed(0);
              const d  = discLabel(p.discovery);
              return `
                <tr>
                  <td><code>${esc(p.ticker)}</code></td>
                  <td style="color:#8b949e;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(p.ticker_name)}</td>
                  <td>
                    <div class="score-bar">
                      <div class="score-track"><div class="score-fill" style="width:${Math.min(p.score||0,1)*100}%"></div></div>
                      <span style="font-size:11px;color:#8b949e;">${sc}</span>
                    </div>
                  </td>
                  <td><span class="badge conf-${p.confidence}">${p.confidence.toUpperCase()}</span></td>
                  <td><span class="${d.cls}">${d.label}</span></td>
                  <td style="color:#7d8590;">${p.n_obs}</td>
                  <td><a class="table-link" href="${esc(yahooUrl(p.ticker))}" target="_blank">Yahoo</a></td>
                </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  /* Synthesizer report */
  if (hasReport) {
    html += `
      <div class="section">
        <div class="section-title">Investment Report</div>
        <div class="report-body" id="report-body"></div>
      </div>
    `;
  }

  // Arbitrage signal banner (uses best pair's discovery)
  const bestDiscovery = best ? best.discovery : "";
  const bestTickerName = best ? best.ticker_name : "";
  html = html.replace("__SIGNAL_BANNER__", signalBanner(bestDiscovery, bestTickerName));

  html += "</div>"; // detail-body
  document.getElementById("detail").innerHTML = html;

  /* Render markdown after DOM update */
  if (hasReport) {
    const el = document.getElementById("report-body");
    if (el && typeof marked !== "undefined") {
      el.innerHTML = marked.parse(REPORT_MD);
    } else if (el) {
      el.textContent = REPORT_MD;
    }
  }
}

/* ── Init ── */
renderFeed();
if (MARKETS.length > 0) selectMarket(MARKETS[0].id);
</script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------
def build_html(market_list, report_md, report_mid, analysis_ts) -> str:
    try:
        ts = datetime.fromisoformat(analysis_ts.replace('Z', '+00:00'))
        ts_str = ts.strftime('%b %d, %Y %H:%M UTC')
    except Exception:
        ts_str = analysis_ts

    n_markets = len(market_list)
    n_pairs   = sum(len(m['pairs']) for m in market_list)

    html = HTML_TEMPLATE
    html = html.replace('__MARKETS__',   json.dumps(market_list,  ensure_ascii=False, default=str))
    html = html.replace('__REPORT__',    json.dumps(report_md,    ensure_ascii=False))
    html = html.replace('__REPORT_MID__',json.dumps(report_mid,   ensure_ascii=False))
    html = html.replace('__N_MARKETS__', str(n_markets))
    html = html.replace('__N_PAIRS__',   str(n_pairs))
    html = html.replace('__TS__',        ts_str)
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate Polymarket Intelligence dashboard')
    parser.add_argument('--trend',  help='Path to trend_analysis JSON')
    parser.add_argument('--csv',    help='Path to polymarket CSV')
    parser.add_argument('--report', help='Path to investment report MD')
    parser.add_argument('--out',    help='Output HTML path (default: docs/index.html)')
    parser.add_argument('--open',   action='store_true', help='Open in browser after generating')
    args = parser.parse_args()

    trend_path  = Path(args.trend)  if args.trend  else find_latest('trend_analysis_*.json')
    csv_path    = Path(args.csv)    if args.csv    else find_latest('polymarket_prod_*.csv')
    report_path = Path(args.report) if args.report else find_latest('investment_report_*.md')
    out_path    = Path(args.out)    if args.out    else ROOT / 'docs' / 'index.html'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not trend_path:
        print('ERROR: No trend_analysis_*.json found in output/. Run the pipeline first.')
        return
    if not csv_path:
        print('ERROR: No polymarket_prod_*.csv found in output/. Run the pipeline first.')
        return

    print(f'  Trend:  {trend_path.name}')
    print(f'  CSV:    {csv_path.name}')
    print(f'  Report: {report_path.name if report_path else "none"}')

    market_list, report_md, report_mid, ts = load_data(trend_path, csv_path, report_path)
    print(f'  Markets: {len(market_list)}  |  Pairs: {sum(len(m["pairs"]) for m in market_list)}')
    if report_mid:
        q = next((m["question"] for m in market_list if m["id"] == report_mid), "")
        print(f'  Report attached to: {q[:70]}')

    fetch_slugs_from_gamma(market_list)
    html = build_html(market_list, report_md, report_mid, ts)
    out_path.write_text(html, encoding='utf-8')
    print(f'\n  Dashboard saved -> {out_path}')

    if args.open:
        webbrowser.open(out_path.as_uri())
        print('  Opened in browser.')


if __name__ == '__main__':
    main()
