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


def _composite_score(stat_score: float, llm_conf: int | None, pearson_r: float | None) -> float:
    """Composite score blending statistical similarity, LLM confidence, and correlation."""
    s = float(stat_score or 0.0)
    c = float(llm_conf or 0) / 100.0  # normalise to 0-1
    r = abs(float(pearson_r or 0.0))  # absolute Pearson r (0-1)
    # Weights: LLM confidence 40%, stat score 35%, |pearson r| 25%
    return round(0.40 * c + 0.35 * s + 0.25 * r, 4)


def _parse_market_summaries(report_md: str) -> dict[str, str]:
    """Extract per-market summaries from the 'Market Signal Summaries' section."""
    summaries: dict[str, str] = {}
    lines = report_md.splitlines()
    in_section = False
    current_q: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if '## ' in line and 'Market Signal Summaries' in line:
            in_section = True
            continue
        if in_section:
            if line.startswith('## '):
                if current_q and current_lines:
                    summaries[current_q] = ' '.join(current_lines).strip()
                break
            elif line.startswith('### '):
                if current_q and current_lines:
                    summaries[current_q] = ' '.join(current_lines).strip()
                current_q = line[4:].strip()
                current_lines = []
            elif current_q and line.strip():
                current_lines.append(line.strip())

    if in_section and current_q and current_lines:
        summaries[current_q] = ' '.join(current_lines).strip()

    return summaries


def _match_summary(question: str, summaries: dict[str, str]) -> str:
    """Fuzzy-match a market question to a summary entry (first 50 chars)."""
    key = question.strip()[:50].lower()
    for q, text in summaries.items():
        if key in q.lower() or q.lower()[:50] in key:
            return text
    return ''


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

    # Pipeline JSON → LLM metadata per (market_id, series_id) + per-market selection rationale
    llm_meta: dict[tuple, dict] = {}  # (market_id, ticker) → {confidence, direction, company_name, rationale}
    sel_rationale: dict[str, str] = {}  # market_id → overall selection rationale
    pipeline_path = find_latest('pipeline_*.json')
    if pipeline_path:
        try:
            with open(pipeline_path, encoding='utf-8') as f:
                pipeline = json.load(f)
            # Load per-ticker rationale from finance_stats
            for stat in pipeline.get('finance_stats', []):
                mid = str(stat.get('polymarket_id', ''))
                ticker = str(stat.get('series_id', ''))
                conf = stat.get('llm_confidence')
                if conf is not None:
                    llm_meta[(mid, ticker)] = {
                        'llm_confidence':         int(conf),
                        'llm_predicted_direction': stat.get('llm_predicted_direction', ''),
                        'llm_company_name':        stat.get('llm_company_name', ''),
                        'llm_rationale':           stat.get('llm_rationale', ''),
                    }
            # Fill in rationale gaps from instrument_selections + capture market-level rationale
            for sel in pipeline.get('instrument_selections', []):
                mid = str(sel.get('market_id', ''))
                if mid and sel.get('selection_rationale'):
                    sel_rationale[mid] = sel['selection_rationale']
                for inst in sel.get('instruments', []):
                    ticker = str(inst.get('ticker', ''))
                    key = (mid, ticker)
                    if key in llm_meta and not llm_meta[key].get('llm_rationale'):
                        llm_meta[key]['llm_rationale'] = inst.get('rationale', '')
        except Exception:
            pass

    # Trend analysis JSON
    with open(trend_path, encoding='utf-8') as f:
        trend = json.load(f)

    markets: dict = {}
    for p in trend.get('pairs', []):
        mid = str(p['polymarket_id'])
        if mid not in markets:
            m = meta.get(mid, {})
            markets[mid] = {
                'id':           mid,
                'question':     p['polymarket_question'],
                'category':     m.get('category', ''),
                'slug':         m.get('slug', ''),
                'price':        m.get('price'),
                'vol24hr':      m.get('vol24hr'),
                'end_date':     m.get('end_date', ''),
                'sel_rationale': sel_rationale.get(mid, ''),
                'pairs':        [],
            }
        ticker = p['ticker']
        corr   = p.get('correlation') or {}
        ses    = p.get('spike_event_study') or {}
        llc    = p.get('lead_lag_ccf') or {}
        div    = p.get('divergence_signal') or {}
        stat_score  = p.get('overall_similarity_score') or 0.0
        pearson_r   = corr.get('pearson_r')
        lm          = llm_meta.get((mid, ticker), {})
        llm_conf    = lm.get('llm_confidence')
        composite   = _composite_score(stat_score, llm_conf, pearson_r)
        markets[mid]['pairs'].append({
            'ticker':            ticker,
            'ticker_name':       p['ticker_name'],
            'score':             stat_score,
            'composite':         composite,
            'confidence':        p.get('confidence_level', 'low'),
            'summary':           p.get('agent_summary', ''),
            'findings':          p.get('key_findings', []),
            'caveats':           p.get('caveats', []),
            'discovery':         ses.get('discovery_direction', ''),
            'pearson_r':         pearson_r,
            'n_obs':             p.get('n_observations', 0),
            'llm_confidence':    llm_conf,
            'llm_direction':     lm.get('llm_predicted_direction', ''),
            'llm_company_name':  lm.get('llm_company_name', ''),
            'llm_rationale':     lm.get('llm_rationale', ''),
            'llm_interp':        llc.get('interpretation', ''),
            'divergence_interp': div.get('interpretation', '') if div.get('signal_direction', 'no_signal') != 'no_signal' else '',
        })

    for m in markets.values():
        # Sort by LLM confidence descending (the new primary ranking key)
        m['pairs'].sort(key=lambda p: (p['llm_confidence'] or 0, p['composite']), reverse=True)
        b = m['pairs'][0] if m['pairs'] else {}
        m['best_ticker']      = b.get('ticker', '')
        m['best_ticker_name'] = b.get('llm_company_name') or b.get('ticker_name', '')
        m['best_score']       = b.get('composite', 0.0)
        m['best_confidence']  = b.get('confidence', 'low')
        m['best_llm_conf']    = b.get('llm_confidence')

    market_list = sorted(
        markets.values(),
        key=lambda m: (m['best_score'], m['vol24hr'] or 0),
        reverse=True,
    )

    # Investment report markdown
    report_md  = ''
    report_mid = ''
    market_summaries: dict[str, str] = {}
    if report_path and report_path.exists():
        report_md = report_path.read_text(encoding='utf-8')
        market_summaries = _parse_market_summaries(report_md)
        for line in report_md.splitlines():
            if line.startswith('# '):
                title = line[2:].strip().lower()
                for m in market_list:
                    if title[:40] in m['question'].lower() or m['question'].lower()[:40] in title:
                        report_mid = m['id']
                        break
                break

    # Attach per-market summaries
    for m in market_list:
        m['market_summary'] = _match_summary(m['question'], market_summaries)

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
.cat-earnings { background: #1a2e12; color: #56d364; }
.cat-ma       { background: #1e1a3b; color: #a5a0ff; }
.cat-social   { background: #1c2128; color: #8b949e; }
.dir-with    { color: #3fb950; font-weight: 600; }
.dir-against { color: #f85149; font-weight: 600; }

/* Info modal */
.info-btn {
  margin-left: auto; background: none; border: 1px solid #30363d; border-radius: 50%;
  width: 24px; height: 24px; color: #7d8590; cursor: pointer; font-size: 13px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.info-btn:hover { border-color: #58a6ff; color: #58a6ff; }
.modal-overlay {
  display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7);
  z-index: 100; align-items: center; justify-content: center;
}
.modal-overlay.open { display: flex; }
.modal-box {
  background: #161b22; border: 1px solid #30363d; border-radius: 10px;
  padding: 24px; max-width: 540px; width: 90%; max-height: 80vh; overflow-y: auto;
}
.modal-box h2 { font-size: 15px; color: #e6edf3; margin-bottom: 16px; }
.modal-term { margin-bottom: 14px; }
.modal-term dt { font-size: 12px; font-weight: 600; color: #58a6ff; margin-bottom: 3px; }
.modal-term dd { font-size: 12px; color: #8b949e; line-height: 1.5; margin: 0; }
.modal-close {
  float: right; background: none; border: none; color: #7d8590;
  font-size: 18px; cursor: pointer; padding: 0; line-height: 1;
}
.modal-close:hover { color: #e6edf3; }

/* Expandable instrument row */
.expand-row { display: none; background: #0d1117; }
.expand-row.open { display: table-row; }
.expand-cell { padding: 10px 14px 14px; border-bottom: 1px solid #21262d; }
.expand-rationale { font-size: 12px; color: #c9d1d9; line-height: 1.5; margin-bottom: 8px; }
.expand-stats { display: flex; gap: 16px; flex-wrap: wrap; font-size: 11px; color: #7d8590; }
.expand-stat-item strong { color: #e6edf3; }
.expand-findings { margin-top: 8px; font-size: 11px; color: #8b949e; }
.pairs-table tr.instrument-row { cursor: pointer; }
.pairs-table tr.instrument-row:hover td { background: #161b22; }

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
  <button class="info-btn" onclick="document.getElementById('glossary-modal').classList.add('open')" title="Glossary">&#x2139;</button>
</div>

<!-- Glossary modal -->
<div class="modal-overlay" id="glossary-modal" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="modal-box">
    <button class="modal-close" onclick="document.getElementById('glossary-modal').classList.remove('open')">&times;</button>
    <h2>Term Definitions</h2>
    <dl>
      <div class="modal-term">
        <dt>Composite Score</dt>
        <dd>Blended signal: 40% Confidence + 35% Stat Score + 25% |Pearson r|. Higher = stronger overall signal.</dd>
      </div>
      <div class="modal-term">
        <dt>Stat Score</dt>
        <dd>Statistical similarity between Polymarket probability changes and financial returns, combining correlation, Granger causality, lead-lag CCF, event study, and volatility spillover tests. 0–1 scale.</dd>
      </div>
      <div class="modal-term">
        <dt>Confidence</dt>
        <dd>LLM-assessed likelihood (1–100) that this financial instrument meaningfully responds to the Polymarket outcome. 90+ = directly named company or instrument. 10–29 = tenuous connection. Below 10 = essentially guessing.</dd>
      </div>
      <div class="modal-term">
        <dt>Alignment</dt>
        <dd><strong>With</strong>: instrument price rises when Polymarket YES probability rises. <strong>Against</strong>: instrument price falls when YES probability rises (inverse relationship).</dd>
      </div>
      <div class="modal-term">
        <dt>Discovery</dt>
        <dd>Which market incorporated information first. <strong>Poly leads</strong>: Polymarket priced it before financial markets — potential trading edge. <strong>Fin leads</strong>: financial markets moved first — Polymarket odds may lag. <strong>Simultaneous</strong>: both markets moved together.</dd>
      </div>
      <div class="modal-term">
        <dt>Pearson r</dt>
        <dd>Linear correlation coefficient between Polymarket probability changes and financial returns over the aligned window. Range −1 to +1. Values near ±1 indicate strong co-movement.</dd>
      </div>
      <div class="modal-term">
        <dt>n obs</dt>
        <dd>Number of aligned hourly observations used in the statistical analysis. More observations = more reliable stats. Below 20 may produce unreliable results.</dd>
      </div>
    </dl>
  </div>
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
  if (c.includes("earnings") || c.includes("corporate")) return "earnings";
  if (c.includes("m&a") || c.includes("ipo")) return "ma";
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
  document.getElementById("feed").innerHTML = MARKETS.map(m => {
    const llmConf = m.best_llm_conf != null ? `<span class="badge conf-medium">${m.best_llm_conf} conf</span>` : "";
    return `
    <div class="market-card" id="card-${esc(m.id)}" onclick="selectMarket('${esc(m.id)}')">
      <div class="card-question">${esc(m.question)}</div>
      <div class="card-badges">
        <span class="badge cat-${catClass(m.category)}">${esc(m.category || "Unknown")}</span>
        <span class="badge conf-${m.best_confidence}">${m.best_confidence.toUpperCase()}</span>
        ${llmConf}
      </div>
      <div class="card-stats">
        <span class="card-price">${m.price != null ? m.price.toFixed(3) : "—"}</span>
        <span class="card-sep">·</span>
        <span class="card-ticker">${esc(m.best_ticker)}</span>
        <span class="card-score-wrap">
          <div class="card-score-num">${((m.best_score||0)*100).toFixed(0)}</div>
          <div class="card-score-label">composite</div>
        </span>
      </div>
    </div>
  `}).join("");
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

  let html = `
    <div class="detail-header">
      <div class="detail-question">${esc(m.question)}</div>
      <div class="detail-meta-row">
        <span class="badge cat-${catClass(m.category)}">${esc(m.category || "Unknown")}</span>
        ${m.price != null ? `<span class="price-badge">${m.price.toFixed(3)}</span>` : ""}
        ${m.end_date ? `<span class="meta-chip">Closes ${fmtDate(m.end_date)}</span>` : ""}
        ${m.vol24hr  ? `<span class="meta-chip">24h vol ${fmtNum(m.vol24hr)}</span>` : ""}
      </div>
      <div class="links-bar">
        ${slug ? `<a class="ext-link poly-link" href="${esc('https://polymarket.com/event/'+slug)}" target="_blank">View on Polymarket</a>` : ""}
        ${topTicker ? `<a class="ext-link yahoo-link" href="${esc(yahooUrl(topTicker))}" target="_blank">${esc(topTicker)} on Yahoo Finance</a>` : ""}
      </div>
    </div>
    <div class="detail-body">
      __SIGNAL_BANNER__
  `;

  /* Best pair summary */
  if (best) {
    const d = discLabel(best.discovery);
    const alignDir = best.llm_direction;
    // Handle legacy up/down values
    const alignNorm = alignDir === "up" ? "with" : alignDir === "down" ? "against" : alignDir;
    const llmDir = alignNorm
      ? `<span class="dir-${alignNorm}">${alignNorm === "with" ? "▲ With" : "▼ Against"}</span>`
      : "";
    const llmConf = best.llm_confidence != null
      ? `<span class="badge conf-medium">Confidence: ${best.llm_confidence}/100</span>`
      : "";
    const compScore = ((best.composite||0)*100).toFixed(1);
    const statScore = ((best.score||0)*100).toFixed(1);
    html += `
      <div class="section">
        <div class="section-title">Top Instrument — ${esc(best.llm_company_name || best.ticker_name)} (${esc(best.ticker)})</div>
        <div class="summary-card">
          <div class="summary-conf-row">
            <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
              ${llmConf}
              ${llmDir ? `<span style="font-size:12px;">Alignment: ${llmDir}</span>` : ""}
              <span class="badge conf-${best.confidence}">${best.confidence.toUpperCase()} stat</span>
            </div>
            <div style="display:flex;gap:14px;font-size:12px;color:#8b949e;text-align:right;">
              <div><div style="color:#e6edf3;font-weight:600;">${compScore}</div><div>Composite</div></div>
              <div><div style="color:#e6edf3;font-weight:600;">${statScore}</div><div>Stat Score</div></div>
              <div><div style="color:#8b949e;font-size:11px;">${best.n_obs} obs</div></div>
            </div>
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

  /* Signal Analysis — built from synthesizer summary + selection rationale + best pair stats */
  {
    const best2 = m.pairs && m.pairs[0];
    const hasAny = m.market_summary || m.sel_rationale || (best2 && best2.llm_interp);
    if (hasAny) {
      let reportHtml = '';

      // Synthesizer paragraph (LLM-generated) — most prominent
      if (m.market_summary) {
        reportHtml += `<p style="margin:0 0 14px;font-size:13px;line-height:1.7;color:#c9d1d9;">${esc(m.market_summary)}</p>`;
      }

      // Lead-lag interpretation from best pair
      if (best2 && best2.llm_interp) {
        reportHtml += `
          <div style="background:#0c2d4d;border:1px solid #388bfd30;border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:#8b949e;line-height:1.5;">
            <strong style="color:#58a6ff;">Price Discovery:</strong> ${esc(best2.llm_interp)}
          </div>`;
      }

      // Divergence signal from best pair
      if (best2 && best2.divergence_interp) {
        reportHtml += `
          <div style="background:#1a2e12;border:1px solid #3fb95030;border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:12px;color:#8b949e;line-height:1.5;">
            <strong style="color:#3fb950;">Live Divergence:</strong> ${esc(best2.divergence_interp)}
          </div>`;
      }

      // Instrument selection rationale (overall assessment from LLM)
      if (m.sel_rationale) {
        reportHtml += `
          <div style="border-top:1px solid #21262d;padding-top:10px;margin-top:4px;font-size:12px;color:#7d8590;line-height:1.6;">
            <strong style="color:#8b949e;">Instrument Selection Assessment:</strong> ${esc(m.sel_rationale)}
          </div>`;
      }

      html += `
        <div class="section">
          <div class="section-title">Signal Analysis</div>
          <div class="summary-card">${reportHtml}</div>
        </div>
      `;
    }
  }

  /* All instruments table (always shown if any pairs) */
  if (m.pairs && m.pairs.length > 0) {
    const rowsHtml = m.pairs.map((p,i) => {
      const comp = ((p.composite||0)*100).toFixed(0);
      const sc   = ((p.score||0)*100).toFixed(0);
      const pr   = p.pearson_r != null ? p.pearson_r.toFixed(3) : "—";
      const d    = discLabel(p.discovery);
      const rowId = `row-expand-${esc(m.id)}-${i}`;
      // Normalize alignment (handle legacy up/down)
      const alignRaw = p.llm_direction;
      const alignNorm = alignRaw === "up" ? "with" : alignRaw === "down" ? "against" : alignRaw;
      const dirHtml = alignNorm
        ? `<span class="dir-${alignNorm}">${alignNorm === "with" ? "▲ With" : "▼ Against"}</span>`
        : "<span style='color:#7d8590;'>—</span>";
      const confHtml = p.llm_confidence != null
        ? `<strong style="color:#e6edf3;">${p.llm_confidence}</strong><span style="color:#7d8590;">/100</span>`
        : "<span style='color:#7d8590;'>—</span>";
      const nameDisp = p.llm_company_name || p.ticker_name;
      const findingsHtml = p.findings && p.findings.length
        ? `<div class="expand-findings"><strong style="color:#7d8590;">Key findings:</strong><ul style="margin:4px 0 0 16px;list-style:disc;">
            ${p.findings.slice(0,3).map(f=>`<li style="margin-bottom:3px;">${esc(f)}</li>`).join("")}
           </ul></div>` : "";
      const caveatsHtml = p.caveats && p.caveats.length
        ? `<div class="expand-findings" style="margin-top:6px;"><strong style="color:#d29922;">Caveats:</strong><ul style="margin:4px 0 0 16px;list-style:disc;">
            ${p.caveats.slice(0,2).map(c=>`<li style="margin-bottom:3px;">${esc(c)}</li>`).join("")}
           </ul></div>` : "";
      const llmInterpHtml = p.llm_interp ? `
        <div style="background:#0c2d4d;border:1px solid #388bfd30;border-radius:5px;padding:8px 12px;margin-top:8px;font-size:11px;color:#8b949e;line-height:1.5;">
          <strong style="color:#58a6ff;">Lead-lag:</strong> ${esc(p.llm_interp)}
        </div>` : "";
      const divInterpHtml = p.divergence_interp ? `
        <div style="background:#1a2e12;border:1px solid #3fb95030;border-radius:5px;padding:8px 12px;margin-top:6px;font-size:11px;color:#8b949e;line-height:1.5;">
          <strong style="color:#3fb950;">Divergence:</strong> ${esc(p.divergence_interp)}
        </div>` : "";
      return `
        <tr class="instrument-row" onclick="toggleRow('${rowId}')">
          <td><code>${esc(p.ticker)}</code></td>
          <td style="color:#8b949e;max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${esc(nameDisp)}">${esc(nameDisp)}</td>
          <td>${confHtml}</td>
          <td>${dirHtml}</td>
          <td>
            <div class="score-bar">
              <div class="score-track"><div class="score-fill" style="width:${Math.min(p.composite||0,1)*100}%"></div></div>
              <span style="font-size:11px;color:#8b949e;">${comp}</span>
            </div>
          </td>
          <td style="color:#7d8590;font-size:11px;">${sc}</td>
          <td><span class="${d.cls}">${d.label}</span></td>
          <td><a class="table-link" href="${esc(yahooUrl(p.ticker))}" target="_blank" onclick="event.stopPropagation()">Yahoo</a></td>
        </tr>
        <tr class="expand-row" id="${rowId}">
          <td class="expand-cell" colspan="8">
            ${p.llm_rationale ? `<div class="expand-rationale"><strong style="color:#58a6ff;">Why selected:</strong> ${esc(p.llm_rationale)}</div>` : ""}
            <div class="expand-stats">
              <span class="expand-stat-item">Pearson r: <strong>${pr}</strong></span>
              <span class="expand-stat-item">n obs: <strong>${p.n_obs}</strong></span>
              <span class="expand-stat-item">Stat score: <strong>${sc}/100</strong></span>
              <span class="expand-stat-item">Composite: <strong>${comp}/100</strong></span>
            </div>
            ${p.summary ? `<div style="margin-top:8px;font-size:12px;color:#8b949e;line-height:1.5;font-style:italic;">${esc(p.summary)}</div>` : ""}
            ${llmInterpHtml}
            ${divInterpHtml}
            ${findingsHtml}
            ${caveatsHtml}
          </td>
        </tr>`;
    }).join("");
    html += `
      <div class="section">
        <div class="section-title">All Instruments (${m.pairs.length}) — click row for details</div>
        <table class="pairs-table">
          <thead><tr>
            <th>Ticker</th><th>Company / Instrument</th>
            <th>Confidence</th><th>Alignment</th>
            <th>Composite</th><th>Stat Score</th>
            <th>Discovery</th><th></th>
          </tr></thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      </div>
    `;
  }

  /* Investment report — shown on all markets when available */
  if (REPORT_MD) {
    html += `
      <div class="section">
        <div class="section-title">Investment Analysis Report</div>
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
  if (REPORT_MD) {
    const el = document.getElementById("report-body");
    if (el && typeof marked !== "undefined") {
      el.innerHTML = marked.parse(REPORT_MD);
    } else if (el) {
      el.textContent = REPORT_MD;
    }
    // Scroll to this market's section in the report if it exists
    if (el) {
      const mq = (m.question || "").slice(0, 50).toLowerCase();
      const headings = el.querySelectorAll("h3");
      for (const h of headings) {
        if (h.textContent.toLowerCase().slice(0, 50).includes(mq.slice(0, 30))) {
          h.scrollIntoView({ behavior: "smooth", block: "start" });
          break;
        }
      }
    }
  }
}

function toggleRow(id) {
  const row = document.getElementById(id);
  if (row) row.classList.toggle('open');
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
