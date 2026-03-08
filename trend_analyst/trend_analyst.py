"""
Trend Analyst Agent  (v0.3.0)
------------------------------
Detects information asymmetry between Polymarket prediction markets and
financial instruments by running per-pair statistical tests with LLM-adaptive
weights, then outputting a structured JSON payload for the Synthesizer.

Pipeline:
  1. Load Polymarket data (CSV with 49-point hourly series, or scan JSON snapshot)
  2. Load financial market data (any workflow JSON format — auto-detected by adapter)
  3. For each (Polymarket market, financial ticker) pair:
       a. Ask OpenAI to assign test weights tailored to the question, sector,
          and instrument type  (falls back to static weights if LLM unavailable)
       b. Run the selected active statistical tests:
            - Pearson/Spearman correlation
            - Granger causality (bidirectional)
            - Lead-lag cross-correlation function (CCF)
            - Event study (absolute probability-change threshold)
            - Spike event study (price discovery direction: priced in or not?)
            - Volatility spillover
       c. Compute weighted composite score + confidence level
       d. Generate key findings and caveats
       e. (Optional) Enrich with OpenAI narrative summary
  4. Output structured JSON for the Synthesizer (local + optional R2 upload)

Key design decision: Polymarket prices are bounded [0, 1] probabilities, not
continuous financial prices. All tests therefore use ABSOLUTE probability changes
(dP = P_t - P_{t-1}) instead of log-returns, which are ill-defined at price
boundaries (log(0) = -inf) and produce NaN on flat-price segments.

Usage:
    # Standard run with CSV + financial data:
    python trend_analyst.py \\
        --polymarket-csv polymarket_prod_2026-03-01.csv \\
        --financial-data financial_output.json \\
        --output data/output/trend_analysis.json

    # Snapshot-only mode (no Polymarket time series — uses financial data as proxy):
    python trend_analyst.py \\
        --polymarket-scan https://pub-758436d0743b4cee966caace6c186999.r2.dev/latest_scan.json \\
        --financial-data financial_output.json

    # Add OpenAI narrative summaries and upload result to Cloudflare R2:
    python trend_analyst.py \\
        --polymarket-csv polymarket_prod_2026-03-01.csv \\
        --financial-data financial_output.json \\
        --use-llm --upload-r2

    # Disable LLM-adaptive weights (use static defaults instead):
    python trend_analyst.py \\
        --polymarket-csv data.csv --financial-data fin.json --no-adaptive-weights

    # Override the financial ticker for a specific Polymarket market:
    python trend_analyst.py --polymarket-csv data.csv --financial-data fin.json \\
        --ticker 916732=USO
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load API keys from secrets.txt if not already in environment.
# Looks in the project dir first, then one level up (lab root).
def _load_secrets():
    for path in [Path(__file__).parent / "secrets.txt",
                 Path(__file__).parent.parent / "secrets.txt"]:
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = val.strip()
            break

_load_secrets()

from config import POLYMARKET_SCAN_URL, OUTPUT_DIR, AGENT_PERSONA, ACTIVE_TESTS
from models.schemas import (
    PairAnalysis,
    TrendAnalystOutput,
)
from tools.data_loader import (
    load_polymarket_scan,
    load_polymarket_csv,
    load_polymarket_from_financial_json,
    build_aligned_series,
)
from tools.financial_adapter import load_financial_workflow_json
from tools.anthropic_client import select_tests_and_weights, generate_agent_summary
from tools.statistical_toolkit import (
    compute_correlation,
    compute_granger_causality,
    compute_lead_lag_ccf,
    compute_event_study,
    compute_spike_event_study,
    compute_volatility_spillover,
    compute_divergence_signal,
    compute_composite_score,
)

# ── Logging setup ────────────────────────────────────────────────────────────

logger = logging.getLogger("trend_analyst")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)


# ── R2 helpers ───────────────────────────────────────────────────────────────

def upload_to_r2(filepath: str, key: str | None = None) -> str | None:
    """Upload analysis output to R2. Returns URL on success, None on failure."""
    try:
        from tools.r2_client import upload_file
        url = upload_file(filepath, key=key)
        logger.info("Uploaded to R2: %s", url)
        return url
    except RuntimeError as e:
        logger.warning("R2 upload skipped: %s", e)
        return None
    except ImportError:
        logger.warning("R2 upload skipped: boto3 not installed")
        return None


def list_previous_runs(prefix: str = "analysis/") -> list[dict]:
    """List previous analysis runs stored in R2."""
    try:
        from tools.r2_client import list_objects
        objects = list_objects(prefix=prefix)
        logger.info("Found %d previous runs in R2", len(objects))
        return objects
    except (RuntimeError, ImportError):
        logger.debug("Cannot list R2 objects (credentials not configured or boto3 missing)")
        return []


# ── LLM enrichment ───────────────────────────────────────────────────────────

def enrich_with_llm(pair_analysis: PairAnalysis) -> PairAnalysis:
    """
    Call OpenAI GPT-4o to generate a nuanced narrative summary that interprets
    the statistical results in context of the specific Polymarket question and
    financial instrument.  Overlays agent_summary, key_findings, and risk_flags
    onto the mechanical analysis.  Falls back silently if OPENAI_API_KEY is not
    set or the API call fails.
    """
    try:
        _ = generate_agent_summary  # verify importable
    except NameError:
        logger.debug("generate_agent_summary not available, skipping LLM enrichment")
        return pair_analysis

    pair_dict = pair_analysis.to_dict()
    result = generate_agent_summary(pair_dict, AGENT_PERSONA)

    if result is None:
        logger.debug("LLM enrichment skipped (no API key or call failed)")
        return pair_analysis

    # Overlay LLM-generated fields onto the mechanical analysis
    if result.get("agent_summary"):
        pair_analysis.agent_summary = result["agent_summary"]
    if result.get("key_findings"):
        pair_analysis.key_findings = result["key_findings"]
    if result.get("risk_flags"):
        pair_analysis.caveats.extend(result["risk_flags"])

    logger.info("LLM enrichment applied for %s x %s",
                pair_analysis.polymarket_id, pair_analysis.ticker)
    return pair_analysis


# ── Core analysis ────────────────────────────────────────────────────────────



def analyze_pair(
    poly_market,
    fin_record,
    recommendation_reasoning: str = "",
    use_llm: bool = False,
    adaptive_weights: bool = True,
) -> PairAnalysis:
    """Run the full statistical suite on a single Polymarket-Equity pair.

    adaptive_weights=True (default): ask OpenAI to assign per-pair test weights
    based on the specific question, sector, and instrument type.  Falls back to
    static _DEFAULT_WEIGHTS when the LLM is unavailable.
    """
    analysis_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()

    aligned = build_aligned_series(poly_market, fin_record)

    if aligned is None:
        return PairAnalysis(
            analysis_id=analysis_id,
            polymarket_id=poly_market.market_id,
            polymarket_question=poly_market.question,
            ticker=fin_record.ticker,
            ticker_name=fin_record.ticker_name,
            recommendation_reasoning=recommendation_reasoning or fin_record.recommendation_reasoning,
            analysis_timestamp=now,
            data_start_date="",
            data_end_date="",
            n_observations=0,
            overall_similarity_score=0.0,
            confidence_level="low",
            agent_summary="Insufficient data to perform analysis.",
            key_findings=["No aligned time series data available"],
            caveats=["Analysis could not be performed due to missing data"],
        )

    n = len(aligned["fin_prices"])
    dates = aligned["dates"]

    logger.info("  %s x %s: %d observations", poly_market.market_id, fin_record.ticker, n)

    # ── Decide which tests to run and their weights ───────────────────────
    dynamic_weights = None
    adaptive_reasoning = None
    if adaptive_weights:
        plan = select_tests_and_weights(
            polymarket_question=poly_market.question,
            polymarket_price=getattr(poly_market, "current_price", 0.5),
            ticker=fin_record.ticker,
            ticker_name=fin_record.ticker_name,
            n_observations=n,
            category=getattr(poly_market, "category", ""),
        )
        if plan:
            selected_tests = set(plan["selected_tests"])
            dynamic_weights = plan["weights"]
            adaptive_reasoning = plan["reasoning"]
        else:
            # LLM unavailable — fall back to all tests with static default weights
            logger.warning(
                "  LLM planner unavailable for %s x %s -- using static default weights",
                poly_market.market_id, fin_record.ticker,
            )
            selected_tests = set(ACTIVE_TESTS)
    else:
        selected_tests = set(ACTIVE_TESTS)

    # Compute absolute Polymarket probability changes (dP = P_t - P_{t-1}).
    # These are bounded [-1, 1] and well-defined even when prices are near 0/1,
    # unlike log-returns which produce NaN/inf at the boundaries.
    poly_prices_arr = np.array(aligned["poly_prices"], dtype=np.float64)
    fin_returns_arr = np.array(aligned["fin_returns"], dtype=np.float64)
    poly_changes = np.diff(poly_prices_arr)  # length N-1, aligned with fin_returns
    # Ensure equal lengths (off-by-one can occur when proxy mode returns N returns for N prices)
    min_len = min(len(poly_changes), len(fin_returns_arr))
    poly_changes = poly_changes[:min_len]
    fin_returns_arr = fin_returns_arr[:min_len]

    # ── Run selected tests ────────────────────────────────────────────────
    correlation = None
    if "correlation" in selected_tests:
        logger.debug("    Correlation...")
        correlation = compute_correlation(poly_changes, fin_returns_arr)

    granger = None
    if "granger" in selected_tests:
        logger.debug("    Granger causality...")
        granger = compute_granger_causality(poly_changes, fin_returns_arr)

    lead_lag = None
    if "lead_lag_ccf" in selected_tests:
        logger.debug("    Lead-lag CCF...")
        lead_lag = compute_lead_lag_ccf(poly_changes, fin_returns_arr)

    event = None
    if "event_study" in selected_tests:
        logger.debug("    Event study...")
        event = compute_event_study(poly_prices_arr, fin_returns_arr)

    spike_event = None
    if "spike_event_study" in selected_tests:
        logger.debug("    Spike event study...")
        spike_event = compute_spike_event_study(poly_prices_arr, fin_returns_arr)

    vol_spill = None
    if "volatility_spillover" in selected_tests:
        logger.debug("    Volatility spillover...")
        vol_spill = compute_volatility_spillover(poly_changes, fin_returns_arr)

    # ── Current divergence signal (always run; not a weighted test) ───────
    logger.debug("    Divergence signal...")
    divergence = compute_divergence_signal(poly_prices_arr, fin_returns_arr)

    # ── Composite score ───────────────────────────────────────────────────
    composite, confidence = compute_composite_score(
        correlation=correlation, granger=granger,
        lead_lag_ccf=lead_lag, event_study=event,
        spike_event_study=spike_event, vol_spillover=vol_spill,
        weights=dynamic_weights,
    )

    # ── Generate key findings ─────────────────────────────────────────────
    key_findings = []

    # Price discovery direction is the most actionable finding — put it first
    if spike_event is not None and spike_event.n_spikes >= 2:
        if spike_event.discovery_direction == "polymarket_leads":
            key_findings.append(
                f"Polymarket LEADS financial market: post-spike directional agreement "
                f"{spike_event.directional_post_agreement:.0%} vs "
                f"{spike_event.directional_pre_agreement:.0%} pre-spike "
                f"({spike_event.n_spikes} spikes >{spike_event.spike_threshold_pp:.0f}pp)"
            )
        elif spike_event.discovery_direction == "financial_leads":
            key_findings.append(
                f"Financial market LEADS Polymarket (info may be priced in): pre-spike "
                f"directional agreement {spike_event.directional_pre_agreement:.0%} "
                f"({spike_event.n_spikes} spikes >{spike_event.spike_threshold_pp:.0f}pp)"
            )

    if correlation is not None and correlation.pearson_p < 0.05:
        key_findings.append(
            f"Significant contemporaneous co-movement: Polymarket dP vs financial returns"
            f"r={correlation.pearson_r:.3f} (p={correlation.pearson_p:.4f})"
        )
    if granger is not None and granger.poly_causes_fin:
        key_findings.append(
            f"Polymarket dP Granger-causes financial returns at lag {granger.poly_to_fin_best_lag}h "
            f"(p={granger.poly_to_fin_p_value:.4f})"
        )
    if granger is not None and granger.fin_causes_poly:
        key_findings.append(
            f"Financial returns Granger-cause Polymarket dP at lag {granger.fin_to_poly_best_lag}h "
            f"(p={granger.fin_to_poly_p_value:.4f})"
        )
    if lead_lag is not None and lead_lag.peak_p_value < 0.05 and lead_lag.peak_lag > 0:
        key_findings.append(
            f"Polymarket leads financial market by {lead_lag.peak_lag}h "
            f"(CCF r={lead_lag.peak_correlation:.3f})"
        )
    elif lead_lag is not None and lead_lag.peak_p_value < 0.05 and lead_lag.peak_lag < 0:
        key_findings.append(
            f"Financial market leads Polymarket by {-lead_lag.peak_lag}h "
            f"(CCF r={lead_lag.peak_correlation:.3f})"
        )
    if event is not None and event.n_events > 0 and event.direction_consistency >= 0.6:
        key_findings.append(
            f"After >{event.threshold_pct:.0f}pp Polymarket moves, financials respond same direction "
            f"{event.direction_consistency:.0%} of the time ({event.n_events} events)"
        )
    if vol_spill is not None and vol_spill.poly_vol_predicts_fin:
        key_findings.append(
            f"Polymarket volatility Granger-predicts financial volatility "
            f"(p={vol_spill.poly_vol_lead_p_value:.4f})"
        )
    if divergence.signal_direction == "underreaction":
        key_findings.append(
            f"CURRENT DIVERGENCE — financial market UNDERREACTION: Polymarket at "
            f"{divergence.current_probability:.0%}, moved {divergence.poly_move_pp:+.1f}pp "
            f"over last {divergence.lookback_hours}h; β implies {divergence.predicted_fin_return_pct:+.2f}% "
            f"financial move but actual was {divergence.actual_fin_return_pct:+.2f}% "
            f"(gap {divergence.divergence_pct:+.2f}%, z={divergence.divergence_z:.2f})"
        )
    elif divergence.signal_direction == "overreaction":
        key_findings.append(
            f"CURRENT DIVERGENCE — financial market OVERREACTION: Polymarket at "
            f"{divergence.current_probability:.0%}, moved {divergence.poly_move_pp:+.1f}pp "
            f"over last {divergence.lookback_hours}h; β implies {divergence.predicted_fin_return_pct:+.2f}% "
            f"but actual was {divergence.actual_fin_return_pct:+.2f}% "
            f"(gap {divergence.divergence_pct:+.2f}%, z={divergence.divergence_z:.2f})"
        )

    if not key_findings:
        key_findings.append("No statistically significant parallel movement detected")

    # ── Generate caveats ──────────────────────────────────────────────────
    caveats = []
    if not aligned.get("has_poly_timeseries", False):
        caveats.append(
            "Polymarket time series not yet available from upstream -- "
            "analysis used financial data as proxy for both series"
        )
    if adaptive_reasoning:
        caveats.append(f"Adaptive weighting: {adaptive_reasoning}")
    if n < 30:
        caveats.append(f"Small sample size ({n} obs) limits statistical power")
    if correlation is not None and correlation.pearson_p > 0.10:
        caveats.append("Correlation not statistically significant at alpha=0.10")

    # ── Agent summary ─────────────────────────────────────────────────────
    summary_parts = [
        f"Composite similarity score: {composite:.2f}/1.00 ({confidence} confidence).",
    ]
    if key_findings[0] != "No statistically significant parallel movement detected":
        summary_parts.append(f"Key finding: {key_findings[0]}.")
    else:
        summary_parts.append(
            "The statistical evidence does not support a strong parallel movement "
            "between these markets at this time."
        )
    summary_parts.append(
        f"Based on {n} observations from {dates[0] if len(dates) > 0 else 'N/A'} "
        f"to {dates[-1] if len(dates) > 0 else 'N/A'}."
    )

    pair = PairAnalysis(
        analysis_id=analysis_id,
        polymarket_id=poly_market.market_id,
        polymarket_question=poly_market.question,
        ticker=fin_record.ticker,
        ticker_name=fin_record.ticker_name,
        recommendation_reasoning=recommendation_reasoning or fin_record.recommendation_reasoning,
        analysis_timestamp=now,
        data_start_date=str(dates[0]) if len(dates) > 0 else "",
        data_end_date=str(dates[-1]) if len(dates) > 0 else "",
        n_observations=n,
        correlation=correlation,
        granger_causality=granger,
        lead_lag_ccf=lead_lag,
        event_study=event,
        spike_event_study=spike_event,
        volatility_spillover=vol_spill,
        divergence_signal=divergence,
        overall_similarity_score=composite,
        confidence_level=confidence,
        agent_summary=" ".join(summary_parts),
        key_findings=key_findings,
        caveats=caveats,
    )

    # ── Optional LLM enrichment ───────────────────────────────────────────
    if use_llm:
        pair = enrich_with_llm(pair)

    return pair


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_trend_analyst(
    polymarket_source: str | None = None,
    polymarket_csv: str | None = None,
    financial_source: str | None = None,
    output_path: str | None = None,
    upload_r2: bool = False,
    use_llm: bool = False,
    adaptive_weights: bool = True,
    ticker_overrides: dict[str, str] | None = None,
) -> TrendAnalystOutput:
    """
    Main entry point for the Trend Analyst agent.

    Args:
        polymarket_source: URL or path to Polymarket scan JSON (snapshot only).
        polymarket_csv:    Path to Polymarket production CSV with hourly prices.
                           When provided this takes priority over polymarket_source
                           for market metadata AND supplies real time series for
                           statistical comparison.
        financial_source:  Path or URL to financial workflow JSON (any supported
                           format — see tools/financial_adapter.py).
        output_path:       Where to save the analysis JSON locally.
        upload_r2:         Whether to upload the output to Cloudflare R2.
        use_llm:           Whether to enrich analysis with OpenAI LLM summaries.
        adaptive_weights:  Use OpenAI to assign per-pair test weights based on the
                           specific question, sector, and instrument type (default True).
                           Falls back to static default weights when LLM is unavailable.
                           Pass False to always use static weights.
        ticker_overrides:  {polymarket_id: ticker} overrides applied after loading.
    """
    now = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 60)
    logger.info("  TREND ANALYST v0.3.1  |  %s", now)
    logger.info("=" * 60)

    # 1. Load Polymarket data
    logger.info("[1/5] Loading Polymarket data...")
    if polymarket_csv:
        logger.info("  Using CSV with hourly time series: %s", polymarket_csv)
        summary, markets = load_polymarket_csv(polymarket_csv)
        poly_source_type = "csv_hourly"
    elif polymarket_source:
        logger.info("  Using scan JSON (snapshot): %s", polymarket_source)
        summary, markets = load_polymarket_scan(polymarket_source)
        poly_source_type = "scan_json_snapshot"
    elif financial_source:
        # Format D financial JSON embeds Polymarket probability series -- use it directly
        logger.info("  No Polymarket source given -- extracting from financial JSON: %s",
                    financial_source)
        try:
            summary, markets = load_polymarket_from_financial_json(financial_source)
            poly_source_type = "financial_json_format_d"
        except ValueError as exc:
            logger.warning("  Could not extract Polymarket data from financial JSON: %s", exc)
            logger.info("  Falling back to upstream Polymarket scan URL")
            summary, markets = load_polymarket_scan(POLYMARKET_SCAN_URL)
            poly_source_type = "scan_json_snapshot"
    else:
        pm_source = POLYMARKET_SCAN_URL
        logger.info("  Using default upstream scan URL: %s", pm_source)
        summary, markets = load_polymarket_scan(pm_source)
        poly_source_type = "scan_json_snapshot"

    n_with_series = sum(1 for m in markets if m.poly_prices)
    logger.info(
        "  Loaded %d markets (%d with hourly time series, scan: %s)",
        len(markets), n_with_series, summary.get("scan_timestamp", "unknown"),
    )

    active_markets = [m for m in markets if m.passed_initial_filter]
    logger.info("  %d markets passed initial filter", len(active_markets))

    # 2. Load financial data
    logger.info("[2/5] Loading financial market data...")
    if financial_source:
        # Use the adapter so all three workflow JSON formats are accepted
        fin_records = load_financial_workflow_json(financial_source)
        logger.info("  Loaded %d financial market pairs", len(fin_records))
    else:
        logger.warning("  No financial data source provided -- will generate placeholder output")
        fin_records = []

    # Apply ticker overrides
    if ticker_overrides and fin_records:
        for fin in fin_records:
            if fin.polymarket_id in ticker_overrides:
                old_ticker = fin.ticker
                fin.ticker = ticker_overrides[fin.polymarket_id]
                logger.info("  Ticker override: %s -> %s (for %s)",
                            old_ticker, fin.ticker, fin.polymarket_id)

    # 3. Match and analyze pairs
    logger.info("[3/5] Running statistical analysis on pairs...")
    pairs = []

    if fin_records:
        market_lookup = {m.market_id: m for m in markets}

        for fin in fin_records:
            poly = market_lookup.get(fin.polymarket_id)
            if poly is None:
                logger.warning("  Polymarket ID %s not found in scan, skipping",
                               fin.polymarket_id)
                continue

            pair_result = analyze_pair(poly, fin, use_llm=use_llm,
                                       adaptive_weights=adaptive_weights)
            pairs.append(pair_result)
    else:
        logger.info("  No financial data pairs to analyze.")

    # 4. Package output
    logger.info("[4/5] Packaging output (%d pair analyses)...", len(pairs))
    scan_timestamp = summary.get("scan_timestamp") or now
    output = TrendAnalystOutput(
        scan_timestamp=scan_timestamp,
        analysis_timestamp=now,
        analyst_version="0.3.1",
        pairs=pairs,
        metadata={
            "total_polymarkets_scanned": len(markets),
            "active_polymarkets": len(active_markets),
            "markets_with_hourly_series": n_with_series,
            "pairs_analyzed": len(pairs),
            "polymarket_source_type": poly_source_type,
            "llm_enrichment": use_llm,
            "adaptive_weights": adaptive_weights,
            "statistical_tests_run": [
                "pearson_spearman_correlation",
                "granger_causality",
                "lead_lag_ccf",
                "event_study_absolute_pp",
                "spike_event_study_price_discovery",
                "volatility_spillover",
            ],
            "agent_persona": AGENT_PERSONA,
        },
    )

    # Save locally
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        output.save(output_path)
        local_path = output_path
    else:
        local_path = os.path.join(OUTPUT_DIR, f"trend_analysis_{now[:10]}.json")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output.save(local_path)

    # 5. Optional R2 upload
    if upload_r2:
        logger.info("[5/5] Uploading to R2...")
        r2_key = f"analysis/trend_analysis_{now[:10]}.json"
        r2_url = upload_to_r2(local_path, key=r2_key)
        if r2_url:
            output.metadata["r2_url"] = r2_url
            # Re-save with R2 URL in metadata
            output.save(local_path)
    else:
        logger.info("[5/5] R2 upload skipped (use --upload-r2 to enable)")

    logger.info("=" * 60)
    logger.info("  ANALYSIS COMPLETE  |  Pairs analyzed: %d", len(pairs))
    logger.info("=" * 60)

    return output


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_ticker_overrides(raw: list[str] | None) -> dict[str, str]:
    """Parse --ticker POLYMARKET_ID=TICKER arguments into a dict."""
    if not raw:
        return {}
    overrides = {}
    for item in raw:
        if "=" not in item:
            logger.warning("Invalid --ticker format '%s', expected POLYMARKET_ID=TICKER", item)
            continue
        pm_id, ticker = item.split("=", 1)
        overrides[pm_id.strip()] = ticker.strip()
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Trend Analyst Agent - Statistical comparison of Polymarket and equity data"
    )
    parser.add_argument(
        "--polymarket-scan", "-p",
        help="Path or URL to Polymarket scan JSON (snapshot, no time series)",
        default=None,
    )
    parser.add_argument(
        "--polymarket-csv", "-c",
        help=(
            "Path to Polymarket production CSV with hourly price columns "
            "(price_t_minus_24 … price_t_plus_24).  "
            "Takes priority over --polymarket-scan and enables real statistical comparison."
        ),
        default=None,
    )
    parser.add_argument(
        "--financial-data", "-f",
        help=(
            "Path or URL to financial market workflow JSON.  "
            "Accepts Format A (native 'pairs'), Format B ('markets/tickers'), "
            "or Format C ('results' list) — see tools/financial_adapter.py."
        ),
        default=None,
    )
    parser.add_argument(
        "--output", "-o",
        help="Output filepath for analysis JSON",
        default=None,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the upstream Polymarket scan URL when no --polymarket-csv or --polymarket-scan is given",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        metavar="POLYMARKET_ID=TICKER",
        help="Override ticker for a polymarket ID (repeatable)",
    )
    parser.add_argument(
        "--upload-r2",
        action="store_true",
        help="Upload output JSON to Cloudflare R2 bucket",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enrich analysis with OpenAI LLM summaries (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--no-adaptive-weights",
        dest="adaptive_weights",
        action="store_false",
        help=(
            "Disable per-pair LLM weight selection and use static default weights instead. "
            "By default the agent asks OpenAI to assign test weights for each pair based on "
            "the question, sector, and instrument type."
        ),
    )
    parser.set_defaults(adaptive_weights=True)
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List previous analysis runs stored in R2 and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Handle --list-runs
    if args.list_runs:
        runs = list_previous_runs()
        if not runs:
            logger.info("No previous runs found in R2 (or credentials not configured)")
        else:
            for r in runs:
                logger.info("  %s  (%s bytes, modified %s)",
                            r["Key"], r["Size"], r["LastModified"])
        return

    # Resolve data sources
    polymarket_source = args.polymarket_scan
    polymarket_csv    = args.polymarket_csv
    financial_source  = args.financial_data

    if args.live:
        # CSV takes priority; only fall back to scan URL if no CSV is given
        if not polymarket_csv and not polymarket_source:
            polymarket_source = POLYMARKET_SCAN_URL
            logger.info("Live mode: using upstream Polymarket scan URL")

    ticker_overrides = parse_ticker_overrides(args.ticker)

    run_trend_analyst(
        polymarket_source=polymarket_source,
        polymarket_csv=polymarket_csv,
        financial_source=financial_source,
        output_path=args.output,
        upload_r2=args.upload_r2,
        use_llm=args.use_llm,
        adaptive_weights=args.adaptive_weights,
        ticker_overrides=ticker_overrides,
    )


if __name__ == "__main__":
    main()
