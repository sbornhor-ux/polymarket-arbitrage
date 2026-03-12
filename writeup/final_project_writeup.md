Team Members: Sam Bornhorst, Kevin Chu, Michael Danielak, Alex Casarella, Jack Goldstein

---

# Press Release

*Chicago Booth School of Business — AI and Financial Information (BUSN 30135)*
*March 2026*

---

## Booth Team Builds Real-Time Signal Platform That Translates Prediction Market Shifts Into Financial Instrument Alerts

**CHICAGO, March 2026** — A five-person team of MBA students at the University of Chicago Booth School of Business has built and deployed a real-time information asymmetry tracker that monitors more than 25,000 Polymarket prediction-market contracts, identifies probability swings relevant to finance and macroeconomics, and automatically maps those swings to correlated traditional financial instruments. The system surfaces its findings on a public live dashboard, runs continuously on cloud infrastructure, and costs approximately $5 per month to operate.

The project addresses a specific blind spot in financial data workflows. Prediction markets, which aggregate crowd beliefs through real-money bets and produce prices bounded in the range [0, 1], often reflect new information about policy outcomes, geopolitical events, and macroeconomic data before that information is fully priced into traditional financial assets. No systematic, automated tool previously existed to detect those probability shifts and translate them into instrument-level signals for event-driven investors. The team built one.

The system operates as a multi-stage pipeline. A data ingestion layer fetches all active Polymarket contracts and applies a keyword filter — covering equities, central bank policy, macroeconomic indicators, currencies, commodities, fixed income, trade policy, and geopolitics — to reduce the universe to the finance-relevant subset. Each remaining market is scored on a composite metric that weights market volume (40%), odds swing magnitude (35%), and volume surge (25%). Markets scoring above 0.05 on this scale proceed to deeper analysis; the top half of that group receive the most intensive treatment. Swing detection runs a rolling z-score against a 30-day trailing baseline across four time windows (1-hour, 6-hour, 1-day, 5-day), and a swing is flagged when the z-score exceeds 2.0. For markets where a statistically significant swing is confirmed, a two-step large language model pipeline selects the 2 to 3 traditional financial instruments with the strongest, most direct exposure to the market outcome: a first LLM call brainstorms seven candidate instruments broadly, and a second LLM call critically evaluates those candidates, selects the best 2 to 3, and writes a three-sentence rationale covering the company's business model, its specific exposure mechanism, and an honest confidence calibration on a 1-to-100 scale. All selected tickers are validated against Polygon.io before being used. Six statistical tests then quantify the relationship between each prediction-market probability series and the historical price series of the selected instrument: Pearson/Spearman correlation, Granger causality (bidirectional), lead-lag cross-correlation, event study, spike event study, and volatility spillover. Polymarket probability changes are computed as absolute ΔP rather than log-returns throughout, because prediction-market prices are bounded at [0, 1] with flat periods and binary terminal jumps that make log-returns ill-defined. Results are written to a SQLite database, uploaded to Cloudflare R2 cloud storage, and rendered as structured investment reports.

The system is differentiated from standard market-monitoring and sentiment analysis tools in two ways. First, it uses prediction-market probabilities — crowd-sourced, real-money, bounded — as the primary input signal, rather than price momentum, news text, or social sentiment. Second, instrument selection is not a fixed mapping or a keyword lookup; it is generated fresh for each market event by an LLM reasoning about specific business-model exposure, then statistically validated. An analyst monitoring a traditional newswire would receive the same information later and without the instrument-level mapping or the quantitative lead-lag analysis.

The system is live today. A public dashboard at https://sbornhor-ux.github.io/polymarket-arbitrage/ displays current market metadata, composite scores, per-market summaries, instrument rationale, and statistical test results for each flagged market. The pipeline runs on Railway with a 60-minute scan interval, using Cloudflare R2 for persistent storage and GitHub Pages for the dashboard front-end. Total infrastructure cost is approximately $5 per month (Railway Hobby plan; Cloudflare R2 free tier covers up to 10 GB storage and 10 million reads per month). A daily summary report is generated at 4:00 PM CST. The codebase uses Pydantic data contracts throughout the pipeline and implements exponential backoff on all external API calls.

The current system is a signal-detection and monitoring tool, not a trading system. It does not connect to a brokerage, execute orders, or provide backtested performance. The statistical tests establish whether a historical relationship between a prediction-market series and a financial instrument series exists; they do not guarantee that the relationship will persist prospectively. Signal accuracy under live conditions has not been systematically measured. Coverage is currently limited to Polymarket; other prediction venues (Kalshi, Manifold Markets) are not yet integrated. The pipeline requires API keys for Polygon.io, OpenAI, and Cloudflare R2. The team views the current build as a functional proof of concept that demonstrates the information-asymmetry thesis and the technical feasibility of the pipeline; prospective development would focus on live-trading integration, a backtesting engine, and expanded venue coverage.

---

> **Figure 1. System Architecture**
> *See Figure 1 (attached separately).*
> A pipeline diagram showing the five stages of the system: (1) Polymarket API ingestion of 25,000+ active contracts; (2) keyword-based finance filter and composite score ranking; (3) swing detection via rolling z-score; (4) LLM-driven instrument selection with Polygon.io validation; and (5) six-test statistical suite with results written to SQLite, uploaded to Cloudflare R2, and displayed on the GitHub Pages dashboard. The diagram illustrates how data flows from raw prediction-market events to structured investment signals.

> **Figure 2. Example Dashboard Signal**
> *See Figure 2 (attached separately).*
> A screenshot or mockup of the live dashboard showing an example flagged market, its composite score, the 2–3 selected financial instruments with confidence scores and rationales, and the results of the statistical tests (e.g., Granger causality p-value, lead-lag cross-correlation, event study cumulative abnormal return). X-axis: calendar date. Y-axis (left): prediction-market probability [0, 1]. Y-axis (right): financial instrument price. The figure illustrates the output a practitioner would use to assess whether a signal is worth acting on.

---

# Frequently Asked Questions

**1. Who is the target customer?**

The primary customer is an event-driven or macro-oriented investor — a discretionary hedge fund analyst, a systematic strategy desk, or a family office portfolio manager — whose investment thesis depends on anticipating how political, regulatory, or macroeconomic outcomes will affect specific financial instruments. These investors already monitor news flow and economic data; this system adds a real-time, quantified view of crowd beliefs with instrument-level mappings they would otherwise have to construct manually. A secondary customer is any researcher or practitioner building alternative-data workflows who wants a structured, programmatic feed of prediction-market signals mapped to financial assets.

**2. What problem does this solve?**

Prediction markets aggregate dispersed information efficiently, often before that information reaches mainstream financial data feeds. But translating a prediction-market probability shift into a tradeable view requires three steps that practitioners currently do manually and inconsistently: identifying which financial instruments are affected, establishing whether a statistical relationship between the prediction-market series and the instrument series exists, and determining the direction of that relationship. This system automates all three steps and presents the output in a structured, daily-updated format.

**3. How does the multi-stage pipeline work?**

The pipeline has five stages. First, the Polymarket API is polled every 60 minutes and all active contracts are retrieved (25,000+ at time of writing, per observed API output). Second, a keyword filter covering finance, macro, policy, geopolitics, and commodities reduces the universe to the finance-relevant subset, and a composite score — weighting market volume (40%), odds-swing magnitude (35%), and volume surge (25%) — ranks the remaining markets; those scoring above 0.05 and in the top half by score receive deep analysis. Third, a rolling z-score against a 30-day trailing baseline detects statistically significant probability swings (z ≥ 2.0) across 1-hour, 6-hour, 1-day, and 5-day windows. Fourth, a two-call LLM pipeline selects the 2–3 traditional financial instruments with the strongest exposure to the predicted outcome and validates each ticker against Polygon.io. Fifth, six statistical tests — Pearson/Spearman correlation, Granger causality, lead-lag cross-correlation (CCF), event study, spike event study, and volatility spillover — quantify the historical relationship between the prediction-market probability series and each selected instrument's price series, using absolute ΔP (not log-returns) throughout, because prediction-market prices are bounded [0, 1] and log-returns are ill-defined at the boundaries.

**4. Why not just trade Polymarket directly?**

Polymarket is a binary-outcome market with limited liquidity in most contracts, US-resident access restrictions for real-money trading, and resolution timelines that can span months. A trader who believes the crowd is mispricing a political outcome on Polymarket may not be able to put on a meaningful position there. The more liquid expression of the same view is through a correlated traditional instrument — an ETF, a Treasury bond, a commodity future — that trades continuously, has deep order books, and does not require a binary outcome resolution to generate returns if the view is correct. This system finds those liquid expressions automatically.

**5. How does the LLM instrument selection work?**

When a statistically significant swing is detected, the system calls an LLM twice. The first call prompts the model to brainstorm seven candidate financial instruments broadly, following explicit coverage rules (include a sector ETF, include the company named in the question if one is named, include macro ETFs for policy questions, etc.). The second call prompts the model to critically evaluate those seven candidates against the specific market question, select the 2–3 with the strongest and most direct exposure, and write a three-sentence rationale for each: one sentence on the company's business model, one on the specific exposure mechanism to the predicted outcome, and one honest assessment of the connection strength with caveats. Each selected ticker is then validated against Polygon.io before being passed downstream. The confidence score assigned to each instrument uses the full 1-to-100 range — a score of 90+ means the company is named directly in the question or the outcome determines its price; a score below 30 indicates tenuous second-order exposure.

**6. What does the live dashboard show?**

The public dashboard at https://sbornhor-ux.github.io/polymarket-arbitrage/ displays the output of the most recent scan. For each flagged market it shows: the market question and metadata (category, end date, current probability, volume, liquidity), the composite score and which flags were triggered, the selected financial instruments with their tickers, alignment (does price rise or fall if YES probability rises?), confidence scores, and LLM-generated rationales, and the results of the six statistical tests. The dashboard is updated at each 60-minute scan cycle and daily summary reports are generated at 4:00 PM CST. All underlying data is also available in JSON and CSV via Cloudflare R2 public URLs for programmatic access.

**7. What are the current limitations?**

The system is a monitoring and signal-detection tool only — it does not connect to any brokerage or execute trades. No live backtesting engine exists; the statistical tests establish historical relationships but do not provide prospective performance estimates. Signal accuracy under live conditions has not been systematically measured, and any practitioner using the output should treat it as one input among several rather than a standalone trading signal. Coverage is Polymarket-only; Kalshi, Manifold Markets, and other prediction venues are not yet integrated. The pipeline requires three API credentials (Polygon.io, OpenAI, Cloudflare R2), and the Polygon.io subscription determines how far back historical price data extends for the statistical tests.

**8. How is this different from traditional market-monitoring tools?**

Most market-monitoring tools track price, volume, and news flow in traditional financial markets. Sentiment tools parse text — headlines, earnings call transcripts, social media — and produce qualitative signals. This system uses a different primary input: the real-money probability of a discrete event, as aggregated by a crowd of bettors. Prediction-market probabilities are bounded, interpretable as crowd-consensus event probabilities, and updated continuously as information arrives. The cross-asset mapping and statistical validation layer then connects those probabilities to liquid financial instruments. The result is a structured, quantified signal that a practitioner can compare to existing market pricing without interpreting raw text or building a custom mapping from scratch.

---

*Export note: repeat the team-member header on each page and add page numbers in the footer in the final PDF export.*
