"""
Final Project Synthesizer - Investment Report Generator

This agent receives data from another agent and generates structured investment
reports in Markdown format. It produces three key sections:
1. Topic Overview
2. Asset Class Correlation Analysis
3. Conclusion & Investment Opportunities

This agent is designed to be called by other agents with pre-processed data.
"""

from datetime import datetime
import json
import os
from langchain.chat_models import init_chat_model


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# Template for Sections 1 & 2: Topic Overview and Asset Class Correlation
section_writer_instructions = """You are an expert investment analyst and technical writer crafting one section of an investment analysis report.

Section Topic:
{section_topic}

Guidelines for writing:

1. Technical Accuracy:
- Include specific data points and metrics
- Reference concrete examples from the financial markets
- Use precise financial terminology
- Support claims with verifiable information

2. Length and Style:
- Strict 150–200 word limit (excluding title and sources)
- No marketing language
- Investment-focused, analytical tone
- Write in clear, professional language
- Start with your single most important insight in **bold**
- Use short paragraphs (2–3 sentences max)

3. Structure:
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2–3 key items (using Markdown table syntax)
  * Or a short list (3–5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title and URL
  * Format: `- [Title] : [URL]`

4. Content Requirements:
- Include at least one specific example or case study
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

5. Use this context to inform your analysis:
{context}

6. Quality Checks:
- Exactly 150–200 words (excluding title and sources)
- Careful use of only ONE structural element (table or list) and only if it helps clarify your point
- One specific example or case study
- Starts with bold insight
- No preamble before content begins
- Sources cited at end"""


# Template for per-market signal summaries
market_summaries_instructions = """You are a senior investment analyst writing a "Market Signal Summaries" \
section for an investment report.

For each Polymarket prediction market listed below, write a focused 2–3 sentence summary that:
1. States what the market is asking and its approximate current probability.
2. Explains what caused it to be flagged as a financial signal — i.e. what price movement, \
swing, or statistical pattern was detected.
3. Names the most relevant financial instrument, its alignment (moves "with" or "against" YES), \
and the key statistical finding (correlation, lead-lag signal, or event study result).

Rules:
- Use a concise, analytical tone — no marketing language.
- Start each entry with ### [Market Question] as a Markdown subheading.
- If confidence in the instrument selection is low (< 40), note that the connection is speculative.
- Do not add a preamble — start directly with the first ### heading.

Markets:
{markets_context}
"""


# Template for Introduction and Conclusion sections
final_section_writer_instructions = """You are an expert investment analyst crafting a section that synthesizes information from the rest of the report.

Section Type:
{section_topic}

Available Report Context:
{context}

1. Section-Specific Approach:

For Introduction (Report Title):
- Use # for report title in Markdown
- 50–100 word limit
- Write in simple, clear, professional language
- Focus on the core investment thesis in 1–2 paragraphs
- Use a clear narrative arc to introduce the report's purpose
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion:
- Use ## for section title in Markdown
- 100–150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill key insights across asset classes or opportunities
    * Keep table entries clear and concise
- For non-comparative reports:
    * Only use ONE optional structural element IF it aids clarity:
    * Either a focused table (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps, recommendations, or forward-looking implications
- No sources section needed

2. Writing Approach:
- Use concrete details and specific recommendations
- Make every word count
- Focus on your single most important takeaway
- No preamble in output
- No word count labels or commentary in output

3. Quality Checks:
- For Introduction: 50–100 words, # for title, no structural elements, no sources
- For Conclusion: 100–150 words, ## for title, max ONE structural element, no sources
- Markdown format throughout
- Clear actionable insights or recommendations"""


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_trend_analysis_data(json_file_path):
    """
    Load trend analysis data from JSON file.

    Args:
        json_file_path (str): Path to the trend analysis JSON file

    Returns:
        tuple: (topic, formatted_context_data)
            - topic: str representing the main investment topic
            - formatted_context_data: str containing formatted analysis data
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract metadata
        scan_timestamp = data.get('scan_timestamp', 'N/A')
        analysis_timestamp = data.get('analysis_timestamp', 'N/A')
        analyst_version = data.get('analyst_version', 'N/A')

        # Extract pairs data and format for context
        pairs = data.get('pairs', [])

        # Use first pair's polymarket question as the topic
        topic = pairs[0]['polymarket_question'] if pairs else "Trend Analysis Report"

        # Format the context data
        context_lines = [
            f"Analysis Metadata:",
            f"- Scan Timestamp: {scan_timestamp}",
            f"- Analysis Timestamp: {analysis_timestamp}",
            f"- Analyst Version: {analyst_version}",
            f"- Number of Asset Pairs Analyzed: {len(pairs)}",
            f"\nKey Findings from Trend Analysis:\n"
        ]

        for i, pair in enumerate(pairs, 1):
            context_lines.append(f"\nPair {i}: {pair['ticker_name']} ({pair['ticker']})")
            context_lines.append(f"  Polymarket Question: {pair['polymarket_question']}")

            # Add correlation analysis
            correlation = pair.get('correlation', {})
            if correlation:
                context_lines.append(f"  Correlation Analysis:")
                context_lines.append(f"    - Pearson r: {correlation.get('pearson_r', 'N/A')}")
                context_lines.append(f"    - P-value: {correlation.get('pearson_p', 'N/A')}")
                context_lines.append(f"    - Interpretation: {correlation.get('interpretation', 'N/A')}")

            # Add cointegration
            cointegration = pair.get('cointegration', {})
            if cointegration:
                context_lines.append(f"  Cointegration Test:")
                context_lines.append(f"    - Is Cointegrated: {cointegration.get('is_cointegrated', 'N/A')}")
                context_lines.append(f"    - Interpretation: {cointegration.get('interpretation', 'N/A')}")

            # Add volume analysis
            volume = pair.get('volume_analysis', {})
            if volume:
                context_lines.append(f"  Volume Analysis:")
                context_lines.append(f"    - Spike Coincidence Ratio: {volume.get('coincidence_ratio', 'N/A')}")
                context_lines.append(f"    - Interpretation: {volume.get('interpretation', 'N/A')}")

            # Add DTW (shape similarity)
            dtw = pair.get('dtw', {})
            if dtw:
                context_lines.append(f"  Price Pattern Similarity (DTW):")
                context_lines.append(f"    - Normalized Distance: {dtw.get('normalized_distance', 'N/A')}")
                context_lines.append(f"    - Interpretation: {dtw.get('interpretation', 'N/A')}")

        formatted_context_data = '\n'.join(context_lines)

        print(f"[OK] Loaded trend analysis data from {json_file_path}")
        print(f"[OK] Topic: {topic}")
        print(f"[OK] Pairs analyzed: {len(pairs)}\n")

        return topic, formatted_context_data

    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        raise


# =============================================================================
# SECTION GENERATION FUNCTIONS
# =============================================================================

def generate_section(llm, section_topic, context, section_type="body"):
    """
    Generate a single section of the report using GPT via LangChain.

    Args:
        llm: LangChain LLM instance (initialized with init_chat_model)
        section_topic: The topic or title of the section
        context: Contextual data or background information
        section_type: Either "body" (uses section_writer_instructions)
                     or "final" (uses final_section_writer_instructions)

    Returns:
        str: The generated section content in Markdown format
    """
    # Select the appropriate prompt template
    if section_type == "body":
        prompt_template = section_writer_instructions
    else:
        prompt_template = final_section_writer_instructions

    # Format the prompt with section-specific content
    system_prompt = prompt_template.format(
        section_topic=section_topic,
        context=context
    )

    try:
        # Call LLM via LangChain
        from langchain_core.messages import HumanMessage

        message = llm.invoke([HumanMessage(content=system_prompt)])

        # Extract and return the generated text
        return message.content

    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise


def generate_topic_overview(llm, topic, context):
    """
    Generate Section 1: Topic Overview

    Args:
        llm: LangChain LLM instance
        topic: The investment topic (e.g., "Global Semiconductor Supply Chain")
        context: Background information or data about the topic

    Returns:
        str: Generated Topic Overview section in Markdown
    """
    print("Generating Topic Overview...")
    return generate_section(
        llm,
        section_topic=f"Topic Overview: {topic}",
        context=context,
        section_type="body"
    )


def generate_correlation_analysis(llm, topic, context):
    """
    Generate Section 2: Asset Class Correlation Analysis

    Args:
        llm: LangChain LLM instance
        topic: The investment topic
        context: Background information including any correlation data

    Returns:
        str: Generated Asset Class Correlation Analysis section in Markdown
    """
    print("Generating Asset Class Correlation Analysis...")
    return generate_section(
        llm,
        section_topic=f"Asset Class Correlation Analysis: {topic}",
        context=context,
        section_type="body"
    )


def generate_investment_opportunities(llm, topic, context):
    """
    Generate Section 3: Conclusion & Investment Opportunities

    Args:
        llm: LangChain LLM instance
        topic: The investment topic
        context: Context including findings from previous sections

    Returns:
        str: Generated Conclusion & Investment Opportunities section in Markdown
    """
    print("Generating Investment Opportunities...")
    return generate_section(
        llm,
        section_topic=f"Conclusion & Investment Opportunities: {topic}",
        context=context,
        section_type="final"
    )


def generate_introduction(llm, topic):
    """
    Generate Introduction section

    Args:
        llm: LangChain LLM instance
        topic: The investment topic

    Returns:
        str: Generated Introduction in Markdown
    """
    print("Generating Introduction...")
    return generate_section(
        llm,
        section_topic="Introduction",
        context=f"Report Topic: {topic}",
        section_type="final"
    )


def generate_market_summaries(llm, pairs: list) -> str:
    """
    Generate Section: Per-Market Signal Summaries.

    Groups pairs by Polymarket market, formats the top instruments and key
    findings for each market, then calls the LLM once to produce a brief
    analyst summary per market.

    Only markets with at least one pair where overall_similarity_score >= 0.05
    are included (filters out markets with no meaningful signal).

    Args:
        llm: LangChain LLM instance
        pairs: List of pair analysis dicts from trend_analysis JSON

    Returns:
        str: Markdown section with one ### subsection per market
    """
    from collections import defaultdict

    # Group pairs by market, keep top 3 by overall_similarity_score
    market_groups: dict[str, dict] = {}
    for p in pairs:
        mid = p.get("polymarket_id", "")
        q   = p.get("polymarket_question", "")
        if not mid or not q:
            continue
        if mid not in market_groups:
            market_groups[mid] = {"question": q, "pairs": []}
        market_groups[mid]["pairs"].append(p)

    # Sort each group by score descending, keep top 3 instruments per market
    for g in market_groups.values():
        g["pairs"].sort(key=lambda x: x.get("overall_similarity_score", 0), reverse=True)
        g["pairs"] = g["pairs"][:3]

    # Include all markets with at least some data (no minimum score filter)
    meaningful = list(market_groups.values())

    # Sort by top pair score descending
    meaningful.sort(key=lambda g: g["pairs"][0].get("overall_similarity_score", 0), reverse=True)

    if not meaningful:
        return "## Market Signal Summaries\n\n*No markets with sufficient signal data.*\n"

    # Build context block for the prompt
    context_lines = []
    for g in meaningful:
        q = g["question"]
        context_lines.append(f"\n### {q}")
        for p in g["pairs"]:
            ticker = p.get("ticker", "")
            ticker_name = p.get("ticker_name", ticker)
            score = p.get("overall_similarity_score", 0)
            conf = p.get("confidence_level", "low")
            findings = p.get("key_findings", [])
            summary = p.get("agent_summary", "")
            # Include agent summary and top 2 key findings
            context_lines.append(f"  Instrument: {ticker} ({ticker_name}) | stat_score={score:.2f} | confidence={conf}")
            if summary:
                context_lines.append(f"  Agent summary: {summary}")
            for f in findings[:2]:
                context_lines.append(f"  Finding: {f}")

    markets_context = "\n".join(context_lines)

    prompt = market_summaries_instructions.format(markets_context=markets_context)

    print("Generating Market Signal Summaries...")
    try:
        from langchain_core.messages import HumanMessage
        message = llm.invoke([HumanMessage(content=prompt)])
        body = message.content.strip()
    except Exception as e:
        print(f"Error generating market summaries: {e}")
        body = "*Market summaries could not be generated.*"

    return f"## Market Signal Summaries\n\n{body}\n"


# =============================================================================
# MAIN REPORT GENERATION PIPELINE
# =============================================================================

def generate_investment_report(llm, topic, context_data, output_file=None, pairs=None):
    """
    Generate a complete investment report with all sections.

    This pipeline:
    1. Receives a topic and context data from another agent
    2. Routes each section to the appropriate prompt template
    3. Calls GPT for each section sequentially
    4. Assembles all sections into a single Markdown document
    5. Saves to a .md file and returns the content

    Args:
        llm: LangChain LLM instance (initialized with init_chat_model)
        topic (str): The main investment topic
        context_data (str): Raw data or background information about the topic
        output_file (str, optional): Path to save the report. Defaults to
                                     "investment_report_[timestamp].md"

    Returns:
        str: The complete report in Markdown format
    """

    print("\n" + "="*70)
    print(f"INVESTMENT REPORT GENERATOR")
    print(f"Topic: {topic}")
    print("="*70 + "\n")

    try:
        # Generate Introduction
        introduction = generate_introduction(llm, topic)

        # Generate Section 1: Topic Overview
        topic_overview = generate_topic_overview(llm, topic, context_data)

        # Generate Section 2: Asset Class Correlation Analysis
        correlation_analysis = generate_correlation_analysis(llm, topic, context_data)

        # Generate Section 3: Per-Market Signal Summaries
        market_summaries = generate_market_summaries(llm, pairs or [])

        # Generate Section 4: Investment Opportunities
        investment_opportunities = generate_investment_opportunities(
            llm,
            topic,
            f"Context from previous sections:\n{topic_overview}\n\n{correlation_analysis}"
        )

        # Assemble all sections into final report
        final_report = f"""{introduction}

{topic_overview}

{correlation_analysis}

{market_summaries}

{investment_opportunities}
"""

        # Save report to file
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"investment_report_{timestamp}.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_report)

        print(f"\nReport generated successfully!")
        print(f"Saved to: {output_file}")
        print("="*70 + "\n")

        return final_report

    except Exception as e:
        print(f"API Error: {e}")
        return None
    except IOError as e:
        print(f"File Error: {e}")
        return None


# =============================================================================
# CONVENIENCE FUNCTION: GENERATE REPORT FROM JSON FILE
# =============================================================================

def generate_report_from_json(json_file_path, output_file=None):
    """
    Convenience function to load trend analysis JSON and generate report.

    This is the main entry point for using the synthesizer agent with
    trend analysis data from another agent. It initializes the LLM and
    generates the complete investment report.

    Args:
        json_file_path (str): Path to the trend_analysis JSON file
        output_file (str, optional): Path to save the report

    Returns:
        str: The complete report in Markdown format
    """
    try:
        # Initialize the LLM using LangChain's init_chat_model
        # This will automatically detect and use your OPENAI_API_KEY
        llm = init_chat_model("gpt-4o-mini", temperature=0, model_provider="openai")

        # Load the trend analysis data
        topic, context_data = load_trend_analysis_data(json_file_path)

        # Also load raw pairs for per-market summaries
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        raw_pairs = raw_data.get('pairs', [])

        # Generate the investment report
        report = generate_investment_report(
            llm,
            topic=topic,
            context_data=context_data,
            output_file=output_file,
            pairs=raw_pairs,
        )

        return report

    except Exception as e:
        print(f"Error generating report from JSON: {e}")
        return None
