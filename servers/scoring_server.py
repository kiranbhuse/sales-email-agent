# File: servers/scoring_server.py
# PURPOSE: Custom MCP server for lead scoring and email quality check.
# HOW IT RUNS: agent.py launches it automatically via mcp_config.json.
# DO NOT run this directly — it communicates via stdio.

import json
import os

import anthropic
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("custom-scoring")
llm = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"  # current fast/cheap tier as of mid-2026


@mcp.tool()
def score_lead(prospect: str, company: str,
               news_summary: str, crm_data: str) -> dict:
    """
    Score a sales lead 1-10 (10 = hottest). Return score + rationale.
    Called AFTER web search and CRM lookup are complete.
    """
    prompt = (
        "Score this lead 1-10 (10 = hottest opportunity).\n"
        f"Prospect: {prospect} at {company}\n"
        f"Recent news: {news_summary}\n"
        f"CRM history: {crm_data}\n"
        "Respond with only valid JSON, no other text: "
        '{"score": N, "rationale": "one sentence"}'
    )

    response = llm.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()

    # The LLM is asked for JSON, but may still wrap it in markdown
    # fences or add stray text — parse defensively rather than
    # trusting the raw string.
    raw_text = raw_text.removeprefix("```json").removeprefix("```")
    raw_text = raw_text.removesuffix("```").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Fail safe rather than crashing the whole agent loop —
        # return a clearly-flagged fallback the agent can react to.
        return {
            "score": 0,
            "rationale": "Could not parse LLM scoring response",
            "raw_response": raw_text,
        }


@mcp.tool()
def check_personalization(email: str, prospect_first_name: str,
                           company: str, news_keyword: str) -> dict:
    """
    Validate email quality. Called AFTER the draft is written.
    Returns {score: 'N/4', passed: [...], failed: [...], action: ...}
    If failed list is non-empty, the agent should revise the email.
    """
    email_lower = email.lower()

    checks = {
        "mentions_first_name": prospect_first_name in email,
        "mentions_company": company in email,
        "references_news": news_keyword.lower() in email_lower,
        "has_soft_cta": any(
            phrase in email_lower
            for phrase in ["chat", "call", "15 min", "curious",
                            "worth a conversation"]
        ),
    }

    passed = [name for name, ok in checks.items() if ok]
    failed = [name for name, ok in checks.items() if not ok]

    return {
        "score": f"{len(passed)}/4",
        "passed": passed,
        "failed": failed,
        "action": "APPROVED" if not failed else f"REVISE: fix {failed}",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")