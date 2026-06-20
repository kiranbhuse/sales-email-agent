# File: agent.py
# PURPOSE: Connects to ALL MCP servers declared in mcp_config.json,
#          merges their tools, then runs the ReAct loop to produce
#          a personalized sales email.
# HOW TO RUN:
#   python agent.py "Jane Smith" "Acme Corp" "VP Sales" \
#       "Salesforce Einstein AI Copilot"
#
# WHY THIS VERSION DIFFERS FROM THE EARLIER DRAFT: the original only
# connected to ONE server (custom-scoring), so tools like
# brave_web_search, fetch, read_query, read_file, and write_file were
# never available to the agent. This version uses AsyncExitStack to
# keep all 5 MCP server connections open at once, merges their tools,
# and routes each tool call to the correct server.

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic

load_dotenv()
client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"  # current fast/cheap tier as of mid-2026

SYSTEM = """
You are a sales email personalization agent.

You have MCP tools available from 5 categories. Follow these steps
IN ORDER. Do not skip any step:

1. SEARCH: brave_web_search — search for the prospect and company
2. SCRAPE: fetch — scrape the company homepage after search
3. CRM: read_query — SELECT from the accounts table in SQLite
4. SCORE: score_lead — call after the CRM lookup
5. TEMPLATE: read_file (templates/email_template.txt) — load before
   drafting the email
6. Draft the email using the template structure
7. VALIDATE: check_personalization — call after drafting; if the
   result includes any failed checks, revise the email and validate
   again before moving on
8. SAVE: write_file (output/{prospect_name}_email.txt) — THIS STEP IS
   MANDATORY. The task is NOT complete until write_file has been
   called successfully.
9. LOG: write_query — INSERT a row into the runs table with this run's
   data. THIS STEP IS ALSO MANDATORY.

CRITICAL RULE: You must call both write_file and write_query before
ending your turn. Never invent facts. Never skip validation.
"""


def load_mcp_config(path="mcp_config.json"):
    with open(path) as f:
        return json.load(f)["mcpServers"]


async def run_agent(prospect: str, company: str, role: str, pitch: str):
    config = load_mcp_config()

    async with AsyncExitStack() as stack:
        sessions = {}        # server_name -> ClientSession
        tool_to_server = {}  # tool_name -> server_name
        all_tools = []       # merged tool list for the Anthropic API

        # Connect to every server declared in mcp_config.json
        for server_name, server_cfg in config.items():
            params = StdioServerParameters(
                command=server_cfg["command"],
                args=server_cfg.get("args", []),
                env={**server_cfg.get("env", {}), **os.environ},
            )
            try:
                read, write = await stack.enter_async_context(
                    stdio_client(params)
                )
                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                sessions[server_name] = session

                tools_response = await session.list_tools()
                tool_names = [t.name for t in tools_response.tools]
                print(f"[{server_name}] tools: {tool_names}")

                for t in tools_response.tools:
                    tool_to_server[t.name] = server_name
                    all_tools.append({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema,
                    })
            except Exception as e:
                print(f"[{server_name}] FAILED to connect: {e}")

        if not all_tools:
            print("No MCP tools available. Check mcp_config.json and "
                  "that each server starts without errors.")
            return

        print(f"\nTotal tools available to agent: "
              f"{[t['name'] for t in all_tools]}\n")

        messages = [{
            "role": "user",
            "content": (
                f"Write a personalized email for {prospect}, {role} at "
                f"{company}. Product: {pitch}"
            ),
        }]

        write_file_called = False
        write_query_called = False
        start = time.time()

        for iteration in range(12):
            print(f"--- Iteration {iteration + 1} ---")
            response = client.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=SYSTEM,
                tools=all_tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                missing_steps = []
                if not write_file_called:
                    missing_steps.append("write_file")
                if not write_query_called:
                    missing_steps.append("write_query")

                if missing_steps:
                    print(f"  [WARNING] Agent stopped without calling: "
                          f"{missing_steps}. Forcing it to continue.")
                    messages.append({
                        "role": "assistant",
                        "content": response.content,
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have not called {missing_steps} yet. "
                            f"You MUST call these now before finishing."
                        ),
                    })
                    continue

                print(f"Done in {time.time() - start:.1f}s")
                for block in response.content:
                    if block.type == "text":
                        print(block.text)
                break

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if block.name == "write_file":
                        write_file_called = True
                    if block.name == "write_query":
                        write_query_called = True

                    server_name = tool_to_server.get(block.name)
                    if not server_name:
                        result_text = (
                            f"ERROR: tool {block.name} not found "
                            f"on any connected server"
                        )
                    else:
                        print(f"  -> Calling {block.name} "
                              f"on [{server_name}] with {block.input}")
                        session = sessions[server_name]
                        result = await session.call_tool(
                            block.name, block.input
                        )
                        result_text = (
                            result.content[0].text
                            if result.content else "no result"
                        )
                        print(f"     Result: {result_text[:150]}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            print("Reached max iterations without a final answer.")


if __name__ == "__main__":
    args = sys.argv[1:]
    asyncio.run(run_agent(
        args[0] if len(args) > 0 else "Jane Smith",
        args[1] if len(args) > 1 else "Acme Corp",
        args[2] if len(args) > 2 else "VP Sales",
        args[3] if len(args) > 3 else "Salesforce Einstein Copilot",
    ))