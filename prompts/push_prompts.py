"""Braintrust prompt definitions for the agent-supervisor project.

Each prompt's text is read from the sibling `.md` files in this directory, so the
files are the single source of truth shared with the running agents (loaded in
src/config.py). Publish to Braintrust with:

    bt functions push prompts/push_prompts.py --if-exists replace -p agent-supervisor

These are pushed as system-message-only prompts. The conversation comes from the
dataset row at eval/playground time via the "append dataset messages" setting
(pointing at the `messages` path), so we do NOT hardcode a `{{{question}}}` user
turn — that would inject an empty user message and the model would just reply
with a generic greeting.

The supervisor prompt's `{{{current_date}}}` placeholder is rendered to a concrete
date at push time, since the playground/eval dataset does not supply a
`current_date` variable. (src/config.py renders it dynamically for local runs.)
"""

from datetime import datetime
from pathlib import Path

import braintrust

PROMPTS_DIR = Path(__file__).resolve().parent
MODEL = "gpt-4o-mini"
TODAY = datetime.now().strftime("%Y-%m-%d")


def _load(filename: str) -> str:
    text = (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()
    return text.replace("{{{current_date}}}", TODAY)


project = braintrust.projects.create(name="agent-supervisor")

project.prompts.create(
    name="Supervisor Agent",
    slug="supervisor-agent",
    description="Orchestrator prompt that routes user queries to the research and math subagents.",
    model=MODEL,
    messages=[
        {"role": "system", "content": _load("supervisor.md")},
    ],
    if_exists="replace",
)

project.prompts.create(
    name="Research Agent",
    slug="research-agent",
    description="Subagent prompt for web search and factual lookups.",
    model=MODEL,
    messages=[
        {"role": "system", "content": _load("research_agent.md")},
    ],
    if_exists="replace",
)

project.prompts.create(
    name="Math Agent",
    slug="math-agent",
    description="Subagent prompt for arithmetic and calculations.",
    model=MODEL,
    messages=[
        {"role": "system", "content": _load("math_agent.md")},
    ],
    if_exists="replace",
)
