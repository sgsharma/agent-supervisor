"""Braintrust prompt definitions for the agent-supervisor project.

Each prompt's text is read from the sibling `.md` files in this directory, so the
files are the single source of truth shared with the running agents (loaded in
src/config.py). Publish to Braintrust with:

    bt functions push prompts/push_prompts.py --if-exists replace -p agent-supervisor

The supervisor prompt keeps its `{{{current_date}}}` Mustache variable so it stays
templated in Braintrust; src/config.py renders the date for local runs.
"""

from pathlib import Path

import braintrust

PROMPTS_DIR = Path(__file__).resolve().parent
MODEL = "gpt-4o-mini"


def _load(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


project = braintrust.projects.create(name="agent-supervisor")

project.prompts.create(
    name="Supervisor Agent",
    slug="supervisor-agent",
    description="Orchestrator prompt that routes user queries to the research and math subagents.",
    model=MODEL,
    messages=[
        {"role": "system", "content": _load("supervisor.md")},
        {"role": "user", "content": "{{{question}}}"},
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
        {"role": "user", "content": "{{{question}}}"},
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
        {"role": "user", "content": "{{{question}}}"},
    ],
    if_exists="replace",
)
