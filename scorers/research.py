"""Research agent eval scorers: web search usage, source attribution,
efficiency, and an LLM judge for answer quality."""

import os
import re

from autoevals import LLMClassifier
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BRAINTRUST_GATEWAY_URL = "https://gateway.braintrust.dev"
GATEWAY_URL = os.getenv("BRAINTRUST_GATEWAY_URL", DEFAULT_BRAINTRUST_GATEWAY_URL)
API_KEY = os.getenv("BRAINTRUST_API_KEY")
JUDGE_MODEL = "gpt-4o"

SEARCH_TOOL = "tavily_search_results_json"


async def web_search_usage_scorer(output, metadata=None):
    """Check if the agent used web search when appropriate."""
    if metadata and metadata.get("used_web_search"):
        return 1.0
    return 0.0


async def source_attribution_scorer(output):
    """Check if the response includes a URL citation."""
    messages = output.get("messages", [])
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""
        if content and role == "assistant":
            if re.search(r"https?://", content):
                return 1.0
            break
    return 0.0


async def efficiency_scorer(output, metadata=None):
    """Score based on minimal tool calls (should use search efficiently)."""
    if not metadata:
        return 0.5

    tool_calls = metadata.get("tool_calls", [])
    num_searches = tool_calls.count(SEARCH_TOOL)

    # Ideal: 1-2 searches, penalize excessive searching
    if num_searches == 1:
        return 1.0
    elif num_searches == 2:
        return 0.9
    elif num_searches <= 4:
        return 0.7
    else:
        return 0.5


# ---------------------------------------------------------------------------
# Answer Quality (LLM-as-a-judge)
# ---------------------------------------------------------------------------

ANSWER_QUALITY_PROMPT = """
You are evaluating a research agent's response to a factual question.

Question: {{input}}
Response: {{output}}

Evaluate the response on:
1. ACCURACY: Is the information correct and factual?
2. COMPLETENESS: Does it answer the question fully?
3. CLARITY: Is it well-structured and clear?
4. RELEVANCE: Does it address what was asked?

Respond with:
EXCELLENT - Accurate, complete, clear, and highly relevant
GOOD - Mostly accurate and complete with minor issues
FAIR - Some accuracy or completeness issues
POOR - Inaccurate, incomplete, or irrelevant
"""

ANSWER_QUALITY_CHOICE_SCORES = {
    "EXCELLENT": 1.0,
    "GOOD": 0.75,
    "FAIR": 0.5,
    "POOR": 0.0,
}

answer_quality_scorer = LLMClassifier(
    name="Answer Quality",
    prompt_template=ANSWER_QUALITY_PROMPT,
    choice_scores=ANSWER_QUALITY_CHOICE_SCORES,
    use_cot=True,
    model=JUDGE_MODEL,
    base_url=GATEWAY_URL,
    api_key=API_KEY,
)
