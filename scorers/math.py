"""Math agent eval scorers: calculation accuracy, tool usage, efficiency,
response format, and an LLM judge for calculation correctness."""

import os
import re

from autoevals import LLMClassifier
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BRAINTRUST_GATEWAY_URL = "https://gateway.braintrust.dev"
GATEWAY_URL = os.getenv("BRAINTRUST_GATEWAY_URL", DEFAULT_BRAINTRUST_GATEWAY_URL)
API_KEY = os.getenv("BRAINTRUST_API_KEY")
JUDGE_MODEL = "gpt-4o"

VALID_MATH_TOOLS = {"add", "subtract", "multiply", "divide"}


async def calculation_accuracy_scorer(input, output, expected):
    """Check if the calculated answer matches the expected value."""
    if not expected or "expected_answer" not in expected:
        return 0.5

    expected_answer = expected["expected_answer"]

    # Extract the final assistant message
    messages = output.get("messages", [])
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""

        if content and role == "assistant":
            # Check if expected answer appears in the response
            if str(expected_answer) in str(content):
                return 1.0
            break

    return 0.0


async def tool_usage_scorer(output, metadata=None):
    """Check if the agent used math tools appropriately."""
    if not metadata:
        return 0.5

    tool_calls = metadata.get("tool_calls", [])
    if not tool_calls:
        return 0.0

    # Check that math tools were used
    tool_names = [tc["name"] for tc in tool_calls]
    used_valid_tools = any(name in VALID_MATH_TOOLS for name in tool_names)
    return 1.0 if used_valid_tools else 0.0


async def efficiency_scorer(output, metadata=None):
    """Score based on minimal unnecessary tool calls."""
    if not metadata:
        return 0.5

    tool_calls = metadata.get("tool_calls", [])
    num_calls = len(tool_calls)

    # Ideal: 1-3 tool calls for most operations
    if num_calls <= 2:
        return 1.0
    elif num_calls <= 4:
        return 0.8
    elif num_calls <= 6:
        return 0.6
    else:
        return 0.4


async def response_format_scorer(output):
    """Check if the response is clear and includes the final answer."""
    messages = output.get("messages", [])

    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""

        if content and role == "assistant":
            # Look for numbers in the final response
            if re.search(r"\d+", content):
                return 1.0
            break

    return 0.0


# ---------------------------------------------------------------------------
# Calculation Correctness (LLM-as-a-judge)
# ---------------------------------------------------------------------------

CALCULATION_CORRECTNESS_PROMPT = """
You are evaluating a math agent's calculation.

Question: {{input}}
Agent's Response: {{output}}
Expected Answer: {{expected}}

Evaluate whether:
1. The calculation is mathematically correct
2. The final answer matches the expected result
3. The reasoning (if shown) is sound

Respond with:
CORRECT - Calculation and answer are correct
INCORRECT - Calculation or answer is wrong
"""

CALCULATION_CORRECTNESS_CHOICE_SCORES = {"CORRECT": 1.0, "INCORRECT": 0.0}

calculation_correctness_scorer = LLMClassifier(
    name="Calculation Correctness",
    prompt_template=CALCULATION_CORRECTNESS_PROMPT,
    choice_scores=CALCULATION_CORRECTNESS_CHOICE_SCORES,
    use_cot=True,
    model=JUDGE_MODEL,
    base_url=GATEWAY_URL,
    api_key=API_KEY,
)
