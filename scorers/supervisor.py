"""Supervisor eval scorers: routing accuracy, response quality, step efficiency."""

import os
from typing import Literal

from autoevals import LLMClassifier
from braintrust.oai import wrap_openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

DEFAULT_BRAINTRUST_GATEWAY_URL = "https://gateway.braintrust.dev"
GATEWAY_URL = os.getenv("BRAINTRUST_GATEWAY_URL", DEFAULT_BRAINTRUST_GATEWAY_URL)
API_KEY = os.getenv("BRAINTRUST_API_KEY")
JUDGE_MODEL = "gpt-4o"

# Async client for the routing scorer's own structured LLM call.
judge_client = wrap_openai(AsyncOpenAI(api_key=API_KEY, base_url=GATEWAY_URL))


# ---------------------------------------------------------------------------
# Routing Accuracy (trace-based code scorer)
# ---------------------------------------------------------------------------


class RoutingAccuracyOutput(BaseModel):
    """Structured output for routing accuracy evaluation."""

    choice: Literal["A", "B", "C", "D"]
    reasoning: str


ROUTING_ACCURACY_PROMPT = """
You are an expert evaluator of AI agent routing systems. Your task is to determine whether a user question was correctly routed to the appropriate agents.

The system has the following specialized agents:
1. **MathAgent**: Should handle mathematical calculations, arithmetic, equations, numerical problems, and any query requiring computation with specific numbers.
2. **ResearchAgent**: Should handle factual questions, information lookup, current events, geography, history, statistics, and any query requiring external knowledge or web search.

The supervisor can:
- Route to a single agent
- Route to multiple agents (if the query requires both research and math)
- Answer directly without routing (for simple greetings, conversational queries, or ambiguous questions)

**User Question**: {input}

**Agents Called**: {agents_called}

**Evaluation Criteria**:

Math queries (e.g., "What is 25 * 4?", "Calculate 100 + 50"):
- SHOULD route to MathAgent only
- Should NOT route to ResearchAgent unless additional context/research is needed

Research queries (e.g., "Who is the president?", "What is the capital of France?"):
- SHOULD route to ResearchAgent only
- Should NOT route to MathAgent unless calculation is involved

Hybrid queries (e.g., "What year was the Eiffel Tower built? Multiply that by 2."):
- SHOULD route to BOTH ResearchAgent (for the fact) AND MathAgent (for the calculation)
- Order may vary

Simple conversational queries (e.g., "hello", "help me understand this"):
- CAN be answered directly by supervisor (no routing)
- Routing is acceptable but not required

**Task**: Evaluate the routing decision and respond with your reasoning, then select ONE of these options:

(A) CORRECT - All routing decisions were appropriate. This includes:
    - Correct agent(s) called for the query type
    - No routing when direct answer is appropriate (simple greetings, chat)
    - Multiple agents called when query requires both research and calculation

(B) MOSTLY_CORRECT - Routing was generally correct but with minor issues:
    - Correct agents called but could have answered directly
    - Correct primary agent but missed a secondary agent for optimal answer

(C) PARTIALLY_WRONG - Significant routing issues:
    - Wrong agent called but got lucky with the answer
    - Correct agent plus unnecessary additional agent(s)
    - Missing critical agent for the query type

(D) INCORRECT - Routing was wrong:
    - Wrong agent(s) called for the query type
    - No routing when specialized agent was clearly needed
    - Multiple wrong agents called
"""

ROUTING_CHOICE_SCORES = {"A": 1.0, "B": 0.7, "C": 0.3, "D": 0.0}


async def routing_accuracy_scorer(input, output, expected, metadata, trace):
    spans = await trace.get_spans(span_type=["task"])
    agents_called_str = "None (supervisor answered directly)"
    agents_called = []
    for span in spans:
        span_name = span.span_attributes.get("name", None)
        if span_name in ["MathAgent", "ResearchAgent"]:
            agents_called.append(span_name)

    if agents_called:
        agents_called_str = ", ".join(agents_called)

    prompt = ROUTING_ACCURACY_PROMPT.format(
        input=input, agents_called=agents_called_str
    )
    response = await judge_client.responses.parse(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        text_format=RoutingAccuracyOutput,
    )
    output = response.output_parsed
    return {
        "name": "Routing Accuracy",
        "score": ROUTING_CHOICE_SCORES.get(output.choice, 0.0) if output else 0.0,
        "metadata": {
            "agents_called": agents_called_str,
            "reasoning": output.reasoning if output else "No output",
            "choice": output.choice if output else "D",
        },
    }


# ---------------------------------------------------------------------------
# Response Quality (LLM-as-a-judge)
# ---------------------------------------------------------------------------

RESPONSE_QUALITY_PROMPT = """
You are an expert evaluator of AI assistant responses. Your task is to assess the quality, accuracy, and completeness of responses.

User Question: {{input}}
AI Response: {{output}}

Evaluate the response based on:
1. ACCURACY: Is the information provided correct?
2. COMPLETENESS: Does it fully answer the question?
3. CLARITY: Is the response clear and well-structured?
4. RELEVANCE: Does it directly address what was asked?

For math questions, check if the calculation is correct.
For factual questions, assess if the information appears accurate and complete.

Respond with:
EXCELLENT - Response is accurate, complete, clear, and highly relevant
GOOD - Response is mostly accurate and complete with minor issues
FAIR - Response has some accuracy or completeness issues
POOR - Response is inaccurate, incomplete, or irrelevant
"""

RESPONSE_QUALITY_CHOICE_SCORES = {
    "EXCELLENT": 1.0,
    "GOOD": 0.75,
    "FAIR": 0.5,
    "POOR": 0.0,
}

response_quality_scorer = LLMClassifier(
    name="Response Quality",
    prompt_template=RESPONSE_QUALITY_PROMPT,
    choice_scores=RESPONSE_QUALITY_CHOICE_SCORES,
    use_cot=True,
    model=JUDGE_MODEL,
    base_url=GATEWAY_URL,
    api_key=API_KEY,
)


# ---------------------------------------------------------------------------
# Step Efficiency (code scorer)
# ---------------------------------------------------------------------------

MAX_STEPS = 8


async def step_efficiency_scorer(output):
    """Score based on the number of steps (messages/tool calls) taken.

    Returns 1.0 when the conversation stays within ``MAX_STEPS`` messages and
    linearly penalizes each extra step beyond that, floored at 0.0.
    """
    messages = output.get("messages", [])
    num_steps = len(messages)
    if num_steps <= MAX_STEPS:
        return 1.0
    return max(0.0, 1.0 - (num_steps - MAX_STEPS) / MAX_STEPS)
