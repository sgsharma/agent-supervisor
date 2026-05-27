import os
from typing import Literal

import braintrust
from pydantic import BaseModel

PROJECT_NAME = os.getenv("BRAINTRUST_PROJECT", "agent-supervisor")
class StepEfficiencyScorer(BaseModel):
    output: list[dict]


async def step_efficiency_scorer(output):
    MAX_STEPS = 8
    messages = output.get("messages", [])
    num_steps = len(messages)
    if num_steps <= MAX_STEPS:
        return 1.0

    return max(0.0, 1.0 - (num_steps - MAX_STEPS) / MAX_STEPS)


project = braintrust.projects.create(name=PROJECT_NAME)

project.scorers.create(
    name="Step Efficiency (Bundled)",
    slug="step-efficiency-bundled",
    description="Evaluates the number of steps taken to answer the question.",
    parameters=StepEfficiencyScorer,
    handler=step_efficiency_scorer,
)


# Routing Accuracy Scorer - Trace-level scorer for routing evaluation


class RoutingAccuracyParameters(BaseModel):
    """Parameters for routing accuracy trace scorer."""
    pass


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


async def routing_accuracy_scorer(input, output, expected, metadata, trace):
    """Trace-level scorer for routing accuracy evaluation."""
    from braintrust.oai import wrap_openai
    from openai import AsyncOpenAI
    
    choice_map = {
        "A": 1.0,
        "B": 0.7,
        "C": 0.3,
        "D": 0.0,
    }
    
    # Extract agents called from trace spans
    spans = await trace.get_spans(span_type=["task"])
    agents_called_str = "None (supervisor answered directly)"
    agents_called = []
    
    for span in spans:
        span_name = span.span_attributes.get("name", None)
        if span_name in ["MathAgent", "ResearchAgent"]:
            agents_called.append(span_name)
    
    if agents_called:
        agents_called_str = ", ".join(agents_called)
    
    # Extract user input from trace
    try:
        input_str = input.get("messages", [{}])[0].get("content", str(input))
    except (IndexError, AttributeError, TypeError):
        input_str = str(input)
    
    prompt = ROUTING_ACCURACY_PROMPT.format(
        input=input_str, agents_called=agents_called_str
    )
    
    # Create wrapped OpenAI client for LLM evaluation
    api_key = os.getenv("BRAINTRUST_API_KEY")
    base_url = os.getenv("BRAINTRUST_GATEWAY_URL", "https://api.braintrust.dev/v1")
    
    client = wrap_openai(
        AsyncOpenAI(api_key=api_key, base_url=base_url)
    )
    
    response = await client.responses.parse(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        text_format=RoutingAccuracyOutput,
    )
    output = response.output_parsed
    
    return {
        "name": "Routing Accuracy",
        "score": choice_map.get(output.choice, 0.0) if output else 0.0,
        "metadata": {
            "agents_called": agents_called_str,
            "reasoning": output.reasoning if output else "No output",
            "choice": output.choice if output else "D",
        },
    }


project.scorers.create(
    name="Routing Accuracy",
    slug="routing-accuracy",
    description="Evaluates whether a user question was correctly routed to appropriate agents using LLM-as-judge.",
    parameters=RoutingAccuracyParameters,
    handler=routing_accuracy_scorer,
)
