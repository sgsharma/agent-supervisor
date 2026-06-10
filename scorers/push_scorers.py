"""Register every agent-supervisor scorer with Braintrust.

The scorer logic lives in the sibling modules (`supervisor`, `math`,
`research`); this file is the single place that publishes them. Push with:

    bt functions push scorers/push_scorers.py --if-exists replace -p agent-supervisor

Code scorers are registered with a `handler` + a pydantic `parameters` model
describing the eval-bound fields they read. LLM-as-a-judge scorers are
registered with `messages` + `choice_scores` so they run on Braintrust's model
proxy.
"""

import os
from typing import Any, Optional

import braintrust
from dotenv import load_dotenv
from pydantic import BaseModel

from scorers.math import (
    CALCULATION_CORRECTNESS_CHOICE_SCORES,
    CALCULATION_CORRECTNESS_PROMPT,
    JUDGE_MODEL,
    calculation_accuracy_scorer,
    response_format_scorer,
    tool_usage_scorer,
)
from scorers.math import efficiency_scorer as math_efficiency_scorer
from scorers.research import (
    ANSWER_QUALITY_CHOICE_SCORES,
    ANSWER_QUALITY_PROMPT,
    source_attribution_scorer,
    web_search_usage_scorer,
)
from scorers.research import efficiency_scorer as research_efficiency_scorer
from scorers.supervisor import (
    RESPONSE_QUALITY_CHOICE_SCORES,
    RESPONSE_QUALITY_PROMPT,
    routing_accuracy_scorer,
    step_efficiency_scorer,
)

load_dotenv()

PROJECT_NAME = os.getenv("BRAINTRUST_PROJECT", "agent-supervisor")
project = braintrust.projects.create(name=PROJECT_NAME)


# ---------------------------------------------------------------------------
# Parameter schemas for the code scorers (mirror each handler's eval-bound args)
# ---------------------------------------------------------------------------


class OutputOnly(BaseModel):
    output: Any


class OutputMetadata(BaseModel):
    output: Any
    metadata: Optional[dict] = None


class InputOutputExpected(BaseModel):
    input: Any = None
    output: Any
    expected: Any = None


class TraceScorerParams(BaseModel):
    input: Any = None
    output: Any
    expected: Any = None
    metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# Supervisor scorers
# ---------------------------------------------------------------------------

project.scorers.create(
    name="Routing Accuracy",
    slug="routing-accuracy",
    description="Trace-based judge of whether the supervisor routed the query to the correct agent(s).",
    parameters=TraceScorerParams,
    handler=routing_accuracy_scorer,
)

project.scorers.create(
    name="Response Quality",
    slug="response-quality",
    description="LLM judge scoring accuracy, completeness, clarity, and relevance of the response.",
    messages=[{"role": "user", "content": RESPONSE_QUALITY_PROMPT}],
    model=JUDGE_MODEL,
    use_cot=True,
    choice_scores=RESPONSE_QUALITY_CHOICE_SCORES,
)

project.scorers.create(
    name="Step Efficiency (Bundled)",
    slug="step-efficiency-bundled",
    description="Evaluates the number of steps taken to answer the question.",
    parameters=OutputOnly,
    handler=step_efficiency_scorer,
)


# ---------------------------------------------------------------------------
# Math agent scorers
# ---------------------------------------------------------------------------

project.scorers.create(
    name="Calculation Accuracy",
    slug="calculation-accuracy",
    description="Checks whether the expected numeric answer appears in the final response.",
    parameters=InputOutputExpected,
    handler=calculation_accuracy_scorer,
)

project.scorers.create(
    name="Tool Usage",
    slug="tool-usage",
    description="Checks whether the math agent used the add/subtract/multiply/divide tools.",
    parameters=OutputMetadata,
    handler=tool_usage_scorer,
)

project.scorers.create(
    name="Math Efficiency",
    slug="math-efficiency",
    description="Scores the math agent on minimal unnecessary tool calls.",
    parameters=OutputMetadata,
    handler=math_efficiency_scorer,
)

project.scorers.create(
    name="Response Format",
    slug="response-format",
    description="Checks the final math response contains a numeric answer.",
    parameters=OutputOnly,
    handler=response_format_scorer,
)

project.scorers.create(
    name="Calculation Correctness",
    slug="calculation-correctness",
    description="LLM judge of whether the math agent's calculation and answer are correct.",
    messages=[{"role": "user", "content": CALCULATION_CORRECTNESS_PROMPT}],
    model=JUDGE_MODEL,
    use_cot=True,
    choice_scores=CALCULATION_CORRECTNESS_CHOICE_SCORES,
)


# ---------------------------------------------------------------------------
# Research agent scorers
# ---------------------------------------------------------------------------

project.scorers.create(
    name="Web Search Usage",
    slug="web-search-usage",
    description="Checks whether the research agent used web search when appropriate.",
    parameters=OutputMetadata,
    handler=web_search_usage_scorer,
)

project.scorers.create(
    name="Source Attribution",
    slug="source-attribution",
    description="Checks whether the research response includes a URL citation.",
    parameters=OutputOnly,
    handler=source_attribution_scorer,
)

project.scorers.create(
    name="Research Efficiency",
    slug="research-efficiency",
    description="Scores the research agent on minimal search tool calls.",
    parameters=OutputMetadata,
    handler=research_efficiency_scorer,
)

project.scorers.create(
    name="Answer Quality",
    slug="answer-quality",
    description="LLM judge scoring accuracy, completeness, clarity, and relevance of a research answer.",
    messages=[{"role": "user", "content": ANSWER_QUALITY_PROMPT}],
    model=JUDGE_MODEL,
    use_cot=True,
    choice_scores=ANSWER_QUALITY_CHOICE_SCORES,
)
