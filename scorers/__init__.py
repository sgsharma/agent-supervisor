"""Scorers for the agent-supervisor evals.

Scorer logic lives in the per-agent modules (`supervisor`, `math`, `research`).
`push_scorers` registers every scorer with Braintrust via
`project.scorers.create`. Eval files import the scorers from here (or from the
submodules directly).
"""

from scorers.math import (
    calculation_accuracy_scorer,
    calculation_correctness_scorer,
    response_format_scorer,
    tool_usage_scorer,
)
from scorers.math import efficiency_scorer as math_efficiency_scorer
from scorers.research import (
    answer_quality_scorer,
    source_attribution_scorer,
    web_search_usage_scorer,
)
from scorers.research import efficiency_scorer as research_efficiency_scorer
from scorers.supervisor import (
    response_quality_scorer,
    routing_accuracy_scorer,
    step_efficiency_scorer,
)

__all__ = [
    # supervisor
    "routing_accuracy_scorer",
    "response_quality_scorer",
    "step_efficiency_scorer",
    # math
    "calculation_accuracy_scorer",
    "tool_usage_scorer",
    "math_efficiency_scorer",
    "response_format_scorer",
    "calculation_correctness_scorer",
    # research
    "web_search_usage_scorer",
    "source_attribution_scorer",
    "research_efficiency_scorer",
    "answer_quality_scorer",
]
