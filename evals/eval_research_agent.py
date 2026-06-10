"""
Research Agent evaluation - focused on web search and information retrieval.
"""

import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from braintrust import Eval, load_parameters  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage

from evals.parameters import (PROJECT_NAME,  # noqa: E402
                              RESEARCH_AGENT_EVAL_PARAMETERS_SLUG,
                              RESEARCH_AGENT_PROMPT_PARAM, parse_prompt_param)
from scorers.research import (answer_quality_scorer,  # noqa: E402
                              efficiency_scorer, source_attribution_scorer,
                              web_search_usage_scorer)
from src.agents.research_agent import get_research_agent  # noqa: E402
from src.config import DEFAULT_RESEARCH_MODEL  # noqa: E402

load_dotenv()


saved_parameters = load_parameters(
    project=PROJECT_NAME,
    slug=RESEARCH_AGENT_EVAL_PARAMETERS_SLUG,
)


def serialize_message(msg: Any) -> dict:
    """Convert a message to JSON-serializable dict."""
    if hasattr(msg, "content"):
        result = {
            "content": msg.content,
            "role": getattr(msg, "role", getattr(msg, "type", "unknown")),
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {"name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in msg.tool_calls
            ]
        return result
    return msg if isinstance(msg, dict) else {"content": str(msg)}


async def run_research_task(input: dict, hooks: Any = None) -> dict:
    """Run a research query through the research agent.

    Args:
        input_data: Dict with 'query' field containing the research question
        hooks: Optional Braintrust hooks for metadata tracking and parameters.
               When running remotely, hooks.parameters contains the configurable
               parameters defined in the Eval() constructor.

    Returns:
        Dict with 'messages' containing the conversation history
    """
    try:
        # Extract parameters if provided (when running remotely)
        params = hooks.parameters if hooks and hasattr(hooks, "parameters") else {}

        # Get parameter values from the shared saved parameters config
        research_agent_prompt = params.get(RESEARCH_AGENT_PROMPT_PARAM)
        research_model = None

        if research_agent_prompt is not None:
            research_agent_prompt, research_model = parse_prompt_param(
                research_agent_prompt
            )

        # Get research agent with custom parameters
        agent = get_research_agent(
            system_prompt=research_agent_prompt,
            model=research_model or DEFAULT_RESEARCH_MODEL,
        )

        # Run the agent
        result = await agent.ainvoke({"messages": [HumanMessage(content=input["query"])]})

        # Extract messages
        messages = result.get("messages", []) if isinstance(result, dict) else []
        serialized = [serialize_message(m) for m in messages]

        # Track metadata for scoring
        tool_calls = []
        for msg in serialized:
            if "tool_calls" in msg:
                tool_calls.extend([tc["name"] for tc in msg["tool_calls"]])

        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update(
                {
                    "tool_calls": tool_calls,
                    "used_web_search": "tavily_search_results_json" in tool_calls,
                    "total_messages": len(serialized),
                }
            )

        return {"messages": serialized}

    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"messages": [{"error": str(e)}]}


# Inline test dataset
RESEARCH_TEST_DATA = [
    {
        "input": {"query": "Who is the current president of France?"},
        "expected": {
            "should_use_search": True,
            "should_have_url": True,
        },
    },
    {
        "input": {"query": "What is the capital of Japan?"},
        "expected": {
            "should_use_search": True,
            "should_have_url": True,
        },
    },
    {
        "input": {"query": "When was the Eiffel Tower built?"},
        "expected": {
            "should_use_search": True,
            "should_have_url": True,
        },
    },
    {
        "input": {"query": "What are the main causes of climate change?"},
        "expected": {
            "should_use_search": True,
            "should_have_url": True,
        },
    },
]


# Evaluation
Eval(
    "agent-supervisor",
    experiment_name="research-agent",
    data=RESEARCH_TEST_DATA,  # type: ignore
    task=run_research_task,
    scores=[
        web_search_usage_scorer,
        source_attribution_scorer,
        efficiency_scorer,
        answer_quality_scorer,
    ],  # type: ignore
    parameters=saved_parameters,
)
