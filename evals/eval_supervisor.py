"""
Simple evaluation script for the Agent Assistant.
Run this file to execute basic evaluations.
"""

import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional

# Ensure project root is on sys.path so `src` package can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from braintrust import Eval, init_dataset, load_parameters  # noqa: E402
from braintrust_langchain import BraintrustCallbackHandler  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evals.parameters import (  # noqa: E402
    MATH_AGENT_PROMPT_PARAM,
    PROJECT_NAME,
    RESEARCH_AGENT_PROMPT_PARAM,
    SUPERVISOR_EVAL_PARAMETERS_SLUG,
    SYSTEM_PROMPT_PARAM,
    parse_prompt_param,
)
from scorers.supervisor import (  # noqa: E402
    response_quality_scorer,
    routing_accuracy_scorer,
    step_efficiency_scorer,
)

# Import our supervisor system
from src.agents.deep_agent import get_supervisor  # noqa: E402
from src.config import AgentConfig  # noqa: E402

load_dotenv()


def unwrap_parameters(params: dict) -> dict:
    """Convert Braintrust prompt parameters into AgentConfig fields."""

    result = {}
    for key, param in params.items():
        if param is None:
            continue

        prompt_text, model = parse_prompt_param(param)

        if key == SYSTEM_PROMPT_PARAM:
            result["system_prompt"] = prompt_text
            if model is not None:
                result["supervisor_model"] = model
        elif key == RESEARCH_AGENT_PROMPT_PARAM:
            result["research_agent_prompt"] = prompt_text
            if model is not None:
                result["research_model"] = model
        elif key == MATH_AGENT_PROMPT_PARAM:
            result["math_agent_prompt"] = prompt_text
            if model is not None:
                result["math_model"] = model
    return result


def serialize_message(msg: Any) -> dict:
    """Convert a LangChain message object to a JSON-serializable dict.

    Args:
        msg: LangChain message object (AIMessage, HumanMessage, etc.)

    Returns:
        Dict with message content and metadata
    """
    # Handle different message types
    if hasattr(msg, "content"):
        result = {
            "content": msg.content,
            "role": getattr(msg, "role", getattr(msg, "type", "unknown")),
        }

        # Add tool calls if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                }
                for tc in msg.tool_calls
            ]

        # Add additional response metadata if present
        if hasattr(msg, "response_metadata") and msg.response_metadata:
            result["response_metadata"] = msg.response_metadata

        return result
    else:
        # Fallback for dict-like objects
        return msg if isinstance(msg, dict) else {"content": str(msg)}


async def run_supervisor_task(input: dict, hooks: Any = None) -> dict[str, list]:
    """Run a single task through the supervisor and return the final response.

    Args:
        input_data: Input data containing messages
        hooks: Optional Braintrust hooks for metadata tracking and parameters.
               When running remotely, hooks.parameters contains the configurable
               parameters defined in the Eval() constructor.

    Returns:
        Dict containing messages from the supervisor execution
    """
    try:
        # Build AgentConfig from parameters (if provided)
        # When running locally: hooks is None, params is empty dict
        # When running remotely: hooks.parameters contains the config values
        params = hooks.parameters if hooks and hasattr(hooks, "parameters") else {}

        config_params = unwrap_parameters(params)
        config = AgentConfig(**config_params) if config_params else None

        supervisor = get_supervisor(config, force_rebuild=True)

        # Use hooks.span as the parent so LangChain spans nest under the eval trace
        span = hooks.span if hooks and hasattr(hooks, "span") and hooks.span else None
        callback = BraintrustCallbackHandler(logger=span)
        result = await supervisor.ainvoke(
            {"messages": input["messages"]},
            config={"callbacks": [callback]},
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []

        # Serialize messages to JSON-serializable format
        serialized_messages = [serialize_message(m) for m in messages]
        return {"messages": serialized_messages}

    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"messages": [{"error": str(e)}]}


saved_parameters = load_parameters(
    project=PROJECT_NAME,
    slug=SUPERVISOR_EVAL_PARAMETERS_SLUG,
)


def get_dataset(
    dataset_name: str = "Tool Routing Correctness",
    tag: Optional[str] = None,
):
    """Load a dataset, optionally filtered by tag via EVAL_TAG env var."""
    dataset_name = os.getenv("EVAL_DATASET", dataset_name)
    tag = os.getenv("EVAL_TAG", tag)

    kwargs: dict[str, Any] = {"project": "agent-supervisor", "name": dataset_name}
    if tag:
        kwargs["_internal_btql"] = {"filter": {"btql": f"tags INCLUDES '{tag}'"}}
    return init_dataset(**kwargs)


# Basic evaluation
Eval(
    "agent-supervisor",
    data=get_dataset(),
    task=run_supervisor_task,
    scores=[
        response_quality_scorer,
        routing_accuracy_scorer,
        step_efficiency_scorer,
    ],  # type: ignore
    parameters=saved_parameters,
)
