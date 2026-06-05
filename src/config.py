"""Configuration for the deep agent supervisor and subagents."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

# Prompt files live in the top-level `prompts/` directory and are the single
# source of truth shared between the running agents and the Braintrust prompts
# (see prompts/push_prompts.py).
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Read a prompt file from the prompts/ directory."""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


# Default prompts and descriptions, loaded from prompts/.
# The supervisor prompt uses a `{{{current_date}}}` Mustache variable so the same
# file can be pushed to Braintrust verbatim; here we render it for local runs.
DEFAULT_SYSTEM_PROMPT = _load_prompt("supervisor.md").replace(
    "{{{current_date}}}", datetime.now().strftime("%Y-%m-%d")
)

DEFAULT_RESEARCH_AGENT_DESCRIPTION = "Research agent."

DEFAULT_MATH_AGENT_DESCRIPTION = "Math agent."

DEFAULT_RESEARCH_AGENT_PROMPT = _load_prompt("research_agent.md")

DEFAULT_MATH_AGENT_PROMPT = _load_prompt("math_agent.md")

# Default model names
DEFAULT_SUPERVISOR_MODEL = "gpt-4o-mini"
DEFAULT_RESEARCH_MODEL = "gpt-4o-mini"
DEFAULT_MATH_MODEL = "gpt-4o-mini"


class AgentConfig(BaseModel):
    """Configuration for the deep agent supervisor and subagents.

    All fields are optional with sensible defaults.
    """

    # Supervisor/System prompt
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Subagent prompts
    research_agent_prompt: str = DEFAULT_RESEARCH_AGENT_PROMPT
    math_agent_prompt: str = DEFAULT_MATH_AGENT_PROMPT

    # Subagent routing descriptions (used by SubAgentMiddleware)
    research_agent_description: str = DEFAULT_RESEARCH_AGENT_DESCRIPTION
    math_agent_description: str = DEFAULT_MATH_AGENT_DESCRIPTION

    # Model selections
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL
    research_model: str = DEFAULT_RESEARCH_MODEL
    math_model: str = DEFAULT_MATH_MODEL

    model_config = ConfigDict(arbitrary_types_allowed=True)
