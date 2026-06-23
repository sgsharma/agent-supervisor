#!/usr/bin/env python3
"""Run a set of image-upload demo queries that produce Braintrust span attachments.

Each trace includes a multimodal user message whose content contains an inline
base64-encoded PNG. Braintrust automatically converts inline base64 image data
into span attachments, which are visible in the Braintrust UI under the trace's
Attachments tab.
"""
import asyncio
import base64
import os
import sys
from pathlib import Path
from typing import List

import braintrust
from braintrust_langchain import BraintrustCallbackHandler
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_graph import get_supervisor  # noqa: E402

load_dotenv()

PROJECT_NAME = os.getenv("BRAINTRUST_PROJECT", "agent-supervisor")
IMAGES_DIR = project_root / "data" / "images"


def _data_url(path: Path) -> str:
    data = path.read_bytes()
    ext = path.suffix.lstrip(".")
    mime = "jpeg" if ext == "jpg" else ext
    return f"data:image/{mime};base64," + base64.b64encode(data).decode()


IMAGE_QUERIES = [
    {
        "label": "chart_analysis",
        "image": "chart_analysis.png",
        "text": (
            "I've uploaded a bar chart showing quarterly sales figures for 2025. "
            "Can you research what factors typically drive quarterly sales variation "
            "in the retail industry?"
        ),
        "tags": ["image-upload", "chart_analysis", "research"],
    },
    {
        "label": "product_photo",
        "image": "product_photo.png",
        "text": (
            "Here's a photo of a consumer electronics product. "
            "What are the current market trends for consumer electronics in 2025?"
        ),
        "tags": ["image-upload", "product_photo", "research"],
    },
    {
        "label": "math_diagram",
        "image": "math_diagram.png",
        "text": (
            "This diagram shows a right triangle with legs measuring 3 cm and 4 cm. "
            "What is the area of the triangle?"
        ),
        "tags": ["image-upload", "math_diagram", "math"],
    },
    {
        "label": "infographic",
        "image": "infographic.png",
        "text": (
            "I've attached an infographic about global carbon emissions. "
            "Which countries are currently the top CO2 emitters?"
        ),
        "tags": ["image-upload", "infographic", "research"],
    },
    {
        "label": "data_table",
        "image": "data_table.png",
        "text": (
            "This image shows a data table with employee counts: "
            "Q1=120, Q2=135, Q3=150, Q4=165. "
            "What is the total headcount across all four quarters?"
        ),
        "tags": ["image-upload", "data_table", "math"],
    },
]


async def run_image_query(
    supervisor, logger, query: dict, idx: int
) -> bool:
    image_path = IMAGES_DIR / query["image"]
    data_url = _data_url(image_path)

    # Multimodal message: image first, then the text question.
    # The inline base64 data URL triggers automatic span attachment creation in Braintrust.
    user_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": query["text"]},
        ],
    }

    print(f"[{idx:03d}] label={query['label']} tags={','.join(query['tags'])}")

    try:
        with logger.start_span(
            name=f"image_query:{query['label']}",
            input={"messages": [user_message]},
            tags=query["tags"],
        ) as span:
            callback = BraintrustCallbackHandler(logger=span)
            # Invoke the agent with plain text only — the base64 image lives in the
            # span input above and doesn't need to be passed through the LLM.
            result = await supervisor.ainvoke(
                {"messages": [("user", query["text"])]},
                config={"callbacks": [callback]},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            output = messages[-1].content if messages else ""
            span.log(output=output)

        return True
    except Exception as e:
        print(f"[{idx:03d}] ❌ {type(e).__name__}: {str(e)[:160]}")
        return False


async def main_async(queries: List[dict]) -> None:
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("Missing BRAINTRUST_API_KEY in environment", file=sys.stderr)
        sys.exit(2)

    logger = braintrust.init_logger(
        project=PROJECT_NAME, api_key=os.environ.get("BRAINTRUST_API_KEY")
    )
    supervisor = get_supervisor()

    print(f"{'=' * 70}")
    print(f"Project: {PROJECT_NAME}")
    print(f"Image queries: {len(queries)}")
    print(f"{'=' * 70}\n")

    tasks = [run_image_query(supervisor, logger, q, i) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = sum(1 for r in results if r is True)
    failures = len(results) - successes

    print(f"\n{'=' * 70}")
    print(f"Completed. successes={successes} failures={failures}")
    print(f"{'=' * 70}")

    print("Flushing traces to Braintrust...")
    logger.flush()
    print("Done.")

    if failures > 0:
        sys.exit(1)


def main() -> None:
    asyncio.run(main_async(IMAGE_QUERIES))


if __name__ == "__main__":
    main()
