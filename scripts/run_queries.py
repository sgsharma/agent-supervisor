#!/usr/bin/env python3
"""Replay scripted multi-turn conversations through the supervisor.

Generates Braintrust traces from a curated list of 1-3 turn conversations
designed to exercise routing to a single subagent (math-only or research-only)
and to both subagents (hybrid). Each turn produces its own root trace.
"""
import argparse
import asyncio
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, TypedDict

from braintrust import init_logger
from braintrust_langchain import BraintrustCallbackHandler, set_global_handler
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage  # type: ignore

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import AgentConfig  # noqa: E402

load_dotenv()

PROJECT_NAME = os.getenv("BRAINTRUST_PROJECT", "agent-supervisor")
MODEL_POOL = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]


class Conversation(TypedDict):
    turns: List[str]
    tags: List[str]


# 55 scripted conversations. Tags describe the expected routing:
#   "math" / "research"      → single-subagent
#   "math" + "research"      → hybrid (both subagents)
# Turn counts vary 1-3.
SCRIPTED_CONVERSATIONS: List[Conversation] = [
    # ─── math only, 1 turn (5) ───
    {"turns": ["What is 145 * 32?"], "tags": ["math", "single-turn"]},
    {"turns": ["Calculate 5678 divided by 23"], "tags": ["math", "single-turn"]},
    {"turns": ["What's 15% of 240?"], "tags": ["math", "single-turn"]},
    {"turns": ["Add 1234 and 5678"], "tags": ["math", "single-turn"]},
    {"turns": ["whats 99 squared"], "tags": ["math", "single-turn", "casual"]},

    # ─── math only, 2 turns (5) ───
    {"turns": ["What is 15 * 24?", "Now divide that by 3"], "tags": ["math", "multi-turn"]},
    {"turns": ["Add 234 + 567", "Multiply the result by 4"], "tags": ["math", "multi-turn"]},
    {"turns": ["What's 50% of 800?", "Subtract 100 from that"], "tags": ["math", "multi-turn"]},
    {"turns": ["Calculate 17 * 23", "What's the square of that number?"], "tags": ["math", "multi-turn"]},
    {"turns": ["What is 1000 divided by 7?", "Round it and add 50"], "tags": ["math", "multi-turn"]},

    # ─── math only, 3 turns (5) ───
    {"turns": [
        "I need to figure out a weekly budget. Start with 5000.",
        "Subtract 1200 for rent.",
        "Now divide the remainder by 4 for weekly groceries.",
    ], "tags": ["math", "multi-turn"]},
    {"turns": [
        "What is 12 * 12?",
        "And 13 * 13?",
        "What's the difference between those two?",
    ], "tags": ["math", "multi-turn"]},
    {"turns": [
        "Add 50 and 75",
        "Now multiply by 3",
        "Divide by 5",
    ], "tags": ["math", "multi-turn"]},
    {"turns": [
        "What is 1000 minus 245?",
        "Now divide that by 5",
        "Multiply the result by 2",
    ], "tags": ["math", "multi-turn"]},
    {"turns": [
        "Calculate 17 * 4",
        "Add 100",
        "Divide by 6",
    ], "tags": ["math", "multi-turn"]},

    # ─── research only, 1 turn (5) ───
    {"turns": ["Who is the current CEO of OpenAI?"], "tags": ["research", "single-turn"]},
    {"turns": ["What year did World War II end?"], "tags": ["research", "single-turn"]},
    {"turns": ["Tell me about the Eiffel Tower"], "tags": ["research", "single-turn"]},
    {"turns": ["What's the capital of Australia?"], "tags": ["research", "single-turn"]},
    {"turns": ["Who invented the lightbulb?"], "tags": ["research", "single-turn"]},

    # ─── research only, 2 turns (5) ───
    {"turns": ["Who founded Microsoft?", "When was it founded?"], "tags": ["research", "multi-turn"]},
    {"turns": ["What is the largest ocean?", "What's the deepest point in it called?"], "tags": ["research", "multi-turn"]},
    {"turns": ["Tell me about Albert Einstein", "What was his most famous equation?"], "tags": ["research", "multi-turn"]},
    {"turns": ["What is the tallest mountain in the world?", "What country is it in?"], "tags": ["research", "multi-turn"]},
    {"turns": ["Who wrote Hamlet?", "What other plays did they write?"], "tags": ["research", "multi-turn"]},

    # ─── research only, 3 turns (5) ───
    {"turns": [
        "Tell me about the Roman Empire",
        "When did the Western Roman Empire fall?",
        "Who was the last Western Roman emperor?",
    ], "tags": ["research", "multi-turn"]},
    {"turns": [
        "What is Tokyo known for?",
        "What's the population of greater Tokyo?",
        "What languages are commonly spoken there?",
    ], "tags": ["research", "multi-turn"]},
    {"turns": [
        "Who is Elon Musk?",
        "What companies does he run?",
        "Which one is the largest by revenue?",
    ], "tags": ["research", "multi-turn"]},
    {"turns": [
        "Tell me about black holes",
        "How are they formed?",
        "What's the closest known one to Earth?",
    ], "tags": ["research", "multi-turn"]},
    {"turns": [
        "What is climate change?",
        "What are the main causes?",
        "Which countries emit the most CO2 annually?",
    ], "tags": ["research", "multi-turn"]},

    # ─── hybrid, 1 turn (5) ───
    {"turns": ["What's the distance from NYC to LA, and how long would it take to drive at 65 mph?"], "tags": ["math", "research", "hybrid", "single-turn"]},
    {"turns": ["What was Apple's revenue in 2023, and what would a 20% increase be?"], "tags": ["math", "research", "hybrid", "single-turn"]},
    {"turns": ["How tall is Mount Everest in meters, and what's that in feet (multiply by 3.28)?"], "tags": ["math", "research", "hybrid", "single-turn"]},
    {"turns": ["What's the population of Canada, and what's 5% of that?"], "tags": ["math", "research", "hybrid", "single-turn"]},
    {"turns": ["What's the speed of light in m/s, and how far does it travel in 10 seconds?"], "tags": ["math", "research", "hybrid", "single-turn"]},

    # ─── hybrid, 2 turns (10) ───
    {"turns": ["What's the population of France?", "What's 12% of that?"], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": ["How tall is the Burj Khalifa in meters?", "Multiply that by 3.28 to get feet"], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": ["What year was the Eiffel Tower built?", "How many years ago is that from 2026?"], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": ["What's the approximate GDP of Japan in trillions of USD?", "What would a 5% increase look like?"], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": ["What's the speed of sound at sea level in m/s?", "How many seconds for sound to travel 1700 meters?"], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": ["When was Einstein born?", "How old would he be in 2026?"], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": ["What's the average resting human heart rate (beats per minute)?", "How many beats per day at that rate?"], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": ["What's the diameter of Earth in km?", "Calculate the circumference (multiply by 3.14159)"], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": ["When did the original iPhone come out?", "How many years has it been from 2026?"], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": ["What's the population of New York City?", "What's 15% of that?"], "tags": ["math", "research", "hybrid", "multi-turn"]},

    # ─── hybrid, 3 turns (5) ───
    {"turns": [
        "What's the population of California?",
        "What about Texas?",
        "What's the difference between the two?",
    ], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": [
        "When was Amazon founded?",
        "How many years has it been since, as of 2026?",
        "Multiply that number by 12 to get months",
    ], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": [
        "What's the height of Mount Kilimanjaro in meters?",
        "What about Mount Everest in meters?",
        "What's the difference in feet (multiply by 3.28)?",
    ], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": [
        "What's the boiling point of water in Celsius?",
        "And the freezing point?",
        "What's the difference between them?",
    ], "tags": ["math", "research", "hybrid", "multi-turn"]},
    {"turns": [
        "What is the speed of light in km/s?",
        "How far is the moon from Earth in km?",
        "How many seconds does light take to reach Earth from the moon?",
    ], "tags": ["math", "research", "hybrid", "multi-turn"]},

    # ─── edge / casual / frustrated (5) ───
    {"turns": ["just tell me 99*99 already"], "tags": ["math", "single-turn", "frustrated"]},
    {"turns": ["whats the cube root of 27", "are u sure thats right?"], "tags": ["math", "multi-turn", "frustrated"]},
    {"turns": ["btw whats the cap of france"], "tags": ["research", "single-turn", "casual"]},
    {"turns": ["hello"], "tags": ["edge-case", "single-turn"]},
    {"turns": ["help me understand percentages", "ok now what is 23% of 540"], "tags": ["math", "multi-turn", "edge-case"]},

    # ─── routing-ambiguous (20) ───
    # Group 1: pure math, but phrased like a factual lookup → may over-route to Research
    {"turns": ["What is the cube root of 27?"], "tags": ["math", "single-turn", "routing-ambiguous", "math-as-fact"]},
    {"turns": ["What's 2 to the 100th power?"], "tags": ["math", "single-turn", "routing-ambiguous", "math-as-fact"]},
    {"turns": ["How many seconds are in a non-leap year?"], "tags": ["math", "single-turn", "routing-ambiguous", "math-as-fact"]},
    {"turns": ["What's the square root of 169?"], "tags": ["math", "single-turn", "routing-ambiguous", "math-as-fact"]},
    {"turns": ["What's 17 factorial?"], "tags": ["math", "single-turn", "routing-ambiguous", "math-as-fact"]},

    # Group 2: pure research / well-known constants, but numeric → may over-route to Math
    {"turns": ["What's the value of pi?"], "tags": ["research", "single-turn", "routing-ambiguous", "fact-as-math"]},
    {"turns": ["What is Avogadro's number?"], "tags": ["research", "single-turn", "routing-ambiguous", "fact-as-math"]},
    {"turns": ["What's the speed of light in m/s?"], "tags": ["research", "single-turn", "routing-ambiguous", "fact-as-math"]},
    {"turns": ["How many bones are in the human body?"], "tags": ["research", "single-turn", "routing-ambiguous", "fact-as-math"]},
    {"turns": ["How many planets are in our solar system?"], "tags": ["research", "single-turn", "routing-ambiguous", "fact-as-math"]},

    # Group 3: statistical-sounding (average/median/percentage), but answer is a published fact
    {"turns": ["What's the average human lifespan?"], "tags": ["research", "single-turn", "routing-ambiguous", "stats-keyword"]},
    {"turns": ["What's the median household income in the US?"], "tags": ["research", "single-turn", "routing-ambiguous", "stats-keyword"]},
    {"turns": ["What percentage of Earth's surface is water?"], "tags": ["research", "single-turn", "routing-ambiguous", "stats-keyword"]},
    {"turns": ["What's the average commute time in Tokyo?"], "tags": ["research", "single-turn", "routing-ambiguous", "stats-keyword"]},

    # Group 4: formula/equation phrasing, but user actually wants a calculation
    {"turns": ["Use the Pythagorean theorem to find the hypotenuse for legs 3 and 4."], "tags": ["math", "single-turn", "routing-ambiguous", "formula-phrasing"]},
    {"turns": ["What's the slope of the line y = 7x - 2?"], "tags": ["math", "single-turn", "routing-ambiguous", "formula-phrasing"]},
    {"turns": ["Convert 100°F to Celsius using the standard formula."], "tags": ["math", "single-turn", "routing-ambiguous", "formula-phrasing"]},

    # Group 5: genuinely hybrid but only one subagent looks needed at first glance
    {"turns": ["How many feet tall is the Eiffel Tower?"], "tags": ["math", "research", "hybrid", "single-turn", "routing-ambiguous", "hybrid-disguised"]},
    {"turns": ["What is 15% of California's population?"], "tags": ["math", "research", "hybrid", "single-turn", "routing-ambiguous", "hybrid-disguised"]},
    {"turns": ["How many marathons fit in the distance from NYC to Boston?"], "tags": ["math", "research", "hybrid", "single-turn", "routing-ambiguous", "hybrid-disguised"]},
]


async def run_conversation(conv: Conversation, idx: int) -> bool:
    """Run a single multi-turn conversation. Each turn is a separate ainvoke (trace)."""
    try:
        from src.agent_graph import get_supervisor  # noqa: E402

        model = random.choice(MODEL_POOL)
        cfg = AgentConfig(
            supervisor_model=model,
            research_model=model,
            math_model=model,
        )
        supervisor = get_supervisor(cfg)
        customer_id = f"customer_{random.randint(1000, 9999)}"

        print(f"[{idx:03d}] model={model:14s} turns={len(conv['turns'])} tags={','.join(conv['tags'])}")

        history: List = []
        for turn_idx, user_text in enumerate(conv["turns"]):
            history.append(HumanMessage(content=user_text))
            result = await supervisor.ainvoke(
                {"messages": history},
                metadata={
                    "customer_id": customer_id,
                    "conversation_idx": idx,
                    "turn_index": turn_idx,
                    "total_turns": len(conv["turns"]),
                    "conversation_model": model,
                    "conversation_tags": conv["tags"],
                },
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if not messages:
                print(f"  [{idx:03d}.{turn_idx}] no messages returned; stopping conversation")
                break
            history = messages

        return True
    except Exception as e:
        print(f"[{idx:03d}] ❌ {type(e).__name__}: {str(e)[:160]}")
        return False


async def main_async(args: argparse.Namespace) -> None:
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("Missing BRAINTRUST_API_KEY in environment", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)
    scripts = list(SCRIPTED_CONVERSATIONS)
    rng.shuffle(scripts)

    # Cycle through the script pool until we reach --count.
    conversations: List[Conversation] = []
    while len(conversations) < args.count:
        conversations.extend(scripts)
    conversations = conversations[: args.count]

    print(f"{'=' * 80}")
    print(f"Project: {PROJECT_NAME}")
    print(f"Conversations: {len(conversations)} (from {len(scripts)} unique scripts)")
    print(f"Concurrency: {args.concurrency}")
    print(f"Model pool: {', '.join(MODEL_POOL)}")
    print(f"{'=' * 80}\n")

    successes = 0
    failures = 0

    for batch_start in range(0, len(conversations), args.concurrency):
        batch = conversations[batch_start : batch_start + args.concurrency]
        tasks = [
            run_conversation(conv, batch_start + i) for i, conv in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if r is True:
                successes += 1
            else:
                failures += 1
        print()

    print(f"{'=' * 80}")
    print(f"Completed. successes={successes} failures={failures}")
    print(f"{'=' * 80}\n")

    if args.fail_on_error and failures > 0:
        sys.exit(1)


def main(logger=None) -> None:
    parser = argparse.ArgumentParser(
        description="Replay scripted multi-turn conversations through the supervisor"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=int(os.environ.get("COUNT", "100")),
        help="Total number of conversations to run (default: 100). Scripts cycle if count > pool size.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "3")),
        help="Number of conversations to run in parallel (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any conversation fails",
    )
    args = parser.parse_args()

    # Initialize tracing — set global handler BEFORE creating agents.
    if logger is None:
        logger = init_logger(
            project=PROJECT_NAME, api_key=os.environ.get("BRAINTRUST_API_KEY")
        )
    set_global_handler(BraintrustCallbackHandler(logger=logger))

    try:
        asyncio.run(main_async(args))
    finally:
        print("Flushing traces to Braintrust...")
        logger.flush()
        print("Done.")


if __name__ == "__main__":
    main()
