from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agents import HeuristicPolicy, LLMPolicy, OpenAICompatibleChatClient
from .runner import run_episode
from .world import AetherVaultWorld


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the virtual world agent harness.")
    parser.add_argument("--agent", choices=("heuristic", "llm"), default="heuristic")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--map-mode", choices=("known", "full", "none"), default="known")
    parser.add_argument("--show-observation", action="store_true")
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        help="Model name for --agent llm.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--llm-json-mode",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("LLM_JSON_MODE", "1") != "0",
        help="Request JSON object responses from the chat completions endpoint.",
    )
    args = parser.parse_args(argv)

    world = AetherVaultWorld()
    if args.agent == "heuristic":
        policy = HeuristicPolicy()
    else:
        client = OpenAICompatibleChatClient.from_env(
            model=args.model,
            api_base=args.api_base,
            json_mode=args.llm_json_mode,
        )
        policy = LLMPolicy(client)

    result = run_episode(
        world=world,
        policy=policy,
        max_steps=args.max_steps,
        map_mode=args.map_mode,
        show_observation=args.show_observation,
    )

    print(result.transcript, end="")
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text(result.transcript, encoding="utf-8")
    if args.replay_file:
        args.replay_file.parent.mkdir(parents=True, exist_ok=True)
        args.replay_file.write_text(json.dumps(result.replay, indent=2), encoding="utf-8")

    return 0 if result.success else 1
