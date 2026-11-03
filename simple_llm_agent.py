from __future__ import annotations

import argparse
import os
from abc import ABC, abstractmethod
from typing import List, Optional   

from config import EnvConfig
from foodtruck_env import FoodTruckEnv


class Agent(ABC):
    @abstractmethod
    def act(self, observation: str) -> str:
        raise NotImplementedError


class SimpleLLMAgent(Agent):
    def __init__(
        self,
    ) -> None:
        # TODO
        pass

    def act(self, observation: str) -> str:
        # TODO
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple LLM agent for FoodTruckEnv.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (defaults to the ANTHROPIC_API_KEY env var).",
    )
    parser.add_argument("--model", default="claude-haiku-5-5", help="Anthropic model name.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for env reset.")
    parser.add_argument("--max-steps", type=int, default=400, help="Safety cap on steps.")
    parser.add_argument(
        "--max-history-tokens",
        type=int,
        default=32000,
        help="Max tokens for transcript history.",
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate for the action.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("No API key found. Set ANTHROPIC_API_KEY in .env or pass --api-key.")

    env = FoodTruckEnv(EnvConfig())
    obs, info = env.reset(seed=args.seed)
    print(obs)

    agent = SimpleLLMAgent(
        # TODO
    )

    for _ in range(args.max_steps):
        action = agent.act(obs)

        print("\n---")
        print(f">>> {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        print(f"reward: {reward:.2f} | terminated: {terminated}")

        if terminated or truncated:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
