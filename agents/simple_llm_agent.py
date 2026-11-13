from __future__ import annotations

import argparse
import os
from typing import List, Optional

import anthropic
import pydantic
from pydantic import BaseModel

from agents.base import (
    FALLBACK_ACTION,
    MAX_RESPONSE_TOKENS,
    Agent,
    clean_actions,
    describe_game,
    get_day,
    run_episode,
)
from foodtruck import EnvConfig, FoodTruckEnv


class DayPlan(BaseModel):
    actions: List[str]


def build_system_prompt(config: EnvConfig) -> str:
    return f"""You run a food truck. {describe_game(config)}

How to respond:
- At the start of each day, you should probably check storage, market, and menu first. Recipes never change, so check them only on day 1. After that they are shown above the transcript. Respond with only those check commands.
- After you see the check results, respond with all your actions for the rest of the day in one list, ending with `end prep`.
- Respond with a JSON object like {{"actions": ["check storage", "check market"]}}. Each action is one command."""


def recipes_section(observation: str) -> Optional[str]:
    """The 'Recipes:' section of an observation, or None if it has none."""
    lines = observation.splitlines()
    if "Recipes:" not in lines:
        return None
    start = lines.index("Recipes:")
    section = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith("- "):
            break
        section.append(line)
    return "\n".join(section)


class SimpleLLMAgent(Agent):
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        config: EnvConfig,
        max_history_tokens: int,
    ) -> None:
        self.client = client
        self.model = model
        self.config = config
        self.system_prompt = build_system_prompt(config)
        self.max_history_tokens = max_history_tokens

        self.transcript: List[str] = []  # alternating observation and action entries, oldest first
        self.recipes_text: Optional[str] = None  # recipes from the last recipe check
        self.queue: List[str] = []  # planned actions not yet sent to the env
        self.queue_day: Optional[int] = None  # the day the queued actions were planned for

    def act(self, observation: str) -> str:
        self.transcript.append(f"Observation:\n{observation}")
        self.recipes_text = recipes_section(observation) or self.recipes_text

        # If a new day started (e.g. prep ops ran out), the old plan no longer applies.
        day = get_day(observation)
        if day != self.queue_day:
            self.queue = []

        if not self.queue:
            self.queue = self._plan()
            self.queue_day = day

        action = self.queue.pop(0)
        self.transcript.append(f"Action: {action}")
        return action

    def _plan(self) -> List[str]:
        """Ask the LLM for the next list of actions."""
        self._truncate_transcript()
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=MAX_RESPONSE_TOKENS,
                system=self.system_prompt,
                messages=self._messages(),
                output_format=DayPlan,
            )
        except pydantic.ValidationError as e:
            # Happens if the JSON is cut off, e.g. by max_tokens.
            print(f"[agent] could not parse LLM response: {e}")
            return [FALLBACK_ACTION]

        plan = response.parsed_output
        if plan is None:
            print(f"[agent] no plan in LLM response (stop_reason: {response.stop_reason})")
            return [FALLBACK_ACTION]

        return clean_actions(plan.actions, self.config) or [FALLBACK_ACTION]

    def _messages(self) -> List[dict]:
        transcript = "\n\n".join(self.transcript)
        content = f"Transcript so far (oldest first):\n\n{transcript}\n\nWhat are your next actions?"
        if self.recipes_text:
            content = f"{self.recipes_text}\n\n{content}"
        return [{"role": "user", "content": content}]

    def _count_tokens(self) -> int:
        return self.client.messages.count_tokens(
            model=self.model,
            system=self.system_prompt,
            messages=self._messages(),
        ).input_tokens

    def _truncate_transcript(self) -> None:
        """Drop the oldest whole entries until the prompt fits in max_history_tokens."""
        while len(self.transcript) > 1:
            tokens = self._count_tokens()
            if tokens <= self.max_history_tokens:
                return

            # Estimate how many characters to drop from the share of tokens over budget,
            # so we only need to re-count once or twice instead of after every entry.
            total_chars = sum(len(entry) for entry in self.transcript)
            chars_to_drop = total_chars * (1 - self.max_history_tokens / tokens)
            dropped = 0
            while len(self.transcript) > 1 and dropped < chars_to_drop:
                dropped += len(self.transcript.pop(0))

            # Start on an observation so no action is left without its context.
            while len(self.transcript) > 1 and self.transcript[0].startswith("Action:"):
                self.transcript.pop(0)


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
    args = parser.parse_args()
    if not args.api_key:
        parser.error("No API key found. Set ANTHROPIC_API_KEY in .env or pass --api-key.")

    config = EnvConfig()
    agent = SimpleLLMAgent(
        client=anthropic.Anthropic(api_key=args.api_key),
        model=args.model,
        config=config,
        max_history_tokens=args.max_history_tokens,
    )
    run_episode(FoodTruckEnv(config), agent, seed=args.seed, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
