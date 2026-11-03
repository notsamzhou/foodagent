from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional

import anthropic
import pydantic
from pydantic import BaseModel

from agents.base import Agent
from foodtruck import EnvConfig, FoodTruckEnv


# Used when the LLM gives us nothing usable. It's harmless and gives the LLM fresh info next time.
FALLBACK_ACTION = "check storage"

# One pattern per command the environment accepts (see README "Action commands").
ACTION_PATTERNS = [
    r"check (storage|market|recipes|menu)",
    r"buy \S+ \d+",
    r"trash \S+ \d+",
    r"set menu \d+ \S+ \d+(\.\d+)?",
    r"clear menu \d+",
    r"end prep",
]


class DayPlan(BaseModel):
    actions: List[str]


def build_system_prompt(config: EnvConfig) -> str:
    # Only rules the agent is allowed to know. Customer demand, price dynamics,
    # and other hidden parameters are left out on purpose.
    return f"""You run a food truck. Your goal is to have as much money as possible at the end of day {config.horizon_days}.

How a day works:
- Each day starts with a prep stage where you can check information, buy or trash ingredients, and set the menu.
- Every command uses one prep op, including checks and invalid commands. You have {config.max_prep_ops_per_day} prep ops per day.
- Prep ends when you send `end prep` or run out of prep ops. Then the truck opens and customers order from your menu.
- Orders are cooked from the ingredients in storage. If you don't have the ingredients, the order goes unfilled.
- Leftover ingredients stay in storage for later days.
- After the day you earn revenue for filled orders, pay ${config.daily_rent:.2f} rent, and see a daily summary.
- You don't know in advance how many customers will order each dish. Learn from the daily summaries.

Commands:
- check storage
- check market
- check recipes
- check menu
- buy <ingredient> <qty>
- trash <ingredient> <qty>
- set menu <slot> <recipe> <price>
- clear menu <slot>
- end prep

Rules:
- <ingredient> and <recipe> can be a name or a 1-based index.
- Menu slots are 1 to {config.max_menu_items}. A recipe can only be in one slot. The menu stays the same between days until you change it.
- Menu prices must be between ${config.price_min:.2f} and ${config.price_max:.2f}, in steps of ${config.price_step:.2f}.
- You can buy at most {config.max_buy_qty} and trash at most {config.max_trash_qty} of an ingredient per command.
- The fridge holds {config.fridge_capacity} items in total and the pantry holds {config.pantry_capacity}.
- Market prices change every day.
- The result of a check only appears in the observation right after that check.

How to respond:
- At the start of each day, you should probably check everything first (storage, market, recipes, menu). Respond with only those check commands.
- After you see the check results, respond with all your actions for the rest of the day in one list, ending with `end prep`.
- Respond with a JSON object like {{"actions": ["check storage", "check market"]}}. Each action is one command."""


def clean_actions(actions: List[str]) -> List[str]:
    """Normalize the LLM's actions and drop any that don't match a command pattern."""
    valid = []
    for action in actions:
        action = " ".join(action.strip().lower().split())
        if any(re.fullmatch(pattern, action) for pattern in ACTION_PATTERNS):
            valid.append(action)
        else:
            print(f"[agent] dropping invalid action: {action!r}")
    return valid


def get_day(observation: str) -> Optional[int]:
    match = re.match(r"Day (\d+)/", observation)
    return int(match.group(1)) if match else None


class SimpleLLMAgent(Agent):
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        system_prompt: str,
        max_history_tokens: int,
        max_response_tokens: int,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens
        self.max_response_tokens = max_response_tokens

        self.transcript: List[str] = []  # alternating observation and action entries, oldest first
        self.queue: List[str] = []  # planned actions not yet sent to the env
        self.queue_day: Optional[int] = None  # the day the queued actions were planned for

    def act(self, observation: str) -> str:
        self.transcript.append(f"Observation:\n{observation}")

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
                max_tokens=self.max_response_tokens,
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

        return clean_actions(plan.actions) or [FALLBACK_ACTION]

    def _messages(self) -> List[dict]:
        transcript = "\n\n".join(self.transcript)
        content = f"Transcript so far (oldest first):\n\n{transcript}\n\nWhat are your next actions?"
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
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=16000,
        help="Max tokens to generate for the action.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("No API key found. Set ANTHROPIC_API_KEY in .env or pass --api-key.")

    config = EnvConfig()
    env = FoodTruckEnv(config)
    obs, info = env.reset(seed=args.seed)
    print(obs)

    agent = SimpleLLMAgent(
        client=anthropic.Anthropic(api_key=args.api_key),
        model=args.model,
        system_prompt=build_system_prompt(config),
        max_history_tokens=args.max_history_tokens,
        max_response_tokens=args.max_response_tokens,
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
