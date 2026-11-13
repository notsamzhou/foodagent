from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from foodtruck import EnvConfig, FoodTruckEnv


# Used when the LLM gives us nothing usable. It's harmless and gives the LLM fresh info next time.
FALLBACK_ACTION = "check storage"

# Max tokens the LLM can generate per call.
MAX_RESPONSE_TOKENS = 16000

# One pattern per command the environment accepts (see README "Action commands").
ACTION_PATTERNS = [
    r"check (storage|market|recipes|menu)",
    r"buy \S+ \d+",
    r"trash \S+ \d+",
    r"set menu \d+ \S+ \d+(\.\d+)?",
    r"clear menu \d+",
    r"end prep",
]


class Agent(ABC):
    @abstractmethod
    def act(self, observation: str) -> str:
        raise NotImplementedError


def describe_game(config: EnvConfig) -> str:
    # Only rules the agent is allowed to know. Customer demand, price dynamics,
    # and other hidden parameters are left out on purpose.
    return f"""Your goal is to have as much money as possible at the end of day {config.horizon_days}.

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
- The result of a check only appears in the observation right after that check."""


def clean_actions(actions: List[str], config: EnvConfig) -> List[str]:
    """Normalize the LLM's actions and drop any that don't match a command pattern or break a rule."""
    valid = []
    for action in actions:
        action = " ".join(action.strip().lower().split())
        if not any(re.fullmatch(pattern, action) for pattern in ACTION_PATTERNS):
            print(f"[agent] dropping invalid action: {action!r}")
        elif not within_limits(action, config):
            print(f"[agent] dropping action that breaks the rules: {action!r}")
        else:
            valid.append(action)
    return valid


def within_limits(action: str, config: EnvConfig) -> bool:
    """Check the numbers in an action against the rules in describe_game."""
    tokens = action.split()
    if tokens[0] == "buy":
        return 1 <= int(tokens[2]) <= config.max_buy_qty
    if tokens[0] == "trash":
        return 1 <= int(tokens[2]) <= config.max_trash_qty
    if tokens[0] in ("set", "clear") and not 1 <= int(tokens[2]) <= config.max_menu_items:
        return False
    if tokens[0] == "set":
        price = float(tokens[4])
        steps = round((price - config.price_min) / config.price_step)
        on_grid = abs(config.price_min + steps * config.price_step - price) <= 1e-6
        return config.price_min <= price <= config.price_max and on_grid
    return True


def get_day(observation: str) -> Optional[int]:
    match = re.match(r"Day (\d+)/", observation)
    return int(match.group(1)) if match else None


def run_episode(env: FoodTruckEnv, agent: Agent, seed: int, max_steps: int) -> None:
    """Run one episode, printing each action and observation."""
    obs, _ = env.reset(seed=seed)
    print(obs)

    for _ in range(max_steps):
        action = agent.act(obs)

        print("\n---")
        print(f">>> {action}")

        obs, reward, terminated, truncated, _ = env.step(action)
        print(obs)
        print(f"reward: {reward:.2f} | terminated: {terminated}")

        if terminated or truncated:
            break
