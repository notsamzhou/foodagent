from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Set, Type, TypeVar

import anthropic
import pydantic
from pydantic import BaseModel

from agents.base import MAX_RESPONSE_TOKENS, Agent, clean_actions, describe_game, get_day, run_episode
from foodtruck import EnvConfig, FoodTruckEnv


class CheckerReport(BaseModel):
    report: str


class MenuDecision(BaseModel):
    actions: List[str]
    notes_for_buyer: str


class BuyPlan(BaseModel):
    actions: List[str]


T = TypeVar("T", bound=BaseModel)


def team_intro(config: EnvConfig) -> str:
    return f"""You are part of a team of assistants that runs a food truck. {describe_game(config)}

How the team works:
Each day's prep stage is handled by three assistants, one after another. Each one only sees what is passed to it.
1. Checker: the routine checks (storage, market, menu, and recipes) have already been run. The checker reads the results and the log of past days, and writes a report with observations and ideas for the other two.
2. Menu setter: reads the checker's report and decides which recipes go on the menu and at what price.
3. Buyer: reads the menu decision and buys the ingredients needed to serve it.
After the buyer, `end prep` is sent automatically and the truck opens.
The log of past days shows, for each day, the menu at the start of the day, market prices, the commands the team sent and their results, and the daily summary of orders and revenue. It is the only way to learn what customers want."""


def build_checker_prompt(config: EnvConfig) -> str:
    return f"""{team_intro(config)}

Your role: checker.
You do not send any commands. Write a report for the menu setter and the buyer. Cover:
- Funds, and how many days are left in the game.
- Stock: for each dish on the current menu, how many orders the current storage can fill, and which ingredient runs out first.
- Costs: the ingredient cost of one serving of each recipe at today's market prices, and the profit per order at the current menu price. Point out ingredients that are much cheaper or more expensive than on past days.
- Demand: from the daily summaries, which dishes sold well, which had unfilled orders (they sold out, so demand was higher than stock), and which got few or no orders, and how orders changed when prices or the menu changed.
- Problems: commands that failed on past days (for example "Insufficient funds" or "Insufficient storage capacity"), storage close to full, stock that no menu dish uses.
- Ideas: concrete suggestions, such as a dish to drop, a price to try, an untried recipe that is cheap to make or uses stock we already have, or how much stock to aim for.
Be specific and use numbers. Keep the report under about 300 words. On day 1 there is no history, so focus on costs and what looks promising.
Respond with a JSON object like {{"report": "..."}}."""


def build_menu_setter_prompt(config: EnvConfig) -> str:
    return f"""{team_intro(config)}

Your role: menu setter.
Decide today's menu. You can only send `set menu <slot> <recipe> <price>` and `clear menu <slot>`. Any other command is dropped.
Your goal is to maximize profit (revenue minus ingredient cost) over the rest of the game.
- The menu stays as it was yesterday unless you change it. Only send commands for slots you want to change, because every command uses a prep op the buyer could use instead.
- Price: use the daily summaries to see how each dish sold at the prices tried so far, and keep the price well above the ingredient cost per serving.
- Stock: the buyer buys ingredients after you, so you are not limited to what is in storage. But dishes that use stock we already have cost nothing extra today, and stock that no menu dish uses is wasted money.
- Unfilled orders mean a dish sold out. That shows demand, so it is a reason to keep the dish and stock more, not to drop it.
- Exploration: the only way to learn how popular a recipe is, is to put it on the menu. Recipes not tried yet are listed in the message. When a slot is empty or a dish is doing poorly, try one of them at a reasonable price, especially early in the game. Don't drop a dish that is clearly profitable just to explore. Late in the game exploring is worth less, because there are few days left to use what you learn.
- A recipe can only be in one slot. To move a recipe to another slot, clear its old slot first.
In notes_for_buyer, list every dish that will be on the menu today with its price and how many orders you expect for it, so the buyer knows how much to stock. Add anything else the buyer should know.
Respond with a JSON object like {{"actions": ["set menu 2 <recipe> <price>"], "notes_for_buyer": "..."}}. Use an empty actions list to keep the menu as it is."""


def build_buyer_prompt(config: EnvConfig) -> str:
    return f"""{team_intro(config)}

Your role: buyer.
Buy the ingredients for today's menu. You can only send `buy <ingredient> <qty>` and `trash <ingredient> <qty>`. Any other command is dropped.
- Work out what is needed: for each dish on today's menu, expected orders times the recipe's ingredient amounts, minus what is already in storage. Add up ingredients that are shared between dishes.
- Every order needs all of a recipe's ingredients. If one is missing, the order goes unfilled.
- A small buffer above the expected orders is fine because leftovers keep. But don't overbuy: money spent on stock that never gets used is lost, and leftovers are worth nothing after day {config.horizon_days}. Near the end of the game, buy only what you expect to use.
- Market prices change every day. If an ingredient you will keep using is unusually cheap today, buying extra can pay off, if storage allows.
- Limits: funds, fridge and pantry capacity (the storage check shows which ingredient goes where), and at most {config.max_buy_qty} per buy command. Repeat a command to buy more.
- Commands run in order. If one fails (not enough funds or space), the rest still run.
- Trash only to free storage space for something more useful. It gives no money back.
- You can send a limited number of commands today. The limit is given in the message, and extra commands are dropped.
Respond with a JSON object like {{"actions": ["buy <ingredient> 6", "buy <ingredient> 4"]}}. Use an empty list if nothing needs to be bought."""


def check_section(observation: str) -> str:
    """The part of a check observation after the header and 'Last action' lines."""
    lines = observation.splitlines()[1:]
    return "\n".join(line for line in lines if not line.startswith("Last action:"))


def last_action_result(observation: str) -> str:
    match = re.search(r"^Last action: (.*)$", observation, re.MULTILINE)
    return match.group(1) if match else "(no result)"


def daily_summary(observation: str) -> str:
    lines = observation.splitlines()
    if "Daily summary:" not in lines:
        return ""
    start = lines.index("Daily summary:")
    summary = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith("- "):
            break
        summary.append(line)
    return "\n".join(summary)


class TeamAgent(Agent):
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        config: EnvConfig,
    ) -> None:
        self.client = client
        self.model = model
        self.config = config

        self.log: List[str] = []  # one entry per finished day, oldest first
        self.recipes_text: Optional[str] = None  # recipes never change, so they are only checked once
        self.tried_recipes: Set[str] = set()  # recipes seen on the menu so far

        self.day: Optional[int] = None
        self.day_header = ""  # first line of the day's first observation, without prep ops left
        self.check_results: Dict[str, str] = {}  # check target -> its section of the observation
        self.results: List[str] = []  # "command -> result" for today's non-check commands
        self.queue: List[str] = []  # actions not yet sent to the env
        self.planned = False  # whether the three agents have run today
        self.ops_used = 0  # actions sent today
        self.last_action: Optional[str] = None

    def act(self, observation: str) -> str:
        self._record_result(observation)

        day = get_day(observation)
        if day != self.day:
            if self.day is not None:
                self._finish_day(observation)
            self._start_day(day, observation)

        if not self.queue and not self.planned:
            self.queue = self._plan_day()
            self.planned = True

        action = self.queue.pop(0) if self.queue else "end prep"
        self.last_action = action
        self.ops_used += 1
        return action

    def _record_result(self, observation: str) -> None:
        """Store what the previous action did."""
        if self.last_action is None or self.last_action == "end prep":
            return
        if self.last_action.startswith("check "):
            target = self.last_action.removeprefix("check ")
            section = check_section(observation)
            if target == "recipes":
                self.recipes_text = section
            else:
                self.check_results[target] = section
            if target == "menu":
                self.tried_recipes.update(re.findall(r"- Slot \d+: (\S+) @", section))
        else:
            self.results.append(f"- {self.last_action} -> {last_action_result(observation)}")

    def _finish_day(self, observation: str) -> None:
        """Add the day that just ended to the log."""
        parts = [
            self.day_header,
            self.check_results.get("menu", "Menu: (not checked)"),
            self.check_results.get("market", "Market prices: (not checked)"),
            "Commands and results:",
            "\n".join(self.results) or "- (none)",
            daily_summary(observation) or "Daily summary: (not available)",
        ]
        self.log.append("\n".join(parts))

    def _start_day(self, day: Optional[int], observation: str) -> None:
        self.day = day
        self.day_header = re.sub(r" \| Prep ops left: \d+", "", observation.splitlines()[0])
        self.check_results = {}
        self.results = []
        self.planned = False
        self.ops_used = 0
        self.queue = ["check storage", "check market", "check menu"]
        if self.recipes_text is None:
            self.queue.append("check recipes")

    def _plan_day(self) -> List[str]:
        """Run the checker, menu setter, and buyer, and return the rest of the day's commands."""
        today = self._today_text()
        log = self._log_text()

        report = self._ask(
            build_checker_prompt(self.config),
            f"{today}\n\n{self._untried_text()}\n\n{log}\n\nWrite your report.",
            CheckerReport,
        )
        checker_report = report.report if report else "(the checker gave no report)"
        print(f"\n[checker]\n{checker_report}")

        decision = self._ask(
            build_menu_setter_prompt(self.config),
            f"{today}\n\n{self._untried_text()}\n\n{log}\n\n"
            f"Checker's report:\n{checker_report}\n\nDecide today's menu.",
            MenuDecision,
        )
        max_actions = self.config.max_prep_ops_per_day - self.ops_used - 1  # one op is left for end prep
        menu_actions = self._keep(decision.actions if decision else [], ("set menu", "clear menu"))[:max_actions]
        notes = decision.notes_for_buyer if decision else "(the menu setter gave no notes; the menu stays as it is)"
        print(f"\n[menu setter] {menu_actions}\n{notes}")

        max_buy_actions = max(max_actions - len(menu_actions), 0)
        menu_text = "\n".join(f"- {a}" for a in menu_actions) or "- (none, the menu stays as it is)"
        plan = self._ask(
            build_buyer_prompt(self.config),
            f"{today}\n\n{log}\n\nChecker's report:\n{checker_report}\n\n"
            f"Menu setter's commands (they run before yours):\n{menu_text}\n\n"
            f"Menu setter's notes:\n{notes}\n\n"
            f"You can send at most {max_buy_actions} commands today.\n\nDecide what to buy.",
            BuyPlan,
        )
        buy_actions = self._keep(plan.actions if plan else [], ("buy", "trash"))[:max_buy_actions]
        print(f"\n[buyer] {buy_actions}")

        return menu_actions + buy_actions + ["end prep"]

    def _keep(self, actions: List[str], prefixes: tuple) -> List[str]:
        """Clean the actions and keep only the commands this agent is allowed to send."""
        kept = []
        for action in clean_actions(actions, self.config):
            if action.startswith(prefixes):
                kept.append(action)
            else:
                print(f"[agent] dropping out-of-role action: {action!r}")
        return kept

    def _today_text(self) -> str:
        sections = [f"Today: {self.day_header}"]
        for target in ["storage", "market", "menu"]:
            sections.append(self.check_results.get(target, f"{target.capitalize()}: (not checked)"))
        sections.append(self.recipes_text or "Recipes: (not checked)")
        return "\n\n".join(sections)

    def _untried_text(self) -> str:
        all_recipes = re.findall(r"^- \d+\. (\S+):", self.recipes_text or "", re.MULTILINE)
        untried = [name for name in all_recipes if name not in self.tried_recipes]
        return f"Recipes not tried yet: {', '.join(untried) or '(none)'}"

    def _log_text(self) -> str:
        if not self.log:
            return "Log of past days: (none yet, this is the first day)"
        return "Log of past days (oldest first):\n\n" + "\n\n".join(self.log)

    def _ask(self, system: str, content: str, output_format: Type[T]) -> Optional[T]:
        """Make one LLM call. Returns None if the response can't be used."""
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=MAX_RESPONSE_TOKENS,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_format=output_format,
            )
        except pydantic.ValidationError as e:
            print(f"[agent] could not parse {output_format.__name__} response: {e}")
            return None

        if response.parsed_output is None:
            print(f"[agent] no {output_format.__name__} in LLM response (stop_reason: {response.stop_reason})")
        return response.parsed_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Checker, menu setter, and buyer agents for FoodTruckEnv.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (defaults to the ANTHROPIC_API_KEY env var).",
    )
    parser.add_argument("--model", default="claude-haiku-5-5", help="Anthropic model name.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for env reset.")
    parser.add_argument("--max-steps", type=int, default=400, help="Safety cap on steps.")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("No API key found. Set ANTHROPIC_API_KEY in .env or pass --api-key.")

    config = EnvConfig()
    agent = TeamAgent(
        client=anthropic.Anthropic(api_key=args.api_key),
        model=args.model,
        config=config,
    )
    run_episode(FoodTruckEnv(config), agent, seed=args.seed, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
