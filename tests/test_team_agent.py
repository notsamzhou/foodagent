from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Dict, List, Optional

from pydantic import BaseModel

from agents.team_agent import (
    BuyPlan,
    CheckerReport,
    MenuDecision,
    TeamAgent,
    build_buyer_prompt,
    build_checker_prompt,
    build_menu_setter_prompt,
)
from foodtruck import EnvConfig, FoodTruckEnv

DAY_1_CHECKS = ["check storage", "check market", "check menu", "check recipes"]
LATER_CHECKS = ["check storage", "check market", "check menu"]
MENU = MenuDecision(actions=["set menu 1 veggie_wrap 7.5"], notes_for_buyer="Expect 3 veggie_wrap orders.")
BUYS = BuyPlan(actions=["buy tomato 3", "buy lettuce 3", "buy tortilla 3"])


class FakeMessages:
    """Stands in for client.messages. Returns pre-written responses for each output format."""

    def __init__(self, responses: Dict[type, List[Optional[BaseModel]]]) -> None:
        self.responses = {fmt: list(items) for fmt, items in responses.items()}
        self.parse_calls: List[dict] = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        parsed = self.responses[kwargs["output_format"]].pop(0)
        return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn")

    def prompt(self, call: int) -> str:
        return self.parse_calls[call]["messages"][0]["content"]


def make_agent(checker=None, menu=None, buyer=None) -> TeamAgent:
    responses = {
        CheckerReport: checker or [CheckerReport(report="Funds are fine. Try veggie_wrap.")],
        MenuDecision: menu or [MENU],
        BuyPlan: buyer or [BUYS],
    }
    return TeamAgent(
        client=SimpleNamespace(messages=FakeMessages(responses)),
        model="test-model",
        config=EnvConfig(),
    )


def run_day(agent: TeamAgent, env: FoodTruckEnv, obs: str) -> tuple:
    """Run until the day changes. Returns the new observation and the actions sent."""
    day = obs.split("/")[0]
    sent = []
    while obs.split("/")[0] == day:
        action = agent.act(obs)
        sent.append(action)
        obs, _, _, _, _ = env.step(action)
    return obs, sent


def test_day_runs_checks_then_menu_setter_then_buyer() -> None:
    env = FoodTruckEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    agent = make_agent()

    _, sent = run_day(agent, env, obs)

    assert sent == DAY_1_CHECKS + MENU.actions + BUYS.actions + ["end prep"]
    # The three agents run in order, each with its own system prompt.
    config = EnvConfig()
    calls = [(call["output_format"], call["system"]) for call in agent.client.messages.parse_calls]
    assert calls == [
        (CheckerReport, build_checker_prompt(config)),
        (MenuDecision, build_menu_setter_prompt(config)),
        (BuyPlan, build_buyer_prompt(config)),
    ]


def test_next_day_skips_recipe_check_and_logs_day_one() -> None:
    env = FoodTruckEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    agent = make_agent(
        checker=[CheckerReport(report="day 1"), CheckerReport(report="day 2")],
        menu=[MENU, MenuDecision(actions=[], notes_for_buyer="keep it")],
        buyer=[BUYS, BuyPlan(actions=[])],
    )

    obs, _ = run_day(agent, env, obs)
    _, sent = run_day(agent, env, obs)

    assert sent == LATER_CHECKS + ["end prep"]
    # Day 1's log has each command with the env's result, and the daily summary.
    day_2_prompt = agent.client.messages.prompt(3)
    assert "- buy tomato 3 -> Bought 3 tomato" in day_2_prompt
    assert "Daily summary:\n- Orders filled:" in day_2_prompt


def storage_and_market_lines(env: FoodTruckEnv) -> List[str]:
    """What the storage and market checks will show, built from the env's own state."""
    fridge, pantry = [], []
    for ing, qty in zip(env.ingredients, env.inventory):
        (fridge if ing["storage"] == "fridge" else pantry).append(f"{ing['name']}: {qty}")
    market = ", ".join(f"{name}: ${price:.2f}" for name, price in zip(env.ingredient_names, env.prices))
    return [f"- Fridge: {', '.join(fridge)}", f"- Pantry: {', '.join(pantry)}", market]


def test_each_agent_gets_the_right_context_each_day() -> None:
    env = FoodTruckEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    menu_actions = [["set menu 1 veggie_wrap 7.5"], ["set menu 2 tomato_salad 6"], []]
    agent = make_agent(
        checker=[CheckerReport(report=f"report day {d}") for d in (1, 2, 3)],
        menu=[MenuDecision(actions=menu_actions[d - 1], notes_for_buyer=f"notes day {d}") for d in (1, 2, 3)],
        buyer=[BuyPlan(actions=["buy tomato 4", "buy lettuce 2", "buy tortilla 2"]) for _ in range(3)],
    )
    messages = agent.client.messages

    for day in (1, 2, 3):
        expected_today = storage_and_market_lines(env)
        obs, _ = run_day(agent, env, obs)

        checker, menu_setter, buyer = [messages.prompt(3 * (day - 1) + i) for i in range(3)]
        for prompt in [checker, menu_setter, buyer]:
            today, log = prompt.split("Log of past days", 1)

            # Today's section has today's check results and the recipes.
            assert today.startswith(f"Today: Day {day}/20")
            for line in expected_today:
                assert line in today
            assert "- 1. veggie_wrap:" in today

            # The log has every finished day, oldest first, and not today.
            assert re.findall(r"^Day (\d+)/", log, re.MULTILINE) == [str(d) for d in range(1, day)]

            # Past days only show what the env returned, never an agent's report or notes.
            for past in range(1, day):
                assert f"report day {past}" not in prompt
                assert f"notes day {past}" not in prompt

            # The only prep op count is the buyer's limit.
            assert "Prep ops left" not in prompt

        # Only the checker and menu setter get the untried list. A recipe is tried once it was on the menu for a day.
        tried = {1: [], 2: ["veggie_wrap"], 3: ["veggie_wrap", "tomato_salad"]}[day]
        untried = ", ".join(name for name in env.recipe_names if name not in tried)
        for prompt in [checker, menu_setter]:
            assert f"Recipes not tried yet: {untried}" in prompt.splitlines()
        assert "Recipes not tried yet:" not in buyer

        # Later agents get today's output from the earlier ones.
        assert f"Checker's report:\nreport day {day}" in menu_setter
        assert f"Checker's report:\nreport day {day}" in buyer
        assert f"notes day {day}" in buyer
        for action in menu_actions[day - 1]:
            assert f"- {action}" in buyer
        if not menu_actions[day - 1]:
            assert "- (none, the menu stays as it is)" in buyer
        # 20 ops - checks (recipes only on day 1) - menu commands - end prep
        checks = 4 if day == 1 else 3
        assert f"at most {20 - checks - len(menu_actions[day - 1]) - 1} commands" in buyer


def test_out_of_role_and_extra_actions_are_dropped() -> None:
    env = FoodTruckEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    agent = make_agent(
        menu=[MenuDecision(actions=["set menu 1 veggie_wrap 7.5", "buy tomato 3"], notes_for_buyer="")],
        buyer=[BuyPlan(actions=["set menu 2 tomato_salad 6", "nonsense"] + ["buy tomato 1"] * 30)],
    )

    _, sent = run_day(agent, env, obs)

    assert sent == DAY_1_CHECKS + ["set menu 1 veggie_wrap 7.5"] + ["buy tomato 1"] * 14 + ["end prep"]


def test_menu_actions_are_capped_to_prep_ops_left() -> None:
    env = FoodTruckEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    agent = make_agent(menu=[MenuDecision(actions=["clear menu 1"] * 30, notes_for_buyer="")])

    _, sent = run_day(agent, env, obs)

    # 20 ops - 4 checks - end prep leaves 15 for the menu, and none for the buyer.
    assert sent == DAY_1_CHECKS + ["clear menu 1"] * 15 + ["end prep"]


def test_day_still_ends_when_llms_give_nothing() -> None:
    env = FoodTruckEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    agent = make_agent(checker=[None], menu=[None], buyer=[None])

    _, sent = run_day(agent, env, obs)

    assert sent == DAY_1_CHECKS + ["end prep"]
    assert "(the checker gave no report)" in agent.client.messages.prompt(1)
