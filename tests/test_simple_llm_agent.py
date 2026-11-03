from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

from agents.simple_llm_agent import (
    FALLBACK_ACTION,
    DayPlan,
    SimpleLLMAgent,
    build_system_prompt,
    clean_actions,
)
from foodtruck import EnvConfig, FoodTruckEnv

CHECKS = ["check storage", "check market", "check recipes", "check menu"]


class FakeMessages:
    """Stands in for client.messages. Returns pre-written plans and counts ~4 chars per token."""

    def __init__(self, plans: List[Optional[List[str]]]) -> None:
        self.plans = list(plans)
        self.parse_calls: List[dict] = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        actions = self.plans.pop(0)
        plan = DayPlan(actions=actions) if actions is not None else None
        return SimpleNamespace(parsed_output=plan, stop_reason="end_turn")

    def count_tokens(self, **kwargs):
        text = kwargs["system"] + kwargs["messages"][0]["content"]
        return SimpleNamespace(input_tokens=len(text) // 4)


def make_agent(plans: List[Optional[List[str]]], max_history_tokens: int = 32000) -> SimpleLLMAgent:
    return SimpleLLMAgent(
        client=SimpleNamespace(messages=FakeMessages(plans)),
        model="test-model",
        system_prompt=build_system_prompt(EnvConfig()),
        max_history_tokens=max_history_tokens,
        max_response_tokens=1000,
    )


def run_steps(agent: SimpleLLMAgent, env: FoodTruckEnv, obs: str, steps: int) -> str:
    for _ in range(steps):
        obs, _, _, _, _ = env.step(agent.act(obs))
    return obs


def test_clean_actions_keeps_valid_and_drops_invalid() -> None:
    actions = [
        "  Check   Market ",
        "buy tomato 3",
        "set menu 1 veggie_wrap 7.5",
        "clear menu 2",
        "end prep",
        "buy tomato",
        "sell tomato 3",
        "I will buy tomatoes",
    ]
    assert clean_actions(actions) == [
        "check market",
        "buy tomato 3",
        "set menu 1 veggie_wrap 7.5",
        "clear menu 2",
        "end prep",
    ]


def test_checks_then_one_call_for_rest_of_day() -> None:
    env = FoodTruckEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    day_plan = ["buy tomato 3", "buy lettuce 3", "buy tortilla 3", "set menu 1 veggie_wrap 7.5", "end prep"]
    agent = make_agent([CHECKS, day_plan, CHECKS])

    obs = run_steps(agent, env, obs, len(CHECKS) + len(day_plan))
    assert len(agent.client.messages.parse_calls) == 2
    assert obs.startswith("Day 2/")

    # Every planned action was sent to the env, in order, and the env applied them.
    sent = [entry.removeprefix("Action: ") for entry in agent.transcript if entry.startswith("Action: ")]
    assert sent == CHECKS + day_plan
    assert agent.queue == []
    assert env.menu[0] == env.recipe_names.index("veggie_wrap")

    # The second call saw the results of all four checks.
    second_prompt = agent.client.messages.parse_calls[1]["messages"][0]["content"]
    for section in ["Storage:", "Market prices:", "Recipes:", "Menu:"]:
        assert section in second_prompt

    # The first call of day 2 saw day 1's results.
    agent.act(obs)
    third_prompt = agent.client.messages.parse_calls[2]["messages"][0]["content"]
    assert "Daily summary:" in third_prompt
    assert "Last action: Bought 3 tomato" in third_prompt
    assert "Last action: Set menu slot 1 to veggie_wrap" in third_prompt


def test_leftover_plan_is_dropped_when_day_ends_early() -> None:
    env = FoodTruckEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    too_many = ["check storage"] * 25  # more than the 20 prep ops in a day
    agent = make_agent([too_many, CHECKS])

    obs = run_steps(agent, env, obs, 20)
    assert obs.startswith("Day 2/")

    assert agent.act(obs) == "check storage"
    assert agent.queue == CHECKS[1:]
    assert len(agent.client.messages.parse_calls) == 2


def test_falls_back_when_llm_gives_no_usable_plan() -> None:
    agent = make_agent([None, ["do something"]])
    assert agent.act("Day 1/20 (Mon)") == FALLBACK_ACTION
    assert agent.act("Day 1/20 (Mon)") == FALLBACK_ACTION


def test_truncation_drops_oldest_whole_entries() -> None:
    agent = make_agent([], max_history_tokens=1000)
    for i in range(50):
        agent.transcript.append(f"Observation:\nDay 1/20 (Mon) step {i} " + "x" * 200)
        agent.transcript.append(f"Action: check storage")

    agent._truncate_transcript()

    assert agent._count_tokens() <= 1000
    assert agent.transcript[0].startswith("Observation:")
    assert agent.transcript[-1] == "Action: check storage"  # newest entries are kept


def test_system_prompt_hides_customer_model() -> None:
    prompt = build_system_prompt(EnvConfig()).lower()
    for hidden in ["elasticity", "weekday", "multiplier", "base demand", "preference", "mean reversion"]:
        assert hidden not in prompt
