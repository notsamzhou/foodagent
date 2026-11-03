# Food Truck Environment

This folder implements a Gymnasium-compatible Food Truck environment with **text-only** observations and actions.

## Install

This project uses [uv](https://docs.astral.sh/uv/). From the project folder:

```bash
uv sync
```

This creates a `.venv` and installs the dependencies from `pyproject.toml`, including dev tools like pytest. Prefix commands with `uv run` to use it (e.g. `uv run python -m examples.example_llm_loop`).

## Project layout

```text
foodtruck/   the environment (FoodTruckEnv, EnvConfig, and its helpers)
agents/      one file per agent; base.py has the Agent base class
examples/    scripts that drive the environment directly
tests/       pytest tests
```

Run scripts from the project folder with `python -m`, e.g. `uv run python -m agents.simple_llm_agent`.

## Quick start

```python
from foodtruck import FoodTruckEnv, EnvConfig

env = FoodTruckEnv(EnvConfig())
obs, info = env.reset(seed=0)
print(obs)

obs, reward, terminated, truncated, info = env.step("check market")
print(obs)
```

## Action commands

All commands are **case-insensitive** and consume **one prep operation**.

### Check commands

- `check storage`
- `check market`
- `check recipes`
- `check menu`

There is **no** `check preferences` command (customer preferences are hidden).

### Direct actions

- `buy <ingredient> <qty>`
- `trash <ingredient> <qty>`
- `set menu <slot> <recipe> <price>`
- `clear menu <slot>`
- `end prep`

### Notes

- `slot` is **1-based** (1..max_menu_items).
- `<ingredient>` and `<recipe>` can be a **name** or a **1-based index**.
- Prices must be within `[price_min, price_max]` and align with `price_step`.

## Observation format

Observation is a single text string with sections. It always includes:

- Day number and day-of-week
- Funds
- Remaining prep ops
- Stage

Checked sections appear only after the corresponding `check ...` command:

- Storage
- Market
- Recipes
- Menu

When `end prep` completes, the observation also includes a **daily summary** (orders, revenue, rent, net).

## Configuration

Hyperparameters are configurable via `EnvConfig` in `foodtruck/config.py` (defaults match the task spec):

- `horizon_days`, `start_funds`, `daily_rent`
- `fridge_capacity`, `pantry_capacity`
- `num_recipes`, `max_menu_items`
- `max_prep_ops_per_day`
- Market and demand dynamics parameters

## Example script

See `examples/example_llm_loop.py` for a minimal loop with scripted commands:

```bash
uv run python -m examples.example_llm_loop
```

## Run the LLM agent (`agents/simple_llm_agent.py`)

This repo includes a reference LLM-driven agent that uses the Anthropic Python SDK.

### Setup

Install dependencies (see [Install](#install)), then create your `.env` and set `ANTHROPIC_API_KEY` in it:

```bash
cp .env.example .env
```

### Run

```bash
uv run --env-file .env python -m agents.simple_llm_agent --seed 1
```

You can also pass the key directly with `--api-key "<YOUR_KEY>"`.

### Useful flags

- `--model`: defaults to `claude-haiku-5-5`
- `--max-steps`: safety cap on environment steps (default 400)
- `--max-history-tokens`: transcript budget sent to the model each step (default 32000). Oldest transcript entries are dropped when over budget.
- `--max-response-tokens`: cap tokens generated per LLM call, including thinking (default 16000)

### How the agent works

- Each day the agent first asks the LLM for its check commands. After the checks run, it makes one more LLM call for the rest of the day's actions, ending with `end prep`.
- The LLM replies with JSON (`{"actions": [...]}`). Each action is checked against the command formats, and invalid ones are dropped.
- `act()` returns one queued action per step. If the day ends early (prep ops run out), leftover actions are discarded.

### Notes

- The script prints an initial observation, then repeatedly prints `>>> <action>` and the resulting observation/reward.
- Token counts come from the Anthropic token counting API (`client.messages.count_tokens`); there is no local tokenizer for Claude models.

## Adding a new agent

1. Create a file in `agents/`, e.g. `agents/my_agent.py`.
2. Subclass `Agent` from `agents.base` and implement `act(observation) -> str`.
3. Add a `main()` with the run loop (see `agents/simple_llm_agent.py`), then run it with `uv run python -m agents.my_agent`.
4. Add tests in `tests/`, e.g. `tests/test_my_agent.py`.

## Tests

The tests use a fake LLM client, so they don't need an API key:

```bash
uv run pytest
```

Useful options:

- `-v`: list each test by name
- `-k <name>`: run only tests whose name contains `<name>`
- `-s`: show `print()` output, such as `[agent] dropping invalid action`
