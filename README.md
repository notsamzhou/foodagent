# Food Truck Environment

This folder implements a Gymnasium-compatible Food Truck environment with **text-only** observations and actions.

## Install

This project uses [uv](https://docs.astral.sh/uv/). From the project folder:

```bash
uv sync
```

This creates a `.venv` and installs the dependencies from `pyproject.toml`. Prefix commands with `uv run` to use it (e.g. `uv run python example_llm_loop.py`).

## Quick start

```python
from foodtruck_env import FoodTruckEnv, EnvConfig

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

Hyperparameters are configurable via `EnvConfig` in `config.py` (defaults match the task spec):

- `horizon_days`, `start_funds`, `daily_rent`
- `fridge_capacity`, `pantry_capacity`
- `num_recipes`, `max_menu_items`
- `max_prep_ops_per_day`
- Market and demand dynamics parameters

## Example script

See `example_llm_loop.py` for a minimal loop with scripted commands.

## Run the LLM agent (`simple_llm_agent.py`)

This repo includes a reference LLM-driven agent that uses the Anthropic Python SDK.

### Setup

Install dependencies (see [Install](#install)), then create your `.env` and set `ANTHROPIC_API_KEY` in it:

```bash
cp .env.example .env
```

### Run

```bash
uv run --env-file .env python simple_llm_agent.py --seed 1
```

You can also pass the key directly with `--api-key "<YOUR_KEY>"`.

### Useful flags

- `--model`: defaults to `claude-haiku-5-5`
- `--max-steps`: safety cap on environment steps (default 400)
- `--max-history-tokens`: transcript budget sent to the model each step (default 32000). Oldest transcript entries are dropped when over budget.
- `--max-response-tokens`: cap tokens generated for the action (default 1024)

### Notes

- The script prints an initial observation, then repeatedly prints `>>> <action>` and the resulting observation/reward.
- Token counts come from the Anthropic token counting API (`client.messages.count_tokens`); there is no local tokenizer for Claude models.
