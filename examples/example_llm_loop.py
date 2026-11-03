from __future__ import annotations

from foodtruck import EnvConfig, FoodTruckEnv


def main() -> None:
    env = FoodTruckEnv(EnvConfig())
    obs, info = env.reset(seed=1)
    print(obs)

    commands = [
        "check recipes",
        "check market",
        "buy tomato 3",
        "buy lettuce 3",
        "buy tortilla 3",
        "set menu 1 veggie_wrap 7.5",
        "check menu",
        "check storage",
        "end prep",
    ]

    for cmd in commands:
        obs, reward, terminated, truncated, info = env.step(cmd)
        print("\n---")
        print(f">>> {cmd}")
        print(obs)
        print(f"reward: {reward:.2f} | terminated: {terminated}")
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
