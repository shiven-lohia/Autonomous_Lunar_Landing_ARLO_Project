import argparse

from garl_lander.config import DEFAULT
from garl_lander.environment import make_env


def main():
    parser = argparse.ArgumentParser(description="One interactive random-policy LunarLander episode")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    env = make_env(DEFAULT, "human")
    try:
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        total = 0.0
        while True:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            if terminated or truncated:
                break
        print(f"Random episode return: {total:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
