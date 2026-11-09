"""Record PPO and hybrid side-by-side on one explicitly selected environment seed."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from garl_lander.environment import make_env
from garl_lander.evaluation import action_for
from garl_lander.experiment import load_agent
from garl_lander.io_utils import new_run, read_json, write_json
from garl_lander.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11, help="Independent training seed")
    parser.add_argument("--episode-seed", type=int, default=40000, help="Predeclared common evaluation start")
    args = parser.parse_args()
    cfg = read_json(args.run / "manifest.json")["config"]
    seed_everything(args.seed, cfg["experiment"]["threads"])
    destination = new_run(Path("videos"), f"ppo_vs_hybrid_start{args.episode_seed}", cfg)
    agents = ["ppo", "hybrid"]
    models = [load_agent(args.run / f"{a}_seed{args.seed}", a) for a in agents]
    environments = [make_env(cfg, "rgb_array") for _ in agents]
    observations = [env.reset(seed=args.episode_seed)[0] for env in environments]
    finished, returns, lengths = [False, False], [0.0, 0.0], [0, 0]
    rng = np.random.default_rng(args.episode_seed)
    try:
        with imageio.get_writer(destination / "comparison.mp4", fps=50, codec="libx264", quality=7,
                                macro_block_size=2) as writer:
            while not all(finished):
                frames = []
                for index, (agent, model, env) in enumerate(zip(agents, models, environments, strict=True)):
                    if not finished[index]:
                        action = action_for(model, observations[index], rng)
                        observations[index], reward, terminated, truncated, _ = env.step(action)
                        returns[index] += reward
                        lengths[index] += 1
                        finished[index] = terminated or truncated
                    frame = Image.fromarray(env.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, frame.width, 44), fill="#102A43")
                    draw.text((12, 6), f"{agent.upper()} | training seed {args.seed} | start {args.episode_seed}", fill="white")
                    draw.text((12, 24), f"Return {returns[index]:.1f} | step {lengths[index]} | "
                              + ("FINISHED" if finished[index] else "running"), fill="white")
                    frames.append(np.asarray(frame))
                writer.append_data(np.concatenate(frames, axis=1))
    finally:
        for env in environments:
            env.close()
    write_json(destination / "manifest.json", {"suite": str(args.run), "training_seed": args.seed,
                                               "episode_seed": args.episode_seed, "selection": "predeclared common start",
                                               "agents": agents, "returns": returns, "steps": lengths,
                                               "extra_evaluation_interactions": sum(lengths)})
    print(destination)


if __name__ == "__main__":
    main()
