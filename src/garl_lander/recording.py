"""Headless MP4 capture, reproducible seed selection, and descriptive overlays."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .environment import make_env
from .evaluation import action_for
from .experiment import load_agent
from .io_utils import new_run, read_json, write_json
from .seeding import seed_everything


def record(suite: Path, agent: str, seed: int, episodes: int = 1, selection: str = "first") -> Path:
    cfg = read_json(suite / "manifest.json")["config"]
    seed_everything(seed, cfg["experiment"]["threads"])
    run = suite / f"{agent}_seed{seed}"
    table = pd.read_csv(run / "evaluation.csv")
    if not 1 <= episodes <= len(table):
        raise ValueError("episodes must be between one and the number of evaluated episodes")
    if selection == "highest":
        chosen = table.nlargest(episodes, "return")
    elif selection == "median":
        chosen = table.iloc[(table["return"] - table["return"].median()).abs().argsort()[:episodes]]
    elif selection == "first":
        chosen = table.head(episodes)
    else:
        raise ValueError("Unknown selection method")
    destination = new_run(Path("videos"), f"{agent}_seed{seed}_{selection}", cfg)
    model = load_agent(run, agent)
    env = make_env(cfg, "rgb_array")
    metadata = []
    try:
        for episode in chosen.rename(columns={"return": "episode_return"}).itertuples():
            obs, _ = env.reset(seed=int(episode.episode_seed))
            rng = np.random.default_rng(int(episode.episode_seed) + 700000)
            total, length = 0.0, 0
            path = destination / f"episode_{episode.episode_seed}.mp4"
            with imageio.get_writer(path, fps=50, codec="libx264", quality=7, macro_block_size=2) as writer:
                while True:
                    obs, reward, terminated, truncated, _ = env.step(action_for(model, obs, rng))
                    total += reward
                    length += 1
                    frame = Image.fromarray(env.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, frame.width, 44), fill="#102A43")
                    draw.text((12, 6), f"{agent} | test seed {episode.episode_seed} | {selection} selection", fill="white")
                    draw.text((12, 24), f"Step {length} | return {total:.1f}", fill="white")
                    writer.append_data(np.asarray(frame))
                    if terminated or truncated:
                        break
            if abs(total - episode.episode_return) > 1e-4:
                raise AssertionError("Recorded return differs from evaluation")
            metadata.append({"path": path.name, "episode_seed": int(episode.episode_seed),
                             "return": float(total), "steps": length, "selection": selection})
    finally:
        env.close()
    write_json(destination / "manifest.json", {"suite": str(suite), "agent": agent,
                                               "training_seed": seed, "episodes": metadata})
    return destination
