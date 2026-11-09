"""Five-agent suite, immutable manifests, saved-model evaluation and sweeps."""

import copy
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from .config import validate
from .evaluation import evaluate, summarize
from .genetic_algorithm import GeneticAlgorithm
from .hybrid_agent import train_hybrid
from .io_utils import new_run, provenance, read_json, write_json
from .policy import load_policy
from .ppo_agent import make_ppo, train_ppo
from .seeding import episode_seeds, seed_everything

AGENTS = ("random", "ga", "ppo", "hybrid", "ppo_short")


def run_seed(cfg: dict, seed: int, suite: Path, agents: tuple = AGENTS) -> None:
    """A worker owns exactly one seed; no concurrent writers share agent directories."""
    hybrid_budget = None
    for agent in agents:
        directory = suite / f"{agent}_seed{seed}"
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
        else:
            if directory.exists():
                archive = suite / "interrupted_attempts"
                archive.mkdir(exist_ok=True)
                directory.rename(archive / f"{directory.name}_{time.time_ns()}")
            print(f"Starting {agent}, seed {seed}: {suite.name}", flush=True)
            summary = run_agent(cfg, seed, agent, directory, hybrid_budget)
        if agent == "hybrid":
            hybrid_budget = summary["ppo_steps"]


def load_agent(directory: Path, agent: str):
    if agent == "random":
        return None
    if agent == "ga":
        return load_policy(directory / "ga" / "selected.npz")
    return PPO.load(directory / "ppo" / "selected.zip", device="cpu")


def run_agent(cfg: dict, seed: int, agent: str, directory: Path, short_budget: int | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, cfg["experiment"]["threads"])
    started = time.perf_counter()
    ga_steps = selection_steps = ppo_steps = generations = 0
    total = cfg["experiment"]["total_optimization_steps"]
    training_info = {}
    if agent == "ga":
        ga = GeneticAlgorithm(cfg, seed)
        ga.train(total, directory / "ga")
        ga_steps, selection_steps, generations = ga.optimization_steps, ga.selection_steps, ga.generation
    elif agent == "hybrid":
        ga, training_info = train_hybrid(cfg, seed, directory)
        ga_steps, generations = ga.optimization_steps, ga.generation
        selection_steps = ga.selection_steps + training_info["selection_steps"]
        ppo_steps = training_info["ppo_steps"]
    elif agent in ("ppo", "ppo_short"):
        if agent == "ppo_short" and short_budget is None:
            raise ValueError("PPO-short requires the actual hybrid PPO allocation")
        ppo_dir = directory / "ppo"
        ppo_dir.mkdir(exist_ok=True)
        model = make_ppo(cfg, seed, ppo_dir)
        try:
            training_info = train_ppo(model, cfg, short_budget if agent == "ppo_short" else total, ppo_dir)
        finally:
            model.get_env().close()
        ppo_steps, selection_steps = training_info["ppo_steps"], training_info["selection_steps"]
    elif agent != "random":
        raise ValueError(f"Unknown agent {agent}")
    training_seconds = time.perf_counter() - started if agent != "random" else 0.0
    model = load_agent(directory, agent)
    test_started = time.perf_counter()
    episodes = evaluate(model, cfg, episode_seeds(cfg, "final"))
    episodes.to_csv(directory / "evaluation.csv", index=False)
    summary = summarize(episodes)
    allocation = 0 if agent == "random" else short_budget if agent == "ppo_short" else total
    summary.update({"agent": agent, "seed": seed, "status": "complete", "ga_steps": ga_steps,
                    "ppo_steps": ppo_steps, "optimization_steps": ga_steps + ppo_steps,
                    "selection_steps": selection_steps, "allocated_steps": allocation,
                    "unused_steps": allocation - ga_steps - ppo_steps,
                    "total_steps": ga_steps + ppo_steps + selection_steps + summary["test_steps"],
                    "training_seconds": training_seconds, "test_seconds": time.perf_counter() - test_started,
                    "generations": generations, "selected_ppo_steps": training_info.get("selected_ppo_steps"),
                    "selection_rule": "best validation mean; earliest checkpoint breaks ties",
                    "return_per_1000_optimization_steps": (summary["mean_return"] * 1000 / (ga_steps + ppo_steps)
                                                           if ga_steps + ppo_steps else None)})
    write_json(directory / "summary.json", summary)
    return summary


def run_suite(cfg: dict, root: Path = Path("results/raw"), agents: tuple = AGENTS,
              resume: Path | None = None, workers: int = 1) -> Path:
    """Resume completed agent runs; interrupted agents restart in a new attempt directory.

    Generation-level exact resume is separately exposed for standalone GA work. PPO
    latest.zip is preserved, but whole-suite restarts are used for deterministic auditability.
    """
    validate(cfg)
    if workers < 1:
        raise ValueError("workers must be positive")
    suite = resume or new_run(root, cfg["name"], cfg)
    manifest = suite / "manifest.json"
    if resume:
        if read_json(manifest)["config"] != cfg:
            raise ValueError("Resume configuration differs from recorded configuration")
    else:
        write_json(manifest, provenance(cfg))
    write_json(suite / "status.json", {"status": "running"})
    try:
        if workers == 1:
            for seed in cfg["experiment"]["seeds"]:
                run_seed(cfg, seed, suite, agents)
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(cfg["experiment"]["seeds"]))) as pool:
                futures = [pool.submit(run_seed, cfg, seed, suite, agents) for seed in cfg["experiment"]["seeds"]]
                for future in as_completed(futures):
                    future.result()
        write_json(suite / "status.json", {"status": "complete"})
    except BaseException as exc:
        write_json(suite / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                           "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    print(f"Completed suite: {suite}", flush=True)
    return suite


def suite_runs(suite: Path):
    for path in sorted(suite.glob("*_seed*/summary.json")):
        yield path.parent, read_json(path)


def reevaluate(suite: Path, stochastic: bool = False) -> Path:
    cfg = read_json(suite / "manifest.json")["config"]
    destination = new_run(suite / "reevaluations", "stochastic" if stochastic else "deterministic", cfg)
    for directory, summary in suite_runs(suite):
        seed_everything(summary["seed"], cfg["experiment"]["threads"])
        rows = evaluate(load_agent(directory, summary["agent"]), cfg, episode_seeds(cfg, "final"),
                        deterministic=not stochastic)
        rows.to_csv(destination / f"{directory.name}.csv", index=False)
    write_json(destination / "manifest.json", {"source": str(suite), "deterministic": not stochastic})
    return destination


def run_sweeps(cfg: dict, root: Path) -> Path:
    """One-factor-at-a-time sweeps, not a confounded Cartesian product."""
    destination = new_run(root, "sweep", cfg)
    records = []
    for parameter, values in cfg["sweep"].items():
        for value in values:
            variant = copy.deepcopy(cfg)
            variant.pop("sweep")
            section, key = parameter.split(".")
            variant[section][key] = value
            variant["name"] = f"{section}_{key}_{value}"
            # Development hold-out is disjoint from the main final-test seed set.
            variant["evaluation"]["final_seed_start"] = 60000
            variant["evaluation_split"] = "development_ablation"
            suite = run_suite(variant, destination, agents=("ga", "hybrid"))
            for directory, summary in suite_runs(suite):
                records.append({"parameter": parameter, "value": value, "suite": str(suite), **summary})
            pd.DataFrame(records).to_csv(destination / "sweep_results.csv", index=False)
    return destination


def robustness(suite: Path, conditions: dict, wait_for_completion: bool = False) -> Path:
    cfg = read_json(suite / "manifest.json")["config"]
    destination = new_run(suite / "robustness", "robustness", cfg)
    records, processed = [], set()
    print(f"Robustness output: {destination}", flush=True)
    while True:
        for directory, summary in suite_runs(suite):
            if directory.name in processed:
                continue
            for condition, overrides in conditions.items():
                variant = copy.deepcopy(cfg)
                variant["environment"].update(overrides)
                validate(variant)
                seed_everything(summary["seed"], cfg["experiment"]["threads"])
                rows = evaluate(load_agent(directory, summary["agent"]), variant, episode_seeds(cfg, "final"))
                rows.to_csv(destination / f"{condition}_{directory.name}.csv", index=False)
                records.append({"condition": condition, "agent": summary["agent"], "seed": summary["seed"],
                                **summarize(rows)})
                pd.DataFrame(records).to_csv(destination / "robustness_results.csv", index=False)
            processed.add(directory.name)
            print(f"  Robustness evaluated {directory.name} on {len(conditions)} conditions", flush=True)
        status = read_json(suite / "status.json")["status"]
        all_finished_names = {directory.name for directory, _ in suite_runs(suite)}
        complete = not wait_for_completion or (status == "complete" and all_finished_names <= processed)
        write_json(destination / "manifest.json", {"source": str(suite), "conditions": conditions,
                                                   "status": "complete" if complete else "waiting_for_training",
                                                   "extra_evaluation_steps": sum(r["test_steps"] for r in records)})
        if complete:
            break
        if status in ("failed", "interrupted"):
            raise RuntimeError("Source training stopped; partial robustness data preserved")
        time.sleep(1)
    return destination
