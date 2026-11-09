"""Transparent NumPy genetic search with generation-boundary resumability."""

import json
import time
from pathlib import Path

import numpy as np

from .evaluation import evaluate
from .io_utils import append_csv
from .policy import GAPolicyNetwork, initialize_chromosome, save_policy
from .seeding import episode_seeds, training_seed


def tournament(fitness: np.ndarray, rng: np.random.Generator, size: int = 3) -> int:
    candidates = rng.choice(len(fitness), size=size, replace=False)
    return int(candidates[np.argmax(fitness[candidates])])


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, kind: str = "uniform") -> np.ndarray:
    if kind == "uniform":
        return np.where(rng.random(a.size) < 0.5, a, b).copy()
    if kind == "blend":
        alpha = rng.random(a.size)
        return (alpha * a + (1 - alpha) * b).astype(np.float32)
    raise ValueError("Unsupported crossover")


def mutate(chromosome: np.ndarray, rng: np.random.Generator, probability: float, sigma: float) -> np.ndarray:
    child = chromosome.copy()
    mask = rng.random(child.size) < probability
    child[mask] += rng.normal(0, sigma, int(mask.sum())).astype(np.float32)
    if not np.isfinite(child).all():
        raise FloatingPointError("Mutation produced non-finite weights")
    return child


class GeneticAlgorithm:
    """Reserve a worst-case complete generation before starting its evaluation.

    This avoids partially evaluated populations, truncated fitness and budget overruns.
    Unused allocation is explicit; the hybrid spends it on full PPO rollouts.
    """

    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        streams = np.random.SeedSequence([seed, 493]).spawn(4)
        self.rngs = [np.random.default_rng(s) for s in streams]
        self.population = np.stack([initialize_chromosome(self.rngs[0])
                                    for _ in range(cfg["ga"]["population_size"])])
        self.generation = 0
        self.optimization_steps = 0
        self.selection_steps = 0
        self.elapsed = 0.0
        self.champion = self.population[0].copy()
        self.champion_validation = None
        self.history = []
        self.initial_return = None

    @property
    def generation_reserve(self) -> int:
        return (self.cfg["ga"]["population_size"] * self.cfg["ga"]["fitness_episodes"]
                * self.cfg["environment"]["max_episode_steps"])

    def breed(self, fitness: np.ndarray) -> np.ndarray:
        g = self.cfg["ga"]
        if not np.isfinite(fitness).all():
            raise FloatingPointError("Non-finite fitness")
        order = np.argsort(-fitness, kind="stable")
        offspring = [self.population[i].copy() for i in order[:g["elite_count"]]]
        immigrants = min(int(g["population_size"] * g["immigrant_rate"]),
                         g["population_size"] - g["elite_count"])
        while len(offspring) < g["population_size"] - immigrants:
            a = self.population[tournament(fitness, self.rngs[1], g["tournament_size"])]
            b = self.population[tournament(fitness, self.rngs[1], g["tournament_size"])]
            child = (crossover(a, b, self.rngs[2], g["crossover_type"])
                     if self.rngs[2].random() < g["crossover_probability"] else a.copy())
            offspring.append(mutate(child, self.rngs[3], g["mutation_probability"], g["mutation_sigma"]))
        offspring.extend(initialize_chromosome(self.rngs[0]) for _ in range(immigrants))
        return np.stack(offspring)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"config": self.cfg, "seed": self.seed, "generation": self.generation,
                    "optimization_steps": self.optimization_steps, "selection_steps": self.selection_steps,
                    "elapsed": self.elapsed, "champion_validation": self.champion_validation,
                    "initial_return": self.initial_return, "history": self.history,
                    "rng_states": [r.bit_generator.state for r in self.rngs]}
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, population=self.population, champion=self.champion,
                                metadata=np.array(json.dumps(metadata, allow_nan=False)))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            ga = cls(meta["config"], meta["seed"])
            ga.population, ga.champion = data["population"].copy(), data["champion"].copy()
        for key in ("generation", "optimization_steps", "selection_steps", "elapsed", "champion_validation",
                    "initial_return", "history"):
            setattr(ga, key, meta[key])
        for rng, state in zip(ga.rngs, meta["rng_states"], strict=True):
            rng.bit_generator.state = state
        return ga

    def train(self, budget: int, directory: Path, stop_after: int | None = None) -> GAPolicyNetwork:
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        prior_elapsed = self.elapsed
        policy = GAPolicyNetwork()
        if self.initial_return is None:
            policy.set_flat_parameters(self.champion)
            initial = evaluate(policy, self.cfg, episode_seeds(self.cfg, "validation"))
            self.selection_steps += int(initial.length.sum())
            self.initial_return = float(initial["return"].mean())
            self.champion_validation = self.initial_return
            save_policy(policy, directory / "initial.npz")
        g = self.cfg["ga"]
        limit = stop_after or g["max_generations"]
        while budget - self.optimization_steps >= self.generation_reserve:
            if limit is not None and self.generation >= limit:
                break
            seeds = [training_seed(self.seed, 0, self.generation * g["fitness_episodes"] + k)
                     for k in range(g["fitness_episodes"])]
            fitness = []
            for index, chromosome in enumerate(self.population):
                policy.set_flat_parameters(chromosome)
                episodes = evaluate(policy, self.cfg, seeds)
                self.optimization_steps += int(episodes.length.sum())
                fitness.append(float(episodes["return"].mean()))
                episodes["generation"], episodes["candidate"] = self.generation + 1, index
                append_csv(directory / "fitness_episodes.csv", episodes.to_dict("records"))
            fitness = np.asarray(fitness)
            if not np.isfinite(fitness).all():
                raise FloatingPointError("Non-finite GA fitness")
            policy.set_flat_parameters(self.population[int(fitness.argmax())])
            val = evaluate(policy, self.cfg, episode_seeds(self.cfg, "validation"))
            self.selection_steps += int(val.length.sum())
            score = float(val["return"].mean())
            if score > self.champion_validation:
                self.champion_validation, self.champion = score, policy.get_flat_parameters()
            self.generation += 1
            self.elapsed = prior_elapsed + time.perf_counter() - started
            self.history.append({"generation": self.generation, "optimization_steps": self.optimization_steps,
                                 "selection_steps": self.selection_steps, "best": float(fitness.max()),
                                 "mean": float(fitness.mean()), "median": float(np.median(fitness)),
                                 "worst": float(fitness.min()), "std": float(fitness.std()),
                                 "diversity": float(self.population.std(axis=0).mean()),
                                 "validation_return": score, "selected_validation_return": self.champion_validation,
                                 "elapsed_seconds": self.elapsed})
            self.population = self.breed(fitness)
            self.save(directory / "checkpoint.npz")
            print(f"  GA seed={self.seed} generation={self.generation} steps={self.optimization_steps} "
                  f"fitness={fitness.max():.1f} validation={score:.1f}", flush=True)
        if self.generation == 0:
            raise ValueError(f"GA allocation {budget} cannot reserve one generation ({self.generation_reserve})")
        self.elapsed = prior_elapsed + time.perf_counter() - started
        self.save(directory / "checkpoint.npz")
        policy.set_flat_parameters(self.champion)
        save_policy(policy, directory / "selected.npz")
        import pandas as pd
        pd.DataFrame(self.history).to_csv(directory / "generations.csv", index=False)
        return policy
