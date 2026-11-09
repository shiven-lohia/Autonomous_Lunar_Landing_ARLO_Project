"""The two stages of GA-initialized PPO and their audit trail."""

from pathlib import Path

import numpy as np

from .genetic_algorithm import GeneticAlgorithm
from .io_utils import write_json
from .ppo_agent import make_ppo, train_ppo
from .weight_transfer import transfer_actor


def train_hybrid(cfg: dict, seed: int, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    total = cfg["experiment"]["total_optimization_steps"]
    ga = GeneticAlgorithm(cfg, seed)
    actor = ga.train(int(total * cfg["hybrid"]["ga_budget_fraction"]), directory / "ga")
    ppo_dir = directory / "ppo"
    ppo_dir.mkdir(exist_ok=True)
    model = make_ppo(cfg, seed, ppo_dir)
    try:
        transfer = transfer_actor(actor, model, cfg)
        before = np.concatenate([p.detach().cpu().numpy().ravel().copy()
                                 for p in model.policy.mlp_extractor.policy_net.parameters()])
        state = train_ppo(model, cfg, total - ga.optimization_steps, ppo_dir,
                          ga.optimization_steps, ga.selection_steps, ga.elapsed)
        after = np.concatenate([p.detach().cpu().numpy().ravel()
                                for p in model.policy.mlp_extractor.policy_net.parameters()])
        transfer["actor_changed_after_ppo"] = bool(np.any(before != after))
        if not transfer["actor_changed_after_ppo"]:
            raise AssertionError("PPO did not update the transferred actor")
        transfer.update({"ga_initial_validation": ga.initial_return,
                         "ga_selected_validation": ga.champion_validation,
                         "post_transfer_validation": state["initial_return"],
                         "ga_generations": ga.generation, "ga_seed": seed, "ppo_seed": seed,
                         "ga_source": "ga/selected.npz"})
        if abs(ga.champion_validation - state["initial_return"]) > 1e-5:
            raise AssertionError("GA and transferred PPO returns differ on identical validation seeds")
        write_json(directory / "transfer.json", transfer)
        return ga, state
    finally:
        model.get_env().close()
