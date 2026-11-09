# Algorithms and implementation choices

## Genetic algorithm

Each chromosome encodes `[W1,b1,W2,b2,W3,b3]`. Each layer samples U(-1/sqrt(fan_in),
1/sqrt(fan_in)), matching ordinary PyTorch linear initialization bounds. Fitness averages
complete deterministic episodes. A generation uses common seeds; the next generation uses
new seeds. Elites are copied, tournaments sample without replacement, crossover chooses genes
independently, and masked Gaussian mutation creates new arrays. Mean per-gene standard deviation
measures diversity. Optional immigrants are disabled in the primary study.

Generation winners compete against the initial actor and earlier winners on validation seeds.
Equal scores retain the earlier checkpoint. The selected champion can therefore differ from
the current-generation fitness winner. Final-test seeds never affect selection.

Before evaluating a generation, reserve `population * K * maximum_episode_length` transitions.
This conservative approach avoids biased incomplete candidate evaluations. Pure GA may leave
more allocation unused than PPO; the report makes that limitation explicit. Hybrid allocation
uses **actual** consumed GA transitions, so unused evolutionary allocation returns to PPO.

## PPO

Stable-Baselines3 implements the clipped policy objective, generalized advantage estimation,
value regression and entropy regularization. The separate actor and critic use two 64-unit
Tanh layers with `ortho_init=False`. No normalization or custom reward shaping is applied.
`DummyVecEnv` serially coordinates several environments. Its four-item step API is handled
inside SB3, while the unified evaluator uses Gymnasium's five-item API.

Training rounds down to complete rollout batches. A wrapper counts actual steps independently
of `model.num_timesteps`; equality is asserted. A subclass hook logs losses and evaluates
checkpoints after `PPO.train()`, avoiding pre-update checkpoint labels. TensorBoard and CSV
records contain losses, entropy, KL, clip fraction, explained variance and learning rate.

## Hybrid

Transfer maps GA layers 0, 2, 4 to SB3 policy-net layers 0, 2 and action head. Every weight and
bias is checked. Random observation batches and real reset states must yield equivalent logits
and deterministic actions. Critic tensors remain unchanged and the fresh optimizer has no state.
Validation returns before/after transfer must also agree. PPO must actually change actor tensors.

GA's argmax actions and PPO's stochastic training actions differ even with identical logits.
This can change early behavior sharply; it is a methodological limitation, not a transfer bug.

## Checkpoints

GA NPZ files include only numeric/string arrays with JSON metadata and are loaded with
`allow_pickle=False`. Saving uses a temporary file and atomic replacement. Continuation restores
all four RNG states and the next population. SB3 ZIP checkpoints include optimizer state;
only trusted checkpoints should be loaded. Exact in-episode PPO physics continuation is not promised.
