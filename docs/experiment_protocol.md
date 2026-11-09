# Experimental protocol

The main benchmark fixes LunarLander-v3 discrete control, no wind, default gravity, original
rewards, raw observations, a 1,000-step limit and common actor dimensions.

Training seeds are 11, 23 and 37. Validation starts are 20,000 onward; primary final starts
are 40,000 onward. Transfer verification uses resets 30,000-30,007 and consumes no transitions.
Exploratory sensitivity evaluation uses 60,000 onward. Optimization environment seeds are
`1_000_000 + experiment_seed*500_000 + stream*100_000 + index`. GA training uses stream 0;
PPO vector resets start in stream 1. Candidate and PPO action sampling have separate seeded RNGs.

Each main trained agent has a 1,000,000-step ceiling. The hybrid reserves 25% for GA and
uses actual remaining allocation for PPO. PPO-short receives exactly its realized PPO stage
budget. Pure GA reserves worst-case complete generations; PPO rounds down to complete
rollout batches. Report unused allocation, rather than claiming identical realized budgets.

Keep five interaction categories: GA fitness, GA selection, PPO training, PPO validation,
and final evaluation. The first and third form optimization cost. Selection and evaluation
are overhead. Hybrid transfer verifies reset observations with zero `step` calls. Gameplay and
robustness consume additional documented evaluation interactions outside main-run ledgers.

Every model is selected on validation mean, retaining the earliest checkpoint in a tie.
No final-test value is used to select a model. Main reported performance is that of the
validation-selected checkpoint; the last update's checkpoint is retained separately.

For uncertainty, aggregate test returns within a training seed, then resample those seed means.
Use paired seed-level differences between hybrid and controls. Three seeds produce preliminary
bootstrap intervals; many evaluation episodes cannot replace independent training runs.
The random baseline is repeated identically for each comparison seed and has zero training
replications. Its interval is omitted. Plotted seed SD bands are not confidence intervals.

Fitness curves use changing training seeds and may be noisy. Learning curves interpolate only
for display inside the common observed range. Threshold crossings use actual recorded validation
points; no crossing is invented between evaluations. Report absent crossings as not reached.

The reduced sweep is exploratory: one seed, 160,000-step ceiling, one fitness episode and
population 10 except the factor being varied. Use its separate development hold-out to examine
trends; confirm them with the full sweep before claiming general sensitivity results.
