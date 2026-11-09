# System architecture

```mermaid
flowchart LR
    C[YAML configuration] --> E[Experiment runner]
    E --> G[Genetic search]
    E --> P[PPO baseline]
    G --> V[Validation-selected GA actor]
    V --> T[Copy 6 actor tensors and verify logits]
    T --> H[PPO fine-tuning with fresh critic]
    H --> S[Derive PPO-short interaction budget]
    E --> R[Random baseline]
    V --> Q[Unified held-out evaluation]
    P --> Q
    H --> Q
    S --> Q
    R --> Q
    Q --> D[CSV rows and JSON ledgers]
    D --> A[Seed-level statistics]
    A --> F[PNG/SVG plots and PDF/TeX/Markdown report]
```

`config.py` validates dimensions and seed separation. `environment.py` owns Gymnasium
construction and terminal classification. `policy.py` owns the 4,996-scalar representation.
`genetic_algorithm.py` has independent initialization, parent-selection, crossover and mutation
RNGs. `weight_transfer.py` maps actor layers and verifies unchanged critic tensors.
`ppo_agent.py` instruments SB3 **after** each optimizer update, so validation is correctly
attributed to the updated model. `experiment.py` tracks allocations and selected checkpoint paths.

Every completed suite records resolved settings and machine/package provenance. Agent directories
are identified by method and training seed. All analyses require an explicit suite path.
