# Course demonstration and viva guide

1. Open `report/full/report.pdf`. State the hypothesis: does a useful evolved initialization justify
   the interactions spent finding it? Do not assume the hybrid should win.
2. Show the architecture figure: eight observations, two Tanh hidden layers, four action logits.
   Explain the 4,996-scalar chromosome and separate PPO critic.
3. Run `.\.venv\Scripts\python.exe -m scripts.doctor`. It verifies a real Box2D episode.
4. Open a hybrid `transfer.json`. Show exact logit agreement, unchanged critic, matching validation
   returns, and changed actor weights after PPO. This distinguishes real transfer from relabeling.
5. Show `analysis/comparison.csv`, `plots/learning_curves.png` and `plots/interaction_accounting.png`.
   Explain equal allocation ceilings, actual step counts, leftover budgets and evaluation overhead.
6. Play the median-selection MP4s. MP4s are not committed to Git; on a fresh clone, retrain
   the full suite (or one seed) and record with `scripts.record_gameplay` first. Explain that return >=200 and a resting physical landing are
   different measurements. The overlay records selection method and episode seed.
7. Show PPO-short. It isolates the value of the initialization when gradient-training time is equal.
8. Show the actual sensitivity and robustness results, while stating their experimental scale.
9. Run the tests or the smoke suite if time permits. A smoke run verifies the pipeline; it is not
   evidence of learning convergence.

Likely questions:

- **Why GA plus PPO?** GA explores behavior through black-box fitness; PPO uses gradient information
  for refinement. Their combination may help or waste budget, which the experiment tests.
- **Why can crossover hurt?** Corresponding coordinates need not encode corresponding features;
  mixing independently learned hidden units can destroy useful structures.
- **Why actor only?** A separate fresh critic keeps the experiment focused on policy initialization.
- **Why a PPO-short control?** It matches the hybrid's gradient-training interactions.
- **Why not use test scores to select models?** That would bias final generalization estimates.
- **Why can a hybrid lose after transfer?** Stochastic action sampling and an untrained critic can
  change behavior and advantage estimates despite exact deterministic forward equivalence.
- **Why only preliminary confidence with three seeds?** Training outcomes vary; 100 episodes from
  one model do not provide 100 independently trained models.
- **Is this the published ERL algorithm?** No. This is two-stage GA-initialized PPO, without replay
  sharing or periodic gradient-policy insertion into an evolutionary population.

Before submission, fill in your name, institution, course section and instructor if required.
Read the final measured findings and the limitations; the report generator does not invent a win.
The measured headline: PPO 268.4, PPO-short 256.2, GA-PPO 244.1, GA -49.5, random -178.3. The hybrid
lost to PPO on all three seeds, so present it as a well-controlled negative result for the hybrid hypothesis.
