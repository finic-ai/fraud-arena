# Fraud Arena

Benchmarks of agents (Claude Code, Codex, ...) that build and keep updating fraud detection at one institution of
[Fraud World](https://github.com/finic-ai/runmere/tree/feat/fraud-world/worlds/fraud-world), a world of the Runmere
engine.

One task, two phases:

1. **Static**: the defender builds and redeploys a detection stack (rules, logistic regression, XGBoost / LightGBM,
   the Grayson decision model) once per simulated period, against Fraud World's scripted fraudsters.
2. **Adversarial**: the same task against agent ring operators, who set their ring's strategy once per period from
   the world's menu and see only their own attempts' outcomes.

Each episode warms the world up, checkpoints it, and forks a candidate run (the agent's bundle) and a reference run
(the institution's built-in rules) from the same state; the score is the difference: fraud losses prevented minus
the cost of friction to good customers, with displacement to other institutions reported. Design:
[docs/design/20-adversarial-benchmarks.md](https://github.com/finic-ai/runmere/blob/docs/fraud-arena-adversarial-plan/docs/design/20-adversarial-benchmarks.md).
