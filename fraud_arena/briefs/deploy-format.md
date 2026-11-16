# bundle/policy.yaml

```yaml
format: 1
rules:                       # ordered; for each decision the first rule whose `when` holds gives the action
  - {decision: brind_card_auth, when: "score.fraud > 0.9", action: decline}
  - {decision: brind_card_auth, when: "score.fraud > 0.6 and obs.amount > 500", action: step_up}
  - {decision: brind_card_auth, action: approve}           # a rule without `when` always holds
models:
  fraud: {file: models/fraud.json}                         # written by fraud_arena.modeltools, or inline:
  velocity: {kind: logistic, features: [txns_1h, new_device], weights: [0.8, 1.9], intercept: -4.2}
```

- A decision with rules in your policy and none that holds takes its default action. A decision of your role that
  you don't mention keeps the institution's built-in rule.
- `when` reads the decision's observation features as `obs.<feature>`, model scores as `score.<model>` (a float),
  and `now`. Operators: `and`, `or`, `not`, comparisons, arithmetic, `in ('a', 'b')`, `coalesce(x, y)`. It can't read
  entities' attributes, histories or anything outside the observation: build what you need into a model.
- A feature can be missing (`none`) for some decisions. Comparing a missing value with a number is an error at run
  time, so wrap features that can be missing: `coalesce(obs.name_match, 1) < 0.5`. A bundle that fails at run time is
  taken back (your previous bundle, or the built-in rules, decides) and you are told why at the next boundary.
- Models read observation features only (numbers, money in major units, bools as 1/0; a missing value is missing):
  - `linear` / `logistic`: `features`, `weights` (one per feature), `intercept`.
  - `trees`: an XGBoost JSON dump (`Booster.get_dump(dump_format='json')`), `base_score` as a margin, `sigmoid`.
- Export trained models with `from fraud_arena.modeltools import from_xgboost, from_sklearn_linear, save` (LightGBM:
  convert its `<=` splits; see the module docstring). Train on the columns that match the decision's features: the
  `decision_points` tool lists each decision's features, and the tables hold the same quantities.
- The policy is checked when it is deployed; an error comes back to you at the next boundary and your previous bundle
  stays in place. Validate locally first: every rule's decision must be one of `decision_points`, every action one of
  that decision's actions, every feature one of its features.
