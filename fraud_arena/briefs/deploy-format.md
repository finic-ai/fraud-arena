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

## Grayson at a gate (optional)

A decision your bundle can't decide well by rule may wait for Grayson: route it with `action: external` and give the
question. The case waits (as a held case does) until Grayson answers at the end of that simulated day, or takes the
decision's default after `timeout`. Works for held cases and reviews (state gates), not for instant ones like card
authorizations. Grayson sees the decision's observation features.

```yaml
rules:
  - {decision: brind_account_open_review, when: "obs.id_match < 0.8", action: external}
  - {decision: brind_account_open_review, action: approve}
external:
  brind_account_open_review:
    timeout: 2d
    grayson:
      instructions: "Decide whether to open this account. Synthetic and stolen identities are common here."
      criteria: {approve: "a real person applying for themselves", decline: "a synthetic or stolen identity"}
```

Or use Grayson's answers as features: ask several questions per case and combine Grayson's answer probabilities with
the case's own features in a logistic layer you fit offline (for example on Grayson answers for your labelled cases,
as in stacking XGBoost with a decision model), with score bands mapped to actions:

```yaml
external:
  brind_account_open_review:
    timeout: 2d
    grayson:
      questions:
        synthetic: {instructions: "Is this a synthetic or stolen identity?", criteria: {yes: "likely", no: "unlikely"}}
        story: {instructions: "Do the occupation, income and purpose fit together?", criteria: {yes: "they fit", no: "they don't"}}
      combine:
        intercept: -2.1
        weights: {synthetic.yes: 3.4, story.no: 1.2, id_match: -2.0}   # "<question>.<option>" or a feature name
        bands: [{min: 0.8, action: decline}, {min: 0.4, action: review}]
        otherwise: approve
```

## Grayson working your review queue (optional)

Every case your rules hold (`hold`, `review`, ...) waits in `REVIEW_QUEUE` until a verdict or its due time. Grayson
can work the queue for you: each simulated day it gets the pending cases (their `REVIEW_QUEUE` fields, nothing else)
and its verdict (`release` or `reject`) is applied. Use either shape above; the actions are `release` and `reject`.
It doesn't use your human team's capacity; you can still review cases yourself in `reviews.json`.

```yaml
review_queue:
  kinds: [p2p_hold, wire_hold, account_application]   # optional: only these case kinds
  grayson:
    instructions: "Release unless this looks like fraud or a scam in progress."
    criteria: {release: "genuine", reject: "fraud or a scam"}
```
