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
It doesn't use your human team's capacity; you can still review cases yourself in `reviews.json`. Grayson looks at the queue every simulated hour and sees each case's `REVIEW_QUEUE` columns. For more context (the customer's history, notes, payees), work the queue from `workflow.py`'s `review(cases)` instead and send Grayson whatever case file you build.

```yaml
review_queue:
  kinds: [p2p, wire, consumer_wire, application]   # optional: only these REVIEW_QUEUE.CASE_TYPE kinds (lower case)
  grayson:
    instructions: "Release unless this looks like fraud or a scam in progress."
    criteria: {release: "genuine", reject: "fraud or a scam"}
```

## Integrations: data and Grayson inside your rules and models (optional)

Like a fraud team's decision platform, your decision points can call data sources when a case arrives, and your
rules and models can use what they return. Declare them in `policy.yaml`:

```yaml
integrations:
  claims_90d:                         # your institution's own records, live: as of the minute the case arrives
    source: own_data
    table: CLAIMS
    where: {CUST_ID: $cust, CLAIM_TS: {">=": "$now - 90d"}}   # a value, a list, or bounds (>=, >, <=, <); $now - 90d/12h/30m
    params: {cust: P2P.CUST_ID}       # a case identifier (<TABLE>.<COLUMN>) or feature.<name>
    sql: "select count(*) from rows"  # optional: SQL over the rows found (named `rows`); without it, the first cell
    default: 0                        # when nothing is found or the lookup fails
  customer_history:                   # SQL over all your tables as of the end of the previous simulated day
    source: own_data
    sql: "select count(*) from CLAIMS where CUST_ID = $cust and CLAIM_TS > $now - interval 90 day"
    params: {cust: P2P.CUST_ID}
  payee_site:                         # the public web: your PUBLIC_* tables (ads, domains, websites, block explorer)
    source: public
    sql: "select * from PUBLIC_WEBSITES where DOMAIN = $d"
    params: {d: feature.payee_domain}
  scam_story:                         # Grayson, given the case and other integrations' rows as its data
    source: grayson
    instructions: "Is this payment part of a scam the customer is being coached through?"
    criteria: {yes: "a scam in progress", no: "an ordinary payment"}
    data: [case, claims_90d, payee_site]
models:
  stack: {kind: logistic, features: [amount, outside.claims_90d, outside.scam_story.yes], weights: [...], intercept: ...}
rules:
  - {decision: brind_payment_hold, when: "obs.amount < 200", action: release}          # never asks anything
  - {decision: brind_payment_hold, when: "score.stack > 0.8", action: hold}
  - {decision: brind_payment_hold, when: "obs.amount > 2000 and outside.scam_story.yes > 0.7", action: hold}
  - {decision: brind_payment_hold, action: release}
```

- A lookup gives one number (the first column of the first row; true/false count as 1/0): `outside.<name>`. A
  Grayson integration gives a probability per option: `outside.<name>.<option>`. Use them in rules and as model
  features (fit your models offline on the same lookups and Grayson answers for past cases).
- Nothing is asked until a rule or model reaches it: put cheap rules first, and a case decided earlier costs nothing.
  Grayson calls count against your budget; when Grayson can't answer, the options are equal (no information).
- A `table` lookup reads the table live, exactly as it stands when the case arrives (fraud reports still arrive as
  late as they do), filtered on its own columns, in about a tenth of a millisecond: use it for per-case history. A
  lookup with only `sql` reads all your tables as of the end of the previous simulated day. Up to 50 rows returned.
- A case's identifiers are the ids its row shows in your tables (e.g. `P2P.CUST_ID`, `P2P.ACCT_ID`); see a pending
  case's `identifiers`. Grayson sees the case's features and identifiers, not its free text until that text is in
  your tables.
- What live Grayson was sent and answered is in `.arena/grayson-live.jsonl` at the next boundary.
