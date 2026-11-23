# Fraud Arena investigation tasks (historical, offline)

Two analyst tasks on a frozen simulation of Runmere's Fraud World, seen from Brindlemark Bank:

- **Task A, alert review.** One alerted customer per case. Decide fraud or not, the typology, the customer's role,
  a calibrated confidence and an action.
- **Task B, incident investigation.** An anomaly yields a few seed customers. Decide whether it is a real incident
  (one ring crew running one kind of scheme, or one fraudster's run of victims) or a benign cluster; if real, find
  every Brindlemark customer in it within the window and give each one's role.

Players are headless Claude Code sessions (or the non-LLM baseline) with one tool, `casefile`, which only ever shows the
bank's own tables as of the case time. Answers are graded against the simulator's truth.

## Format

The benchmark is fully public, in the style of tau-bench and FrontierSWE:

- **Task pages**: one per task family (`publish` writes their data), with every case's inputs, the answer keys of
  published sets, and example traces (every `casefile` command and the final answer) from runs.
- **Seed sets** (`task-sets.json`): `dev` (seeds 11, 12) for development, examples and answers; one `official-<season>`
  set per season (e.g. `official-2027s1`: 201, 202, 203), the leaderboard set, chosen per season, with its answers
  published after the season. Every set uses the same world, scale (8000 roots, 120 days) and case counts (40 alerts,
  12 incidents of which up to 4 are benign clusters). A new season is new seeds of the same world.
- **k trials** per case (`--trials K`, a fresh workspace each), scored as mean@K with the worst@K / best@K range,
  pass^K on the verdict, and a 95% bootstrap confidence interval over cases (Scoring below).
- **Harbor task folders** (`export-harbor`): each case as a self-contained task (instruction, environment with the
  bank tables as of the case time and the tool, verifier with the answer key, reference solution).

```bash
python run_tasks.py build --split dev --seed 11 --out cases/dev-s11      # ~5 min simulation + ~30 s build
python run_tasks.py run --task alert --cases cases/dev-s11 --model sonnet --trials 3 --out runs/dev-s11-sonnet
python run_tasks.py run --task incident --cases cases/dev-s11 --model baseline --out runs/dev-s11-baseline
python run_tasks.py score --runs runs/dev-s11-sonnet runs/dev-s11-baseline
python run_tasks.py leaderboard --runs runs/* --out site-data/leaderboard.json
python run_tasks.py publish --split dev --cases cases/dev-s11 cases/dev-s12 --runs runs/* \
    --examples dev-s11/A006,dev-s11/B002 --out site-data
python run_tasks.py export-harbor --cases cases/dev-s11 --out harbor/dev-s11
python tools/check_domains.py cases/dev-s11 cases/dev-s12
```

`build` checks that the seed belongs to the named set (`--allow-any-seed` for scratch builds) and reuses an existing
simulation in `cases/NAME/truth/` when seed/roots/days match (`--resimulate` to run it again). Builds are
deterministic for a given simulation. `run` copies each case workspace into the run folder once per trial, runs four
in parallel (`--parallel`), and keeps `answer.json`, the command log and the Claude transcript per trial.

## World

`worlds/fraud-world-tasks.yaml` is `worlds/fraud-world-arena.yaml` (the 1v1 world) with two differences:
`fraud_density` is back at the world default of 0.62, so independent fraudsters (ATO, card, scams, first-party,
mules...) act alongside the ten ring crews (R1-R10); and ring R1 runs a direct-deposit sign-up bonus churn: each
identity opens a checking account, receives two or three biweekly payroll-looking credits from one of the crew's
rotating fake employers (some under real small employers' names), collects the sign-up bonus, and the balance leaves
by ACH within days to one of the crew's collection accounts at Dovrex or a prepaid card; referrals are occasional and
the openers use their own varied email addresses. No defender or attacker policy is deployed: the bank runs its
built-in rules and the rings their scripts.

## Layout

```
task-sets.json                     the seed sets (dev, official-<season>) and case counts
cases/NAME/world.json              seed, roots, days
cases/NAME/bank/<TABLE>.parquet    Brindlemark's observer tables + LOAD_TS, end-state columns masked, exported (below)
cases/NAME/truth/                  simulator truth (write_truth), raw observer output, rows.parquet, entity maps,
                                   export.json (amount factors, renames), keys/<CASE>.json answer keys
cases/NAME/alert/A001/ ...         workspaces: case.json, brief.md, notes.md, drafts/
cases/NAME/incident/B001/ ...
cases/NAME/index.json              split, seed, case list with template ids, build summary
runs/NAME/<task>/<CASE>/trial-<k>/ the player's copy of the workspace: answer.json, log.jsonl (every command)
runs/NAME/<task>/<CASE>/trial-<k>.claude.json   Claude Code's JSON result (turns, cost, final message)
runs/NAME/<task>/run.json, score.json
site-data/                         leaderboard.json, alert.json, incident.json (publish)
harbor/NAME/<set>-<case>/          Harbor task folders (export-harbor)
```

`cases/`, `runs/`, `site-data/` and `harbor/` are git-ignored (they are generated).

### As-of view and leak control

- Every bank row gets `LOAD_TS`: the time the row reached the bank (`row_links.load_time`). Rows about entities
  (customers, accounts, cards, devices, beneficiaries) are written once in their latest version, so their `LOAD_TS` is
  the entity's creation time instead.
- `casefile` loads only rows with `LOAD_TS <= case_time` into an in-memory DuckDB, then switches off file access and
  locks the configuration, so player SQL cannot read any file (truth included). Harbor images hold only those rows.
- Columns that hold an end-of-run state are masked in `bank/`: account `STATUS`, `RESTRICTED_FLG`, `BALANCE`; card
  `STATUS`; dispute `MERCHANT_WON`; review-queue decision columns; ACH `STATUS = RETURNED` reads `POSTED` (returns are
  in `ACH_RETURNS` once loaded); a posted check's `RETURN_REASON` is cleared.
- Known gap: `CUSTOMERS` is the latest profile, so a contact change made after the case time can show (the change
  itself, in `PROFILE_CHANGES`, stays hidden until it loads).

### Export (`arena/tasks/export.py`)

After `materialize`, the bank tables are rewritten deterministically per seed; the world's logic is untouched.

**Contact details.** The world writes reserved names. Each address maps to a real-looking one, by a pure function of
the address, so it is identical in every table (`CUSTOMERS.EMAIL`, `P2P.CP_HANDLE`, `PAYEES.PAYEE_HANDLE`,
`PROFILE_CHANGES.NEW_EMAIL`, `TREASURY_USERS.EMAIL`, `REVIEW_QUEUE.CP_HANDLE`, `BUSINESS_APPLICATIONS.WEBSITE`) and in
answer keys and anomaly texts:

| World domain | What the world means | Exported as |
|---|---|---|
| example.com | the largest free provider, more among the young | gmail.com |
| example.net | an older-skewed portal | yahoo.com 70%, aol.com 30% |
| example.org | a mid-age webmail | outlook.com 55%, hotmail.com 45% |
| inbox.example | a device maker's mail, young-skewed | icloud.com |
| isp.example | an internet provider's mail, older people | comcast.net 35%, att.net 25%, verizon.net 20%, sbcglobal.net 20% |
| privatemail.example | a privacy provider | proton.me 70%, tutanota.com 30% |
| tempbox.example | a disposable-address service | guerrillamail.com 60%, 10minutemail.com 40% |
| employers' `*.example.com/org` | company mail | invented domains (`COMPANY_DOMAINS`), e.g. hessmark-mfg-works.com |
| generated business sites | company websites | `<slug>-bizhq.<tld>` |

The split between two providers is drawn from a hash of the address. Personal local parts get a 4-6 digit suffix from
a hash of the address (`omar.jackson` -> `omar.jackson58213@yahoo.com`), so the addresses very likely belong to no
real person. Addresses a crew generates from one shared template keep a shared look (their suffix comes from the
template: `karenking18`, `dcampbell17` -> `karenking98918@gmail.com`, `dcampbell98917@gmail.com`). Company domains
were checked to be unregistered (`tools/check_domains.py`: DNS NS/SOA and whois; 34 domains free in dev).

**Per-seed variation.** Every fraud scheme's amounts are scaled by a factor drawn per (seed, scheme) in 0.6-1.7
(never within 12% of 1), keeping round-number granularity; derived rows (returns, disputes, claims, tags, review
queue) get the same values. Structuring keeps its amounts (it is defined by the cash-report threshold). Counterparties
that only fraud uses (fake employers, mule shells) get names and entry descriptions drawn per seed from a pool of
payroll, gig-payout, loan and settlement names. `truth/export.json` records the factors and renames.

## Truth: roles and typologies (`arena/tasks/truth.py`, `schema.py`)

Every Brindlemark row with a customer is joined to its truth event and cause (`cause_scenario`, `cause_instance`,
`cause_label`, actor). An *involvement* is one fraud scenario instance touching one customer. How the customer took
part decides the role:

| Situation | Role |
|---|---|
| a stolen or synthetic identity opened by a fraudster (any scenario on it) | perpetrator; typology = the scheme it was opened for |
| first-party scenario by the customer's own person (friendly fraud, refund abuse, refund service, first-party ACH return, courier fraud, structuring...) | perpetrator |
| a mule scenario, or a real customer whose account a ring took over as a mule (account `ring` set) and that moved ring money | mule |
| a fraudster acted on the customer's account or card (ATO, card testing / CNP, vishing for card details, R7's pulls) | victim |
| a scam in which the customer paid (romance, pig butchering, impersonation, tech support, grandparent...) | victim |
| only *received* money from a fraud, nothing else | incidental: excluded from sampling |

Scenario to typology family (full map in `schema.py`): ATO scenarios -> `ato`; card testing, CNP, vishing card
details, triangulation -> `card_compromise`; authorised scams, diverted payroll, BEC -> `scam_victim`; mule flows ->
`mule`; synthetic bust-out, new-account ID theft, ACH-return and ACATS fraud -> `synthetic_identity`; R1 bonus
churning, R10 promo farm -> `promo_abuse`; R5 check kite -> `check_fraud`; structuring -> `other`. A customer with
several involvements takes the highest-priority role (perpetrator > mule > victim), earliest first.

## Templates

Every case has a template id (in `index.json` and the answer key; never in the player's workspace):

- **Alert templates** `alert/<monitor>/<scheme>`: the monitor that fired and the customer's scheme (`<ring>-<typology>`,
  `indep-<scenario>` for an independent fraudster, or `benign`).
- **Incident templates** `incident/<ring>-<typology>` (a ring crew's customers in one typology: e.g. R1 bonus churning,
  R2 vishing for card details and wallet provisioning, R5 synthetic identities, R6 card testing, R7 ACH pulls from
  breached logins, R9 elder scams, R10 refund service and farmed accounts, R1/R3/R4 mule recruits, R4 romance and
  pig-butchering victims), `incident/indep-<scenario>` (one independent fraudster's run, grouped by the fraudster who
  acted), and `incident/benign-<kind>` (a cluster of untouched customers on a shared IP, employer payroll, merchant,
  payee, address or device).

All templates go through the same rules: the export variation above, and the seeding rules below.

## Task A: alert review

**Alerts** come from 15 bank-side monitors (`monitors.py`), written in SQL over `bank/` only: card disputes (fraud /
non-receipt), P2P unauthorized claims, ACH returns, ACH callbacks, P2P holds, wire callbacks / large wires, card
step-ups, referral-reward velocity, new-device contact change, risky logins (hosting/VPN/proxy, SIM change, remote
session), referred or high-risk applications, P2P pass-through, near-threshold cash, fraud-report calls. The bank's
own payment-rule outcomes (holds, callbacks, step-ups) are read from the bank's tables, not from truth decisions.
An alert fires at the triggering row's `LOAD_TS`; the case time is 24 h later.

**Sampling.** Fraud alerts: the customer's fraud had started by the case time and at least one of its rows had
loaded by then. Benign alerts: customers the world never touched at all (no fraud involvement, no ring marks, real
identity). Fraud cases are spread over typologies; benign cases are drawn first from the same monitors as the fraud
ones (look-alikes, e.g. honest "mistaken fraud claim" disputes next to real card compromise), one case per customer.
The fraud share (default 45%) is stated in every brief.

**Answer**: `disposition`, `typology` (ato, scam_victim, mule, synthetic_identity, first_party, card_compromise,
promo_abuse, check_fraud, other, none), `customer_role` (perpetrator, victim, mule, legitimate), `confidence`,
`recommended_action`, `rationale`.

## Task B: incident investigation

**Truth groups.** Customers resolved as above, grouped by ring crew and typology or, for independent fraudsters, by
scenario and the fraudster who acted. Groups are cut into waves of at most 12 members within a window drawn per group
from 21/28/42 days. Each group gives at most one case per seed, and appears in at most 2 builds of a seed set
(`builds_per_group`; the builds are picked by a hash of template and seed), so no scheme dominates a season.

**Seeding (drawn per case).** The case time is when a drawn 60-90% of the wave's members are visible, plus 1-4 days;
the window starts a day before the first member's fraud. The seed key's kind is drawn uniformly among the kinds that
link 2+ visible members (device, IP, merchant, payee, ACH originator, referral, address, P2P sender; keys shared by at
most 40 customers bank-wide), and the seed is the key's first 2..min(5, half the members) holders. Up to four draws
are tried per wave, keeping the first with members to find. When no key leads further (typical of one crew's
victims), the seed comes from the monitor that fired on most members in the window, and the anomaly says how many
customers fired it overall.

**Findability.** A member counts toward recall (`to_find`) only if `casefile links` reaches it from the seed: over the
anomaly's own key, or over keys shared by at most 12 customers, following members only, within 3 hops (for monitor
seeds: it fired the same monitor in the window). Unreachable members (`unreachable` in the key), the same crew's
customers outside the window, and members not yet visible are neutral. A real incident with nothing findable is kept
as a verdict-only case (`expansion: none`): scored on verdict and typology.

**Benign clusters.** At case times drawn from the real incidents, keys (IP, ACH originator, merchant, address, payee,
device) shared by 4-30 customers, 3+ of them active in the last week, *none* of whom the world ever touched: carrier
or office IPs, an employer's payroll, a popular merchant or payee, a household. Seeds have the same sizes (2-4) and
the anomaly text the same form as real ones.

**Answer**: `is_incident`, `typology`, `members` [{cust_id, role}], `link_keys`, `summary`, `recommended_actions`.

## Scoring (`arena/tasks/score.py`)

**Per case and trial** (0-1):
- Alert: 0.6 disposition + 0.2 typology + 0.2 role (benign cases: not_fraud / none / legitimate). Missing answer: 0.
- Incident, real with members to find: 0.3 verdict + 0.1 typology + 0.6 member F1 (F1 of the members added beyond
  the seed, neutral members ignored; 0 when the verdict is wrong).
- Incident, verdict-only (`expansion: none`): 0.75 verdict + 0.25 typology.
- Benign cluster: 0 if called an incident; else 1 minus 0.06 per customer named beyond the seed (at most 0.3 off).

**Over K trials**: per case the mean, min and max over trials; per model and task family **mean@K** (mean over cases
of the per-case mean), **worst@K** / **best@K** (mean of the per-case min / max), **pass^K** (share of cases whose
verdict was right in every trial), and a **95% bootstrap CI** of mean@K over cases (2000 resamples).

**Pooled metrics** (every trial one row), as before: alert accuracy, fraud precision / recall / F1, typology and role
accuracy on fraud cases, Brier score of `confidence`, cost per case (missed fraud 5, false alarm 1) next to the cheaper
trivial policy's; incident verdict accuracy, typology accuracy, member precision / recall / F1 (cases with members to
find), role accuracy, benign-cluster verdict accuracy and names.

`leaderboard` pools the case sets of a split per model and task family into `site-data/leaderboard.json`:
`models.<model>.<split>.<task>` = {n_cases, k, mean_at_k, worst_at_k, best_at_k, pass_hat_k_verdict, ci95, brier,
cost_usd, case_sets, pooled}.

## Publishing (`publish`)

`publish --split S` writes `site-data/<task>.json` per task family: the one-line description, every case's inputs
(`case.json` without paths, the brief), answer keys (always for `dev`; for an official set only with `--with-keys`,
after its season), and for `--examples` the traces of every run trial (commands, final answer, turns, cost). It
refuses case sets of another split, and, when `$FA_PUBLISH_CHECK` is set, runs that command on its output.

## Harbor export (`export-harbor`, `arena/tasks/harbor.py`)

```
<set>-<case>/instruction.md        the brief + how to work and submit
<set>-<case>/task.toml             metadata (split, seed, case), agent 20 min, verifier, environment (no network)
<set>-<case>/environment/          Dockerfile (python + duckdb), bank/ as of the case time, app/ workspace,
                                   casefile/ (the tool and its wrapper)
<set>-<case>/tests/                test.sh -> verify.py: scores /app/answer.json with the per-case score, writes
                                   /logs/verifier/reward.txt and reward.json; answer_key.json; score.py, schema.py
<set>-<case>/solution/             solve.sh + answer.json: the reference answer (scores 1.0)
```

Images are not built by the exporter.

## Tools: `casefile`

The player's only command.

```
casefile brief | tables | notes
casefile query "SQL"                 DuckDB over the as-of tables, 200 rows max
casefile customer CUST_ID [--all]    profile, accounts, cards, timeline across all tables (latest 150 events)
casefile links CUST_ID --by KIND     device ip phone email address ach_originator payee merchant referral p2p_sender:
                                     the customers sharing each key, and how many customers share it in total
casefile note "TEXT"
casefile submit drafts/answer.json   validates the JSON and saves answer.json (resubmit allowed)
```

Every command is appended to the workspace's `log.jsonl` (args, ok/error, seconds).

## Players

- **Claude**: `claude -p <brief + how-to> --model M --tools Bash,Write,Read --allowedTools "Bash(casefile:*)"
  "Edit(drafts/**)" "Read(drafts/**)" --strict-mcp-config --mcp-config <empty> --setting-sources project
  --no-session-persistence --max-turns N --output-format json`, cwd = the workspace, `PATH` prefixed with a folder
  holding the workspace's `casefile` wrapper. 30 turns for alerts, 60 for incidents, 20 min timeout. If
  `CLAUDE_CODE_OAUTH_TOKEN` is not set, it is read from `~/.claude-token` (or `$CLAUDE_TOKEN_FILE`).
- **Baseline** (`baseline.py`): hand-set points on as-of signals (account age, application risk, referrals and
  bonuses, risky logins, profile changes, disputes, claims, unauthorized returns, P2P fan-in, wires by older
  customers, near-threshold cash, tiny card auths). Incidents: the seed's risk decides the verdict; expansion is one
  hop over keys shared by at most 8 customers, keeping customers who look risky or share two keys with the seed.

## Known gaps

- `CUSTOMERS` shows the latest profile (see above). Alert case time is a fixed 24 h after the alert.
- Some fraud cases are weakly evidenced at the case time (e.g. a synthetic identity whose only visible rows are its
  application and KYC check); they are kept: real queues have them too.
- Monitor-seeded incidents (victims of one crew) are scored only on members that fired the same monitor.
- Independent fraudsters rarely touch 3+ Brindlemark customers, so most incident templates are ring crews and benign
  clusters; mule-recruit and scam-victim groups are small.
- Truth roles are rule-derived from event actors and account control; witting vs unwitting mules are not separated.

## Results

Dev set (seeds 11 and 12, 8000 roots, 120 days): 80 alert cases (45% fraud) and 21 incident cases (13 real, of which
2 verdict-only; 8 benign clusters). Real incidents: R1 bonus churning x2, R2 x2, R9 x2, R5, R6, R7, R10 refund
service, R10 farmed accounts, R4 mule recruits, one independent card tester. Unreachable members per real incident:
0-5 (neutral).

| Player | Task | Cases | K | mean@K | worst@K / best@K | pass^K (verdict) | 95% CI | pooled |
|---|---|---|---|---|---|---|---|---|
| baseline | alert | 80 | 1 | 0.573 | 0.573 / 0.573 | 0.61 | 0.47-0.67 | accuracy 0.60 / 0.63, fraud F1 0.53 / 0.62, Brier 0.24 / 0.23 (seed 11 / 12) |
| baseline | incident | 21 | 1 | 0.509 | 0.509 / 0.509 | 0.67 | 0.33-0.68 | verdict 0.67, member F1 0.07 / 0.00, benign verdicts 1.0 |
| Sonnet (smoke) | alert | 2 | 2 | 0.50 | 0.50 / 0.50 | 0.50 | - | the fraud case missed in both trials, the benign one right |
| Sonnet (smoke) | incident | 2 | 2 | 0.62 | 0.59 / 0.65 | 1.00 | 0.40-0.84 | verdicts right, member F1 0.45, typology 0.5 |

The Sonnet rows are a smoke test of the runner and scoring (4 cases, $0.67), not a measurement.
