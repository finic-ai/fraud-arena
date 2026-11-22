# Fraud Arena investigation tasks (historical, offline)

Two analyst tasks on a frozen simulation of Runmere's Fraud World, seen from Brindlemark Bank:

- **Task A, alert review.** One alerted customer per case. Decide fraud or not, the typology, the customer's role,
  a calibrated confidence and an action.
- **Task B, incident investigation.** An anomaly yields a few seed customers. Decide whether it is a real incident
  (one ring crew running one kind of scheme, or one fraudster's run of victims) or a benign cluster; if real, find
  every Brindlemark customer in it within the window and give each one's role.

Players are headless Claude Code sessions (or the non-LLM baseline) with one tool, `casefile`, which only ever shows the
bank's own tables as of the case time. Answers are graded against the simulator's truth.

```bash
python run_tasks.py build --seed 11 --roots 8000 --days 120 --out cases/pilot     # ~6 min simulation + ~10 s build
python run_tasks.py run --task alert --cases cases/pilot --model baseline --out runs/pilot-baseline
python run_tasks.py run --task alert --cases cases/pilot --model sonnet --n 12 --out runs/pilot-sonnet
python run_tasks.py run --task incident --cases cases/pilot --model sonnet --ids B001,B003 --out runs/pilot-sonnet
python run_tasks.py score --cases cases/pilot --runs runs/pilot-baseline runs/pilot-sonnet
```

`build` reuses an existing simulation in `cases/NAME/truth/` when seed/roots/days match (pass `--resimulate` to run it
again). Builds are deterministic for a given simulation. `run` copies each case workspace into the run folder, runs
four cases in parallel (`--parallel`), and writes `answer.json`, the command log and the Claude transcript per case.

## World

`worlds/fraud-world-tasks.yaml` is `worlds/fraud-world-arena.yaml` with `fraud_density` back at the world default of
0.62, so independent fraudsters (ATO, card, scams, first-party, mules...) act alongside the ten ring crews (R1-R10).
Nothing else differs (`diff` shows only the header comment and that value). No defender or attacker policy is
deployed: the bank runs its built-in rules and the rings their scripts.

## Layout

```
cases/NAME/world.json              seed, roots, days
cases/NAME/bank/<TABLE>.parquet    Brindlemark's observer tables + LOAD_TS, with end-state columns masked
cases/NAME/truth/                  simulator truth (write_truth), raw observer output, rows.parquet, entity maps,
                                   keys/<CASE>.json answer keys                       -- never visible to players
cases/NAME/alert/A001/ ...         workspaces: case.json, brief.md, notes.md, drafts/
cases/NAME/incident/B001/ ...
cases/NAME/index.json              case list + build summary
runs/NAME/<task>/<CASE>/           the player's copy of the workspace: answer.json, log.jsonl (every command)
runs/NAME/<task>/<CASE>.claude.json   Claude Code's JSON result (turns, cost, final message)
runs/NAME/<task>/run.json, score.json
```

`cases/` and `runs/` are git-ignored.

### As-of view and leak control

- Every bank row gets `LOAD_TS`: the time the row reached the bank (`row_links.load_time`). Rows about entities
  (customers, accounts, cards, devices, beneficiaries) are written once in their latest version, so their `LOAD_TS` is
  the entity's creation time instead.
- `casefile` loads only rows with `LOAD_TS <= case_time` into an in-memory DuckDB, then switches off file access and
  locks the configuration, so player SQL cannot read any file (truth included).
- Columns that hold an end-of-run state are masked in `bank/`: account `STATUS`, `RESTRICTED_FLG`, `BALANCE`; card
  `STATUS`; dispute `MERCHANT_WON`; review-queue decision columns; ACH `STATUS = RETURNED` reads `POSTED` (returns are
  in `ACH_RETURNS` once loaded); a posted check's `RETURN_REASON` is cleared.
- Workspaces contain only the case file, the brief and the player's own notes/drafts. The workspace's `case.json`
  points at the shared `bank/` folder; players have no file tools outside `drafts/`.
- Known gap: `CUSTOMERS` is the latest profile, so a contact change made after the case time can show (the change
  itself, in `PROFILE_CHANGES`, stays hidden until it loads).

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
`mule`; synthetic bust-out, new-account ID theft, ACH-return and ACATS fraud -> `synthetic_identity`; R1 bank hopping,
R10 promo farm -> `promo_abuse`; R5 check kite -> `check_fraud`; structuring -> `other`. Ring cover activity marks a
ring account but defines no family. A customer with several involvements takes the highest-priority role
(perpetrator > mule > victim), earliest first.

## Task A: alert review

**Alerts** come from 15 bank-side monitors (`monitors.py`), written in SQL over `bank/` only: card disputes (fraud /
non-receipt), P2P unauthorized claims, ACH returns, ACH callbacks, P2P holds, wire callbacks / large wires, card
step-ups, referral-reward velocity, new-device contact change, risky logins (hosting/VPN/proxy, SIM change, remote
session), referred or high-risk applications, P2P pass-through, near-threshold cash, fraud-report calls. The bank's
own payment-rule outcomes (holds, callbacks, step-ups) are read from the bank's tables, not from truth decisions.
An alert fires at the triggering row's `LOAD_TS`; the case time is 24 h later.

**Sampling.** Fraud alerts: the customer's fraud had started by the case time and at least one of its rows had
loaded by then. Benign alerts: customers the world never touched at all (no fraud involvement, no ring marks, real
identity). Customers touched only later, or only incidentally, are left out. Fraud cases are spread over typologies;
benign cases are drawn first from the same monitors as the fraud ones (look-alikes, e.g. the world's honest
"mistaken fraud claim" disputes next to real card compromise), one case per customer. The fraud share (default 45%)
is stated in every brief.

**Answer**: `disposition`, `typology` (ato, scam_victim, mule, synthetic_identity, first_party, card_compromise,
promo_abuse, check_fraud, other, none), `customer_role` (perpetrator, victim, mule, legitimate), `confidence`,
`recommended_action`, `rationale`.

**Score**: accuracy, fraud precision / recall / F1, typology and role accuracy on the fraud cases, Brier score of
`confidence`, and cost per case (missed fraud 5, false alarm 1) next to the cheaper trivial policy's cost. A missing
answer counts as not_fraud with confidence 0.5.

## Task B: incident investigation

**Truth groups.** Customers resolved as above, grouped by ring crew and typology (e.g. R1/promo_abuse,
R6/card_compromise) or, for independent fraudsters, by scenario and instance anchor (one fraudster's run). Large
groups are cut into waves (35-day windows, at least 4 members; up to 3 waves per group). The case time is when 80% of
the wave's members are visible, plus two days; the window starts a day before the first member's fraud.

**Seed.** The bank-visible key (device, IP, merchant, payee, ACH originator, referrer, address, P2P sender) held by the
most visible members and shared by at most 40 customers bank-wide. The seed is the first 35% of its holders (2-6).
When no rare key joins the members (typical of one crew's victims, who share no devices), the seed comes from the
monitor that fired on most members in the window instead, and the anomaly says how many customers fired it overall.

**Answerability.** A member counts toward recall only if it is reachable from the seed within 3 hops over keys
shared by at most 25 customers (for monitor seeds: if it fired the same monitor in the window). Unreachable members,
the same crew's customers outside the window, and members not yet visible are neutral: neither required nor counted
as wrong.

**False-positive incidents.** At case times drawn from the real incidents, keys (IP, ACH originator, merchant,
address, payee, device) shared by 4-30 customers, 3+ of them active in the last week, *none* of whom the world ever
touched: carrier/office IPs, an employer's payroll, a popular merchant, a household. Seeds have the same sizes (2-4)
as real ones and the anomaly text has the same form.

**Answer**: `is_incident`, `typology`, `members` [{cust_id, role}], `link_keys`, `summary`, `recommended_actions`.

**Score**: verdict accuracy; typology accuracy on true incidents; precision / recall / F1 of the members added beyond
the seed (averaged over true incidents; a false "not an incident" scores 0); role accuracy on correctly named
members; on benign clusters the verdict accuracy and how many customers were named.

## Tools: `casefile`

The player's only command (the spec called it `case`, but `case` is a shell keyword: neither the shell nor Claude
Code's permission matcher can run it, so it is `casefile`).

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
  holding the workspace's `casefile` wrapper. 30 turns for alerts, 60 for incidents, 20 min timeout. The environment is
  passed through; if `CLAUDE_CODE_OAUTH_TOKEN` is not set, it is read from `~/.claude-token` (or `$CLAUDE_TOKEN_FILE`).
- **Baseline** (`baseline.py`): hand-set points on as-of signals (account age, application risk, referrals and
  bonuses, risky logins, profile changes, disputes, claims, unauthorized returns, P2P fan-in, wires by older
  customers, near-threshold cash, tiny card auths). Incidents: the seed's risk decides the verdict; expansion is one
  hop over keys shared by at most 8 customers, keeping customers who look risky or share two keys with the seed.

## Known gaps

- Pilot scale: one seed, 40 alerts, 12 incidents; too few for model comparisons beyond a smoke test.
- `CUSTOMERS` shows the latest profile (see above). Alert case time is a fixed 24 h after the alert.
- Some fraud cases are weakly evidenced at the case time (e.g. a synthetic identity whose only visible rows are its
  application and KYC check); they are kept: real queues have them too.
- Monitor-seeded incidents (victims of one crew) are scored only on members that fired the same monitor.
- Truth roles are rule-derived from event actors and account control; witting vs unwitting mules are not separated.
