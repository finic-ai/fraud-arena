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

## Replays and the viewer

A game writes `<out>/replay/replay.jsonl` as it runs (`--replay off` to skip it): one JSON frame per simulated day
and run, one per period boundary with the players' turns, and the score at each boundary. When the match ends the
file is rebuilt from the scoring's truth exports, so the finished replay is exact. The format is described in
`fraud_arena/replay.py`. It is truth: public after the match, never shown to a player during it.

    fraud-arena game ... --watch       # serves the viewer on 127.0.0.1 and follows the match live
    fraud-arena watch <out>            # the same for a running or finished match
    fraud-arena replay <out>           # rebuilds <out>/replay/ from a match's outputs

The viewer (`viewer/`, TypeScript and PixiJS) builds into `fraud_arena/web/viewer.js` and `viewer.css`
(`cd viewer && npm ci && npm run build`). To embed it, load both files and leave this element on the page:

    <div id="fraud-arena-viewer" data-replay="https://.../replay.jsonl"></div>

`viewer/sample/replay.jsonl` is a synthetic replay (a toy economy, `python -m fraud_arena.demo <out>`) for demos.
Open `viewer/index.html` over HTTP to play it; `#day=60&focus=R2&ghost=1&paused=1&turns=35` deep-links a view.

## The sandbox

Each player (each defender and each attacker) runs as its own Unix user, `arena-p1` to `arena-p12`
(`sandbox/setup-box.sh`). Its folder under `--sandbox` holds its workspace, its tables and its deployed bundles, and
only that user and the harness can read it, so no defender can read an attacker's `view.json`, which is truth.
Players reach only loopback.

Model access is a choice per run, `--credentials`:

- `subscription` (the default): Claude Code players run on `CLAUDE_CODE_OAUTH_TOKEN`, and every Codex player runs as
  `arena-codex` with its one ChatGPT login, so Codex players share that user and its folders (the harness refuses
  Codex on both sides of a game). Players reach the providers through the allowlisting proxy (`--proxy`). A player's
  own token is in its session's environment, so code it runs can read it.
- `api-key`: no model credential is in a player's sandbox. The harness runs a model gateway
  (`fraud_arena/gateway.py`) that holds `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`; each player gets only the gateway's
  address and its own gateway token. The gateway accepts only the run's models, logs every call's tokens to
  `<out>/gateway/usage.jsonl`, and stops the run at `--model-budget-tokens`, the budget the owner approved.

Either way, Grayson's key stays in the harness's broker, and sessions get their environment from a file in the
player's folder (`sandbox/arena-exec`), never on a command line, where `ps` shows it to every user.
`sandbox/check-box.sh` checks all of this from the players' side. For a local run without player users, pass
`--no-player-users`.
