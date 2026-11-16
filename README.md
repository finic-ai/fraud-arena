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
