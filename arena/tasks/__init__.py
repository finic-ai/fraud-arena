"""Fraud Arena historical (offline) investigation tasks: alert review (task A) and incident investigation (task B).

The world is simulated once per seed (`world.py`); `truth.py` derives who did what to whom from the simulator's truth;
`monitors.py` raises the bank-visible alerts and anomalies; `build.py` samples cases and writes one truth-free
workspace per case; `cli.py` is the `case` command players use; `baseline.py` and `score.py` grade answers."""
