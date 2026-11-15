"""A queue of episodes that runs a few at a time and backs off when a subscription hits its usage limit.

Subscriptions meter usage in rolling windows, so a full pass takes days: an episode that stops on `UsageLimited`
keeps its state (see `Episode`) and goes back on the queue until the limit's reset time (or a default wait), and
every runner sharing that subscription waits too.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .runners import UsageLimited

DEFAULT_WAIT_S = 30 * 60


@dataclass
class Job:
    name: str
    account: str
    run: Callable[[], dict]
    not_before: float = 0.0
    attempts: int = 0


class Queue:
    def __init__(self, log: Path, *, parallel: int = 2, clock=time.time, sleep=time.sleep, max_attempts: int = 50):
        self.log, self.parallel, self.clock, self.sleep = Path(log), parallel, clock, sleep
        self.max_attempts = max_attempts
        self.blocked_until: dict[str, float] = {}

    def _note(self, **rec):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a") as f:
            f.write(json.dumps({"time": self.clock(), **rec}) + "\n")

    def run(self, jobs: list[Job]) -> dict[str, dict | Exception]:
        pending, results = list(jobs), {}
        running: dict[Future, Job] = {}
        with ThreadPoolExecutor(self.parallel) as pool:
            while pending or running:
                now = self.clock()
                ready = [j for j in pending
                         if j.not_before <= now and self.blocked_until.get(j.account, 0) <= now]
                while ready and len(running) < self.parallel:
                    j = ready.pop(0)
                    pending.remove(j)
                    j.attempts += 1
                    self._note(event="start", job=j.name, attempt=j.attempts)
                    running[pool.submit(j.run)] = j
                if not running:
                    soonest = min(max(j.not_before, self.blocked_until.get(j.account, 0)) for j in pending)
                    self.sleep(max(1.0, soonest - self.clock()))
                    continue
                done, _ = wait(running, timeout=60, return_when=FIRST_COMPLETED)
                for f in done:
                    j = running.pop(f)
                    try:
                        results[j.name] = f.result()
                        self._note(event="done", job=j.name)
                    except UsageLimited as e:
                        until = e.retry_at or (self.clock() + DEFAULT_WAIT_S)
                        self.blocked_until[j.account] = max(self.blocked_until.get(j.account, 0), until)
                        if j.attempts >= self.max_attempts:
                            results[j.name] = e
                            self._note(event="gave_up", job=j.name, reason=str(e))
                        else:
                            j.not_before = until
                            pending.append(j)
                            self._note(event="usage_limited", job=j.name, until=until, account=j.account)
                    except Exception as e:  # recorded; other jobs go on
                        results[j.name] = e
                        self._note(event="failed", job=j.name, error=f"{type(e).__name__}: {e}")
        return results
