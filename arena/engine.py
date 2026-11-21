"""Runs batches of Runmere's `auth_vs_adversary` duel with each role played by a policy (or the world's built-in)."""

from pathlib import Path

import runmere

from .policy import DECISIONS, decide

WORLD = Path(__file__).resolve().parent.parent / "worlds" / "payments-duel.world.yaml"
EPISODE = "auth_vs_adversary"
ROLE_OF_INDEX = {0: "defender", 1: "attacker"}  # the engine's roles are [defender, adversary]


class Duel:
    def __init__(self, world=WORLD, episode=EPISODE):
        self.world = runmere.World.load(str(world))
        self.episodes = self.world.episodes(episode)
        spec = self.episodes.spec()
        self.decisions = {d["index"]: d for d in spec["decisions"]}
        self.term_names = spec["reward_terms"]
        for d in self.decisions.values():
            known = DECISIONS[d["name"]]
            assert known["actions"] == d["actions"], (d["name"], d["actions"])
            assert list(known["features"]) == d["features"], (d["name"], d["features"])
        self._prepared = set()

    def run_batch(self, seed, n, policies):
        """Play episodes 0..n-1 of `seed`. `policies` maps "defender"/"attacker" to a policy dict, or None for the
        world's built-in policy. Returns per-episode results and every decision made, for building player views."""
        if seed not in self._prepared:
            self.episodes.prepare(seed)
            self._prepared.add(seed)
        episodes, decisions = [], []
        for index in range(n):
            ep, step = self.episodes.reset(seed, index)
            while step["request"] is not None:
                req = step["request"]
                spec = self.decisions[req["decision"]]
                role = "defender" if spec["role"] == "defender" else "attacker"
                policy = policies.get(role)
                features = dict(zip(spec["features"], req["values"]))
                row = {"episode": index, "time": req["time"], "role": role, "decision": spec["name"],
                       "features": features}
                if policy is None:
                    row["source"] = "builtin"
                    row["action"] = spec["actions"][ep.builtin_action()] if hasattr(ep, "builtin_action") else None
                    step = ep.act_builtin()
                else:
                    action, rule = decide(policy, spec["name"], features)
                    idx = spec["actions"].index(action)
                    if not req["mask"][idx]:
                        idx = req["default_action"]
                        row["source"] = "unavailable->default"
                    else:
                        row["source"] = "rule" if rule is not None else "default"
                    row["rule"] = rule
                    row["action"] = spec["actions"][idx]
                    step = ep.act(idx)
                decisions.append(row)
            info = ep.info() if callable(ep.info) else ep.info
            terms = {ROLE_OF_INDEX[r]: dict(zip(self.term_names[r], info["term_totals"][r])) for r in (0, 1)}
            episodes.append({"episode": index, "defender": info["totals"][0], "attacker": info["totals"][1],
                             "terms": terms, "weight": info["weight"], "done": info["done"]})
        return {"seed": seed, "n": n, "episodes": episodes, "decisions": decisions,
                "defender": sum(e["defender"] for e in episodes), "attacker": sum(e["attacker"] for e in episodes)}
