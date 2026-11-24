"""The defender task is rules-only in v1: the policy checker refuses models and score reads.

    python -m unittest tests.test_rules_only
"""

import json
import tempfile
import unittest
from pathlib import Path

from arena.fw.cli import NO_MODELS, check

DECISIONS = [{"name": "brind_card_auth", "actions": ["approve", "decline"], "default_action": "approve",
              "features": [["amount", "number"], ["new_device", "bool"]]}]


class RulesOnly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        (self.work / "decisions.json").write_text(json.dumps(DECISIONS))

    def tearDown(self):
        self.tmp.cleanup()

    def test_rules_pass(self):
        policy = {"format": 1, "rules": [{"decision": "brind_card_auth", "when": "obs.amount > 500 and obs.new_device",
                                          "action": "decline"}]}
        self.assertEqual(check(self.work, "defender", policy), [])

    def test_models_are_refused(self):
        policy = {"format": 1, "rules": [], "models": {"m": {"kind": "logistic", "features": ["amount"],
                                                              "weights": [0.1], "intercept": 0}}}
        self.assertIn(f"models: {NO_MODELS}", check(self.work, "defender", policy))

    def test_score_reads_are_refused(self):
        policy = {"format": 1, "rules": [{"decision": "brind_card_auth", "when": "score.m > 0.5", "action": "decline"}]}
        self.assertEqual(check(self.work, "defender", policy), [f"rules[0]: score.m: {NO_MODELS}"])


if __name__ == "__main__":
    unittest.main()
