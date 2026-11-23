"""The defender's deployments on an engine that records cases natively: no outside code unless the defender has lists.

    python -m unittest tests.test_native_cases
"""
import unittest

from arena.fw.deployment import compose

BUILTIN = [{"decision": "auth", "when": "obs.amount > 100", "action": "decline"},
           {"decision": "auth", "action": "approve"},
           {"decision": "review", "action": "approve"}]
GATES = {"auth": "event", "review": "wait"}


class NativeCases(unittest.TestCase):
    def test_native_logging_needs_no_outside(self):
        dep = compose({}, BUILTIN, GATES, native=True)
        self.assertNotIn("outside", dep)
        self.assertNotIn("external", dep)
        self.assertTrue(dep["record_cases"])
        self.assertEqual(dep["format"], 7)
        self.assertFalse(any("outside." in r.get("when", "") for r in dep["rules"]))

    def test_counterfactuals_record_nothing(self):
        dep = compose({}, BUILTIN, GATES, log=False, native=True)
        self.assertNotIn("record_cases", dep)
        self.assertNotIn("outside", dep)

    def test_only_lists_read_outside(self):
        dep = compose({}, BUILTIN, GATES, native=True, lists={"bad": {"key": "card_id", "set": {"1"}}})
        self.assertEqual(set(dep["outside"]), {"list_bad"})
        self.assertTrue(dep["record_cases"])

    def test_old_engine_keeps_the_logging_rule(self):
        dep = compose({}, BUILTIN, GATES)
        self.assertIn("arena_log", dep["outside"])
        self.assertIn("auth", dep["external"])


if __name__ == "__main__":
    unittest.main()
