"""Score weights and institutions, from the v0 harness (fraud_arena/config.py on feat/defender-harness)."""
from dataclasses import dataclass

INSTITUTIONS = {
    "brindlemark": {"role": "brind_defender", "analyst": "brind_analyst", "observer": "brindlemark"},
    "dovrex": {"role": "dovrex_defender", "analyst": "dovrex_analyst", "observer": "dovrex"},
    "hopcrate": {"role": "hop_defender", "analyst": "hop_analyst", "observer": "hopcrate"},
    "kellmont": {"role": "kell_defender", "analyst": "kell_analyst", "observer": "kellmont"},
}


@dataclass(frozen=True)
class ScoreWeights:
    """Dollar costs of friction to good customers. Defaults are a stated starting point, not the owner's decision
    (ADVERSARIAL.md open question 2); every report shows the components unweighted too."""

    decline_fixed: float = 3.0
    decline_share_of_amount: float = 0.10
    step_up: float = 0.50
    hold: float = 2.0
    review: float = 15.0
    # The analyst team's capacity for operational reviews (the analyst role's reviews_per_day). Raised alerts already
    # wait in the analysts' queue; `review` actions beyond the day's capacity are worked as overtime or outsourced.
    review_capacity_per_day: int = 10
    review_overtime: float = 75.0
    alert: float = 8.0
    block: float = 25.0
    close_account: float = 200.0
    # Per dollar of mule pass-through prevented, against 1.0 for a dollar of loss: a stated default, not the owner's.
    mule_flow: float = 0.10
    displacement_in_composite: bool = False

    def action_cost(self, action: str, amount: float) -> float:
        a = action.lower()
        if a in ("decline", "reject", "deny", "refuse", "return_debit"):  # a returned good debit fails like a decline
            return self.decline_fixed + self.decline_share_of_amount * max(amount, 0.0)
        if a == "step_up":
            return self.step_up
        if a in ("hold", "standard"):
            return self.hold
        if a == "review":
            return self.review
        if a == "raise_alert":
            return self.alert
        if a in ("block", "restrict"):
            return self.block
        if a == "close_account":
            return self.close_account
        return 0.0


