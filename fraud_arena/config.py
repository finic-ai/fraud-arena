from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The defender role at each observed institution, and its manual review team (`analyst`, with a daily capacity),
# as Fraud World declares them. A defender's bundle covers both: its detection decides at the gates, and its review
# rules decide the cases its team reviews.
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
    alert: float = 8.0
    block: float = 25.0
    close_account: float = 200.0
    # Per dollar of mule pass-through prevented, against 1.0 for a dollar of loss: a stated default, not the owner's.
    mule_flow: float = 0.10
    displacement_in_composite: bool = False

    def action_cost(self, action: str, amount: float) -> float:
        a = action.lower()
        if a in ("decline", "reject", "deny", "refuse"):
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


@dataclass(frozen=True)
class EpisodeConfig:
    world: Path
    institution: str
    seed: int
    roots: int | None = None
    partitions: int = 4
    start_day: int = 180
    period_days: int = 30
    horizon_days: int = 120
    phase: str = "static"
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    grayson_calls_per_period: int = 500

    def __post_init__(self):
        if self.institution not in INSTITUTIONS:
            raise ValueError(f"no defender role at {self.institution!r}; one of {sorted(INSTITUTIONS)}")
        if self.phase not in ("static", "adversarial"):
            raise ValueError("phase is 'static' or 'adversarial'")
        if self.period_days <= 0 or self.horizon_days < self.period_days:
            raise ValueError("period_days must be positive and no longer than horizon_days")

    @property
    def role(self) -> str:
        return INSTITUTIONS[self.institution]["role"]

    @property
    def roles(self) -> list[str]:
        i = INSTITUTIONS[self.institution]
        return [i["role"]] + ([i["analyst"]] if i.get("analyst") else [])

    @property
    def observer(self) -> str:
        return INSTITUTIONS[self.institution]["observer"]

    def boundaries(self) -> list[int]:
        end = self.start_day + self.horizon_days
        return list(range(self.start_day, end, self.period_days)) + [end]
