from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The defender role and the decisions it owns at each observed institution, as Fraud World declares them. Hopcrate
# and Kellmont declare no defender decisions yet, so they can't host a defender episode.
INSTITUTIONS = {
    "brindlemark": {"role": "brind_defender", "observer": "brindlemark"},
    "dovrex": {"role": "dovrex_defender", "observer": "dovrex"},
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
    displacement_in_composite: bool = False

    def action_cost(self, action: str, amount: float) -> float:
        a = action.lower()
        if a in ("decline", "reject"):
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
    def observer(self) -> str:
        return INSTITUTIONS[self.institution]["observer"]

    def boundaries(self) -> list[int]:
        end = self.start_day + self.horizon_days
        return list(range(self.start_day, end, self.period_days)) + [end]
