"""Online calibration of how cautious the governor is allowed to be.

The thesis mechanism failed for a reason worth stating plainly: an informative
signal was wired to attention logits, where its effect on driving was an order
of magnitude smaller than its effect on attention. The governor wires the same
signal to behaviour instead, and that raises the question attention never had to
answer -- how much slowing down is the right amount. A hand-set threshold would
be one more knob tuned on the routes it is then scored on.

Conformal decision theory answers it without a threshold. Hold one scalar that
says how conservative to be, watch a surrogate risk that is observable at
runtime, and push the scalar up when risk shows up and down when it does not:

    lambda <- clip(lambda + step * (risk - target), 0, ceiling)

Two properties make this the right shape. It needs no model of how risk depends
on caution, only the sign of the error, so it cannot be wrong about a
relationship nobody has measured. And its long-run realised risk rate converges
to the target from any start, at a rate set by the step size -- so the target is
chosen, in units a reader understands, rather than the threshold.

What it does not give is a per-frame guarantee. The bound is on the average over
a run, so a single tick can be arbitrarily wrong; that is inherent to the method
and is why the mapping this drives keeps a floor under the speed rather than
letting the calibrator stop the car.
"""

from dataclasses import dataclass, field


@dataclass
class ConformalCautionCalibrator:
    """Adapts one conservativeness scalar to hold a surrogate risk at a target.

    Attributes:
        target_risk: Long-run rate of surrogate risk events to converge to.
        step_size: How far one observation moves the scalar. Larger adapts
            faster and oscillates more.
        ceiling: Upper bound on the scalar, so a pathological run cannot drive
            the governor into permanent standstill.
        value: The current conservativeness, in ``[0, ceiling]``.
        num_updates: Observations seen, for reporting.
        num_risk_events: Risk events seen, for reporting.
    """

    target_risk: float = 0.05
    step_size: float = 0.05
    ceiling: float = 1.0
    value: float = 0.0
    num_updates: int = 0
    num_risk_events: int = 0
    _history: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Reject a configuration that could not converge.

        Raises:
            ValueError: If the target is not a rate, or the step is not
                positive, or the ceiling is not positive.
        """
        if not 0.0 < self.target_risk < 1.0:
            raise ValueError(
                f"target_risk must be a rate strictly inside (0, 1), got "
                f"{self.target_risk}.",
            )
        if self.step_size <= 0.0:
            raise ValueError(f"step_size must be positive, got {self.step_size}.")
        if self.ceiling <= 0.0:
            raise ValueError(f"ceiling must be positive, got {self.ceiling}.")
        self.value = min(max(self.value, 0.0), self.ceiling)

    def update(self, risk_event: bool) -> float:
        """Fold one tick's surrogate risk into the conservativeness.

        Args:
            risk_event: Whether this tick produced a surrogate risk event.

        Returns:
            The updated conservativeness.
        """
        observed = 1.0 if risk_event else 0.0
        self.value = min(
            max(self.value + self.step_size * (observed - self.target_risk), 0.0),
            self.ceiling,
        )
        self.num_updates += 1
        self.num_risk_events += int(risk_event)
        self._history.append(self.value)
        return self.value

    @property
    def realised_risk(self) -> float:
        """The risk rate actually seen so far; zero before the first update."""
        if self.num_updates == 0:
            return 0.0
        return self.num_risk_events / self.num_updates

    def state(self) -> dict[str, float]:
        """Report the calibrator's state for the run log.

        Returns:
            The scalar, the realised rate and the target, so a run can be read
            back without the calibrator object.
        """
        return {
            "caution_lambda": self.value,
            "realised_risk": self.realised_risk,
            "target_risk": self.target_risk,
            "num_updates": float(self.num_updates),
        }
