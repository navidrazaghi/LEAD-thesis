"""Add an escape-from-stall creep to the evaluation agent.

Off by default, so every existing result stays reproducible.
"""

import io
import pathlib
import sys

ROOT = pathlib.Path.home() / "LEAD/lead"
CONFIG = ROOT / "src/lead/config/evaluation/inference_config.py"
AGENT = ROOT / "src/lead/evaluation/agents/transfuser/transfuser_agent.py"

CONFIG_ANCHOR = '''    # --- Control which output is used for controlling ---'''

CONFIG_BLOCK = '''    # --- Escape from a stalled state ---
    # The controller derives its demand from the predicted plan, and zero
    # demand means zero throttle. A stopped car then sees a scene in which it
    # is stopped and predicts stopping again, so the state is self-
    # reinforcing: nothing in the control path can restore motion, and a route
    # ends with the vehicle motionless and no fault recorded anywhere.
    #
    # With this on, a small throttle is applied once the vehicle has been
    # continuously stationary for ``creep_after_seconds``, for
    # ``creep_seconds`` at a time, and then the wait starts again.
    #
    # Off by default, and it must stay off to reproduce this project's
    # results: every closed-loop number was collected without it, so a run
    # with it on is not comparable with those tables.
    creep_when_stuck: bool = False
    # How long the vehicle must be continuously stationary before it creeps.
    creep_after_seconds: float = 30.0
    # Speed below which the vehicle counts as stationary, in m/s.
    creep_speed_threshold: float = 0.1
    # Throttle applied while creeping, in [0, 1].
    creep_throttle: float = 0.4
    # How long one creep lasts, in seconds.
    creep_seconds: float = 1.0

'''

STATE_ANCHOR = '''        self.caution = 0.0
'''

STATE_BLOCK = '''        # Ticks the vehicle has been continuously stationary, and ticks left
        # in the creep now running. A fresh agent is built per route, so both
        # start at zero for every route rather than carrying across the sweep.
        self.stationary_ticks = 0
        self.creep_ticks_left = 0
'''

METHOD_ANCHOR = '''    def _post_process_target_speed(
'''

METHOD_BLOCK = '''    def _creep_if_stuck(
        self,
        throttle: float,
        brake: float,
        ego_speed: float,
    ) -> tuple[float, float]:
        """Nudge the vehicle forward when it has been stationary too long.

        Throttle is ignored while the brake is held, so a creep has to release
        the brake as well as raise the throttle or the vehicle does not move.

        Steering is deliberately left as it is. By the time a creep triggers
        the vehicle is braking and stationary, and the caller has already
        zeroed the steering for that case, so the nudge is straight ahead. It
        is meant to break the loop, not to follow the route.

        Args:
            throttle: The throttle the trackers asked for.
            brake: The brake the trackers asked for.
            ego_speed: Current speed of the vehicle in m/s.

        Returns:
            The throttle and brake to apply this tick.
        """
        inference = self.lead_config.evaluation.inference
        if not inference.creep_when_stuck:
            return throttle, brake

        ticks_per_second = self.lead_config.expert.simulation.carla_fps
        if ego_speed < inference.creep_speed_threshold:
            self.stationary_ticks += 1
        else:
            self.stationary_ticks = 0
            self.creep_ticks_left = 0

        if self.creep_ticks_left == 0 and self.stationary_ticks >= (
            inference.creep_after_seconds * ticks_per_second
        ):
            self.creep_ticks_left = max(
                1,
                round(inference.creep_seconds * ticks_per_second),
            )

        if self.creep_ticks_left > 0:
            self.creep_ticks_left -= 1
            if self.creep_ticks_left == 0:
                # Wait a full interval before trying again. Without this the
                # trigger condition is still true on the next tick, and a
                # nudge that failed to free the car becomes a throttle held
                # down for the rest of the route.
                self.stationary_ticks = 0
            return inference.creep_throttle, 0.0

        return throttle, brake

''' + METHOD_ANCHOR

CALL_ANCHOR = '''        return AgentPrediction(
            prediction=prediction,'''

CALL_BLOCK = '''        throttle, brake = self._creep_if_stuck(throttle, brake, float(ego_speed))

''' + CALL_ANCHOR


def patch(path: pathlib.Path, pairs: list[tuple[str, str]]) -> None:
    """Apply one anchor/replacement pair at a time, refusing anything ambiguous."""
    text = io.open(path, encoding="utf-8").read()
    for anchor, replacement in pairs:
        found = text.count(anchor)
        if found != 1:
            sys.exit(f"FATAL: {path.name}: anchor appears {found} times, expected 1:\n{anchor[:80]}")
        text = text.replace(anchor, replacement)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)
    print(f"  patched {path.relative_to(ROOT)}")


patch(CONFIG, [(CONFIG_ANCHOR, CONFIG_BLOCK + CONFIG_ANCHOR)])
patch(AGENT, [
    (STATE_ANCHOR, STATE_ANCHOR + STATE_BLOCK),
    (METHOD_ANCHOR, METHOD_BLOCK),
    (CALL_ANCHOR, CALL_BLOCK),
])
print("done")
