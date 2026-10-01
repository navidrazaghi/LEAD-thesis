"""The tick log the agent writes is the tick log the harness reads.

Two changes meet at this file and neither is useful without the other. The agent
stopped rewriting ``metric_info.json`` on every tick and now flushes on an
interval, and ``run_evaluation.py`` reads that same file when a route is killed
at the wall-clock cap, to record whether the car was stuck or the machine was
slow. If the writer's format and the reader's parser drift apart, nothing fails
loudly: the killed routes simply go back to carrying no information, which is
the state this was meant to fix.

So the writer is exercised for real and its output handed to the real reader.
"""

import importlib.util
import json
import pathlib
import types

import pytest

from lead.api.abstract_driving_agent import (
    _METRIC_FLUSH_EVERY,
    AbstractDrivingAgent,
)

_HARNESS = (
    pathlib.Path(__file__).resolve().parents[3]
    / "scripts"
    / "common"
    / "run_evaluation.py"
)


def _harness() -> types.ModuleType:
    """The evaluation harness, loaded from its path.

    ``scripts/common`` is not a package, so it cannot simply be imported.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("run_evaluation", _HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent(save_path: pathlib.Path, log: dict) -> types.SimpleNamespace:
    """A stand-in carrying only what ``write_metric_info`` touches.

    The real agent needs a running simulator, and none of that is involved in
    writing the log.

    Args:
        save_path: Where the log should be written.
        log: The accumulated tick log.

    Returns:
        An object the unbound method accepts as ``self``.
    """
    return types.SimpleNamespace(
        metric_info=log,
        lead_config=types.SimpleNamespace(
            evaluation=types.SimpleNamespace(save_path=str(save_path)),
        ),
    )


def _straight_line(ticks: int, step: float) -> dict:
    """A log of a car moving a fixed distance each tick.

    Args:
        ticks: How many ticks to write.
        step: Metres between consecutive ticks.

    Returns:
        A tick log in the shape the agent produces.
    """
    return {
        str(index): {"location": [index * step, 0.0, 0.0]}
        for index in range(ticks)
    }


def test_flush_interval_is_a_sane_stride() -> None:
    """An interval of one would restore the quadratic cost this removed."""
    assert isinstance(_METRIC_FLUSH_EVERY, int)
    assert _METRIC_FLUSH_EVERY > 1


def test_written_log_round_trips(tmp_path: pathlib.Path) -> None:
    """What is written parses back to what was held."""
    log = _straight_line(50, 1.0)
    AbstractDrivingAgent.write_metric_info(_agent(tmp_path, log))
    written = json.loads((tmp_path / "metric_info.json").read_text())
    assert written == log


def test_empty_log_writes_nothing(tmp_path: pathlib.Path) -> None:
    """A route that produced no ticks should not leave an empty file behind."""
    AbstractDrivingAgent.write_metric_info(_agent(tmp_path, {}))
    assert not (tmp_path / "metric_info.json").exists()


def test_harness_reads_what_the_agent_writes(tmp_path: pathlib.Path) -> None:
    """The diagnostics recover the motion that was written."""
    AbstractDrivingAgent.write_metric_info(_agent(tmp_path, _straight_line(101, 2.0)))
    motion = _harness().motion_summary(tmp_path)
    assert motion["ticks"] == 101
    assert motion["distance_m"] == pytest.approx(200.0)
    assert motion["stationary_frac"] == 0.0


def test_harness_separates_a_stuck_car_from_a_moving_one(
    tmp_path: pathlib.Path,
) -> None:
    """The stationary share is what tells a wedged route from a slow one."""
    # A car that creeps: it holds still, edges forward a few centimetres, and
    # holds still again. Monotonic, because a real one does not slide back.
    stuck = {
        str(index): {"location": [(index // 10) * 0.04, 0.0, 0.0]}
        for index in range(200)
    }
    AbstractDrivingAgent.write_metric_info(_agent(tmp_path, stuck))
    motion = _harness().motion_summary(tmp_path)
    assert motion["stationary_frac"] > 0.5
    assert motion["distance_m"] < 1.0


def test_harness_survives_a_log_truncated_by_the_kill(
    tmp_path: pathlib.Path,
) -> None:
    """A route killed mid-write still reports, on the ticks that landed.

    This is the case the whole change exists for: the routes that reach the cap
    are killed, and a kill can land in the middle of a flush.
    """
    AbstractDrivingAgent.write_metric_info(_agent(tmp_path, _straight_line(300, 1.0)))
    destination = tmp_path / "metric_info.json"
    whole = destination.read_text()
    destination.write_text(whole[: int(len(whole) * 0.6)])

    motion = _harness().motion_summary(tmp_path)
    assert 1 < motion["ticks"] < 300
    assert motion["distance_m"] > 0
