"""The harness keeps which infractions a route had, not only how many.

The leaderboard's checkpoint file lists every infraction by kind, but the
harness clears that file before the next route and used to keep only the count.
Under camera loss the kind is the question -- a red light is a failure of
information only the camera carries, a stall is not -- so the kinds now go into
the results CSV, and these tests hold that in place.

The records below have the shape the Bench2Drive leaderboard writes
(``RouteRecord.to_json``: each kind mapped to the list of its events), built by
hand because the leaderboard module needs a simulator to import.
"""

import csv
import importlib.util
import io
import json
import pathlib
import types

_HARNESS = (
    pathlib.Path(__file__).resolve().parents[3]
    / "scripts"
    / "common"
    / "run_evaluation.py"
)


def _harness() -> types.ModuleType:
    """The evaluation harness, loaded from its path (``scripts/common`` is not a package)."""
    spec = importlib.util.spec_from_file_location("run_evaluation", _HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(**events: int) -> dict:
    """A route record as the leaderboard writes it, with ``events`` of each kind."""
    kinds = (
        "collisions_layout", "collisions_pedestrian", "collisions_vehicle", "red_light",
        "stop_infraction", "outside_route_lanes", "min_speed_infractions",
        "yield_emergency_vehicle_infractions", "scenario_timeouts", "route_dev",
        "vehicle_blocked", "route_timeout",
    )
    return {
        "index": 0,
        "route_id": "RouteScenario_0",
        "status": "Completed",
        "num_infractions": sum(events.values()),
        "infractions": {kind: [f"{kind} event"] * events.get(kind, 0) for kind in kinds},
        "scores": {"score_route": 100.0, "score_penalty": 0.49, "score_composed": 49.0},
        "meta": {"route_length": 150.0},
        "town_name": "Town12",
    }


def test_kinds_are_counted_sorted_and_zeros_left_out():
    harness = _harness()
    summary = harness.summarize_infractions(_record(red_light=2, collisions_vehicle=1))
    assert summary == "collisions_vehicle:1;red_light:2"


def test_a_clean_route_gives_an_empty_field():
    harness = _harness()
    assert harness.summarize_infractions(_record()) == ""


def test_a_record_without_the_key_gives_an_empty_field():
    harness = _harness()
    assert harness.summarize_infractions({"status": "Completed"}) == ""
    assert harness.summarize_infractions({"infractions": None}) == ""


def test_integer_counts_are_read_as_counts():
    # The global record stores counts rather than lists; accept either.
    harness = _harness()
    summary = harness.summarize_infractions({"infractions": {"red_light": 3, "route_dev": 0}})
    assert summary == "red_light:3"


def test_read_score_carries_the_kinds_and_leaves_the_rest_unchanged(tmp_path):
    harness = _harness()
    endpoint = tmp_path / "checkpoint_endpoint.json"
    record = _record(vehicle_blocked=1, red_light=1)
    endpoint.write_text(json.dumps({"_checkpoint": {"records": [record]}}))

    scores = harness.read_score(endpoint)

    assert scores["infractions"] == "red_light:1;vehicle_blocked:1"
    assert scores["num_infractions"] == 2
    assert scores["driving_score"] == 49.0
    assert scores["route_completion"] == 100.0
    assert scores["infraction_penalty"] == 0.49
    assert scores["status"] == "Completed"
    assert scores["town"] == "Town12"


def test_the_new_column_adds_to_the_schema_without_reordering_it():
    harness = _harness()
    before = (
        "model", "modality", "severity", "route", "driving_score", "route_completion",
        "infraction_penalty", "status", "town", "num_infractions", "seconds", "ticks",
        "distance_m", "stationary_frac",
    )
    fields = harness._FIELDS
    assert fields.count("infractions") == 1
    assert tuple(f for f in fields if f != "infractions") == before


def test_the_field_survives_a_csv_round_trip():
    harness = _harness()
    row = dict.fromkeys(harness._FIELDS, "")
    row["infractions"] = harness.summarize_infractions(_record(red_light=1, collisions_pedestrian=1))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=harness._FIELDS)
    writer.writeheader()
    writer.writerow(row)
    buffer.seek(0)
    back = next(csv.DictReader(buffer))
    assert back["infractions"] == "collisions_pedestrian:1;red_light:1"


def test_an_unscored_route_leaves_the_field_empty():
    # A route that never scored is written from dict.fromkeys(_SCORE_FIELDS).
    harness = _harness()
    row = dict.fromkeys(harness._SCORE_FIELDS)
    assert "infractions" in row and row["infractions"] is None
