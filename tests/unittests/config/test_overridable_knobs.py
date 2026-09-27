"""Every config key a run script passes must actually be overridable.

The config tree has two kinds of attribute that look identical from a command
line and behave nothing alike. An annotated class attribute is a knob and takes
an override; a ``@property`` is derived and refuses one, by raising.

That refusal is correct, and its timing is the problem. A dotlist reaches the
child process, the config is built there, and the raise happens before the
driving agent is ever constructed -- so the leaderboard reports "Agent couldn't
be set up" with no traceback in the sweep's own log, identically on every route.
A twenty-route calibration run failed that way twenty times over forty minutes
before the cause was found, and the cause was one token naming a derived
property.

These tests read the tokens the run scripts actually pass and check each one
against the config, so the same mistake fails here in a second instead of on the
machine an hour later.
"""

import pathlib
import re

import pytest

from lead.config.lead_config import load_lead_config

_SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "common"

# A dotted config key followed by '=', as it appears in a shell command.
_TOKEN = re.compile(r"\b((?:evaluation|policy|training|expert)(?:\.[a-z_0-9]+)+)=")


def _tokens_in(path: pathlib.Path) -> set[str]:
    """Config keys a script passes, whatever the value is."""
    return set(_TOKEN.findall(path.read_text(encoding="utf-8")))


def _override(key: str, value: str) -> None:
    """Apply one dotlist token to a fresh config tree.

    Args:
        key: The dotted config key.
        value: Any value; the type only has to survive coercion.
    """
    node = load_lead_config()
    parts = key.split(".")
    for part in parts[:-1]:
        node = getattr(node, part)
    node.apply_overrides({parts[-1]: value}, is_user_override=True)


def _script_files() -> list[pathlib.Path]:
    return sorted(p for p in _SCRIPTS.glob("*.sh") if _tokens_in(p))


@pytest.mark.parametrize(
    "script",
    _script_files(),
    ids=lambda p: p.name,
)
def test_every_key_a_script_passes_is_overridable(script: pathlib.Path) -> None:
    """A derived property here means the run dies before the agent is built."""
    refused = []
    for key in sorted(_tokens_in(script)):
        try:
            _override(key, "0")
        except AttributeError as error:
            if "derived and cannot be overridden" in str(error):
                refused.append(key)
        except Exception:  # noqa: BLE001 - a type error is not what this checks
            continue
    assert not refused, (
        f"{script.name} passes derived properties that cannot be overridden: "
        f"{refused}. The run would fail at config build, before the agent "
        f"exists, and report 'Agent couldn't be set up' on every route."
    )


class TestTheCheckItself:
    """The check has to be able to fail, or it is decoration."""

    def test_a_derived_property_is_refused(self) -> None:
        with pytest.raises(AttributeError, match="derived and cannot be overridden"):
            _override("evaluation.save_path", "/tmp/anywhere")

    def test_a_real_knob_is_accepted(self) -> None:
        _override("evaluation.inference.caution_calibration_log", "/tmp/x.jsonl")

    def test_the_scripts_are_actually_being_read(self) -> None:
        """A regex that matched nothing would make every test above vacuous."""
        found = {key for path in _script_files() for key in _tokens_in(path)}
        assert len(found) > 3, found
