"""Tests for the open-loop pre-screen and, more importantly, its validation.

The screen exists to spend GPU nights well, and its failure mode is silent: a
screen that does not predict still emits a confident ordering, and the models it
discards never get the run that would have shown it was wrong. So the tests that
matter here are the ones on the validator -- that it recognises a screen which
agrees with the simulator, and refuses one that does not.
"""

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "scripts"
    / "common"
    / "openloop_prescreen.py"
)


def _load_module():
    """Import the script by path; scripts/ is not an installed package."""
    spec = importlib.util.spec_from_file_location("openloop_prescreen", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["openloop_prescreen"] = module
    spec.loader.exec_module(module)
    return module


prescreen = _load_module()


class TestSpearman:
    """The rank correlation, since it is implemented here rather than imported."""

    def test_perfect_agreement(self) -> None:
        assert prescreen.spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)

    def test_perfect_disagreement(self) -> None:
        assert prescreen.spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)

    def test_it_reads_ranks_not_values(self) -> None:
        """A monotone transform of either side must not change the answer."""
        straight = prescreen.spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0])
        assert straight == pytest.approx(1.0)

    def test_a_constant_side_is_zero_not_undefined(self) -> None:
        assert prescreen.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0

    def test_ties_share_their_average_rank(self) -> None:
        assert prescreen._average_ranks([5.0, 1.0, 5.0]) == [2.5, 1.0, 2.5]

    def test_too_short_to_rank_is_zero(self) -> None:
        assert prescreen.spearman([1.0], [2.0]) == 0.0


class TestValidator:
    """What the validator concludes, which is what gates the GPU nights."""

    @staticmethod
    def _cells(pairs: dict) -> dict:
        return dict(pairs)

    def test_a_screen_that_agrees_is_accepted(self) -> None:
        """Lower open-loop error, higher driving score, in every condition."""
        openloop = self._cells(
            {
                ("a", "none:0"): 0.4, ("b", "none:0"): 0.6, ("c", "none:0"): 0.8,
                ("a", "camera:1.0"): 0.5, ("b", "camera:1.0"): 0.7, ("c", "camera:1.0"): 0.9,
            },
        )
        closedloop = self._cells(
            {
                ("a", "none:0"): 50.0, ("b", "none:0"): 40.0, ("c", "none:0"): 30.0,
                ("a", "camera:1.0"): 45.0, ("b", "camera:1.0"): 35.0, ("c", "camera:1.0"): 25.0,
            },
        )
        assert prescreen.validate(openloop, closedloop) == 0

    def test_a_screen_that_inverts_the_ordering_is_refused(self) -> None:
        openloop = self._cells(
            {
                ("a", "none:0"): 0.4, ("b", "none:0"): 0.6, ("c", "none:0"): 0.8,
                ("a", "camera:1.0"): 0.5, ("b", "camera:1.0"): 0.7, ("c", "camera:1.0"): 0.9,
            },
        )
        closedloop = self._cells(
            {
                ("a", "none:0"): 30.0, ("b", "none:0"): 40.0, ("c", "none:0"): 50.0,
                ("a", "camera:1.0"): 25.0, ("b", "camera:1.0"): 35.0, ("c", "camera:1.0"): 45.0,
            },
        )
        assert prescreen.validate(openloop, closedloop) == 2

    def test_pooled_agreement_does_not_rescue_a_bad_within_condition_order(
        self,
    ) -> None:
        """The trap the validator exists to catch.

        Conditions differing strongly from each other can carry the pooled
        correlation while the models inside each condition are ordered wrongly,
        and it is the within-condition order the screen is actually used for.
        """
        openloop = self._cells(
            {
                ("a", "none:0"): 0.40, ("b", "none:0"): 0.42,
                ("a", "camera:1.0"): 0.90, ("b", "camera:1.0"): 0.92,
            },
        )
        closedloop = self._cells(
            {
                ("a", "none:0"): 40.0, ("b", "none:0"): 50.0,
                ("a", "camera:1.0"): 10.0, ("b", "camera:1.0"): 20.0,
            },
        )
        pooled = -prescreen.spearman(
            [openloop[key] for key in sorted(openloop)],
            [closedloop[key] for key in sorted(openloop)],
        )
        assert pooled > 0.5
        assert prescreen.validate(openloop, closedloop) == 2

    def test_too_little_overlap_is_refused_rather_than_guessed(self) -> None:
        openloop = self._cells({("a", "none:0"): 0.4, ("b", "none:0"): 0.6})
        closedloop = self._cells({("a", "none:0"): 50.0})
        assert prescreen.validate(openloop, closedloop) == 1


class TestReaders:
    """Joining the two CSVs on a condition key they spell differently."""

    def test_closed_loop_rows_without_a_score_are_skipped(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        path = tmp_path / "closed.csv"
        path.write_text(
            "model,modality,severity,route,driving_score\n"
            "a,none,0,r1.xml,50\n"
            "a,none,0,r2.xml,\n"
            "a,none,0,r3.xml,30\n",
            encoding="utf-8",
        )
        assert prescreen.read_closedloop(path) == {("a", "none:0"): 40.0}

    def test_open_loop_frames_average_per_cell(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        path = tmp_path / "open.csv"
        path.write_text(
            "model,condition,frame,l2\n"
            "a,none:0,0,0.2\n"
            "a,none:0,1,0.4\n"
            "b,none:0,0,1.0\n",
            encoding="utf-8",
        )
        assert prescreen.read_openloop(path) == {
            ("a", "none:0"): pytest.approx(0.3),
            ("b", "none:0"): pytest.approx(1.0),
        }


class TestAgainstTheRecordedRuns:
    """The verdict on this stack, pinned so a change to it is deliberate.

    The screen picks the gated rung in every condition and the simulator ranks
    it last or middle in every condition. That is the same effect the ablation
    reported from the other side -- a model whose waypoint error improves while
    its driving does not -- so the screen fails here for a reason, not by
    accident, and it should not silently start being trusted.
    """

    def test_the_screen_does_not_predict_closed_loop_here(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[3]
        openloop = repo / "results" / "openloop_frames.csv"
        closedloop = repo / "results" / "closed_loop.csv"
        if not openloop.exists() or not closedloop.exists():
            pytest.skip("recorded results are not present in this checkout")
        verdict = prescreen.validate(
            prescreen.read_openloop(openloop),
            prescreen.read_closedloop(closedloop),
        )
        assert verdict == 2
