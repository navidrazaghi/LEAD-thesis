"""Drive a matrix of checkpoints and sensor conditions through CARLA.

The ablation needs every model scored on the same routes under the same
conditions, which is several hundred closed-loop runs. Doing that by hand is
not realistic, and the failure modes are quiet: a crashed CARLA leaves a route
unscored, a stale result directory silently reports an old run's number.

So this owns the whole loop. It starts CARLA, runs one route at a time as a
subprocess, reads the score out of the leaderboard's own JSON, and appends a
row per run to a CSV. Re-running skips rows already in that CSV, so an
interrupted sweep resumes instead of restarting.

Example, a camera-degradation sweep over two models::

    python scripts/common/run_evaluation.py
        --models rung0=outputs/rung0_baseline_post
                 rung1=outputs/rung1_deformable_free_post
        --routes src/lead/routes/eval_sets/degradation.txt
        --conditions none:0 camera:0.25 camera:0.5 camera:0.75 camera:1.0
        --out results/degradation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
# CARLA needs this long to answer RPC after launch on this machine; it is a
# ceiling, not a sleep, so a fast boot costs nothing.
_CARLA_BOOT_TIMEOUT_S = 180
# A route that has not finished by now is wedged. The floor is the
# leaderboard's own hard cap of 4000 simulation ticks: a model that stalls
# burns all of them, which measured at about 1300 s on an idle machine.
#
# This machine is shared. Under a load average near 30 the same 4000 ticks ran
# past 2400 s, and the guard then discarded the run -- so the baseline lost
# exactly the routes where it stalls, which are the routes the comparison most
# needs. Because the simulation is synchronous with a fixed time step, that
# contention costs wall clock and nothing else: the scores are unaffected, only
# the time taken to produce them.
#
# Raising it to 5400 s did not help: the cell it was raised for failed again at
# 5400 s, which means that cell is wedged rather than slow and no ceiling
# reaches it. So the ceiling is set from the other direction -- by what a wedge
# costs. The longest healthy run measured is 1152 s, and 2700 s keeps more than
# twice that as headroom while halving the price of each wedge.
_ROUTE_TIMEOUT_S = 2700
_SCORE_FIELDS = (
    "driving_score",
    "route_completion",
    "infraction_penalty",
    "status",
    "town",
    "num_infractions",
)
_FIELDS = ("model", "modality", "severity", "route", *_SCORE_FIELDS, "seconds")
# Statuses that describe the simulator giving up rather than the agent driving
# badly. A row carrying one of these is not a measurement, so it is neither
# treated as done on resume nor left to poison the next few runs: CARLA is
# restarted immediately after one.
# A TickRuntime is deliberately absent here. It is raised by scenario_manager
# at a hard cap of 4000 simulation ticks, so it means the agent used its whole
# step budget without reaching the goal -- driving too slowly or stalling. The
# leaderboard still records a full score for those routes, and dropping them
# discards the worst runs unevenly across conditions, which flatters whichever
# condition stalls most.
_INFRASTRUCTURE_FAILURES = ("NoResult", "Agent timed out")


def _unbuffer_stdout() -> None:
    """Flush progress lines as they happen, not when the pipe buffer fills.

    Redirected stdout is block buffered, so a sweep that takes hours writes
    nothing to its log until it ends. That makes a healthy run look dead, and
    makes any grep over the log quietly answer about an empty file.
    """
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)


def route_seed(route: pathlib.Path) -> int:
    """A stable seed for one route's sensor damage.

    Args:
        route: The route file being driven.

    Returns:
        A seed derived from the route name, equal across runs and machines.
    """
    return zlib.crc32(route.name.encode()) % (2**31)


def free_port(start: int) -> int:
    """The first free TCP port at or above ``start``.

    The default traffic-manager port is often taken by another user's job, and
    the failure it produces — "Failed to connect to CARLA Traffic Manager" —
    does not name the port, so the ports are probed rather than assumed.

    Args:
        start: Where to start looking.

    Returns:
        A port nothing is listening on.

    Raises:
        RuntimeError: If nothing in the scanned range is free.
    """
    for port in range(start, start + 200):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"no free port in [{start}, {start + 200})")


class Carla:
    """A CARLA server, started and stopped around a batch of routes."""

    def __init__(self, carla_root: pathlib.Path, port: int):
        """Prepare to run a server without starting it.

        Args:
            carla_root: Directory holding ``CarlaUE4.sh``.
            port: RPC port to serve on.
        """
        self.carla_root = carla_root
        self.port = port
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        """Launch the server and wait until it answers on its port.

        Raises:
            RuntimeError: If it does not answer within the boot timeout.
        """
        self.process = subprocess.Popen(  # noqa: S603
            [
                "./CarlaUE4.sh",
                f"-world-port={self.port}",
                "-nosound",
                "-RenderOffScreen",
                f"-carla-streaming-port={self.port + 1}",
            ],
            cwd=self.carla_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + _CARLA_BOOT_TIMEOUT_S
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                    time.sleep(5)
                    return
            time.sleep(3)
        self.stop()
        raise RuntimeError(f"CARLA did not answer on port {self.port}")

    def stop(self) -> None:
        """Kill the server, then sweep for any orphan holding its port.

        The launcher shell exits once the engine binary is up, which reparents
        the binary to init and out of the group ``killpg`` reaches. Left alone
        it keeps ~6 GB of VRAM and its port, and a sweep that restarts CARLA
        every twenty routes would accumulate one of those each time.
        """
        if self.process is not None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process = None
        self._kill_orphans()
        time.sleep(3)

    def _kill_orphans(self) -> None:
        """Kill any CarlaUE4 process still serving this instance's port."""
        marker = f"-world-port={self.port}".encode()
        for entry in pathlib.Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if marker in cmdline and b"CarlaUE4" in cmdline:
                try:
                    os.kill(int(entry.name), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    def alive(self) -> bool:
        """Whether the server is still running.

        Returns:
            True while the process has not exited.
        """
        return self.process is not None and self.process.poll() is None


def read_score(endpoint: pathlib.Path) -> dict | None:
    """Pull one route's scores out of the leaderboard's checkpoint file.

    Args:
        endpoint: The ``checkpoint_endpoint.json`` a run wrote.

    Returns:
        The score fields, or None if the file holds no finished record.
    """
    try:
        records = json.loads(endpoint.read_text())["_checkpoint"]["records"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if not records:
        return None
    record = records[0]
    scores = record.get("scores", {})
    return {
        "driving_score": scores.get("score_composed"),
        "route_completion": scores.get("score_route"),
        "infraction_penalty": scores.get("score_penalty"),
        "status": record.get("status"),
        "town": record.get("town_name"),
        "num_infractions": record.get("num_infractions"),
    }


def run_route(
    model: pathlib.Path,
    route: pathlib.Path,
    modality: str,
    severity: float,
    carla: Carla,
    tm_port: int,
    work_dir: pathlib.Path,
) -> dict | None:
    """Drive one route once and return its scores.

    Args:
        model: Checkpoint directory, holding ``config.yaml`` and one weights file.
        route: The route XML.
        modality: Which sensor to damage, or ``"none"``.
        severity: How much to damage it.
        carla: The running server.
        tm_port: Traffic-manager port for this run.
        work_dir: Where the run writes its outputs; cleared first so a stale
            result can never be read back as this run's.

    Returns:
        The scores, or None if the run produced no finished record.
    """
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    environment = dict(os.environ)
    environment.update(
        {
            "LEAD_RUNTIME_TYPE_CHECKING": "false",
            "TIMM_USE_OLD_CACHE": "1",
            "WANDB_MODE": "offline",
            "OMP_NUM_THREADS": "1",
            # ``python -m lead`` is argparse-only and rejects key=value
            # arguments, unlike the training entry point. Config for an
            # evaluation run travels in this dotlist instead.
            "LEAD_CONFIG": (
                # The demo video is on by default and costs about 200 MB and a
                # slab of CPU per route, encoded alongside the simulation it is
                # competing with. Across a 270-run sweep that is pure overhead:
                # nothing in the results table is read off a video, and each
                # one is deleted with the scratch directory on the next run.
                "evaluation.produce_demo_video=false "
                f"evaluation.inference.degrade_modality={modality} "
                f"evaluation.inference.degrade_severity={severity} "
                # Keyed on the route alone, so every checkpoint meets the
                # identical damage there and the models can be compared route
                # by route, while different routes still get different noise.
                f"evaluation.inference.degrade_seed={route_seed(route)}"
            ),
        },
    )
    command = [
        sys.executable,
        "-m",
        "lead",
        "--checkpoint",
        str(model),
        "--routes",
        str(route),
        "--bench2drive",
        "--port",
        str(carla.port),
        "--traffic-manager-port",
        str(tm_port),
        "--output-dir",
        str(work_dir),
    ]
    transcript = work_dir / "run.log"
    with transcript.open("w", encoding="utf-8") as sink:
        # Not subprocess.run: its timeout kills only the direct child, and
        # ``python -m lead`` spawns the leaderboard evaluator underneath it.
        # That grandchild would survive, keep its CARLA connection and its
        # slice of GPU memory, and drag on every remaining run of the sweep.
        # start_new_session puts the pair in one process group so both go.
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=ROOT,
            env=environment,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=_ROUTE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"      | timed out after {_ROUTE_TIMEOUT_S}s, killing the group")
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                    process.wait(timeout=30)
                    break
                except (ProcessLookupError, PermissionError):
                    break
                except subprocess.TimeoutExpired:
                    continue
            return None

    for endpoint in sorted(work_dir.rglob("checkpoint_endpoint.json")):
        scores = read_score(endpoint)
        if scores is not None:
            return scores
    if transcript.exists():
        tail = transcript.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in tail[-4:]:
            print(f"      | {line}")
    return None


def is_measurement(status: str | None) -> bool:
    """Whether a status reflects the agent's driving rather than a sim failure.

    Args:
        status: The status the leaderboard reported, if any.

    Returns:
        True when the row is a usable measurement.
    """
    if not status:
        return False
    return not any(marker in status for marker in _INFRASTRUCTURE_FAILURES)


def load_done(out: pathlib.Path) -> set[tuple[str, str, str, str]]:
    """Which runs the results CSV already holds a usable measurement for.

    A row whose status names a simulator failure is not counted, so re-running
    the sweep retries it instead of carrying the failure into the final table.
    A cell that has failed twice is the exception: it is treated as settled,
    because a third attempt costs a full route timeout and has never yet
    produced a measurement.

    Args:
        out: The results CSV.

    Returns:
        The ``(model, modality, severity, route)`` keys already measured.
    """
    if not out.exists():
        return set()
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    measured = {
        (row["model"], row["modality"], row["severity"], row["route"])
        for row in rows
        if is_measurement(row.get("status"))
    }
    # A cell that failed once may have been unlucky: a server that died, a port
    # still closing. A cell that failed twice is failing for a reason another
    # attempt will not fix, and each attempt costs the whole route timeout. It
    # is counted as settled so the budget goes to cells that can still be
    # measured; the missing row stays visible in the CSV as a failure.
    failures: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        key = (row["model"], row["modality"], row["severity"], row["route"])
        if not is_measurement(row.get("status")):
            failures[key] = failures.get(key, 0) + 1
    abandoned = {key for key, count in failures.items() if count >= 2}
    if abandoned:
        print(f"  {len(abandoned)} cell(s) abandoned after two failures")
        for model, modality, severity, route in sorted(abandoned):
            print(f"    {model} {modality}:{severity} {route}")
    return measured | abandoned


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME=DIR",
        help="Checkpoint directories to score, named for the results table.",
    )
    parser.add_argument(
        "--routes",
        type=pathlib.Path,
        required=True,
        help="File listing route XML names, one per line.",
    )
    parser.add_argument(
        "--route-dir",
        type=pathlib.Path,
        default=ROOT / "src/lead/routes/benchmark_routes/bench2drive",
        help="Directory the listed route names live in.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["none:0"],
        metavar="MODALITY:SEVERITY",
        help="Sensor conditions to score under.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="Results CSV; existing rows are kept and skipped.",
    )
    parser.add_argument(
        "--carla-root",
        type=pathlib.Path,
        default=pathlib.Path.home() / "CARLA/standard_0916",
    )
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument(
        "--restart-every",
        type=int,
        default=8,
        help="Restart CARLA after this many routes; it degrades over long runs.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Report the matrix and stop, without starting CARLA or driving.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the matrix, appending each finished route to the results CSV."""
    _unbuffer_stdout()
    args = parse_args()
    models = [pair.split("=", 1) for pair in args.models]
    conditions = [pair.split(":", 1) for pair in args.conditions]
    routes = [
        line.strip()
        for line in args.routes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Models innermost, on purpose. Model-major order means an interruption
    # at 60% leaves two models complete and the third untouched -- and the
    # third is the one the whole comparison is about. This way every prefix of
    # the sweep holds complete model-triples, so a sweep that dies early is
    # still a smaller version of the same experiment rather than a useless one.
    # It costs nothing: every route is a fresh subprocess that loads its
    # checkpoint anyway, so switching models per run is free.
    jobs = [
        (name, pathlib.Path(path), modality, severity, route)
        for modality, severity in conditions
        for route in routes
        for name, path in models
    ]
    done = load_done(args.out)
    pending = [job for job in jobs if (job[0], job[2], job[3], job[4]) not in done]
    print(
        f"{len(jobs)} runs in the matrix, {len(done)} already done, "
        f"{len(pending)} to go",
    )
    for name, _, modality, severity, _ in pending[:1]:
        print(f"  first pending: {name} at {modality}:{severity}")
    if args.plan_only:
        print("plan only; nothing started")
        return
    if not pending:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.out.exists()
    handle = args.out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=_FIELDS)
    if is_new:
        writer.writeheader()
        handle.flush()

    port = free_port(args.port)
    tm_port = free_port(port + 100)
    carla = Carla(args.carla_root, port)
    work_dir = ROOT / "outputs" / "eval_scratch"
    print(f"CARLA on {port}, traffic manager on {tm_port}")
    restart_next = False
    carla.start()

    try:
        for index, (name, model, modality, severity, route) in enumerate(pending, 1):
            if index % args.restart_every == 0 or not carla.alive() or restart_next:
                print("  restarting CARLA")
                carla.stop()
                carla.start()
                restart_next = False
            started = time.time()
            scores = run_route(
                model,
                args.route_dir / route,
                modality,
                float(severity),
                carla,
                tm_port,
                work_dir,
            )
            elapsed = round(time.time() - started, 1)
            row = {
                "model": name,
                "modality": modality,
                "severity": severity,
                "route": route,
                "seconds": elapsed,
                **(scores or dict.fromkeys(_SCORE_FIELDS)),
            }
            if scores is None:
                # Recorded rather than dropped: a route that never scored is a
                # fact about the run, and silently missing rows would look like
                # a smaller sample instead of a failure.
                row["status"] = "NoResult"
            writer.writerow(row)
            handle.flush()
            # One sim failure is usually the first of several: the server is
            # already unwell and the next runs burn twenty minutes each before
            # failing the same way. Replace it now rather than after it dies.
            restart_next = not is_measurement(row["status"])
            print(
                f"[{index}/{len(pending)}] {name} {modality}:{severity} {route} "
                f"-> DS {row['driving_score']} ({row['status']}, {elapsed}s)"
                f"{'  [sim failure, will restart]' if restart_next else ''}",
            )
    finally:
        carla.stop()
        handle.close()
        print(f"results in {args.out}")


if __name__ == "__main__":
    main()
