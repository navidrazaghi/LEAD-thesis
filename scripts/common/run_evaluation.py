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

ROOT = pathlib.Path(__file__).resolve().parents[2]
# CARLA needs this long to answer RPC after launch on this machine; it is a
# ceiling, not a sleep, so a fast boot costs nothing.
_CARLA_BOOT_TIMEOUT_S = 180
# A route that has not finished by now is wedged; the observed time is about
# five minutes, so this is generous rather than tight.
_ROUTE_TIMEOUT_S = 1800
_SCORE_FIELDS = (
    "driving_score",
    "route_completion",
    "infraction_penalty",
    "status",
    "town",
    "num_infractions",
)
_FIELDS = ("model", "modality", "severity", "route", *_SCORE_FIELDS, "seconds")


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
        """Kill the server and every process it spawned."""
        if self.process is None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process = None

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
        f"evaluation.inference.degrade_modality={modality}",
        f"evaluation.inference.degrade_severity={severity}",
    ]
    try:
        subprocess.run(  # noqa: S603
            command,
            cwd=ROOT,
            env=environment,
            timeout=_ROUTE_TIMEOUT_S,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None

    for endpoint in sorted(work_dir.rglob("checkpoint_endpoint.json")):
        scores = read_score(endpoint)
        if scores is not None:
            return scores
    return None


def load_done(out: pathlib.Path) -> set[tuple[str, str, str, str]]:
    """Which runs the results CSV already holds.

    Args:
        out: The results CSV.

    Returns:
        The ``(model, modality, severity, route)`` keys already written.
    """
    if not out.exists():
        return set()
    with out.open(newline="", encoding="utf-8") as handle:
        return {
            (row["model"], row["modality"], row["severity"], row["route"])
            for row in csv.DictReader(handle)
        }


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
        default=20,
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
    args = parse_args()
    models = [pair.split("=", 1) for pair in args.models]
    conditions = [pair.split(":", 1) for pair in args.conditions]
    routes = [
        line.strip()
        for line in args.routes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    jobs = [
        (name, pathlib.Path(path), modality, severity, route)
        for name, path in models
        for modality, severity in conditions
        for route in routes
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
    carla.start()

    try:
        for index, (name, model, modality, severity, route) in enumerate(pending, 1):
            if index % args.restart_every == 0 or not carla.alive():
                print("  restarting CARLA")
                carla.stop()
                carla.start()
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
            print(
                f"[{index}/{len(pending)}] {name} {modality}:{severity} {route} "
                f"-> DS {row['driving_score']} ({row['status']}, {elapsed}s)",
            )
    finally:
        carla.stop()
        handle.close()
        print(f"results in {args.out}")


if __name__ == "__main__":
    main()
