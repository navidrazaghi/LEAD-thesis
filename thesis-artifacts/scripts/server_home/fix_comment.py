"""Rephrase one comment so ruff stops reading it as commented-out code."""

import io
import pathlib
import sys

path = pathlib.Path.home() / "LEAD/lead/src/lead/config/expert/sensor_rig_config.py"
text = io.open(path, encoding="utf-8").read()

old = (
    "    # sensor-level corruption suite is built:\n"
    "    #\n"
    '    #   LEAD_CONFIG="expert.sensor_rig.lidar_channels=16"\n'
    "    #\n"
    "    # Note that CARLA's own defaults are already a corruption: 45 per cent of\n"
    "    # returns are dropped before anything in this project touches them."
)
new = (
    "    # sensor-level corruption suite is built: one override per axis,\n"
    "    # through the same config path every other setting uses, rather than\n"
    "    # a code fork.\n"
    "    #\n"
    "    # Note that CARLA's own defaults are already a corruption: 45 per cent of\n"
    "    # returns are dropped before anything in this project touches them."
)

if text.count(old) != 1:
    sys.exit(f"FATAL: anchor appears {text.count(old)} times, expected 1")
io.open(path, "w", encoding="utf-8", newline="\n").write(text.replace(old, new))
print("rephrased")
