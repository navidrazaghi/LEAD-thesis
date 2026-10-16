"""Make the LiDAR's own degradation attributes configurable.

Three edits, and the third is the one that makes the other two matter.

Weather never touches the ray-cast LiDAR, so every LiDAR corruption this
project has measured was applied to the rasterised grid after the sweep had
already been taken. Corrupting the sensor instead needs the attributes CARLA
exposes -- beam count, dropoff, atmospheric attenuation, range noise -- to be
reachable from the config.

They were not, and adding them to lidar_sensor_setup alone would have changed
nothing: the Bench2Drive wrapper builds its attribute dict from scratch and
hardcodes every LiDAR value, so whatever the agent asks for is discarded. The
third edit makes the wrapper read the spec and fall back to the value it used
to hardcode.

Every default equals what the wrapper writes today, so a run that overrides
nothing behaves exactly as before.
"""

import io
import pathlib
import sys

ROOT = pathlib.Path.home() / "LEAD/lead"
RIG = ROOT / "src/lead/config/expert/sensor_rig_config.py"
SETUP = ROOT / "src/lead/common/sensors/av_sensor_setup.py"
WRAPPER = (ROOT / "3rd_party/leaderboard/bench2drive/leaderboard/leaderboard"
           / "autoagents/agent_wrapper.py")

RIG_ANCHOR = """    # --- Camera Configuration ---"""

RIG_BLOCK = '''    # --- LiDAR sensor attributes ---
    # What the sensor itself does to the sweep, as opposed to what the
    # degradation curriculum does to the rasterised grid afterwards. Every
    # default here is the value the evaluation harness used to hardcode, so a
    # run that overrides none of them is unchanged. Overriding one is how a
    # sensor-level corruption suite is built:
    #
    #   LEAD_CONFIG="expert.sensor_rig.lidar_channels=16"
    #
    # Note that CARLA's own defaults are already a corruption: 45 per cent of
    # returns are dropped before anything in this project touches them.

    # Range in metres.
    lidar_range_meter: float = 85.0
    # Sweeps per second.
    lidar_rotation_frequency: float = 10.0
    # Beam count. Halving it is the cheapest sensor a rig could ship.
    lidar_channels: int = 64
    # Vertical field of view, in degrees above and below the horizon. The
    # lower bound sets how close to the vehicle the ground is first seen.
    lidar_upper_fov: float = 10.0
    lidar_lower_fov: float = -30.0
    # Points emitted per second, over all beams.
    lidar_points_per_second: int = 600000
    # Atmospheric attenuation per metre. Rain and fog raise this in reality;
    # CARLA holds it fixed whatever the weather preset says.
    lidar_atmosphere_attenuation_rate: float = 0.004
    # Fraction of returns dropped regardless of intensity.
    lidar_dropoff_general_rate: float = 0.45
    # Intensity below which the dropoff applies at all.
    lidar_dropoff_intensity_limit: float = 0.8
    # Probability of dropping a zero-intensity return.
    lidar_dropoff_zero_intensity: float = 0.4
    # Gaussian noise on the measured distance, in metres.
    lidar_noise_stddev: float = 0.0

'''

SETUP_ANCHOR = """                "id": lidar_id,
            },
        )"""

SETUP_BLOCK = """                "id": lidar_id,
                # Sensor attributes travel with the spec so the harness can
                # honour them; see agent_wrapper._preprocess_sensor_spec.
                "range": config.sensor_rig.lidar_range_meter,
                "rotation_frequency": config.sensor_rig.lidar_rotation_frequency,
                "channels": config.sensor_rig.lidar_channels,
                "upper_fov": config.sensor_rig.lidar_upper_fov,
                "lower_fov": config.sensor_rig.lidar_lower_fov,
                "points_per_second": config.sensor_rig.lidar_points_per_second,
                "atmosphere_attenuation_rate": (
                    config.sensor_rig.lidar_atmosphere_attenuation_rate
                ),
                "dropoff_general_rate": config.sensor_rig.lidar_dropoff_general_rate,
                "dropoff_intensity_limit": (
                    config.sensor_rig.lidar_dropoff_intensity_limit
                ),
                "dropoff_zero_intensity": (
                    config.sensor_rig.lidar_dropoff_zero_intensity
                ),
                "noise_stddev": config.sensor_rig.lidar_noise_stddev,
            },
        )"""

WRAPPER_ANCHOR = """            attributes['range'] = str(85)
            attributes['rotation_frequency'] = str(10)
            attributes['channels'] = str(64)
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(600000)
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(0.45)
            attributes['dropoff_intensity_limit'] = str(0.8)
            attributes['dropoff_zero_intensity'] = str(0.4)"""

WRAPPER_BLOCK = """            # Read what the agent asked for and fall back to the value this
            # branch used to hardcode. Without this the sensor attributes are
            # unreachable from config and every LiDAR corruption has to be
            # applied to the raster after the sweep is already taken.
            attributes['range'] = str(sensor_spec.get('range', 85))
            attributes['rotation_frequency'] = str(
                sensor_spec.get('rotation_frequency', 10))
            attributes['channels'] = str(sensor_spec.get('channels', 64))
            attributes['upper_fov'] = str(sensor_spec.get('upper_fov', 10))
            attributes['lower_fov'] = str(sensor_spec.get('lower_fov', -30))
            attributes['points_per_second'] = str(
                sensor_spec.get('points_per_second', 600000))
            attributes['atmosphere_attenuation_rate'] = str(
                sensor_spec.get('atmosphere_attenuation_rate', 0.004))
            attributes['dropoff_general_rate'] = str(
                sensor_spec.get('dropoff_general_rate', 0.45))
            attributes['dropoff_intensity_limit'] = str(
                sensor_spec.get('dropoff_intensity_limit', 0.8))
            attributes['dropoff_zero_intensity'] = str(
                sensor_spec.get('dropoff_zero_intensity', 0.4))
            attributes['noise_stddev'] = str(sensor_spec.get('noise_stddev', 0.0))"""


def patch(path: pathlib.Path, anchor: str, replacement: str) -> None:
    """Replace one unique anchor, refusing anything ambiguous."""
    text = io.open(path, encoding="utf-8").read()
    found = text.count(anchor)
    if found != 1:
        sys.exit(f"FATAL: {path.name}: anchor appears {found} times, expected 1")
    io.open(path, "w", encoding="utf-8", newline="\n").write(
        text.replace(anchor, replacement))
    print(f"  patched {path.relative_to(ROOT)}")


patch(RIG, RIG_ANCHOR, RIG_BLOCK + RIG_ANCHOR)
patch(SETUP, SETUP_ANCHOR, SETUP_BLOCK)
patch(WRAPPER, WRAPPER_ANCHOR, WRAPPER_BLOCK)
print("done")
