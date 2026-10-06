"""Tests for the configurable LiDAR sensor attributes.

The property that matters most here is the boring one: with nothing
overridden, the sensor is spawned with exactly the values the harness used to
hardcode. Every closed-loop number this project has collected was measured
under those values, and a silent change to any of them would make the next
run incomparable with all of them without anything failing to announce it.

The rest check that an override actually reaches the spec, since the whole
point of the change is to make a sensor-level corruption suite reachable from
config rather than from a code fork.
"""

import pytest

from lead.common.sensors.av_sensor_setup import lidar_sensor_setup
from lead.config import LeadConfig

# What the Bench2Drive wrapper wrote before the spec was consulted. These are
# CARLA's own defaults for everything except range, rotation frequency and
# points per second, which the harness raised.
HARNESS_DEFAULTS = {
    "range": 85.0,
    "rotation_frequency": 10.0,
    "channels": 64,
    "upper_fov": 10.0,
    "lower_fov": -30.0,
    "points_per_second": 600000,
    "atmosphere_attenuation_rate": 0.004,
    "dropoff_general_rate": 0.45,
    "dropoff_intensity_limit": 0.8,
    "dropoff_zero_intensity": 0.4,
    "noise_stddev": 0.0,
}


class TestDefaultsAreUnchanged:
    """Nothing moves unless a run asks for it."""

    def test_every_attribute_matches_what_the_harness_hardcoded(self) -> None:
        """A run that overrides nothing must spawn the sensor it always did."""
        specs = lidar_sensor_setup(LeadConfig().expert)
        for spec in specs:
            for name, expected in HARNESS_DEFAULTS.items():
                assert spec[name] == pytest.approx(expected), name

    def test_both_lidars_carry_the_attributes(self) -> None:
        """The rig has two, and a corruption has to reach both."""
        specs = lidar_sensor_setup(LeadConfig().expert)
        assert len(specs) == 2
        for spec in specs:
            assert set(HARNESS_DEFAULTS) <= set(spec)

    def test_mounting_pose_is_untouched(self) -> None:
        """The attributes are new; the geometry is not."""
        config = LeadConfig().expert
        first, second = lidar_sensor_setup(config)
        assert [first["x"], first["y"], first["z"]] == config.sensor_rig.lidar_pos_1
        assert [second["x"], second["y"], second["z"]] == config.sensor_rig.lidar_pos_2


class TestOverridesReachTheSpec:
    """The point of the change: a corruption is a config line."""

    def test_beam_count(self) -> None:
        """Halving the beams is the cheapest sensor a rig could ship."""
        config = LeadConfig()
        config.expert.sensor_rig.lidar_channels = 16
        for spec in lidar_sensor_setup(config.expert):
            assert spec["channels"] == 16

    def test_dropoff(self) -> None:
        """Sparse returns, the failure distance and rain produce."""
        config = LeadConfig()
        config.expert.sensor_rig.lidar_dropoff_general_rate = 0.8
        for spec in lidar_sensor_setup(config.expert):
            assert spec["dropoff_general_rate"] == pytest.approx(0.8)

    def test_atmospheric_attenuation(self) -> None:
        """The one CARLA leaves fixed however heavy the weather preset is."""
        config = LeadConfig()
        config.expert.sensor_rig.lidar_atmosphere_attenuation_rate = 0.05
        for spec in lidar_sensor_setup(config.expert):
            assert spec["atmosphere_attenuation_rate"] == pytest.approx(0.05)

    def test_range_noise(self) -> None:
        """Spurious returns rather than missing ones."""
        config = LeadConfig()
        config.expert.sensor_rig.lidar_noise_stddev = 0.1
        for spec in lidar_sensor_setup(config.expert):
            assert spec["noise_stddev"] == pytest.approx(0.1)

    def test_a_clean_lidar_is_reachable_too(self) -> None:
        """CARLA's default already drops 45 per cent; zero is a valid setting."""
        config = LeadConfig()
        config.expert.sensor_rig.lidar_dropoff_general_rate = 0.0
        for spec in lidar_sensor_setup(config.expert):
            assert spec["dropoff_general_rate"] == pytest.approx(0.0)
