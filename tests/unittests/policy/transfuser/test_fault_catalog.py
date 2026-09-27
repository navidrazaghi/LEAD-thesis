"""The structured fault catalogue: each property a wrong score would hide.

An evaluation fault that silently does nothing, or does the wrong thing, still
produces a number, and the number looks like a measurement. These tests pin the
properties whose failure would go unnoticed that way: that a fault damages its
own sensor and only that one, that zero severity is exact, that a fault's
geometry holds still across the ticks of a route, that spatial faults respect
the seams between stitched cameras, that LiDAR faults are structured by range
and bearing from the sensor rather than uniform, and that the evaluator routes
every catalogue name to the fault rather than to an undamaged run.
"""

import importlib.util
import math
import pathlib

import pytest
import torch

from lead.policy.transfuser.utils.fault_catalog import (
    CAMERA_FAULTS,
    FAULT_CATALOG,
    LIDAR_FAULTS,
    FaultGeometry,
    apply_fault,
)
from lead.policy.transfuser.utils.sensor_degradation import degrade_batch_family

GEOMETRY = FaultGeometry()  # LEAD's layout: 3 cameras, 320 x 384 raster at 4 px/m
H, W = 48, 3 * 64  # a small stitched image, three 64-wide tiles
BEV_H, BEV_W = 320, 384
EGO_ROW, EGO_COL = 160, 128


def _batch(batch_size: int = 2, uint8: bool = False) -> dict:
    """A batch with a textured image and a partly occupied LiDAR raster."""
    g = torch.Generator().manual_seed(7)
    rgb = torch.rand(batch_size, 3, H, W, generator=g) * 200.0 + 20.0
    lidar = (torch.rand(batch_size, 1, BEV_H, BEV_W, generator=g) < 0.3).float() * 0.6
    return {
        "rgb": rgb.round().to(torch.uint8) if uint8 else rgb,
        "rasterized_lidar": lidar,
        "ego_speed": torch.full((batch_size, 1), 5.0),
    }


def _full(severity: float, batch_size: int = 2) -> torch.Tensor:
    return torch.full((batch_size,), severity)


@pytest.mark.parametrize("fault", FAULT_CATALOG)
def test_a_fault_damages_its_own_sensor_and_leaves_the_other_alone(fault: str) -> None:
    batch = _batch()
    before = {k: v.clone() for k, v in batch.items()}
    after = apply_fault(batch, fault, _full(1.0), torch.Generator().manual_seed(0), 3, GEOMETRY)
    mine, other = ("rgb", "rasterized_lidar") if fault in CAMERA_FAULTS else ("rasterized_lidar", "rgb")
    assert not torch.equal(before[mine], after[mine]), f"{fault} did not change {mine}"
    assert torch.equal(before[other], after[other]), f"{fault} touched {other}"


@pytest.mark.parametrize("fault", FAULT_CATALOG)
def test_zero_severity_is_exactly_a_no_op(fault: str) -> None:
    batch = _batch()
    before = {k: v.clone() for k, v in batch.items()}
    after = apply_fault(batch, fault, _full(0.0), torch.Generator().manual_seed(0), 3, GEOMETRY)
    for key in before:
        assert torch.equal(before[key], after[key])


@pytest.mark.parametrize("fault", FAULT_CATALOG)
def test_outputs_stay_in_range_and_keep_their_dtype(fault: str) -> None:
    batch = _batch(uint8=True)
    after = apply_fault(batch, fault, _full(1.0), torch.Generator().manual_seed(1), 3, GEOMETRY)
    assert after["rgb"].dtype == torch.uint8
    assert after["rgb"].shape == (2, 3, H, W)
    lidar = after["rasterized_lidar"]
    assert lidar.dtype == torch.float32
    assert float(lidar.min()) >= 0.0 and float(lidar.max()) <= 1.0


def test_a_lens_fault_holds_still_across_ticks() -> None:
    """Mud does not move: same route seed, different tick streams, same image."""
    a = apply_fault(_batch(), "cam_local_occlusion", _full(0.8), torch.Generator().manual_seed(1), 42, GEOMETRY)
    b = apply_fault(_batch(), "cam_local_occlusion", _full(0.8), torch.Generator().manual_seed(2), 42, GEOMETRY)
    assert torch.equal(a["rgb"], b["rgb"])


def test_a_blocked_lidar_wedge_holds_still_across_ticks() -> None:
    def blinded(tick_seed: int) -> torch.Tensor:
        batch = {"rasterized_lidar": torch.full((1, 1, BEV_H, BEV_W), 0.5)}
        out = apply_fault(batch, "lid_frustum_occlusion", _full(0.6, 1),
                          torch.Generator().manual_seed(tick_seed), 42, GEOMETRY)
        # Inside 4 m the wedge carries random backscatter that changes every
        # tick by design; the blind geometry is what must hold still.
        return (out["rasterized_lidar"][0, 0] == 0) & (_distance_m() > 5.0)

    assert torch.equal(blinded(1), blinded(2))


def test_a_different_route_gets_a_different_fault() -> None:
    a = apply_fault(_batch(), "cam_local_occlusion", _full(0.8), None, 1, GEOMETRY)
    b = apply_fault(_batch(), "cam_local_occlusion", _full(0.8), None, 2, GEOMETRY)
    assert not torch.equal(a["rgb"], b["rgb"])


def test_a_lens_fault_stays_on_one_camera() -> None:
    batch = _batch(batch_size=1)
    before = batch["rgb"].clone()
    after = apply_fault(batch, "cam_local_occlusion", _full(1.0, 1), None, 5, GEOMETRY)["rgb"]
    changed = [not torch.equal(before[..., k * 64:(k + 1) * 64], after[..., k * 64:(k + 1) * 64])
               for k in range(3)]
    assert sum(changed) == 1, changed


@pytest.mark.parametrize("fault", ["cam_motion_blur", "cam_ghosting"])
def test_spatial_faults_do_not_bleed_across_the_camera_seams(fault: str) -> None:
    """A black camera between two white ones must stay black after blur or ghosting."""
    rgb = torch.full((1, 3, H, W), 255.0)
    rgb[..., 64:128] = 0.0
    out = apply_fault({"rgb": rgb}, fault, _full(1.0, 1), torch.Generator().manual_seed(0), 3, GEOMETRY)["rgb"]
    assert float(out[..., 64:128].max()) == 0.0


def _distance_m() -> torch.Tensor:
    rows = torch.arange(BEV_H).view(-1, 1).float()
    cols = torch.arange(BEV_W).view(1, -1).float()
    return torch.sqrt(((rows - EGO_ROW) / 4.0) ** 2 + ((cols - EGO_COL) / 4.0) ** 2)


def test_range_dropout_loses_far_returns_before_near_ones() -> None:
    lidar = torch.full((1, 1, BEV_H, BEV_W), 0.6)
    out = apply_fault({"rasterized_lidar": lidar}, "lid_range_dropout", _full(1.0, 1),
                      torch.Generator().manual_seed(0), 3, GEOMETRY)["rasterized_lidar"][0, 0]
    d = _distance_m()
    near_kept = ((out > 0) & (d > 8) & (d < 15)).sum() / ((d > 8) & (d < 15)).sum()
    far_kept = ((out > 0) & (d > 55)).sum() / (d > 55).sum()
    assert float(near_kept) > float(far_kept) + 0.3


def test_the_blind_wedge_has_its_apex_at_the_sensor() -> None:
    lidar = torch.full((1, 1, BEV_H, BEV_W), 0.6)
    out = apply_fault({"rasterized_lidar": lidar}, "lid_frustum_occlusion", _full(0.5, 1),
                      torch.Generator().manual_seed(0), 9, GEOMETRY)["rasterized_lidar"][0, 0]
    rows = torch.arange(BEV_H).view(-1, 1).float() - EGO_ROW
    cols = torch.arange(BEV_W).view(1, -1).float() - EGO_COL
    bearing = torch.atan2(rows, cols)
    blind = (out == 0) & (_distance_m() > 5.0)
    assert blind.any()
    angles = bearing[blind]
    # Angular width of the blind cells as seen from the sensor. The raster is
    # not symmetric about the ego (64 m ahead, 32 m behind), so the wedge is
    # clipped unevenly and its mean bearing is not its centre; the full width
    # is what a wedge from the sensor bounds, and a blob elsewhere would not.
    mean = torch.atan2(torch.sin(angles).mean(), torch.cos(angles).mean())
    spread = torch.remainder(angles - mean + math.pi, 2 * math.pi) - math.pi
    width = float(spread.max() - spread.min())
    assert width <= 2 * math.radians(8 + 40 * 0.5) + 0.03


def test_speckle_only_adds_returns_where_there_were_none() -> None:
    batch = _batch(batch_size=1)
    before = batch["rasterized_lidar"].clone()
    after = apply_fault(batch, "lid_local_speckle", _full(1.0, 1), torch.Generator().manual_seed(0),
                        3, GEOMETRY)["rasterized_lidar"]
    occupied = before > 0
    assert torch.equal(before[occupied], after[occupied])
    assert (after[~occupied] > 0).any()


def test_the_dispatcher_reaches_the_catalogue_and_still_rejects_unknown_names() -> None:
    batch = _batch()
    before = batch["rgb"].clone()
    out = degrade_batch_family(batch, "cam_night_lowlight", 1.0, torch.Generator().manual_seed(0),
                               persistent_seed=3, geometry=GEOMETRY)
    assert float(out["rgb"].mean()) < float(before.mean())
    with pytest.raises(ValueError):
        degrade_batch_family(_batch(), "cam_not_a_fault", 1.0, torch.Generator().manual_seed(0))


def test_the_evaluator_routes_every_catalogue_name_to_the_fault() -> None:
    """A name missing from run_evaluation would be sent as a modality and score clean."""
    path = pathlib.Path(__file__).resolve().parents[4] / "scripts/common/run_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(FAULT_CATALOG) <= set(module._DEPLOYMENT_FAMILIES)
    assert set(CAMERA_FAULTS) | set(LIDAR_FAULTS) == set(FAULT_CATALOG)
