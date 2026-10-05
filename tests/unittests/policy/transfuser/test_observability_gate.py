"""Tests for the fusion modality gate and the sensor-degradation curriculum."""

import pytest
import torch
import torch.nn.functional as F

from lead.config import LeadConfig, load_lead_config
from lead.policy.transfuser.dataloader.observability import (
    NUM_OBSERVABILITY_CHANNELS,
    ObservabilityChannel,
)
from lead.policy.transfuser.encoder.deformable_attention import (
    MultiScaleDeformableAttention,
)
from lead.policy.transfuser.encoder.observability_gate import (
    ObservabilityGate,
    ObservabilityTokenTargets,
    gate_loss,
)
from lead.policy.transfuser.utils.sensor_degradation import (
    apply_sensor_degradation,
    degrade_batch,
    degrade_camera,
)

IMAGE_SHAPE = (12, 36)
BEV_SHAPE = (10, 12)
SPATIAL_SHAPES = (IMAGE_SHAPE, BEV_SHAPE)
NUM_IMAGE_TOKENS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
NUM_TOKENS = NUM_IMAGE_TOKENS + BEV_SHAPE[0] * BEV_SHAPE[1]


@pytest.fixture
def lead_config() -> LeadConfig:
    """Fixture providing the root config tree."""
    return load_lead_config()


@pytest.fixture
def attention() -> MultiScaleDeformableAttention:
    """Fixture providing a deformable operator over both anchor grids."""
    torch.manual_seed(0)
    return MultiScaleDeformableAttention(
        n_embd=64,
        n_head=4,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        spatial_shapes=SPATIAL_SHAPES,
        num_points=4,
    ).eval()


class TestGateInitialization:
    """Tests that a gated model starts where the ungated one does."""

    def test_projection_starts_at_zero(self) -> None:
        gate = ObservabilityGate(64, 2, load_lead_config())
        assert (gate.head.weight == 0.0).all()
        assert (gate.head.bias == 0.0).all()

    def test_leaves_the_operator_unchanged(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        gate = ObservabilityGate(64, 2, load_lead_config())
        x = torch.randn(2, NUM_TOKENS, 64)

        torch.testing.assert_close(attention(x, gate(x)), attention(x))

    def test_predicts_one_logit_per_token_per_modality(self) -> None:
        gate = ObservabilityGate(64, 2, load_lead_config())
        assert gate(torch.randn(2, NUM_TOKENS, 64)).shape == (2, NUM_TOKENS, 2)


class TestGateEffect:
    """Tests that the gate redirects which modality a query reads."""

    def test_a_negative_logit_suppresses_that_modality(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        x = torch.randn(2, NUM_TOKENS, 64)
        suppress_image = torch.zeros(2, NUM_TOKENS, 2)
        suppress_image[:, :, 0] = -12.0

        with torch.no_grad():
            logits = attention.attention_weights(x).view(2, NUM_TOKENS, 4, 2, 4)
            open_weights = F.softmax(logits.flatten(-2), -1).view(
                2, NUM_TOKENS, 4, 2, 4
            )
            shut_weights = F.softmax(
                (logits + suppress_image[:, :, None, :, None]).flatten(-2),
                -1,
            ).view(2, NUM_TOKENS, 4, 2, 4)

        assert open_weights[..., 0, :].sum(-1).mean() > 0.4
        assert shut_weights[..., 0, :].sum(-1).mean() < 0.01

    def test_the_gated_output_stays_finite_and_differs(
        self,
        attention: MultiScaleDeformableAttention,
    ) -> None:
        x = torch.randn(2, NUM_TOKENS, 64)
        bias = torch.zeros(2, NUM_TOKENS, 2)
        bias[:, :, 0] = -12.0

        gated = attention(x, bias)

        assert torch.isfinite(gated).all()
        assert not torch.allclose(gated, attention(x))


class TestTokenTargets:
    """Tests for reducing the dense targets onto the fusion token grids."""

    @pytest.fixture
    def token_targets(self, lead_config: LeadConfig) -> ObservabilityTokenTargets:
        return ObservabilityTokenTargets(lead_config)

    @pytest.fixture
    def dense_shape(self, lead_config: LeadConfig) -> tuple[int, int, int, int]:
        config = lead_config.policy.transfuser
        return (
            1,
            NUM_OBSERVABILITY_CHANNELS,
            config.lidar_height_pixel // config.bev_downsample_factor,
            config.lidar_width_pixel // config.bev_downsample_factor,
        )

    def test_bev_tokens_average_only_the_supervised_cells(
        self,
        token_targets: ObservabilityTokenTargets,
        dense_shape: tuple[int, int, int, int],
    ) -> None:
        target = torch.zeros(dense_shape)
        mask = torch.zeros(dense_shape)
        # The first BEV token covers exactly the first 8x8 block of cells; only
        # a quarter of it is supervised, and at a value the block mean would
        # not produce.
        target[0, :, 0:4, 0:4] = 0.75
        mask[0, :, 0:4, 0:4] = 1.0

        token_target, token_mask = token_targets(target, mask)

        assert token_mask[0, NUM_IMAGE_TOKENS].all()
        torch.testing.assert_close(
            token_target[0, NUM_IMAGE_TOKENS],
            torch.full((NUM_OBSERVABILITY_CHANNELS,), 0.75),
        )

    def test_a_bev_token_with_no_measurement_is_unsupervised(
        self,
        token_targets: ObservabilityTokenTargets,
        dense_shape: tuple[int, int, int, int],
    ) -> None:
        target = torch.zeros(dense_shape)
        mask = torch.zeros(dense_shape)
        mask[0, :, 0:8, 0:8] = 1.0

        _, token_mask = token_targets(target, mask)

        assert token_mask[0, NUM_IMAGE_TOKENS].all()
        assert token_mask[0, NUM_IMAGE_TOKENS + 50].sum() == 0.0

    def test_image_tokens_read_where_their_rays_land(
        self,
        lead_config: LeadConfig,
        token_targets: ObservabilityTokenTargets,
        dense_shape: tuple[int, int, int, int],
    ) -> None:
        config = lead_config.policy.transfuser
        target = torch.zeros(dense_shape)
        mask = torch.zeros(dense_shape)
        # A patch ahead of the ego, where the forward cameras actually look.
        cells_per_meter = dense_shape[3] / (
            config.bev_max_x_meter - config.bev_min_x_meter
        )
        near = int((8.0 - config.bev_min_x_meter) * cells_per_meter)
        far = int((38.0 - config.bev_min_x_meter) * cells_per_meter)
        rows_per_meter = dense_shape[2] / (
            config.bev_max_y_meter - config.bev_min_y_meter
        )
        left = int((-10.0 - config.bev_min_y_meter) * rows_per_meter)
        right = int((10.0 - config.bev_min_y_meter) * rows_per_meter)
        target[0, :, left:right, near:far] = 0.6
        mask[0, :, left:right, near:far] = 1.0

        token_target, token_mask = token_targets(target, mask)

        supervised = token_mask[0, :NUM_IMAGE_TOKENS, 1] > 0
        assert supervised.any(), "forward image tokens must pick the patch up"
        values = token_target[0, :NUM_IMAGE_TOKENS, 1][supervised]
        torch.testing.assert_close(values, torch.full_like(values, 0.6))

    def test_image_tokens_above_the_horizon_are_unsupervised(
        self,
        lead_config: LeadConfig,
        token_targets: ObservabilityTokenTargets,
        dense_shape: tuple[int, int, int, int],
    ) -> None:
        config = lead_config.policy.transfuser
        _, token_mask = token_targets(
            torch.ones(dense_shape),
            torch.ones(dense_shape),
        )
        rows = token_mask[0, :NUM_IMAGE_TOKENS, 1].reshape(
            config.img_vert_anchors,
            config.img_horz_anchors,
        )
        # The cameras carry no pitch, so the horizon is the middle of the image.
        assert rows[: config.img_vert_anchors // 2].sum() == 0.0
        assert rows[config.img_vert_anchors // 2 :].sum() > 0.0


class TestGateLoss:
    """Tests for the masked supervision of every block's gate."""

    def test_unsupervised_tokens_contribute_nothing(self) -> None:
        target = torch.zeros(1, NUM_TOKENS, 2)
        mask = torch.zeros(1, NUM_TOKENS, 2)
        mask[0, 5] = 1.0
        logits = torch.full((1, NUM_TOKENS, 2), 20.0)
        logits[0, 5] = -20.0

        assert gate_loss([logits, logits], target, mask).item() == pytest.approx(
            0.0,
            abs=1e-6,
        )

    def test_supervised_tokens_drive_the_loss(self) -> None:
        target = torch.zeros(1, NUM_TOKENS, 2)
        mask = torch.zeros(1, NUM_TOKENS, 2)
        mask[0, 5] = 1.0
        logits = torch.full((1, NUM_TOKENS, 2), 20.0)

        assert gate_loss([logits], target, mask).item() > 10.0

    def test_every_block_receives_gradient(self) -> None:
        blocks = [torch.randn(1, NUM_TOKENS, 2, requires_grad=True) for _ in range(3)]

        gate_loss(
            blocks,
            torch.rand(1, NUM_TOKENS, 2),
            torch.ones(1, NUM_TOKENS, 2),
        ).backward()

        assert all(block.grad is not None for block in blocks)
        assert all(torch.isfinite(block.grad).all() for block in blocks)

    def test_nothing_supervised_is_finite(self) -> None:
        loss = gate_loss(
            [torch.randn(1, NUM_TOKENS, 2)],
            torch.zeros(1, NUM_TOKENS, 2),
            torch.zeros(1, NUM_TOKENS, 2),
        )
        assert torch.isfinite(loss)


class TestSensorDegradation:
    """Tests for the curriculum that teaches the gate what damage looks like."""

    @pytest.fixture
    def batch(self) -> dict:
        torch.manual_seed(3)
        return {
            "rgb": torch.full((8, 3, 32, 96), 200, dtype=torch.uint8),
            "rasterized_lidar": torch.rand(8, 1, 64, 96),
            "observability": torch.ones(8, NUM_OBSERVABILITY_CHANNELS, 20, 24),
        }

    def test_degrades_one_modality_per_sample(self, batch: dict) -> None:
        # The gate exists to move reading onto an intact modality, so leaving
        # one intact is the property that makes the curriculum trainable.
        out = apply_sensor_degradation(batch, probability=1.0, max_severity=1.0)
        camera = out["observability"][:, ObservabilityChannel.CAMERA].amax(dim=(1, 2))
        lidar = out["observability"][:, ObservabilityChannel.LIDAR].amax(dim=(1, 2))

        assert torch.all((camera == 1.0) | (lidar == 1.0))

    def test_a_lower_camera_target_means_a_more_damaged_image(
        self,
        batch: dict,
    ) -> None:
        out = apply_sensor_degradation(batch, probability=1.0, max_severity=1.0)
        camera = out["observability"][:, ObservabilityChannel.CAMERA].amax(dim=(1, 2))
        brightness = out["rgb"].float().mean(dim=(1, 2, 3))

        hit = camera < 0.999
        if not hit.any():
            pytest.skip("no camera was degraded in this draw")
        order = torch.argsort(camera[hit])
        assert brightness[hit][order][0] <= brightness[hit][order][-1] + 1e-3

    def test_a_lower_lidar_target_means_a_thinner_sweep(self, batch: dict) -> None:
        out = apply_sensor_degradation(batch, probability=1.0, max_severity=1.0)
        lidar = out["observability"][:, ObservabilityChannel.LIDAR].amax(dim=(1, 2))
        occupancy = (out["rasterized_lidar"] > 0).float().mean(dim=(1, 2, 3))

        hit = lidar < 0.999
        if not hit.any():
            pytest.skip("no lidar was degraded in this draw")
        order = torch.argsort(lidar[hit])
        assert occupancy[hit][order][0] <= occupancy[hit][order][-1] + 1e-3

    def test_zero_probability_is_a_no_op(self, batch: dict) -> None:
        original = {key: value.clone() for key, value in batch.items()}

        out = apply_sensor_degradation(batch, probability=0.0, max_severity=1.0)

        for key, value in original.items():
            assert torch.equal(out[key], value), key

    def test_a_batch_without_sensors_is_left_alone(self) -> None:
        batch = {"observability": torch.ones(2, 2, 4, 4)}
        out = apply_sensor_degradation(batch, probability=1.0, max_severity=1.0)
        torch.testing.assert_close(out["observability"], torch.ones(2, 2, 4, 4))


class TestGateLossWeight:
    """Tests for how the gate's task weight follows its toggles."""

    def test_zero_without_the_observability_targets(
        self,
        lead_config: LeadConfig,
    ) -> None:
        config = lead_config.policy.transfuser
        config.use_observability = False
        config.use_observability_gate = True

        assert config.per_task_loss_weights(0)["loss_observability_gate"] == 0.0

    def test_zero_while_the_gate_is_off(self, lead_config: LeadConfig) -> None:
        config = lead_config.policy.transfuser
        config.use_observability = True
        config.use_observability_gate = False

        assert config.per_task_loss_weights(0)["loss_observability_gate"] == 0.0

    def test_nonzero_with_both_on(self, lead_config: LeadConfig) -> None:
        config = lead_config.policy.transfuser
        config.use_observability = True
        config.use_observability_gate = True

        assert config.per_task_loss_weights(0)["loss_observability_gate"] > 0.0


class TestInferenceDegradation:
    """Tests for the fixed degradation that traces the robustness curve."""

    @pytest.fixture
    def batch(self) -> dict:
        torch.manual_seed(5)
        return {
            "rgb": torch.full((4, 3, 32, 96), 200, dtype=torch.uint8),
            "rasterized_lidar": torch.rand(4, 1, 64, 96),
        }

    def test_none_is_a_no_op(self, batch: dict) -> None:
        original = {key: value.clone() for key, value in batch.items()}
        out = degrade_batch(batch, "none", 1.0)
        for key, value in original.items():
            assert torch.equal(out[key], value), key

    def test_zero_severity_is_a_no_op(self, batch: dict) -> None:
        original = {key: value.clone() for key, value in batch.items()}
        out = degrade_batch(batch, "camera", 0.0)
        for key, value in original.items():
            assert torch.equal(out[key], value), key

    def test_camera_degradation_leaves_lidar_alone(self, batch: dict) -> None:
        lidar = batch["rasterized_lidar"].clone()
        brightness = batch["rgb"].float().mean()

        out = degrade_batch(batch, "camera", 0.8)

        assert out["rgb"].float().mean() < brightness
        assert torch.equal(out["rasterized_lidar"], lidar)

    def test_lidar_degradation_leaves_the_camera_alone(self, batch: dict) -> None:
        rgb = batch["rgb"].clone()
        occupancy = (batch["rasterized_lidar"] > 0).float().mean()

        out = degrade_batch(batch, "lidar", 0.8)

        assert (out["rasterized_lidar"] > 0).float().mean() < occupancy
        assert torch.equal(out["rgb"], rgb)

    def test_every_sample_gets_the_same_damage(self) -> None:
        # The training curriculum randomises severity and modality per sample;
        # a curve point must not, or the run averages over severities instead
        # of measuring one. Both still add per-pixel noise, so the samples are
        # never bit-identical: what separates them is how far apart their means
        # land, which is an order of magnitude tighter under fixed damage.
        def spread(images: torch.Tensor) -> float:
            per_sample = images.float().mean(dim=(1, 2, 3))
            return (per_sample.max() - per_sample.min()).item()

        torch.manual_seed(5)
        fixed = degrade_batch(
            {"rgb": torch.full((16, 3, 64, 192), 200, dtype=torch.uint8)},
            "camera",
            0.5,
        )
        torch.manual_seed(5)
        random = apply_sensor_degradation(
            {"rgb": torch.full((16, 3, 64, 192), 200, dtype=torch.uint8)},
            probability=1.0,
            max_severity=1.0,
        )

        assert spread(fixed["rgb"]) < 0.2 * spread(random["rgb"])

    def test_severity_is_monotone(self) -> None:
        brightness = []
        for severity in (0.2, 0.5, 0.9):
            torch.manual_seed(5)
            batch = {"rgb": torch.full((2, 3, 16, 48), 200, dtype=torch.uint8)}
            out = degrade_batch(batch, "camera", severity)
            brightness.append(out["rgb"].float().mean().item())
        assert brightness[0] > brightness[1] > brightness[2]

    def test_an_unknown_modality_is_refused(self, batch: dict) -> None:
        with pytest.raises(ValueError, match="camera"):
            degrade_batch(batch, "radar", 0.5)


class TestDegradationUnderAutocast:
    """The training step runs under autocast, and that changed the dtypes.

    The blur inside the camera degradation is a convolution, so autocast hands
    it back in bfloat16 while the image it is mixed with is still float32. A
    bare call on the CPU never sees that, which is why it reached a training
    run before it was caught.
    """

    def test_camera_degradation_survives_autocast(self) -> None:
        rgb = torch.full((4, 3, 32, 96), 200, dtype=torch.uint8)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = degrade_camera(rgb, torch.full((4,), 0.6))
        assert out.dtype == torch.uint8
        assert out.float().mean() < rgb.float().mean()

    def test_the_training_curriculum_survives_autocast(self) -> None:
        batch = {
            "rgb": torch.full((4, 3, 32, 96), 200, dtype=torch.uint8),
            "rasterized_lidar": torch.rand(4, 1, 64, 96),
            "observability": torch.ones(4, NUM_OBSERVABILITY_CHANNELS, 20, 24),
        }
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = apply_sensor_degradation(batch, probability=1.0, max_severity=1.0)
        assert torch.isfinite(out["rasterized_lidar"]).all()
        assert out["rgb"].dtype == torch.uint8

    def test_the_inference_degradation_survives_autocast(self) -> None:
        batch = {"rgb": torch.full((4, 3, 32, 96), 200, dtype=torch.uint8)}
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = degrade_batch(batch, "camera", 0.6)
        assert out["rgb"].dtype == torch.uint8


def test_degrade_camera_accepts_the_float_batch_inference_hands_it() -> None:
    """The inference path passes float32, not uint8, and must not be rejected.

    ``features_to_batch`` casts every model input to float32 before the driving
    agent calls ``degrade_batch``, while training degrades the collated uint8
    batch. Annotating only uint8 made every degraded evaluation run raise under
    the default runtime type checking, and the pilot only survived because an
    unrelated flag happened to disable it.
    """
    rgb = torch.full((2, 3, 16, 16), 128.0)
    out = degrade_camera(rgb, torch.tensor([0.5, 0.5]))

    assert out.dtype == torch.float32
    assert out.shape == rgb.shape
    assert not torch.equal(out, rgb)


def test_both_dtypes_are_damaged_by_the_same_amount() -> None:
    """uint8 and float32 carry 0-255 values, so the damage must match.

    Guards the claim the comparison rests on: a training-time degradation and
    an evaluation-time one of the same severity are the same degradation, so a
    robustness number measures the model rather than a scale mismatch.
    """
    severity = torch.tensor([0.7])
    as_float = torch.full((1, 3, 16, 16), 200.0)
    as_uint8 = as_float.to(torch.uint8)

    torch.manual_seed(0)
    from_float = degrade_camera(as_float, severity)
    torch.manual_seed(0)
    from_uint8 = degrade_camera(as_uint8, severity)

    # uint8 truncates on the way back; anything beyond that is a scale bug.
    assert (from_float.round() - from_uint8.float()).abs().max() <= 1.0


@pytest.mark.parametrize(
    ("modality", "key"),
    [("camera", "rgb"), ("lidar", "rasterized_lidar")],
)
def test_one_seed_reproduces_one_sequence_of_damage(modality, key) -> None:
    """Two runs under the same seed must meet byte-identical damage.

    This is what makes the per-route paired comparison mean anything: the two
    checkpoints being compared face the same noise and the same dropped
    returns, so the difference between them is the model.
    """

    def sequence(seed: int) -> list[torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        frames = []
        for _ in range(3):
            batch = {
                "rgb": torch.full((1, 3, 16, 16), 128.0),
                "rasterized_lidar": torch.ones(1, 2, 16, 16),
            }
            frames.append(degrade_batch(batch, modality, 0.5, generator)[key].clone())
        return frames

    first, again, other = sequence(7), sequence(7), sequence(8)

    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    # Not strict, and the mismatch is the point: a run of eight ticks draws a
    # different stream from a run of seven, so only the frames both runs have
    # are comparable.
    assert any(not torch.equal(a, b) for a, b in zip(first, other, strict=False))
    # A generator rebuilt per tick would freeze one pattern onto every frame,
    # which is not what a failing sensor does.
    assert not torch.equal(first[0], first[1])


def test_unseeded_degradation_still_works_for_training() -> None:
    """Training draws from the global stream and must not need a generator."""
    batch = {"rgb": torch.full((2, 3, 16, 16), 128.0)}
    out = degrade_batch(batch, "camera", 0.5)

    assert out["rgb"].shape == (2, 3, 16, 16)
