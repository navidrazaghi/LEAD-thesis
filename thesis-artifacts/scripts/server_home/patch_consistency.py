# -*- coding: utf-8 -*-
"""Degradation-consistency self-distillation: plan the same with a broken sensor.

The degradation curriculum shows the model damaged inputs but lets it answer
them however it likes: the imitation target is the expert's trajectory, and
nothing ties the plan made from a damaged frame to the plan the same model makes
from the intact one. The thesis measured what that freedom costs -- destroying
the camera moves the deformable run's planned waypoints by 1.55 m on average --
and this turns that measured quantity into a training objective.

Each training step sees the batch twice. The intact copy goes through the model
without gradient and acts as the teacher; the damaged copy, exactly as the
curriculum would have produced it, is the student and carries the imitation
loss as before. On the samples whose sensors were actually damaged, the
student's waypoints and route are pulled towards the teacher's with an L1 term.
Undamaged samples add nothing, since teacher and student saw the same input.

This is the consistency-regularization idea of AugMix (Hendrycks et al., ICLR
2020) moved from class posteriors to a driving plan, and from image corruptions
to sensor failure.

Off by default (weight 0.0), and when off the code path is the one that existed
before this patch, draw for draw: the clean copy is taken without consuming any
random draw, and only when the weight is positive.

The planning outputs exist only once the planning decoder is on, so the term is
inert in pretraining. A consistency run can therefore post-train from the
pretrain of the curriculum run it is compared against, and differ from it in
this term alone.
"""
import io
import pathlib
import sys

REPO = pathlib.Path.home() / "LEAD/lead"
CFG = REPO / "src/lead/config/training/data_config.py"
POLICY = REPO / "src/lead/policy/transfuser/transfuser.py"
TRAIN = REPO / "src/lead/training/train.py"

CONFIG_OLD = '''    sensor_degradation_misalignment_probability: float = 0.0'''
CONFIG_NEW = '''    sensor_degradation_misalignment_probability: float = 0.0
    # Degradation-consistency self-distillation: weight of the L1 pull of the
    # damaged sample's plan towards the plan from its intact copy. 0 is off.
    degradation_consistency_weight: float = 0.0'''

POLICY_OLD = '''    def degrade_batch(self, batch: TransfuserForwardBatch) -> TransfuserForwardBatch:
        """Inherited, see superclass."""'''
POLICY_NEW = '''    def augment_batch_with_clean(
        self,
        batch: TransfuserForwardBatch,
    ) -> tuple[TransfuserForwardBatch, TransfuserForwardBatch]:
        """The augmented batch as the student sees it, and its undamaged copy.

        The colour augmentation is shared, so the two differ only in the sensor
        damage. The copy is taken between the two augmentations and consumes no
        random draw, so the student's batch is exactly what augment_batch would
        have returned for the same random state.

        Args:
            batch: The collated batch, modified in place.

        Returns:
            The damaged batch, and the intact copy the teacher reads.
        """
        data_config = self.lead_config.training.data
        if not self.training:
            return batch, batch
        if data_config.use_color_augmentation and "rgb" in batch:
            batch["rgb"] = augment_rgb_batch(
                batch["rgb"],
                data_config.color_augmentation_probability,
            )
        clean = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        if data_config.use_sensor_degradation:
            batch = apply_sensor_degradation(
                batch,
                data_config.sensor_degradation_probability,
                data_config.sensor_degradation_max_severity,
                data_config.deployment_perturbation_families,
                data_config.sensor_degradation_independent_modalities,
                data_config.sensor_degradation_full_failure_probability,
                data_config.sensor_degradation_misalignment_probability,
                float(self.get_policy_config().bev_pixels_per_meter),
            )
        return batch, clean

    def degrade_batch(self, batch: TransfuserForwardBatch) -> TransfuserForwardBatch:
        """Inherited, see superclass."""'''

TRAIN_OLD = '''        batch = self._raw_model.augment_batch(batch)
        predictions = self.model(batch)
        losses, extra_metrics = self.model.compute_loss(predictions, batch)

        total_loss = torch.zeros(1, device=self.device)
        for key, value in losses.items():
            # Reshape as sanity check if the loss is a scalar
            total_loss = total_loss + self.normalized_loss_weights[
                key
            ] * value.float().reshape(1)
'''
TRAIN_NEW = '''        consistency_weight = self.config.training.data.degradation_consistency_weight
        clean_batch = None
        if consistency_weight > 0.0:
            batch, clean_batch = self._raw_model.augment_batch_with_clean(batch)
        else:
            batch = self._raw_model.augment_batch(batch)
        predictions = self.model(batch)
        losses, extra_metrics = self.model.compute_loss(predictions, batch)

        total_loss = torch.zeros(1, device=self.device)
        for key, value in losses.items():
            # Reshape as sanity check if the loss is a scalar
            total_loss = total_loss + self.normalized_loss_weights[
                key
            ] * value.float().reshape(1)

        if clean_batch is not None:
            consistency = self._degradation_consistency(predictions, batch, clean_batch)
            if consistency is not None:
                total_loss = total_loss + consistency_weight * consistency.reshape(1)
                extra_metrics["losses/degradation_consistency"] = consistency.detach()
'''

HELPER_ANCHOR = '''    def transfer_batch_to_device(
        self,'''
HELPER_NEW = '''    def _degradation_consistency(
        self,
        predictions: typing.Any,
        batch: dict,
        clean_batch: dict,
    ) -> torch.Tensor | None:
        """L1 distance from the damaged sample's plan to its intact copy's plan.

        The teacher is the same network on the intact input, without gradient;
        its outputs are cloned at once so no compiled graph can hand back a
        buffer the student's pass reuses. Only samples whose camera or LiDAR
        tensor actually differs from the intact copy count, so undamaged samples
        neither add signal nor dilute it.

        Args:
            predictions: The student's predictions on the damaged batch.
            batch: The damaged batch.
            clean_batch: The intact copy.

        Returns:
            The mean per-sample distance over damaged samples, or None when the
            policy predicts no plan (pretraining) or nothing was damaged.
        """
        student_waypoints = getattr(predictions, "future_waypoints", None)
        if student_waypoints is None:
            return None
        damaged = torch.zeros(
            student_waypoints.shape[0],
            dtype=torch.bool,
            device=student_waypoints.device,
        )
        for key in ("rgb", "rasterized_lidar"):
            if key in batch and key in clean_batch:
                difference = (batch[key].float() - clean_batch[key].float()).flatten(1)
                damaged |= difference.abs().amax(dim=1) > 0
        if not bool(damaged.any()):
            return None
        with torch.no_grad():
            teacher = self.model(clean_batch)
        teacher_waypoints = teacher.future_waypoints.detach().float().clone()
        per_sample = (student_waypoints.float() - teacher_waypoints).abs().mean(dim=(1, 2))
        student_route = getattr(predictions, "route", None)
        teacher_route = getattr(teacher, "route", None)
        if student_route is not None and teacher_route is not None:
            per_sample = per_sample + (
                student_route.float() - teacher_route.detach().float().clone()
            ).abs().mean(dim=(1, 2))
        weight = damaged.to(per_sample.dtype)
        return (per_sample * weight).sum() / weight.sum()

    def transfer_batch_to_device(
        self,'''

EDITS = [
    (CFG, CONFIG_OLD, CONFIG_NEW),
    (POLICY, POLICY_OLD, POLICY_NEW),
    (TRAIN, TRAIN_OLD, TRAIN_NEW),
    (TRAIN, HELPER_ANCHOR, HELPER_NEW),
]


def main() -> None:
    """Apply every edit, refusing anything ambiguous."""
    for path, anchor, replacement in EDITS:
        text = io.open(path, encoding="utf-8").read()
        found = text.count(anchor)
        if found != 1:
            sys.exit(f"FATAL: {path.name}: anchor appears {found} times, expected 1:\n{anchor[:120]}")
        io.open(path, "w", encoding="utf-8", newline="\n").write(text.replace(anchor, replacement))
        print(f"  patched {path.name}")


main()
