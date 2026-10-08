"""Project-local active-axis weighted ACT loss adapter.

The raw activity masks are captured before normalization. The adapter runs the
upstream forward exactly once and replaces only its L1 contribution, retaining
the original variational/KL contribution and standard ACT checkpoint type.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

RAW_ACTIVE = "day8.raw_active"
# Historical alias kept for Day8 tests/callers.
RAW_Y_ACTIVE = RAW_ACTIVE
RAW_CLOSE = "day8.raw_close"
RAW_POSITIVE_Z = "day8.raw_positive_z"


@dataclass(frozen=True)
class ActionLossSettings:
    """Loss settings for one or more active Cartesian axes."""

    active_weight: float = 4.0
    close_weight: float = 1.0
    activity_epsilon: float = 1e-6
    active_channel: int = 1
    active_name: str = "y"
    active_axes: tuple[str, ...] | None = None
    x_weight: float | None = None
    y_weight: float | None = None
    z_weight: float | None = None
    positive_z_weight: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.active_weight) or self.active_weight < 1:
            raise ValueError("Active-axis weight must be finite and >= 1 (1 is the control).")
        if not math.isfinite(self.close_weight) or self.close_weight < 1:
            raise ValueError("Close weight must be finite and >= 1 (1 is the control).")
        if not math.isfinite(self.activity_epsilon) or self.activity_epsilon <= 0:
            raise ValueError("Activity epsilon must be finite and > 0.")
        for axis, weight in (("x", self.x_weight), ("y", self.y_weight), ("z", self.z_weight)):
            if weight is not None and (not math.isfinite(weight) or weight < 1):
                raise ValueError(f"{axis.upper()} weight must be finite and >= 1 (1 is the control).")
        if self.positive_z_weight is not None and (
            not math.isfinite(self.positive_z_weight) or self.positive_z_weight < 1
        ):
            raise ValueError("Positive-Z weight must be finite and >= 1 (1 is the control).")

        axes = tuple(self.active_axes) if self.active_axes is not None else (self.active_name,)
        if not axes or any(axis not in ("x", "y", "z") for axis in axes) or len(set(axes)) != len(axes):
            raise ValueError("active_axes must be a non-empty tuple containing unique 'x', 'y', and/or 'z' values.")
        object.__setattr__(self, "active_axes", axes)
        if self.positive_z_weight is not None and "z" not in axes:
            raise ValueError("positive_z_weight requires active_axes to include 'z'.")

        if len(axes) == 1:
            expected_channel = {"x": 0, "y": 1, "z": 2}[axes[0]]
            if self.active_channel != expected_channel:
                raise ValueError("active_channel and active_name/active_axes disagree.")
        elif self.active_channel not in (0, 1, 2):
            raise ValueError("active_channel must be 0, 1, or 2.")

    @property
    def active_channels(self) -> tuple[int, ...]:
        return tuple({"x": 0, "y": 1, "z": 2}[axis] for axis in self.active_axes or ())

    @property
    def active_weights(self) -> tuple[float, ...]:
        """Return the effective weight for each configured active axis."""
        overrides = {"x": self.x_weight, "y": self.y_weight, "z": self.z_weight}
        return tuple(
            self.active_weight if overrides[axis] is None else overrides[axis]
            for axis in self.active_axes or ()
        )

    @property
    def active_axis_label(self) -> str:
        return "+".join(self.active_axes or ())


# Backward-compatible name for Day8 code/tests.
YLossSettings = ActionLossSettings


def raw_activity(actions: Tensor, settings: ActionLossSettings) -> Tensor:
    """Return raw active-axis labels.

    For one active axis, the result is ``[batch, time]`` for backward
    compatibility. For multiple axes, the result is ``[batch, time, axes]``.
    """
    if actions.ndim != 3 or actions.shape[-1] != 4:
        raise ValueError("Expected ACT action chunks [batch, time, 4] (x, y, z, gripper).")
    mask = actions[..., list(settings.active_channels)].abs() > settings.activity_epsilon
    return mask[..., 0] if len(settings.active_channels) == 1 else mask.detach().clone()


def raw_y_activity(actions: Tensor, settings: ActionLossSettings) -> Tensor:
    """Backward-compatible Day8 helper for a single active Y axis."""
    if tuple(settings.active_axes or ()) != ("y",):
        raise ValueError("raw_y_activity requires active_axes=('y',)")
    return raw_activity(actions, settings)


def raw_close_activity(actions: Tensor) -> Tensor:
    """Return gripper-close labels from raw, unnormalized actions."""
    if actions.ndim != 3 or actions.shape[-1] != 4:
        raise ValueError("Expected ACT action chunks [batch, time, 4] (x, y, z, gripper).")
    return (actions[..., 3] >= 1.5).detach().clone()


def raw_positive_z_activity(actions: Tensor, activity_epsilon: float = 1e-6) -> Tensor:
    """Return raw positive-Z action labels used for the lift phase."""
    if actions.ndim != 3 or actions.shape[-1] != 4:
        raise ValueError("Expected ACT action chunks [batch, time, 4] (x, y, z, gripper).")
    if not math.isfinite(activity_epsilon) or activity_epsilon <= 0:
        raise ValueError("Activity epsilon must be finite and > 0.")
    return (actions[..., 2] > activity_epsilon).detach().clone()


def weighted_action_l1(
    predictions: Tensor,
    targets: Tensor,
    is_pad: Tensor,
    raw_active: Tensor,
    settings: ActionLossSettings,
    raw_close: Tensor | None = None,
    raw_positive_z: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Compute weighted normalized L1 while excluding padded action positions."""
    if predictions.shape != targets.shape or targets.ndim != 3 or targets.shape[-1] != 4:
        raise ValueError("Predictions and targets must have matching [batch, time, 4] shapes.")
    if is_pad.shape != targets.shape[:2]:
        raise ValueError("Padding mask must have shape [batch, time].")

    n_axes = len(settings.active_channels)
    if n_axes == 1:
        if raw_active.shape != is_pad.shape:
            raise ValueError("Single-axis activity mask must have shape [batch, time].")
        active_masks = raw_active.unsqueeze(-1)
    else:
        if raw_active.shape != (*is_pad.shape, n_axes):
            raise ValueError("Multi-axis activity mask must have shape [batch, time, axes].")
        active_masks = raw_active

    if raw_close is None:
        # This fallback is useful only for small tests that pass raw targets.
        # The actual adapter always captures the raw mask before normalization.
        raw_close = targets[..., 3] >= 1.5
    if raw_close.shape != is_pad.shape:
        raise ValueError("Raw close mask must have shape [batch, time].")
    if raw_positive_z is None:
        raw_positive_z = targets[..., 2] > settings.activity_epsilon
    if raw_positive_z.shape != is_pad.shape:
        raise ValueError("Raw positive-Z mask must have shape [batch, time].")
    if is_pad.dtype != torch.bool or active_masks.dtype != torch.bool or raw_close.dtype != torch.bool or raw_positive_z.dtype != torch.bool:
        raise TypeError("Padding and activity masks must be boolean.")

    valid = ~is_pad.to(targets.device)
    active_masks = active_masks.to(targets.device) & valid.unsqueeze(-1)
    active_any = active_masks.any(dim=-1)
    neutral = valid & ~active_any
    close = raw_close.to(targets.device) & valid
    positive_z = raw_positive_z.to(targets.device) & valid
    valid_scalars = valid.unsqueeze(-1).expand_as(targets)
    errors = torch.where(valid_scalars, F.l1_loss(predictions, targets, reduction="none"), 0.0)

    weights = valid_scalars.to(errors.dtype).clone()
    for axis_index, (axis_name, channel, axis_weight) in enumerate(
        zip(settings.active_axes or (), settings.active_channels, settings.active_weights, strict=True)
    ):
        channel_weights = valid.to(errors.dtype) + (axis_weight - 1) * active_masks[..., axis_index]
        if axis_name == "z" and settings.positive_z_weight is not None:
            lift_mask = positive_z & active_masks[..., axis_index]
            channel_weights = channel_weights + (settings.positive_z_weight - axis_weight) * lift_mask
        weights[..., channel] = channel_weights
    weights[..., 3] = valid.to(errors.dtype) + (settings.close_weight - 1) * close

    count = valid_scalars.sum()
    baseline = errors.sum() / count.clamp_min(1)
    weighted = (weights * errors).sum() / weights.sum().clamp_min(1)
    metrics: dict[str, Tensor] = {"unweighted_l1_loss": baseline.detach()}
    for axis_index, (axis_name, channel) in enumerate(zip(settings.active_axes or (), settings.active_channels, strict=True)):
        axis_errors = errors[..., channel]
        axis_active = active_masks[..., axis_index]
        metrics[f"{axis_name}_active_mae_norm"] = (
            torch.where(axis_active, axis_errors, 0.0).sum().detach() / axis_active.sum().clamp_min(1)
        )
        metrics[f"{axis_name}_neutral_mae_norm"] = (
            torch.where(neutral, axis_errors, 0.0).sum().detach() / neutral.sum().clamp_min(1)
        )
        metrics[f"{axis_name}_active_target_ratio"] = axis_active.sum().detach() / valid.sum().clamp_min(1)
    metrics["close_active_mae_norm"] = torch.where(close, errors[..., 3], 0.0).sum().detach() / close.sum().clamp_min(1)
    metrics["close_active_target_ratio"] = close.sum().detach() / valid.sum().clamp_min(1)
    if "z" in (settings.active_axes or ()) and settings.positive_z_weight is not None:
        z_errors = errors[..., 2]
        lift_mask = positive_z & valid
        metrics["positive_z_active_mae_norm"] = (
            torch.where(lift_mask, z_errors, 0.0).sum().detach() / lift_mask.sum().clamp_min(1)
        )
        metrics["positive_z_active_target_ratio"] = lift_mask.sum().detach() / valid.sum().clamp_min(1)
    return weighted, baseline, metrics


def weighted_y_l1(predictions: Tensor, targets: Tensor, is_pad: Tensor, raw_active: Tensor, settings: ActionLossSettings, raw_close: Tensor | None = None):
    """Backward-compatible Day8 helper for a single active Y axis."""
    if tuple(settings.active_axes or ()) != ("y",):
        raise ValueError("weighted_y_l1 requires active_axes=('y',)")
    return weighted_action_l1(predictions, targets, is_pad, raw_active, settings, raw_close)


def attach_weighted_forward(policy: Any, settings: ActionLossSettings) -> None:
    """Attach an instance-local training forward, leaving inference/state intact."""
    if "forward" in policy.__dict__:
        raise RuntimeError("Refusing to replace a policy forward that is already customized.")
    original_forward = policy.forward
    verified = False

    def forward(self: Any, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        nonlocal verified
        if RAW_ACTIVE not in batch:
            raise RuntimeError("Raw Y mask missing (active-axis mask missing); use the project training adapter, not vanilla training.")
        if settings.close_weight > 1 and RAW_CLOSE not in batch:
            raise RuntimeError("Raw close mask missing while close_weight > 1; use the project training adapter.")
        if settings.positive_z_weight is not None and RAW_POSITIVE_Z not in batch:
            raise RuntimeError("Raw positive-Z mask missing while positive_z_weight is enabled; use the project training adapter.")
        model_batch = {
            key: value
            for key, value in batch.items()
            if key not in (RAW_ACTIVE, RAW_CLOSE, RAW_POSITIVE_Z)
        }
        captured: list[Tensor] = []

        def capture_prediction(_module: Any, _inputs: Any, output: Any) -> None:
            if not isinstance(output, tuple) or not isinstance(output[0], Tensor):
                raise RuntimeError("Upstream ACT model output changed; recheck the active-axis adapter.")
            captured.append(output[0])

        handle = self.model.register_forward_hook(capture_prediction)
        try:
            original_loss, original_metrics = original_forward(model_batch)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one upstream ACT model forward.")
        weighted, baseline, metrics = weighted_action_l1(
            captured[0],
            batch["action"],
            batch["action_is_pad"],
            batch[RAW_ACTIVE],
            settings,
            batch.get(RAW_CLOSE),
            batch.get(RAW_POSITIVE_Z),
        )
        if not verified:
            if "l1_loss" not in original_metrics or not math.isclose(original_metrics["l1_loss"], baseline.item(), rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError("Upstream ACT L1 reduction changed; adapter cannot safely replace it.")
            verified = True
        loss = original_loss if settings.active_weight == 1 and settings.close_weight == 1 else original_loss + (weighted - baseline)
        result = dict(original_metrics)
        result["l1_loss"] = weighted.item()
        result.update({key: value.item() for key, value in metrics.items()})
        return loss, result

    policy.forward = MethodType(forward, policy)


@contextmanager
def adapt_trainer(trainer: Any, settings: ActionLossSettings, on_policy: Callable[[Any], None] | None = None, on_checkpoint: Callable[[Any], None] | None = None) -> Iterator[None]:
    """Temporarily adapt local upstream training entry points."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    original_preprocess, original_factory, original_save = trainer._preprocess_dataset_batch, trainer.make_policy, trainer.save_checkpoint
    policies: list[Any] = []

    def preprocess(batch: dict[str, Any], camera_keys: list[str], rename_map: dict, preprocessor: Any) -> Any:
        mask = raw_activity(batch["action"], settings)
        close_mask = raw_close_activity(batch["action"])
        positive_z_mask = raw_positive_z_activity(batch["action"], settings.activity_epsilon)
        processed = dict(original_preprocess(batch, camera_keys, rename_map, preprocessor))
        processed[RAW_ACTIVE] = mask.to(processed["action"].device)
        processed[RAW_CLOSE] = close_mask.to(processed["action"].device)
        processed[RAW_POSITIVE_Z] = positive_z_mask.to(processed["action"].device)
        return processed

    def factory(*args: Any, **kwargs: Any) -> Any:
        policy = original_factory(*args, **kwargs)
        if not isinstance(policy, ACTPolicy):
            raise TypeError("The active-axis adapter supports ACTPolicy only.")
        attach_weighted_forward(policy, settings); policies.append(policy)
        if on_policy is not None: on_policy(policy)
        return policy

    def save(*args: Any, **kwargs: Any) -> Any:
        result = original_save(*args, **kwargs)
        if on_checkpoint is not None: on_checkpoint(kwargs["checkpoint_dir"] if "checkpoint_dir" in kwargs else args[0])
        return result

    trainer._preprocess_dataset_batch, trainer.make_policy, trainer.save_checkpoint = preprocess, factory, save
    try: yield
    finally:
        trainer._preprocess_dataset_batch, trainer.make_policy, trainer.save_checkpoint = original_preprocess, original_factory, original_save
        for policy in policies: del policy.forward
