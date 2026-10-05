"""Project-local active-axis weighted ACT loss adapter.

The raw activity mask is captured before normalization. The adapter runs the
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

# Kept under the historical name for compatibility with the Day8 tests and
# existing project-local callers. Its contents are now axis-independent.
RAW_ACTIVE = "day8.raw_active"
RAW_Y_ACTIVE = RAW_ACTIVE
RAW_CLOSE = "day8.raw_close"


@dataclass(frozen=True)
class ActionLossSettings:
    active_weight: float = 4.0
    close_weight: float = 1.0
    activity_epsilon: float = 1e-6
    active_channel: int = 1
    active_name: str = "y"

    def __post_init__(self) -> None:
        if not math.isfinite(self.active_weight) or self.active_weight < 1:
            raise ValueError("Active-axis weight must be finite and >= 1 (1 is the control).")
        if not math.isfinite(self.close_weight) or self.close_weight < 1:
            raise ValueError("Close weight must be finite and >= 1 (1 is the control).")
        if not math.isfinite(self.activity_epsilon) or self.activity_epsilon <= 0:
            raise ValueError("Activity epsilon must be finite and > 0.")
        if self.active_channel not in (0, 1, 2, 3):
            raise ValueError("Active channel must be one of 0 (x), 1 (y), 2 (z), or 3 (gripper).")
        if self.active_name not in ("x", "y"):
            raise ValueError("Active axis name must be 'x' or 'y'.")
        expected_channel = 0 if self.active_name == "x" else 1
        if self.active_channel != expected_channel:
            raise ValueError("active_channel and active_name disagree.")


# Backward-compatible name for Day8 code/tests.
YLossSettings = ActionLossSettings


def raw_activity(actions: Tensor, settings: ActionLossSettings) -> Tensor:
    """Return [batch, time] activity labels from raw, unnormalized actions."""
    if actions.ndim != 3 or actions.shape[-1] != 4:
        raise ValueError("Expected ACT action chunks [batch, time, 4] (x, y, z, gripper).")
    return (actions[..., settings.active_channel].abs() > settings.activity_epsilon).detach().clone()


def raw_y_activity(actions: Tensor, settings: ActionLossSettings) -> Tensor:
    """Backward-compatible Day8 helper; requires the default Y settings."""
    if settings.active_channel != 1:
        raise ValueError("raw_y_activity requires settings.active_channel == 1")
    return raw_activity(actions, settings)


def raw_close_activity(actions: Tensor) -> Tensor:
    """Return gripper-close labels from raw, unnormalized actions."""
    if actions.ndim != 3 or actions.shape[-1] != 4:
        raise ValueError("Expected ACT action chunks [batch, time, 4] (x, y, z, gripper).")
    return (actions[..., 3] >= 1.5).detach().clone()


def weighted_action_l1(
    predictions: Tensor,
    targets: Tensor,
    is_pad: Tensor,
    raw_active: Tensor,
    settings: ActionLossSettings,
    raw_close: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Weighted mean over valid scalar action targets, excluding padding."""
    if predictions.shape != targets.shape or targets.ndim != 3 or targets.shape[-1] != 4:
        raise ValueError("Predictions and targets must have matching [batch, time, 4] shapes.")
    if is_pad.shape != targets.shape[:2] or raw_active.shape != is_pad.shape:
        raise ValueError("Padding and activity masks must have shape [batch, time].")
    if raw_close is None:
        # Only for small unit-test calls where targets are still raw actions.
        # The real training adapter supplies this mask before normalization.
        raw_close = targets[..., 3] >= 1.5
    if raw_close.shape != is_pad.shape:
        raise ValueError("Raw close mask must have shape [batch, time].")
    if is_pad.dtype != torch.bool or raw_active.dtype != torch.bool or raw_close.dtype != torch.bool:
        raise TypeError("Padding and activity masks must be boolean.")

    valid = ~is_pad.to(targets.device)
    active = raw_active.to(targets.device) & valid
    neutral = valid & ~active
    close = raw_close.to(targets.device) & valid
    valid_scalars = valid.unsqueeze(-1).expand_as(targets)
    errors = torch.where(valid_scalars, F.l1_loss(predictions, targets, reduction="none"), 0.0)
    weights = valid_scalars.to(errors.dtype).clone()
    channel = settings.active_channel
    weights[..., channel] = valid.to(errors.dtype) + (settings.active_weight - 1) * active
    weights[..., 3] = valid.to(errors.dtype) + (settings.close_weight - 1) * close
    count = valid_scalars.sum()
    baseline = errors.sum() / count.clamp_min(1)
    weighted = (weights * errors).sum() / weights.sum().clamp_min(1)
    active_errors = errors[..., channel]
    prefix = settings.active_name
    metrics = {
        "unweighted_l1_loss": baseline.detach(),
        f"{prefix}_active_mae_norm": torch.where(active, active_errors, 0.0).sum().detach()
        / active.sum().clamp_min(1),
        f"{prefix}_neutral_mae_norm": torch.where(neutral, active_errors, 0.0).sum().detach()
        / neutral.sum().clamp_min(1),
        f"{prefix}_active_target_ratio": active.sum().detach() / valid.sum().clamp_min(1),
        "close_active_mae_norm": torch.where(close, errors[..., 3], 0.0).sum().detach()
        / close.sum().clamp_min(1),
        "close_active_target_ratio": close.sum().detach() / valid.sum().clamp_min(1),
    }
    return weighted, baseline, metrics


def weighted_y_l1(
    predictions: Tensor,
    targets: Tensor,
    is_pad: Tensor,
    raw_active: Tensor,
    settings: ActionLossSettings,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Backward-compatible Day8 helper for Y-weighted loss."""
    if settings.active_channel != 1:
        raise ValueError("weighted_y_l1 requires settings.active_channel == 1")
    return weighted_action_l1(predictions, targets, is_pad, raw_active, settings)


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
        model_batch = {key: value for key, value in batch.items() if key not in (RAW_ACTIVE, RAW_CLOSE)}
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
            captured[0], batch["action"], batch["action_is_pad"], batch[RAW_ACTIVE], settings,
            batch.get(RAW_CLOSE),
        )
        if not verified:
            if "l1_loss" not in original_metrics or not math.isclose(
                original_metrics["l1_loss"], baseline.item(), rel_tol=1e-5, abs_tol=1e-6
            ):
                raise RuntimeError("Upstream ACT L1 reduction changed; adapter cannot safely replace it.")
            verified = True
        loss = original_loss if settings.active_weight == 1 else original_loss + (weighted - baseline)
        result = dict(original_metrics)
        result["l1_loss"] = weighted.item()
        result.update({key: value.item() for key, value in metrics.items()})
        return loss, result

    policy.forward = MethodType(forward, policy)


@contextmanager
def adapt_trainer(
    trainer: Any,
    settings: ActionLossSettings,
    on_policy: Callable[[Any], None] | None = None,
    on_checkpoint: Callable[[Any], None] | None = None,
) -> Iterator[None]:
    """Temporarily adapt local upstream training entry points."""
    from lerobot.policies.act.modeling_act import ACTPolicy

    original_preprocess = trainer._preprocess_dataset_batch
    original_factory = trainer.make_policy
    original_save = trainer.save_checkpoint
    policies: list[Any] = []

    def preprocess(batch: dict[str, Any], camera_keys: list[str], rename_map: dict, preprocessor: Any) -> Any:
        mask = raw_activity(batch["action"], settings)
        close_mask = raw_close_activity(batch["action"])
        processed = original_preprocess(batch, camera_keys, rename_map, preprocessor)
        processed = dict(processed)
        processed[RAW_ACTIVE] = mask.to(processed["action"].device)
        processed[RAW_CLOSE] = close_mask.to(processed["action"].device)
        return processed

    def factory(*args: Any, **kwargs: Any) -> Any:
        policy = original_factory(*args, **kwargs)
        if not isinstance(policy, ACTPolicy):
            raise TypeError("The active-axis adapter supports ACTPolicy only.")
        attach_weighted_forward(policy, settings)
        policies.append(policy)
        if on_policy is not None:
            on_policy(policy)
        return policy

    def save(*args: Any, **kwargs: Any) -> Any:
        result = original_save(*args, **kwargs)
        if on_checkpoint is not None:
            directory = kwargs["checkpoint_dir"] if "checkpoint_dir" in kwargs else args[0]
            on_checkpoint(directory)
        return result

    trainer._preprocess_dataset_batch = preprocess
    trainer.make_policy = factory
    trainer.save_checkpoint = save
    try:
        yield
    finally:
        trainer._preprocess_dataset_batch = original_preprocess
        trainer.make_policy = original_factory
        trainer.save_checkpoint = original_save
        for policy in policies:
            del policy.forward
