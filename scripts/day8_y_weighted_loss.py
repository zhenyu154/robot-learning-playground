"""Project-local ACT loss adapter; it never edits the installed LeRobot source.

The raw activity mask is captured before normalization. The adapter runs the
upstream forward exactly once and replaces only its L1 contribution, retaining
the original variational/KL contribution and the standard ACT checkpoint type.
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

RAW_Y_ACTIVE = "day8.raw_y_active"


@dataclass(frozen=True)
class YLossSettings:
    active_weight: float = 4.0
    activity_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not math.isfinite(self.active_weight) or self.active_weight < 1:
            raise ValueError("Y active weight must be finite and >= 1 (1 is the control).")
        if not math.isfinite(self.activity_epsilon) or self.activity_epsilon <= 0:
            raise ValueError("Y activity epsilon must be finite and > 0.")


def raw_y_activity(actions: Tensor, settings: YLossSettings) -> Tensor:
    """Return [batch, time] activity labels from original, unnormalized actions."""
    if actions.ndim != 3 or actions.shape[-1] != 4:
        raise ValueError("Expected ACT action chunks [batch, time, 4] (x, y, z, gripper).")
    return (actions[..., 1].abs() > settings.activity_epsilon).detach().clone()


def weighted_y_l1(
    predictions: Tensor,
    targets: Tensor,
    is_pad: Tensor,
    raw_active: Tensor,
    settings: YLossSettings,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Weighted mean over valid scalar action targets, not over padded slots.

    w = active_weight only for valid Y-active scalars; otherwise w = 1.
    L = sum(w * abs(error)) / sum(w). Padding has weight zero. This global
    normalization changes the common gradient scale, but the active-Y to all
    other error coefficients has exactly the requested ratio.
    """
    if predictions.shape != targets.shape or targets.ndim != 3 or targets.shape[-1] != 4:
        raise ValueError("Predictions and targets must have matching [batch, time, 4] shapes.")
    if is_pad.shape != targets.shape[:2] or raw_active.shape != is_pad.shape:
        raise ValueError("Padding and raw Y activity must have shape [batch, time].")
    if is_pad.dtype != torch.bool or raw_active.dtype != torch.bool:
        raise TypeError("Padding and raw Y activity masks must be boolean.")

    valid = ~is_pad.to(targets.device)
    active = raw_active.to(targets.device) & valid
    neutral = valid & ~active
    valid_scalars = valid.unsqueeze(-1).expand_as(targets)
    # where, rather than multiplication, also excludes nonfinite padded targets.
    errors = torch.where(valid_scalars, F.l1_loss(predictions, targets, reduction="none"), 0.0)
    weights = valid_scalars.to(errors.dtype).clone()
    weights[..., 1] = valid.to(errors.dtype) + (settings.active_weight - 1) * active
    count = valid_scalars.sum()
    baseline = errors.sum() / count.clamp_min(1)
    weighted = (weights * errors).sum() / weights.sum().clamp_min(1)
    y_errors = errors[..., 1]
    metrics = {
        "unweighted_l1_loss": baseline.detach(),
        "y_active_mae_norm": torch.where(active, y_errors, 0.0).sum().detach() / active.sum().clamp_min(1),
        "y_neutral_mae_norm": torch.where(neutral, y_errors, 0.0).sum().detach() / neutral.sum().clamp_min(1),
        "y_active_target_ratio": active.sum().detach() / valid.sum().clamp_min(1),
    }
    return weighted, baseline, metrics


def attach_weighted_forward(policy: Any, settings: YLossSettings) -> None:
    """Attach an instance-local training forward, leaving inference/state_dict intact."""
    if "forward" in policy.__dict__:
        raise RuntimeError("Refusing to replace a policy forward that is already customized.")
    original_forward = policy.forward
    verified = False

    def forward(self: Any, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        nonlocal verified
        if RAW_Y_ACTIVE not in batch:
            raise RuntimeError("Raw Y mask missing: use the Day8 preprocessing adapter, not vanilla training.")
        model_batch = {key: value for key, value in batch.items() if key != RAW_Y_ACTIVE}
        captured: list[Tensor] = []

        def capture_prediction(_module: Any, _inputs: Any, output: Any) -> None:
            if not isinstance(output, tuple) or not isinstance(output[0], Tensor):
                raise RuntimeError("Upstream ACT model output changed; recheck the Day8 adapter.")
            captured.append(output[0])

        handle = self.model.register_forward_hook(capture_prediction)
        try:
            original_loss, original_metrics = original_forward(model_batch)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one upstream ACT model forward.")
        weighted, baseline, metrics = weighted_y_l1(
            captured[0], batch["action"], batch["action_is_pad"], batch[RAW_Y_ACTIVE], settings
        )
        if not verified:
            if "l1_loss" not in original_metrics or not math.isclose(
                original_metrics["l1_loss"], baseline.item(), rel_tol=1e-5, abs_tol=1e-6
            ):
                raise RuntimeError("Upstream ACT L1 reduction changed; Day8 cannot safely replace it.")
            verified = True
        # Keep the upstream graph/KL intact, and never perform a second stochastic
        # model forward. Return the original loss exactly for the weight=1 control.
        loss = original_loss if settings.active_weight == 1 else original_loss + (weighted - baseline)
        result = dict(original_metrics)
        result["l1_loss"] = weighted.item()
        result.update({key: value.item() for key, value in metrics.items()})
        return loss, result

    policy.forward = MethodType(forward, policy)


@contextmanager
def adapt_trainer(
    trainer: Any,
    settings: YLossSettings,
    on_policy: Callable[[Any], None] | None = None,
    on_checkpoint: Callable[[Any], None] | None = None,
) -> Iterator[None]:
    """Temporarily adapt the local, single-process upstream training entry points."""
    from lerobot.policies.act.modeling_act import ACTPolicy

    original_preprocess = trainer._preprocess_dataset_batch
    original_factory = trainer.make_policy
    original_save = trainer.save_checkpoint
    policies: list[Any] = []

    def preprocess(batch: dict[str, Any], camera_keys: list[str], rename_map: dict, preprocessor: Any) -> Any:
        # Processors may change action values or drop arbitrary extra keys. Save
        # the raw mask separately, then attach it AFTER the standard pipeline.
        mask = raw_y_activity(batch["action"], settings)
        processed = original_preprocess(batch, camera_keys, rename_map, preprocessor)
        processed = dict(processed)
        processed[RAW_Y_ACTIVE] = mask.to(processed["action"].device)
        return processed

    def factory(*args: Any, **kwargs: Any) -> Any:
        policy = original_factory(*args, **kwargs)
        if not isinstance(policy, ACTPolicy):
            raise TypeError("The Day8 adapter supports ACTPolicy only.")
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
            del policy.forward  # restore the original ACTPolicy class method
