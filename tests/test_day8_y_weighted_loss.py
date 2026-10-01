"""CPU correctness tests only: no dataset videos, training loops, or GPU required."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.scripts import lerobot_train

from scripts.day8_y_weighted_loss import (
    RAW_Y_ACTIVE,
    YLossSettings,
    adapt_trainer,
    attach_weighted_forward,
    raw_y_activity,
    weighted_y_l1,
)


def small_config(use_vae=True):
    return ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (18,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (3,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
        device="cpu", push_to_hub=False,
        chunk_size=3, n_action_steps=2,
        dim_model=16, n_heads=2, dim_feedforward=32,
        n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1,
        latent_dim=2, use_vae=use_vae, pretrained_backbone_weights=None,
    )


class FixedPredictions(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictions = nn.Parameter(torch.linspace(-0.3, 0.8, 24).reshape(2, 3, 4))
        self.mu = nn.Parameter(torch.full((2, 2), 0.2))
        self.logvar = nn.Parameter(torch.full((2, 2), -0.1))
        self.calls = 0

    def forward(self, batch):
        if RAW_Y_ACTIVE in batch:
            raise AssertionError("Training-only raw mask must not enter the ACT model.")
        self.calls += 1
        return self.predictions, (self.mu, self.logvar)


def fixed_policy():
    # Exercise the real upstream ACTPolicy.forward with a tiny deterministic
    # prediction module, without constructing its normal-sized vision network.
    policy = ACTPolicy.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    policy.config = small_config()
    policy.model = FixedPredictions()
    policy.reset()
    return policy


def example_batch():
    raw = torch.tensor([
        [[0, -0.25, 0, 1], [0, 0, -0.5, 1], [0, 0.25, 0, 2]],
        [[0, 0.25, 0, 1], [0, 0, 0.5, 1], [0, -0.25, 0, 2]],
    ], dtype=torch.float32)
    return {
        "action": raw, "action_is_pad": torch.tensor([[False, False, False], [False, True, True]]),
        RAW_Y_ACTIVE: raw_y_activity(raw, YLossSettings()),
    }


class LossTests(unittest.TestCase):
    def test_invalid_weights(self):
        for weight in (0, 0.5, -1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                YLossSettings(active_weight=weight)

    def test_invalid_epsilon(self):
        for epsilon in (0, -1, float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                YLossSettings(activity_epsilon=epsilon)

    def test_raw_mask_is_independent_of_normalized_values(self):
        raw = torch.zeros(1, 2, 4)
        raw[0, 1, 1] = 0.25
        mask = raw_y_activity(raw, YLossSettings())
        # A nonzero mean makes raw zero nonzero after normalization, while
        # a real active command can become normalized zero. Mask stays raw.
        raw[..., 1] = (raw[..., 1] - 0.25) / 0.1
        self.assertEqual(mask.tolist(), [[False, True]])

    def test_mask_epsilon(self):
        raw = torch.zeros(1, 3, 4)
        raw[0, :, 1] = torch.tensor([0, 1e-8, -0.25])
        self.assertEqual(raw_y_activity(raw, YLossSettings()).tolist(), [[False, False, True]])

    def test_bad_raw_shape(self):
        with self.assertRaises(ValueError):
            raw_y_activity(torch.zeros(3, 4), YLossSettings())

    def test_weight_one_matches_original_loss_and_gradients(self):
        batch = example_batch()
        prediction = torch.full_like(batch["action"], 0.3, requires_grad=True)
        loss, baseline, _ = weighted_y_l1(
            prediction, batch["action"], batch["action_is_pad"], batch[RAW_Y_ACTIVE], YLossSettings(1)
        )
        mask = ~batch["action_is_pad"].unsqueeze(-1)
        original = (F.l1_loss(batch["action"], prediction, reduction="none") * mask).sum() / (mask.sum() * 4)
        torch.testing.assert_close(loss, original, rtol=0, atol=0)
        torch.testing.assert_close(baseline, original, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(loss, prediction)[0], torch.autograd.grad(original, prediction)[0],
            rtol=0, atol=0,
        )

    def test_weight_four_hand_calculation_and_relative_gradients(self):
        prediction = torch.ones(1, 2, 4, requires_grad=True)
        target = torch.zeros_like(prediction)
        with torch.no_grad():
            prediction[0, 0, 1] = 2
        loss, baseline, _ = weighted_y_l1(
            prediction, target, torch.zeros(1, 2, dtype=torch.bool),
            torch.tensor([[True, False]]), YLossSettings(4),
        )
        torch.testing.assert_close(loss, torch.tensor(15 / 11))
        torch.testing.assert_close(baseline, torch.tensor(9 / 8))
        gradient = torch.autograd.grad(loss, prediction)[0]
        torch.testing.assert_close(gradient[0, 0, 1], torch.tensor(4 / 11))
        # X/Z/gripper and neutral Y share weight one, not an extra multiplier.
        torch.testing.assert_close(gradient[0, 1], torch.full((4,), 1 / 11))
        self.assertAlmostEqual((gradient[0, 0, 1] / gradient[0, 0, 0]).item(), 4)

    def test_activity_at_later_chunk_position_is_weighted(self):
        prediction = torch.ones(1, 3, 4, requires_grad=True)
        loss, _, _ = weighted_y_l1(
            prediction, torch.zeros_like(prediction), torch.zeros(1, 3, dtype=torch.bool),
            torch.tensor([[False, False, True]]), YLossSettings(4),
        )
        gradient = torch.autograd.grad(loss, prediction)[0]
        self.assertAlmostEqual((gradient[0, 2, 1] / gradient[0, 0, 1]).item(), 4)

    def test_no_activity_is_identical_to_baseline(self):
        prediction = torch.ones(1, 2, 4)
        loss, baseline, metrics = weighted_y_l1(
            prediction, torch.zeros_like(prediction), torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool), YLossSettings(4),
        )
        torch.testing.assert_close(loss, baseline, rtol=0, atol=0)
        self.assertEqual(metrics["y_active_mae_norm"].item(), 0)

    def test_padding_does_not_change_value_or_gradient(self):
        prediction = torch.ones(1, 2, 4, requires_grad=True)
        with torch.no_grad():
            prediction[0, 1] = 1e20
        pad = torch.tensor([[False, True]])
        active = torch.tensor([[True, True]])
        loss, _, _ = weighted_y_l1(prediction, torch.zeros_like(prediction), pad, active, YLossSettings())
        self.assertEqual(loss.item(), 1)
        self.assertEqual(torch.autograd.grad(loss, prediction)[0][0, 1].abs().sum().item(), 0)

    def test_all_padding_is_finite_zero(self):
        prediction = torch.ones(1, 2, 4, requires_grad=True)
        loss, baseline, metrics = weighted_y_l1(
            prediction, torch.zeros_like(prediction), torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool), YLossSettings(),
        )
        self.assertEqual(loss.item(), 0)
        self.assertEqual(baseline.item(), 0)
        self.assertTrue(all(torch.isfinite(v) for v in metrics.values()))
        self.assertEqual(torch.autograd.grad(loss, prediction)[0].abs().sum().item(), 0)

    def test_nonfinite_padding_is_excluded_from_value(self):
        prediction = torch.ones(1, 2, 4)
        target = torch.zeros_like(prediction)
        target[0, 1] = float("nan")
        loss, _, _ = weighted_y_l1(
            prediction, target, torch.tensor([[False, True]]),
            torch.ones(1, 2, dtype=torch.bool), YLossSettings(),
        )
        self.assertEqual(loss.item(), 1)

    def test_mask_and_action_shapes_are_checked(self):
        with self.assertRaises(ValueError):
            weighted_y_l1(
                torch.ones(1, 2, 4), torch.ones(1, 2, 4), torch.zeros(1, 3, dtype=torch.bool),
                torch.ones(1, 2, dtype=torch.bool), YLossSettings(),
            )
        with self.assertRaises(TypeError):
            weighted_y_l1(
                torch.ones(1, 2, 4), torch.ones(1, 2, 4), torch.zeros(1, 2),
                torch.ones(1, 2, dtype=torch.bool), YLossSettings(),
            )


class AdapterTests(unittest.TestCase):
    def test_weight_one_real_upstream_forward_and_gradients(self):
        original = fixed_policy()
        adapted = copy.deepcopy(original)
        attach_weighted_forward(adapted, YLossSettings(1))
        batch = example_batch()
        base_loss, _ = original({k: v for k, v in batch.items() if k != RAW_Y_ACTIVE})
        new_loss, _ = adapted(batch)
        torch.testing.assert_close(new_loss, base_loss, rtol=0, atol=0)
        base_loss.backward()
        new_loss.backward()
        for left, right in zip(original.parameters(), adapted.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        self.assertEqual(adapted.model.calls, 1)
        self.assertEqual(len(adapted.model._forward_hooks), 0)

    def test_weight_four_preserves_upstream_kl_contribution(self):
        original = fixed_policy()
        adapted = copy.deepcopy(original)
        batch = example_batch()
        base_loss, _ = original({k: v for k, v in batch.items() if k != RAW_Y_ACTIVE})
        weighted, baseline, _ = weighted_y_l1(
            original.model.predictions, batch["action"], batch["action_is_pad"], batch[RAW_Y_ACTIVE], YLossSettings()
        )
        attach_weighted_forward(adapted, YLossSettings())
        new_loss, metrics = adapted(batch)
        torch.testing.assert_close(new_loss, base_loss - baseline + weighted)
        base_loss.backward()
        new_loss.backward()
        # In this test prediction and KL parameters are independent, so we can
        # check exactly that the KL gradient was neither removed nor doubled.
        torch.testing.assert_close(adapted.model.mu.grad, original.model.mu.grad)
        torch.testing.assert_close(adapted.model.logvar.grad, original.model.logvar.grad)
        self.assertIn("kld_loss", metrics)
        self.assertEqual(adapted.model.calls, 1)

    def test_missing_raw_mask_fails_instead_of_using_normalized_y(self):
        policy = fixed_policy()
        attach_weighted_forward(policy, YLossSettings())
        with self.assertRaisesRegex(RuntimeError, "Raw Y mask missing"):
            policy({"action": torch.zeros(2, 3, 4)})

    def test_double_install_is_rejected(self):
        policy = fixed_policy()
        attach_weighted_forward(policy, YLossSettings())
        with self.assertRaises(RuntimeError):
            attach_weighted_forward(policy, YLossSettings())

    def test_preprocessor_uses_raw_mask_before_real_normalization(self):
        cfg = small_config()
        stats = {
            "action": {"mean": torch.full((4,), 0.25), "std": torch.full((4,), 0.1)},
            "observation.state": {"mean": torch.zeros(18), "std": torch.ones(18)},
        }
        preprocessor, _ = make_act_pre_post_processors(cfg, stats)
        policy = fixed_policy()
        trainer = SimpleNamespace(
            _preprocess_dataset_batch=lerobot_train._preprocess_dataset_batch,
            make_policy=lambda **kwargs: policy,
            save_checkpoint=lambda *args, **kwargs: None,
        )
        raw = example_batch()
        raw.pop(RAW_Y_ACTIVE)
        raw["observation.state"] = torch.zeros(2, 18)
        expected = raw_y_activity(raw["action"], YLossSettings())
        with adapt_trainer(trainer, YLossSettings()):
            processed = trainer._preprocess_dataset_batch(raw, [], {}, preprocessor)
            torch.testing.assert_close(processed[RAW_Y_ACTIVE], expected)
            # raw neutral Y becomes -2.5: still NOT marked active.
            self.assertLess(processed["action"][0, 1, 1].item(), -2)
            self.assertFalse(processed[RAW_Y_ACTIVE][0, 1])
            trainer.make_policy(cfg=cfg)(processed)

    def test_hooks_and_instance_forward_are_restored_after_exception(self):
        policy = fixed_policy()
        trainer = SimpleNamespace(
            _preprocess_dataset_batch=lambda *args: {},
            make_policy=lambda **kwargs: policy,
            save_checkpoint=lambda *args, **kwargs: None,
        )
        original_functions = dict(trainer.__dict__)
        with self.assertRaisesRegex(RuntimeError, "test exit"):
            with adapt_trainer(trainer, YLossSettings()):
                trainer.make_policy()
                self.assertIn("forward", policy.__dict__)
                raise RuntimeError("test exit")
        for name, function in original_functions.items():
            self.assertIs(getattr(trainer, name), function)
        self.assertNotIn("forward", policy.__dict__)

    def test_metadata_callbacks_and_checkpoint_keyword(self):
        events = []
        policy = fixed_policy()
        trainer = SimpleNamespace(
            _preprocess_dataset_batch=lambda *args: {},
            make_policy=lambda **kwargs: policy,
            save_checkpoint=lambda **kwargs: events.append("saved"),
        )
        with adapt_trainer(
            trainer, YLossSettings(),
            lambda p: events.append(p), lambda path: events.append(path),
        ):
            trainer.make_policy()
            trainer.save_checkpoint(checkpoint_dir=Path("test-checkpoint"))
        self.assertEqual(events, [policy, "saved", Path("test-checkpoint")])

    def test_checkpoint_remains_standard_act_and_inference_needs_no_mask(self):
        policy = ACTPolicy(small_config(use_vae=False)).eval()
        attach_weighted_forward(policy, YLossSettings())
        state_keys = tuple(policy.state_dict())
        observation = {
            "observation.state": torch.zeros(1, 18),
            "observation.environment_state": torch.zeros(1, 3),
        }
        expected = policy.select_action(observation)
        with tempfile.TemporaryDirectory() as directory:
            policy.save_pretrained(directory)
            restored = ACTPolicy.from_pretrained(directory).eval()
            self.assertIs(type(restored), ACTPolicy)
            self.assertEqual(tuple(restored.state_dict()), state_keys)
            torch.testing.assert_close(restored.select_action(observation), expected)
            self.assertNotIn("forward", restored.__dict__)


if __name__ == "__main__":
    unittest.main()
