"""Train ordinary ACT checkpoints with a project-local Y-active weighted L1.

Use the shell launcher for the Day7-v2-controlled experiment. No installed
LeRobot files are modified. --check-only validates config/metadata without
constructing a policy, decoding videos, or performing optimizer updates.
"""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

if __package__:
    from .day8_y_weighted_loss import ActionLossSettings, adapt_trainer
else:
    from day8_y_weighted_loss import ActionLossSettings, adapt_trainer


def validate_day8_config(cfg):
    """Fail closed for training modes outside this deliberately narrow experiment."""
    if cfg.policy is None or cfg.policy.type != "act":
        raise ValueError("Day8 supports --policy.type=act only.")
    if cfg.steps <= 0 or cfg.batch_size <= 0:
        raise ValueError("Training steps and batch size must be positive.")
    if cfg.resume or cfg.policy.pretrained_path is not None:
        raise ValueError("Day8 starts from scratch; resume/fine-tuning is not supported.")
    if cfg.is_reward_model_training or cfg.job.is_remote:
        raise ValueError("Day8 supports local policy training only.")
    degrees = (
        cfg.parallelism.dp_replicate, cfg.parallelism.dp_shard,
        cfg.parallelism.cp_size, cfg.parallelism.cfg_parallel,
    )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or any(degree != 1 for degree in degrees):
        raise ValueError("Day8 currently supports one local process only.")
    if cfg.accelerator.compile.enabled or cfg.accelerator.activation_checkpointing.mode != "none":
        raise ValueError("Day8's forward capture requires compile/checkpointing disabled.")
    if (
        cfg.policy.use_amp or cfg.accelerator.mixed_precision != "no"
        or cfg.accelerator.gradient_accumulation.steps != 1
    ):
        raise ValueError("Use the Day7 control settings: no mixed precision and no accumulation.")
    if cfg.policy.device not in {"cpu", "xpu"}:
        raise ValueError("Day8 currently supports --policy.device=cpu or xpu.")
    if cfg.sample_weighting is not None or cfg.peft is not None or cfg.ema.enable:
        raise ValueError("Do not combine Day8 with sample weighting, PEFT, or EMA.")
    if cfg.dataset.streaming or cfg.env is not None or cfg.env_eval_freq != 0:
        raise ValueError("Use a local dataset and separate online evaluation (--env_eval_freq=0).")
    if cfg.policy.push_to_hub or cfg.save_checkpoint_to_hub:
        raise ValueError("Day8 saves local checkpoints only; disable Hub publishing.")
    if cfg.output_dir is None or Path(cfg.output_dir).exists():
        raise ValueError("Specify a new, nonexistent --output_dir; never overwrite the Day7 baseline.")
    if cfg.dataset.root is None:
        raise ValueError("A local --dataset.root is required.")
    root = Path(cfg.dataset.root)
    info = json.loads((root / "meta/info.json").read_text())
    action = info["features"]["action"]
    if action["shape"] != [4] or action.get("names") != ["delta_x", "delta_y", "delta_z", "gripper"]:
        raise ValueError("Dataset action order must be [delta_x, delta_y, delta_z, gripper].")
    return info


def source_sha256(value):
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def git_head(directory):
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_recipe(cfg, settings, info, trainer):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    script_dir = Path(__file__).resolve().parent
    root = Path(cfg.dataset.root)
    return {
        "experiment": f"active_{settings.active_axis_label}_weighted_l1",
        "active_weight": settings.active_weight,
        "close_weight": settings.close_weight,
        "activity_epsilon_raw_units": settings.activity_epsilon,
        "active_axes": list(settings.active_axes or ()),
        "active_channels": list(settings.active_channels),
        "mask": "abs(raw action[..., active channel(s)]) > epsilon; exclude action_is_pad",
        "l1_reduction": "sum(valid scalar weight * normalized absolute error) / sum(valid scalar weight)",
        "other_scalar_weights": 1.0,
        "kl_weight": cfg.policy.kl_weight,
        "inference": "standard ACTPolicy; no action multiplier or intervention",
        "resume_supported": False,
        "dataset": {
            "repo_id": cfg.dataset.repo_id, "root": str(root),
            "frames": info["total_frames"], "episodes": info["total_episodes"], "fps": info["fps"],
            "info_sha256": hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest(),
            "stats_sha256": hashlib.sha256((root / "meta/stats.json").read_bytes()).hexdigest(),
        },
        "training": {
            "steps": cfg.steps, "batch_size": cfg.batch_size, "seed": cfg.seed,
            "chunk_size": cfg.policy.chunk_size, "n_action_steps": cfg.policy.n_action_steps,
        },
        "source": {
            "torch_version": torch.__version__,
            "project_git_head": git_head(script_dir.parent),
            "lerobot_git_head": git_head(Path(inspect.getfile(trainer)).parent),
            "upstream_act_forward_sha256": source_sha256(ACTPolicy.forward),
            "upstream_preprocess_sha256": source_sha256(trainer._preprocess_dataset_batch),
            "project_file_sha256": {
                name: hashlib.sha256((script_dir / name).read_bytes()).hexdigest()
                for name in ("day8_y_weighted_loss.py", "day8_train_y_weighted.py")
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Remaining arguments are passed to LeRobot. Use the Day8 shell launcher for the control settings.",
        allow_abbrev=False,
    )
    parser.add_argument("--active-weight", "--y-active-weight", dest="active_weight", type=float, default=4.0)
    parser.add_argument("--active-axis", choices=("x", "y", "xy"), default="y")
    parser.add_argument("--close-weight", type=float, default=1.0)
    parser.add_argument("--activity-epsilon", "--y-activity-epsilon", dest="activity_epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--progress-miniters",
        type=int,
        default=50,
        help="Refresh the LeRobot training progress bar after this many steps.",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate only; do not train or write outputs.")
    options, upstream_args = parser.parse_known_args()
    if options.progress_miniters <= 0:
        raise ValueError("--progress-miniters must be positive")
    active_axes = tuple(options.active_axis) if options.active_axis == "xy" else (options.active_axis,)
    settings = ActionLossSettings(
        active_weight=options.active_weight,
        close_weight=options.close_weight,
        activity_epsilon=options.activity_epsilon,
        active_channel=0 if active_axes[0] == "x" else 1,
        active_name=active_axes[0],
        active_axes=active_axes,
    )

    # LeRobot's parser introspects concrete annotations; do not enable deferred
    # annotations in this entry point. All custom flags must be stripped first.
    from lerobot.configs import parser as lr_parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train as trainer
    from lerobot.utils.import_utils import register_third_party_plugins

    @lr_parser.wrap()
    def run(cfg: TrainPipelineConfig):
        cfg.validate()
        info = validate_day8_config(cfg)
        recipe = build_recipe(cfg, settings, info, trainer)
        print(json.dumps(recipe, indent=2), flush=True)
        if options.check_only:
            print("Day8 configuration check: PASS (no training or output writes)", flush=True)
            return

        def write_recipe(policy):
            from lerobot.utils.constants import ACTION

            if tuple(policy.config.output_features[ACTION].shape) != (4,):
                raise ValueError("The constructed ACT policy must have four action channels.")
            directory = Path(cfg.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "day8_loss_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")

        def checkpoint_recipe(directory):
            path = Path(directory) / "pretrained_model/day8_loss_recipe.json"
            path.write_text(json.dumps(recipe, indent=2) + "\n")

        with adapt_trainer(trainer, settings, write_recipe, checkpoint_recipe):
            # The ordinary trainer owns model creation, sampling, optimizer,
            # seeds, processors, checkpoints, and logging, exactly as in Day7.
            # Override only the local tqdm constructor. This does not modify
            # the installed LeRobot checkout or alter training semantics.
            original_tqdm = trainer.tqdm
            trainer.tqdm = partial(original_tqdm, miniters=options.progress_miniters)
            try:
                trainer.train(cfg)
            finally:
                trainer.tqdm = original_tqdm

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *upstream_args]
        register_third_party_plugins()
        run()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
