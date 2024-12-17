# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import pathlib

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{SCRIPT_DIR}/..")
sys.path.insert(0, f"{SCRIPT_DIR}/../rl_baseline/sample-factory")
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.run_algorithm import run_algorithm
from sample_factory.utils.utils import str2bool

# Needs to be imported to register models and envs
import rl_baseline.tasks_nle
import rl_baseline.encoders_nle
import rl_baseline.env_nle
from datetime import datetime


def add_extra_params(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser

    # ==== wandb args ====
    p.add_argument("--wandb_entity", type=str, default="rl-llm")
    p.add_argument("--wandb_proj", type=str, default="online-reward-model")

    # ==== reward function setup ===
    p.add_argument(
        "--extrinsic_reward_coeff",
        default=1.0,
        type=float,
        help="Coefficient for the environment reward",
    )
    p.add_argument(
        "--llm_reward_coeff",
        default=0.0,
        type=float,
        help="Coefficient for adding the reward learned through LLM preferences.",
    )
    p.add_argument(
        "--llm_reward_type",
        type=str,
        choices=[
            "motif",
            "online_cls",
            "online_cls_reward_model",
            "online_motif",
            "cosine-bow",
            "none",
        ],
    )

    p.add_argument(
        "--remove_duplicated_obs",
        default=0,
        type=int,
    )

    p.add_argument(
        "--online_dataset_subsample",
        default=10,
        type=int,
    )

    # used when using querying online LLM (reward_type: online_cls, online_cls_reward_model, online_motif)
    p.add_argument("--llm_server_addr", type=str)
    p.add_argument("--llm_model", type=str, default="Meta-LLama-3.1-8B-Instruct")
    p.add_argument("--llm_batch_size", type=int, default=100)
    p.add_argument("--prompt_version", type=str, default="default")
    p.add_argument("--goal_key", type=str, default="defaultgoal")


    # online used for learning online reward model (reward_type: online_cls_reward_model)
    p.add_argument("--reward_model_batch_size", type=int, default=256)
    p.add_argument("--reward_model_lr", type=float, default=1e-4)
    p.add_argument("--reward_model_gradient_steps_per_update", type=int, default=32)
    p.add_argument("--classification_threshold", type=float, default=0.5)

    # motif
    p.add_argument(
        "--train_dir",
        default=os.path.join(os.getcwd(), "train_dir"),
        type=str,
        help="Root for all experiments",
    )
    # if you have a reward model offline trained: only for motif
    p.add_argument(
        "--reward_dir",
        default=os.path.join(os.getcwd(), "train_dir/rl_pairs_dataset_random_seed1"),
        type=str,
        help="Root dir to load reward model from.",
    )
    p.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["gpu", "cpu"],
        help="CPU training is only recommended for smaller e.g. MLP policies",
    )

    p.add_argument("--seed", default=1, type=int, help="Set a fixed seed value")

    # below are from AlgorithmBase and ReinforcementLearningAlgorithm classes.
    p.add_argument("--save_every_sec", default=120, type=int, help="Checkpointing rate")
    p.add_argument(
        "--save_every_steps", default=1e8, type=int, help="Checkpointing rate"
    )
    p.add_argument(
        "--keep_checkpoints",
        default=3,
        type=int,
        help="Number of model checkpoints to keep",
    )
    p.add_argument(
        "--checkpoint_id",
        default=-1,
        type=int,
        help="Checkpoint id to load from folder",
    )
    p.add_argument(
        "--save_milestones_sec",
        default=-1,
        type=int,
        help="Save intermediate checkpoints in a separate folder for later evaluation (default=never)",
    )

    p.add_argument(
        "--stats_avg",
        default=100,
        type=int,
        help="How many episodes to average to measure performance (avg. reward etc)",
    )

    p.add_argument("--learning_rate", default=1e-4, type=float, help="LR")

    p.add_argument(
        "--train_for_env_steps",
        default=int(1e10),
        type=int,
        help="Stop after all policies are trained for this many env steps",
    )
    p.add_argument(
        "--train_for_seconds",
        default=int(1e10),
        type=int,
        help="Stop training after this many seconds",
    )

    # observation preprocessing
    p.add_argument(
        "--obs_subtract_mean",
        default=0.0,
        type=float,
        help="Observation preprocessing, mean value to subtract from observation (e.g. 128.0 for 8-bit RGB)",
    )
    p.add_argument(
        "--obs_scale",
        default=1.0,
        type=float,
        help="Observation preprocessing, divide observation tensors by this scalar (e.g. 128.0 for 8-bit RGB)",
    )

    p.add_argument(
        "--pop_bl",
        default=True,
        type=str2bool,
        help="To remove or not the bl_stats from the observation",
    )
    # RL
    p.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    p.add_argument(
        "--reward_scale",
        default=1.0,
        type=float,
        help=(
            "Multiply all rewards by this factor before feeding into RL algorithm."
            "Sometimes the overall scale of rewards is too high which makes value estimation a harder regression task."
            "Loss values become too high which requires a smaller learning rate, etc."
        ),
    )
    p.add_argument(
        "--reward_clip",
        default=10.0,
        type=float,
        help="Clip rewards between [-c, c]. Default [-10, 10] virtually means no clipping for most envs",
    )

    # policy size and configuration
    p.add_argument(
        "--encoder_type",
        default="conv",
        type=str,
        help="Type of the encoder. Supported: conv, mlp, resnet (feel free to define more)",
    )
    p.add_argument(
        "--encoder_subtype",
        default="convnet_simple",
        type=str,
        help="Specific encoder design (see model.py)",
    )
    p.add_argument(
        "--encoder_custom",
        default="nle_rgbcrop_encoder",
        type=str,
        help="Use custom encoder class from the registry (see model_utils.py)",
    )
    p.add_argument(
        "--encoder_extra_fc_layers",
        default=1,
        type=int,
        help='Number of fully-connected layers of size "hidden size" to add after the basic encoder (e.g. convolutional)',
    )
    p.add_argument(
        "--hidden_size",
        default=512,
        type=int,
        help="Size of hidden layer in the model, or the size of RNN hidden state in recurrent model (e.g. GRU)",
    )
    p.add_argument(
        "--nonlinearity",
        default="elu",
        choices=["elu", "relu", "tanh"],
        type=str,
        help="Type of nonlinearity to use",
    )
    p.add_argument(
        "--policy_initialization",
        default="orthogonal",
        choices=["orthogonal", "xavier_uniform"],
        type=str,
        help="NN weight initialization",
    )
    p.add_argument(
        "--policy_init_gain",
        default=1.0,
        type=float,
        help="Gain parameter of PyTorch initialization schemas (i.e. Xavier)",
    )
    p.add_argument(
        "--actor_critic_share_weights",
        default=True,
        type=str2bool,
        help="Whether to share the weights between policy and value function",
    )

    # TODO: Right now this only applies to custom encoders. Make sure generic policies also factor in this arg
    p.add_argument(
        "--use_spectral_norm",
        default=False,
        type=str2bool,
        help="Use spectral normalization to smoothen the gradients and stabilize training. Only supports fully connected layers",
    )

    p.add_argument(
        "--adaptive_stddev",
        default=True,
        type=str2bool,
        help="Only for continuous action distributions, whether stddev is state-dependent or just a single learned parameter",
    )
    p.add_argument(
        "--initial_stddev",
        default=1.0,
        type=float,
        help="Initial value for non-adaptive stddev. Only makes sense for continuous action spaces",
    )


def parse_all_args(argv=None, evaluation=False):
    parser = arg_parser(argv, evaluation=evaluation)
    add_extra_params(parser)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    # needed for llama3
    user = os.environ["USER"]
    os.environ["OUTLINES_CACHE_DIR"] = f"/tmp/.outlines.{user}"

    """Script entry point."""
    cfg = parse_all_args()
    # if "WANDB_API_KEY" in os.environ and cfg.wandb:

    if cfg.llm_reward_type == "none":
        assert cfg.llm_reward_coeff == 0
    else:
        assert cfg.llm_reward_coeff > 0

    print(f"llm reward type: {cfg.llm_reward_type}")

    if cfg.wandb:

        wandb_dir = os.path.abspath(cfg.train_dir)
        os.environ["WANDB_DIR"] = wandb_dir
        os.makedirs(wandb_dir, exist_ok=True)

        import wandb

        wandb.login(host="https://fairwandb.org")

        fname = f"{cfg.train_dir}/{cfg.experiment}/wandb_id"
        resume = os.path.exists(fname)
        kwargs = {}
        if resume:
            with open(fname, "r") as f:
                wandb_id = f.read()
            kwargs["id"] = wandb_id
            kwargs["resume"] = "must"

            print(f"Resuming wandb from {wandb_id}")

        run = wandb.init(
            project=cfg.wandb_proj,
            entity=cfg.wandb_entity,
            config=cfg,
            save_code=True,
            name=cfg.experiment,
            sync_tensorboard=True,
            **kwargs,
        )

        if not resume:
            os.makedirs(f"{cfg.train_dir}/{cfg.experiment}", exist_ok=True)
            with open(fname, "w") as f:
                f.write(run.id)
    status = run_algorithm(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
