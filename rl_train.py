import argparse
import os
import pickle
from pathlib import Path

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
from ray import tune, train
import ray.rllib
from ray.rllib.algorithms.ppo import PPOConfig, PPO, PPO
from ray import shutdown

from gym_envs.envs.blind_env import BlindEnv
from gym_envs.envs.shop_env import ShopEnv

from copy import deepcopy

from modeling.generic_blind_model import BalatroBlindModel
from modeling.generic_shop_model import BalatroShopModel
from ray.rllib.models import ModelCatalog
from modeling.distributions import AutoregressiveCardDist
from modeling.distributions import ShopActionDist
from gymnasium import spaces as sp
from ray.rllib.algorithms.algorithm import Algorithm
from gym_envs.envs.blind_env import BlindEnv
from gym_envs.envs.blind_shop_env import BlindShopEnv

import numpy as np
import torch
from gym_envs.round_logger_callback import RoundLoggerCallback

from numpy import array, float32
from ray.tune.search.optuna import OptunaSearch
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
import os
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import wandb
from typing import Any, Dict as typed_dict
from gym_envs.components.deck import Deck, ErraticDeck
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

from ray.air.config import CheckpointConfig
from config_utils import load_config_file, apply_overrides, deep_merge

os.environ["WANDB_SILENT"] = "true"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "blind_shop.toml"
DEFAULT_STORAGE_PATH = BASE_DIR / "run_data"
DEFAULT_RAY_CONFIG: dict[str, Any] = {"ignore_reinit_error": True}

PARAMETER_DEFAULTS: dict[str, Any] = {
    "max_jokers": 5,
    "max_hand_size": 10,
    "subset_hand_types": None,
    "subset_scoring_cards": None,
    "demo_mode": False,
    "shared_encoder": False,
    "blind_action_dist": "ar_custom_dist",
}

ACTION_MODE_LOOKUP = {
    "ar_custom_dist": "ar_custom_dist",
}

_OPTIONAL_SENTINELS = {"", "none", "null"}


class CuriosityTorchPolicy(PPOTorchPolicy):
    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        info = super().extra_action_out(input_dict, state_batches, model, action_dist)
        return info

class CuriosityPPO(PPO):
    @classmethod
    def get_default_policy_class(cls, config):
        return CuriosityTorchPolicy
    
def register_all():
    ModelCatalog.register_custom_action_dist("ar_custom_dist", AutoregressiveCardDist)
    ModelCatalog.register_custom_action_dist("shop_custom_dist", ShopActionDist)
    ModelCatalog.register_custom_model("generic_blind_model", BalatroBlindModel)
    ModelCatalog.register_custom_model("generic_shop_model", BalatroShopModel)
        
register_all()

def policy_mapping(agent_id, episode, **kwargs):
    if agent_id == "blind":
        return "blind_policy"
    else:
        return "shop_policy"

def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, typed_dict):
            v = dict(v)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (int, float, str, bool)):
            items.append((new_key, v))
    return dict(items)

def log_results(i, result):
    keys_to_remove = ["sampler_results", "env_runner_results"]
    for key in keys_to_remove:
        if key in result:
            del result[key]
    keys_to_remove = []
    for key in result["custom_metrics"]:
        if key.endswith("_min") or key.endswith("_max"):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del result["custom_metrics"][key]
    wandb.log(
        flatten_dict(result),
        step=i,
        commit=i % 5 == 0,
    )

def _normalize_optional(value: Any) -> Any:
    if isinstance(value, str) and value.lower() in _OPTIONAL_SENTINELS:
        return None
    return value

def parse_args():
    parser = argparse.ArgumentParser(description="Train Balatro RL agents with PPO.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a TOML configuration file.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values using dot notation keys.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and exit without launching training.",
    )
    return parser.parse_args()

def apply_ppo_overrides(
    config: PPOConfig, overrides: dict[str, Any] | None
) -> PPOConfig:
    if not overrides:
        return config
    config_copy = config.copy(copy_frozen=False)
    overrides = dict(overrides)

    def _apply_builder(name: str):
        if name not in overrides:
            return
        args = overrides.pop(name)
        builder = getattr(config_copy, name)
        if args is None:
            builder()
        elif isinstance(args, dict):
            builder(**args)
        else:
            builder(args)

    for builder_name in (
        "resources",
        "env_runners",
        "training",
        "multi_agent",
        "evaluation",
    ):
        _apply_builder(builder_name)

    if overrides:
        config_copy.update_from_dict(overrides)
    return config_copy


def default_blind_env_config(params: dict[str, Any]) -> dict[str, Any]:
    subset_hand_types = params.get("subset_hand_types")
    subset_scoring_cards = params.get("subset_scoring_cards")
    target_hand_reward = params.get("target_hand_reward", 0.0)
    max_hand_size = params["max_hand_size"]
    max_jokers = params["max_jokers"]
    config = {
        "objective_mode": "blind_grind",
        "max_hand_size": max_hand_size,
        "action_mode": ACTION_MODE_LOOKUP[params["blind_action_dist"]],
        "hand_mode": "base_card",
        "imagined_trajectories": False,
        "expert_pretraining": False,
        "deck_cls": Deck,
        "deck_obs": False,
        "deck_counts_obs": False,
        "embed_cards": True,
        "discard_penalty": 0.0,
        "correct_reward": 0.0,
        "incorrect_penalty": 0.0,
        "discard_potential_reward": 0.0,
        "goal_progress_reward": 0.0,
        "suit_homogeneity_bonus": 0.0,
        "joker_synergy_bonus": 0.0,
        "flattened_rank_chips": False,
        "cannot_discard_obs": True,
        "contained_hand_types_obs": True,
        "subset_hand_types_obs": subset_hand_types is not None,
        "scoring_cards_mask_obs": subset_scoring_cards is not None,
        "force_play": True,
        "chips_reward_weight": 0.0,
        "bias": 0.0,
        "rarity_bonus": 0.0,
        "target_hand_obs": target_hand_reward > 0.0,
        "max_jokers": max_jokers,
        "joker_count_range": (0, 0),
        "hand_level_range": (1, 1),
        "joker_count_bias_exponent": -1.0,
        "round_range": (1, 1),
    }
    return config

def default_blind_model_config(params: dict[str, Any]) -> dict[str, Any]:
    subset_hand_types = params.get("subset_hand_types")
    subset_scoring_cards = params.get("subset_scoring_cards")
    config = {
        "custom_model": "generic_blind_model",
        "custom_action_dist": params["blind_action_dist"],
        "custom_model_config": {
            "embed_cards": True,
            "action_method": "autoregressive",
            "discard_as_intent": True,
            "hand_representation_method": "context_token",
            "jokers_in_hand_attention": False,
            "max_jokers": params["max_jokers"],
            "max_supported_hand_size": params["max_hand_size"],
            "joker_types": 151,
            "forced_play_head": False,
            "subset_hand_types": subset_hand_types,
            "scoring_cards_masks": subset_scoring_cards,
            "FiLM_mode": None,
            "deck_obs": False,
            "shared_encoder": params["shared_encoder"],
            "noisy_layers": [],
            "allow_illegal_actions": False,
            "valid_card_count_coeff": 0.0,
            "suit_rank_entropy_coeff": 0.0,
            "suit_matching_aux_coeff": 0.0,
            "rank_matching_aux_coeff": 0.0,
            "option_variation_coeff": 0.0,
            "available_hand_types_coeff": 0.0,
            "intent_similarity_coeff": 0.0,
            "joker_identity_coeff": 0.0,
            "suit_count_aux_coeff": 0.0,
            "weight_decay_coeff": 0.0,
            "hand_score_aux_coeff": 0.0,
            "joker_spread_loss_coeff": 0.0,
        },
    }
    return config

def default_shop_model_config(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "custom_model": "generic_shop_model",
        "custom_action_dist": "shop_custom_dist",
        "custom_model_config": {
            "max_hand_size": params["max_hand_size"],
        },
    }

def default_blind_shop_env_config(
    params: dict[str, Any], blind_env_config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "start_phase": "blind",
        "stake": 0,
        "shop_starts_enabled": True,
        "cash_gained_reward": 0.0,
        "run_win_reward": 0.0,
        "round_won_reward": 0.0,
        "blind_env_config": blind_env_config,
        "shop_env_config": {
            "ignore_rarity": True,
            "max_hand_size": params["max_hand_size"],
        },
        "connect_to_balatro": params["demo_mode"],
        "cycle_blind_agents": True,
        "truncate_blind_agents": False,
        "catchup_probability": 0.0,
    }

def shop_config(model_config=None, ppo_overrides=None):
    env_config = {}
    model_config = model_config or {}
    config = (
        PPOConfig()
        .environment(
            env=ShopEnv,
            env_config=env_config,
            observation_space=ShopEnv(env_config).build_observation_space(),
            action_space=ShopEnv(env_config).build_action_space(),
        )
        .callbacks(RoundLoggerCallback)
        .framework("torch", torch_compile_learner=True)
        .resources(num_gpus=1, num_cpus_per_worker=1)
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            sample_timeout_s=60,
            rollout_fragment_length="auto",
            batch_mode="complete_episodes",
        )
        .training(
            train_batch_size=int(2**10),
            sgd_minibatch_size=int(2**10),
            num_sgd_iter=1,
            lr=2e-4,
            grad_clip=0.5,
            clip_param=0.3,
            kl_coeff=0.0,
            vf_loss_coeff=0.5,
            vf_clip_param=10.0,
            entropy_coeff=0.00,
            lambda_=0.95,
            gamma=0.99,
            model=model_config,
        )
    )

    return apply_ppo_overrides(config, ppo_overrides)


def blind_config(
    blind_env_config=None,
    blind_model_config=None,
    ppo_overrides=None,
):
    blind_env_config = blind_env_config or {}
    blind_model_config = blind_model_config or {}
    hand_size = 8
    config = (
        PPOConfig()
        .environment(
            env=BlindEnv,
            env_config=blind_env_config,
            observation_space=BlindEnv(blind_env_config).observation_space,
            action_space=BlindEnv(blind_env_config).action_space,
        )
        .callbacks(RoundLoggerCallback)
        .framework("torch", torch_compile_learner=True)
        .resources(
            num_gpus=1,
            num_cpus_per_worker=1,
        )
        .env_runners(
            num_env_runners=14,
            num_envs_per_env_runner=100,
            sample_timeout_s=60,
            rollout_fragment_length="auto",
            batch_mode="complete_episodes",
        )
        .training(
            train_batch_size=int(2**13),
            sgd_minibatch_size=int(2**11),
            num_sgd_iter=10,
            lr=2e-4,
            grad_clip=0.5,
            clip_param=0.1,
            kl_coeff=0.0,
            vf_loss_coeff=0.5,
            vf_clip_param=10.0,
            entropy_coeff=0.0,
            lambda_=0.95,
            gamma=0.99,
            model=blind_model_config,
        )
    )
    
#    class WeightInitCallback(RoundLoggerCallback):
#        def on_algorithm_init(self, *, algorithm, **kwargs):
#            manual_weight_init(algorithm)
#            super().on_algorithm_init(algorithm=algorithm, **kwargs)
#
#    config.callbacks(WeightInitCallback)

    return apply_ppo_overrides(config, ppo_overrides)

def blind_shop_config(
    blind_env_config=None,
    blind_model_config=None,
    blind_shop_env_config=None,
    shop_model_config=None,
    ppo_overrides=None,
):
    smoke = False
    blind_env_config = blind_env_config or {}
    blind_model_config = blind_model_config or {}
    blind_shop_env_config = blind_shop_env_config or {}
    shop_model_config = shop_model_config or {}
    blind_env = BlindEnv(blind_env_config)
    shop_env = ShopEnv(blind_shop_env_config["shop_env_config"])

    remote_env = blind_shop_env_config.get("connect_to_balatro", False)
    eval_env_config = deepcopy(blind_shop_env_config)
    eval_env_config["shop_starts_enabled"] = True
    eval_env_config["catchup_probability"] = 0.0
    eval_env_config["shop_env_config"]["ignore_rarity"] = False

    POLICIES = {
        "blind_agent": (
            None,
            blind_env.observation_space,
            blind_env.action_space,
            {
                "model": blind_model_config,
                "kl_coeff": 0.0,
                "lr": 2e-5,
                "explore": not remote_env,
                "lambda_": 0.99,
            },
        ),
        "shop_agent": (
            None,
            shop_env.observation_space,
            shop_env.action_space,
            {
                "model": shop_model_config,
                "kl_coeff": 0.0,
                "entropy_coeff": 0.001,
                "lr": 1e-4,
                "explore": not remote_env,
                "lambda_": 0.95,
            },
        ),
    }

    def policy_mapper(agent_id, episode, **kwargs):
        if "-" in agent_id:
            agent_id = agent_id.split("-")[0]
        return agent_id

    config = (
        PPOConfig(CuriosityPPO)
        .environment(env=BlindShopEnv, env_config=blind_shop_env_config)
        .callbacks(RoundLoggerCallback)
        .framework("torch")
        .experimental(_enable_new_api_stack=False)
        .resources(num_gpus=1, num_cpus_per_worker=1)
        .env_runners(
            num_env_runners=(10 if not remote_env and not smoke else 0),
            num_envs_per_env_runner=6 if not remote_env and not smoke else 1,
            explore=not remote_env,
            sample_timeout_s=60,
            rollout_fragment_length="auto",
            batch_mode="complete_episodes",
        )
        .multi_agent(
            policies=POLICIES,
            policy_mapping_fn=policy_mapper,
            policies_to_train=(["blind_agent", "shop_agent"] if not remote_env else []),
        )
        .training(
            train_batch_size=int(2**15),
            sgd_minibatch_size=int(2**11),
            num_sgd_iter=3,
            lr=1e-4,
            grad_clip=3,
            clip_param=0.3,
            vf_clip_param=3,
            entropy_coeff=0.001,
            gamma=0.99,
            vf_loss_coeff=0.25,
        )
    )
    
#    class WeightInitCallback(RoundLoggerCallback):
#        def on_algorithm_init(self, *, algorithm, **kwargs):
#            manual_weight_init(algorithm)
#            super().on_algorithm_init(algorithm=algorithm, **kwargs)
#
#    config.callbacks(WeightInitCallback)

    if not remote_env and not smoke:
        config = config.evaluation(
            evaluation_num_env_runners=2,
            evaluation_interval=1,
            evaluation_config={"explore": False, "env_config": eval_env_config},
            evaluation_parallel_to_training=True,
            evaluation_duration="auto",
        )

    return apply_ppo_overrides(config, ppo_overrides)

def manual_weight_init(algorithm):
    best_blind_ckpt = r""
    state_path = os.path.join(best_blind_ckpt, "policies", "default_policy", "policy_state.pkl")
    with open(state_path, "rb") as f:
        state_dict = pickle.load(f)
    algorithm.get_policy("blind_agent").set_state(state_dict)
    algorithm.workers.sync_weights()

def run_training(
    config_path: Path | str | None = None,
    overrides: list[str] | None = None,
    dry_run: bool = False,
):
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    overrides = list(overrides or [])

    config_from_file = load_config_file(config_path)
    config_data = apply_overrides(config_from_file, overrides)

    general_cfg = config_data.get("general", {})
    training_mode = general_cfg.get("training_mode", "blind_shop")

    ray_cfg = deep_merge(DEFAULT_RAY_CONFIG, config_data.get("ray", {}))
    ray_cfg.setdefault("ignore_reinit_error", True)
    ray.init(**ray_cfg)

    analysis = None
    try:

        parameters_raw = deep_merge(
            PARAMETER_DEFAULTS, config_data.get("parameters", {})
        )
        parameters = {k: _normalize_optional(v) for k, v in parameters_raw.items()}

        blind_action_dist = parameters["blind_action_dist"]
        if blind_action_dist not in ACTION_MODE_LOOKUP:
            raise ValueError(
                f"Unknown blind_action_dist '{blind_action_dist}'. Known values: {list(ACTION_MODE_LOOKUP)}"
            )

        blind_env_config = deep_merge(
            default_blind_env_config(parameters), config_data.get("blind_env", {})
        )
        blind_model_config = deep_merge(
            default_blind_model_config(parameters), config_data.get("blind_model", {})
        )
        shop_model_config = deep_merge(
            default_shop_model_config(parameters), config_data.get("shop_model", {})
        )
        blind_shop_env_config = deep_merge(
            default_blind_shop_env_config(parameters, blind_env_config),
            config_data.get("blind_shop_env", {}),
        )
        blind_shop_env_config["blind_env_config"] = blind_env_config
        blind_shop_env_config["shop_env_config"] = deep_merge(
            {"ignore_rarity": True, "max_hand_size": parameters["max_hand_size"]},
            blind_shop_env_config.get("shop_env_config", {}),
        )

        ppo_overrides = config_data.get("ppo_overrides", {})

        if training_mode == "blind":
            algo_config = blind_config(
                blind_env_config=blind_env_config,
                blind_model_config=blind_model_config,
                ppo_overrides=ppo_overrides.get("blind"),
            )
        elif training_mode == "shop_only":
            algo_config = shop_config(
                model_config=shop_model_config,
                ppo_overrides=ppo_overrides.get("shop"),
            )
        elif training_mode == "blind_shop":
            algo_config = blind_shop_config(
                blind_env_config=blind_env_config,
                blind_model_config=blind_model_config,
                blind_shop_env_config=blind_shop_env_config,
                shop_model_config=shop_model_config,
                ppo_overrides=ppo_overrides.get("blind_shop"),
            )
        else:
            raise ValueError(
                f"Unknown training mode '{training_mode}'. Expected 'blind', 'shop_only', or 'blind_shop'."
            )

        wandb_cfg = config_data.get("wandb", {})
        callbacks = []
        if wandb_cfg.get("enabled", True):
            wandb_kwargs = {
                "project": wandb_cfg.get("project", "balatro-rl"),
                "log_config": wandb_cfg.get("log_config", True),
                "save_checkpoints": wandb_cfg.get("save_checkpoints", False),
            }
            for optional_key in ("entity", "group", "api_key_file", "job_type"):
                if optional_key in wandb_cfg:
                    wandb_kwargs[optional_key] = wandb_cfg[optional_key]
            callbacks.append(WandbLoggerCallback(**wandb_kwargs))

        run_settings = config_data.get("run", {})
        run_name = run_settings.get("name", training_mode)
        storage_path_value = run_settings.get("storage_path", DEFAULT_STORAGE_PATH)
        storage_path = Path(storage_path_value)
        if not storage_path.is_absolute():
            storage_path = (BASE_DIR / storage_path).resolve()

        trial_format = run_settings.get("trial_name_format", "{run_name}_{trial_id}")

        def trial_name_creator(trial):
            template_vars = {
                "run_name": run_name,
                "trial_id": trial.trial_id,
                "experiment_tag": getattr(trial, "experiment_tag", ""),
            }
            try:
                return trial_format.format(**template_vars)
            except KeyError:
                return f"{run_name}_{trial.trial_id}"

        checkpoint_cfg = config_data.get("checkpoint", {})
        checkpoint_kwargs = {
            "num_to_keep": checkpoint_cfg.get("num_to_keep", 10),
            "checkpoint_frequency": checkpoint_cfg.get("checkpoint_frequency", 10),
        }
        if "checkpoint_at_end" in checkpoint_cfg:
            checkpoint_kwargs["checkpoint_at_end"] = checkpoint_cfg["checkpoint_at_end"]
        checkpoint_config = CheckpointConfig(**checkpoint_kwargs)

        if dry_run:
            print("Resolved training mode:", training_mode)
            print("Config file:", config_path)
            print("Ray init kwargs:", ray_cfg)
            print("Storage path:", storage_path)
            print("Callbacks:", [cb.__class__.__name__ for cb in callbacks])
            print("PPO override keys:", list(ppo_overrides.keys()))
            return None

        tune_kwargs: dict[str, Any] = {
            "config": algo_config,
            "name": run_name,
            "callbacks": callbacks,
            "storage_path": str(storage_path),
            "checkpoint_config": checkpoint_config,
            "trial_name_creator": trial_name_creator,
        }

        if run_settings.get("stop"):
            tune_kwargs["stop"] = run_settings["stop"]
        if "num_samples" in run_settings:
            tune_kwargs["num_samples"] = run_settings["num_samples"]
        if run_settings.get("restore"):
            tune_kwargs["restore"] = run_settings["restore"]
        if run_settings.get("metric"):
            tune_kwargs["metric"] = run_settings["metric"]
        if run_settings.get("mode"):
            tune_kwargs["mode"] = run_settings["mode"]

        analysis = tune.run(CuriosityPPO, **tune_kwargs)
        return analysis
    finally:
        ray.shutdown()


def main():
    args = parse_args()
    run_training(args.config, args.overrides, args.dry_run)


if __name__ == "__main__":
    main()
