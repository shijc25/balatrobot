import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gym_envs.envs.blind_env import BlindEnv


CORE_OBS_KEYS = {"discards_left", "hands_left", "hand_stats", "dollars"}


def assert_basic_observation(obs, env):
    assert isinstance(obs, dict)
    missing_keys = CORE_OBS_KEYS - set(obs.keys())
    assert not missing_keys, f"Missing observation keys: {missing_keys}"

    if env.hand_mode == "base_card":
        assert "hand" in obs
        assert isinstance(obs["hand"], dict)
    else:
        assert "hand_indices" in obs
        assert obs["hand_indices"].shape == (env.G.max_hand_size,)

    for key in ("discards_left", "hands_left", "dollars"):
        assert obs[key].shape == (1,)

    if env.blind_obs:
        assert "blind_index" in obs
        assert obs["blind_index"].shape == (1,)

    if env.objective_mode == "blind_grind":
        assert "chip_goals" in obs
        assert obs["chip_goals"].shape == (5,)
        assert "goal_progress" in obs
        assert obs["goal_progress"].shape == (6,)

    assert isinstance(obs["hand_stats"], dict)
    if "jokers" in obs:
        assert isinstance(obs["jokers"], dict)

    return obs


def test_reset_produces_valid_observation():
    env = BlindEnv()
    obs, info = env.reset()

    assert_basic_observation(obs, env)
    assert info == {}


def test_combo_index_play_and_discard_flow():
    env = BlindEnv()
    env.reset()

    initial_hands_left = env.hands_left
    initial_discards_left = env.discards_left

    play_action = np.array([1, 0, 0], dtype=np.int64)
    obs, reward, terminated, truncated, info = env.step(play_action)

    assert env.hands_left == initial_hands_left - 1
    assert not truncated
    assert_basic_observation(obs, env)
    assert isinstance(info, dict)

    assert env.discards_left == initial_discards_left
    discard_action = np.array([0, 0, 0], dtype=np.int64)
    obs, reward, terminated, truncated, info = env.step(discard_action)

    assert env.discards_left == initial_discards_left - 1
    assert_basic_observation(obs, env)
    assert isinstance(info, dict)


def test_multi_binary_action_mode_flow():
    env = BlindEnv({"action_mode": "multi_binary"})
    env.reset()

    play_action = np.zeros(env.G.max_hand_size + 1, dtype=np.int64)
    play_action[0] = 1  # choose PLAY
    play_action[1] = 1  # select the first card slot
    obs, reward, terminated, truncated, info = env.step(play_action)

    assert not truncated
    assert_basic_observation(obs, env)

    discard_action = np.zeros(env.G.max_hand_size + 1, dtype=np.int64)
    discard_action[0] = 0  # choose DISCARD
    discard_action[1] = 1
    obs, reward, terminated, truncated, info = env.step(discard_action)

    assert_basic_observation(obs, env)


def test_stacked_binary_masks_action_mode_flow():
    env = BlindEnv({"action_mode": "stacked_binary_masks"})
    env.reset()

    action = np.zeros((6, env.G.max_hand_size + 2), dtype=np.int64)
    action[0, 0] = 1  # choose PLAY
    action[0, 1] = 1  # pick the first card slot
    obs, reward, terminated, truncated, info = env.step(action)

    assert_basic_observation(obs, env)
    assert env.last_hand_played.sum() >= 0


def test_reset_with_catchup_round():
    env = BlindEnv()
    obs, info = env.reset(options={"catchup_round": 3})

    assert env.round == 3
    assert_basic_observation(obs, env)


def test_blind_grind_training_configuration():
    config = {
        "objective_mode": "blind_grind",
        "max_hand_size": 15,
        "action_mode": "stacked_binary_masks",
        "hand_mode": "base_card",
        "cannot_discard_obs": True,
        "contained_hand_types_obs": True,
        "deck_counts_obs": True,
        "goal_progress_reward": 0.5,
        "suit_homogeneity_bonus": 0.1,
        "discard_penalty": 0.0,
    }

    env = BlindEnv(config)
    obs, info = env.reset()

    assert env.chip_goal > 0
    assert_basic_observation(obs, env)
    assert "chip_goals" in obs
    assert "goal_progress" in obs

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert_basic_observation(obs, env)
    assert isinstance(info, dict)
