from gym_envs.pseudo.blind_env import PseudoBlindEnv
from gym_envs.pseudo.shop_env import PseudoShopEnv
from gym_envs.pseudo.blind_shop_env import BlindShopEnv
import numpy as np

# env = PseudoBlindEnv(
#     env_config={
#         "max_hand_size": 8,
#         "action_mode": "multi_binary",
#         "hand_mode": "indices",
#         "target_hand_obs": False,
#         "chips_reward_weight": 0.015,
#         "hand_type_reward_weight": 0.0,
#         "infinite_deck": False,
#         "bias": 0.0,
#         "rarity_bonus": 0.0,
#         "joker_count_range": (0, 2),
#         "max_jokers": 2,
#     }
# )

# episode_rewards = []
# for _ in range(50):
#     obs, _ = env.reset()
#     done, R = False, 0
#     while not done:
#         action = env.action_space.sample()
#         obs, r, done, _, _ = env.step(action)
#         R += r
#     episode_rewards.append(R)
# print("Average episode reward:", np.mean(episode_rewards))
# print("Max episode reward:", np.max(episode_rewards))
# print("Min episode reward:", np.min(episode_rewards))
# print("Reward Variance:", np.var(episode_rewards))
# print("Episode rewards:", episode_rewards)


# obs, _ = env.reset()
# print(len(env.deck.remaining_cards))
# print(sorted([str(x) for x in env.deck.remaining_cards]))
# print("Initial hand:", sorted([str(x) for x in env.hand]))
# action = [0, 1, 0, 0, 0, 0, 0, 0, 0]
# print(env.action_vector_to_action(action))
# print(env.step(action))
# print(len(env.deck.remaining_cards))
# print("New hand:", sorted([str(x) for x in env.hand]))
# action = [0, 0, 0, 0, 0, 0, 0, 0, 1]
# print(env.action_vector_to_action(action))
# print(env.step(action))
# print(len(env.deck.remaining_cards))
# print("New hand:", sorted([str(x) for x in env.hand]))


# for x in range(100):
#     obs, _ = env.reset()
#     done, R = False, 0
#     while not done:
#         action = env.action_space.sample()
#         obs, r, done, _, _ = env.step(action)
#         # Check if there are any duplicate cards in the hand or deck
#         if len(set(env.hand.cards)) != len(env.hand.cards):
#             print("Duplicate cards in hand:", env.hand)


# for x in range(100):
#     obs, _ = env.reset()
#     print(env.jokers)
#     print(obs["joker_indices"])
#     print(obs)


# shop_env = PseudoShopEnv({})
# shop_env.reset()
# print(shop_env.get_obs())


# blind_shop_env = BlindShopEnv()
# obs = blind_shop_env.reset()
# print("Initial Observation:", obs)
# action = blind_shop_env.shop_env.action_space.sample()
# print("Sample Action:", action)
# print(blind_shop_env.step({"shop_agent": action}))


import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, Tuple, MultiBinary

MAX_JOKERS = 3
NUM_TYPES = 5

# Joker definition
single_joker_space = Dict(
    {
        "type_id": Discrete(NUM_TYPES),
        "cost": Box(0.0, 100.0, shape=()),
        "mult": Box(0.0, 10.0, shape=()),
    }
)

# Whole observation space
obs_space = Dict(
    {
        "jokers": Dict(
            {
                "features": Tuple([single_joker_space for _ in range(MAX_JOKERS)]),
                "mask": MultiBinary(MAX_JOKERS),
            }
        ),
        "dummy_scalar": Box(0.0, 1.0, shape=()),
    }
)
