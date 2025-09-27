from gym_envs.envs.blind_env import BlindEnv
from itertools import combinations
from gym_envs.components.hand import Hand
import time
import cProfile, pstats


class RoundOneBot:
    def choose_action(self, env: BlindEnv):
        best_hand_type, scoring_cards = env.G.hand.evaluate()
        hand_score = (
            env.G.hand_stats[best_hand_type].chips
            + sum(x.chip_value() for x in scoring_cards)
        ) * env.G.hand_stats[best_hand_type].mult
        if (
            best_hand_type in ["Four of a Kind", "Full House", "Straight Flush"]
            or hand_score >= env.chip_goal - env.chips
        ):
            for combo in combinations(env.G.hand.cards, 5):
                sub_hand = Hand(combo)
                sub_best_hand_type, sub_scoring_cards = sub_hand.evaluate()
                if sub_best_hand_type == best_hand_type:
                    scoring_cards = sub_scoring_cards
                    break
            action = [1, 0]
            for i, card in enumerate(env.G.hand.cards):
                if card in scoring_cards:
                    action.append(1)
                else:
                    action.append(0)
            # print(sum(action))
            return action

        multiples = env.G.hand.multiples()
        trips = [m for m in multiples if multiples[m] >= 3]
        doubles = [m for m in multiples if multiples[m] == 2][:2]  # limit to two pairs
        forced_play = env.discards_left == 0
        if len(trips) > 0:
            action = [forced_play, 0]
            discard_count = 0
            for i, card in enumerate(env.G.hand.cards):
                if card.value in trips or discard_count == 5:
                    action.append(0)
                else:
                    action.append(1)
                    discard_count += 1
            return action
        elif len(doubles) > 0:
            action = [forced_play, 0]
            discard_count = 0
            for i, card in enumerate(env.G.hand.cards):
                if card.value in doubles or discard_count == 5:
                    action.append(0)
                else:
                    action.append(1)
                    discard_count += 1
            return action
        else:
            return [forced_play, 0, 1, 1, 1, 1, 1, 0, 0, 0]


max_jokers = 5
hand_size = 8
target_hand_reward = 0.0
num_experts = 5  # Number of experts for the linear experts model
demo_mode = False

blind_action_dist = "mode_count_binary_dist"
action_mode_lookup = {
    "mode_count_binary_dist": "mode_count_binary",
    "play_discard_binary_dist": "multi_binary",
}
blind_env_config = {
    "max_hand_size": hand_size,
    "action_mode": action_mode_lookup[blind_action_dist],
    "hand_mode": "base_card",
    "num_experts": num_experts,
    "correct_reward": 1.0,
    "incorrect_penalty": 0.0,
    "discard_potential_reward": 0.05,
    "goal_progress_reward": 0.3,
    "suit_homogeneity_bonus": 0.00,
    "joker_synergy_bonus": 0.05,
    "discard_penalty": 0.0,  # negative penalty = reward for discarding
    "flattened_rank_chips": False,
    "cannot_discard_obs": True,
    "chips_reward_weight": 1.0 / 1000,
    "hand_type_reward_weight": target_hand_reward,
    "infinite_deck": False,
    "bias": 0.0,
    "rarity_bonus": 0.0,
    "target_hand_obs": target_hand_reward > 0.0,
    "max_jokers": max_jokers,
    "chip_reward_normalization": "log_joker",
    "deck_obs": False,
    "deck_counts_obs": True,
    "imagined_trajectories": False,
    "objective_mode": "blind_grind",
    "joker_count_range": (0, 0),
    "hand_level_range": (1, 1),
    "joker_count_bias_exponent": 2,
    "round_range": (
        1,
        1,
    ),  # Defines starting round. Usually (1, 1), but can be changed for curriculum or testing
}


start = time.time()
B = 1000
hands_in_round_1 = []
steps = 0
env = BlindEnv(blind_env_config)
profiler = cProfile.Profile()
profiler.enable()
for e in range(B):
    env.reset()
    bot = RoundOneBot()
    while True:
        action = bot.choose_action(env)
        # print(f"Round {e + 1}: Action chosen: {action}")
        obs, reward, done, trunc, info = env.step(action)
        steps += 1
        if reward < 0:
            print(f"Negative reward encountered: {reward}. Action: {action}")
            hands_in_round_1 = hands_in_round_1 + [5]
        if done or info.get("won_round"):
            if reward > 0:
                hands_in_round_1.append(env.hands_played_in_round_1)
            break

profiler.disable()
stop = time.time()
print(f"Average hands played in round 1: {sum(hands_in_round_1)/len(hands_in_round_1)}")
print(f"Time taken for {steps} steps: {stop - start} seconds")
profiler.dump_stats("env_profile.prof")
