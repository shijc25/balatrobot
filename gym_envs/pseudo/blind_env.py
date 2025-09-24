from collections import OrderedDict
import gymnasium as gym
import numpy as np
from balatro_connection import State, Actions
import time
from gamestates import cache_state
from gym_envs.pseudo.hand_type import HandType
from gymnasium import spaces as sp
from balatro_connection import BalatroConnection
from gym_envs.balatro_constants import Suit, rank_lookup
from random import randint, choices, random, choice
from ray.rllib.utils.spaces.repeated import Repeated
from gym_envs.joker_effects import (
    joker_card_score_effects,
    joker_triggered_effects,
    joker_discard_effects,
    joker_round_win_effects,
    joker_round_start_effects,
)
from math import comb
from gym_envs.pseudo.hand import Hand
from gym_envs.pseudo.card import Card
from copy import deepcopy
from gym_envs.joker import Joker
from gym_envs.blind import Blind
from gym_envs.base_card import BaseCard
from gym_envs.pseudo.deck import Deck
from itertools import combinations

from gym_envs.shared_gamestate import SharedGamestate


class PseudoBlindEnv(gym.Env):
    metadata = {"name": "BalatroPseudoBlindEnv-v0"}

    def __init__(self, env_config={"max_hand_size": 15}):
        self.G = SharedGamestate()
        self.G.max_hand_size = env_config.get("max_hand_size", self.G.max_hand_size)
        self.max_hand_size = self.G.max_hand_size

        self.infinite_deck = env_config.get("infinite_deck", False)
        initial_bias = env_config.get("bias", 0.0)

        self.correct_reward = env_config.get("correct_reward", 1.0)
        self.incorrect_penalty = env_config.get("incorrect_penalty", 1.0)
        self.discard_penalty = env_config.get("discard_penalty", 0.05)
        self.rarity_bonus = env_config.get("rarity_bonus", 0.0)
        self.action_mode = env_config.get("action_mode", "combo_index")
        self.hand_mode = env_config.get("hand_mode", "indices")
        self.deck_obs = env_config.get("deck_obs", False)
        self.force_play = env_config.get("force_play", True)
        self.hand_level_randomization = env_config.get(
            "hand_level_randomization", "per_hand"
        )
        self.hands = [
            "High Card",  # 0
            "Pair",  # 1
            "Two Pair",  # 2
            "Three of a Kind",  # 3
            "Straight",  # 4
            "Flush",  # 5
            "Full House",  # 6
            "Four of a Kind",  # 7
            "Straight Flush",  # 8
        ]

        self.biases = {hand: initial_bias for hand in self.hands}
        self.hit_rates = {hand: 0 for hand in self.hands}
        self.target_counts = {hand: 0 for hand in self.hands}
        self.hand_counts = {k: 0 for k in self.hands}
        self.card_slot_counts = {i: 0 for i in range(1, self.max_hand_size + 1)}
        self.count_counts = {i: 0 for i in range(0, 6)}
        self.discard_count_counts = {i: 0 for i in range(0, 6)}
        self.hands_played = 0
        self.discards_played = 0
        self.confusion_matrix = np.zeros((9, 9), dtype=np.float32)
        self.rarities = {hand: 0 for hand in self.hands}
        self.scored_ranks = {
            hand: np.zeros(13, dtype=np.float32) for hand in self.hands
        }
        self.scored_suits = {hand: np.zeros(4, dtype=np.float32) for hand in self.hands}

        self.hand_to_id = {x: i for i, x in enumerate(self.hands)}
        self.id_to_hand = {i: x for i, x in enumerate(self.hands)}

        self.reward_range = (-float("inf"), float("inf"))

        self.objective_mode = env_config.get("objective_mode", "max_chips")
        self.max_discards = 3 if self.objective_mode != "one_hand_easy" else 0
        self.max_plays = 5 if self.objective_mode != "one_hand_easy" else 10
        self.chips_reward_weight = env_config.get("chips_reward_weight", 1.0)
        self.hand_type_reward_weight = env_config.get("hand_type_reward_weight", 0.0)
        self.target_hand_obs = env_config.get("target_hand_obs", False)
        self.cannot_discard_obs = env_config.get("cannot_discard_obs", False)
        self.G.joker_limit = env_config.get("max_jokers", 0)
        self.joker_count_range = env_config.get("joker_count_range", (0, 0))
        self.chip_reward_normalization = env_config.get(
            "chip_reward_normalization", None
        )
        self.suit_homogeneity_bonus = env_config.get("suit_homogeneity_bonus", 0.0)
        self.imagined_trajectories = env_config.get("imagined_trajectories", False)
        self.discard_potential_reward = env_config.get("discard_potential_reward", 0.0)
        self.joker_synergy_bonus = env_config.get("joker_synergy_bonus", 0.0)
        self.flattened_rank_chips = env_config.get("flattened_rank_chips", False)
        self.deck_counts_obs = env_config.get("deck_counts_obs", False)
        self.hand_level_range = env_config.get("hand_level_range", (1, 1))

        self.goal_progress_reward = env_config.get("goal_progress_reward", 0.0)
        self.joker_count_bias_exponent = env_config.get(
            "joker_count_bias_exponent", 0.0
        )
        self.starting_dollars = env_config.get("starting_dollars", 5.0)
        self.blind_obs = env_config.get("blind_obs", True)
        self.round_range = env_config.get("round_range", (1, 1))
        self.contained_hand_types_obs = env_config.get(
            "contained_hand_types_obs", False
        )
        self.deck_cls = env_config.get("deck_cls", Deck)
        self.expert_pretraining = env_config.get("expert_pretraining", False)
        self.subset_hand_types_obs = env_config.get("subset_hand_types_obs", False)
        self.scoring_cards_mask_obs = env_config.get("scoring_cards_mask_obs", False)
        self.failure_progress_penalty = env_config.get("failure_progress_penalty", 0.0)
        self.stake = 0
        self.missed_wins = []
        self.missed_wins_length = 20
        self.missed_wins_p = 0.5
        self.missed_wins_decay_p = 0.5

        self.num_experts = env_config.get("num_experts", 0)

        self.G.owned_jokers = []
        self.forcing_play = False

        self.action_space = PseudoBlindEnv.build_action_space(
            self.G.max_hand_size,
            action_mode=self.action_mode,
            num_experts=self.num_experts,
        )
        self.subset_masks = self.build_action_to_card_mask()
        self.observation_space = self.build_observation_space()

        self.rank_history = []
        self.expert_history = []
        self.average_expert_rewards = {x: 0.5 for x in range(self.num_experts)}
        self.alpha = 0.01

        # Hack to let us switch between multi agent and single agent without breaking the logger
        self.blind_env = self

    def build_action_to_card_mask(self):
        masks = []
        for n in range(1, 6):
            combos = list(combinations(range(self.G.max_hand_size), n))
            # self.num_actions += len(combos)  # count total actions
            for combo in combos:
                mask = np.zeros(self.G.max_hand_size, dtype=bool)
                mask[list(combo)] = True
                masks.append(mask)
        return np.stack(masks, axis=0).astype(bool)

    def hand_subsets(self):
        subhands = []
        scoring_masks = []
        for i in self.subset_masks:
            subhand = Hand()
            for j in range(len(i)):
                if i[j] and len(self.G.hand.cards) > j:
                    subhand.add_card(self.G.hand.cards[j])
            h_type, scored_hand = subhand.evaluate()
            scoring_mask = np.zeros_like(i)
            for j in range(len(self.G.hand.cards)):
                if any(x == self.G.hand.cards[j] for x in scored_hand.cards):
                    scoring_mask[j] = True
            subhands.append(subhand)
            scoring_masks.append(scoring_mask)
        return subhands, scoring_masks

    def subset_available_hands(self, subhands=None):
        if subhands is None:
            subhands, scoring_masks = self.hand_subsets()
        hand_types = []
        for hand in subhands:
            available_hands = np.zeros(8, dtype=np.float32)
            for hand_type in hand.contained_hand_types():
                if hand_type != "High Card":
                    available_hands[self.hand_to_id[hand_type] - 1] = 1.0
            hand_types.append(available_hands)
        return hand_types

    @staticmethod
    def build_action_space(hand_size, action_mode="combo_index", num_experts=0):
        if action_mode == "multi_binary":
            return sp.MultiDiscrete([2] + [2] * hand_size)
        if action_mode == "option_multi_binary":
            return sp.MultiDiscrete([num_experts] + [2] + [2] * hand_size)
        if action_mode == "combo_index":
            return sp.MultiDiscrete([2, 5, 70])
        if action_mode == "stacked_binary_masks":
            return sp.MultiBinary([6, hand_size + 2])
        if action_mode == "mode_count_binary":
            return sp.MultiDiscrete([2, 5] + [2] * hand_size)
        if action_mode == "expert_mode_counts":
            return sp.MultiDiscrete([num_experts] + [2, 5] + [2] * hand_size)
        if action_mode == "subset_index":
            num_combos = sum(comb(hand_size, i) for i in range(1, 6))
            return sp.Discrete(num_combos * 2)
        if action_mode == "dual_subset":
            num_combos = sum(comb(hand_size, i) for i in range(0, 6))
            return sp.Tuple([sp.Discrete(num_combos), sp.Discrete(num_combos)])
        if action_mode == "play_subset_discard_mask":
            num_combos = sum(comb(hand_size, i) for i in range(0, 6))
            return sp.Tuple(
                [sp.Discrete(num_combos), sp.MultiDiscrete([2] * hand_size)]
            )
        raise ValueError(
            f"Unknown action mode: {action_mode}. Supported modes are 'combo_index', 'multi_binary', and 'stacked_binary_masks'."
        )

    def build_observation_space(self):
        hand_indices = sp.Box(
            low=0, high=52, shape=(self.G.max_hand_size,), dtype=np.int8
        )
        deck_indices = sp.Box(low=0, high=52, shape=(52,), dtype=np.int8)

        # hand_suits = sp.Box(low=0, high=4, shape=(self.G.max_hand_size,), dtype=np.int8)
        hand_suits = sp.MultiDiscrete([4] * self.G.max_hand_size)
        hand_ranks = sp.Box(
            low=0, high=13, shape=(self.G.max_hand_size,), dtype=np.int8
        )
        hand_ranks_ordinal = sp.MultiDiscrete([13] * self.G.max_hand_size)

        card_relation_counts = sp.Box(
            low=1,
            high=self.G.max_hand_size,
            shape=(self.G.max_hand_size,),
            dtype=np.float32,
        )

        deck_relation_counts = sp.Box(low=0, high=52, shape=(52 - 8,), dtype=np.float32)

        space = {}
        if self.hand_mode == "indices":
            space["hand_indices"] = hand_indices
        elif self.hand_mode == "suits_ranks":
            space["hand_suits"] = hand_suits
            space["hand_ranks"] = hand_ranks
        elif self.hand_mode == "suits_ranks_w_ordinal":
            space["hand_suits"] = hand_suits
            space["hand_ranks"] = hand_ranks
            space["hand_rank_ordinals"] = hand_ranks_ordinal
        elif self.hand_mode == "base_card":
            space["hand"] = BaseCard.observation_space(self.G.max_hand_size)
        if self.deck_obs:
            space["deck_indices"] = deck_indices

        discards_left = sp.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        hands_left = sp.Box(low=0, high=20, shape=(1,), dtype=np.float32)
        target_hand_types = sp.Discrete(9)

        if self.cannot_discard_obs:
            space["cannot_discard"] = sp.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )

        space["discards_left"] = discards_left
        space["hands_left"] = hands_left
        if self.target_hand_obs:
            space["target_hand_types"] = target_hand_types

        if self.G.joker_limit > 0:
            # space["jokers"] = Joker.observation_space(max_jokers)
            space["jokers"] = BaseCard.observation_space(self.G.joker_limit)
        # space["last_hand_played"] = sp.Box(low=0, high=1, shape=(9,), dtype=np.float32)

        if self.deck_counts_obs:
            space["deck_ranks"] = sp.Box(low=0, high=5, shape=(13,), dtype=np.float32)
            space["deck_suits"] = sp.Box(low=0, high=10, shape=(4,), dtype=np.float32)

        if self.objective_mode == "blind_grind":
            space["round"] = sp.Box(low=-15, high=40, shape=(1,), dtype=np.float32)
            space["chip_goals"] = sp.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
            )
            space["goal_progress"] = sp.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
            )

        if self.blind_obs:
            space["blind_index"] = sp.Box(low=0, high=30, shape=(1,), dtype=np.float32)

        if self.num_experts > 0:
            # In option_multi_binary mode, we need to add an option mask if we have already locked in an action
            space["option_mask"] = sp.Box(
                low=0, high=1, shape=(self.num_experts,), dtype=np.float32
            )

        space["hand_stats"] = HandType.stats_obs_space()
        if self.contained_hand_types_obs:
            space["available_hand_types"] = sp.Box(
                low=0, high=1, shape=(8,), dtype=np.float32
            )

        if self.subset_hand_types_obs:
            num_subsets = self.subset_masks.shape[0]
            space["subset_hand_types"] = sp.Box(
                low=0, high=1, shape=(num_subsets, 8), dtype=np.float32
            )
        if self.scoring_cards_mask_obs:
            space["scoring_cards_masks"] = sp.Box(
                low=0, high=1, shape=(num_subsets, 8), dtype=np.float32
            )
        space["dollars"] = sp.Box(low=-20, high=20, shape=(1,), dtype=np.float32)

        return sp.Dict(space)

    def reset_hand_watermarks(self):
        if len(self.G.hand.cards) == 0:
            self.hand_score_watermark = 0.0
            self.suit_homogeneity_watermark = 0.0
            return
        self.suit_homogeneity_watermark = self.G.hand.suit_homogeneity()
        hand_type = self.G.hand.evaluate()[0]
        chips, mult = self.G.hand_stats[hand_type].scores()
        self.hand_score_watermark = chips * mult

    def calc_suit_homogeneity_bonus(self):
        if self.suit_homogeneity_bonus > 0:
            new_homogeneity = self.G.hand.suit_homogeneity()
            net_homogeneity = new_homogeneity - self.suit_homogeneity_watermark
            if net_homogeneity > 0:
                self.suit_homogeneity_watermark = new_homogeneity
                return net_homogeneity * self.suit_homogeneity_bonus
        return 0.0

    def calc_hand_potential_bonus(self):
        hand_type = self.G.hand.evaluate()[0]
        chips, mult = self.G.hand_stats[hand_type].scores()
        hand_score = chips * mult
        net_hand_score = hand_score - self.hand_score_watermark
        # if net_hand_score > 0:
        self.hand_score_watermark = hand_score
        if self.objective_mode == "max_chips":
            return (
                net_hand_score
                * self.chips_reward_weight
                * self.discard_potential_reward
            )
        elif self.objective_mode == "blind_grind":
            if net_hand_score < 0:
                return 0.0
            return net_hand_score * self.discard_potential_reward / self.chip_goal

    def get_obs(self, reset_hand=False, new_hand=None):
        obs = {}
        rank_counts, suit_counts = self.G.hand.card_dupe_counts()
        available_hands = np.zeros(8, dtype=np.float32)
        for hand in self.G.hand.contained_hand_types():
            if hand != "High Card":
                available_hands[self.hand_to_id[hand] - 1] = 1.0
        if self.contained_hand_types_obs:
            obs["available_hand_types"] = available_hands

        if self.hand_mode == "indices":
            obs["hand_indices"] = np.array(
                [card.index() for card in self.G.hand]
                + [52] * (self.G.max_hand_size - len(self.G.hand)),
                dtype=np.int32,
            )
        elif self.hand_mode == "suits_ranks":
            obs["hand_suits"] = np.array(
                [card.suit_index() for card in self.G.hand], dtype=np.int8
            )
            obs["hand_ranks"] = np.array(
                [card.value - 2 for card in self.G.hand], dtype=np.int8
            )
        elif self.hand_mode == "suits_ranks_w_ordinal":
            obs["hand_suits"] = np.array(
                [card.suit_index() for card in self.G.hand], dtype=np.int8
            )
            obs["hand_ranks"] = np.array(
                [card.value - 2 for card in self.G.hand], dtype=np.int8
            )
            obs["hand_rank_ordinals"] = np.array(
                [card.value - 2 for card in self.G.hand], dtype=np.int8
            )
        elif self.hand_mode == "base_card":
            obs["hand"] = BaseCard.observe_list(
                self.G.hand,
                self.G.max_hand_size,
                override_segment=BaseCard.Segments.HAND,
            )

        if self.deck_obs:
            obs["deck_indices"] = np.array(
                [card.index() for card in self.deck_hand], dtype=np.int32
            )

        if self.cannot_discard_obs:
            obs["cannot_discard"] = np.array(
                [self.discards_left <= 0], dtype=np.float32
            )
        obs["discards_left"] = np.array([self.discards_left], dtype=np.float32)
        obs["hands_left"] = np.array([self.hands_left], dtype=np.float32)
        if self.target_hand_obs:
            obs["target_hand_types"] = np.argmax(self.target_hand_types)

        if self.G.joker_limit > 0:
            obs["jokers"] = BaseCard.observe_list(
                self.G.owned_jokers,
                self.G.joker_limit,
                override_segment=BaseCard.Segments.JOKER,
            )

        if self.deck_counts_obs:
            deck_ranks = np.zeros(13, dtype=np.float32)
            deck_suits = np.zeros(4, dtype=np.float32)
            for card in self.G.deck.remaining_cards:
                deck_ranks[card.value - 2] += 1
                deck_suits[card.suit_index()] += 1
            obs["deck_ranks"] = deck_ranks / 4
            obs["deck_suits"] = deck_suits / 13

        if self.objective_mode == "blind_grind":

            def chip_norms(chips):
                lin = (chips / 50000) - 1
                return [
                    np.log(max(chips, 1) / 300) - 3,
                    (chips / 300) ** (1 / 2) - 3,
                    lin,
                    np.sin(lin * np.pi / 2) * 2,
                    np.cos(lin * np.pi / 2) * 2,
                ]

            obs["chip_goals"] = np.array(chip_norms(self.chip_goal), dtype=np.float32)
            obs["round"] = np.array([self.round], dtype=np.float32)

            remaining_chips = self.chip_goal - self.chips
            remaining_chips = max(remaining_chips, 0)

            obs["goal_progress"] = np.array(
                [np.clip(remaining_chips / self.chip_goal, 0, 1)]
                + chip_norms(remaining_chips),
                dtype=np.float32,
            )

        if self.blind_obs:
            obs["blind_index"] = np.array(
                [self.G.current_blind.index], dtype=np.float32
            )

        if self.num_experts > 0:
            obs["option_mask"] = np.zeros(self.num_experts, dtype=np.float32)
            if self.active_expert is not None:
                obs["option_mask"] = np.ones(self.num_experts, dtype=np.float32)
                obs["option_mask"][int(self.active_expert)] = 0.0

        obs["hand_stats"] = HandType.observe_stats(self.G.hand_stats)

        if self.subset_hand_types_obs:
            subhands, scoring_masks = self.hand_subsets()
            obs["subset_hand_types"] = self.subset_available_hands(subhands=subhands)
            if self.scoring_cards_mask_obs:
                obs["scoring_cards_masks"] = np.stack(scoring_masks, axis=0).astype(
                    np.float32
                )

        scaled_dollars = (self.G.dollars - 100) / 10.0
        scaled_dollars = np.clip(scaled_dollars, -20, 20)
        obs["dollars"] = np.array([scaled_dollars], dtype=np.float32)

        if reset_hand:
            self.last_hand_played = np.zeros(9, dtype=np.float32)

        return obs

    def check_illegal_actions(self, action):
        fail_reasons = super().check_illegal_actions(action)
        if self.discards_left <= 0 and action[0] == Actions.DISCARD_HAND:
            fail_reasons.append("No discards left")

        return fail_reasons

    def step(self, action):
        action = self.action_vector_to_action(action)
        last_hand_played = np.zeros(9, dtype=np.float32)

        illegal_reasons = self.check_illegal_actions(action)
        if len(illegal_reasons) > 0:
            print("YOU CAN'T DO THAT")
            print(illegal_reasons)
            return (self.get_obs(reset_hand=True), -0.1, True, False, {})

        before_hand = deepcopy(self.G.hand)
        could_have = self.G.hand.contained_hand_types()
        before_chips = self.chips
        played_hand = self.G.hand.pop_cards(action[1])
        # if there are no cards played, we auto fail the episode. Shouldn't be possible if the hand is full
        # But can happen if the deck gets thinned enough that the hand is not full
        if len(played_hand) == 0:
            print("YOU CAN'T PLAY AN EMPTY HAND")
            return (self.get_obs(), -0.1, True, False, {})
        if action[0] == Actions.DISCARD_HAND:
            self.discards_played += 1
            self.discards_left -= 1
            self.discard_count_counts[len(played_hand)] += 1

            reward = -self.discard_penalty
            effects = joker_discard_effects(self.G.owned_jokers, played_hand)
            self.handle_callbacks(effects["callbacks"])
            reward += effects["synergy"] * self.joker_synergy_bonus

            if self.round == 1:
                # Check if any of the available hand types would have won the round if we played that instead
                for hand_type in could_have:
                    continue
                    if hand_type in [
                        "Flush",
                        "Straight",
                        "Straight Flush",
                        "Four of a Kind",
                        "Full House",
                    ]:
                        chips, mult = self.G.hand_stats[hand_type].scores()
                        # Assume some small amount of chips from the cards themselves for now
                        # Really this needs to be more sophisticated, at least based on number of scoring cards
                        chips += 20
                        could_have_score = chips * mult
                        if could_have_score + before_chips > self.chip_goal:
                            reward -= self.goal_progress_reward * 0.3
                            self.missed_wins.append(before_hand)

            self.draw_cards()

            # reward += self.calc_hand_potential_bonus()
            reward += self.calc_suit_homogeneity_bonus()
            self.reset_hand_watermarks()
            if len(self.G.hand) == 0:
                return (self.get_obs(reset_hand=True), -0.1, True, False, {})

            return (
                self.get_obs(reset_hand=True),
                reward,
                False,
                False,
                {},
            )
        elif action[0] == Actions.PLAY_HAND:
            self.hands_played += 1
            if self.round == 1:
                self.hands_played_in_round_1 += 1
            self.hands_left -= 1

            play_result = self.determine_play_hand_outcome(played_hand)
            self.update_scored_card_stats(
                play_result["scored_cards"], played_hand, play_result["hand_type"]
            )

            imagined_result = {}
            if self.imagined_trajectories:
                imagined_result = self.imagine_play_hand(played_hand, action)

            if not play_result.get("won_round", False) and not self.hands_left == 0:
                self.draw_cards()
                # Trying to keep the reward signals pure
                # If the agent wants to discard, but we force it to play, then it shouldn't get any reward for the cards played
                # Otherwise we might incentivize discarding valuable hand types accidentally
                if self.forcing_play:
                    play_result["reward"] = -self.discard_penalty

            self.reset_hand_watermarks()
            self.chips += play_result["hand_score"]
            if self.objective_mode in ["blind_grind"]:
                cheat_death = (
                    self.chips < self.chip_goal
                    and self.chips > self.chip_goal * 0.25
                    and self.hands_left == 0
                    and any(j.name == "Mr. Bones" for j in self.G.owned_jokers)
                )

                if self.chips >= self.chip_goal or cheat_death:
                    if cheat_death:
                        self.G.owned_jokers = [
                            j for j in self.G.owned_jokers if j.name != "Mr. Bones"
                        ]

                    play_result["reward"] += (
                        self.hands_left * 0.1 * self.goal_progress_reward
                    )
                    effects = joker_round_win_effects(self.G)
                    self.handle_callbacks(effects["callbacks"])
                    for card in self.G.hand.cards:
                        if card.enhancement == BaseCard.Enhancements.GOLD:
                            self.G.dollars += 3
                    self.round += 1
                    play_result["won_round"] = True
                elif play_result["game_over"] and not self.expert_pretraining:
                    play_result["reward"] -= (
                        self.goal_progress_reward * self.failure_progress_penalty
                    )

                elif len(self.G.hand) == 0:
                    play_result["game_over"] = True

            obs = self.get_obs(reset_hand=True)

            self.last_hand_played[self.hand_to_id[play_result["hand_type"]]] = 1

            info = {
                "imagined_result": imagined_result,
                "won_round": play_result.get("won_round", False),
            }

            # Add supplemental info about joker score contribution
            if not self.forcing_play:
                joker_count = len(self.G.owned_jokers)
                score = play_result.get("joker_marginal", 0)
                chips = play_result.get("joker_chips", 0)
                mult = play_result.get("joker_mult", 0)
                mult_mult = play_result.get("joker_mult_mult", 1.0)

                info["joker_marginal"] = score if joker_count > 0 else 0
                info["joker_chips"] = chips if joker_count > 0 else 0
                info["joker_mult"] = mult if joker_count > 0 else 0
                info["joker_mult_mult"] = mult_mult if joker_count > 0 else 1

            return (obs, play_result["reward"], play_result["game_over"], False, info)
        else:
            raise ValueError(f"Invalid action {action[0]}")

    def determine_play_hand_outcome(self, played_hand):
        game_over = self.hands_left == 0
        no_op_result = {
            "reward": -0.05,
            "game_over": game_over,
            "hand_type": "High Card",
            "scored_cards": played_hand,
            "hand_score": 0,
        }
        if self.G.current_blind.name == "The Psychic":
            if len(played_hand) < 5:
                return no_op_result
        elif self.G.current_blind.name == "The Hook":
            # Pop 2 random cards from the played hand and discard them
            if len(self.G.hand) > 0:
                if len(self.G.hand) == 1:
                    discard_cards = self.G.hand.pop_cards([1])
                else:
                    discard_indices = [randint(1, len(self.G.hand)) for _ in range(2)]
                    while discard_indices[0] == discard_indices[1]:
                        discard_indices[1] = randint(1, len(self.G.hand))
                    discard_cards = self.G.hand.pop_cards(discard_indices)

                effects = joker_discard_effects(self.G.owned_jokers, discard_cards)
                self.handle_callbacks(effects["callbacks"])

        disabled_joker = None
        if self.G.current_blind.name == "Crimson Heart":
            # Disable a random joker for this hand
            if len(self.G.owned_jokers) > 0:
                joker = choice(self.G.owned_jokers)
                self.G.owned_jokers.remove(joker)
                disabled_joker = deepcopy(joker)

        four_finger_joker = any(
            joker.name == "Four Fingers" for joker in self.G.owned_jokers
        )
        hand_type, scored_cards = played_hand.evaluate(
            allow_4_flush=four_finger_joker,
            allow_4_straight=four_finger_joker,
        )

        if self.G.current_blind.name == "The Eye":  # No repeats
            if self.G.hand_stats[hand_type].played_this_blind:
                return no_op_result
        if self.G.current_blind.name == "The Mouth":  # only one hand type per blind
            if not self.G.hand_stats[hand_type].played_this_blind:
                if any(
                    self.G.hand_stats[h].played_this_blind for h in self.G.hand_stats
                ):
                    return no_op_result
        if self.G.current_blind.name == "The Arm":  # Decrease level of played hand
            if self.G.hand_stats[hand_type].level > 1:
                self.G.hand_stats[hand_type].add_level(-1)

        self.G.hand_stats[hand_type].play_count += 1
        self.G.hand_stats[hand_type].played_this_blind = True

        splash = any(joker.name == "Splash" for joker in self.G.owned_jokers)
        if splash:
            scored_cards = played_hand

        if (
            any([j.name == "Space Joker" for j in self.G.owned_jokers])
            and random() < 0.25
        ):
            self.G.hand_stats[hand_type].add_level(1)

        reward = 0
        chips, mult = self.G.hand_stats[hand_type].scores()
        if self.G.current_blind.name == "The Flint":
            chips = max(int(chips / 2), 1)
            mult = max(int(mult / 2), 1)

        if self.flattened_rank_chips:
            chips += 7 * len(scored_cards)
        else:
            chips += sum([card.chip_value() for card in scored_cards])
        chips += sum(
            50 if card.enhancement == BaseCard.Enhancements.STONE else 0
            for card in played_hand.cards
        )
        chips += sum(
            (
                50
                if card.enhancement == BaseCard.Enhancements.STONE
                and card.seal == BaseCard.Seals.RED
                else 0
            )
            for card in played_hand.cards
        )
        pre_joker_score = chips * mult

        scored_cards.cards = self.G.current_blind.filter_scored_cards(
            scored_cards.cards
        )
        aggregate_joker_effects = {"chips": 0, "mult": 0, "mult_mult": 1.0}
        for card in scored_cards.cards:
            if card.enhancement == BaseCard.Enhancements.BONUS:
                chips += 30
            elif card.enhancement == BaseCard.Enhancements.MULT:
                mult += 4
            elif card.enhancement == BaseCard.Enhancements.GLASS:
                mult *= 2
                if random() < 0.25:
                    self.G.destroy_card(card)
                    for j in self.G.owned_jokers:
                        if j.name == "Glass Joker":
                            j.state["mult_mult"] += 0.75
            elif card.enhancement == BaseCard.Enhancements.LUCKY:
                triggers = 0
                if random() < 0.20:
                    mult += 20
                    triggers += 1
                if random() < 0.067:
                    self.G.dollars += 20
                    triggers += 1

                for j in self.G.owned_jokers:
                    if j.name == "Lucky Cat":
                        j.state["mult_mult"] += 0.25 * triggers

            if card.edition == BaseCard.Editions.FOIL:
                chips += 50
            elif card.edition == BaseCard.Editions.HOLOGRAPHIC:
                mult += 10
            elif card.edition == BaseCard.Editions.POLYCHROME:
                mult *= 1.5

            effects = joker_card_score_effects(self.G.owned_jokers, card, self.G)
            aggregate_joker_effects["chips"] += effects["chips"]
            aggregate_joker_effects["mult"] += effects["mult"]
            aggregate_joker_effects["mult_mult"] *= effects["mult_mult"]
            if card.seal == BaseCard.Seals.GOLD:
                self.G.dollars += 3
            elif card.seal == BaseCard.Seals.RED:
                chips += card.chip_value()
                effects["chips"] *= 2
                effects["mult"] *= 2
                effects["mult_mult"] = effects["mult_mult"] ** 2

            self.handle_callbacks(effects["callbacks"])
            chips += effects["chips"]
            mult += effects["mult"]
            mult *= effects["mult_mult"]
            reward += effects["synergy"] * self.joker_synergy_bonus

        for card in self.G.hand.cards:
            if card.enhancement == BaseCard.Enhancements.STEEL:
                mult *= 1.5

        effects = joker_triggered_effects(
            self.G.owned_jokers,
            played_hand,
            scored_cards,
            self.G.hand,
            hand_type,
            self.hands_left,
            self.discards_left,
            self.G.hand_stats,
            self.G.deck,
            self.G.dollars,
            self.G,
        )
        self.handle_callbacks(effects["callbacks"])
        chips += effects["chips"]
        mult += effects["mult"]
        mult *= effects["mult_mult"]
        reward += effects["synergy"] * self.joker_synergy_bonus
        aggregate_joker_effects["chips"] += effects["chips"]
        aggregate_joker_effects["mult"] += effects["mult"]
        aggregate_joker_effects["mult_mult"] *= effects["mult_mult"]

        post_joker_score = chips * mult
        joker_marginal = post_joker_score - pre_joker_score

        if self.objective_mode == "max_chips":
            norm = self.chip_reward_normalization
            crw = self.chips_reward_weight
            if norm == "log_joker":
                reward += pre_joker_score * crw
                reward += np.log(joker_marginal * crw + 1)
            elif norm == "sqrt_joker":
                reward += pre_joker_score * crw
                reward += np.sqrt(joker_marginal * crw)
            elif norm == "log":
                reward += np.log(post_joker_score * crw + 1)
            elif norm == "sqrt":
                reward += np.sqrt(post_joker_score * crw)
            else:
                reward += post_joker_score * crw
        elif self.objective_mode == "blind_grind":
            remaining_progress = (self.chip_goal - self.chips) / self.chip_goal
            progress = post_joker_score / self.chip_goal
            # reward += min(remaining_progress, progress) * self.goal_progress_reward
            overkill = max(0, progress - remaining_progress)
            # if overkill > 0:
            #     # reward += overkill * self.goal_progress_reward * 1.0
            #     reward += np.log(overkill + 1) * self.goal_progress_reward
        elif self.objective_mode == "one_hand_easy":
            if hand_type != self.easy_hand_type and self.easy_hand_type == "Flush":
                reward = 0.5 * max(played_hand.suit_counts().values()) / 10
                if len(played_hand) < 5:
                    reward = 0.0
            elif hand_type == "High Card":
                reward = 0.0
            elif hand_type == self.easy_hand_type:
                reward = 0.5
            else:
                reward = 0.1

        reward += (
            self.target_hand_types[self.hand_to_id[hand_type]]
            * self.hand_type_reward_weight
        )

        avg_rarity = sum(self.rarities.values()) / len(self.rarities)
        rarity = self.rarities[hand_type]
        if rarity > avg_rarity:
            reward += (rarity - avg_rarity) * self.rarity_bonus

        if len(played_hand) == 5:
            reward += played_hand.suit_homogeneity() * self.suit_homogeneity_bonus

        suit_counts = [0, 0, 0, 0]
        for card in played_hand.cards:
            suit_counts[card.suit_index()] += 1

        game_over = self.hands_left == 0
        result = {
            "reward": reward,
            "game_over": game_over,
            "hand_type": hand_type,
            "scored_cards": scored_cards,
            "suit_counts": suit_counts,
            "hand_score": post_joker_score,
            "pre_joker_score": pre_joker_score,
            "joker_marginal": joker_marginal,
            "joker_chips": aggregate_joker_effects["chips"],
            "joker_mult": aggregate_joker_effects["mult"],
            "joker_mult_mult": aggregate_joker_effects["mult_mult"],
            "played_hand": played_hand,
        }

        # retuned = self.expert_pretraining_adjustment(result)
        # if self.active_expert is not None:
        #     self.average_expert_rewards[self.active_expert] = (
        #         self.average_expert_rewards[self.active_expert] * (1 - self.alpha)
        #         + result["reward"] * self.alpha
        #     )
        #     retuned["reward"] /= self.average_expert_rewards[self.active_expert]
        if disabled_joker is not None:
            self.G.owned_jokers.append(disabled_joker)
        return result
        # return retuned

    def check_illegal_actions(self, action):
        fail_reasons = []
        if len(action[1]) < 1 or len(action[1]) > 5:
            fail_reasons.append(f"Invalid number of cards selected: {len(action[1])}")

        return fail_reasons

    def update_scored_card_stats(self, scored_cards, played_cards, hand_type):
        if scored_cards is None:
            return
        suit_map = {
            "Clubs": 0,
            "Diamonds": 1,
            "Hearts": 2,
            "Spades": 3,
        }
        for card in scored_cards:
            self.scored_ranks[hand_type][card.value - 2] += 1
            self.scored_suits[hand_type][suit_map[card.suit]] += 1
        self.hand_counts[hand_type] += 1
        if hand_type == self.target_hand_type:
            self.hit_rates[hand_type] += 1

        self.count_counts[len(played_cards)] += 1

    def action_vector_to_action(self, action_vector, override_action_mode=None):
        action_mode = override_action_mode or self.action_mode
        if action_mode == "combo_index":
            play_or_discard = action_vector[0]
            card_count = action_vector[1] + 1
            index = action_vector[2]
            if index >= len(self.index_to_card_combo[card_count]):
                print("INVALID INDEX (may be dummy sample during initialization)")
                index = 0

            action = [
                Actions.PLAY_HAND if play_or_discard else Actions.DISCARD_HAND,
                list(self.index_to_card_combo[card_count][index]),
            ]

            return action
        elif action_mode == "subset_index":
            i = action_vector
            mode = not (i // self.subset_masks.shape[0])
            card_mask = self.subset_masks[i % self.subset_masks.shape[0]]
            return self.action_vector_to_action(
                [mode, *card_mask], override_action_mode="multi_binary"
            )
        elif action_mode == "dual_subset":
            play_i, discard_i = action_vector
            mode = play_i > 0
            card_mask = np.zeros(self.subset_masks.shape[1], dtype=np.float32)
            if discard_i > 0:
                card_mask = self.subset_masks[discard_i - 1]
            if mode:
                play_mask = self.subset_masks[play_i - 1]
                card_mask = np.logical_or(card_mask, play_mask).astype(np.float32)
            return self.action_vector_to_action(
                [mode, *card_mask], override_action_mode="multi_binary"
            )
        elif action_mode == "play_subset_discard_mask":
            # print(action_vector)
            play_i, discard_mask = action_vector
            mode = play_i > 0
            card_mask = np.zeros(self.subset_masks.shape[1], dtype=np.float32)
            if mode:
                card_mask = self.subset_masks[int(play_i) - 1].copy()
                # if card_mask[8] or card_mask[9]:
                #     print(
                #         "Warning, high index play subset selected:", play_i, card_mask
                #     )
            # if discard_mask[8] or discard_mask[9]:
            #     print("Warning, high index discard mask selected:", discard_mask)

            # limit the mask to at most 5 cards
            for i in range(len(discard_mask)):
                if discard_mask[i] and np.sum(card_mask) < 5:
                    card_mask[i] = 1.0

            # Force selecting at least one card
            if np.sum(card_mask) == 0:
                card_mask[0] = 1.0

            return self.action_vector_to_action(
                [mode, *card_mask], override_action_mode="multi_binary"
            )
        elif action_mode == "multi_binary":
            play_or_discard = action_vector[0]
            if self.discards_left <= 0 and play_or_discard == 0 and self.force_play:
                play_or_discard = 1
                self.forcing_play = True
                # print("forcing play")
            else:
                self.forcing_play = False
            cards = action_vector[1:]
            if len(cards) != self.G.max_hand_size:
                raise ValueError(
                    f"Expected {self.G.max_hand_size} cards, got {len(cards)}"
                )
            card_indices = self.binary_hand_to_card_indices(cards)
            action = [
                Actions.PLAY_HAND if play_or_discard else Actions.DISCARD_HAND,
                card_indices,
            ]
            return action
        elif action_mode == "mode_count_binary":
            mode = action_vector[0]
            count = action_vector[1]
            cards = action_vector[2:]
            # print(mode, count, cards)
            # print([mode] + cards)
            without_count = np.concatenate(([mode], cards), axis=0)
            return self.action_vector_to_action(
                without_count, override_action_mode="multi_binary"
            )
        elif action_mode == "expert_mode_counts":
            if self.active_expert is None:
                self.expert_history.append(action_vector[0])
                # self.active_expert = action_vector[0]
            action_vector = action_vector[1:]  # remove the expert index
            return self.action_vector_to_action(
                action_vector, override_action_mode="mode_count_binary"
            )

        elif action_mode == "option_multi_binary":
            if self.active_expert is None:
                self.expert_history.append(action_vector[0])
                self.active_expert = action_vector[0]
            return self.action_vector_to_action(
                action_vector[1:], override_action_mode="multi_binary"
            )
        elif action_mode == "stacked_binary_masks":
            flattened = np.sum(action_vector, axis=0)
            if np.any(flattened > 1):
                print(
                    "Warning: Some elements in flattened are greater than 1:", flattened
                )

            play_or_discard = flattened[0]
            cards = flattened[
                1:-1
            ]  # exclude the last mask which is the stop flag during sampling
            if len(cards) != self.G.max_hand_size:
                raise ValueError(
                    f"Expected {self.G.max_hand_size} cards, got {len(cards)}"
                )
            card_indices = self.binary_hand_to_card_indices(cards)
            action = [
                Actions.PLAY_HAND if play_or_discard else Actions.DISCARD_HAND,
                card_indices,
            ]
            for i in action[1]:
                self.card_slot_counts[i] += 1
            return action

    def binary_hand_to_card_indices(self, binary_hand):
        return (np.where(binary_hand)[0] + 1).tolist()

    def draw_cards(self):
        if self.objective_mode == "one_hand_easy":
            # hand_type = choice(self.hands)
            hand_i = randint(0, len(self.hands) - 1)
            self.active_expert = hand_i
            hand_type = self.hands[hand_i]
            self.easy_hand_type = hand_type
            self.G.hand = Hand.random(hand_type, self.G.max_hand_size)
            self.G.hand.shuffle()
            return

        starting_size = len(self.G.hand)
        while len(self.G.hand) < self.G.current_hand_size:
            if not self.infinite_deck:
                if len(self.G.deck.remaining_cards) == 0:
                    break
                bias = self.biases[self.target_hand_type]
                if bias is None or bias <= 0:
                    card = self.G.deck.draw()
                else:
                    biasers = [lambda x: 0]
                    if starting_size > 0:
                        if self.target_hand_type in [
                            "Pair",
                            "Three of a Kind",
                            "Four of a Kind",
                            "Two Pair",
                            "Full House",
                        ]:
                            biasers.append(self.G.hand.rank_biaser())
                        if self.target_hand_type in ["Flush"]:
                            biasers.append(self.G.hand.suit_biaser())
                        if self.target_hand_type in ["Straight"]:
                            biasers.append(self.G.hand.straight_biaser())
                        if self.target_hand_type in ["Straight Flush"]:
                            biasers.append(self.G.hand.straight_flush_biaser())
                    biaser = lambda x: sum([b(x) for b in biasers])
                    if len(self.G.deck.remaining_cards) == 0:
                        break
                    card = self.G.deck.draw_biased(biaser, bias)
            else:
                card = Card.random()
            self.G.hand.add_card(card)

        # Sort the hand to make it easier for the model to learn
        # self.G.hand = sorted(self.G.hand, key=lambda x: (x.value, x.suit))
        # self.G.hand.sort()

        # Alternatively, shuffle the hand to force the model to generalize
        self.G.hand.shuffle()

    def get_bias(self):
        return self.biases

    def set_bias(self, bias):
        self.biases = bias

    def get_and_reset_stats(self):
        stats = {
            "confusion": self.confusion_matrix.copy(),
            "target_counts": self.target_counts.copy(),
            "hit_rates": self.hit_rates.copy(),
            "scored_ranks": self.scored_ranks.copy(),
            "scored_suits": self.scored_suits.copy(),
            "hand_counts": self.hand_counts.copy(),
        }
        self.confusion_matrix = np.zeros((9, 9), dtype=np.float32)
        self.target_counts = {hand: 0 for hand in self.hands}
        self.hit_rates = {hand: 0 for hand in self.hands}
        self.scored_ranks = {
            hand: np.zeros(13, dtype=np.float32) for hand in self.hands
        }
        self.scored_suits = {hand: np.zeros(4, dtype=np.float32) for hand in self.hands}
        return stats

    def update_target_hand_type(self):
        self.target_hand_types = np.zeros(9, dtype=np.float32)
        self.target_hand_types -= self.incorrect_penalty
        self.target_hand_type = choice(self.hands)
        self.target_counts[self.target_hand_type] += 1
        chosen_index = self.hand_to_id[self.target_hand_type]
        self.target_hand_types[chosen_index] = self.correct_reward

    def set_rarities(self, rarities):
        self.rarities = rarities

    def expert_pretraining_adjustment(self, old_result):
        result = deepcopy(old_result)
        if not self.expert_pretraining or self.active_expert is None:
            return result

        # 0 is reserved for the generic expert that just tries to maximize score without modification
        hand_type_experts = {
            1: ["Flush", "Straight Flush", "Flush house", "Flush five"],
            2: ["Straight", "Straight Flush"],
            3: ["Straight Flush"],
            4: ["Four of a Kind", "Five of a kind"],
            5: ["Full House", "Flush house"],
            6: ["Three of a Kind"],
            7: ["Two Pair"],
            8: ["Pair"],
        }

        hand_type_shaping = {
            1: lambda hand, played, scored: max(played.suit_counts().values()) * 10,
            2: lambda hand, played, scored: played.longest_run() * 5,
            3: lambda hand, played, scored: (
                played.longest_run() * 20 if hand != "Straight" else 75
            )
            + (max(played.suit_counts().values()) * 20 if hand != "Flush" else 75),
            4: lambda hand, played, scored: {"Three of a Kind": 150, "Pair": 50}.get(
                hand, 0
            ),
            5: lambda hand, played, scored: {
                "Three of a Kind": 150,
                "Two Pair": 100,
                "Pair": 50,
            }.get(hand, 0),
            6: lambda hand, played, scored: {"Pair": 50, "Four of a Kind": 50}.get(
                hand, 0
            ),
            7: lambda hand, played, scored: {"Pair": 50, "Full House": 50}.get(hand, 0),
            8: lambda hand, played, scored: {"Three of a Kind": 50}.get(hand, 0),
        }

        suit_experts = {
            # 9: ["Clubs"],
            # 10: ["Diamonds"],
            # 11: ["Hearts"],
            # 12: ["Spades"],
        }

        rank_experts = {
            # 13: [2, 4, 6, 8, 10],  # Even ranks
            # 14: [3, 5, 7, 9, 14],  # Odd ranks
            # 15: [11, 12, 13],  # Face cards
        }

        if self.active_expert == 0:
            # Expert 0: Generic expert
            return result

        for expert in hand_type_experts:
            if self.active_expert == expert:
                if result["hand_type"] not in hand_type_experts[expert]:
                    result["hand_score"] = hand_type_shaping[expert](
                        result["hand_type"],
                        result["played_hand"],
                        result["scored_cards"],
                    )
                    result["reward"] = result["hand_score"] / self.chip_goal
                else:
                    result["hand_score"] = 500
                    if result["hand_type"] == "Pair":
                        result["hand_score"] = 150
                    result["reward"] = result["hand_score"] / self.chip_goal
                return result
        for expert in suit_experts:
            if self.active_expert == expert:
                match_suit = 0
                off_suit = 0
                for card in result["scored_cards"].cards:
                    if (
                        card.suit in suit_experts[expert]
                        or card.enhancement == BaseCard.Enhancements.WILD
                    ):
                        match_suit += 1
                    else:
                        off_suit += 1
                total_cards = match_suit + off_suit
                if total_cards == 0:
                    match_suit_ratio = 0.1
                else:
                    match_suit_ratio = match_suit / (match_suit + off_suit)
                    match_suit_ratio = max(match_suit_ratio, 0.1)
                # score = (match_suit**2) * match_suit_ratio / 10
                score = max(0.1, match_suit * 2)
                result["hand_score"] = 25 * score
                result["reward"] = result["hand_score"] / self.chip_goal
                return result

        for expert in rank_experts:
            if self.active_expert == expert:
                match_rank = 0
                off_rank = 0
                for card in result["scored_cards"].cards:
                    if (
                        card.value in rank_experts[expert]
                        or card.enhancement == BaseCard.Enhancements.WILD
                    ):
                        match_rank += 1
                    else:
                        off_rank += 1
                total_cards = match_rank + off_rank
                if total_cards == 0:
                    match_rank_ratio = 0.1
                else:
                    match_rank_ratio = match_rank / (match_rank + off_rank)
                    match_rank_ratio = max(match_rank_ratio, 0.1)
                # score = (match_rank**2) * match_rank_ratio / 10
                score = max(0.1, match_rank * 2)
                result["hand_score"] = 25 * score
                result["reward"] = result["hand_score"] / self.chip_goal

        return result

    def handle_callbacks(self, callbacks):
        for callback in callbacks:
            if callback[0] == "destroy_joker":
                if callback[1] in self.G.owned_jokers:
                    self.G.destroy_card(callback[1])
            elif callback[0] == "unlock_joker":
                self.G.unlocked_jokers.add(callback[1])
            elif callback[0] == "add_joker":
                if len(self.G.owned_jokers) < self.G.current_joker_limit:
                    if type(callback[1]) == Joker:
                        self.G.owned_jokers.append(
                            callback[1]
                        )  # check for unimplemented jokers
            elif callback[0] == "earn_money":
                self.G.dollars += callback[1]
            elif callback[0] == "hand_level":
                self.G.hand_stats[callback[1]].add_level(1)
            else:
                print(f"Unhandled callback: {callback}")

    def imagine_play_hand(self, actually_played, action):
        """
        Simulate playing a hand without actually modifying the game state.
        Returns the imagined reward and whether the game would end.
        """
        mutated_hand = actually_played.mutate()
        result = self.determine_play_hand_outcome(mutated_hand)

        # We now have to reconstruct the hand as if these were the cards played by the action
        # The cards must be put back into the slots specified by the action
        imagined_hand = Hand([])
        j = 0
        k = 0
        for i in range(len(self.G.hand) + len(mutated_hand)):
            if i + 1 in action[1]:
                imagined_hand.cards.append(mutated_hand.cards[j])
                j += 1
            else:
                imagined_hand.cards.append(self.G.hand.cards[k])
                k += 1

        result["imagined_hand"] = imagined_hand
        return result

    def fresh_blind(self, fake_reset=False):
        self.G.deck.reset()
        self.G.hand = Hand([])
        if not fake_reset:
            self.G.current_blind = self.next_blind
            self.next_blind = Blind.random(self.round + 1)

        if any(j.name == "Chicot" for j in self.G.owned_jokers):
            # Chicot disables boss blinds, so we'll act like this is a big blind if it's a boss
            if self.G.current_blind.is_boss():
                # most effects are based on name, but large goal blinds need to be adjusted
                if self.G.current_blind.name == "Violet Vessel":
                    self.G.current_blind.chip_goal /= 3
                elif self.G.current_blind.name == "The Wall":
                    self.G.current_blind.chip_goal /= 2
                self.G.current_blind.name = "Big Blind"
                self.G.current_blind.index = 1

        self.chip_goal = self.G.current_blind.chip_goal
        if not self.expert_pretraining:
            self.active_expert = None

        if (
            self.round == 1
            and random() < self.missed_wins_p
            and len(self.missed_wins) > 0
        ):
            i = randint(0, len(self.missed_wins) - 1)
            self.G.hand = deepcopy(self.missed_wins[i])
            if random() < self.missed_wins_decay_p:
                print(f"Missed wins before pop: {len(self.missed_wins)}")
                self.missed_wins.pop(i)
        else:
            self.draw_cards()
        self.chips = 0
        self.reset_hand_watermarks()

        if type(self.max_plays) == dict:
            self.hands_left = self.max_plays[self.target_hand_type]
            self.discards_left = self.max_discards[self.target_hand_type]
        else:
            self.hands_left = self.max_plays
            self.discards_left = self.max_discards

        if self.G.current_blind.name == "The Water":
            # The Water blind allows 0 discards
            self.discards_left = 0
        if self.G.current_blind.name == "The Needle":
            # The Needle blind allows only 1 hand
            self.hands_left = 1

        for h in self.G.hand_stats:
            self.G.hand_stats[h].played_this_blind = False

        if not fake_reset:
            effects = joker_round_start_effects(
                self.G.owned_jokers, self.G.hand, self.hands_left, self.discards_left
            )
            self.hands_left = effects["hands_left"]
            self.discards_left = effects["discards_left"]
            self.handle_callbacks(effects["callbacks"])
        if self.stake >= 4:
            self.discards_left -= 1
            self.discards_left = max(self.discards_left, 0)

    # Initializes various aspects of the gamestate to a reasonable level depending on the round
    # Varies the jokers, deck, hand levels, dollars, etc.
    def catchup(self):
        if self.round <= 1:
            return

        self.G.dollars += self.round * 4
        expected_joker_count = (self.round / 3) + 1
        weights = [1] * 6
        for i in range(0, 6):
            weights[i] = (abs(expected_joker_count - i) + 1) ** -2
        joker_count = choices(range(0, 6), weights=weights, k=1)[0]

        # Ignore rarity since we are assuming that rare jokers are more likely to be purchased when they appear
        # Somewhat offsetting their rarity. Mainly to help the model learn the value of rare jokers better
        self.G.owned_jokers = [
            Joker.random(sparse_pool=False, ignore_rarity=True)
            for _ in range(joker_count)
        ]
        # Add some random state for any jokers that need to scale
        for joker in self.G.owned_jokers:
            if "chips" in joker.state and joker.state["chips"] == 0:
                joker.state["chips"] = int(random() * self.round * 5)
            if "mult" in joker.state and joker.state["mult"] == 0:
                joker.state["mult"] = int(random() * self.round * 2)
            if "mult_mult" in joker.state and joker.state["mult_mult"] == 1:
                joker.state["mult_mult"] = (10 + int(random() * self.round / 10)) / 10

        self.G.dollars -= sum([j.value for j in self.G.owned_jokers])
        self.G.dollars = max(self.G.dollars, 0)

        # Select a "favorite" hand type group
        favorite = choice(self.hands)
        associated_hand_types = {
            "Flush": ["Straight Flush"],
            "Straight": ["Straight Flush"],
            "Straight Flush": ["Flush", "Straight"],
            "Full House": ["Two Pair", "Three of a Kind"],
            "Two Pair": ["Full House", "Three of a Kind", "Pair"],
        }.get(favorite, [])
        main_level = randint(self.round - 5, self.round + 3)
        main_level = max(main_level, 1)
        self.G.hand_stats[favorite].set_level(main_level)
        self.G.hand_stats[favorite].play_count = self.round * 2
        for hand_type in associated_hand_types:
            if hand_type in self.G.hand_stats:
                self.G.hand_stats[hand_type].set_level(max(main_level // 2, 1))
                self.G.hand_stats[hand_type].play_count = self.round
        self.G.hand_stats["High Card"].play_count = self.round * 2

        self.G.dollars -= main_level * 2
        self.G.dollars = max(self.G.dollars, 0)

        # Randomly choose a suit and rank homogeneity level, assuming higher homogeneity is more likely at later rounds
        suit_homogeneity = random() * self.round / 24
        rank_homogeneity = random() * self.round / 24
        random_card_rate = random() * self.round / 10

        preferred_suit = choice(Card.SUITS)
        preferred_rank = choice(Card.RANKS)

        # Randomize deck "thickness"
        deck_size = randint(52 - self.round, 52 + self.round)
        while len(self.G.deck.all_cards) < deck_size:
            self.G.deck.add_card(Card.random())
        while len(self.G.deck.all_cards) > deck_size:
            self.G.deck.all_cards.pop(randint(0, len(self.G.deck.all_cards) - 1))

        # For each card in the deck, randomly decide if it should be made "homogenous" based on the homogeneity levels
        for card in self.G.deck.all_cards:
            if random() < suit_homogeneity:
                card.suit = preferred_suit
            if random() < rank_homogeneity:
                card.value = preferred_rank
            if (
                random() < random_card_rate
                and card.enhancement == BaseCard.Enhancements.NORMAL
                and card.seal == BaseCard.Seals.NO_SEAL
                and card.edition == BaseCard.Editions.NO_EDITION
            ):
                sample_card = Card.random()
                card.enhancement = sample_card.enhancement
                card.seal = sample_card.seal
                card.edition = sample_card.edition

        self.G.dollars -= int(suit_homogeneity)
        self.G.dollars -= int(rank_homogeneity)
        self.G.dollars = max(self.G.dollars, 0)

        # Add a little extra random money to simulate the player having some money saved up for interest
        self.G.dollars += int(random() * self.round * 2)

    def reset(self, seed=None, options=None):
        self.G = SharedGamestate()
        self.G.max_hand_size = self.max_hand_size
        self.update_target_hand_type()
        # self.draw_cards()

        self.hand_counts = {k: 0 for k in self.hands}
        self.card_slot_counts = {i: 0 for i in range(1, self.max_hand_size + 1)}
        self.count_counts = {i: 0 for i in range(0, 6)}
        self.discard_count_counts = {i: 0 for i in range(0, 6)}
        self.hands_played = 0
        self.discards_played = 0

        self.active_expert = None
        self.G.deck = self.deck_cls()
        self.G.hand_stats = HandType.all_hands()
        self.G.owned_jokers = []
        self.G.dollars = self.starting_dollars
        if options and "catchup_round" in options:
            self.round = options["catchup_round"]
            self.catchup()
        else:
            self.round = randint(self.round_range[0], self.round_range[1])

            self.G.unlocked_jokers = set()
            if self.G.current_joker_limit > 0:
                joker_count = choices(
                    range(self.joker_count_range[0], self.joker_count_range[1] + 1),
                    weights=[
                        x**self.joker_count_bias_exponent
                        for x in range(
                            1, self.joker_count_range[1] - self.joker_count_range[0] + 2
                        )
                    ],
                    k=1,
                )[0]
                self.G.owned_jokers = [
                    Joker.random(sparse_pool=False) for _ in range(joker_count)
                ]

            hands_to_randomize = []
            if self.hand_level_randomization == "per_hand":
                hands_to_randomize = self.hands
            elif self.hand_level_randomization == "single_hand":
                hands_to_randomize = [choice(self.hands)]
            for hand_type in hands_to_randomize:
                chosen_hand_level = choices(
                    range(self.hand_level_range[0], self.hand_level_range[1] + 1),
                    weights=[
                        x**0
                        for x in range(
                            1, self.hand_level_range[1] - self.hand_level_range[0] + 2
                        )
                    ],
                    k=1,
                )[0]
                self.G.hand_stats[hand_type].set_level(chosen_hand_level, force=True)
            if self.hand_level_randomization == "single_hand":
                for hand_type in self.hands:
                    if hand_type not in hands_to_randomize:
                        self.G.hand_stats[hand_type].set_level(0, force=True)
            if self.expert_pretraining:
                # For expert pretraining, we randomly select an expert
                self.active_expert = randint(0, self.num_experts - 1)

        # self.draw_cards()

        self.chips = 0
        if type(self.max_plays) == dict:
            self.hands_left = self.max_plays[self.target_hand_type]
            self.discards_left = self.max_discards[self.target_hand_type]
        else:
            self.hands_left = self.max_plays
            self.discards_left = self.max_discards

        self.slots_discarded = np.zeros(0, dtype=np.float32)
        self.slots_played = np.zeros(0, dtype=np.float32)
        self.last_hand_played = np.zeros(9, dtype=np.float32)
        self.next_blind = Blind.random(self.round)
        self.fresh_blind()
        self.reset_hand_watermarks()
        self.rank_history = []
        self.expert_history = []
        self.hands_played_in_round_1 = 0
        return self.get_obs(), {}

    # imports gamestate info from the real game to synchronize the environment
    def load_gamestate(self, G):
        self.G.hand = Hand.from_gamestate_hand(G["hand"])
        self.G.deck = Deck.from_gamestate_deck(G["deck"])
        for k in G["handscores"]:
            # k_lower = k[0] + k[1:].lower()
            self.G.hand_stats[k].chips = G["handscores"][k]["chips"]
            self.G.hand_stats[k].mult = G["handscores"][k]["mult"]
        self.G.dollars = G["dollars"]
        self.chips = G["chips"]
        self.chip_goal = G["current_round"]["chips_required"]
        self.round = G["round"]
        self.hands_left = G["current_round"]["hands_left"]
        self.discards_left = G["current_round"]["discards_left"]
        self.G.current_blind = Blind.from_gamestate(G)
        self.enforce_segments()

    def enforce_segments(self):
        for x in self.G.hand.cards:
            x.segment = BaseCard.Segments.HAND
        for x in self.G.owned_jokers:
            x.segment = BaseCard.Segments.JOKER
