from __future__ import annotations

from copy import deepcopy
from random import choices, randint, sample, random
from typing import Dict, Optional

import gymnasium as gym
import numpy as np

from balatro_connection import Actions
from gym_envs.base_card import BaseCard
from gym_envs.blind import Blind
from gym_envs.joker import Joker
from gym_envs.joker_effects import joker_discard_effects, joker_round_win_effects
from gym_envs.envs.blind_env_actions import BlindActionHelper
from gym_envs.envs.blind_env_gameplay import BlindGameplayHelper
from gym_envs.envs.blind_env_observations import BlindObservationHelper
from gym_envs.components.deck import Deck
from gym_envs.components.hand_type import HandType
from gym_envs.shared_gamestate import SharedGamestate


class BlindEnv(gym.Env):
    metadata = {"name": "BalatroBlindEnv-v0"}

    def __init__(self, env_config: Optional[Dict] = None):
        env_config = env_config or {"max_hand_size": 15}

        self.G = SharedGamestate()
        self.G.max_hand_size = env_config.get("max_hand_size", self.G.max_hand_size)
        self.max_hand_size = self.G.max_hand_size

        self.infinite_deck = env_config.get("infinite_deck", False)
        initial_bias = env_config.get("bias", 0.0)

        self.correct_reward = env_config.get("correct_reward", 0.0)
        self.incorrect_penalty = env_config.get("incorrect_penalty", 0.0)
        self.discard_penalty = env_config.get("discard_penalty", 0.0)
        self.rarity_bonus = env_config.get("rarity_bonus", 0.0)
        self.action_mode = env_config.get("action_mode", "combo_index")
        self.hand_mode = env_config.get("hand_mode", "indices")
        self.deck_obs = env_config.get("deck_obs", False)
        self.force_play = env_config.get("force_play", True)
        self.hand_level_randomization = env_config.get(
            "hand_level_randomization", "per_hand"
        )
        self.hands = [
            "High Card",
            "Pair",
            "Two Pair",
            "Three of a Kind",
            "Straight",
            "Flush",
            "Full House",
            "Four of a Kind",
            "Straight Flush",
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
        self.max_discards = 3
        self.max_plays = 5
        self.chips_reward_weight = env_config.get("chips_reward_weight", 0.0)
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

        self._action_helper = BlindActionHelper(self)
        self._observation_helper = BlindObservationHelper(self)
        self._gameplay_helper = BlindGameplayHelper(self)

        self.action_space = self.build_action_space(
            self.G.max_hand_size,
            action_mode=self.action_mode,
            num_experts=self.num_experts,
        )
        self.initialize_action_lookup()
        self.subset_masks = self.build_action_to_card_mask()
        self.observation_space = self.build_observation_space()

        self.rank_history = []
        self.expert_history = []
        self.average_expert_rewards = {x: 0.5 for x in range(self.num_experts)}
        self.alpha = 0.01

        self.deck_hand = []

        # Hack to let us switch between multi agent and single agent without breaking the logger
        self.blind_env = self

    # -- Action helper proxies -------------------------------------------------

    def build_action_space(self, *args, **kwargs):
        return self._action_helper.build_action_space(*args, **kwargs)

    def initialize_action_lookup(self):
        return self._action_helper.initialize_action_lookup()

    def build_action_to_card_mask(self):
        return self._action_helper.build_action_to_card_mask()

    def action_vector_to_action(self, *args, **kwargs):
        return self._action_helper.action_vector_to_action(*args, **kwargs)

    def binary_hand_to_card_indices(self, *args, **kwargs):
        return self._action_helper.binary_hand_to_card_indices(*args, **kwargs)

    def check_illegal_actions(self, *args, **kwargs):
        return self._action_helper.check_illegal_actions(*args, **kwargs)

    # -- Observation helper proxies -------------------------------------------

    def hand_subsets(self, *args, **kwargs):
        return self._observation_helper.hand_subsets(*args, **kwargs)

    def subset_available_hands(self, *args, **kwargs):
        return self._observation_helper.subset_available_hands(*args, **kwargs)

    def build_observation_space(self):
        return self._observation_helper.build_observation_space()

    def reset_hand_watermarks(self):
        return self._observation_helper.reset_hand_watermarks()

    def calc_suit_homogeneity_bonus(self):
        return self._observation_helper.calc_suit_homogeneity_bonus()

    def calc_hand_potential_bonus(self):
        return self._observation_helper.calc_hand_potential_bonus()

    def get_obs(self, *args, **kwargs):
        return self._observation_helper.get_obs(*args, **kwargs)

    # -- Gameplay helper proxies ----------------------------------------------

    def determine_play_hand_outcome(self, *args, **kwargs):
        return self._gameplay_helper.determine_play_hand_outcome(*args, **kwargs)

    def update_scored_card_stats(self, *args, **kwargs):
        return self._gameplay_helper.update_scored_card_stats(*args, **kwargs)

    def draw_cards(self, *args, **kwargs):
        return self._gameplay_helper.draw_cards(*args, **kwargs)

    def handle_callbacks(self, *args, **kwargs):
        return self._gameplay_helper.handle_callbacks(*args, **kwargs)

    def imagine_play_hand(self, *args, **kwargs):
        return self._gameplay_helper.imagine_play_hand(*args, **kwargs)

    def fresh_blind(self, *args, **kwargs):
        return self._gameplay_helper.fresh_blind(*args, **kwargs)

    def catchup(self, *args, **kwargs):
        return self._gameplay_helper.catchup(*args, **kwargs)

    def load_gamestate(self, *args, **kwargs):
        return self._gameplay_helper.load_gamestate(*args, **kwargs)

    def enforce_segments(self, *args, **kwargs):
        return self._gameplay_helper.enforce_segments(*args, **kwargs)

    def step(self, action):
        action = self.action_vector_to_action(action)

        illegal_reasons = self.check_illegal_actions(action)
        if len(illegal_reasons) > 0:
            print("YOU CAN'T DO THAT")
            print(illegal_reasons)
            return (self.get_obs(reset_hand=True), 0, True, False, {})
        
        played_hand = self.G.hand.pop_cards(action[1])
        if len(played_hand) == 0:
            print("YOU CAN'T PLAY AN EMPTY HAND")
            return (self.get_obs(), 0, True, False, {})

        if action[0] == Actions.DISCARD_HAND:
            self.discards_played += 1
            self.discards_left -= 1
            self.discard_count_counts[len(played_hand)] += 1

            reward = 0.0

            effects = joker_discard_effects(self.G.owned_jokers, played_hand)
            self.handle_callbacks(effects["callbacks"])

            self.draw_cards()
            self.reset_hand_watermarks()
            if len(self.G.hand) == 0:
                return (self.get_obs(reset_hand=True), 0, True, False, {})
            
            return (
                self.get_obs(reset_hand=True),
                reward,
                False,
                False,
                {},
            )

        if action[0] == Actions.PLAY_HAND:
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
                        1.0 + self.hands_left * 0.1
                    )
                    effects = joker_round_win_effects(self.G)
                    self.handle_callbacks(effects["callbacks"])
                    for card in self.G.hand.cards:
                        if card.enhancement == BaseCard.Enhancements.GOLD:
                            self.G.dollars += 3
                    self.round += 1
                    play_result["won_round"] = True
                    
            if len(self.G.hand) == 0:
                play_result["game_over"] = True

            obs = self.get_obs(reset_hand=True)
            self.last_hand_played[self.hand_to_id[play_result["hand_type"]]] = 1

            info = {
                "imagined_result": imagined_result,
                "won_round": play_result.get("won_round", False),
            }

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

        raise ValueError(f"Invalid action {action[0]}")

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
        self.target_hand_type = choices(self.hands)[0]
        self.target_counts[self.target_hand_type] += 1
        chosen_index = self.hand_to_id[self.target_hand_type]
        self.target_hand_types[chosen_index] = self.correct_reward

    def set_rarities(self, rarities):
        self.rarities = rarities

    def reset(self, seed=None, options=None):
        self.G = SharedGamestate()
        self.G.max_hand_size = self.max_hand_size
        self.update_target_hand_type()

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
                    Joker.random(sparse_pool=False) for _ in 
                    range(joker_count)
                ]

            hands_to_randomize = []
            if self.hand_level_randomization == "per_hand":
                hands_to_randomize = self.hands
            elif self.hand_level_randomization == "single_hand":
                hands_to_randomize = [choices(self.hands)[0]]
            for h in self.hands:
                self.G.hand_stats[h].set_level(1, force=True)
            num_upgraded_hands = randint(1, 2)
            upgraded_hands = sample(self.hands, num_upgraded_hands)
            for hand_type in upgraded_hands:
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
                self.active_expert = randint(0, self.num_experts - 1)

        self.chips = 0
        if isinstance(self.max_plays, dict):
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
