"""Observation helpers for :mod:`gym_envs.envs.blind_env`."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces as sp

from gym_envs.base_card import BaseCard
from gym_envs.components.hand import Hand
from gym_envs.components.hand_type import HandType


class BlindObservationHelper:
    """All routines responsible for observation construction and bookkeeping."""

    def __init__(self, env) -> None:
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def hand_subsets(self) -> Tuple[List[Hand], List[np.ndarray]]:
        subhands: List[Hand] = []
        scoring_masks: List[np.ndarray] = []
        for mask in self.subset_masks:
            subhand = Hand()
            for idx, include in enumerate(mask):
                if include and len(self.G.hand.cards) > idx:
                    subhand.add_card(self.G.hand.cards[idx])
            hand_type, scored_hand = subhand.evaluate()
            scoring_mask = np.zeros_like(mask)
            for idx in range(len(self.G.hand.cards)):
                if any(card == self.G.hand.cards[idx] for card in scored_hand.cards):
                    scoring_mask[idx] = True
            subhands.append(subhand)
            scoring_masks.append(scoring_mask)
        return subhands, scoring_masks

    def subset_available_hands(self, subhands: Optional[List[Hand]] = None):
        if subhands is None:
            subhands, _ = self.hand_subsets()
        hand_types: List[np.ndarray] = []
        for hand in subhands:
            available_hands = np.zeros(8, dtype=np.float32)
            for hand_type in hand.contained_hand_types():
                if hand_type != "High Card":
                    available_hands[self.hand_to_id[hand_type] - 1] = 1.0
            hand_types.append(available_hands)
        return hand_types

    def build_observation_space(self) -> sp.Dict:
        hand_indices = sp.Box(low=0, high=52, shape=(self.G.max_hand_size,), dtype=np.int8)
        deck_indices = sp.Box(low=0, high=52, shape=(52,), dtype=np.int8)

        hand_suits = sp.MultiDiscrete([4] * self.G.max_hand_size)
        hand_ranks = sp.Box(low=0, high=13, shape=(self.G.max_hand_size,), dtype=np.int8)
        hand_ranks_ordinal = sp.MultiDiscrete([13] * self.G.max_hand_size)

        card_relation_counts = sp.Box(
            low=1,
            high=self.G.max_hand_size,
            shape=(self.G.max_hand_size,),
            dtype=np.float32,
        )

        deck_relation_counts = sp.Box(low=0, high=52, shape=(52 - 8,), dtype=np.float32)

        space: Dict[str, sp.Space] = {}
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
            space["cannot_discard"] = sp.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        space["discards_left"] = discards_left
        space["hands_left"] = hands_left
        if self.target_hand_obs:
            space["target_hand_types"] = target_hand_types

        if self.G.joker_limit > 0:
            space["jokers"] = BaseCard.observation_space(self.G.joker_limit)

        if self.deck_counts_obs:
            space["deck_ranks"] = sp.Box(low=0, high=5, shape=(13,), dtype=np.float32)
            space["deck_suits"] = sp.Box(low=0, high=10, shape=(4,), dtype=np.float32)

        if self.objective_mode == "blind_grind":
            space["round"] = sp.Box(low=-15, high=40, shape=(1,), dtype=np.float32)
            space["chip_goals"] = sp.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
            space["goal_progress"] = sp.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        if self.blind_obs:
            space["blind_index"] = sp.Box(low=0, high=30, shape=(1,), dtype=np.float32)

        if self.num_experts > 0:
            space["option_mask"] = sp.Box(low=0, high=1, shape=(self.num_experts,), dtype=np.float32)

        space["hand_stats"] = HandType.stats_obs_space()
        if self.contained_hand_types_obs:
            space["available_hand_types"] = sp.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        if self.subset_hand_types_obs:
            num_subsets = self.subset_masks.shape[0]
            space["subset_hand_types"] = sp.Box(
                low=0, high=1, shape=(num_subsets, 8), dtype=np.float32
            )
            if self.scoring_cards_mask_obs:
                space["scoring_cards_masks"] = sp.Box(
                    low=0, high=1, shape=(num_subsets, 8), dtype=np.float32
                )
        elif self.scoring_cards_mask_obs:
            num_subsets = self.subset_masks.shape[0]
            space["scoring_cards_masks"] = sp.Box(
                low=0, high=1, shape=(num_subsets, 8), dtype=np.float32
            )
        space["dollars"] = sp.Box(low=-20, high=20, shape=(1,), dtype=np.float32)

        return sp.Dict(space)

    def reset_hand_watermarks(self) -> None:
        if len(self.G.hand.cards) == 0:
            self.hand_score_watermark = 0.0
            self.suit_homogeneity_watermark = 0.0
            return
        self.suit_homogeneity_watermark = self.G.hand.suit_homogeneity()
        hand_type = self.G.hand.evaluate()[0]
        chips, mult = self.G.hand_stats[hand_type].scores()
        self.hand_score_watermark = chips * mult

    def calc_suit_homogeneity_bonus(self) -> float:
        if self.suit_homogeneity_bonus > 0:
            new_homogeneity = self.G.hand.suit_homogeneity()
            net_homogeneity = new_homogeneity - self.suit_homogeneity_watermark
            if net_homogeneity > 0:
                self.suit_homogeneity_watermark = new_homogeneity
                return net_homogeneity * self.suit_homogeneity_bonus
        return 0.0

    def calc_hand_potential_bonus(self) -> float:
        hand_type = self.G.hand.evaluate()[0]
        chips, mult = self.G.hand_stats[hand_type].scores()
        hand_score = chips * mult
        net_hand_score = hand_score - self.hand_score_watermark
        self.hand_score_watermark = hand_score
        if self.objective_mode == "max_chips":
            return net_hand_score * self.chips_reward_weight * self.discard_potential_reward
        if self.objective_mode == "blind_grind":
            if net_hand_score < 0:
                return 0.0
            return net_hand_score * self.discard_potential_reward / self.chip_goal
        return 0.0

    def get_obs(self, reset_hand: bool = False, new_hand=None):
        obs: Dict[str, np.ndarray] = {}
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
            obs["cannot_discard"] = np.array([
                self.discards_left <= 0
            ], dtype=np.float32)
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
        elif self.scoring_cards_mask_obs:
            _, scoring_masks = self.hand_subsets()
            obs["scoring_cards_masks"] = np.stack(scoring_masks, axis=0).astype(
                np.float32
            )

        scaled_dollars = (self.G.dollars - 100) / 10.0
        scaled_dollars = np.clip(scaled_dollars, -20, 20)
        obs["dollars"] = np.array([scaled_dollars], dtype=np.float32)

        if reset_hand:
            self.last_hand_played = np.zeros(9, dtype=np.float32)

        return obs
