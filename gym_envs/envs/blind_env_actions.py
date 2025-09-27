"""Action handling utilities for :mod:`gym_envs.envs.blind_env`."""

from itertools import combinations
from math import comb
from typing import Dict, Iterable, List, Optional

import numpy as np
from gymnasium import spaces as sp

from balatro_connection import Actions


class BlindActionHelper:
    """Utility object that handles action space definition and decoding."""

    def __init__(self, env) -> None:
        self._env = env
        self.index_to_card_combo: Dict[int, np.ndarray] = {}

    def __setattr__(self, name, value):
        if name == "_env" or not hasattr(self, "_env"):
            object.__setattr__(self, name, value)
        elif name == "index_to_card_combo":
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def initialize_action_lookup(self) -> None:
        """Pre-compute lookup tables for action decoding.

        The default ``combo_index`` action mode expects a mapping from a card
        count to the list of possible index combinations.  Historically this
        lived directly inside :class:`BlindEnv`; moving it here keeps the
        environment focussed on high level flow while still exposing the same
        behaviour.
        """

        self.index_to_card_combo = {
            count: np.array(
                list(combinations(range(1, self.G.max_hand_size + 1), count)),
                dtype=np.int32,
            )
            for count in range(1, 6)
        }

    def build_action_to_card_mask(self) -> np.ndarray:
        masks: List[np.ndarray] = []
        for n in range(1, 6):
            combos = list(combinations(range(self.G.max_hand_size), n))
            for combo in combos:
                mask = np.zeros(self.G.max_hand_size, dtype=bool)
                mask[list(combo)] = True
                masks.append(mask)
        return np.stack(masks, axis=0).astype(bool)

    @staticmethod
    def build_action_space(
        hand_size: int, action_mode: str = "combo_index", num_experts: int = 0
    ) -> sp.Space:
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
            f"Unknown action mode: {action_mode}. Supported modes are 'combo_index',"
            " 'multi_binary', and 'stacked_binary_masks'."
        )

    def action_vector_to_action(
        self, action_vector, override_action_mode: Optional[str] = None
    ):
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
            play_i, discard_mask = action_vector
            mode = play_i > 0
            card_mask = np.zeros(self.subset_masks.shape[1], dtype=np.float32)
            if mode:
                card_mask = self.subset_masks[int(play_i) - 1].copy()
            for i in range(len(discard_mask)):
                if discard_mask[i] and np.sum(card_mask) < 5:
                    card_mask[i] = 1.0
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
            without_count = np.concatenate(([mode], cards), axis=0)
            return self.action_vector_to_action(
                without_count, override_action_mode="multi_binary"
            )
        elif action_mode == "expert_mode_counts":
            if self.active_expert is None:
                self.expert_history.append(action_vector[0])
            action_vector = action_vector[1:]
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
                print("Warning: Some elements in flattened are greater than 1:", flattened)

            play_or_discard = flattened[0]
            cards = flattened[1:-1]
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
        else:
            raise ValueError(f"Unknown action mode: {action_mode}")

    def binary_hand_to_card_indices(self, binary_hand: Iterable[int]) -> List[int]:
        return (np.where(binary_hand)[0] + 1).tolist()

    def check_illegal_actions(self, action) -> List[str]:
        fail_reasons: List[str] = []
        if len(action[1]) < 1 or len(action[1]) > 5:
            fail_reasons.append(f"Invalid number of cards selected: {len(action[1])}")
        if self.discards_left <= 0 and action[0] == Actions.DISCARD_HAND:
            fail_reasons.append("No discards left")
        return fail_reasons
