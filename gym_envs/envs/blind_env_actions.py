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
        if action_mode == "ar_custom_dist":
            return sp.MultiDiscrete([2, 11, 11, 11, 11, 11])
        raise ValueError(
            f"Unknown action mode: {action_mode}. Supported modes are 'combo_index',"
            " 'multi_binary', and 'stacked_binary_masks'."
        )

    def action_vector_to_action(
        self, action_vector, override_action_mode: Optional[str] = None
    ):
        action_mode = override_action_mode or self.action_mode
        if action_mode == "ar_custom_dist":
            mode = action_vector[0]
            cards = []
            hand_actual_size = len(self.G.hand.cards)
            
            for c in action_vector[1:]:
                if c == 10: break
                if c < hand_actual_size:
                    cards.append(c + 1)
                
            if mode == 0 and self.discards_left <= 0: mode = 1
            
            if len(cards) == 0: 
                cards = [1]
                
            return [Actions.PLAY_HAND if mode else Actions.DISCARD_HAND, cards]
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
