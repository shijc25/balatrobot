from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class BlindModelConfig:
    embed_cards: bool = True
    card_embedding_size: int = 32
    embed_suits_ranks: Optional[str] = None
    hidden_size: int = 1024
    hidden_layer_count: int = 1
    action_method: str = "linear_logits"
    ar_head_hidden_size: int = 64
    hand_representation_method: str = "concat"
    self_attention_layers: int = 1
    self_attention_heads: int = 4
    max_jokers: int = 0
    deck_obs: bool = True
    joker_identity_coeff: float = 0.0
    joker_types: int = 151
    suit_rank_entropy_coeff: float = 0.0
    FiLM_mode: Optional[str] = None
    suit_matching_aux_coeff: float = 0.0
    rank_matching_aux_coeff: float = 0.0
    allow_illegal_actions: bool = False
    valid_card_count_coeff: float = 0.0
    joker_spread_loss_coeff: float = 0.0
    weight_decay_coeff: float = 0.0
    blind_obs: bool = True
    num_blind_types: int = 30
    blind_embedding_size: int = 16
    num_experts: int = 3
    available_hand_types_coeff: float = 0.0
    expert_sample_separate: bool = True
    jokers_in_hand_attention: bool = False
    option_variation_coeff: float = 0.0
    suit_count_aux_coeff: float = 0.0
    intent_similarity_coeff: float = 0.0
    forced_play_head: bool = False
    noisy_layers: Sequence[str] = field(default_factory=list)
    subset_hand_types: Optional[str] = None
    scoring_cards_masks: Optional[str] = None
    hand_score_aux_coeff: float = 0.0
    max_supported_hand_size: int = 8
    invalidate_non_minimal: bool = True
    shared_encoder: bool = False
    discard_as_intent: bool = False

    _extra_options: Dict[str, object] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_kwargs(cls, **custom_options: object) -> "BlindModelConfig":
        field_names = {f.name for f in fields(cls) if f.init}
        known: Dict[str, object] = {
            key: value for key, value in custom_options.items() if key in field_names
        }
        config = cls(**known)
        config._extra_options = {
            key: value for key, value in custom_options.items() if key not in field_names
        }
        return config

    def merged_options(self) -> Dict[str, object]:
        base = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        base.update(self._extra_options)
        return base

    def copy_with_overrides(self, **overrides: object) -> "BlindModelConfig":
        merged = self.merged_options()
        merged.update(overrides)
        return self.from_kwargs(**merged)

    def as_kwargs(self, include_extras: bool = True) -> Dict[str, object]:
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        if include_extras:
            data.update(self._extra_options)
        return data

    def require(self, keys: Iterable[str]) -> None:
        missing: List[str] = [key for key in keys if getattr(self, key, None) is None]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
