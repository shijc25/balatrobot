"""Legacy distribution implementations retained for reference."""

from .all_combos import AllCombosDistribution, AllCombosDist
from .combo_index import ComboIndexDistribution, ComboIndexDist
from .mixed_logits import MixedLogitsDistribution, MixedLogitsDist
from .modal_multibinary import ModalMultibinaryDistribution, ModalMultibinaryDist
from .modal_multinomial import ModalMultinomialDistribution, ModalMultinomialDist
from .n_choose_k_sequential import NChooseKSequentialDistribution, NChooseKDistribution
from .n_choose_k_simultaneous import (
    NChooseKSimultaneousDistribution,
    NChooseKSimultaneousDist,
)
from .play_discard_choose import PlayDiscardChooseDistribution, PlayDiscardChooseDist
from .sequential_choice import SequentialChoiceDistribution, SequentialChoiceDist

__all__ = [
    "AllCombosDist",
    "AllCombosDistribution",
    "ComboIndexDist",
    "ComboIndexDistribution",
    "MixedLogitsDist",
    "MixedLogitsDistribution",
    "ModalMultibinaryDist",
    "ModalMultibinaryDistribution",
    "ModalMultinomialDist",
    "ModalMultinomialDistribution",
    "NChooseKDistribution",
    "NChooseKSequentialDistribution",
    "NChooseKSimultaneousDist",
    "NChooseKSimultaneousDistribution",
    "PlayDiscardChooseDist",
    "PlayDiscardChooseDistribution",
    "SequentialChoiceDist",
    "SequentialChoiceDistribution",
]
