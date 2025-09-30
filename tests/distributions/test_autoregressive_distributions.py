import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from modeling.distributions import ARBinaryDistribution, ARChooseOrStopStackedDistribution


class FakeModel:
    def __init__(self, hand_size=5, allow_illegal_actions=False, cards_to_allow=None):
        self.max_supported_hand_size = hand_size
        self.allow_illegal_actions = allow_illegal_actions
        self.cards_to_allow = cards_to_allow if cards_to_allow is not None else hand_size

    def ar_step(self, selected_so_far, hidden_state):
        selected_so_far = selected_so_far[:, 1:]
        batch = selected_so_far.shape[0]
        logits = torch.zeros(batch, self.max_supported_hand_size + 1)
        logits[selected_so_far.bool()] = float("-inf")
        logits[:, self.cards_to_allow : -1] = float("-inf")
        logits[:, -1] = 0.0
        return logits


@pytest.fixture
def base_inputs():
    batch = 4
    hand_size = 5
    hidden_size = 4
    inputs = torch.randn(batch, 2 + hidden_size + 1)
    inputs[:, -1] = 0.0  # allow discards
    return inputs


def test_sample_shape_and_finitemass(base_inputs):
    model = FakeModel(hand_size=5, allow_illegal_actions=False)
    dist = ARChooseOrStopStackedDistribution(base_inputs, model)

    samples = dist.sample(deterministic=False)

    assert samples.ndim == 3
    assert samples.shape[0] == base_inputs.shape[0]
    assert samples.shape[-1] == model.max_supported_hand_size + 2
    logp = dist.logp(samples)
    assert torch.isfinite(logp).all(), "log probabilities should be finite for legal samples"


def test_card_cap_enforced_in_logp(base_inputs):
    model = FakeModel(hand_size=5, cards_to_allow=1)
    dist = ARChooseOrStopStackedDistribution(base_inputs, model)

    samples = dist.sample(deterministic=True)
    logp = dist.logp(samples)
    assert torch.isfinite(logp).all()

    # craft illegal sample selecting two distinct cards before stopping
    batch = base_inputs.shape[0]
    hand_size = model.max_supported_hand_size
    illegal = torch.zeros(batch, samples.shape[1], hand_size + 2)
    illegal[:, 0, 0] = 1  # choose play mode
    illegal[:, 1, 1] = 1  # select first card
    illegal[:, 2, 2] = 1  # select second card -> exceeds allowance
    illegal[:, 3, hand_size + 1] = 1  # attempt to stop

    illegal_logp = dist.logp(illegal)
    assert (illegal_logp <= -20).all(), "illegal sequences should receive extremely low log-probability"


class BinaryDummyModel:
    def __init__(self, hand_size=3, allow_illegal_actions=False):
        self.hand_size = hand_size
        self.allow_illegal_actions = allow_illegal_actions

    def ar_step(self, prev, hidden_state):
        batch = hidden_state.shape[0]
        logits = torch.zeros(batch, 2, device=hidden_state.device)
        logits[:, 1] = 0.5
        return logits, hidden_state


def test_ar_binary_distribution_samples_valid(base_inputs):
    model = BinaryDummyModel(hand_size=3, allow_illegal_actions=False)
    inputs = base_inputs.clone()
    dist = ARBinaryDistribution(inputs, model)

    action = dist.sample(deterministic=False)
    assert action.shape == (inputs.shape[0], model.hand_size + 1)
    assert torch.isfinite(dist.logp(action)).all()


def test_ar_binary_distribution_respects_illegal_mask(base_inputs):
    model = BinaryDummyModel(hand_size=3, allow_illegal_actions=False)
    inputs = base_inputs.clone()
    inputs[:, -1] = 1.0  # cannot discard
    dist = ARBinaryDistribution(inputs, model)

    hidden = dist.hidden_state_logits()
    prev = torch.zeros(inputs.shape[0], 1, device=inputs.device)
    mode_logits, _ = model.ar_step(prev, hidden)
    adjusted = dist.adjust_mode_logits(mode_logits, dist.cannot_discard_flags())
    assert torch.all(adjusted[:, 0] == -1e9)
