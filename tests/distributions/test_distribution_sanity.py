import sys
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.distributions import (
    DualSubsetDistribution,
    ModeCountBinaryDistribution,
    PlayDiscardBinaryDistribution,
    ShopActionAndHandTargetsDistribution,
)


@pytest.fixture
def batch_size() -> int:
    return 4


def make_play_discard_distribution(batch: int) -> PlayDiscardBinaryDistribution:
    inputs = torch.randn(batch, 19)
    model = SimpleNamespace(allow_illegal_actions=False)
    return PlayDiscardBinaryDistribution(inputs, model)


def make_mode_count_distribution(batch: int) -> ModeCountBinaryDistribution:
    inputs = torch.randn(batch, 64)
    model = SimpleNamespace(allow_illegal_actions=False)
    return ModeCountBinaryDistribution(inputs, model)


def make_shop_distribution(batch: int) -> ShopActionAndHandTargetsDistribution:
    inputs = torch.randn(batch, 17 + 8 + 5)
    model = SimpleNamespace(max_hand_size=8)
    return ShopActionAndHandTargetsDistribution(inputs, model)


def make_dual_subset_distribution(batch: int) -> DualSubsetDistribution:
    # number of subsets for max_supported_hand_size=6 equals sum_{k=0..6} C(6,k) = 64
    inputs = torch.randn(batch, 64, 2)
    model = SimpleNamespace(max_supported_hand_size=6)
    return DualSubsetDistribution(inputs, model)


def test_play_discard_sample_and_logp(batch_size: int):
    dist = make_play_discard_distribution(batch_size)
    action = dist.sample()
    assert action.shape == (batch_size, 9)

    logp = dist.logp(action)
    assert logp.shape == (batch_size,)
    assert torch.isfinite(logp).all()


def test_mode_count_sample_and_logp(batch_size: int):
    dist = make_mode_count_distribution(batch_size)
    action = dist.sample()
    assert action.shape[0] == batch_size

    logp = dist.logp(action)
    assert logp.shape == (batch_size,)
    assert torch.isfinite(logp).all()


def test_shop_action_sample_shapes(batch_size: int):
    dist = make_shop_distribution(batch_size)
    sample = dist.sample()
    assert set(sample.keys()) == {"action", "hand_targets"}
    assert sample["action"].shape == (batch_size,)
    assert sample["hand_targets"].shape == (batch_size, 8)

    logp = dist.logp(sample)
    assert logp.shape == (batch_size,)
    assert torch.isfinite(logp).all()


def test_dual_subset_sample_range(batch_size: int):
    dist = make_dual_subset_distribution(batch_size)
    play_subset, discard_subset = dist.sample(deterministic=False)
    assert play_subset.shape == (batch_size,)
    assert discard_subset.shape == (batch_size,)

    # indices are expected to be within possible subset count (0..63 inclusive)
    assert torch.all((play_subset >= 0) & (play_subset < 64))
    assert torch.all((discard_subset >= 0) & (discard_subset < 64))
