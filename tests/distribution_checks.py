from modeling.play_discard_multi_binary import PlayDiscardBinaryDist
from modeling.mode_count_multi_binary import ModeCountBinaryDist
import torch
from itertools import combinations
from time import time


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.allow_illegal_actions = False


# dummy_inputs = torch.arange(19).view(1, 19)  # Dummy logits
# dummy_inputs[0, -1] = 1
# dist = PlayDiscardBinaryDist(
#     dummy_inputs,
#     DummyModel(),
# )


# print(dist.mode_logits())
# print(dist.mode_distribution().probs)
# print(dist.cannot_discard_flags())

import torch
import pytest
from modeling.ar_choose_or_stop_stacked import ARChooseOrStopStacked


class DummyModel:
    def __init__(self):
        self.allow_illegal_actions = False
        self.hand_size = 8
        self.max_hand_size = 8

    def ar_step(self, selected_so_far, h):
        B = selected_so_far.size(0)
        logits = torch.full((B, 9), -1e9, device=selected_so_far.device)
        # logits[:, -2:] = 10.0
        logits[:, 0] = 10.0
        logits[:, -1] = 10.0
        return logits


class DummyDist(ARChooseOrStopStacked):
    def __init__(self, inputs):
        super().__init__(inputs, DummyModel())

    def hidden_state_logits(self):
        return torch.zeros(self.inputs.shape[0], 1, device=self.inputs.device)

    def mode_logits(self):
        B = self.inputs.shape[0]
        logits = torch.zeros(B, 2, device=self.inputs.device)
        logits[:, 1] = -1e9  # force mode 0
        return logits

    def adjust_mode_logits(self, logits, cannot_discard):
        return logits

    def cannot_discard_flags(self):
        return torch.zeros(
            self.inputs.shape[0], dtype=torch.bool, device=self.inputs.device
        )


@pytest.fixture
def batch_inputs():
    return torch.randn(5, 2 + 64)


def test_sample_shape_and_one_hot(batch_inputs):
    dist = DummyDist(batch_inputs)
    samples = dist.sample(deterministic=False)
    assert samples.shape == (5, 6, 10)

    # one-hot check: sum over actions per step == 1
    step_sums = samples.sum(dim=2)
    assert torch.all(step_sums >= 0)
    assert torch.all(step_sums <= 1)


def test_logp_shape_and_finite(batch_inputs):
    dist = DummyDist(batch_inputs)
    samples = dist.sample()
    lp = dist.logp(samples)
    assert lp.shape == (5,)
    assert torch.isfinite(lp).all()


def test_reproducible_with_seed(batch_inputs):
    torch.manual_seed(123)
    dist = DummyDist(batch_inputs)
    s1 = dist.sample()
    torch.manual_seed(123)
    dist2 = DummyDist(batch_inputs)
    s2 = dist2.sample()
    assert torch.equal(s1, s2)


def test_stop_flag_masking_on_first_step(batch_inputs):
    dist = DummyDist(batch_inputs)
    samples = dist.sample()
    # ensure no stop (index 9) is chosen in first pick
    assert torch.all(samples[:, 1, 9] == 0)


def test_all_samples_one_card(batch_inputs):
    dist = DummyDist(batch_inputs)
    samples = dist.sample()

    # ensure at least one card is selected in the first step
    assert torch.all(samples[:, 1, :].sum(dim=1) > 0)

    # stop is always the last action
    assert torch.all(samples[:, 2, -1] == 1)


class ModeCountDummyModel:
    def __init__(self):
        self.allow_illegal_actions = False
        self.hand_size = 8


def test_mode_count_shape_and_finite(B):
    dist = ModeCountBinaryDist(torch.randn(B, 64), ModeCountDummyModel())
    samples = dist.sample()
    lp = dist.logp(samples)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()


def test_mode_count_counts_equal_card_counts(B):
    dist = ModeCountBinaryDist(torch.randn(B, 64), ModeCountDummyModel())
    samples = dist.sample()
    # print(samples)
    counts = samples[:, 1]
    cards = samples[:, 2:]
    assert (counts >= 0).all()
    assert (counts <= 4).all()
    assert torch.all(counts + 1 == cards.sum(dim=1)), "Counts do not match card counts"


def test_mode_count_logp_reasonable(B):
    dist = ModeCountBinaryDist(torch.randn(B, 64), ModeCountDummyModel())
    samples = dist.sample()
    lp = dist.logp(samples)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()
    # Check that log probabilities are not all zero or negative infinity
    assert (lp != 0).all() and (
        lp != -float("inf")
    ).all(), "Log probabilities should be reasonable"
    assert torch.all(lp >= -100), "Log probabilities should not be too negative"
    assert torch.all(dist.entropy() >= 0), "Entropy should be non-negative"
    assert torch.all(dist.entropy() < 20), "Entropy should not be too high"


def test_mode_count_forced_logits():
    for mode in [0, 1]:
        for count in [0, 1, 2, 3, 4]:
            for combos in combinations([0, 1, 2, 3, 4, 5, 6, 7], count + 1):
                logits = torch.randn(1, 64)
                logits[0, mode] = 1e9  # force the mode
                logits[0, 2 + mode * 5 + count] = 1e9  # force the count
                for card_index in combos:
                    logits[0, 2 + 10 + 8 * mode + card_index] = (
                        1e9  # force one of the cards to be selected
                    )
                logits[0, 2 + 10 + 16 :] = 0  # disable cannot discard
                dist = ModeCountBinaryDist(logits, ModeCountDummyModel())
                samples = dist.sample()
                # Check that the mode is forced correctly
                assert samples[0, 0] == mode, f"Mode {mode} not forced correctly"
                # Check that the count is forced correctly
                assert samples[0, 1] == count, f"Count {count} not forced correctly"
                for card_index in combos:
                    # Check that the card is selected correctly
                    assert (
                        samples[0, 2 + card_index] == 1
                    ), f"Card {card_index} not selected correctly"
                assert (
                    dist.entropy() < 1e-5
                ), "Entropy should be very low for forced samples"
                # print(dist.logp(samples))
                assert (
                    dist.logp(samples).item() > -0.1
                ), "Log probability should be very high for forced samples"


from modeling.shop_action_and_hand_targets import ShopActionAndHandTargetsDist


def random_shop_inputs(B):
    # Generate random inputs for testing
    action_and_hand_logits = torch.randn(B, 17 + 8)  # 17 actions + 8 hand targets
    target_counts = torch.randint(0, 3, (B, 5))
    inputs = torch.cat(
        [action_and_hand_logits, target_counts], dim=1
    )  # Concatenate logits and counts
    return inputs


def test_shop_action_and_hand_targets(B):
    inputs = random_shop_inputs(B)
    dist = ShopActionAndHandTargetsDist(inputs, DummyModel())

    # Test sample shape
    samples = dist.sample(deterministic=False)
    assert samples["action"].shape == (B,)
    assert samples["hand_targets"].shape == (B, 8)

    # Test logp shape and finite values
    lp = dist.logp(samples)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()
    assert torch.all(lp >= -10), "Log probabilities should not be too negative"
    assert torch.all(lp <= 0), "Log probabilities should not be positive"
    assert torch.all(
        lp <= -0.1
    ), "Log probabilities should not be too high when randomly sampling logits"
    assert torch.all(dist.entropy() >= 0), "Entropy should be non-negative"
    assert torch.all(
        dist.entropy() >= 1
    ), "Entropy should be fairly high when randomly sampling logits"
    assert torch.all(dist.entropy() < 20), "Entropy should not be too high"


def test_shop_action_and_hand_targets_forced(B):
    # Test forced logits for specific actions and targets
    for base_action in range(17):
        for target_count in range(4):
            for hand_target_set in combinations(range(8), target_count):
                inputs = random_shop_inputs(B)
                inputs[:, base_action] = 1e9
                if base_action > 11:
                    inputs[:, 17 + 8 + base_action - 12] = target_count
                # for target in hand_target_set:
                #     inputs[:, 17 + target] = 1e9
                dist = ShopActionAndHandTargetsDist(inputs, DummyModel())
                samples = dist.sample()

                assert (
                    samples["action"] == base_action
                ).all(), f"Action {base_action} not forced correctly"
                if base_action > 11:
                    assert (
                        samples["hand_targets"].sum(dim=1) == target_count
                    ).all(), f"Target count {target_count} not forced correctly"
                    # assert (
                    #     samples["hand_targets"][:, hand_target_set].sum(dim=1)
                    #     == target_count
                    # ).all(), f"Hand targets {hand_target_set} not forced correctly"
                # print(dist.entropy())
                # assert torch.all(
                #     dist.entropy() < 1e-5
                # ), "Entropy should be very low for forced samples"
                # assert torch.all(
                #     dist.logp(samples) > -0.1
                # ), "Log probability should be very high for forced samples"


def benchmark_shop_action_and_hand_targets(B):
    inputs = random_shop_inputs(B)
    timing_samples = 1000  # Number of samples to benchmark
    start_time = time()
    for _ in range(timing_samples):
        dist = ShopActionAndHandTargetsDist(inputs, DummyModel())
    end_time = time()
    print(
        f"Constructing shop action and hand targets dist took {1000*(end_time - start_time)/timing_samples:.2f}ms for batch size {B}."
    )

    start_time = time()
    for _ in range(timing_samples):  # Run 1000 samples
        dist.sample(deterministic=False)
    end_time = time()

    print(
        f"Sampling shop action and hand targets dist took {1000*(end_time - start_time)/timing_samples:.2f}ms for batch size {B}."
    )


from modeling.dual_subset_dist import DualSubsetDist


class DummyDualSubsetModel:
    def __init__(self):
        # Should be able to test all paths with just 6 cards
        self.max_supported_hand_size = 6


import math
import numpy as np


def build_action_to_card_mask(max_hand_size):
    masks = []
    for n in range(1, 6):
        combos = list(combinations(range(max_hand_size), n))
        # self.num_actions += len(combos)  # count total actions
        for combo in combos:
            mask = np.zeros(max_hand_size, dtype=bool)
            mask[list(combo)] = True
            masks.append(mask)
    return np.stack(masks, axis=0).astype(bool)


def test_dual_subset_dist_legal_counts(B):
    num_subsets = 0
    for i in range(6):
        num_subsets += math.comb(6, i)
    inputs = torch.randn(B, num_subsets, 2)
    dist = DualSubsetDist(inputs, DummyDualSubsetModel())

    samples = dist.sample(deterministic=False)
    play_subset_i, discard_subset_i = samples
    action_masks = build_action_to_card_mask(6)

    for i in range(B):
        card_masks = np.zeros((6))
        if play_subset_i[i] > 0:
            card_masks = action_masks[play_subset_i[i] - 1]
        if discard_subset_i[i] > 0:
            card_masks = np.logical_or(
                card_masks, action_masks[discard_subset_i[i] - 1]
            ).astype(np.float32)
        # discard_masks = action_masks[discard_subset_i - 1]
        # play_counts = play_masks.sum(axis=1)
        # discard_counts = discard_masks.sum(axis=1)
        # total_counts = play_counts + discard_counts
        # print(dist.comb_count_indices())
        # print(dist.indices_to_sizes(play_subset_i))
        # print(play_counts)
        # print(dist.indices_to_sizes(play_subset_i) - play_counts)
        # print(card_masks)
        total_counts = card_masks.sum()
        # print(total_counts)

        if total_counts == 0 or total_counts > 5:
            print(
                f"Invalid counts for sample {i}: play_subset_i={play_subset_i[i]}, discard_subset_i={discard_subset_i[i]}, total_counts={total_counts}"
            )
            if play_subset_i[i] > 0:
                print(
                    f"Play subset: {action_masks[play_subset_i[i] - 1]}, indices: {play_subset_i[i]}"
                )
                print("play counts:", action_masks[play_subset_i[i] - 1].sum())
            if discard_subset_i[i] > 0:
                print(
                    f"Discard subset: {action_masks[discard_subset_i[i] - 1]}, indices: {discard_subset_i[i]}"
                )
                print("discard counts:", action_masks[discard_subset_i[i] - 1].sum())
            print(dist.indices_to_sizes(play_subset_i[i]))

    # test_discard_counts = torch.tensor([0, 1, 2, 3, 4, 5])
    # print(dist.max_discard_index(test_discard_counts))

    # print(dist.discard_mask(0, dist.max_discard_index(torch.tensor(0))))

    # assert torch.all(torch.from_numpy(total_counts > 0)) and torch.all(
    #     torch.from_numpy(total_counts <= 5)
    # ), "Total counts must be greater than 0 and less than or equal to 5"


def benchmark_dual_subset_dist(B):
    inputs = torch.randn(B, 200, 2)  # Adjust size as needed
    timing_samples = 1000  # Number of samples to benchmark
    start_time = time()
    for _ in range(timing_samples):
        dist = DualSubsetDist(inputs, DummyDualSubsetModel())
    end_time = time()
    print(
        f"Constructing dual subset dist took {1000*(end_time - start_time)/timing_samples:.2f}ms for batch size {B}."
    )

    start_time = time()
    for _ in range(timing_samples):  # Run 1000 samples
        dist.sample(deterministic=False)
    end_time = time()

    print(
        f"Sampling dual subset dist took {1000*(end_time - start_time)/timing_samples:.2f}ms for batch size {B}."
    )


if __name__ == "__main__":
    B = 10
    # AR Dist tests
    # test_all_samples_one_card(batch_inputs=torch.randn(5, 2 + 64))
    # test_logp_shape_and_finite(batch_inputs=torch.randn(5, 2 + 64))

    # Mode Count Dist tests
    # test_mode_count_shape_and_finite(B)
    # test_mode_count_counts_equal_card_counts(B)
    # test_mode_count_logp_reasonable(B)
    # test_mode_count_forced_logits()

    # Shop Action and Hand Targets Dist tests
    # test_shop_action_and_hand_targets(B)
    # test_shop_action_and_hand_targets_forced(B)
    # benchmark_shop_action_and_hand_targets(B)

    # Dual Subset Dist tests
    # test_dual_subset_dist_legal_counts(B)
    # benchmark_dual_subset_dist(B)
