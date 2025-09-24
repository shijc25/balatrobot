import torch
from modeling.ar_choose_or_stop_stacked import ARChooseOrStopStacked


class FakeModel:
    def __init__(self, hand_size=5, allow_illegal_actions=False, cards_to_allow=None):
        if cards_to_allow is not None:
            self.cards_to_allow = cards_to_allow
        else:
            self.cards_to_allow = hand_size
        self.allow_illegal_actions = allow_illegal_actions
        self.max_supported_hand_size = hand_size

    def ar_step(self, selected_so_far, h):
        selected_so_far = selected_so_far[:, 1:]
        B = selected_so_far.shape[0]
        # return logits for  (cards + STOP)
        hand_size = self.max_supported_hand_size
        logits = torch.zeros(B, hand_size + 1)

        # Set all selected values to -inf
        logits[selected_so_far.bool()] = float("-inf")

        # mask any more cards than we allow
        logits[:, self.cards_to_allow : -1] = float("-inf")

        # make STOP last column have low value normally
        logits[:, -1] = 0
        return logits


def run_test():
    B = 4
    hand_size = 5
    # inputs: mode logits (2), hidden state (4), cannot_discard (1) => total 7
    inputs = torch.randn(B, 2 + 4 + 1)
    # set cannot_discard to zeros
    inputs[:, -1] = 0.0
    model = FakeModel(hand_size=hand_size, allow_illegal_actions=False)
    dist = ARChooseOrStopStacked(inputs, model)

    print("Calling sample...")
    samples = dist.sample(deterministic=False)
    print("Sample shape:", samples.shape)

    print("Computing logp for samples...")
    lp = dist.logp(samples)
    print("Logp:", lp)

    # Test early-stop: craft samples where stop selected immediately
    stop_samples = torch.zeros(B, 6, hand_size + 2)
    stop_samples[:, 0, 0] = 1
    # mark STOP chosen at step 1
    stop_idx = hand_size  # last index (cards + STOP)
    stop_samples[:, 1, 1] = 1  # Choose only the first card
    stop_samples[:, 2, stop_idx + 1] = 1
    print("Logp stop_samples:", dist.logp(stop_samples))

    one_card_model = FakeModel(
        hand_size=hand_size, allow_illegal_actions=False, cards_to_allow=1
    )
    one_card_dist = ARChooseOrStopStacked(inputs, one_card_model)

    print("Calling sample...")
    samples = one_card_dist.sample(deterministic=False)
    print("Sample shape:", samples.shape)

    print("Computing logp for samples...")
    lp = one_card_dist.logp(samples)
    print("Logp:", lp)

    # Test early-stop: craft samples where stop selected immediately
    stop_samples = torch.zeros(B, 6, hand_size + 2)
    stop_samples[:, 0, 0] = 1
    # mark STOP chosen at step 1
    stop_idx = hand_size  # last index (cards + STOP)
    stop_samples[:, 1, 1] = 1  # Choose only the first card
    stop_samples[:, 2, stop_idx + 1] = 1
    print("Logp stop_samples:", one_card_dist.logp(stop_samples))

    impossible_samples = torch.zeros(B, 6, hand_size + 2)
    impossible_samples[:, 0, 0] = 1
    # mark STOP chosen at step 1
    stop_idx = hand_size  # last index (cards + STOP)
    impossible_samples[:, 1, 2] = 1  # Choose only the second card
    impossible_samples[:, 2, stop_idx + 1] = 1
    print("Logp impossible_samples:", one_card_dist.logp(impossible_samples))


if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("EXCEPTION:", e)
