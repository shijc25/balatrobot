import unittest
import torch
from bitwise_hand_types import BitwiseHandTypes


class TestBitwiseHandTypes(unittest.TestCase):
    def setUp(self):
        # Example card_obs for 5 cards: ranks 2, 3, 4, 5, 6 (straight), suits 1
        self.straight_hand = {
            "rank": torch.tensor([[1, 2, 3, 4, 5]]),  # 2-3-4-5-6
            "suit": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        # Example for flush: ranks 2, 5, 7, 9, 13 (Ace), suits all 2
        self.flush_hand = {
            "rank": torch.tensor([[1, 4, 6, 8, 13]]),
            "suit": torch.tensor([[2, 2, 2, 2, 2]]),
        }
        # Example for full house: ranks 2, 2, 2, 3, 3, suits arbitrary (1-4)
        self.full_house_hand = {
            "rank": torch.tensor([[1, 1, 1, 2, 2]]),
            "suit": torch.tensor([[1, 2, 3, 4, 1]]),
        }
        # Example for four of a kind: ranks 5, 5, 5, 5, 2
        self.four_kind_hand = {
            "rank": torch.tensor([[4, 4, 4, 4, 1]]),
            "suit": torch.tensor([[1, 2, 3, 4, 1]]),
        }
        # Example for two pair: ranks 2, 2, 3, 3, 4
        self.two_pair_hand = {
            "rank": torch.tensor([[1, 1, 2, 2, 3]]),
            "suit": torch.tensor([[1, 2, 3, 4, 1]]),
        }
        # Example for three of a kind: ranks 2, 2, 2, 3, 4
        self.three_kind_hand = {
            "rank": torch.tensor([[1, 1, 1, 2, 3]]),
            "suit": torch.tensor([[1, 2, 3, 4, 1]]),
        }
        # Example for pair: ranks 2, 2, 3, 4, 5
        self.pair_hand = {
            "rank": torch.tensor([[1, 1, 2, 3, 4]]),
            "suit": torch.tensor([[1, 2, 3, 4, 1]]),
        }

    def assertArrayEqual(self, a, b):
        self.assertTrue(torch.equal(a, b))

    def test_contained_hand_types(self):
        # Test straight
        hand_types = BitwiseHandTypes.contained_hand_types(self.straight_hand)
        self.assertTrue(hand_types[0, 3].item())  # Straight
        # Test flush
        hand_types = BitwiseHandTypes.contained_hand_types(self.flush_hand)
        self.assertTrue(hand_types[0, 4].item())  # Flush
        # Test full house
        hand_types = BitwiseHandTypes.contained_hand_types(self.full_house_hand)
        self.assertTrue(hand_types[0, 6].item())  # Full House
        # Test four of a kind
        hand_types = BitwiseHandTypes.contained_hand_types(self.four_kind_hand)
        self.assertTrue(hand_types[0, 2].item())  # Four of a Kind
        # Test two pair
        hand_types = BitwiseHandTypes.contained_hand_types(self.two_pair_hand)
        self.assertTrue(hand_types[0, 5].item())  # Two Pair
        # Test three of a kind
        hand_types = BitwiseHandTypes.contained_hand_types(self.three_kind_hand)
        self.assertTrue(hand_types[0, 1].item())  # Three of a Kind
        # Test pair
        hand_types = BitwiseHandTypes.contained_hand_types(self.pair_hand)
        self.assertTrue(hand_types[0, 0].item())  # Pair

    def test_possible_hand_types(self):
        # Test reachable hand types with must_include and illegal masks
        # Example: must include 2, 3, illegal 4, 5, 6, rest legal
        card_obs = {
            "rank": torch.tensor([[1, 2, 3, 4, 5]]),
            "suit": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        must_include_mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        illegal_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
        hand_types = BitwiseHandTypes.possible_hand_types(
            card_obs, illegal_mask, must_include_mask
        )
        # Shouldn't be able to make any hand type
        self.assertFalse(hand_types.any())

        # Test with all cards legal, none must include
        must_include_mask = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.bool)
        illegal_mask = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.bool)
        hand_types = BitwiseHandTypes.possible_hand_types(
            card_obs, illegal_mask, must_include_mask
        )
        self.assertArrayEqual(
            hand_types,
            torch.tensor([[False, False, False, True, True, False, False, True]]),
        )

        # Test with only one card legal
        must_include_mask = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.bool)
        illegal_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
        hand_types = BitwiseHandTypes.possible_hand_types(
            card_obs, illegal_mask, must_include_mask
        )
        # Shouldn't be able to make any hand type
        self.assertFalse(hand_types.any())

    def test_batch_dimension(self):
        card_obs = {
            "rank": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 1, 2, 1]]),
            "suit": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        }

        illegal_mask = torch.zeros_like(card_obs["rank"], dtype=torch.bool)
        must_include_mask = torch.zeros_like(card_obs["rank"], dtype=torch.bool)

        # print("asdf", BitwiseHandTypes.subset_metadata(card_obs, ~illegal_mask))

        hand_types = BitwiseHandTypes.possible_hand_types(
            card_obs, illegal_mask, must_include_mask
        )
        self.assertArrayEqual(
            hand_types,
            torch.tensor(
                [
                    [False, False, False, True, True, False, False, True],
                    [True, True, False, False, True, True, True, False],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
