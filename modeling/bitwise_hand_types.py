import torch.nn.functional as F
import torch


class BitwiseHandTypes:
    # simple per-device cache to avoid .to() churn
    _pat_cache = {}

    @staticmethod
    def _patterns(device):
        key = (device.type, device.index)
        t = BitwiseHandTypes._pat_cache.get(key)
        if t is None:
            base = torch.tensor(
                [
                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2-3-4-5-6
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 10-J-Q-K-A
                    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 2-3-4-5-A
                ],
                dtype=torch.float32,
            )
            t = base.to(device, non_blocking=True)
            BitwiseHandTypes._pat_cache[key] = t
        return t

    # Calculates information about a subset of the cards in the batch of observation tensors
    def subset_metadata(card_obs, subset_mask):
        cards_ranks = card_obs["rank"]
        cards_suits = card_obs["suit"]

        card_count = subset_mask.sum(dim=-1)
        rank_oh = F.one_hot(cards_ranks.long(), num_classes=14).float()
        suit_oh = F.one_hot(cards_suits.long(), num_classes=6).float()

        rank_counts = (rank_oh * subset_mask.unsqueeze(-1)).sum(dim=1)
        suit_counts = (suit_oh * subset_mask.unsqueeze(-1)).sum(dim=1)

        max_same_rank = rank_counts.max(dim=-1).values
        max_same_suit = suit_counts.max(dim=-1).values

        rank_existance = (rank_counts > 0).float()
        straight_hits = (
            rank_existance @ BitwiseHandTypes._patterns(rank_existance.device).T
        )
        max_straight_hits = straight_hits.max(dim=-1).values

        num_pairs = (rank_counts > 1).sum(dim=-1)

        return {
            "rank_counts": rank_counts,
            "suit_counts": suit_counts,
            "max_straight_hits": max_straight_hits,
            "max_same_rank": max_same_rank,
            "max_same_suit": max_same_suit,
            "num_pairs": num_pairs,
            "card_count": card_count,
        }

    # The maximum achieveable metrics across all possible remaining inclusion sets
    # Note that not all of these metrics are simultaneously achievable
    # I think this could sometimes cause false positives for full houses and straight flushes
    # Other hand types should be okay
    def potential_metadata(must_include_md, could_include_md):
        # maximum_chooseable_cards: [batch_size]
        maximum_chooseable_cards = 5 - must_include_md["card_count"]  # [batch_size]
        # For broadcasting, unsqueeze to [batch_size, 1]
        max_cards = maximum_chooseable_cards.unsqueeze(-1)  # [batch_size, 1]

        # Clamp could_include counts so you can't add more than max_cards per batch
        potential_rank_counts = must_include_md["rank_counts"] + torch.clamp(
            could_include_md["rank_counts"], max=max_cards
        )  # [batch_size, num_ranks]
        max_same_rank = potential_rank_counts.max(dim=1).values  # [batch_size]
        max_same_rank = torch.clamp(
            max_same_rank,
            max=maximum_chooseable_cards - must_include_md["max_same_rank"],
        )  # [batch_size]

        potential_suit_counts = must_include_md["suit_counts"] + torch.clamp(
            could_include_md["suit_counts"], max=max_cards
        )  # [batch_size, num_suits]
        max_same_suit = potential_suit_counts.max(dim=1).values  # [batch_size]
        max_same_suit = torch.clamp(
            max_same_suit,
            max=maximum_chooseable_cards - must_include_md["max_same_suit"],
        )  # [batch_size]

        # Pairs logic
        potential_num_pairs_delta = (potential_rank_counts > 1).sum(
            dim=1
        ) - must_include_md[
            "num_pairs"
        ]  # [batch_size]
        potential_num_pairs_delta = torch.clamp(
            potential_num_pairs_delta, max=maximum_chooseable_cards
        )  # [batch_size]
        potential_num_pairs = (
            must_include_md["num_pairs"] + potential_num_pairs_delta
        )  # [batch_size]

        potential_card_count = torch.clamp(
            must_include_md["card_count"] + could_include_md["card_count"], max=5
        )  # [batch_size]

        potential_rank_existence = (
            potential_rank_counts > 0
        ).float()  # [batch_size, num_ranks]
        potential_straight_hits = (
            potential_rank_existence
            @ BitwiseHandTypes._patterns(potential_rank_existence.device).T
        )  # [batch_size, num_patterns]

        # Clamp straight hits
        potential_straight_hits = torch.clamp(
            potential_straight_hits,
            max=(must_include_md["max_straight_hits"].unsqueeze(-1) + max_cards),
        )  # [batch_size, num_patterns]
        max_potential_straight_hits = potential_straight_hits.max(
            dim=1
        ).values  # [batch_size]

        return {
            "rank_counts": potential_rank_counts,
            "suit_counts": potential_suit_counts,
            "max_straight_hits": max_potential_straight_hits,
            "max_same_rank": max_same_rank,
            "max_same_suit": max_same_suit,
            "num_pairs": potential_num_pairs,
            "card_count": potential_card_count,
        }

    # Produces a boolean vector of size 8 representing each hand type contained in the subset based on this metadata
    def metadata_to_hand_types(metadata):
        B = metadata["rank_counts"].shape[0]
        hand_types = torch.zeros(
            (B, 8), dtype=torch.bool, device=metadata["rank_counts"].device
        )

        hand_types[:, 0] = (metadata["card_count"] >= 2) & (
            metadata["max_same_rank"] >= 2
        )  # Pair
        hand_types[:, 1] = (metadata["card_count"] >= 3) & (
            metadata["max_same_rank"] >= 3
        )  # Three of a Kind
        hand_types[:, 2] = (metadata["card_count"] >= 4) & (
            metadata["max_same_rank"] >= 4
        )  # Four of a Kind
        hand_types[:, 3] = (metadata["card_count"] >= 5) & (
            metadata["max_straight_hits"] >= 5
        )  # Straight
        hand_types[:, 4] = (metadata["card_count"] >= 5) & (
            metadata["max_same_suit"] >= 5
        )  # Flush
        hand_types[:, 5] = (metadata["card_count"] >= 4) & (
            metadata["num_pairs"] >= 2
        )  # Two Pair
        hand_types[:, 6] = (
            (metadata["card_count"] >= 5)
            & (metadata["num_pairs"] >= 2)
            & (metadata["max_same_rank"] >= 3)
        )  # Full House
        hand_types[:, 7] = hand_types[:, 3] & hand_types[:, 4]  # Straight Flush

        return hand_types

    def contained_hand_types(card_obs, mask=None):
        if mask is None:
            mask = torch.ones_like(card_obs["rank"], dtype=torch.bool)

        metadata = BitwiseHandTypes.subset_metadata(card_obs, mask)
        return BitwiseHandTypes.metadata_to_hand_types(metadata)

    # Produces a boolean vector of size 8 representing each hand type
    # Where 1 represents that hand type still being "reachable"
    def possible_hand_types(card_obs, illegal_mask, must_include_mask):
        S = 5  # Maximum number of selected cards

        legal_mask = ~illegal_mask

        chooseable_mask = legal_mask & ~must_include_mask

        must_include_metadata = BitwiseHandTypes.subset_metadata(
            card_obs, must_include_mask
        )

        chooseable_metadata = BitwiseHandTypes.subset_metadata(
            card_obs, chooseable_mask
        )

        potential_metadata = BitwiseHandTypes.potential_metadata(
            must_include_metadata, chooseable_metadata
        )
        return BitwiseHandTypes.metadata_to_hand_types(potential_metadata)
