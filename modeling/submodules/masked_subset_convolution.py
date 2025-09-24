from copy import deepcopy
from gymnasium import spaces as sp
import numpy as np
from random import randint, choice, random
import torch
import torch.nn as nn
from gym_envs.base_card import BaseCard
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from itertools import combinations
import torch.utils.bottleneck

from torch.nn.utils.rnn import pad_sequence


class MaskedSubsetConvolutionModel(nn.Module):
    def __init__(
        self,
        d_model=128,
        flat_size=128,
        per_subset_info_size=0,
        projected_queries=False,
        expect_scoring_mask=False,
        aux_outputs=0,
        max_num_cards=8,
        invalidate_non_minimal=False,
        include_zero_subset=False,
        dual_action_logits=False,
        discard_as_intent=False,
        intent_size=32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_num_cards = max_num_cards
        self.num_actions = 0
        self.num_out_layers = 2
        self.expect_scoring_mask = expect_scoring_mask
        self.invalidate_non_minimal = invalidate_non_minimal
        self.include_zero_subset = include_zero_subset
        self.dual_action_logits = dual_action_logits
        self.discard_as_intent = discard_as_intent
        self.intent_size = intent_size

        self.projected_queries = projected_queries

        self.subset_metadata_size = (
            8 + 14 + 6 + 5
        )  # Hand types, rank counts, suit counts, other

        summaries = 0
        if self.expect_scoring_mask:
            summaries += 1
        in_size = int(
            d_model * summaries
            + flat_size
            + per_subset_info_size
            + self.subset_metadata_size * 2.0
        )
        self.num_action_logits = 1
        if dual_action_logits:
            self.num_action_logits = 2
            if discard_as_intent:
                self.num_action_logits = 1 + intent_size
        out_size = self.num_action_logits + aux_outputs

        h_start = 256
        self.out = nn.Sequential(
            # nn.LayerNorm(in_size),
            nn.Linear(in_size, h_start),
            nn.GELU(),
            nn.Linear(h_start, h_start // 2),
            nn.GELU(),
            nn.Linear(h_start // 2, h_start // 4),
            nn.GELU(),
            nn.Linear(h_start // 4, self.num_action_logits),
        )

        self.aux_head = nn.Sequential(
            nn.Linear(in_size, h_start),
            nn.GELU(),
            nn.Linear(h_start, h_start // 2),
            nn.GELU(),
            nn.Linear(h_start // 2, aux_outputs),
        )

        mask_np = self.build_action_to_card_mask()
        self.register_buffer("mask_mat", torch.from_numpy(mask_np).float())

        # Always starts with 0 because we are handling "null" indices
        straight_patterns = torch.tensor(
            [
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2-3-4-5-6
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 3-4-5-6-7
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 4-5-6-7-8
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 5-6-7-8-9
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # 6-7-8-9-10
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],  # 7-8-9-10-J
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],  # 8-9-10-J-Q
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],  # 9-10-J-Q-K
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 10-J-Q-K-A
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 2-3-4-5-A
            ],
            dtype=torch.float32,
        )
        self.register_buffer("straight_patterns", straight_patterns.float())

    def build_action_to_card_mask(self):
        masks = []
        if self.include_zero_subset:
            masks.append(np.zeros(self.max_num_cards, dtype=bool))
            self.num_actions += 1
        for n in range(1, 6):
            combos = list(combinations(range(self.max_num_cards), n))
            self.num_actions += len(combos)  # count total actions
            for combo in combos:
                mask = np.zeros(self.max_num_cards, dtype=bool)
                mask[list(combo)] = True
                masks.append(mask)
        return np.stack(masks, axis=0).astype(bool)

    @staticmethod
    def _counts_using_scatter(cards_vals, subset_masks, n_classes):
        counts = torch.zeros(
            (cards_vals.size(0), n_classes),
            dtype=torch.float32,
            device=cards_vals.device,
        )
        counts.scatter_add_(
            1, cards_vals.to(torch.long), subset_masks.to(torch.float32)
        )
        return counts

    def subset_metadata(
        self, subset_masks, cards, forced_is_invalid=None, invalidate_non_minimal=False
    ):
        B = cards["rank"].size(0)

        if forced_is_invalid is None:
            invalid_cards = (
                (cards["rank"] == 0)
                & (cards["suit"] == 0)
                & (cards["enhancement"] == 0)
            )
            invalid_counts = torch.einsum(
                "bh,mh->bm", invalid_cards.float(), subset_masks
            )
            is_invalid = invalid_counts > 0
        else:
            is_invalid = forced_is_invalid.clone()
        is_valid = ~is_invalid
        flat_idx = is_valid.view(-1).nonzero(as_tuple=False).squeeze(1)  # (N_valid,)
        batch_idx = flat_idx // self.num_actions
        subset_idx = flat_idx % self.num_actions

        # Check if there are no valid subsets in the batch, in which case we return empty metadata
        if is_invalid.all():
            return (
                None,
                is_invalid,
            )

        cards_ranks = cards["rank"][batch_idx]
        cards_suits = cards["suit"][batch_idx]
        valid_masks = subset_masks[subset_idx]

        # includes null class that has to be clipped off
        # rank_oh = F.one_hot(cards_ranks.long(), num_classes=14).float()
        # # includes wild and null
        # suit_oh = F.one_hot(cards_suits.long(), num_classes=6).float()
        # # rank_counts = torch.einsum("bhr,mh->bmr", rank_oh, valid_masks)
        # # suit_counts = torch.einsum("bhs,mh->bms", suit_oh, valid_masks)
        # rank_counts = (rank_oh * valid_masks.unsqueeze(-1)).sum(dim=1)  # [N_valid, 14]
        # suit_counts = (suit_oh * valid_masks.unsqueeze(-1)).sum(dim=1)  # [N_valid, 6]
        rank_counts = self._counts_using_scatter(cards_ranks, valid_masks, n_classes=14)
        suit_counts = self._counts_using_scatter(cards_suits, valid_masks, n_classes=6)
        sizes = valid_masks.sum(dim=-1)

        # 0 the count of null ranks and suits to avoid counting non-existent cards
        rank_counts[:, 0] = 0.0  # null rank
        suit_counts[:, 0] = 0.0  # null suit

        max_same_rank = rank_counts.max(dim=-1).values  # [B, A]
        distinct_ranks = (rank_counts > 0).sum(dim=-1)  # [B, A]
        max_same_suit = suit_counts.max(dim=-1).values
        # Add the count of wilds to the max of same suit
        max_same_suit += suit_counts[:, 5]  # wilds are at index 5

        # straight_hits = torch.einsum("br,qr->bq", rank_counts, self.straight_patterns)
        straight_hits = rank_counts @ self.straight_patterns.T
        max_straight_hits = straight_hits.max(dim=-1).values  # [N_valid]
        is_straight = (sizes >= 5) & (max_straight_hits == 5) & (distinct_ranks >= 5)
        is_flush = (sizes >= 5) & (max_same_suit >= 5)
        is_straight_flush = is_straight & is_flush

        has_pair = (sizes >= 2) & (max_same_rank >= 2)
        has_three = (sizes >= 3) & (max_same_rank >= 3)
        has_four = (sizes >= 4) & (max_same_rank >= 4)
        has_full_house = (sizes >= 5) & has_three & ~has_four & (distinct_ranks == 2)
        num_pairs = (rank_counts >= 2).sum(dim=-1)
        has_two_pair = (sizes >= 4) & (num_pairs == 2)

        num_single_features = 13
        num_ranks = 14
        num_suits = 6

        metadata = torch.empty(
            has_two_pair.size(0),
            num_single_features + num_ranks + num_suits,
            device=rank_counts.device,
        )
        metadata[:, 0] = has_pair.float()
        metadata[:, 1] = has_three.float()
        metadata[:, 2] = has_four.float()
        metadata[:, 3] = has_full_house.float()
        metadata[:, 4] = has_two_pair.float()
        metadata[:, 5] = is_straight.float()
        metadata[:, 6] = is_flush.float()
        metadata[:, 7] = is_straight_flush.float()
        metadata[:, 8] = max_same_rank.float()
        metadata[:, 9] = max_same_suit.float()
        metadata[:, 10] = distinct_ranks.float()
        metadata[:, 11] = max_straight_hits.float()
        metadata[:, 12] = sizes.float()

        rank_start = num_single_features
        metadata[:, rank_start : rank_start + num_ranks] = rank_counts  # [N_valid, 14]
        suit_start = rank_start + num_ranks
        metadata[:, suit_start : suit_start + num_suits] = suit_counts  # [N_valid,  6]

        minimal = (
            (is_straight | is_flush | has_full_house)
            | (has_pair & (sizes == 2))
            | (has_three & (sizes == 3))
            | (has_four & (sizes == 4))
            | (has_two_pair & (sizes == 4))
            | (sizes == 1)
            | (sizes == 0)  # to handle optional zero subset setting
        )

        # If we are using dual action logits, we can't ignore the non-minimal subsets
        # Because discards need them. So we need to return them for future masking
        # UNLESS we are using discard as intent, in which case we can still ignore them
        if invalidate_non_minimal and (
            (not self.dual_action_logits) or self.discard_as_intent
        ):
            # Flag the subsets that are minimal, i.e., have the least number of cards
            # that still satisfy the hand type
            is_invalid[batch_idx, subset_idx] |= ~minimal
            # metadata = [m[minimal] for m in metadata]  # Filter out invalid subsets
            metadata = metadata[minimal]  # Filter out invalid subsets
        # metadata = torch.cat(metadata, dim=-1)

        return metadata, is_invalid, minimal  # [B, A, 8 + 14 + 6 + 5]

    def forward(
        self,
        embeds,
        flat_info,
        per_subset_info=None,
        scoring_mask=None,
        cards=None,
        hand_embeddings=None,
        hand_padding=None,
    ):
        X = embeds  # [B, 8, d_model]
        B = X.size(0)

        # Convert masks to floats: 1.0 for “select”, 0.0 for “ignore”
        mask_in_f = self.mask_mat  # [B, A, 8]
        mask_out_f = 1 - self.mask_mat

        in_metadata, invalid_mask, minimal = self.subset_metadata(
            mask_in_f, cards, invalidate_non_minimal=self.invalidate_non_minimal
        )
        out_metadata, _, _ = self.subset_metadata(
            mask_out_f, cards, forced_is_invalid=invalid_mask
        )
        if invalid_mask.all():
            return (
                torch.zeros(
                    (B, self.num_actions),
                    dtype=torch.float32,
                    device=flat_info.device,
                ),
                torch.zeros(
                    (B, self.num_actions, 1),
                    dtype=torch.float32,
                    device=flat_info.device,
                ),
            )

        is_valid = ~invalid_mask
        flat_idx = is_valid.view(-1).nonzero(as_tuple=False).squeeze(1)  # (N_valid,)
        batch_idx = flat_idx // self.num_actions
        subset_idx = flat_idx % self.num_actions

        summary = torch.cat(
            [
                flat_info[batch_idx],
                in_metadata,
                out_metadata,
                (
                    per_subset_info[batch_idx]
                    if per_subset_info is not None
                    else torch.zeros_like(in_metadata[:, :0])
                ),
            ],
            dim=-1,
        )

        out = self.out(summary)
        if (
            self.dual_action_logits
            and self.invalidate_non_minimal
            and not self.discard_as_intent
        ):
            # We couldn't remove the non-minimal before, so we need to mask them now
            out[:, 0] = out[:, 0].masked_fill(~minimal, -1e9)
        valid_subset_lengths = is_valid.sum(dim=-1)

        if self.discard_as_intent:
            play_logits = out[:, 0:1]
            discard_intents = out[:, 1:]
            discard_logits = torch.einsum(
                "bnd,bd->bn", hand_embeddings[batch_idx], discard_intents
            )
            # mask any discard logits for padding cards
            # discard_logits = discard_logits * (~hand_padding[batch_idx, 1:]).float()
            discard_logits = discard_logits.masked_fill(
                hand_padding[batch_idx, 1:], -1e9
            )
            out = torch.cat([play_logits, discard_logits], dim=1)

        # Even though padding keeps the output rectangular, it will still be different
        # Shapes between batches, and annoyingly RLLIB does not handle that well.
        # So we pad the output to some arbitrary length, and hope that we aren't often exceeding it
        # Depending on VRAM availability later on we can increase this to the
        # max number of subsets to prevent any truncation issues.
        subset_count_to_pad = 500

        logit_splits = out.split(valid_subset_lengths.tolist(), dim=0)
        padded_logit_splits = pad_sequence(
            logit_splits, batch_first=True, padding_value=-1e9
        )
        subset_idx_splits = subset_idx.split(valid_subset_lengths.tolist(), dim=0)
        padded_subset_idx_splits = pad_sequence(
            subset_idx_splits, batch_first=True, padding_value=-1
        )
        if padded_logit_splits.shape[1] >= subset_count_to_pad:
            # If we exceed the padding length, we truncate the output
            print(
                f"Warning: Number of subsets {padded_logit_splits.shape[1]} exceeds padding length {subset_count_to_pad}. Truncating."
            )
            padded_logit_splits = padded_logit_splits[:, :subset_count_to_pad, :]
            padded_subset_idx_splits = padded_subset_idx_splits[:, :subset_count_to_pad]
        else:
            # if we are below the padding length, we pad to the max
            padded_logit_splits = F.pad(
                padded_logit_splits,
                (0, 0, 0, subset_count_to_pad - padded_logit_splits.shape[1]),
                value=-1e9,
            )
            padded_subset_idx_splits = F.pad(
                padded_subset_idx_splits,
                (0, subset_count_to_pad - padded_subset_idx_splits.shape[1]),
                value=-9999,
            )
            # print(padded_logit_splits.shape, padded_subset_idx_splits.shape)

        return padded_logit_splits, padded_subset_idx_splits

        full_logits = torch.full(
            (B, self.num_actions, out.shape[-1]),
            -1e9,
            device=out.device,
            dtype=out.dtype,
        )
        full_logits[batch_idx, subset_idx] = out
        action_logits = full_logits[:, :, : self.num_action_logits]
        if self.num_action_logits == 1:
            action_logits = action_logits.squeeze(-1)

        aux_logits = self.aux_head(summary)
        full_aux_logits = torch.full(
            (B, self.num_actions, aux_logits.shape[-1]),
            -1e9,
            device=aux_logits.device,
            dtype=aux_logits.dtype,
        )
        full_aux_logits[batch_idx, subset_idx] = aux_logits
        return action_logits, full_aux_logits
