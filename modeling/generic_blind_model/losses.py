from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.sample_batch import SampleBatch

from gym_envs.base_card import BaseCard
from gym_envs.joker import Joker


class AuxiliaryLossMixin:
    def aux_logits(self):
        if self.aux_outputs > 0:
            return self._last_aux_outputs
        else:
            return None

    def joker_aux_loss(self, policy_loss, loss_inputs, aux_logits):
        # try to predict the jokers that were in the input
        # reshape to be a 3D tensor of shape [batch_size, max_jokers, 151]
        batch_size = aux_logits.shape[0]
        max_jokers = self.max_jokers
        aux_logits = aux_logits.view(batch_size, self.joker_types, max_jokers)
        mask = Joker.implemented_mask()
        mask = (
            mask.to(aux_logits.device)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, -1, max_jokers)
        )
        aux_logits = torch.where(
            mask, aux_logits, torch.tensor(-1e9, device=aux_logits.device)
        )

        # Need to get the joker indices from the observation, but first we undo the flattening
        # of the joker indices in the input_dict
        original_obs = restore_original_dimensions(
            loss_inputs["obs"].cpu(), self.obs_space, "torch"
        )
        actual_jokers = original_obs["jokers"]["indices"].to(aux_logits.device)
        actual_jokers = actual_jokers.to(torch.int64)
        actual_jokers = actual_jokers - BaseCard.FIRST_JOKER_INDEX + 1
        actual_jokers = torch.clamp(actual_jokers, min=0)

        ce_loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        aux_loss = ce_loss(aux_logits, actual_jokers)
        self._last_joker_aux_loss = aux_loss.detach().cpu().item()
        return aux_loss

    def suit_rank_entropy_loss(self, loss_inputs):
        """
        Computes the *global* rank‐ and suit‐entropy for all “done” & valid hands in the batch,
        and returns (H_rank_global + H_suit_global).  Concretely:

        1) Filter to only those timesteps where done==True and reward>=0 (valid plays).
        2) From each such obs, extract the first SLOTS entries of `hand_indices` so that
        SLOTS = self.action_logits.shape[1].  (This ensures we only look at exactly
        the card‐slots your policy is choosing between.)
        3) For each hand b in that filtered set, compute
            q_{b,r} = 1 - ∏_{i : rank[b,i] = r} (1 - p_{b,i}),
        where p_{b,i} = sigmoid(logit_{b,i}), and rank[b,i] = (hand_indices[b,i] % 13).
        4) Average q_{b,r} over b to get \overline{q}_r.  Normalize p_r = \overline{q}_r / ∑_r \overline{q}_r.
        5) H_rank_global = -∑_r p_r log(p_r + eps).  (Similarly for suits.)
        6) Return (H_rank_global + H_suit_global).

        This exactly matches “entropy of the aggregated rank probabilities,” so if your logging
        callback also computes
        p_r_agg = scored_ranks_flat[r] / ∑_r scored_ranks_flat[r]
        H_metric = -∑_r p_r_agg log(p_r_agg + eps),
        they will be proportional (in fact, numerically identical).
        """

        # 1) Build the “valid play” mask: done & non-negative reward
        done_mask = loss_inputs["dones"].bool()  # shape (TOTAL_STEPS,)
        done_mask &= loss_inputs["rewards"] >= 0.0  # filter out discards if reward<0

        if not done_mask.any():
            # No valid “done & play” steps in this batch → zero entropy
            return torch.tensor(0.0, device=self.action_logits.device)

        # 2) Extract only those observations & logits
        obs_done = loss_inputs["obs"][
            done_mask
        ]  # a dict; e.g. obs_done["hand_indices"] is (B, 18)
        logits_done = self.action_logits[
            done_mask, : self.hand_size
        ]  # shape (B, SLOTS)

        # 3) Restore dims to get the original “hand_indices” as LongTensor (B, 18)
        original_obs = restore_original_dimensions(
            obs_done.cpu(), self.obs_space, "torch"
        )
        hand_indices_full = original_obs["hand_indices"].to(
            self.action_logits.device
        )  # (B, 18)

        # 4) Slice exactly the first SLOTS columns so that we match logits_done.shape[1]
        SLOTS = logits_done.shape[1]  # e.g. 8
        hand_indices = hand_indices_full[:, :SLOTS]  # shape (B, SLOTS)

        # 5) Compute p_{b,i} = sigmoid(logit_{b,i})  for each slot
        probs = torch.sigmoid(logits_done)  # shape (B, SLOTS)

        # 6) ---- RANK side: build mask and q_{b,r} ----

        # a) Decode rank[b,i] = hand_indices[b,i] % 13, shape (B, SLOTS)
        hand_ranks = (hand_indices % 13).long()  # dtype=long

        # b) Build a boolean mask of shape (B, 13, SLOTS):
        #    mask_rank[b, r, i] = 1 if hand_ranks[b,i] == r, else 0
        device = probs.device
        rank_range = torch.arange(13, device=device).view(1, 13, 1)  # shape (1,13,1)
        mask_rank = (
            hand_ranks.unsqueeze(1) == rank_range
        ).float()  # shape (B,13,SLOTS)

        # c) Compute for each (b,r):  prod_{i: rank[b,i]==r} (1 - p[b,i])
        one_minus = (1.0 - probs).unsqueeze(1)  # shape (B, 1, SLOTS)
        # (one_minus * mask_rank) picks out (1-p_{b,i}) only for i with rank r
        # (1 - mask_rank) is 1 for i not in that rank, so those factors contribute a “1” to the product.
        prod_terms_r = (one_minus * mask_rank + (1.0 - mask_rank)).prod(
            dim=2
        )  # shape (B,13)

        # d) q_ranks[b,r] = 1 - product_term
        q_ranks = 1.0 - prod_terms_r  # shape (B,13), each entry in [0,1]

        # e) Average across batch to get \overline{q}_r
        mean_q_ranks = q_ranks.mean(dim=0)  # shape (13,)

        # f) Normalize into a proper distribution p_r = mean_q_ranks / (sum + eps)
        eps = 1e-8
        sum_qr = mean_q_ranks.sum()
        p_r = mean_q_ranks / (sum_qr + eps)  # shape (13,), sums ≈ 1

        # g) Compute global rank-entropy: H_rank_global = -∑_r p_r log(p_r + eps)
        H_rank_global = -torch.sum(p_r * torch.log(p_r + eps))  # scalar

        # Optionally store for debugging
        self._last_rank_entropy_loss = H_rank_global.detach().cpu().item()

        # 7) ---- SUIT side: build mask and q_{b,s} ----

        # a) Decode suit[b,i] = hand_indices[b,i] // 13, shape (B, SLOTS)
        hand_suits = (hand_indices // 13).long()

        # b) Build a boolean mask of shape (B, 4, SLOTS):
        suit_range = torch.arange(4, device=device).view(1, 4, 1)
        mask_suit = (hand_suits.unsqueeze(1) == suit_range).float()  # (B,4,SLOTS)

        # c) Compute prod_{i: suit[b,i]==s} (1 - p[b,i]) same as above
        prod_terms_s = (one_minus * mask_suit + (1.0 - mask_suit)).prod(
            dim=2
        )  # shape (B,4)

        # d) q_suits[b,s] = 1 - product
        q_suits = 1.0 - prod_terms_s  # shape (B,4)

        # e) Average across batch to get \overline{q}_s
        mean_q_suits = q_suits.mean(dim=0)  # shape (4,)

        # f) Normalize into p_s = mean_q_suits / (sum + eps)
        sum_qs = mean_q_suits.sum()
        p_s = mean_q_suits / (sum_qs + eps)  # shape (4,)

        # g) Compute global suit‐entropy: H_suit_global = -∑_s p_s log(p_s + eps)
        H_suit_global = -torch.sum(p_s * torch.log(p_s + eps))  # scalar

        self._last_suit_entropy_loss = H_suit_global.detach().cpu().item()

        # 8) Return the sum of both global entropies
        return H_rank_global  # + H_suit_global

    def suit_aux_loss(self, policy_loss, loss_inputs, aux_logits):
        original_obs = restore_original_dimensions(
            loss_inputs["obs"].cpu(), self.obs_space, "torch"
        )
        hand_suits = original_obs["hand"]["suit"].to(self.action_logits.device) - 1
        hand_suits = hand_suits.long()
        # Count of each suit in the hand
        suit_counts = torch.zeros(
            (hand_suits.shape[0], 4), device=hand_suits.device, dtype=torch.float32
        )
        for suit in range(4):
            suit_counts[:, suit] = (hand_suits == suit).sum(dim=1)

        # Normalize the counts to get probabilities
        num_suits = suit_counts.sum(dim=1, keepdim=True)
        # Avoid division by zero
        num_suits = torch.clamp(num_suits, min=1.0)
        suit_probs = suit_counts.float() / num_suits.float()

        pred_probs = torch.sigmoid(aux_logits)
        suit_count_loss = F.mse_loss(pred_probs, suit_probs, reduction="mean")

        # suit_count_loss = F.binary_cross_entropy_with_logits(
        #     aux_logits, suit_probs, reduction="mean"
        # )
        self._last_suit_loss = suit_count_loss.detach().cpu().item()

        return suit_count_loss

    def suit_matching_aux_loss(self, policy_loss, loss_inputs, aux_logits):
        original_obs = restore_original_dimensions(
            loss_inputs["obs"].cpu(), self.obs_space, "torch"
        )
        hand_suits = original_obs["hand"]["suit"].to(self.action_logits.device) - 1
        s1 = hand_suits.unsqueeze(1)  # (B, 1, 8)
        s2 = hand_suits.unsqueeze(2)  # (B, 8, 1)
        suit_matching_mask = (s1 == s2).float()

        H = hand_suits.shape[1]  # Hand size, e.g., 8
        tri_mask = torch.triu(torch.ones(H, H, device=hand_suits.device), diagonal=1)
        target = suit_matching_mask[:, tri_mask.bool()].float()  # (B, unordered_pairs)

        num_positive = target.sum()
        num_negative = target.numel() - num_positive
        num_positive = torch.clamp(num_positive, min=1.0)  # Avoid division by zero
        pos_weight = num_negative / num_positive

        suit_matching_loss = F.binary_cross_entropy_with_logits(
            aux_logits, target, reduction="mean", pos_weight=pos_weight
        )
        self._last_suit_matching_loss = suit_matching_loss.detach().cpu().item()
        return suit_matching_loss

    def rank_matching_aux_loss(self, policy_loss, loss_inputs, aux_logits):
        original_obs = restore_original_dimensions(
            loss_inputs["obs"].cpu(), self.obs_space, "torch"
        )
        # hand_indices = original_obs["hand_indices"].to(
        #     self.action_logits.device
        # )  # (B, 8)
        # hand_ranks = hand_indices % 13  # (B, 8) -> rank indices in [0, 3]
        hand_ranks = (
            original_obs["hand"]["rank"].to(self.action_logits.device) - 1
        )  # Go back to 0 index since there are no null ranks being considered
        s1 = hand_ranks.unsqueeze(1)  # (B, 1, 8)
        s2 = hand_ranks.unsqueeze(2)  # (B, 8, 1)
        rank_matching_mask = (s1 == s2).float()

        H = hand_ranks.shape[1]  # Hand size, e.g., 8
        tri_mask = torch.triu(torch.ones(H, H, device=hand_ranks.device), diagonal=1)
        target = rank_matching_mask[:, tri_mask.bool()].float()  # (B, unordered_pairs)

        num_positive = target.sum()
        num_negative = target.numel() - num_positive
        pos_weight = num_negative / (num_positive + 1e-8)
        # pos_weight = torch.clamp(pos_weight, min=1.0, max=10.0)

        rank_matching_loss = F.binary_cross_entropy_with_logits(
            aux_logits, target, reduction="mean", pos_weight=pos_weight
        )
        self._last_rank_matching_loss = rank_matching_loss.detach().cpu().item()
        return rank_matching_loss

    def valid_card_count_aux_loss(self, policy_loss, loss_inputs):
        # Penalize the model for selecting too many or too few cards
        play_logits = self.action_logits[:, : self.hand_size]
        discard_logits = self.action_logits[:, self.hand_size : self.hand_size * 2]

        play_probs = torch.sigmoid(play_logits)
        discard_probs = torch.sigmoid(discard_logits)

        # Penalize E[play_count] over 5 or under 1 and the same for discards
        play_count = play_probs.sum(dim=1)  # shape (B,)
        discard_count = discard_probs.sum(dim=1)  # shape (B,)
        play_loss = F.relu(play_count - 5.5).mean() + F.relu(0.5 - play_count).mean()
        discard_loss = (
            F.relu(discard_count - 5.5).mean() + F.relu(0.5 - discard_count).mean()
        )
        valid_card_count_loss = play_loss + discard_loss
        self._last_valid_card_count_loss = valid_card_count_loss.detach().cpu().item()
        return valid_card_count_loss

    def weight_decay_loss(self):
        weight_decay_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                weight_decay_loss += param.pow(2).sum()
        weight_decay_loss *= self.weight_decay_coeff
        self._last_weight_decay_loss = weight_decay_loss.detach().cpu().item()
        return weight_decay_loss

    def option_variation_loss(self):
        if self.action_method != "intent_vectors":
            return 0.0
        # gating_probs:    [B, K]  (softmax over experts)
        dist = torch.distributions.Categorical(logits=self.option_logits)
        H_cond = dist.entropy().mean()

        # marginal: you still have to do it manually,
        # but you can clamp that one:
        eps = 1e-8
        marginal = torch.softmax(self.option_logits, dim=-1).mean(dim=0).clamp(min=eps)
        H_marg = -(marginal * marginal.log()).sum()
        info_bonus = H_marg - H_cond
        info_loss = -info_bonus  # we want to minimize loss, so we negate it
        # info_loss = info_bonus

        self._last_option_variation_loss = info_loss.detach().cpu().item()

        return info_loss

    def intent_cosine_loss(self, play_intents, discard_intents):
        """
        play_intents: Tensor of shape (B, E, D)
        discard_intents: Tensor of shape (B, E, D)
        Returns: scalar loss = mean_pairwise_cos(play) + mean_pairwise_cos(discard)
        """

        def pairwise_cos_loss(intents):
            # 1) normalize each vector to unit length
            normed = F.normalize(intents, dim=-1)  # (B, E, D)

            # 2) compute pairwise cosine sims per batch
            sims = torch.einsum("bnd,bmd->bnm", normed, normed)  # (B, E, E)

            # 3) average over batch
            avg_sims = sims.mean(dim=0)  # (E, E)

            # 4) zero out the diagonal
            E = avg_sims.size(0)
            off_diag = avg_sims.flatten()[~torch.eye(E, dtype=bool).flatten()]

            # 5) return mean of squared cosines
            return torch.mean(off_diag**2)

        play_loss = pairwise_cos_loss(play_intents)
        discard_loss = pairwise_cos_loss(discard_intents)

        # total loss; positive when experts are similar, and minimizing
        # drives their mean cosine toward -1 (maximally dissimilar)
        return play_loss + discard_loss

    def intent_similarity_loss(self):
        if self.action_method != "intent_vectors" or self.num_experts <= 1:
            return 0.0

        # Compute the cosine similarity between play and discard intents of the pairs of experts
        play_intents = self.play_intents
        discard_intents = self.discard_intents

        intent_similarity_loss = self.intent_cosine_loss(play_intents, discard_intents)
        self._last_intent_similarity_loss = intent_similarity_loss.detach().cpu().item()
        return intent_similarity_loss

    def available_hand_types_aux_loss(
        self, policy_loss, loss_inputs, available_hand_types_logits
    ):
        d = available_hand_types_logits.device
        original_obs = restore_original_dimensions(
            loss_inputs["obs"], self.obs_space, "torch"
        )
        target_hand_types = original_obs["available_hand_types"].to(d).float()
        pos_counts = target_hand_types.sum(dim=0)
        neg_counts = target_hand_types.shape[0] - pos_counts
        pos_weight = neg_counts.clamp_min(1.0) / pos_counts.clamp_min(1.0)
        pos_weight = pos_weight.to(d)

        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # available_hand_types_loss = criterion(
        #     available_hand_types_logits,
        #     target_hand_types,
        # )
        available_hand_types_loss = F.binary_cross_entropy_with_logits(
            available_hand_types_logits,
            target_hand_types,
            pos_weight=pos_weight,
            reduction="mean",
        )
        self._last_available_hand_types_loss = (
            available_hand_types_loss.detach().cpu().item()
        )
        return available_hand_types_loss

    def hand_score_loss(self, loss_inputs):
        aux_preds = self.play_aux
        num_aux = aux_preds.size(1)
        if self.action_method == "subset_attention":
            actions = loss_inputs[SampleBatch.ACTIONS]
            is_play = actions < num_aux
        elif self.action_method == "dual_subset":
            actions = loss_inputs[SampleBatch.ACTIONS][:, 0]
            is_play = actions > 0
        true_scores = loss_inputs["joker_scores"]
        is_play_w_joker = is_play & (self._has_jokers)
        # Print the count of non-minimum entries in each column of true_scores
        min_vals = true_scores.min(dim=0, keepdim=True)[0]
        non_min_mask = true_scores > min_vals + 1e-6  # Avoid numerical issues
        non_min_counts = non_min_mask.sum(dim=0)
        nans = torch.isnan(true_scores[is_play_w_joker])
        nans_count = nans.sum(dim=0)
        # print("NaN counts per column in true_scores:", nans_count.tolist())

        is_play_w_joker = is_play_w_joker & (~torch.isnan(true_scores).any(dim=1))
        if (actions >= aux_preds.size(1)).any():
            print("Warning: Some actions are out of bounds!")
        if torch.isnan(aux_preds).any().item():
            print("Warning: aux_preds contains NaNs!")
        if is_play_w_joker.any():
            valid_actions = actions[is_play_w_joker]  # [N]
            valid_true_scores = true_scores[is_play_w_joker]  # [N, 4]
            valid_aux_preds = aux_preds[is_play_w_joker]  # [N, 1586, 4]

            # gather the predicted scores at the chosen action index for each valid row
            pred_scores = valid_aux_preds.gather(
                1,
                valid_actions.view(-1, 1, 1).expand(
                    -1, 1, aux_preds.size(2)
                ),  # [N, 1, 4]
            ).squeeze(1)
            aux_loss = F.mse_loss(pred_scores, valid_true_scores)
        else:
            aux_loss = torch.tensor(0.0, device=aux_preds.device)

        self._last_hand_score_loss = aux_loss.detach().cpu().item()
        return aux_loss

    def joker_spread_loss(self, loss_inputs):
        joker_embeddings = self.universal_card_encoder.general_index_embedding(
            torch.arange(
                BaseCard.FIRST_JOKER_INDEX,
                BaseCard.FIRST_JOKER_INDEX + 150,
                device=self.action_logits.device,
            )
        )
        diffs = joker_embeddings.unsqueeze(0) - joker_embeddings.unsqueeze(1)
        dists = torch.norm(diffs, dim=-1)
        mask = ~torch.eye(dists.size(0), dtype=torch.bool, device=dists.device)
        loss = torch.exp(-0.5 * dists[mask]).mean()
        self._last_joker_spread_loss = loss.detach().cpu().item()

        pca = PCA()
        joker_np = joker_embeddings.detach().cpu().numpy()
        pca.fit(joker_np)
        explained = pca.explained_variance_ratio_
        self._last_joker_pca_evs = explained.tolist()

        return loss

    def custom_loss(self, policy_loss, loss_inputs):
        if self.aux_outputs > 0:
            aux_logits = self._last_aux_outputs
            if self.suit_count_aux_coeff:
                suit_logits = aux_logits[:, :4]
                aux_logits = aux_logits[:, 4:]
                suit_loss = self.suit_aux_loss(policy_loss, loss_inputs, suit_logits)
                policy_loss = [x + suit_loss for x in policy_loss]

            if self.joker_identity_coeff > 0:
                joker_logits = aux_logits[:, : self.max_jokers * self.joker_types]
                aux_logits = aux_logits[:, self.max_jokers * self.joker_types :]
                joker_loss = self.joker_aux_loss(policy_loss, loss_inputs, joker_logits)
                policy_loss = [
                    x + joker_loss * self.joker_identity_coeff for x in policy_loss
                ]

            if self.suit_matching_aux_coeff > 0:
                unordered_pairs = self.hand_size * (self.hand_size - 1) // 2
                suit_matching_logits = aux_logits[:, :unordered_pairs]
                aux_logits = aux_logits[:, unordered_pairs:]
                suit_matching_loss = self.suit_matching_aux_loss(
                    policy_loss, loss_inputs, suit_matching_logits
                )
                policy_loss = [
                    x + suit_matching_loss * self.suit_matching_aux_coeff
                    for x in policy_loss
                ]

            if self.rank_matching_aux_coeff > 0:
                unordered_pairs = self.hand_size * (self.hand_size - 1) // 2
                rank_matching_logits = aux_logits[:, :unordered_pairs]
                aux_logits = aux_logits[:, unordered_pairs:]
                rank_matching_loss = self.rank_matching_aux_loss(
                    policy_loss, loss_inputs, rank_matching_logits
                )
                policy_loss = [
                    x + rank_matching_loss * self.rank_matching_aux_coeff
                    for x in policy_loss
                ]

            if self.available_hand_types_coeff > 0:
                available_hand_types_logits = aux_logits[:, :8]
                aux_logits = aux_logits[:, 8:]
                available_hand_types_loss = self.available_hand_types_aux_loss(
                    policy_loss, loss_inputs, available_hand_types_logits
                )
                policy_loss = [
                    x + available_hand_types_loss * self.available_hand_types_coeff
                    for x in policy_loss
                ]

        if self.suit_rank_entropy_coeff > 0:
            entropy_loss = self.suit_rank_entropy_loss(loss_inputs)
            policy_loss = [
                x - entropy_loss * self.suit_rank_entropy_coeff for x in policy_loss
            ]

        if self.valid_card_count_coeff > 0:
            valid_card_count_loss = self.valid_card_count_aux_loss(
                policy_loss, loss_inputs
            )
            policy_loss = [
                x + valid_card_count_loss * self.valid_card_count_coeff
                for x in policy_loss
            ]

        if self.num_experts > 1:
            option_variation_loss = self.option_variation_loss()
            policy_loss = [
                x + option_variation_loss * self.option_variation_coeff
                for x in policy_loss
            ]

        if self.intent_similarity_coeff > 0:
            intent_similarity_loss = self.intent_similarity_loss()
            policy_loss = [
                x + intent_similarity_loss * self.intent_similarity_coeff
                for x in policy_loss
            ]

        if self.weight_decay_coeff > 0:
            weight_decay_loss = self.weight_decay_loss()
            policy_loss = [x + weight_decay_loss for x in policy_loss]

        if self.hand_score_aux_coeff > 0:
            assert self.action_method in ["subset_attention", "dual_subset"]
            hand_score_loss = self.hand_score_loss(loss_inputs)
            policy_loss = [
                x + hand_score_loss * self.hand_score_aux_coeff for x in policy_loss
            ]

        if self.joker_spread_loss_coeff > 0:
            joker_spread_loss = self.joker_spread_loss(loss_inputs)
            policy_loss = [
                x + joker_spread_loss * self.joker_spread_loss_coeff
                for x in policy_loss
            ]

        self.custom_losses_have_been_calced = True
        return policy_loss

    def metrics(self):
        m = {}

        if self._last_suit_loss is not None:
            m["suit_aux_loss"] = self._last_suit_loss
        if self._last_joker_aux_loss is not None:
            m["joker_aux_loss"] = self._last_joker_aux_loss
        if self._last_suit_entropy_loss is not None:
            m["suit_entropy_loss"] = self._last_suit_entropy_loss
            m["rank_entropy_loss"] = self._last_rank_entropy_loss
        if self._last_suit_matching_loss is not None:
            m["suit_matching_loss"] = self._last_suit_matching_loss
        if self._last_rank_matching_loss is not None:
            m["rank_matching_loss"] = self._last_rank_matching_loss
        if self._last_valid_card_count_loss is not None:
            m["valid_card_count_loss"] = self._last_valid_card_count_loss
        if self._last_weight_decay_loss is not None:
            m["weight_decay_loss"] = self._last_weight_decay_loss
        if self._last_available_hand_types_loss is not None:
            m["available_hand_types_loss"] = self._last_available_hand_types_loss
        if self._last_option_variation_loss is not None:
            m["option_variation_loss"] = self._last_option_variation_loss
        if self._last_intent_similarity_loss is not None:
            m["intent_similarity_loss"] = self._last_intent_similarity_loss
        if self._last_hand_score_loss is not None:
            m["hand_score_loss"] = self._last_hand_score_loss
        if self._last_joker_spread_loss is not None:
            m["joker_spread_loss"] = self._last_joker_spread_loss
        if len(self._last_joker_pca_evs) > 0:
            for i in range(len(self._last_joker_pca_evs)):
                m[f"joker_pca_ev_top_{i}"] = sum(self._last_joker_pca_evs[: i + 1])
        if self._last_curiosity_bonus is not None:
            m["blind_curiosity_bonus"] = self._last_curiosity_bonus
        # for i, param in enumerate(self.parameters()):
        #     if param.requires_grad:
        #         m[f"param_{i}_{str(param.shape)}_norm"] = param.norm().item()

        return m
