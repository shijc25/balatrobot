from collections import Counter
from typing import Dict, Tuple
import argparse
import gymnasium as gym
import numpy as np
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.algorithm import Algorithm
from functools import partial
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    # EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from copy import deepcopy
from ray.air.integrations.wandb import setup_wandb
import wandb
import torch
from gym_envs.joker_effects import supported_jokers
from ray.rllib.algorithms.ppo import PPO
from gym_envs.joker import Joker
from ray.rllib.env import MultiAgentEnv
from gym_envs.pseudo.card import Card
from gym_envs.blind import Blind
from gym_envs.base_card import BaseCard
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch


def _update_bias(worker, bias):
    worker.foreach_env(lambda env: env.blind_env.set_bias(bias))


def _update_rarities(worker, rarities):
    worker.foreach_env(lambda env: env.blind_env.set_rarities(rarities))


def get_biases(worker):
    return worker.foreach_env(lambda env: env.blind_env.get_bias())


def get_and_reset_stats(worker):
    return worker.foreach_env(lambda env: env.blind_env.get_and_reset_stats())


class RoundLoggerCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.max_starting_jokers = 0
        self.shop_enabled = False
        self.curriculum = [
            "0_jokers",
            "1_joker",
            "deck_variation",
            "2_jokers",
            "shop_enabled",
        ]
        self.curriculum_index = 0
        self.shop_joker_counts = Counter()

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        # row = {}
        blind_env = base_env.get_sub_environments()[env_index].blind_env
        shop_env = base_env.get_sub_environments()[env_index].shop_env
        # blind_env = base_env.get_sub_environments()[env_index]
        # shop_env = None
        episode.custom_metrics["round"] = blind_env.round
        for i in range(1, 30):
            if blind_env.round > i:
                episode.custom_metrics[f"round_{i}_won"] = 1.0
            else:
                episode.custom_metrics[f"round_{i}_won"] = 0.0

        for blind_name in Blind.all_blind_names():
            if blind_env.G.current_blind.name == blind_name:
                episode.custom_metrics[f"blind_{blind_name}_death_rate"] = 1.0
            else:
                episode.custom_metrics[f"blind_{blind_name}_death_rate"] = 0.0

        hand_counts = blind_env.hand_counts
        hand_counts_flat = []
        for i, hand_type in enumerate(sorted(hand_counts.keys())):
            if hand_counts[hand_type] > 0:
                hand_counts_flat += [i] * hand_counts[hand_type]

        # episode.hist_data["hands_counts"] = hand_counts_flat
        # row["hands_counts"] = hand_counts_flat

        count_counts = blind_env.count_counts
        count_counts_flat = []
        for i, count_type in enumerate(sorted(count_counts.keys())):
            if count_counts[count_type] > 0:
                count_counts_flat += [i] * count_counts[count_type]
        episode.custom_metrics["card_count_mean"] = np.mean(count_counts_flat)

        discard_count_counts = blind_env.discard_count_counts
        discard_count_counts_flat = []
        for i, count_type in enumerate(sorted(discard_count_counts.keys())):
            if discard_count_counts[count_type] > 0:
                discard_count_counts_flat += [i] * discard_count_counts[count_type]
        episode.custom_metrics["discard_count_mean"] = np.mean(
            discard_count_counts_flat
        )

        # episode.hist_data["count_counts"] = count_counts_flat
        # row["count_counts"] = count_counts_flat

        card_slot_counts = blind_env.card_slot_counts
        card_slot_counts_flat = []
        for i, card_slot_hits in enumerate(sorted(card_slot_counts.keys())):
            # if card_slot_counts[card_slot_hits] > 0:
            #     card_slot_counts_flat += [i] * card_slot_counts[card_slot_hits]
            episode.custom_metrics[f"card_slot_{i+1}_count"] = card_slot_counts[
                card_slot_hits
            ]

        # episode.user_data.setdefault("card_slot_counts", np.zeros(8, dtype=np.float32)) += np.array(
        #     card_slot_counts_flat, dtype=np.float32
        # )

        # episode.hist_data["card_slot_counts"] = card_slot_counts_flat
        # row["card_slot_counts"] = card_slot_counts_flat

        total_hands = sum(hand_counts.values())
        for hand_type in hand_counts:
            episode.custom_metrics[f"hand_{hand_type}"] = (
                hand_counts[hand_type] / total_hands
            )
            # row[f"hand_{hand_type}"] = hand_counts[hand_type]

        if shop_env is not None:
            for x in shop_env.telemetry:
                episode.custom_metrics[x] = shop_env.telemetry[x]
            episode.custom_metrics["terminal_dollars"] = shop_env.G.dollars

        rank_counts = {r: 0 for r in Card.RANKS}
        # for rank in Card.RANKS:
        #     count = 0
        for card in blind_env.G.deck.all_cards:
            rank_counts[card.value] += 1
        for rank, count in rank_counts.items():
            episode.custom_metrics[f"deck_ranks/{rank}"] = count

        suit_counts = {suit: 0 for suit in Card.SUITS}
        # for suit in Card.SUITS:
        #     count = 0
        for card in blind_env.G.deck.all_cards:
            suit_counts[card.suit] += 1
            # count += 1
        for suit, count in suit_counts.items():
            episode.custom_metrics[f"deck_suits/{suit}"] = count

        enh_counts = {
            enh: 0
            for enh in BaseCard.Enhancements.ALL.values()
            if enh != BaseCard.Enhancements.NORMAL
        }
        # for enhancement, id in BaseCard.Enhancements.ALL.items():
        #     if id == BaseCard.Enhancements.NORMAL:
        #         continue
        #     count = 0
        for card in blind_env.G.deck.all_cards:
            if card.enhancement in enh_counts:
                enh_counts[card.enhancement] += 1

        for enhancement, count in enh_counts.items():
            episode.custom_metrics[f"deck_enhancements/{enhancement}"] = count

        edition_counts = {
            edition: 0
            for edition in BaseCard.Editions.ALL.values()
            if edition != BaseCard.Editions.NO_EDITION
        }
        for card in blind_env.G.deck.all_cards:
            if card.edition in edition_counts:
                edition_counts[card.edition] += 1
        for edition, count in edition_counts.items():
            episode.custom_metrics[f"deck_editions/{edition}"] = count
        # for edition, id in BaseCard.Editions.ALL.items():
        #     if id == BaseCard.Editions.NO_EDITION:
        #         continue
        #     count = 0
        #     for card in blind_env.G.deck.all_cards:
        #         if card.edition == id:
        #             count += 1
        #     episode.custom_metrics[f"deck_editions/{edition}"] = count

        seal_counts = {
            seal: 0
            for seal in BaseCard.Seals.ALL.values()
            if seal != BaseCard.Seals.NO_SEAL
        }
        for card in blind_env.G.deck.all_cards:
            if card.seal in seal_counts:
                seal_counts[card.seal] += 1
        for seal, count in seal_counts.items():
            episode.custom_metrics[f"deck_seals/{seal}"] = count
        # for seal, id in BaseCard.Seals.ALL.items():
        #     if id == BaseCard.Seals.NO_SEAL:
        #         continue
        #     count = 0
        #     for card in blind_env.G.deck.all_cards:
        #         if card.seal == id:
        #             count += 1
        #     episode.custom_metrics[f"deck_seals/{seal}"] = count

        episode.custom_metrics["deck_size"] = len(blind_env.G.deck.all_cards)
        if blind_env.hands_played_in_round_1 > 0:
            episode.custom_metrics["hands_played_in_round_1"] = (
                blind_env.hands_played_in_round_1
            )

        # calculate a general metric that indicates how much the agent has homeogenized the deck by ranks and suits (separately)
        rank_count_array = np.zeros(13, dtype=np.float32)
        suit_count_array = np.zeros(4, dtype=np.float32)
        for card in blind_env.G.deck.all_cards:
            rank_count_array[card.value - 2] += 1
            suit_count_array[Card.SUITS.index(card.suit)] += 1
        rank_entropy = -np.sum(
            (rank_count_array / np.sum(rank_count_array))
            * np.log(rank_count_array / np.sum(rank_count_array) + 1e-10)
        )
        suit_entropy = -np.sum(
            (suit_count_array / np.sum(suit_count_array))
            * np.log(suit_count_array / np.sum(suit_count_array) + 1e-10)
        )

        episode.custom_metrics["deck_rank_entropy"] = rank_entropy
        episode.custom_metrics["deck_suit_entropy"] = suit_entropy

        episode.custom_metrics["hands_played"] = blind_env.hands_played
        # row["hands_played"] = blind_env.hands_played
        episode.custom_metrics["discards_played"] = blind_env.discards_played
        # row["discards_played"] = blind_env.discards_played

        episode.custom_metrics["chips"] = blind_env.chips
        # row["chips"] = blind_env.chips

        episode.custom_metrics[f"{len(blind_env.G.owned_jokers)}_jokers_chips"] = (
            blind_env.chips
        )
        episode.custom_metrics[f"{len(blind_env.G.owned_jokers)}_jokers_round"] = (
            blind_env.round
        )

        # Makes the null ratio 1.0, and the best ratio 2.0
        def normalized_ratio(correct, total, null_ratio):
            best_ratio = 1.0 / null_ratio
            if total > 0:
                observed_ratio = correct / total
                ratio_score = observed_ratio / null_ratio
                if ratio_score > 1:
                    ratio_score = 1 + (ratio_score - 1) / (best_ratio - 1)
                return ratio_score
            else:
                return 0.0

        episode.custom_metrics["joker_count"] = len(blind_env.G.owned_jokers)
        joker_names = list(sorted([x.name for x in blind_env.G.owned_jokers]))
        # if len(blind_env.G.owned_jokers) > 0:
        #     episode.custom_metrics[f"{'_'.join(joker_names)}_chips"] = blind_env.chips
        # for rank in range(13):
        #     episode.custom_metrics[f"{'_'.join(joker_names)}_rank_{rank+1}"] = sum(
        #         blind_env.scored_ranks[k][rank] for k in blind_env.scored_ranks
        #     )
        # suit_map = {
        #     0: "Clubs",
        #     1: "Diamonds",
        #     2: "Hearts",
        #     3: "Spades",
        # }
        # for suit in range(4):
        #     episode.custom_metrics[
        #         f"{'_'.join(joker_names)}_suit_{suit_map[suit]}"
        #     ] = sum(blind_env.scored_suits[k][suit] for k in blind_env.scored_ranks)

        #     suit_jokers = {
        #         "Lusty Joker": 2,
        #         "Greedy Joker": 1,
        #         "Wrathful Joker": 3,
        #         "Gluttonous Joker": 0,
        #     }
        #     for joker in suit_jokers:
        #         if joker in joker_names:
        #             suit = suit_jokers[joker]
        #             cards_on_suit = sum(
        #                 blind_env.scored_suits[k][suit] for k in blind_env.scored_suits
        #             )
        #             cards_total = sum(
        #                 sum(blind_env.scored_suits[k]) for k in blind_env.scored_suits
        #             )
        #             null_ratio = 1 / 4

        #             if cards_total > 0:
        #                 episode.custom_metrics["{}_suit_match_rate".format(joker)] = (
        #                     normalized_ratio(cards_on_suit, cards_total, null_ratio)
        #                 )

        #     rank_jokers = {
        #         "Even Steven": [2, 4, 6, 8, 10],
        #         "Odd Todd": [3, 5, 7, 9, 14],
        #         "Scholar": [14],
        #         "Walkie Talkie": [10, 4],
        #         "Smiley Face": [11, 12, 13],
        #         "Scary Face": [11, 12, 13],
        #         "Fibonacci": [2, 3, 5, 8, 14],
        #     }

        #     for joker in rank_jokers:
        #         if joker in joker_names:
        #             ranks = rank_jokers[joker]
        #             cards_on_rank = sum(
        #                 blind_env.scored_ranks[k][rank - 2]  # Adjust for 0-indexing
        #                 for k in blind_env.scored_ranks
        #                 for rank in ranks
        #             )
        #             cards_total = sum(
        #                 sum(blind_env.scored_ranks[k]) for k in blind_env.scored_ranks
        #             )

        #             null_ratio = len(ranks) / 13.0
        #             if cards_total > 0:
        #                 episode.custom_metrics["{}_rank_match_rate".format(joker)] = (
        #                     normalized_ratio(cards_on_rank, cards_total, null_ratio)
        #                 )

        # hand_type_jokers = {
        #     "Jolly Joker": ["Pair", "Two Pair", "Three of a Kind", "Full House"],
        #     "Zany Joker": ["Three of a Kind", "Full House", "Four of a Kind"],
        #     "Mad Joker": ["Two Pair", "Full House"],
        #     "Crazy Joker": ["Straight", "Straight Flush"],
        #     "Droll Joker": ["Flush", "Straight Flush"],
        #     "Sly Joker": [
        #         "Pair",
        #         "Two Pair",
        #         "Three of a Kind",
        #         "Full House",
        #         "Four of a Kind",
        #     ],
        #     "The Duo": [
        #         "Pair",
        #         "Two Pair",
        #         "Three of a Kind",
        #         "Full House",
        #         "Four of a Kind",
        #     ],
        #     "Wily Joker": ["Three of a Kind", "Full House", "Four of a Kind"],
        #     "The Trio": ["Three of a Kind", "Full House", "Four of a Kind"],
        #     "The Family": ["Four of a Kind"],
        #     "Clever Joker": ["Two Pair"],
        #     "Devious Joker": ["Straight", "Straight Flush"],
        #     "The Order": ["Straight", "Straight Flush"],
        #     "Crafty Joker": ["Flush", "Straight Flush"],
        #     "The Tribe": ["Flush", "Straight Flush"],
        #     "Four Fingers": ["Flush", "Straight Flush", "Straight"],
        # }

        # for joker in hand_type_jokers:
        #     if joker in joker_names:
        #         hand_types = hand_type_jokers[joker]
        #         hands_match_joker = sum(
        #             blind_env.G.hand_stats[hand_type].play_count
        #             for hand_type in blind_env.G.hand_stats
        #             if hand_type in hand_types
        #         )
        #         hands_total = sum(
        #             blind_env.G.hand_stats[hand_type].play_count
        #             for hand_type in blind_env.G.hand_stats
        #         )

        #         null_ratio = len(hand_types) / len(blind_env.G.hand_stats)
        #         if hands_total > 0:
        #             episode.custom_metrics["{}_hand_match_rate".format(joker)] = (
        #                 normalized_ratio(hands_match_joker, hands_total, null_ratio)
        #             )

        for joker in Joker.all_supported_joker_names():
            num_of_joker = joker_names.count(joker)
            episode.custom_metrics[f"{joker}_owned"] = num_of_joker

        # for joker in blind_env.G.owned_jokers:
        #     episode.custom_metrics[f"joker_rounds/{joker.name}"] = blind_env.round

        episode.custom_metrics["unique_experts"] = len(set(blind_env.expert_history))

        for hand_type in blind_env.G.hand_stats:
            episode.custom_metrics[f"hand_level_{hand_type}"] = blind_env.G.hand_stats[
                hand_type
            ].get_level()

        # Duplicate all stats on a per-expert basis
        # active_expert = blind_env.active_expert
        # if active_expert is not None:
        #     episode.hist_data.setdefault("active_expert", []).append(active_expert)
        #     for metric in list(episode.custom_metrics.keys()):
        #         if "hands_played" in metric or "hand_" in metric or "round" in metric:
        #             episode.custom_metrics[f"expert_{active_expert}/{metric}"] = (
        #                 episode.custom_metrics[metric]
        #             )

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        hands = [
            "High Card",
            "Pair",
            "Two Pair",
            "Three of a Kind",
            "Straight",
            "Flush",
            "Full House",
            "Four of a Kind",
            "Straight Flush",
        ]
        # 0    "Flush",
        # 1    "Four of a Kind",
        # 2    "Full House",
        # 3    "High Card",
        # 4    "Pair",
        # 5    "Straight",
        # 6    "Straight Flush",
        # 7    "Three of a Kind",
        # 8    "Two Pair",

        # confusion, target_counts, hit_rates, scored_ranks, scored_suits
        stats = algorithm.workers.foreach_worker(func=get_and_reset_stats)
        stats = [item for sublist in stats for item in sublist]
        # confusion = np.sum([stat[0] for stat in stats], axis=0)
        target_counts = {hand: 0 for hand in hands}
        hit_rates = {hand: 0 for hand in hands}
        scored_ranks = {hand: np.zeros(13, dtype=np.float32) for hand in hands}
        scored_suits = {hand: np.zeros(4, dtype=np.float32) for hand in hands}
        for stat in stats:
            for hand in hands:
                target_counts[hand] += stat["target_counts"][hand]
                hit_rates[hand] += stat["hit_rates"][hand]
                scored_ranks[hand] += stat["scored_ranks"][hand]
                scored_suits[hand] += stat["scored_suits"][hand]
        hands_on_target = {
            hand: (
                hit_rates[hand] / target_counts[hand]
                if target_counts[hand] > 0
                else 0.0
            )
            for hand in hands
        }

        scored_ranks_flat = np.zeros(13, dtype=np.float32)
        scored_suits_flat = np.zeros(4, dtype=np.float32)
        for hand in hands:
            scored_ranks_flat += scored_ranks[hand]
            scored_suits_flat += scored_suits[hand]

        result[ENV_RUNNER_RESULTS]["custom_metrics"]["rank_entropy"] = -np.sum(
            (scored_ranks_flat / np.sum(scored_ranks_flat))
            * np.log(scored_ranks_flat / np.sum(scored_ranks_flat) + 1e-10)
        )
        result[ENV_RUNNER_RESULTS]["custom_metrics"]["suit_entropy"] = -np.sum(
            (scored_suits_flat / np.sum(scored_suits_flat))
            * np.log(scored_suits_flat / np.sum(scored_suits_flat) + 1e-10)
        )

        total_hands = sum(target_counts.values())
        result[ENV_RUNNER_RESULTS]["custom_metrics"]["high_low_rank_spread"] = (
            scored_ranks_flat.max() - scored_ranks_flat.min()
        ) / total_hands
        result[ENV_RUNNER_RESULTS]["custom_metrics"]["high_low_suit_spread"] = (
            scored_suits_flat.max() - scored_suits_flat.min()
        ) / total_hands

        # rank_hist = []

        # biases = algorithm.workers.foreach_worker(func=get_biases)
        # # Flatten the list of lists of biases
        # biases = [item for sublist in biases for item in sublist]
        # biases = biases[0]
        # new_biases = deepcopy(biases)
        # new_biases = {
        #     hand: (bias * 0.99 if bias > 0.01 else 0.0)
        #     for hand, bias in new_biases.items()
        # }

        # flushes = result[ENV_RUNNER_RESULTS]["custom_metrics"]["hand_Flush_mean"]
        # mean_return = result[ENV_RUNNER_RESULTS]["episode_return_mean"]

        # for hand in hands:
        #     target_hit_rate = 0.2
        #     actual_hit_rate = hands_on_target[hand]
        #     # new_biases[hand] -= 0.005
        #     if actual_hit_rate < target_hit_rate:
        #         new_biases[hand] *= 0.99
        #         pass
        #     elif actual_hit_rate > target_hit_rate:
        #         # new_biases[hand] -= 0.05
        #         new_biases[hand] *= 0.95
        #     if new_biases[hand] < 0.01:
        #         new_biases[hand] = 0.00
        #     new_biases[hand] = np.clip(new_biases[hand], 0.0, 0.99)
        #     result[ENV_RUNNER_RESULTS]["custom_metrics"][f"bias_{hand}"] = new_biases[
        #         hand
        #     ]
        #     result[ENV_RUNNER_RESULTS]["custom_metrics"][f"hit_rate_{hand}"] = (
        #         hands_on_target[hand]
        #     )
        # average_bias = np.mean(list(new_biases.values()))
        # result[ENV_RUNNER_RESULTS]["custom_metrics"]["average_bias"] = average_bias

        hand_means = {}
        for hand_type in hands:
            hand_means[hand_type] = result[ENV_RUNNER_RESULTS]["custom_metrics"][
                f"hand_{hand_type}_mean"
            ]
        denom = max(hand_means.values())
        hand_rarities = {k: (denom - v) / denom for k, v in hand_means.items()}

        keys_to_remove = ["sampler_results", "env_runner_results"]
        for key in keys_to_remove:
            if key in result:
                del result[key]
        keys_to_remove = []
        for key in result["custom_metrics"]:
            # Remove autogenerated min and max keys
            if key.endswith("_min") or key.endswith("_max"):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del result["custom_metrics"][key]

        if "evaluation" in result:
            eval_res = result["evaluation"]
            for key in ("sampler_results", "env_runner_results"):
                eval_res.pop(key, None)
            bad = [
                k
                for k in eval_res.get("custom_metrics", {})
                if k.endswith("_min") or k.endswith("_max")
            ]
            for k in bad:
                eval_res["custom_metrics"].pop(k, None)

        # new_bias = average_bias
        # if mean_return < target_return:
        #     new_bias = average_bias + 0.01
        # elif mean_return > target_return:
        #     new_bias = average_bias - 0.05

        # new_bias = np.clip(new_bias, 0.0, 0.9)

        # algorithm.workers.foreach_worker(func=partial(_update_bias, bias=new_biases))
        # algorithm.workers.foreach_worker(
        #     func=partial(_update_rarities, rarities=hand_rarities)
        # )

        def enable_shop_agent(env):
            env.shop_enabled = True
            env.blind_env.joker_count_range = (0, 0)

        def set_one_joker(env):
            env.blind_env.joker_count_range = (0, 2)

        # import gc

        # def list_top_gpu_tensors(n=10):
        #     # Gather (tensor, size_in_bytes) for every live CUDA tensor
        #     tensor_sizes = []
        #     for obj in gc.get_objects():
        #         try:
        #             if torch.is_tensor(obj) and obj.is_cuda:
        #                 size = obj.element_size() * obj.nelement()
        #                 tensor_sizes.append((obj, size))
        #             elif (
        #                 hasattr(obj, "data")
        #                 and torch.is_tensor(obj.data)
        #                 and obj.data.is_cuda
        #             ):
        #                 t = obj.data
        #                 size = t.element_size() * t.nelement()
        #                 tensor_sizes.append((t, size))
        #         except Exception:
        #             continue

        #     # Sort descending, take top n
        #     tensor_sizes.sort(key=lambda x: x[1], reverse=True)
        #     print(f"Top {n} live CUDA tensors:")
        #     for t, size in tensor_sizes[:n]:
        #         mb = size / 1e6
        #         print(f" • {tuple(t.shape)}  dtype={str(t.dtype)}   {mb} MB")
        #     print(
        #         len([x for x in tensor_sizes if x[1] > 0]), "live CUDA tensors in total"
        #     )

        # # Call it right after your curiosity code
        # list_top_gpu_tensors(10)

        # if result["custom_metrics"]["round_mean"] > 3.5 and self.curriculum_index == 0:
        #     # Enable the shop agent if the average chips with 0 jokers is above 400
        #     print("Enabling shop agent due to curriculum progression")
        #     algorithm.workers.foreach_worker(
        #         lambda worker: worker.foreach_env(lambda env: enable_shop_agent(env))
        #     )
        #     self.shop_enabled = True
        #     self.curriculum_index += 1

        # def scale_joker_presence(env, score):
        #     # scale the exponent from -10 to 0 based on the score
        #     env.joker_count_bias_exponent = min(-10 + (score / 60), 0)
        #     env.round_range = (1, 4 + score // 60)

        # algorithm.workers.foreach_worker(
        #     lambda worker: worker.foreach_env(
        #         lambda env: scale_joker_presence(
        #             env, result["custom_metrics"]["0_jokers_chips_mean"]
        #         )
        #     )
        # )

        # blind_ckpt = r"C:\Users\giewe\AppData\Roaming\Balatro\Mods\balatrobot\saved_models\PPO_PseudoBlindEnv_0bfd2_00000_0_2025-06-11_16-47-26\checkpoint_000054"
        # Load everything (weights, optimizer, timesteps) in one go:
        # if not getattr(algorithm, "_blind_injected", False):
        #     # load the old blind algo just to pull its state
        #     print("Injecting blind policy state from checkpoint:", blind_ckpt)
        #     old = PPO.from_checkpoint(blind_ckpt)
        #     print("Old blind policy state:", old.get_policy("blind_agent").get_state())
        #     state = old.get_policy().get_state()
        #     # print("Old blind policy state:", state)
        #     # set that state into the new multi-agent trainer
        #     print("Injecting blind policy state into new algorithm")
        #     algorithm.get_policy("blind_agent").set_state(state)

        #     # mark as done so we don’t do this again
        #     algorithm._blind_injected = True

        # learner_stats = result["info"]["learner"]["default_policy"]

        # policy_clip_frac = learner_stats.get("policy_clip_fraction")
        # vf_clip_frac = learner_stats.get("vf_clip_fraction")
        # print(policy_clip_frac, vf_clip_frac)

        # result["custom_metrics"]["policy_clip_fraction"] = policy_clip_frac
        # result["custom_metrics"]["vf_clip_fraction"] = vf_clip_frac
        # print(result.keys())
        # print(result["sampler_results"])
        # print(result["env_runner_results"])
        # print(result["info"])

        # jokers = result["hist_stats"].get("jokers", [])
        # chips = result["hist_stats"].get("chips", [])
        # hand_counts = result["hist_stats"].get("hand_counts", [])
        # scored_suits = result["hist_stats"].get("scored_suits", [])
        # scored_ranks = result["hist_stats"].get("scored_ranks", [])

        # overall_suit_counts = np.zeros(4, dtype=np.float32)
        # for ep in scored_suits:
        #     for hand_type in ep:
        #         overall_suit_counts += ep[hand_type]

        # overall_rank_counts = np.zeros(13, dtype=np.float32)
        # for ep in scored_ranks:
        #     for hand_type in ep:
        #         overall_rank_counts += ep[hand_type]

        # suit_entropy = -np.sum(
        #     (overall_suit_counts / np.sum(overall_suit_counts))
        #     * np.log(overall_suit_counts / np.sum(overall_suit_counts) + 1e-10)
        # )
        # rank_entropy = -np.sum(
        #     (overall_rank_counts / np.sum(overall_rank_counts))
        #     * np.log(overall_rank_counts / np.sum(overall_rank_counts) + 1e-10)
        # )
        # result[ENV_RUNNER_RESULTS]["custom_metrics"]["suit_entropy"] = suit_entropy
        # result[ENV_RUNNER_RESULTS]["custom_metrics"]["rank_entropy"] = rank_entropy

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kw,
    ):
        if "blind" in agent_id:
            policy = policies[policy_id]
            infos = postprocessed_batch[SampleBatch.INFOS]
            targets = {
                "joker_marginal": [],
                "joker_chips": [],
                "joker_mult": [],
                "joker_mult_mult": [],
            }
            # Use np.nan for missing values
            for target in targets:
                targets[target] = [
                    info[target] if target in info else np.nan for info in infos
                ]
                last_info = episode.last_info_for(agent_id)
                targets[target].append(
                    last_info[target] if last_info and target in last_info else np.nan
                )
                targets[target] = targets[target][1:]
                targets[target] = np.array(targets[target], dtype=np.float32)

            # Apply scaling, but keep np.nan for missing values
            joker_marginal = np.where(
                np.isnan(targets["joker_marginal"]),
                np.nan,
                (np.log(np.maximum(targets["joker_marginal"], 1) / 100) - 1) / 3,
            )
            joker_chips = np.where(
                np.isnan(targets["joker_chips"]),
                np.nan,
                (((targets["joker_chips"] - 200) / 100) + 1.5) * 1.2,
            )
            joker_mult = np.where(
                np.isnan(targets["joker_mult"]),
                np.nan,
                (((targets["joker_mult"] - 60) / 30) + 1.5) * 2.0,
            )
            joker_mult_mult = np.where(
                np.isnan(targets["joker_mult_mult"]),
                np.nan,
                (((targets["joker_mult_mult"] - 2.0) / 2) + 0.3) * 2.0,
            )

            targets_arr = np.stack(
                [joker_marginal, joker_chips, joker_mult, joker_mult_mult], axis=-1
            )
            postprocessed_batch["joker_scores"] = targets_arr
            return
            # Add a curiosity bonus based on the accuracy of the predicted joker scores
            # No idea why actions sometimes aren't included, might be an evaluation mode bug?
            if (
                "hand_scores_aux_output" in postprocessed_batch
                and "actions" in postprocessed_batch
            ):
                batch = postprocessed_batch
                # print(batch)
                # print(agent_id)
                rewards = batch["rewards"]
                actions = batch[SampleBatch.ACTIONS]
                preds = batch["hand_scores_aux_output"]
                true_scores = batch["joker_scores"]
                has_jokers = batch["has_jokers"].astype(bool)

                num_aux = preds.shape[1]
                method = policy.config["model"]["custom_model_config"]["action_method"]
                if method == "subset_attention":
                    is_play = actions < num_aux
                elif method == "dual_subset":
                    actions = actions[:, 0]  # only care about the play action
                    is_play = actions > 0
                else:
                    raise ValueError(f"Unknown action_method {method}")

                valid = is_play & has_jokers & (~np.isnan(true_scores).any(axis=1))

                if not valid.any():
                    # print("No valid curiosity steps found, skipping curiosity bonus.")
                    return

                chosen_preds = preds[valid, actions[valid], :]
                chosen_true = true_scores[valid, :]
                with torch.no_grad():
                    errs = np.mean((chosen_preds - chosen_true) ** 2, axis=1)
                    errs = np.clip(errs, 0.0, 3.0)
                    bonus = 0.3 * errs
                    # print(f"Blind curiosity bonus: {bonus.mean():.4f} ± {bonus.std():.4f}")
                    # rewards[valid] += bonus
                    # batch["blind_curiosity_bonus"] = bonus # This breaks something idk why
                    # batch["rewards"] = rewards
                    postprocessed_batch["advantages"][valid] += bonus

                # done = batch[SampleBatch.DONES][-1]
                # last_val = 0.0 if done else batch[SampleBatch.VF_PREDS][-1]
                # new_batch = compute_gae_for_sample_batch(
                #     policy=policy,
                #     sample_batch=batch,
                #     episode=episode,
                # )
                # postprocessed_batch["advantages"] = new_batch["advantages"]
                # postprocessed_batch["value_targets"] = new_batch["value_targets"]

        elif "shop" in agent_id:
            return
            # Add a curiosity bonus based on the rarity of the "build" when the shop is ended
            policy = policies[policy_id]
            infos = postprocessed_batch[SampleBatch.INFOS]
            targets = {
                "shop_ended": [],
                "owned_joker_ids": [],
            }

            for target in targets:
                targets[target] = [
                    info[target] if target in info else np.nan for info in infos
                ]
                last_info = episode.last_info_for(agent_id)
                targets[target].append(
                    last_info[target] if last_info and target in last_info else np.nan
                )
                targets[target] = targets[target][1:]

            shop_ended = targets["shop_ended"]
            owned_joker_ids = targets["owned_joker_ids"]
            rewards = postprocessed_batch["rewards"]

            all_bonuses = []
            single_episode_counts = Counter()
            for i in range(len(shop_ended)):
                all_bonuses.append(0.0)
                if np.isnan(shop_ended[i]):
                    continue
                shop_ended[i] = bool(shop_ended[i])

                if not shop_ended[i]:
                    continue

                if (
                    type(owned_joker_ids[i]) is not list
                    and np.isnan(owned_joker_ids[i]).any()
                ):
                    continue

                bonus = 0.0
                for j in owned_joker_ids[i]:
                    if np.isnan(j):
                        continue
                    self.shop_joker_counts[j] += 1
                    single_episode_counts[j] += 1
                    # Calculate rarity score as ratio of count to most common
                    rarity_score = 1 - self.shop_joker_counts[j] / max(
                        self.shop_joker_counts.values()
                    )
                    # bonus += 1 / (self.shop_joker_counts[j] ** 0.25)
                    # bonus += rarity_score / (single_episode_counts[j] ** 0.5)
                    bonus += rarity_score
                    # if (
                    #     j == 46 + BaseCard.FIRST_JOKER_INDEX - 1
                    # ):  # Egg curiosity for testing
                    #     bonus += 10
                all_bonuses[i] = bonus * 2.0
                # rewards[i] += bonus * 1.0

            # for i in range(len(postprocessed_batch["advantages"])):
            #     postprocessed_batch["advantages"][i] += all_bonuses[i]

            # curiosity_rewards = np.array(all_bonuses, dtype=np.float32)
            # curiosity_dones = np.array(shop_ended, dtype=np.bool_)
            # curiosity_batch = SampleBatch(
            #     {
            #         **postprocessed_batch,
            #         SampleBatch.REWARDS: curiosity_rewards,
            #         SampleBatch.DONES: curiosity_dones,
            #     }
            # )
            # with torch.no_grad():
            #     curiosity_batch = compute_gae_for_sample_batch(
            #         policy=policy,
            #         sample_batch=curiosity_batch,
            #         episode=episode,
            #     )
            # postprocessed_batch["advantages"] += curiosity_batch["advantages"]
            # postprocessed_batch["value_targets"] += curiosity_batch["value_targets"]

            # curiosity_batch["rewards"] += curiosity_rewards

            # old_gamma = policy.config["gamma"]
            # old_lambda = policy.config["lambda"]
            # policy.config["gamma"] = 0.0
            # policy.config["lambda"] = 1.0
            # curiosity_batch = compute_gae_for_sample_batch(
            #     policy=policy,
            #     sample_batch=curiosity_batch,
            #     episode=episode,
            # )
            # policy.config["gamma"] = old_gamma
            # policy.config["lambda"] = old_lambda

            # if len(all_bonuses) > 0:
            #     print(
            #         f"Shop curiosity bonuses: {sum(all_bonuses)/len(all_bonuses):.4f} ± {np.std(all_bonuses):.4f}"
            #     )

            # new_batch = compute_gae_for_sample_batch(
            #     policy=policy, sample_batch=postprocessed_batch, episode=episode
            # )

            # postprocessed_batch["advantages"] = new_batch["advantages"]
            # postprocessed_batch["value_targets"] = new_batch["value_targets"]
