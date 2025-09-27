"""Gameplay helpers extracted from :mod:`gym_envs.envs.blind_env`."""

from __future__ import annotations

from copy import deepcopy
from random import choice, choices, randint, random
from typing import Dict, Iterable, List

import numpy as np

from gym_envs.base_card import BaseCard
from gym_envs.blind import Blind
from gym_envs.joker import Joker
from gym_envs.joker_effects import (
    joker_card_score_effects,
    joker_discard_effects,
    joker_round_start_effects,
    joker_round_win_effects,
    joker_triggered_effects,
)
from gym_envs.components.card import Card
from gym_envs.components.hand import Hand
from gym_envs.components.deck import Deck


class BlindGameplayHelper:
    """Encapsulates the heavy gameplay logic used by :class:`BlindEnv`."""

    def __init__(self, env) -> None:
        self._env = env

    def __setattr__(self, name, value):
        if name == "_env" or not hasattr(self, "_env"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def determine_play_hand_outcome(self, played_hand):
        game_over = self.hands_left == 0
        no_op_result = {
            "reward": -0.05,
            "game_over": game_over,
            "hand_type": "High Card",
            "scored_cards": played_hand,
            "hand_score": 0,
        }
        if self.G.current_blind.name == "The Psychic":
            if len(played_hand) < 5:
                return no_op_result
        elif self.G.current_blind.name == "The Hook":
            if len(self.G.hand) > 0:
                if len(self.G.hand) == 1:
                    discard_cards = self.G.hand.pop_cards([1])
                else:
                    discard_indices = [randint(1, len(self.G.hand)) for _ in range(2)]
                    while discard_indices[0] == discard_indices[1]:
                        discard_indices[1] = randint(1, len(self.G.hand))
                    discard_cards = self.G.hand.pop_cards(discard_indices)

                effects = joker_discard_effects(self.G.owned_jokers, discard_cards)
                self.handle_callbacks(effects["callbacks"])

        disabled_joker = None
        if self.G.current_blind.name == "Crimson Heart":
            if len(self.G.owned_jokers) > 0:
                joker = choice(self.G.owned_jokers)
                self.G.owned_jokers.remove(joker)
                disabled_joker = deepcopy(joker)

        four_finger_joker = any(
            joker.name == "Four Fingers" for joker in self.G.owned_jokers
        )
        hand_type, scored_cards = played_hand.evaluate(
            allow_4_flush=four_finger_joker,
            allow_4_straight=four_finger_joker,
        )

        if self.G.current_blind.name == "The Eye":
            if self.G.hand_stats[hand_type].played_this_blind:
                return no_op_result
        if self.G.current_blind.name == "The Mouth":
            if not self.G.hand_stats[hand_type].played_this_blind:
                if any(
                    self.G.hand_stats[h].played_this_blind for h in self.G.hand_stats
                ):
                    return no_op_result
        if self.G.current_blind.name == "The Arm":
            if self.G.hand_stats[hand_type].level > 1:
                self.G.hand_stats[hand_type].add_level(-1)

        self.G.hand_stats[hand_type].play_count += 1
        self.G.hand_stats[hand_type].played_this_blind = True

        splash = any(joker.name == "Splash" for joker in self.G.owned_jokers)
        if splash:
            scored_cards = played_hand

        if any([j.name == "Space Joker" for j in self.G.owned_jokers]) and random() < 0.25:
            self.G.hand_stats[hand_type].add_level(1)

        reward = 0
        chips, mult = self.G.hand_stats[hand_type].scores()
        if self.G.current_blind.name == "The Flint":
            chips = max(int(chips / 2), 1)
            mult = max(int(mult / 2), 1)

        if self.flattened_rank_chips:
            chips += 7 * len(scored_cards)
        else:
            chips += sum([card.chip_value() for card in scored_cards])
        chips += sum(
            50 if card.enhancement == BaseCard.Enhancements.STONE else 0
            for card in played_hand.cards
        )
        chips += sum(
            (
                50
                if card.enhancement == BaseCard.Enhancements.STONE
                and card.seal == BaseCard.Seals.RED
                else 0
            )
            for card in played_hand.cards
        )
        pre_joker_score = chips * mult

        scored_cards.cards = self.G.current_blind.filter_scored_cards(
            scored_cards.cards
        )
        aggregate_joker_effects = {"chips": 0, "mult": 0, "mult_mult": 1.0}
        for card in scored_cards.cards:
            if card.enhancement == BaseCard.Enhancements.BONUS:
                chips += 30
            elif card.enhancement == BaseCard.Enhancements.MULT:
                mult += 4
            elif card.enhancement == BaseCard.Enhancements.GLASS:
                mult *= 2
                if random() < 0.25:
                    self.G.destroy_card(card)
                    for j in self.G.owned_jokers:
                        if j.name == "Glass Joker":
                            j.state["mult_mult"] += 0.75
            elif card.enhancement == BaseCard.Enhancements.LUCKY:
                triggers = 0
                if random() < 0.20:
                    mult += 20
                    triggers += 1
                if random() < 0.067:
                    self.G.dollars += 20
                    triggers += 1

                for j in self.G.owned_jokers:
                    if j.name == "Lucky Cat":
                        j.state["mult_mult"] += 0.25 * triggers

            if card.edition == BaseCard.Editions.FOIL:
                chips += 50
            elif card.edition == BaseCard.Editions.HOLOGRAPHIC:
                mult += 10
            elif card.edition == BaseCard.Editions.POLYCHROME:
                mult *= 1.5

            effects = joker_card_score_effects(self.G.owned_jokers, card, self.G)
            aggregate_joker_effects["chips"] += effects["chips"]
            aggregate_joker_effects["mult"] += effects["mult"]
            aggregate_joker_effects["mult_mult"] *= effects["mult_mult"]
            if card.seal == BaseCard.Seals.GOLD:
                self.G.dollars += 3
            elif card.seal == BaseCard.Seals.RED:
                chips += card.chip_value()
                effects["chips"] *= 2
                effects["mult"] *= 2
                effects["mult_mult"] = effects["mult_mult"] ** 2

            self.handle_callbacks(effects["callbacks"])
            chips += effects["chips"]
            mult += effects["mult"]
            mult *= effects["mult_mult"]
            reward += effects["synergy"] * self.joker_synergy_bonus

        for card in self.G.hand.cards:
            if card.enhancement == BaseCard.Enhancements.STEEL:
                mult *= 1.5

        effects = joker_triggered_effects(
            self.G.owned_jokers,
            played_hand,
            scored_cards,
            self.G.hand,
            hand_type,
            self.hands_left,
            self.discards_left,
            self.G.hand_stats,
            self.G.deck,
            self.G.dollars,
            self.G,
        )
        self.handle_callbacks(effects["callbacks"])
        chips += effects["chips"]
        mult += effects["mult"]
        mult *= effects["mult_mult"]
        reward += effects["synergy"] * self.joker_synergy_bonus
        aggregate_joker_effects["chips"] += effects["chips"]
        aggregate_joker_effects["mult"] += effects["mult"]
        aggregate_joker_effects["mult_mult"] *= effects["mult_mult"]

        post_joker_score = chips * mult
        joker_marginal = post_joker_score - pre_joker_score

        if self.objective_mode == "max_chips":
            norm = self.chip_reward_normalization
            crw = self.chips_reward_weight
            if norm == "log_joker":
                reward += pre_joker_score * crw
                reward += np.log(joker_marginal * crw + 1)
            elif norm == "sqrt_joker":
                reward += pre_joker_score * crw
                reward += np.sqrt(joker_marginal * crw)
            elif norm == "log":
                reward += np.log(post_joker_score * crw + 1)
            elif norm == "sqrt":
                reward += np.sqrt(post_joker_score * crw)
            else:
                reward += post_joker_score * crw
        elif self.objective_mode == "blind_grind":
            remaining_progress = (self.chip_goal - self.chips) / self.chip_goal
            progress = post_joker_score / self.chip_goal
            overkill = max(0, progress - remaining_progress)
        elif self.objective_mode == "one_hand_easy":
            if hand_type != self.easy_hand_type and self.easy_hand_type == "Flush":
                reward = 0.5 * max(played_hand.suit_counts().values()) / 10
                if len(played_hand) < 5:
                    reward = 0.0
            elif hand_type == "High Card":
                reward = 0.0
            elif hand_type == self.easy_hand_type:
                reward = 0.5
            else:
                reward = 0.1

        reward += (
            self.target_hand_types[self.hand_to_id[hand_type]]
            * self.hand_type_reward_weight
        )

        avg_rarity = sum(self.rarities.values()) / len(self.rarities)
        rarity = self.rarities[hand_type]
        if rarity > avg_rarity:
            reward += (rarity - avg_rarity) * self.rarity_bonus

        if len(played_hand) == 5:
            reward += played_hand.suit_homogeneity() * self.suit_homogeneity_bonus

        suit_counts = [0, 0, 0, 0]
        for card in played_hand.cards:
            suit_counts[card.suit_index()] += 1

        game_over = self.hands_left == 0
        result = {
            "reward": reward,
            "game_over": game_over,
            "hand_type": hand_type,
            "scored_cards": scored_cards,
            "suit_counts": suit_counts,
            "hand_score": post_joker_score,
            "pre_joker_score": pre_joker_score,
            "joker_marginal": joker_marginal,
            "joker_chips": aggregate_joker_effects["chips"],
            "joker_mult": aggregate_joker_effects["mult"],
            "joker_mult_mult": aggregate_joker_effects["mult_mult"],
            "played_hand": played_hand,
        }
        if disabled_joker is not None:
            self.G.owned_jokers.append(disabled_joker)
        return result

    def update_scored_card_stats(self, scored_cards, played_cards, hand_type):
        if scored_cards is None:
            return
        suit_map = {
            "Clubs": 0,
            "Diamonds": 1,
            "Hearts": 2,
            "Spades": 3,
        }
        for card in scored_cards:
            self.scored_ranks[hand_type][card.value - 2] += 1
            self.scored_suits[hand_type][suit_map[card.suit]] += 1
        self.hand_counts[hand_type] += 1
        if hand_type == self.target_hand_type:
            self.hit_rates[hand_type] += 1

        self.count_counts[len(played_cards)] += 1

    def draw_cards(self):
        if self.objective_mode == "one_hand_easy":
            hand_i = randint(0, len(self.hands) - 1)
            self.active_expert = hand_i
            hand_type = self.hands[hand_i]
            self.easy_hand_type = hand_type
            self.G.hand = Hand.random(hand_type, self.G.max_hand_size)
            self.G.hand.shuffle()
            return

        starting_size = len(self.G.hand)
        while len(self.G.hand) < self.G.current_hand_size:
            if not self.infinite_deck:
                if len(self.G.deck.remaining_cards) == 0:
                    break
                bias = self.biases[self.target_hand_type]
                if bias is None or bias <= 0:
                    card = self.G.deck.draw()
                else:
                    biasers = [lambda x: 0]
                    if starting_size > 0:
                        if self.target_hand_type in [
                            "Pair",
                            "Three of a Kind",
                            "Four of a Kind",
                            "Two Pair",
                            "Full House",
                        ]:
                            biasers.append(self.G.hand.rank_biaser())
                        if self.target_hand_type in ["Flush"]:
                            biasers.append(self.G.hand.suit_biaser())
                        if self.target_hand_type in ["Straight"]:
                            biasers.append(self.G.hand.straight_biaser())
                        if self.target_hand_type in ["Straight Flush"]:
                            biasers.append(self.G.hand.straight_flush_biaser())
                    biaser = lambda x: sum([b(x) for b in biasers])
                    if len(self.G.deck.remaining_cards) == 0:
                        break
                    card = self.G.deck.draw_biased(biaser, bias)
            else:
                card = Card.random()
            self.G.hand.add_card(card)

        self.G.hand.shuffle()

    def handle_callbacks(self, callbacks: Iterable):
        for callback in callbacks:
            if callback[0] == "destroy_joker":
                if callback[1] in self.G.owned_jokers:
                    self.G.destroy_card(callback[1])
            elif callback[0] == "unlock_joker":
                self.G.unlocked_jokers.add(callback[1])
            elif callback[0] == "add_joker":
                if len(self.G.owned_jokers) < self.G.current_joker_limit:
                    if type(callback[1]) == Joker:
                        self.G.owned_jokers.append(callback[1])
            elif callback[0] == "earn_money":
                self.G.dollars += callback[1]
            elif callback[0] == "hand_level":
                self.G.hand_stats[callback[1]].add_level(1)
            else:
                print(f"Unhandled callback: {callback}")

    def imagine_play_hand(self, actually_played, action):
        mutated_hand = actually_played.mutate()
        result = self.determine_play_hand_outcome(mutated_hand)

        imagined_hand = Hand([])
        j = 0
        k = 0
        for i in range(len(self.G.hand) + len(mutated_hand)):
            if i + 1 in action[1]:
                imagined_hand.cards.append(mutated_hand.cards[j])
                j += 1
            else:
                imagined_hand.cards.append(self.G.hand.cards[k])
                k += 1

        result["imagined_hand"] = imagined_hand
        return result

    def fresh_blind(self, fake_reset: bool = False):
        self.G.deck.reset()
        self.G.hand = Hand([])
        if not fake_reset:
            self.G.current_blind = self.next_blind
            self.next_blind = Blind.random(self.round + 1)

        if any(j.name == "Chicot" for j in self.G.owned_jokers):
            if self.G.current_blind.is_boss():
                if self.G.current_blind.name == "Violet Vessel":
                    self.G.current_blind.chip_goal /= 3
                elif self.G.current_blind.name == "The Wall":
                    self.G.current_blind.chip_goal /= 2
                self.G.current_blind.name = "Big Blind"
                self.G.current_blind.index = 1

        self.chip_goal = self.G.current_blind.chip_goal
        if not self.expert_pretraining:
            self.active_expert = None

        if self.round == 1 and random() < self.missed_wins_p and len(self.missed_wins) > 0:
            i = randint(0, len(self.missed_wins) - 1)
            self.G.hand = deepcopy(self.missed_wins[i])
            if random() < self.missed_wins_decay_p:
                print(f"Missed wins before pop: {len(self.missed_wins)}")
                self.missed_wins.pop(i)
        else:
            self.draw_cards()
        self.chips = 0
        self.reset_hand_watermarks()

        if type(self.max_plays) == dict:
            self.hands_left = self.max_plays[self.target_hand_type]
            self.discards_left = self.max_discards[self.target_hand_type]
        else:
            self.hands_left = self.max_plays
            self.discards_left = self.max_discards

        if self.G.current_blind.name == "The Water":
            self.discards_left = 0
        if self.G.current_blind.name == "The Needle":
            self.hands_left = 1

        for h in self.G.hand_stats:
            self.G.hand_stats[h].played_this_blind = False

        if not fake_reset:
            effects = joker_round_start_effects(
                self.G.owned_jokers, self.G.hand, self.hands_left, self.discards_left
            )
            self.hands_left = effects["hands_left"]
            self.discards_left = effects["discards_left"]
            self.handle_callbacks(effects["callbacks"])
        if self.stake >= 4:
            self.discards_left -= 1
            self.discards_left = max(self.discards_left, 0)

    def catchup(self):
        if self.round <= 1:
            return

        self.G.dollars += self.round * 4
        expected_joker_count = (self.round / 3) + 1
        weights = [1] * 6
        for i in range(0, 6):
            weights[i] = (abs(expected_joker_count - i) + 1) ** -2
        joker_count = choices(range(0, 6), weights=weights, k=1)[0]

        self.G.owned_jokers = [
            Joker.random(sparse_pool=False, ignore_rarity=True)
            for _ in range(joker_count)
        ]
        for joker in self.G.owned_jokers:
            if "chips" in joker.state and joker.state["chips"] == 0:
                joker.state["chips"] = int(random() * self.round * 5)
            if "mult" in joker.state and joker.state["mult"] == 0:
                joker.state["mult"] = int(random() * self.round * 2)
            if "mult_mult" in joker.state and joker.state["mult_mult"] == 1:
                joker.state["mult_mult"] = (10 + int(random() * self.round / 10)) / 10

        self.G.dollars -= sum([j.value for j in self.G.owned_jokers])
        self.G.dollars = max(self.G.dollars, 0)

        favorite = choice(self.hands)
        associated_hand_types = {
            "Flush": ["Straight Flush"],
            "Straight": ["Straight Flush"],
            "Straight Flush": ["Flush", "Straight"],
            "Full House": ["Two Pair", "Three of a Kind"],
            "Two Pair": ["Full House", "Three of a Kind", "Pair"],
        }.get(favorite, [])
        main_level = randint(self.round - 5, self.round + 3)
        main_level = max(main_level, 1)
        self.G.hand_stats[favorite].set_level(main_level)
        self.G.hand_stats[favorite].play_count = self.round * 2
        for hand_type in associated_hand_types:
            if hand_type in self.G.hand_stats:
                self.G.hand_stats[hand_type].set_level(max(main_level // 2, 1))
                self.G.hand_stats[hand_type].play_count = self.round
        self.G.hand_stats["High Card"].play_count = self.round * 2

        self.G.dollars -= main_level * 2
        self.G.dollars = max(self.G.dollars, 0)

        suit_homogeneity = random() * self.round / 24
        rank_homogeneity = random() * self.round / 24
        random_card_rate = random() * self.round / 10

        preferred_suit = choice(Card.SUITS)
        preferred_rank = choice(Card.RANKS)

        deck_size = randint(52 - self.round, 52 + self.round)
        while len(self.G.deck.all_cards) < deck_size:
            self.G.deck.add_card(Card.random())
        while len(self.G.deck.all_cards) > deck_size:
            self.G.deck.all_cards.pop(randint(0, len(self.G.deck.all_cards) - 1))

        for card in self.G.deck.all_cards:
            if random() < suit_homogeneity:
                card.suit = preferred_suit
            if random() < rank_homogeneity:
                card.value = preferred_rank
            if (
                random() < random_card_rate
                and card.enhancement == BaseCard.Enhancements.NORMAL
                and card.seal == BaseCard.Seals.NO_SEAL
                and card.edition == BaseCard.Editions.NO_EDITION
            ):
                sample_card = Card.random()
                card.enhancement = sample_card.enhancement
                card.seal = sample_card.seal
                card.edition = sample_card.edition

        self.G.dollars -= int(suit_homogeneity)
        self.G.dollars -= int(rank_homogeneity)
        self.G.dollars = max(self.G.dollars, 0)

        self.G.dollars += int(random() * self.round * 2)

    def load_gamestate(self, G: Dict):
        self.G.hand = Hand.from_gamestate_hand(G["hand"])
        self.G.deck = Deck.from_gamestate_deck(G["deck"])
        for k in G["handscores"]:
            self.G.hand_stats[k].chips = G["handscores"][k]["chips"]
            self.G.hand_stats[k].mult = G["handscores"][k]["mult"]
        self.G.dollars = G["dollars"]
        self.chips = G["chips"]
        self.chip_goal = G["current_round"]["chips_required"]
        self.round = G["round"]
        self.hands_left = G["current_round"]["hands_left"]
        self.discards_left = G["current_round"]["discards_left"]
        self.G.current_blind = Blind.from_gamestate(G)
        self.enforce_segments()

    def enforce_segments(self):
        for card in self.G.hand.cards:
            card.segment = BaseCard.Segments.HAND
        for card in self.G.owned_jokers:
            card.segment = BaseCard.Segments.JOKER
