import gymnasium as gym
from gymnasium import spaces as sp
from gym_envs.components.consumable import Consumable
from gym_envs.components.spectral import SpectralCard
from balatro_connection import State, Actions
import numpy as np
from gym_envs.components.planet import PlanetCard

from random import randint, choices
from ray.rllib.utils.spaces.repeated import Repeated
from gym_envs.joker import Joker
from gym_envs.components.booster import Booster
from gym_envs.base_card import BaseCard
from gym_envs.components.hand_type import HandType
from gym_envs.components.card import Card
from gym_envs.components.deck import Deck
from gym_envs.components.hand import Hand
from gym_envs.components.tarot import TarotCard
from gym_envs.blind import Blind
from gym_envs.shared_gamestate import SharedGamestate


class ShopEnv(gym.Env):
    metadata = {"label": "ShopEnv-v0"}

    def __init__(self, env_config={}):
        super().__init__()

        self.illegal_action_reward = env_config.get("illegal_action_reward", -0.2)
        self.starting_dollars = env_config.get("starting_dollars", 5.0)

        self.G = SharedGamestate()
        self.G.dollars = 0
        self.round = 1
        self.shop_cards = 2
        self.reroll_cost = 5
        self.shop_jokers = []
        self.boosters = []
        self.booster_contents = []
        self.G.unlocked_jokers = set()
        self.G.owned_jokers = []
        self.G.joker_limit = 5
        self.max_boosters = 2
        self.max_booster_contents = 5
        self.G.deck = Deck()
        self.first_shop = True
        self.G.max_hand_size = env_config.get("max_hand_size", 15)
        self.max_hand_size = self.G.max_hand_size
        self.allow_joker_selling_before_cap = env_config.get(
            "allow_joker_selling_before_cap", False
        )
        self.allow_zero_rolling = env_config.get("allow_zero_rolling", False)
        self.ignore_rarity = env_config.get("ignore_rarity", False)
        self.next_blind = Blind.random(round=1)
        self.stake = 0

        self.min_dollars = -25.0
        self.max_dollars = 200.0

        self.action_space = self.build_action_space()
        self.observation_space = self.build_observation_space()

    def get_card_list(self):
        special_cards = [
            BaseCard(
                u_index=BaseCard.SpecialTokens.SHOP_CONTEXT
                + BaseCard.FIRST_SPECIAL_TOKEN_INDEX,
                scalar_properties=[0, 0, 0, 0],
            ),
            BaseCard(
                u_index=BaseCard.SpecialTokens.END_SHOP
                + BaseCard.FIRST_SPECIAL_TOKEN_INDEX,
                scalar_properties=[0, 0, 0, 0],
            ),
            BaseCard(
                u_index=BaseCard.SpecialTokens.REROLL_SHOP
                + BaseCard.FIRST_SPECIAL_TOKEN_INDEX,
                scalar_properties=[
                    (self.reroll_cost - 4)
                    / 10.0,  # Reroll cost following the joker scaling logic
                    0,
                    0,
                    0,
                ],
            ),
            BaseCard(
                u_index=BaseCard.SpecialTokens.END_BOOSTER
                + BaseCard.FIRST_SPECIAL_TOKEN_INDEX,
                scalar_properties=[0, 0, 0, 0],
            ),
        ]
        for x in special_cards:
            x.segment = BaseCard.Segments.SPECIAL_TOKEN

        shop_cards = self.shop_jokers + [BaseCard()] * (
            self.shop_cards - len(self.shop_jokers)
        )
        owned_jokers = self.G.owned_jokers + [BaseCard()] * (
            self.G.joker_limit - len(self.G.owned_jokers)
        )
        boosters = self.boosters + [BaseCard()] * (
            self.max_boosters - len(self.boosters)
        )
        booster_contents = self.booster_contents + [BaseCard()] * (
            self.max_booster_contents - len(self.booster_contents)
        )
        hand = self.G.hand.cards + [BaseCard()] * (
            self.G.max_hand_size - len(self.G.hand.cards)
        )
        hand = hand[
            : self.G.max_hand_size
        ]  # Ensure we don't exceed max hand size e.g. in spectral packs

        all_cards = (
            special_cards
            + shop_cards
            + owned_jokers
            + boosters
            + booster_contents
            + hand
        )
        return all_cards

    def round_info(self):
        chips_estimate = Blind.estimate_chips_for_round(self.round)
        lin = (chips_estimate / 50000) - 1
        return [
            self.round / 10 - 1,
            (chips_estimate / 300) ** 0.5 - 3,
            np.log(chips_estimate + 1) - 3,
            lin,
            np.sin(lin * np.pi / 2) * 2,
            np.cos(lin * np.pi / 2) * 2,
        ]

    def get_obs(self):
        self.enforce_segments()
        deck_size = np.array([len(self.G.deck.all_cards)], dtype=np.float32)
        deck_ranks = np.zeros(13, dtype=np.float32)
        deck_suits = np.zeros(4, dtype=np.float32)
        deck_enhancements = np.zeros(BaseCard.num_enhancements, dtype=np.float32)
        deck_editions = np.zeros(BaseCard.num_editions, dtype=np.float32)
        deck_seals = np.zeros(BaseCard.num_seals, dtype=np.float32)
        for card in self.G.deck.all_cards:
            deck_ranks[card.value - 2] += 1
            deck_suits[card.suit_index()] += 1
            deck_enhancements[card.enhancement] += 1
            deck_editions[card.edition] += 1
            deck_seals[card.seal] += 1
        deck_stats = np.concatenate(
            [
                (deck_ranks - 4) / 4,
                (deck_suits - 13) / 13,
                (deck_enhancements - 2) / 10,
                (deck_editions - 2) / 10,
                (deck_seals - 2) / 10,
                (deck_size - 52) / 10,
            ]
        )

        target_counts = np.zeros(self.max_booster_contents, dtype=np.int32)
        for i in range(len(self.booster_contents)):
            if isinstance(self.booster_contents[i], Consumable):
                target_counts[i] = self.booster_contents[i].num_targets

        return {
            "dollars": np.array(
                np.clip(self.G.dollars, self.min_dollars, self.max_dollars),
                dtype=np.float32,
            ),
            "round_info": np.array(self.round_info(), dtype=np.float32),
            "all_cards": BaseCard.observe_list(
                self.get_card_list(), self.card_list_size
            ),
            "action_mask": self.get_action_mask(),
            "hand_stats": HandType.observe_stats(self.G.hand_stats),
            "deck_stats": deck_stats,
            "num_targets": target_counts,
            "blind_index": np.array([self.next_blind.index], dtype=np.int32),
            "owned_joker_count": np.array(len(self.G.owned_jokers), dtype=np.float32),
            "max_owned_jokers": np.array(self.G.current_joker_limit, dtype=np.float32),
            "jokers_at_capacity": np.array(
                int(len(self.G.owned_jokers) >= self.G.current_joker_limit),
                dtype=np.float32,
            ),
        }

    def build_observation_space(self):
        self.card_list_size = len(self.get_card_list())
        return sp.Dict(
            {
                "dollars": sp.Box(
                    low=self.min_dollars,
                    high=self.max_dollars,
                    shape=(),
                    dtype=np.float32,
                ),
                "round_info": sp.Box(low=-20, high=200.0, shape=(6,), dtype=np.float32),
                "all_cards": BaseCard.observation_space(self.card_list_size),
                "action_mask": sp.Box(
                    low=0, high=1, shape=(17 + self.G.max_hand_size,), dtype=np.float32
                ),
                "deck_stats": sp.Box(low=-20, high=20, shape=(37,), dtype=np.float32),
                "hand_stats": HandType.stats_obs_space(),
                "num_targets": sp.Box(
                    low=0, high=3, shape=(self.max_booster_contents,), dtype=np.int32
                ),
                "blind_index": sp.Box(low=0, high=30, shape=(1,), dtype=np.int32),
                "owned_joker_count": sp.Box(low=0, high=20, shape=(), dtype=np.float32),
                "max_owned_jokers": sp.Box(low=0, high=20, shape=(), dtype=np.float32),
                "jokers_at_capacity": sp.Box(low=0, high=1, shape=(), dtype=np.float32),
            }
        )

    def can_pay(self, cost):
        have_cc = any(j.name == "Credit Card" for j in self.G.owned_jokers)
        return self.G.dollars >= cost or (have_cc and self.G.dollars - cost >= -25)

    def build_action_space(self):
        # 0) End shop
        # 1) Reroll shop
        # 2) End Booster
        # 3-4) Buy jokers
        # 5-9) Sell jokers
        # 10-11) Buy boosters
        # 12-16) Select booster contents
        # return sp.MultiDiscrete([17])
        return sp.Dict(
            {
                "action": sp.Discrete(17),
                "hand_targets": sp.MultiDiscrete(
                    [2] * self.G.max_hand_size
                ),  # 0 or 1 for each card in hand
            }
        )

    def get_action_mask(self):
        mask = np.zeros(17 + self.G.max_hand_size, dtype=np.float32)
        in_booster = len(self.booster_contents) > 0
        # We can't end the shop from inside a booster
        if in_booster:
            mask[0] = 1.0
        # Can we reroll
        if (
            not self.can_pay(self.reroll_cost)
            or in_booster
            or (not self.allow_zero_rolling and not self.can_pay(self.reroll_cost + 5))
        ):
            mask[1] = 1.0
        if not in_booster:
            mask[2] = 1.0  # Can we skip the booster
        # Can we purchase each card
        for i in range(1, self.shop_cards + 1):
            if not self.check_card_purchaseable([Actions.BUY_CARD, [i]]) or in_booster:
                mask[i + 2] = 1.0

        # Can we sell each owned joker
        for i in range(1, 6):
            joker_i = i + 2 + self.shop_cards
            if len(self.G.owned_jokers) < i:
                mask[joker_i] = 1.0
            elif (
                not self.allow_joker_selling_before_cap
                and len(self.G.owned_jokers) < self.G.joker_limit
            ):
                mask[joker_i] = 1.0
            elif self.G.owned_jokers[i - 1].seal == BaseCard.Seals.ETERNAL:
                mask[joker_i] = 1.0  # Can't sell eternal jokers

        # Can we buy each booster
        for i in range(1, self.max_boosters + 1):
            if (
                not self.check_card_purchaseable([Actions.BUY_BOOSTER, [i]])
                or in_booster
            ):
                mask[i + 2 + self.shop_cards + self.G.joker_limit] = 1.0

        # Can we select booster options
        for i in range(1, self.max_booster_contents + 1):
            index = i + 2 + self.shop_cards + self.G.joker_limit + self.max_boosters
            if len(self.booster_contents) < i:
                mask[index] = 1.0
            elif self.booster_contents[i - 1].get_universal_index() == 0:
                # Padding card from real environment
                mask[index] = 1.0
            # Make sure we aren't going over joker limit if the booster contents are jokers
            elif len(self.G.owned_jokers) >= self.G.joker_limit:
                if isinstance(self.booster_contents[i - 1], Joker):
                    mask[index] = 1.0
            elif isinstance(self.booster_contents[i - 1], TarotCard) or isinstance(
                self.booster_contents[i - 1], SpectralCard
            ):
                if len(self.G.hand) < self.booster_contents[i - 1].num_targets:
                    # Not enough cards in hand to target
                    mask[index] = 1.0

        # Can we target cards in the hand for tarot cards
        for i in range(1, self.G.max_hand_size + 1):
            if (
                len(self.G.hand) < i or type(self.G.hand[i - 1]) != Card
            ):  # Padding card from real environment
                mask[
                    i
                    + 2
                    + self.shop_cards
                    + self.G.joker_limit
                    + self.max_boosters
                    + self.max_booster_contents
                ] = 1.0

        return mask

    def check_card_purchaseable(self, action):
        if action[0] == Actions.BUY_CARD:
            i = action[1][0] - 1
            if i >= len(self.shop_jokers):
                return False
            if len(self.G.owned_jokers) >= self.G.joker_limit:
                return False
            joker = self.shop_jokers[i]
            if joker.get_universal_index() == 0:
                return False  # padding card from real environment
            if type(joker) != Joker:  # Disallow buying tarot and planet cards for now
                return False

            affordable = self.can_pay(joker.value)
            return affordable
        elif action[0] == Actions.BUY_BOOSTER:
            i = action[1][0] - 1
            if i >= len(self.boosters):
                return False
            booster = self.boosters[i]
            if booster.get_universal_index() == 0:
                return False  # padding card from real environment
            affordable = self.can_pay(booster.value)
            return affordable
        else:
            raise ValueError(f"Unknown action type for purchasing: {action[0]}")

    def action_vector_to_action(self, action):
        if action["action"] == 0:
            return [Actions.END_SHOP]
        elif action["action"] == 1:
            return [Actions.REROLL_SHOP]
        elif action["action"] == 2:
            return [Actions.SKIP_BOOSTER_PACK]
        elif action["action"] in [3, 4]:
            return [Actions.BUY_CARD, [action["action"] - 2]]
        elif action["action"] in range(5, 10):
            return [Actions.SELL_JOKER, [action["action"] - 4]]
        elif action["action"] in range(10, 12):
            return [Actions.BUY_BOOSTER, [action["action"] - 9]]
        elif action["action"] in range(12, 17):
            hand_targets = []
            for i, x in enumerate(action["hand_targets"]):
                if x == 1:
                    hand_targets.append(i + 1)
            # print(action["hand_targets"], hand_targets)
            return [Actions.SELECT_BOOSTER_CARD, [action["action"] - 11], hand_targets]

    def roll_jokers(self):
        self.shop_jokers = []
        # Generate a random joker and ensure it is unique and not already owned
        while len(self.shop_jokers) < 2:
            r = randint(1, 28)
            if r <= 20:
                joker = Joker.random(
                    unlocked_jokers=self.G.unlocked_jokers,
                    sparse_pool=True,
                    ignore_rarity=self.ignore_rarity,
                    stake=self.stake,
                )
                joker.segment = BaseCard.Segments.SHOP_BUYABLE
                if any(
                    j.is_dupe(joker) for j in self.shop_jokers + self.G.owned_jokers
                ):
                    continue
                self.shop_jokers.append(joker)
            elif r <= 24:
                planet = PlanetCard.random(unlocked=self.G.unlocked_jokers)
                planet.segment = BaseCard.Segments.SHOP_BUYABLE
                if any(
                    j.is_dupe(planet) for j in self.shop_jokers + self.G.owned_jokers
                ):
                    continue
                self.shop_jokers.append(planet)
            elif r <= 28:
                tarot = TarotCard.random()
                tarot.segment = BaseCard.Segments.SHOP_BUYABLE
                if any(
                    j.is_dupe(tarot) for j in self.shop_jokers + self.G.owned_jokers
                ):
                    continue
                self.shop_jokers.append(tarot)

    def roll_boosters(self):
        self.boosters = []
        # Generate a random booster and ensure it is unique
        if self.first_shop:
            self.first_shop = False
            self.boosters.append(Booster("Buffoon", "Normal", 1.2))
        while len(self.boosters) < self.max_boosters:
            self.boosters.append(Booster.random())
            if any(j.name == "Astronomer" for j in self.G.owned_jokers):
                if self.boosters[-1].name == "Celestial":
                    self.boosters[-1].value = 0

    def roll_shop(self):
        self.roll_jokers()

    def new_shop(self):
        self.roll_shop()
        self.roll_boosters()
        self.booster_contents = []
        self.G.hand = Hand([])
        self.reroll_cost = 5
        if any([j.name == "Chaos the Clown" for j in self.G.owned_jokers]):
            # Chaos the Clown allows rerolling once for free
            self.reroll_cost = 0

    def reset(self, seed=None, options=None):
        self.first_shop = True
        self.G = SharedGamestate()
        self.G.max_hand_size = self.max_hand_size
        self.new_shop()
        self.G.dollars = self.starting_dollars
        self.G.owned_jokers = []
        self.G.unlocked_jokers = set()
        self.jokers_purchased = 0
        self.jokers_sold = 0
        self.reroll_count = 0
        self.boosters_purchased = 0
        self.boosters_skipped = 0
        self.round = 1
        self.G.hand_stats = HandType.all_hands()
        self.telemetry = {
            k: 0
            for k in [
                "jokers_purchased",
                "jokers_sold",
                "rerolls",
                "jokers_sold_before_limit",
                "boosters_purchased",
                "boosters_skipped",
            ]
            + [f"{x.full_name()}_purchased" for x in Booster.all()]
            + [f"booster_choices/planet/{x.name}" for x in PlanetCard.all()]
            + [f"booster_choices/rank/{x}" for x in Card.RANKS]
            + [f"booster_choices/suit/{x}" for x in Card.SUITS]
            + [f"booster_choices/tarot/{x.name}" for x in TarotCard.all()]
        }
        return self.get_obs(), {}

    def step(self, action):
        action = self.action_vector_to_action(action)
        if action[0] == Actions.BUY_CARD or action[0] == Actions.BUY_BOOSTER:
            if not self.check_card_purchaseable(action):
                # End the shop if the action is invalid for any reason
                reward = self.illegal_action_reward
                print(
                    f"Illegal action: Cannot make purchase {action[0]} with parameters {action[1]}, dollars: {self.G.dollars}, owned_jokers: {len(self.G.owned_jokers)}, boosters: {len(self.boosters)}"
                )
                action = [Actions.END_SHOP]
            else:
                reward = 0.03  # Small positive reward for valid purchase action
        elif action[0] == Actions.END_SHOP:
            reward = 0
        elif action[0] == Actions.REROLL_SHOP:
            if not self.can_pay(self.reroll_cost):
                reward = self.illegal_action_reward
                print("Illegal action: Not enough dollars to reroll shop")
                action = [Actions.END_SHOP]
            elif not self.can_pay(self.reroll_cost + 5):
                reward = -0.05
            else:
                reward = 0.01
        elif action[0] == Actions.SELL_JOKER:
            if len(self.G.owned_jokers) < action[1][0]:
                reward = self.illegal_action_reward
                print("Illegal action: Not enough jokers to sell")
                action = [Actions.END_SHOP]
            elif len(self.G.owned_jokers) < self.G.joker_limit:
                reward = -0.05
            else:
                reward = -0.01
        else:
            reward = 0

        self.take_action(action)
        obs = self.get_obs()
        return (
            obs,
            reward,
            action[0] == Actions.END_SHOP,
            False,
            {},
            # {
            #     "shop_ended": action[0] == Actions.END_SHOP,
            #     "owned_joker_ids": [
            #         j.get_universal_index() for j in self.G.owned_jokers
            #     ],
            # },
        )

    def take_action(self, action):
        if action[0] == Actions.BUY_CARD:
            self.telemetry["jokers_purchased"] += 1
            purchased_joker = self.shop_jokers.pop(action[1][0] - 1)
            purchased_joker.segment = BaseCard.Segments.JOKER
            self.G.dollars -= purchased_joker.value
            # Loses half the value when you drive it off the lot
            purchased_joker.value = max(int(purchased_joker.value / 2.0), 1)
            self.G.owned_jokers.append(purchased_joker)
            self.jokers_purchased += 1
            if type(purchased_joker) == BaseCard:
                raise ValueError(f"Purchased joker is not a Joker: {purchased_joker}")
        elif action[0] == Actions.REROLL_SHOP:
            self.telemetry["rerolls"] += 1
            self.G.dollars -= self.reroll_cost
            if self.reroll_cost == 0:
                self.reroll_cost = 5  # Reset to default cost after free reroll
            else:
                self.reroll_cost += 2
            self.roll_jokers()
            self.reroll_count += 1
            for j in self.G.owned_jokers:
                if j.name == "Flash Card":
                    j.state["mult"] += 2
        elif action[0] == Actions.SELL_JOKER:
            self.telemetry["jokers_sold"] += 1
            if len(self.G.owned_jokers) < min(
                self.G.joker_limit, self.G.current_joker_limit
            ):
                self.telemetry["jokers_sold_before_limit"] += 1
            sold_joker = self.G.owned_jokers.pop(action[1][0] - 1)
            self.G.dollars += sold_joker.value
            self.jokers_sold += 1
        elif action[0] == Actions.END_SHOP:
            self.roll_shop()
        elif action[0] == Actions.BUY_BOOSTER:
            self.telemetry["boosters_purchased"] += 1
            booster = self.boosters.pop(action[1][0] - 1)
            self.telemetry[f"{booster.full_name()}_purchased"] += 1
            self.booster_choices_left = booster.num_choices
            self.G.dollars -= booster.value
            self.booster_contents = booster.open(
                self.G.owned_jokers + self.shop_jokers,
                self.G.unlocked_jokers,
                ignore_rarity=self.ignore_rarity,
                stake=self.stake,
            )
            if booster.name in ["Arcana", "Spectral"]:
                self.G.hand = Hand([])
                self.G.deck.reset()
                self.G.hand = Hand(
                    [self.G.deck.draw() for _ in range(self.G.current_hand_size)]
                )
            self.boosters_purchased += 1
        elif action[0] == Actions.SELECT_BOOSTER_CARD:
            card = self.booster_contents.pop(action[1][0] - 1)
            if isinstance(card, Joker):
                card.segment = BaseCard.Segments.JOKER
                card.value = max(int(card.value / 2.0), 1)
                self.G.owned_jokers.append(card)
            elif isinstance(card, Consumable):
                targets = []
                if len(action) > 2:
                    for x in action[2]:
                        targets.append(self.G.hand[x - 1])  # Convert to 0-indexed
                result = card.trigger(targets=targets, gamestate=self.G)
                if isinstance(card, PlanetCard):
                    self.telemetry[f"booster_choices/planet/{card.name}"] += 1
                elif isinstance(card, TarotCard):
                    self.telemetry[f"booster_choices/tarot/{card.name}"] += 1
            elif isinstance(card, Card):
                self.telemetry[f"booster_choices/rank/{card.value}"] += 1
                self.telemetry[f"booster_choices/suit/{card.suit}"] += 1
                card.segment = BaseCard.Segments.HAND
                self.G.create_card(card, no_hand=True)
            else:
                raise ValueError(f"Unknown card type in booster contents: {type(card)}")
            self.booster_choices_left -= 1
            if self.booster_choices_left <= 0:
                self.booster_contents = []
                self.G.hand = Hand([])
        elif action[0] == Actions.SKIP_BOOSTER_PACK:
            self.telemetry["boosters_skipped"] += 1
            self.booster_contents = []
            self.booster_choices_left = 0
            self.boosters_skipped += 1
            self.G.hand = Hand([])

    def load_gamestate(self, gamestate):
        self.G.dollars = gamestate.get("dollars", self.starting_dollars)
        self.round = gamestate.get("round", 2) + 1
        shop_data = gamestate.get("shop", {})
        self.reroll_cost = shop_data.get("reroll_cost", 5)
        self.shop_jokers = [
            BaseCard.from_gamestate_card(card) for card in shop_data.get("cards", [])
        ]
        self.G.owned_jokers = [
            Joker.from_gamestate_card(card) for card in gamestate.get("jokers", [])
        ]
        self.boosters = [
            BaseCard.from_gamestate_card(card) for card in shop_data.get("boosters", [])
        ]
        self.booster_contents = [
            BaseCard.from_gamestate_card(card)
            for card in shop_data.get("pack_cards", [])
        ]
        self.G.deck = Deck.from_gamestate_deck(gamestate.get("deck", []))
        self.hand = Hand.from_gamestate_hand(gamestate.get("hand", []))
        self.enforce_segments()
        # print(self.get_action_mask())
        # self.G.unlocked_jokers = set(gamestate.get("unlocked_jokers", []))
        # self.G.hand_stats = HandType.load_from_gamestate(gamestate.get("hand_stats", {}))

    def enforce_segments(self):
        for x in self.shop_jokers:
            # x.segment = BaseCard.Segments.SHOP_BUYABLE
            x.segment = BaseCard.Segments.SHOP_JOKER
        for x in self.G.owned_jokers:
            x.segment = BaseCard.Segments.JOKER
        for x in self.boosters:
            x.segment = BaseCard.Segments.SHOP_BUYABLE
        for x in self.booster_contents:
            x.segment = BaseCard.Segments.BOOSTER_CHOICE
        for x in self.G.hand:
            x.segment = BaseCard.Segments.HAND
