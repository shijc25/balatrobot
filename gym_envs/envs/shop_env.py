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

        self.hand_to_id = env_config.get("hand_to_id", {})
        
        self.illegal_action_reward = env_config.get("illegal_action_reward", 0)
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
        
        self.in_pack_selection = False
        self.num_booster_contents = 5

    def get_obs(self):
        self.enforce_segments()
        
        shop_prices = np.full(2, 999.0, dtype=np.float32)
        for i, j in enumerate(self.shop_jokers):
            if hasattr(j, 'value'): shop_prices[i] = j.value

        booster_prices = np.full(2, 999.0, dtype=np.float32)
        for i, b in enumerate(self.boosters):
            if hasattr(b, 'value'): booster_prices[i] = b.value
        
        is_joker_on_shelf = np.zeros(2, dtype=np.bool_)
        is_planet_on_shelf = np.zeros(2, dtype=np.bool_)
        is_buffoon_pack = np.zeros(2, dtype=np.bool_)
        is_celestial_pack = np.zeros(2, dtype=np.bool_)
        
        for i, j in enumerate(self.shop_jokers):
            if i < 2:
                is_joker_on_shelf[i] = isinstance(j, Joker)
                is_planet_on_shelf[i] = isinstance(j, PlanetCard)
        
        for i, b in enumerate(self.boosters):
            if i < 2:
                is_buffoon_pack[i] = (b.name == "Buffoon") if hasattr(b, 'name') else False
                is_celestial_pack[i] = (b.name == "Celestial") if hasattr(b, 'name') else False
        
        current_pack_is_buffoon = False
        current_pack_is_celestial = False
        if self.in_pack_selection and hasattr(self, 'current_opened_pack_name'):
            current_pack_is_buffoon = (self.current_opened_pack_name == "Buffoon")
            current_pack_is_celestial = (self.current_opened_pack_name == "Celestial")
            
        pack_card_is_joker = np.zeros(5, dtype=np.bool_)
        pack_card_is_planet = np.zeros(5, dtype=np.bool_)
        pack_card_level_indices = np.full((5,), -1, dtype=np.int32)
        
        for i, card in enumerate(self.booster_contents):
            if isinstance(card, Joker):
                pack_card_is_joker[i] = True
            elif isinstance(card, PlanetCard):
                pack_card_is_planet[i] = True
                pack_card_level_indices[i] = self.hand_to_id.get(card.hand_type, -1)
                
        pack_cards_obs = BaseCard.observe_list(self.booster_contents, 5)
        
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
                deck_ranks,
                deck_suits,
                deck_enhancements,
                deck_editions,
                deck_seals,
                deck_size,
            ]
        )

        target_counts = np.zeros(self.max_booster_contents, dtype=np.int32)
        for i in range(len(self.booster_contents)):
            if isinstance(self.booster_contents[i], Consumable):
                target_counts[i] = self.booster_contents[i].num_targets

        return {
            "dollars": np.array(
                [np.clip(self.G.dollars, self.min_dollars, self.max_dollars)],
                dtype=np.float32,
            ),
            "deck_stats": deck_stats,
            "owned_joker_count": np.array([len(self.G.owned_jokers)], dtype=np.float32),
            
            "reroll_price": np.array([self.reroll_cost], dtype=np.float32),
            "shop_prices": shop_prices,
            "booster_prices": booster_prices,
            "goal": np.array([self.next_blind.chip_goal], dtype=np.float32),
            "hand_stats": HandType.observe_stats(self.G.hand_stats),
            "shop_cards": BaseCard.observe_list(self.shop_jokers, 2),
            "booster_cards": BaseCard.observe_list(self.boosters, 2),
            "owned_jokers": BaseCard.observe_list(self.G.owned_jokers, 5),
            "pack_cards": pack_cards_obs,
            
            "is_joker_on_shelf": is_joker_on_shelf.astype(np.int8),
            "is_planet_on_shelf": is_planet_on_shelf.astype(np.int8),
            "is_buffoon_pack": is_buffoon_pack.astype(np.int8),
            "is_celestial_pack": is_celestial_pack.astype(np.int8),
            
            "in_pack_selection": np.array([self.in_pack_selection], dtype=np.int8),
            "current_pack_is_buffoon": np.array([current_pack_is_buffoon], dtype=np.int8),
            "current_pack_is_celestial": np.array([current_pack_is_celestial], dtype=np.int8),
            
            "pack_card_is_joker": pack_card_is_joker.astype(np.int8),
            "pack_card_is_planet": pack_card_is_planet.astype(np.int8),
            "pack_card_level_indices": pack_card_level_indices,
            
            "blind_index": np.array([self.next_blind.index], dtype=np.int32),
            "round": np.array([self.round], dtype=np.int32),
        }

    def build_observation_space(self):
        bool_space_2 = sp.Box(0, 1, (2,), dtype=np.int8)
        bool_space_5 = sp.Box(0, 1, (5,), dtype=np.int8)
        bool_space_1 = sp.Box(0, 1, (1,), dtype=np.int8)
        
        return sp.Dict(
            {
                "dollars": sp.Box(low=self.min_dollars, high=self.max_dollars, shape=(1,), dtype=np.float32),
                "reroll_price": sp.Box(0, 1000, shape=(1,), dtype=np.float32),
                "shop_prices": sp.Box(0, 1000, shape=(2,), dtype=np.float32),
                "booster_prices": sp.Box(0, 1000, shape=(2,), dtype=np.float32),
                "owned_joker_count": sp.Box(0, 20, shape=(1,), dtype=np.float32),
                "goal": sp.Box(0, 1e18, shape=(1,), dtype=np.float32),
                
                "hand_stats": HandType.stats_obs_space(),
                "deck_stats": sp.Box(low=-200, high=200, shape=(37,), dtype=np.float32),
                
                "is_joker_on_shelf": bool_space_2,
                "is_planet_on_shelf": bool_space_2,
                "is_buffoon_pack": bool_space_2,
                "is_celestial_pack": bool_space_2,
                
                "in_pack_selection": bool_space_1,
                "current_pack_is_buffoon": bool_space_1,
                "current_pack_is_celestial": bool_space_1,
                
                "pack_card_is_joker": bool_space_5,
                "pack_card_is_planet": bool_space_5,
                "pack_card_level_indices": sp.Box(-1, 12, shape=(5,), dtype=np.int32),
                
                "shop_cards": BaseCard.observation_space(2),
                "booster_cards": BaseCard.observation_space(2),
                "owned_jokers": BaseCard.observation_space(5),
                "pack_cards": BaseCard.observation_space(5),
                
                "blind_index": sp.Box(0, 30, shape=(1,), dtype=np.int32),
                "round": sp.Box(0, 30, shape=(1,), dtype=np.int32),
            }
        )

    def can_pay(self, cost):
        have_cc = any(j.name == "Credit Card" for j in self.G.owned_jokers)
        return self.G.dollars >= cost or (have_cc and self.G.dollars - cost >= -25)

    def build_action_space(self):
        return sp.Discrete(17)

    def check_card_purchaseable(self, action):
        if action[0] == Actions.BUY_CARD:
            i = action[1][0] - 1
            if i >= len(self.shop_jokers):
                return False
            joker = self.shop_jokers[i]
            if isinstance(joker, Joker):
                if len(self.G.owned_jokers) >= self.G.joker_limit:
                    return False
            if joker.get_universal_index() == 0:
                return False  # padding card from real environment
            if not isinstance(joker, (Joker, PlanetCard)):
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

    def action_vector_to_action(self, action_idx):
        if action_idx == 0:
            return [Actions.END_SHOP]
        elif action_idx == 1:
            return [Actions.REROLL_SHOP]
        elif action_idx == 2:
            return [Actions.BUY_CARD, [1]]
        elif action_idx == 3:
            return [Actions.BUY_CARD, [2]]
        elif action_idx == 4:
            return [Actions.BUY_BOOSTER, [1]]
        elif action_idx == 5:
            return [Actions.BUY_BOOSTER, [2]]
        elif action_idx in range(6, 11):
            joker_slot = action_idx - 5
            return [Actions.SELL_JOKER, [joker_slot]]
        elif action_idx in range(11, 16):
            pack_slot = action_idx - 10
            return [Actions.SELECT_BOOSTER_CARD, [pack_slot], []]
        elif action_idx == 16:
            return [Actions.SKIP_BOOSTER_PACK]
        else:
            return [Actions.END_SHOP]

    def roll_jokers(self):
        self.shop_jokers = []
        # Generate a random joker and ensure it is unique and not already owned
        while len(self.shop_jokers) < 2:
            r = randint(1, 28)
            if r <= 20:
                joker = Joker.random(
                    unlocked_jokers=self.G.unlocked_jokers,
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
        if self.round == 2:
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
        
        is_legal = True
        if action[0] in [Actions.BUY_CARD, Actions.BUY_BOOSTER]:
            is_legal = self.check_card_purchaseable(action)
        elif action[0] == Actions.REROLL_SHOP:
            is_legal = self.can_pay(self.reroll_cost)
        elif action[0] == Actions.SELL_JOKER:
            is_legal = len(self.G.owned_jokers) >= action[1][0]
            
        if not is_legal:
            reward = self.illegal_action_reward
            print(f"DEBUG: Model picked ILLEGAL {action}.")
            action = [Actions.END_SHOP]
        else:
            reward = 0.005 if action[0] != Actions.END_SHOP else 0.0
        
        self.take_action(action)
        
        if is_legal and action[0] == Actions.BUY_BOOSTER:
            self.in_pack_selection = True
        elif action[0] in [Actions.SELECT_BOOSTER_CARD, Actions.SKIP_BOOSTER_PACK]:
            if self.booster_choices_left <= 0 or action[0] == Actions.SKIP_BOOSTER_PACK:
                self.in_pack_selection = False
                self.booster_contents = []
        
        obs = self.get_obs()
        done = (action[0] == Actions.END_SHOP) and (not self.in_pack_selection)

        return (
            obs,
            reward,
            done,
            False,
            {},
        )

    def take_action(self, action):
        if action[0] == Actions.BUY_CARD:
            purchased_joker = self.shop_jokers.pop(action[1][0] - 1)
            self.G.dollars -= purchased_joker.value
            if isinstance(purchased_joker, PlanetCard):
                purchased_joker.trigger(targets=[], gamestate=self.G)
            else:
                self.telemetry["jokers_purchased"] += 1
                purchased_joker.segment = BaseCard.Segments.JOKER
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
            self.current_opened_pack_name = booster.name
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
                        if 0 <= x - 1 and x - 1 < len(self.G.hand.cards):
                            targets.append(self.G.hand[x - 1])
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

    def enforce_segments(self):
        for x in self.shop_jokers:
            x.segment = BaseCard.Segments.SHOP_JOKER
        for x in self.G.owned_jokers:
            x.segment = BaseCard.Segments.JOKER
        for x in self.boosters:
            x.segment = BaseCard.Segments.SHOP_BUYABLE
        for x in self.booster_contents:
            x.segment = BaseCard.Segments.BOOSTER_CHOICE
        for x in self.G.hand:
            x.segment = BaseCard.Segments.HAND
