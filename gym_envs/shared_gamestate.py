from gym_envs.base_card import BaseCard
from gym_envs.blind import Blind
from gym_envs.joker import Joker
from gym_envs.components.deck import Deck
from gym_envs.components.hand import Hand
from gym_envs.components.hand_type import HandType


# For managing aspects of the game state that are similar between blind, shop, etc.
class SharedGamestate:
    def __init__(self):
        self.owned_jokers = []
        self.dollars = 0
        self.hand_stats = HandType.all_hands()
        self.deck = Deck()
        self.hand = Hand()
        self.joker_limit = 5
        self.tarot_used = 0
        self.consumables = []
        self.max_consumables = 2
        self.max_hand_size = 12
        self.hand_size_mod = 0
        self.current_blind = Blind.random(round=1)
        self.unlocked_jokers = set()

    @property
    def current_joker_limit(self):
        return self.joker_limit

    @property
    def current_consumable_size(self):
        return self.max_consumables

    @property
    def current_hand_size(self):
        # return 3
        size = 8
        for j in self.owned_jokers:
            if j.name == "Merry Andy":
                size -= 1
            elif j.name == "Stuntman":
                size -= 2
            elif j.name == "Juggler":
                size += 1
            elif j.name == "Troubadour":
                size += 2
        if self.current_blind.name == "Manacle":
            size -= 1

        size += self.hand_size_mod
        return max(min(size, self.max_hand_size), 1)

    def add_consumable(self, consumable):
        if len(self.consumables) < self.max_consumables:
            self.consumables.append(consumable)

    def add_joker(self, joker, fix_price=True):
        if len(self.owned_jokers) < self.joker_limit:
            if fix_price:
                joker.value = max(int(joker.value / 2.0), 1)

            # We can't allow the agent to actually have an unimplemented joker, so we act as if it was immediately sold
            if type(joker) == BaseCard:
                self.dollars += joker.value
            else:
                self.owned_jokers.append(joker)

    def destroy_card(self, card):
        if isinstance(card, Joker):
            if card in self.owned_jokers and card.seal != BaseCard.Seals.ETERNAL:
                self.owned_jokers.remove(card)
        else:
            if card in self.hand.cards:
                self.hand.cards.remove(card)
            if card in self.deck.all_cards:
                self.deck.all_cards.remove(card)
            if card.is_face_card(jokers=self.owned_jokers):
                for j in self.owned_jokers:
                    if j.name == "Canio":
                        j.state["mult_mult"] += 1.0

    def update_unlocked_jokers(self):
        for enh, joker in {
            BaseCard.Enhancements.GLASS: "Glass Joker",
            BaseCard.Enhancements.STEEL: "Steel Joker",
            BaseCard.Enhancements.STONE: "Stone Joker",
            BaseCard.Enhancements.LUCKY: "Lucky Cat",
            BaseCard.Enhancements.GOLD: "Golden Ticket",
        }.items():
            if joker not in self.unlocked_jokers:
                if any(c.enhancement == enh for c in self.deck.all_cards):
                    self.unlocked_jokers.add(joker)

    def create_card(self, card, no_hand=False):
        if not no_hand:
            self.hand.cards.append(card)
        self.deck.all_cards.append(card)

        for j in self.owned_jokers:
            if j.name == "Hologram":
                j.state["mult_mult"] += 0.25
        self.update_unlocked_jokers()

    def use_planet(self, planet):
        for j in self.owned_jokers:
            if j.name == "Constellation":
                j.state["mult_mult"] += 0.1
        self.hand_stats[planet.hand_type].add_level(1, planet=True)

    def black_hole(self):
        for x in self.hand_stats.values():
            x.add_level(1)
