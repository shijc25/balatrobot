from random import choice, randint, random
from gym_envs.base_card import BaseCard
import numpy as np


class Card(BaseCard):
    SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
    RANKS = list(range(2, 15))

    def __init__(self, suit, value):
        super().__init__()
        self.suit = suit
        self.value = value
        self.seal = 0
        self.enhancement = 0
        self.edition = 0
        self.cost = 1
        self.hiker_chips = 0
        self.glass_breaking = False

    def get_enhancement(self):
        return self.enhancement

    def get_edition(self):
        return self.edition

    def get_seal(self):
        return self.seal

    def copy_from(self, other):
        self.suit = other.suit
        self.value = other.value
        self.seal = other.seal
        self.enhancement = other.enhancement
        self.edition = other.edition

    def suit_index(self):
        return self.SUITS.index(self.suit) if self.suit in self.SUITS else None

    def get_universal_index(self):
        if self.enhancement == BaseCard.Enhancements.STONE:
            return BaseCard.STONE_INDEX
        return 0  # Testing no universal index to see if suit/rank embeddings are better alone
        # if self.suit is None or self.value is None:
        #     return 0
        # return self.index() + self.FIRST_PLAYING_CARD_INDEX

    def get_u_rank_index(self):
        if self.enhancement == BaseCard.Enhancements.STONE:
            return 0
        return self.value - 1 if self.value in self.RANKS else 0

    def get_u_suit_index(self):
        if self.enhancement == BaseCard.Enhancements.WILD:
            return 5
        if self.enhancement == BaseCard.Enhancements.STONE:
            return 0
        return self.suit_index() + 1 if self.suit in self.SUITS else 0

    def chip_value(self):
        if self.enhancement == BaseCard.Enhancements.STONE:
            return 50
        if self.value == 14:
            return 11 + self.hiker_chips
        return min(self.value, 10) + self.hiker_chips

    def get_scalar_properties(self):
        p = np.zeros(4, dtype=np.float32)
        p[0] = (self.cost - 4) / 10
        p[1] = self.chip_value()
        return p

    def index(self):
        if self.suit is None or self.value is None:
            return 52

        suit_map = {
            "Clubs": 0,
            "Diamonds": 1,
            "Hearts": 2,
            "Spades": 3,
        }
        return suit_map[self.suit] * 13 + self.value - 2

    def is_face_card(self, pareidolia=False, jokers=[]):
        if not pareidolia and any(j.name == "Pareidolia" for j in jokers):
            return self.is_face_card(pareidolia=True)
        return pareidolia or self.value in [11, 12, 13]

    @staticmethod
    def from_gamestate_card(gamestate_card):
        value = gamestate_card.get("value", None)
        suit = gamestate_card.get("suit", None)
        if suit is None or value is None:
            return None
        try:
            value = int(value)
        except ValueError:
            value = {"Jack": 11, "Queen": 12, "King": 13, "Ace": 14}.get(value, value)

        return Card(suit, value)

    @staticmethod
    def index_to_card(index):
        if index < 0 or index >= 52:
            return None
        suit = Card.SUITS[index // 13]
        value = index % 13 + 2
        return Card(suit, value)

    def __str__(self):
        return f"{self.value} of {self.suit}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.suit == other.suit and self.value == other.value

    def __hash__(self):
        return hash((self.suit, self.value))

    def smeared_suit(self):
        if self.suit in ["Hearts", "Diamonds"]:
            return "Red"
        elif self.suit in ["Clubs", "Spades"]:
            return "Black"
        return None

    @staticmethod
    def random(vanilla_only=False):
        c = Card(choice(Card.SUITS), choice(Card.RANKS))
        if vanilla_only:
            return c
        edition_r = random()
        if edition_r < 0.012:
            c.edition = BaseCard.Editions.POLYCHROME
        elif edition_r < 0.040:
            c.edition = BaseCard.Editions.HOLOGRAPHIC
        elif edition_r < 0.080:
            c.edition = BaseCard.Editions.FOIL

        if random() < 0.40:
            c.enhancement = randint(1, BaseCard.num_enhancements - 1)

        if random() < 0.2:
            c.seal = randint(1, BaseCard.num_seals - 1)

        return c

    @staticmethod
    def random_flush(hand):
        flush_suit = choice(Card.SUITS)
        if len(hand) > 0:
            suit_counts = {suit: 0 for suit in Card.SUITS}
            for card in hand:
                suit_counts[card.suit] += 1
            flush_suit = max(suit_counts, key=suit_counts.get)
        return Card(flush_suit, choice(Card.RANKS))

    @staticmethod
    def random_straight(hand):
        valid_ranks = []
        for card in hand:
            if card.value == 2:
                valid_ranks.append(14)
            else:
                valid_ranks.append(card.value - 1)

            if card.value == 14:
                valid_ranks.append(2)
            else:
                valid_ranks.append(card.value + 1)
        if len(hand) == 0:
            return Card.random()

        for card in hand:
            valid_ranks = [x for x in valid_ranks if x != card.value]
        return Card(choice(Card.SUITS), choice(valid_ranks))

    @staticmethod
    def random_straight_flush(hand):
        flush = Card.random_flush(hand)
        straight = Card.random_straight(hand)

        return Card(flush.suit, straight.value)

    # For pair, Three of a Kind, and four of a kind
    @staticmethod
    def random_dupe(hand):
        if len(hand) == 0:
            return Card.random()
        dupe_rank = choice([x.value for x in hand if x.value in Card.RANKS])

        return Card(choice(Card.SUITS), dupe_rank)

    @staticmethod
    def random_two_pair(hand):
        if len(hand) == 0:
            return Card.random()
        multiples = hand.multiples()

        # We want to return a dupe of any card that is not in a pair
        non_pair_hand = [x.value for x in hand if x.value not in multiples]
        if len(non_pair_hand) == 0:
            rank = choice([x for x in Card.RANKS if x not in multiples])
            return Card(choice(Card.SUITS), rank)
        return Card(choice(Card.SUITS), choice(non_pair_hand))

    @staticmethod
    def random_full_house(hand):
        if len(hand) == 0:
            return Card.random()
        multiples = hand.multiples()
        triple_ranks = [k for k, v in multiples.items() if v > 2]
        non_triple_ranks = set(Card.RANKS) - set(triple_ranks)
        if len(non_triple_ranks) == 0:
            return Card.random()
        double_ranks = [k for k, v in multiples.items() if v == 2]
        if len(triple_ranks) == 0:
            if len(double_ranks) != 0:
                return Card(choice(Card.SUITS), choice(double_ranks))
            else:
                return Card.random_dupe(hand)
        return Card(choice(Card.SUITS), choice(list(non_triple_ranks)))
