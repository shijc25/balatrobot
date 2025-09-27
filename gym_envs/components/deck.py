from gym_envs.components.card import Card
from random import choice, randint, choices, sample, shuffle


class Deck:
    def __init__(self, infinite=False):
        self.remaining_cards = []
        self.infinite = infinite
        self.all_cards = [
            Card(suit, value) for suit in Card.SUITS for value in Card.RANKS
        ]
        self.reset()

    @staticmethod
    def from_gamestate_deck(gamestate_deck):
        deck = Deck(infinite=False)
        deck.remaining_cards = [
            Card.from_gamestate_card(card) for card in gamestate_deck
        ]
        deck.all_cards = deck.remaining_cards.copy()
        return deck

    def reset(self):
        self.remaining_cards = self.all_cards.copy()
        shuffle(self.remaining_cards)

    def add_card(self, card):
        if self.infinite:
            return
        self.all_cards.append(card)

    def draw(self):
        if self.infinite:
            return choice(self.all_cards)
        if len(self.remaining_cards) == 0:
            return None
        i = randint(0, len(self.remaining_cards) - 1)
        return self.remaining_cards.pop(i)

    def draw_biased(self, bias_calculator, bias):
        weights = [1] * len(self.remaining_cards)
        for i, card in enumerate(self.remaining_cards):
            weights[i] += ((bias * bias_calculator(card) + 1) ** (3)) - 1
        i = choices(range(len(self.remaining_cards)), weights=weights)[0]
        return self.remaining_cards.pop(i)

    def __len__(self):
        return len(self.remaining_cards)

    def __str__(self):
        return str(self.remaining_cards)


class ErraticDeck(Deck):
    def __init__(self, infinite=False):
        super().__init__(infinite=infinite)
        self.all_cards = [Card.random(vanilla_only=True) for _ in range(52)]
        self.reset()
