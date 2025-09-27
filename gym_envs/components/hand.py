from gym_envs.components.card import Card
from random import shuffle, choice, choices, sample, randint
from copy import deepcopy
from gym_envs.base_card import BaseCard


class Hand:
    def __init__(self, cards=[]):
        self.cards = cards[:]

    @staticmethod
    def from_gamestate_hand(gamestate_hand):
        cards = []
        for card in gamestate_hand:
            cards.append(Card.from_gamestate_card(card))
        return Hand(cards)

    def add_card(self, card):
        self.cards.append(card)

    def pop_cards(self, indices):
        popped = []
        for i in sorted(indices, reverse=True):
            if i < 1 or i > len(self.cards):
                print(
                    "Warning: Attempted to pop card at index",
                    i,
                    "from hand of size",
                    len(self.cards),
                )
                continue
            popped.append(self.cards.pop(i - 1))
        hand = Hand()
        hand.cards = popped
        return hand

    def sort(self):
        self.cards = list(sorted(self.cards, key=lambda x: x.value))

    def shuffle(self):
        shuffle(self.cards)

    def suit_counts(self):
        suit_counts = {suit: 0 for suit in Card.SUITS}
        for card in self.cards:
            suit_counts[card.suit] += 1
        return suit_counts

    def suit_homogeneity(self):
        suit_counts = self.suit_counts()
        if len(self.cards) < 4:
            return 0.0
        if len(self.cards) == 4:
            return (max(suit_counts.values()) - 1) / 3
        if len(self.cards) == 5:
            return (
                max(suit_counts.values()) - 2
            ) / 3  # Can't have a max count of 1 with 5 cards, so min is 2
        if len(self.cards) == 8:
            return (max(suit_counts.values()) - 2) / 6
        return max(suit_counts.values()) / len(self.cards)

    # Change the ranks and suits of the cards in the hand randomly
    # Maintains the number of cards, and the hand type
    # (Does not currently handle some jokers)
    def mutate(self):
        hand = deepcopy(self)
        hand_type, scored_cards = hand.evaluate()
        if hand_type in ["Straight Flush", "Straight"]:
            bounds = min(x.value for x in scored_cards.cards), max(
                x.value for x in scored_cards.cards
            )
            if bounds[0] == 2 and bounds[1] == 14:
                bump_range = range(
                    1, 10
                )  # Ace is low, so we can bump up to 9 (making it a royal straight)
            else:
                bump_range = range(2 - bounds[0], 14 - bounds[1] + 1)

            # Never keep the rank unchanged
            bump_range = [x for x in bump_range if x != 0]
            bump = choice(bump_range)
            for card in hand.cards:
                card.value += bump
        else:
            # We can freely change the ranks as long as any multiples are maintained, so we can use a random mapping
            reordered_ranks = list(Card.RANKS)
            shuffle(reordered_ranks)
            rank_mapping = {x: y for x, y in zip(Card.RANKS, reordered_ranks)}
            for card in scored_cards.cards:
                if card.value is not None:
                    card.value = rank_mapping[card.value]

        # Change the suits randomly
        reordered_suits = list(Card.SUITS)
        shuffle(reordered_suits)
        suit_mapping = {x: y for x, y in zip(Card.SUITS, reordered_suits)}
        for card in hand.cards:
            if card.value is not None:
                card.suit = suit_mapping[card.suit]
        return hand

    def card_dupe_counts(self):
        rank_counts = []
        suit_counts = []
        for card in self.cards:
            if card.value is None:
                rank_counts.append(0)
                suit_counts.append(0)
                continue
            rank_counts.append(len([x for x in self.cards if x.value == card.value]))
            suit_counts.append(len([x for x in self.cards if x.suit == card.suit]))
        return rank_counts, suit_counts

    def card_run_counts(self, suited=False):
        ranks = {
            card.value
            for card in self.cards
            if card.enhancement != BaseCard.Enhancements.STONE
        }
        run_counts = []
        for card in self.cards:
            if card.value is None:
                run_counts.append(0)
                continue
            if suited:
                ranks = {c.value for c in self.cards if c.suit == card.suit}
            run_up = 0
            run_down = 0
            v = card.value
            low_v = v
            if v == 14:
                low_v = 1
            for i in range(low_v + 1, 15):
                if i in ranks:
                    run_up += 1
                else:
                    break

            for i in range(v - 1, 0, -1):
                if i in ranks:
                    run_down += 1
                elif i == 1 and 14 in ranks:
                    run_down += 1
                else:
                    break
            run_counts.append(run_up + run_down + 1)
        return run_counts

    @staticmethod
    def random(hand_type, size=8):
        hand = Hand.random_prototype(hand_type)
        suit_counts = hand.suit_counts()
        banned_ranks = set([card.value for card in hand.cards])
        while len(hand.cards) < size:
            suit = choice(Card.SUITS)
            if suit_counts[suit] >= 4:
                continue
            value = choice([x for x in Card.RANKS if x not in banned_ranks])
            test_hand = Hand(hand.cards + [Card(suit, value)])
            if hand_type not in ["Straight", "Straight Flush"]:
                if test_hand.longest_run() >= 5:
                    continue
            hand.add_card(Card(suit, value))

            suit_counts[suit] += 1
            banned_ranks.add(value)
        hand.shuffle()
        return hand

    @staticmethod
    def random_prototype(hand_type):
        cards = []  # high card prototype
        if hand_type in ["Pair", "Three of a Kind", "Four of a Kind"]:
            dupe_count = {"Pair": 2, "Three of a Kind": 3, "Four of a Kind": 4}[
                hand_type
            ]
            value = choice(Card.RANKS)
            suits = choices(Card.SUITS, k=dupe_count)
            cards = [Card(suit, value) for suit in suits]
        elif hand_type in ["Two Pair", "Full House"]:
            first_size = {"Two Pair": 2, "Full House": 3}[hand_type]
            value1 = choice(Card.RANKS)
            value2 = choice([x for x in Card.RANKS if x != value1])
            suits = choices(Card.SUITS, k=2 + first_size)
            cards = [Card(suit, value1) for suit in suits[:2]]
            cards += [Card(suit, value2) for suit in suits[2 : 2 + first_size]]
            # Make sure we don't have a flush
            if len(set(suit for suit in suits)) == 1:
                suits = sample([suit for suit in Card.SUITS if suit not in suits], k=1)
                cards[randint(0, len(cards) - 1)].suit = suits[0]
        elif hand_type in ["Straight", "Straight Flush"]:
            start_value = randint(1, 10)  # include ace on low end
            end_value = start_value + 4
            suits = choices(Card.SUITS, k=5)
            cards = [
                Card(suit, value if value > 1 else 14)
                for suit, value in zip(suits, range(start_value, end_value + 1))
            ]
            if hand_type == "Straight Flush":
                # Make sure all suits are the same
                suit = choice(suits)
                for card in cards:
                    card.suit = suit
            else:
                # Make sure we don't have a flush
                if len(set(suit for suit in suits)) == 1:
                    suits = sample(
                        [suit for suit in Card.SUITS if suit not in suits], k=1
                    )
                    cards[randint(0, len(cards) - 1)].suit = suits[0]
        elif hand_type == "Flush":
            suit = choice(Card.SUITS)
            values = sample(Card.RANKS, k=5)
            cards = [Card(suit, value) for value in values]
            while Hand(cards).longest_run() >= 5:
                # If we accidentally made a straight, change one of the cards
                i = randint(0, len(cards) - 1)
                new_value = choice([x for x in Card.RANKS if not x in values])
                cards[i] = Card(suit, new_value)
        elif hand_type == "High Card":
            cards = [Card(choice(Card.SUITS), choice(Card.RANKS))]
        return Hand(cards)

    def multiples(self):
        multiples = {}
        for card in self.cards:
            if card.enhancement == BaseCard.Enhancements.STONE:
                continue
            if card.value in multiples:
                multiples[card.value] += 1
            else:
                multiples[card.value] = 1

        multiples = {k: v for k, v in multiples.items() if v > 1}
        return multiples

    def contained_hand_types(
        self, allow_4_flush=False, allow_4_straight=False, smeared=False
    ):
        available = set()
        multiples = self.multiples()
        multiple_counts = list(multiples.values())
        if len(self.cards) > 0:
            available.add("High Card")
        if len(multiples) > 0:
            available.add("Pair")
            if len(multiples) > 1:
                available.add("Two Pair")
                if any([x >= 3 for x in multiple_counts]):
                    available.add("Full House")
            if any([x >= 3 for x in multiple_counts]):
                available.add("Three of a Kind")
            if any([x >= 4 for x in multiple_counts]):
                available.add("Four of a Kind")

        run = self.longest_run()
        if run >= 5:
            available.add("Straight")
        elif allow_4_straight and run >= 4:
            available.add("Straight")
        for suit in (Card.SUITS + ["Red", "Black"]) if smeared else Card.SUITS:
            suit_hand = Hand(
                [
                    card
                    for card in self.cards
                    if card.suit == suit
                    or card.smeared_suit() == suit
                    or card.enhancement == BaseCard.Enhancements.WILD
                ]
            )
            if len(suit_hand) >= 5:
                available.add("Flush")
            elif allow_4_flush and len(suit_hand) >= 4:
                available.add("Flush")
        if "Flush" in available and "Straight" in available:
            available.add("Straight Flush")

        return available

    def evaluate(self, allow_4_flush=False, allow_4_straight=False, smeared=False):
        contained_hands = self.contained_hand_types(
            allow_4_flush=allow_4_flush,
            allow_4_straight=allow_4_straight,
            smeared=smeared,
        )
        hand = "High Card"  # Default to high card if no other hand is found, should only happen if self.cards is empty, but that can happen in some edge cases
        for hand_type in [
            "Straight Flush",
            "Four of a Kind",
            "Full House",
            "Flush",
            "Straight",
            "Three of a Kind",
            "Two Pair",
            "Pair",
            "High Card",
        ]:
            if hand_type in contained_hands:
                hand = hand_type
                break
        if hand in ["Straight Flush", "Flush", "Straight", "Full House"]:
            return hand, Hand(self.cards)  # , Hand([])
        multiples = self.multiples()
        if hand == "High Card":
            if len(self.cards) == 0:
                return hand, Hand([])
            high_card = max(self.cards, key=lambda x: x.value)
            return (
                hand,
                Hand([high_card]),
                # Hand([x for x in self.cards if x.value != high_card.value]),
            )

        return (
            hand,
            Hand(
                [
                    x
                    for x in self.cards
                    if x.value in multiples
                    and x.enhancement != BaseCard.Enhancements.STONE
                ]
            ),
            # Hand([x for x in self.cards if x.value not in multiples]),
        )

    def longest_run(self):
        ranks = {
            card.value
            for card in self.cards
            if card.enhancement != BaseCard.Enhancements.STONE
        }
        if 14 in ranks:
            ranks.add(1)
        rank_hand = sorted(list(ranks))
        highest_run = 1
        run = 1
        for i in range(1, len(rank_hand)):
            if rank_hand[i] - rank_hand[i - 1] == 1:
                run += 1
            else:
                highest_run = max(highest_run, run)
                run = 1
        highest_run = max(highest_run, run)

        return highest_run

    def general_biaser(self):
        rank_counts = {value: 0 for value in Card.RANKS}
        suit_counts = {suit: 0 for suit in Card.SUITS}
        for card in self.cards:
            rank_counts[card.value] += 1
            suit_counts[card.suit] += 1

        def calculate_bias(card):
            return rank_counts[card.value] + suit_counts[card.suit] * 2

        return calculate_bias

    def rank_biaser(self):
        rank_counts = {value: 0 for value in Card.RANKS}
        for card in self.cards:
            rank_counts[card.value] += 1

        def calculate_bias(card):
            return rank_counts[card.value] * 15

        return calculate_bias

    def suit_biaser(self):
        suit_counts = {suit: 0 for suit in Card.SUITS}
        for card in self.cards:
            suit_counts[card.suit] += 1

        def calculate_bias(card):
            return suit_counts[card.suit] * 10

        return calculate_bias

    def straight_biaser(self):
        adjacency_counts = {value: 0 for value in Card.RANKS}
        for card in self.cards:
            if card.value == 14:
                # adjacency_counts[2] += 1
                pass
            else:
                adjacency_counts[card.value + 1] += 1

            if card.value == 2:
                # adjacency_counts[14] += 1
                pass
            else:
                adjacency_counts[card.value - 1] += 1

        # We don't want to draw duplicates when we have a partial straight
        for card in self.cards:
            adjacency_counts[card.value] = 0

        def calculate_bias(card):
            return adjacency_counts[card.value] * 15

        return calculate_bias

    def straight_flush_biaser(self):
        adjacency_counts = {
            (suit, value): 0 for value in Card.RANKS for suit in Card.SUITS
        }
        for card in self.cards:
            if card.value == 14:
                # adjacency_counts[(card.suit, 2)] += 1
                pass
            else:
                adjacency_counts[(card.suit, card.value + 1)] += 1

            if card.value == 2:
                # adjacency_counts[(card.suit, 14)] += 1
                pass
            else:
                adjacency_counts[(card.suit, card.value - 1)] += 1

        # We don't want to draw duplicates when we have a partial straight
        for card in self.cards:
            adjacency_counts[(card.suit, card.value)] = 0

        def calculate_bias(card):
            return adjacency_counts[(card.suit, card.value)] * 10

        return calculate_bias

    def __str__(self):
        return f"Hand: {self.cards}"

    def __len__(self):
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, key):
        return self.cards[key]
