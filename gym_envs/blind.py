from random import choice, randint


class Blind:
    ante_base_chips = [100, 300, 800, 2000, 5000, 11000, 20000, 35000, 50000]
    green_stake_chips = [100, 300, 900, 2600, 8000, 20000, 36000, 60000, 100000]
    purple_stake_chips = [100, 300, 100, 3200, 9000, 25000, 60000, 110000, 200000]
    boss_blinds = [
        ("The Club", 0, 2),
        ("The Goad", 0, 2),
        ("The Window", 0, 2),
        ("The Head", 0, 2),
        ("The Hook", 0, 2),
        ("The Psychic", 0, 2),
        ("The Manacle", 0, 2),  # -1 hand size
        # ("The Pillar", 0, 2), # debuff previously played
        ("The House", 2, 2),  # First hand face down
        ("The Wall", 2, 4),
        ("The Arm", 2, 2),  # Decrease level of played hand
        ("The Fish", 2, 2),  # Cards drawn after playing hands are face down
        ("The Water", 2, 2),  # 0 discards
        ("The Mouth", 2, 2),  # Only 1 hand type allowed
        ("The Needle", 2, 1),  # Only 1 hand
        ("The Flint", 2, 2),  # Base chips and mult halved
        ("The Mark", 2, 2),  # All face cards face down
        ("The Eye", 3, 2),  # No repeat hand types
        # ("The Tooth", 3, 2), # lose $1 per card played
        ("The Plant", 4, 2),  # Face cards debuffed
        # ("The Serpent", 5, 2),  # Always draw 3 (ignore hand size)
        # ("The Ox", 6, 2), # Playing your most played hand sets money to 0
    ]

    finisher_blinds = [
        ("Amber Acorn", 8, 2),  # Flip and shuffle all jokers
        # ("Verdant Leaf", 8, 2), # All cards debuffed until 1 joker sold
        ("Violet Vessel", 8, 6),
        ("Crimson Heart", 8, 2),  # Random joker disabled every hand
        # ("Cerulean Bell", 8, 2) # 1 card always force selected
    ]

    def __init__(self, name, round, chip_goal, index=None, reward=0):
        self.name = name
        self.round = round
        self.chip_goal = chip_goal
        self.index = index if index is not None else self.lookup_index()
        self.gold_reward = reward

    def is_boss(self):
        return self.index >= 2

    @staticmethod
    def all_blind_names():
        names = ["Small Blind", "Big Blind"]
        names.extend([boss[0] for boss in Blind.boss_blinds])
        names.extend([finisher[0] for finisher in Blind.finisher_blinds])
        return names

    @staticmethod
    def base_chips_for_stake(stake=0):
        if stake >= 5:
            return Blind.purple_stake_chips
        elif stake >= 2:
            return Blind.green_stake_chips
        else:
            return Blind.ante_base_chips

    @staticmethod
    def estimate_chips_for_round(round, stake=0):
        base_chip_table = Blind.base_chips_for_stake(stake)

        ante = (round - 1) // 3 + 1
        base_chips = Blind.ante_base_chips[min(ante, len(Blind.ante_base_chips) - 1)]
        if ante > len(Blind.ante_base_chips) - 1:
            base_chips *= 10 ** (ante - len(Blind.ante_base_chips) + 1)

        if round % 3 == 0:
            return base_chips * 2
        elif round % 3 == 1:
            return base_chips * 1.5
        else:
            return base_chips

    @staticmethod
    def from_gamestate(gamestate):
        name = gamestate["current_round"]["blind_name"]
        round = gamestate["round"]
        ante = (round - 1) // 3 + 1
        base_chips = Blind.ante_base_chips[min(ante, len(Blind.ante_base_chips) - 1)]
        if ante > len(Blind.ante_base_chips) - 1:
            base_chips *= 10 ** (ante - len(Blind.ante_base_chips) + 1)

        if name == "Small Blind":
            return Blind("Small Blind", round, base_chips, 0)
        elif name == "Big Blind":
            return Blind("Big Blind", round, base_chips * 1.5, 1)
        else:
            for i, (boss_name, min_ante, multiplier) in enumerate(Blind.boss_blinds):
                if name == boss_name:
                    return Blind(boss_name, round, base_chips * multiplier, i + 2)
            for i, (finisher_name, _, _) in enumerate(Blind.finisher_blinds):
                if name == finisher_name:
                    return Blind(
                        finisher_name,
                        round,
                        base_chips * 2,
                        i + 2 + len(Blind.boss_blinds),
                    )
        print("Warning: Blind not found in gamestate:", name)
        return Blind("Big Blind", round, base_chips * 2, 1)

    def lookup_index(self):
        if self.name == "Small Blind":
            return 0
        elif self.name == "Big Blind":
            return 1
        else:
            for i, (boss_name, min_ante, _) in enumerate(Blind.boss_blinds):
                if self.name == boss_name:
                    return i + 2
        for i, (finisher_name, _, _) in enumerate(Blind.finisher_blinds):
            if self.name == finisher_name:
                return i + 2 + len(Blind.boss_blinds)
        return -1

    def filter_scored_cards(self, scored_cards):
        suit_map = {
            "The Club": "Clubs",
            "The Goad": "Spades",
            "The Window": "Diamonds",
            "The Head": "Hearts",
        }

        if self.name in suit_map:
            suit = suit_map[self.name]
            scored_cards = [card for card in scored_cards if card.suit != suit]

        elif self.name == "The Plant":
            scored_cards = [
                card for card in scored_cards if card.value not in (11, 12, 13)
            ]

        return scored_cards

    @staticmethod
    def random(round, stake=0):
        orig_round = round
        ante = (round - 1) // 3 + 1
        round = (round - 1) % 3
        base_chips_table = Blind.base_chips_for_stake(stake)
        base_chips = base_chips_table[min(ante, len(base_chips_table) - 1)]
        if ante > len(base_chips_table) - 1:
            base_chips *= 10 ** (ante - len(base_chips_table) + 1)

        if round == 0:
            # Small blind
            return Blind(
                "Small Blind", orig_round, base_chips, 0, reward=3 if stake <= 0 else 0
            )
        elif round == 1:
            # Big blind
            return Blind("Big Blind", orig_round, base_chips * 1.5, 1, reward=4)
        else:
            # Boss blind
            if ante % 8 == 0:
                possible_blinds = Blind.finisher_blinds
            else:
                possible_blinds = [x for x in Blind.boss_blinds if x[1] <= ante]
            if not possible_blinds:
                return Blind("Boss Blind", orig_round, base_chips * 2, reward=8)
            boss_blind_index = randint(0, len(possible_blinds) - 1)
            boss_blind = possible_blinds[boss_blind_index]
            name, min_ante, multiplier = boss_blind
            return Blind(
                name,
                orig_round,
                base_chips * multiplier,
                index=boss_blind_index + 2,
                reward=5,
            )
