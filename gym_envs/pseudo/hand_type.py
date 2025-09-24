from gymnasium import spaces as sp
import numpy as np


class HandType:
    def __init__(self, name, mult, chips, mult_inc, chip_inc, i):
        self.name = name
        self.index = i
        self.chips = chips
        self.mult = mult
        self.chip_inc = chip_inc
        self.mult_inc = mult_inc
        self.level = 1
        self.play_count = 0
        self.played_this_blind = False
        self.planet_used_this_run = False

    def scores(self):
        return self.chips, self.mult

    def get_value(self):
        return self.mult * self.chips

    def get_level(self):
        return self.level

    def add_level(self, x, force=False, planet=False):
        if x == 0:
            return

        if x < 0 and not force:
            if self.level == 1:
                return

        self.level += x
        if self.level < 1:
            self.level = 0
            self.chips = 0
            self.mult = 0
        self.chips += self.chip_inc * x
        self.chips = max(0, self.chips)
        self.mult += self.mult_inc * x
        self.mult = max(0, self.mult)

    def set_level(self, x, force=False):
        if x < 1 and not force:
            return

        if x == self.level:
            return

        diff = x - self.level
        self.add_level(diff, force=force)

    @staticmethod
    def observe_stats(stats):
        levels = np.ones(12, dtype=np.float32)
        chips = np.zeros(12, dtype=np.float32)
        mults = np.zeros(12, dtype=np.float32)
        played_count = np.zeros(12, dtype=np.float32)
        played_this_blind = np.zeros(12, dtype=np.float32)
        for hand in stats.values():
            index = hand.index
            levels[index] = hand.get_level()
            chips[index] = hand.chips / 100.0
            mults[index] = hand.mult / 10.0
            played_count[index] = hand.play_count / 10.0
            played_this_blind[index] = 1.0 if hand.played_this_blind else 0.0

        return {
            "level": levels,
            "chips": chips,
            "mult": mults,
            "played_count": played_count,
            "played_this_blind": played_this_blind,
        }

    @staticmethod
    def stats_obs_space():
        return sp.Dict(
            {
                "level": sp.Box(0, 50, (12,), np.float32),
                "chips": sp.Box(-1, 20, (12,), np.float32),
                "mult": sp.Box(-1, 20, (12,), np.float32),
                "played_count": sp.Box(0, 30, (12,), np.float32),
                "played_this_blind": sp.Box(0, 1, (12,), np.float32),
            }
        )

    @staticmethod
    def all_hands():
        hands = [
            HandType("High Card", 1, 5, 1, 10, 0),
            HandType("Pair", 2, 10, 1, 15, 1),
            HandType("Two Pair", 2, 20, 1, 20, 2),
            HandType("Three of a Kind", 3, 30, 2, 20, 3),
            HandType("Straight", 4, 30, 3, 30, 4),
            HandType("Flush", 4, 35, 2, 15, 5),
            # HandType("Flush", 5, 40, 2, 15, 5), # stronger flush to see if it helps learn them early
            HandType("Full House", 4, 40, 2, 25, 6),
            HandType("Four of a Kind", 7, 60, 3, 30, 7),
            HandType("Straight Flush", 8, 100, 4, 40, 8),
            HandType("Five of a Kind", 12, 120, 3, 35, 9),
            HandType("Flush House", 14, 140, 4, 40, 10),
            HandType("Flush Five", 16, 160, 3, 50, 11),
        ]

        return {x.name: x for x in hands}
