from copy import deepcopy
from gymnasium import spaces as sp
import numpy as np
from random import randint, choice, random
import torch
from gym_envs.base_card import BaseCard


class Joker(BaseCard):
    _implemented_mask = None

    def __init__(
        self,
        name,
        index,
        value=1,
        rarity=0,
        state={},
        rank_affinity=[],
        suit_affinity=[],
        hand_affinity=[],
    ):
        super().__init__(segment=BaseCard.Segments.JOKER)
        self.name = name
        self.index = index
        self.value = value
        self.rarity = rarity
        self.state = state
        self.edition = 0
        self.rank_affinity = rank_affinity
        self.suit_affinity = suit_affinity
        self.hand_affinity = hand_affinity
        self.copy_priority = 0
        self.seal = BaseCard.Seals.NO_SEAL  # Using seals as stickers

    def get_universal_index(self):
        if self.index == 0:
            return 0
        return self.index + BaseCard.FIRST_JOKER_INDEX - 1

    def get_edition(self):
        return self.edition

    def get_seal(self):
        return self.seal

    def get_scalar_properties(self):
        return np.array(
            [
                (self.value - 4) / 10.0,
                self.state.get("chips", 0) / 100.0,
                self.state.get("mult", 0) / 10.0,
                self.state.get("mult_mult", 0) / 2.0,
            ],
            dtype=np.float32,
        )

    def copyable(self):
        return self.name not in [
            "Blueprint",
            "Brainstorm",
            "Four Fingers",
            "Credit Card",
            "Chaos the Clown",
            "Delayed Gratification",
            "Pareidolia",
            "Egg",
            "Splash",
            "Sixth Sense",
            "Shortcut",
            "Cloud 9",
            "Rocket",
            "Midas Mask",
            "Gift Card",
            "Turtle Bean",
            "To the Moon",
            "Juggler",
            "Drunkard",
            "Golden Joker",
            "Trading Card",
            "Mr. Bones",
            "Troubadour",
            "Smeared Joker",
            "Showman",
            "Merry Andy",
            "Oops! All 6s",
            "Invisible Joker",
            "Satellite",
            "Astronomer",
            "Chicot",
        ]

    @staticmethod
    def from_gamestate_card(gamestate_card):
        label = gamestate_card.get("label", None)
        if label is None:
            return None
        if label not in label_lookup:
            return None
        joker = deepcopy(label_lookup[label])
        joker.value = gamestate_card.get("cost", joker.value)
        ability = gamestate_card.get("ability", {})
        if ability:
            joker.state = {
                "chips": ability.get("t_chips", 0),
                "mult": max([ability.get(x, 0) for x in ["mult", "h_mult", "t_mult"]]),
                "mult_mult": ability.get("x_mult", 0),
            }
        return joker

    # For masking aux loss etc to ignore jokers that are not yet emulated
    @staticmethod
    def implemented_mask():
        if Joker._implemented_mask is not None:
            return Joker._implemented_mask
        implemented_mask = torch.zeros(151, dtype=torch.bool)
        implemented_mask[0] = True  # Index 0 is always implemented (no joker)
        for joker in jokers:
            if joker.index > 0:
                implemented_mask[joker.index] = True
        Joker._implemented_mask = implemented_mask
        return implemented_mask

    @staticmethod
    def random(
        force_rarity=None,
        unlocked_jokers=set(),
        sparse_pool=True,
        ignore_rarity=False,
        stake=0,  # Only for determining sticker pool
    ):
        locked_jokers = {
            "Glass Joker",
            "Steel Joker",
            "Cavendish",
            "Stone Joker",
            "Lucky Cat",
            "Golden Ticket",
        }

        locked_jokers = locked_jokers - unlocked_jokers

        if force_rarity is not None:
            rarity = force_rarity
        else:
            rarity_p = random()
            if rarity_p < 0.7:
                rarity = 0
            elif rarity_p < 0.95:
                rarity = 1
            else:
                rarity = 2

        possible_jokers = [
            joker
            for joker in jokers
            if (joker.rarity == rarity or ignore_rarity)
            and joker.name not in locked_jokers
        ]

        if sparse_pool:
            # If we are in a sparse pool, we check how many of this rarity have been implemented vs the full pool
            full_counts = [61, 64, 20, 5]
            implemented_count = sum(
                1 for joker in possible_jokers if joker.rarity == rarity
            )
            hit_rate = implemented_count / full_counts[rarity]
            if random() > hit_rate:
                # If we are above the hit rate, we return a null joker
                return BaseCard()

        stickers = []
        if stake >= 3:
            stickers.append(BaseCard.Seals.ETERNAL)
        if stake >= 6:
            stickers.append(BaseCard.Seals.PERISHABLE)
        if stake >= 7:
            stickers.append(BaseCard.Seals.RENTAL)

        if possible_jokers:
            joker = deepcopy(choice(possible_jokers))
            edition_r = random()
            if edition_r < 0.003:
                joker.edition = BaseCard.Editions.NEGATIVE
            elif edition_r < 0.006:
                joker.edition = BaseCard.Editions.POLYCHROME
            elif edition_r < 0.02:
                joker.edition = BaseCard.Editions.HOLOGRAPHIC
            elif edition_r < 0.04:
                joker.edition = BaseCard.Editions.FOIL

            # Stickers currently overwrite eachother since we are reusing "seal", need a way to represent multiple stickers
            if random() < 0.3 and stake >= 3:
                joker.seal = BaseCard.Seals.ETERNAL
            if random() < 0.3 and stake >= 6:
                joker.seal = BaseCard.Seals.PERISHABLE
            if random() < 0.3 and stake >= 7:
                joker.seal = BaseCard.Seals.RENTAL

            if joker.seal == BaseCard.Seals.RENTAL:
                joker.value = 1

            return joker
        return None

    @staticmethod
    def all_supported_joker_names():
        return [joker.name for joker in jokers]

    @staticmethod
    def observation_space(max_count):
        return sp.Dict(
            {
                "indices": sp.Box(0, 150, (max_count,), np.int32),
                "values": sp.Box(0, 2, (max_count,), np.float32),
                "mult_state": sp.Box(0, 2, (max_count,), np.float32),
                "mult_mult_state": sp.Box(0, 2, (max_count,), np.float32),
                "chips_state": sp.Box(0, 2, (max_count,), np.float32),
            }
        )

    @staticmethod
    def state_vector_size():
        return 4


jokers = [
    Joker(
        name="Even Steven",
        index=39,
        value=4,
        rarity=0,
        state={},
        rank_affinity=[2, 4, 6, 8, 10],
    ),
    Joker(
        name="Odd Todd",
        index=40,
        value=4,
        rarity=0,
        state={},
        rank_affinity=[3, 5, 7, 9, 14],
    ),
    Joker(
        name="Greedy Joker",
        index=2,
        value=5,
        rarity=0,
        state={},
        suit_affinity=["Diamonds"],
    ),
    Joker(
        name="Lusty Joker",
        index=3,
        value=5,
        rarity=0,
        state={},
        suit_affinity=["Hearts"],
    ),
    Joker(
        name="Wrathful Joker",
        index=4,
        value=5,
        rarity=0,
        state={},
        suit_affinity=["Spades"],
    ),
    Joker(
        name="Gluttonous Joker",
        index=5,
        value=5,
        rarity=0,
        state={},
        suit_affinity=["Clubs"],
    ),
    Joker(
        name="Fibonacci",
        index=31,
        value=8,
        rarity=1,
        state={},
        rank_affinity=[14, 2, 3, 5, 8],
    ),
    Joker(name="Scholar", index=41, value=4, rarity=0, state={}, rank_affinity=[14]),
    Joker(
        name="Scary Face",
        index=33,
        value=4,
        rarity=0,
        state={},
        rank_affinity=[11, 12, 13],
    ),
    Joker(
        name="Walkie Talkie",
        index=101,
        value=4,
        rarity=0,
        state={},
        rank_affinity=[10, 4],
    ),
    Joker(
        name="Smiley Face",
        index=104,
        value=4,
        rarity=0,
        state={},
        rank_affinity=[11, 12, 13],
    ),
    Joker(name="Joker", index=1, value=2, rarity=0, state={}),
    Joker(
        name="Jolly Joker",
        index=6,
        value=3,
        rarity=0,
        state={},
        hand_affinity=["Pair", "Three of a Kind", "Four of a Kind", "Full House"],
    ),
    Joker(name="Zany Joker", index=7, value=4, rarity=0, state={}),
    Joker(name="Mad Joker", index=8, value=4, rarity=0, state={}),
    Joker(name="Crazy Joker", index=9, value=4, rarity=0, state={}),
    Joker(name="Droll Joker", index=10, value=4, rarity=0, state={}),
    Joker(name="The Tribe", index=135, value=8, rarity=2, state={}),
    Joker(name="Sly Joker", index=11, value=3, rarity=0, state={}),
    Joker(name="Wily Joker", index=12, value=4, rarity=0, state={}),
    Joker(name="Clever Joker", index=13, value=4, rarity=0, state={}),
    Joker(name="Devious Joker", index=14, value=4, rarity=0, state={}),
    Joker(name="Crafty Joker", index=15, value=4, rarity=0, state={}),
    Joker(name="Half Joker", index=16, value=5, rarity=0, state={}),
    Joker(name="Four Fingers", index=18, value=7, rarity=1, state={}),
    Joker(name="Banner", index=22, value=5, rarity=0, state={}),
    Joker(name="Mystic Summit", index=23, value=5, rarity=0, state={}),
    Joker(name="Raised Fist", index=29, value=5, rarity=0, state={}),
    Joker(name="Misprint", index=27, value=4, rarity=0, state={}),
    Joker(name="Gros Michel", index=38, value=5, rarity=0, state={}),
    Joker(name="Abstract Joker", index=34, value=4, rarity=0),
    Joker(name="Egg", index=46, value=4, rarity=0, state={}),
    Joker(name="Ride the Bus", index=44, value=6, rarity=0, state={"mult": 0}),
    Joker(name="Ice Cream", index=50, value=5, rarity=0, state={"chips": 100}),
    Joker(name="Splash", index=52, value=3, rarity=0, state={}),
    Joker(name="Faceless Joker", index=57, value=4, rarity=0, state={}),
    Joker(name="Green Joker", index=58, value=4, rarity=0, state={"mult": 0}),
    Joker(name="Cavendish", index=61, value=4, rarity=0, state={}),
    Joker(name="Spare Trousers", index=98, value=6, rarity=1, state={"mult": 0}),
    Joker(name="Golden Joker", index=90, value=6, rarity=0, state={}),
    Joker(name="To the Moon", index=84, value=5, rarity=1, state={}),
    Joker(name="Ramen", index=100, value=6, rarity=1, state={"mult_mult": 2}),
    Joker(name="Swashbuckler", index=110, value=4, rarity=0, state={}),
    Joker(name="Smeared Joker", index=113, value=7, rarity=1, state={}),
    Joker(name="Acrobat", index=108, value=6, rarity=1, state={}),
    Joker(name="The Duo", index=131, value=8, rarity=2, state={}),
    Joker(name="The Trio", index=132, value=8, rarity=2, state={}),
    Joker(name="The Family", index=133, value=8, rarity=2, state={}),
    Joker(name="The Order", index=134, value=8, rarity=2, state={}),
    Joker(name="Shoot the Moon", index=140, value=5, rarity=0, state={}),
    Joker(name="Credit Card", index=20, value=1, rarity=0, state={}),
    Joker(name="Chaos the Clown", index=30, value=4, rarity=0, state={}),
    Joker(name="Burglar", index=47, value=6, rarity=1, state={}),
    Joker(name="Wee Joker", index=124, value=8, rarity=2, state={"chips": 0}),
    Joker(name="Supernova", index=43, value=5, rarity=0, state={}),
    Joker(name="Space Joker", index=45, value=5, rarity=1, state={}),
    Joker(name="Blue Joker", index=53, value=5, rarity=0),
    Joker(name="Hiker", index=56, value=5, rarity=1, state={}),
    Joker(name="Constellation", index=55, value=6, rarity=1, state={"mult_mult": 1}),
    Joker(name="Square Joker", index=65, value=4, rarity=0, state={"chips": 0}),
    Joker(name="Riff-Raff", index=67, value=6, rarity=0, state={}),
    Joker(name="Rough Gem", index=116, value=7, rarity=1, state={}),
    Joker(name="Bloodstone", index=117, value=7, rarity=1, state={}),
    Joker(name="Arrowhead", index=118, value=7, rarity=1, state={}),
    Joker(name="Onyx Agate", index=119, value=7, rarity=1, state={}),
    Joker(name="Flower Pot", index=122, value=6, rarity=1, state={}),
    Joker(name="Seeing Double", index=128, value=6, rarity=1, state={}),
    Joker(name="Hit the Road", index=130, value=8, rarity=2, state={"mult": 1}),
    Joker(name="Astronomer", index=143, value=8, rarity=1, state={}),
    Joker(name="Erosion", index=81, value=6, rarity=1, state={}),
    Joker(name="Pareidolia", index=37, value=5, rarity=1, state={}),
    Joker(name="Business Card", index=42, value=4, rarity=0, state={}),
    Joker(name="Card Sharp", index=62, value=6, rarity=1, state={}),
    Joker(name="Cloud 9", index=73, value=7, rarity=1, state={}),
    Joker(name="Popcorn", index=97, value=5, rarity=0, state={"mult": 20}),
    Joker(name="Blackboard", index=48, value=6, rarity=1, state={}),
    Joker(name="Bull", index=93, value=6, rarity=1, state={}),
    Joker(name="Bootstraps", index=145, value=7, rarity=1, state={}),
    Joker(name="Vampire", index=68, value=7, rarity=1, state={"mult_mult": 1}),
    Joker(name="Driver's License", index=141, value=7, rarity=2, state={}),
    Joker(name="Hologram", index=70, value=7, rarity=1, state={"mult_mult": 1}),
    Joker(name="Photograph", index=78, value=5, rarity=0, state={}),
    Joker(name="Flash Card", index=96, value=5, rarity=1, state={"mult": 0}),
    Joker(name="Drunkard", index=88, value=4, rarity=0, state={}),
    Joker(name="Baron", index=72, value=8, rarity=2, state={}),
    Joker(name="Mr. Bones", index=107, value=5, rarity=1, state={}),
    Joker(name="Gift Card", index=79, value=6, rarity=1, state={}),
    Joker(name="Baseball Card", index=92, value=8, rarity=2, state={}),
    Joker(name="Blueprint", index=123, value=10, rarity=2, state={}),
    Joker(name="Brainstorm", index=138, value=10, rarity=2, state={}),
    Joker(name="Runner", index=49, value=5, rarity=0, state={"chips": 0}),
    Joker(name="Troubadour", index=111, value=6, rarity=1, state={}),
    Joker(name="Juggler", index=87, value=4, rarity=0, state={}),
    Joker(name="Merry Andy", index=125, value=7, rarity=1, state={}),
    Joker(name="Stuntman", index=136, value=7, rarity=2, state={}),
    Joker(name="Joker Stencil", index=17, value=8, rarity=1, state={"mult": 1}),
    Joker(name="Satellite", index=139, value=6, rarity=1, state={}),
    Joker(name="Fortune Teller", index=86, value=6, rarity=0, state={}),
    Joker(name="Steel Joker", index=32, value=7, rarity=1),
    Joker(name="Glass Joker", index=120, value=6, rarity=1, state={"mult_mult": 1}),
    Joker(name="Stone Joker", index=89, value=6, rarity=1, state={}),
    Joker(name="Lucky Cat", index=91, value=6, rarity=1, state={"mult_mult": 1}),
    Joker(name="Golden Ticket", index=106, value=5, rarity=0, state={}),
    ######################
    # Currently implementable
    # Joker(name="Burnt Joker", index=144, value=8, rarity=2, state={}),
    ####
    # Joker(name="Trading Card", index=95, value=6, rarity=1, state={}),
    # Joker(name="Marble Joker", index=24, value=6, rarity=1, state={}),
    # Joker(name="Delayed Gratification", index=35, value=4, rarity=0, state={}),
    # Joker(name="Red Card", index=63, value=5, rarity=0, state={}),
    # Joker(name="DNA", index=51, value=8, rarity=2, state={}),
    # Joker(name="Shortcut", index=69, value=7, rarity=1, state={}),
    # Joker(name="Reserved Parking", index=82, value=6, rarity=0, state={}),
    # Joker(name="Madness", index=64, value=7, rarity=1, state={"mult": 1}),
    # Joker(name="Rocket", index=74, value=6, rarity=1, state={}),
    # Joker(name="Obelisk", index=75, value=8, rarity=2, state={"mult": 1}),
    # Joker(name="Midas Mask", index=76, value=7, rarity=1, state={}),
    # Joker(name="Campfire", index=105, value=9, rarity=2, state={}),
    # Joker(name="Certificate", index=112, value=6, rarity=1, state={}),
    # Joker(name="Showman", index=121, value=5, rarity=1, state={}),
    # Joker(name="Matador", index=129, value=7, rarity=1, state={}),
    ######################
    # Requires consumeable slots
    # Joker(name="8 Ball", index=26, value=5, rarity=0, state={}),
    # Joker(name="Sixth Sense", index=54, value=6, rarity=1, state={}),
    # Joker(name="Superposition", index=59, value=4, rarity=0, state={}),
    # Joker(name="Seance", index=66, value=6, rarity=1, state={}),
    # Joker(name="Vagabond", index=71, value=8, rarity=2, state={}),
    # Joker(name="Hallucination", index=85, value=4, rarity=0, state={}),
    # Joker(name="Cartomancer", index=142, value=6, rarity=1, state={}),
    ######################
    # Requires Retriggering
    # Joker(name="Dusk", index=28, value=5, rarity=1, state={}),
    # Joker(name="Hack", index=36, value=6, rarity=1, state={}),
    # Joker(name="Seltzer", index=102, value=6, rarity=1, state={}),
    # Joker(name="Sock and Buskin", index=109, value=6, rarity=1, state={}),
    # Joker(name="Hanging Chad", index=115, value=4, rarity=0, state={}),
    ######################
    # Requires joker positioning/targeting
    # Joker(name="Ceremonial Dagger", index=21, value=6, rarity=1, state={"mult": 0}),
    # Joker(name="Mime", index=19, value=5, rarity=1, state={}),
    ######################
    # Requires expanded joker state vector
    # Joker(name="Loyalty Card", index=25, value=5, rarity=1, state={"remaining": 5}),
    # Joker(name="To Do List", index=60, value=4, rarity=0, state={}),
    # Joker(name="Mail-In Rebate", index=83, value=4, rarity=0, state={}),
    # Joker(name="Castle", index=103, value=6, rarity=1, state={}),
    # Joker(name="Ancient Joker", index=99, value=8, rarity=2, state={}),
    # Joker(name="The Idol", index=127, value=6, rarity=1, state={}),
    # Joker(name="Invisible Joker", index=137, value=8, rarity=2, state={}),
    # Joker(name="Turtle Bean", index=80, value=6, rarity=1, state={}),
    ######################
    # Legendary (Requires spectral etc)
    Joker(name="Canio", index=146, value=20, rarity=3, state={"mult_mult": 1}),
    Joker(name="Triboulet", index=147, value=20, rarity=3, state={}),
    Joker(
        name="Yorick",
        index=148,
        value=20,
        rarity=3,
        state={"mult_mult": 1, "discards_left": 23},
    ),
    Joker(name="Chicot", index=149, value=20, rarity=3, state={}),
    # Joker(name="Perkeo", index=150, value=20, rarity=3, state={}),
    ######################
    # Joker(name="Luchador", index=77, value=5, rarity=1, state={}),
    # Joker(name="Diet Cola", index=94, value=6, rarity=1, state={}),
    # Joker(name="Throwback", index=114, value=6, rarity=1, state={"mult": 1}),
    # Joker(name="Oops! All 6s", index=126, value=4, rarity=1, state={}),
]

label_lookup = {
    joker.name: joker for joker in jokers if joker.index > 0
}  # Exclude the first joker (index 0) which is a placeholder
