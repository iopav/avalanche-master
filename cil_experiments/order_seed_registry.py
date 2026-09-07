"""Single source of truth for seeds and dataset-specific task-grouped orders.

Each order is stored only once as a tuple of task groups. Flattening the groups
gives the class order; the group lengths give the task split.
"""

from __future__ import annotations



SEEDS_BY_DATASET: dict[str, tuple[int, ...]] = {
    "spike": (63, 65, 67, 69, 71),
    "texture": (52,53,54,55,56),
    "uwave": (48,50,52,54,56)
}


def get_search_seed(dataset: str) -> int:
    """Seed used by the legacy single-seed search path."""
    return SEEDS_BY_DATASET[dataset][0]


def get_formal_seeds(dataset: str) -> tuple[int, ...]:
    """Held-out seeds used by the legacy formal path."""
    return SEEDS_BY_DATASET[dataset][1:]

ORDERS_BY_DATASET: dict[str, dict[int, tuple[tuple[int, ...], ...]]] = {
    "spike": {
        1: ((7, 18, 12), (1,), (11,), (9,), (13,), (8,), (2,), (17,), (10,), (6,), (4,), (5,), (15,), (3,), (19,), (0,), (14,), (16,)),
        2: ((16, 19, 0), (9,), (8,), (17,), (1,), (12,), (11,), (5,), (10,), (4,), (18,), (13,), (15,), (7,), (14,), (6,), (3,), (2,)),
        3: ((14, 1, 19), (2,), (16,), (3,), (5,), (7,), (6,), (13,), (12,), (0,), (8,), (11,), (10,), (18,), (4,), (9,), (17,), (15,)),

    },
    "texture": {
        1: ((2, 8, 6), (3,), (7,), (1,), (10,), (0,), (4,), (11,), (5,), (9,)),
        2: ((6, 7, 10), (3,), (11,), (8,), (9,), (4,), (2,), (5,), (1,), (0,)),
        3: ((10, 8, 2), (11,), (7,), (6,), (4,), (0,), (9,), (1,), (3,), (5,)),


    },
    "uwave": {
        1: ((4, 0, 3), (6,), (5,), (2,), (7,), (1,)),
        2: ((5, 2, 6), (0,), (7,), (1,), (3,), (4,)),
        3: ((2, 6, 0), (4,), (7,), (5,), (3,), (1,)),

    },
}
