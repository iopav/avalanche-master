"""Shared settings for the fixed-split learning-rate search."""

from __future__ import annotations


LR_CANDIDATES = (0.1, 0.05, 0.01)
ACC_TOLERANCE = 0.005
SEARCH_EPOCHS = 3
VALIDATION_FRACTION = 0.2
VALIDATION_SPLIT_SEED = 62

JOINT_METHOD = "joint"
PIPELINE_METHODS = (
    "ewc",
    "icarl",
    "er_ace",
    "fecam",
    "tagfex",
    "cwr_star",
)

def normalize_method(value: str) -> str:
    aliases = {
        "erace": "er_ace",
        "cwr*": "cwr_star",
    }
    method = aliases.get(value.lower(), value.lower())
    if method not in PIPELINE_METHODS:
        raise ValueError(f"Unknown method {value!r}; choose from {PIPELINE_METHODS}")
    return method
