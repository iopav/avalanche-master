"""User-supplied hyperparameter-search FLOPs for formal summaries.

Fill each dataset-method entry after its complete search has been measured with
the same FLOP convention as formal training. ``None`` deliberately blocks a
formal run instead of silently publishing an uncertain zero.
"""

from __future__ import annotations


HYPERPARAMETER_SEARCH_FLOPS: dict[str, dict[str, int | None]] = {
    "spike": {
        "er_ace": None,
        "ewc": None,
        "cwr_star": None,
        "icarl": None,
        "fecam": None,
    },
    "texture": {
        "er_ace": None,
        "ewc": None,
        "cwr_star": None,
        "icarl": None,
        "fecam": None,
    },
    "uwave": {
        "er_ace": None,
        "ewc": None,
        "cwr_star": None,
        "icarl": None,
        "fecam": None,
    },
}


def get_hyperparameter_search_flops(dataset: str, method: str) -> int:
    try:
        value = HYPERPARAMETER_SEARCH_FLOPS[dataset][method]
    except KeyError as exc:
        raise KeyError(f"Missing hyperparameter-search FLOPs entry for {dataset}/{method}") from exc
    if value is None:
        raise RuntimeError(
            f"Hyperparameter-search FLOPs are not filled for {dataset}/{method}. "
            "Update cil_experiments/hyperparameter_search_flops.py before a formal run."
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Hyperparameter-search FLOPs for {dataset}/{method} must be a non-negative integer"
        )
    return value
