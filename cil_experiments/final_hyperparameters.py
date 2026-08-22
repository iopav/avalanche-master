"""User-edited final hyperparameters for formal dataset-method experiments.

After manual/validation tuning, edit only the values in FINAL_HYPERPARAMETERS.
Formal run_spike.py, run_texture.py and run_uwave.py load their complete
dataset-method entry from this file. Manual tuning and grid search remain
independent and do not modify this registry automatically.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


# Keep False while any of the 15 entries is still a provisional tuning value.
# Set True only after all dataset-method entries have been reviewed and finalized.
FINAL_HYPERPARAMETERS_LOCKED = False


# Shared keys are repeated deliberately: each dataset-method entry is a complete,
# independently editable formal configuration rather than an implicit fallback.
FINAL_HYPERPARAMETERS: dict[str, dict[str, dict[str, Any]]] = {
    "spike": {
        "er_ace": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "memory_size": 200, "batch_size_mem": 10,
        },
        "ewc": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "ewc_lambda": 0.4, "mode": "separate",
        },
        "cwr_star": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "cwr_layer_name": "classifier.classifier",
        },
        "icarl": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "memory_size": 2000, "fixed_memory": True,
        },
        "fecam": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "tukey": False, "shrinkage": True, "shrink1": 1.0,
            "shrink2": 1.0, "covnorm": True,
        },
    },
    "texture": {
        "er_ace": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "memory_size": 200, "batch_size_mem": 10,
        },
        "ewc": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "ewc_lambda": 0.4, "mode": "separate",
        },
        "cwr_star": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "cwr_layer_name": "classifier.classifier",
        },
        "icarl": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "memory_size": 2000, "fixed_memory": True,
        },
        "fecam": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "tukey": False, "shrinkage": True, "shrink1": 1.0,
            "shrink2": 1.0, "covnorm": True,
        },
    },
    "uwave": {
        "er_ace": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "memory_size": 200, "batch_size_mem": 10,
        },
        "ewc": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "ewc_lambda": 0.4, "mode": "separate",
        },
        "cwr_star": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "cwr_layer_name": "classifier.classifier",
        },
        "icarl": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "memory_size": 2000, "fixed_memory": True,
        },
        "fecam": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "tukey": False, "shrinkage": True, "shrink1": 1.0,
            "shrink2": 1.0, "covnorm": True,
        },
    },
}


SHARED_KEYS = {
    "optimizer", "learning_rate", "momentum", "weight_decay", "foreach",
    "train_mb_size", "eval_mb_size", "num_workers", "epochs_per_experience",
}
METHOD_KEYS = {
    "er_ace": {"memory_size", "batch_size_mem"},
    "ewc": {"ewc_lambda", "mode"},
    "cwr_star": {"cwr_layer_name"},
    "icarl": {"memory_size", "fixed_memory"},
    "fecam": {"tukey", "shrinkage", "shrink1", "shrink2", "covnorm"},
}


def validate_final_hyperparameter_entry(
    dataset: str, method: str, parameters: dict[str, Any]
) -> None:
    if dataset not in FINAL_HYPERPARAMETERS:
        raise KeyError(f"Unknown final-hyperparameter dataset: {dataset}")
    if method not in METHOD_KEYS:
        raise KeyError(f"Unknown final-hyperparameter method: {method}")
    expected = SHARED_KEYS | METHOD_KEYS[method]
    actual = set(parameters)
    if actual != expected:
        raise ValueError(
            f"Final hyperparameter keys differ for {dataset}/{method}: "
            f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
        )
    if parameters["optimizer"] != "SGD":
        raise ValueError("Only SGD is implemented by the shared formal strategy builder")
    for key in ("learning_rate", "train_mb_size", "eval_mb_size", "epochs_per_experience"):
        if float(parameters[key]) <= 0:
            raise ValueError(f"{dataset}/{method} requires {key} > 0")
    if int(parameters["num_workers"]) < 0:
        raise ValueError(f"{dataset}/{method} requires num_workers >= 0")
    if method in {"er_ace", "icarl"}:
        memory_size = int(parameters["memory_size"])
        if memory_size <= 0 or memory_size > 2000:
            raise ValueError(f"{dataset}/{method} memory_size must be in [1, 2000]")
    if method == "er_ace":
        replay_batch = int(parameters["batch_size_mem"])
        if replay_batch <= 0 or replay_batch > int(parameters["memory_size"]):
            raise ValueError(
                f"{dataset}/{method} batch_size_mem must be in [1, memory_size]"
            )
    if method == "ewc":
        if float(parameters["ewc_lambda"]) < 0:
            raise ValueError(f"{dataset}/{method} ewc_lambda must be non-negative")
        if parameters["mode"] != "separate":
            raise ValueError("The formal EWC protocol currently requires mode='separate'")
    if method == "icarl" and parameters["fixed_memory"] is not True:
        raise ValueError("The local iCaRL adapter implements only fixed_memory=True")
    if method == "fecam":
        if float(parameters["shrink1"]) < 0 or float(parameters["shrink2"]) < 0:
            raise ValueError("FeCAM shrink strengths must be non-negative")


def get_final_hyperparameters(
    dataset: str,
    method: str,
    overrides: dict[str, Any] | None = None,
    require_locked: bool = False,
) -> dict[str, Any]:
    if require_locked and not FINAL_HYPERPARAMETERS_LOCKED:
        raise RuntimeError(
            "Formal hyperparameters are not locked. Fill all 15 entries in "
            "cil_experiments/final_hyperparameters.py, then set "
            "FINAL_HYPERPARAMETERS_LOCKED = True."
        )
    try:
        parameters = copy.deepcopy(FINAL_HYPERPARAMETERS[dataset][method])
    except KeyError as exc:
        raise KeyError(f"Missing final hyperparameters for {dataset}/{method}") from exc
    if overrides:
        unknown = set(overrides) - set(parameters)
        if unknown:
            raise KeyError(f"Unknown overrides for {dataset}/{method}: {sorted(unknown)}")
        parameters.update(copy.deepcopy(overrides))
    validate_final_hyperparameter_entry(dataset, method, parameters)
    return parameters


def final_hyperparameter_hash(parameters: dict[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_final_hyperparameter_registry() -> None:
    if set(FINAL_HYPERPARAMETERS) != {"spike", "texture", "uwave"}:
        raise ValueError("Final hyperparameter registry must contain spike, texture and uwave")
    for dataset, methods in FINAL_HYPERPARAMETERS.items():
        if set(methods) != set(METHOD_KEYS):
            raise ValueError(f"Final hyperparameter method coverage differs for {dataset}")
        for method, parameters in methods.items():
            validate_final_hyperparameter_entry(dataset, method, parameters)


validate_final_hyperparameter_registry()
