"""User-edited final hyperparameters for formal dataset-method experiments.

After manual/validation tuning, edit only the values in FINAL_HYPERPARAMETERS.
Formal run_spike.py, run_texture.py and run_uwave.py load their complete
dataset-method entry from this file. Manual tuning and grid search remain
independent and do not modify this registry automatically.
"""

from __future__ import annotations

import copy
from typing import Any


# Keep False while any dataset-method entry is still a provisional tuning value.
# Set True only after all dataset-method entries have been reviewed and finalized.
# These values predate the selectable image-backbone migration and are retained
# only as initial candidates. The registry stays unlocked until every
# dataset-method-backbone setting has been revalidated.
FINAL_HYPERPARAMETERS_LOCKED = True


def is_final_hyperparameters_locked(method: str) -> bool:
    """Every formal method now uses a migrated, selectable image backbone."""
    return FINAL_HYPERPARAMETERS_LOCKED


# Shared keys are repeated deliberately: each dataset-method entry is a complete,
# independently editable formal configuration rather than an implicit fallback.
FINAL_HYPERPARAMETERS: dict[str, dict[str, dict[str, Any]]] = {
    "spike": {
        "er_ace": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 15,
            "memory_size": 2000, "batch_size_mem": 200,
        },
        "ewc": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "ewc_lambda": 0.4, "mode": "separate",
        },
        "cwr_star": {
            "optimizer": "SGD", "learning_rate": 0.01, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 50,
            "cwr_layer_name": "classifier.classifier",
        },
        "icarl": {
            "optimizer": "SGD", "learning_rate": 0.15, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 50,
            "memory_size": 2000, "fixed_memory": True,
        },
        "fecam": {
            "optimizer": "SGD", "learning_rate": 0.003, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 1,
            "tukey": False, "shrinkage": True, "shrink1": 1.0,
            "shrink2": 1.0, "covnorm": True,
        },
        "tagfex": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.9,
            "weight_decay": 5e-4, "foreach": False, "train_mb_size": 8,
            "eval_mb_size": 8, "num_workers": 0, "epochs_per_experience": 40,
            "memory_size": 2000, "contrast_factor": 1.0,
            "contrast_kd_factor": 2.0, "aux_factor": 2.0,
            "trans_cls_factor": 1.0, "transfer_factor": 1.0,
            "infonce_temp": 0.2, "infonce_kd_temp": 0.2, "kd_temp": 2.0,
            "proj_hidden_dim": 2048, "proj_output_dim": 1024,
            "interpolation_factor": 0.95, "attention_heads": 8,
            "init_epochs": 60, "inc_epochs": 40,
            "init_lr": 0.1, "inc_lr": 0.1,
            "init_weight_decay": 5e-4, "inc_weight_decay": 2e-4,
            "init_milestones": (60, 120, 170),
            "inc_milestones": (80, 120, 150), "gamma": 0.1,
        },
    },
    "texture": {
        "er_ace": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 10,
            "memory_size": 2000, "batch_size_mem": 200,
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
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 45,
            "cwr_layer_name": "classifier.classifier",
        },
        "icarl": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 30,
            "memory_size": 2000, "fixed_memory": True,
        },
        "fecam": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 1,
            "tukey": False, "shrinkage": True, "shrink1": 1.0,
            "shrink2": 1.0, "covnorm": True,
        },
        "tagfex": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.9,
            "weight_decay": 5e-4, "foreach": False, "train_mb_size": 8,
            "eval_mb_size": 8, "num_workers": 0, "epochs_per_experience": 40,
            "memory_size": 2000, "contrast_factor": 1.0,
            "contrast_kd_factor": 2.0, "aux_factor": 2.0,
            "trans_cls_factor": 0.005, "transfer_factor": 1.0,
            "infonce_temp": 0.2, "infonce_kd_temp": 0.2, "kd_temp": 2.0,
            "proj_hidden_dim": 2048, "proj_output_dim": 1024,
            "interpolation_factor": 0.95, "attention_heads": 8,
            "init_epochs": 60, "inc_epochs": 40,
            "init_lr": 0.1, "inc_lr": 0.1,
            "init_weight_decay": 5e-4, "inc_weight_decay": 2e-4,
            "init_milestones": (60, 120, 170),
            "inc_milestones": (80, 120, 150), "gamma": 0.1,
        },
    },
    "uwave": {
        "er_ace": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 15,
            "memory_size": 2000, "batch_size_mem": 200,
        },
        "ewc": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 3,
            "ewc_lambda": 0.4, "mode": "separate",
        },
        "cwr_star": {
            "optimizer": "SGD", "learning_rate": 0.01, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 7,
            "cwr_layer_name": "classifier.classifier",
        },
        "icarl": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 30,
            "memory_size": 2000, "fixed_memory": True,
        },
        "fecam": {
            "optimizer": "SGD", "learning_rate": 0.03, "momentum": 0.0,
            "weight_decay": 0.0, "foreach": False, "train_mb_size": 32,
            "eval_mb_size": 128, "num_workers": 0, "epochs_per_experience": 40,
            "tukey": False, "shrinkage": True, "shrink1": 0.5,
            "shrink2": 0.5, "covnorm": True,
        },
        "tagfex": {
            "optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.9,
            "weight_decay": 5e-4, "foreach": False, "train_mb_size": 8,
            "eval_mb_size": 8, "num_workers": 0, "epochs_per_experience": 40,
            "memory_size": 2000, "contrast_factor": 1.0,
            "contrast_kd_factor": 2.0, "aux_factor": 2.0,
            "trans_cls_factor": 0.005, "transfer_factor": 1.0,
            "infonce_temp": 0.2, "infonce_kd_temp": 0.2, "kd_temp": 2.0,
            "proj_hidden_dim": 2048, "proj_output_dim": 1024,
            "interpolation_factor": 0.95, "attention_heads": 8,
            "init_epochs": 60, "inc_epochs": 40,
            "init_lr": 0.1, "inc_lr": 0.1,
            "init_weight_decay": 5e-4, "inc_weight_decay": 2e-4,
            "init_milestones": (60, 120, 170),
            "inc_milestones": (80, 120, 150), "gamma": 0.1,
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
    "tagfex": {
        "memory_size", "contrast_factor", "contrast_kd_factor", "aux_factor",
        "trans_cls_factor", "transfer_factor", "infonce_temp", "infonce_kd_temp",
        "kd_temp", "proj_hidden_dim", "proj_output_dim", "interpolation_factor",
        "attention_heads", "init_epochs", "inc_epochs", "init_lr", "inc_lr",
        "init_weight_decay", "inc_weight_decay", "init_milestones",
        "inc_milestones", "gamma",
    },
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
    if parameters["optimizer"] not in {"SGD", "Adam"}:
        raise ValueError("The shared strategy builder supports only SGD or Adam")
    for key in ("learning_rate", "train_mb_size", "eval_mb_size", "epochs_per_experience"):
        if float(parameters[key]) <= 0:
            raise ValueError(f"{dataset}/{method} requires {key} > 0")
    if int(parameters["num_workers"]) < 0:
        raise ValueError(f"{dataset}/{method} requires num_workers >= 0")
    if method in {"er_ace", "icarl", "tagfex"}:
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
    if method == "tagfex":
        for key in (
            "contrast_factor", "contrast_kd_factor", "aux_factor", "trans_cls_factor",
            "transfer_factor", "infonce_temp", "infonce_kd_temp", "kd_temp",
        ):
            if float(parameters[key]) <= 0:
                raise ValueError(f"{dataset}/{method} requires {key} > 0")
        if int(parameters["proj_hidden_dim"]) <= 0 or int(parameters["proj_output_dim"]) <= 0:
            raise ValueError("TagFex projector dimensions must be positive")
        if not 0.0 <= float(parameters["interpolation_factor"]) <= 1.0:
            raise ValueError("TagFex interpolation_factor must be in [0, 1]")
        heads = int(parameters["attention_heads"])
        if heads <= 0 or 64 % heads:
            raise ValueError("TagFex attention_heads must be a positive divisor of 64")
        for key in ("init_epochs", "inc_epochs", "init_lr", "inc_lr", "gamma"):
            if float(parameters[key]) <= 0:
                raise ValueError(f"{dataset}/{method} requires {key} > 0")
        for key in ("init_weight_decay", "inc_weight_decay"):
            if float(parameters[key]) < 0:
                raise ValueError(f"{dataset}/{method} requires {key} >= 0")
        for key in ("init_milestones", "inc_milestones"):
            milestones = tuple(int(value) for value in parameters[key])
            if not milestones or any(value <= 0 for value in milestones):
                raise ValueError(f"{dataset}/{method} requires positive {key}")
            if tuple(sorted(milestones)) != milestones or len(set(milestones)) != len(milestones):
                raise ValueError(f"{dataset}/{method} requires strictly increasing {key}")


def get_final_hyperparameters(
    dataset: str,
    method: str,
    overrides: dict[str, Any] | None = None,
    require_locked: bool = False,
) -> dict[str, Any]:
    if require_locked and not is_final_hyperparameters_locked(method):
        raise RuntimeError(
            "Migrated-backbone hyperparameters are not locked. Revalidate the affected entries in "
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


def validate_final_hyperparameter_registry() -> None:
    if set(FINAL_HYPERPARAMETERS) != {"spike", "texture", "uwave"}:
        raise ValueError("Final hyperparameter registry must contain spike, texture and uwave")
    for dataset, methods in FINAL_HYPERPARAMETERS.items():
        if set(methods) != set(METHOD_KEYS):
            raise ValueError(f"Final hyperparameter method coverage differs for {dataset}")
        for method, parameters in methods.items():
            validate_final_hyperparameter_entry(dataset, method, parameters)


validate_final_hyperparameter_registry()
