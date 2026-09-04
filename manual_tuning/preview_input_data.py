from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "dataset_mini"
OUTPUT_ROOT = PROJECT_ROOT / "manual_tuning" / "data_preview"


def _three_classes(labels: np.ndarray) -> list[int]:
    return [int(np.flatnonzero(labels == label)[0]) for label in np.unique(labels)[:3]]


def _stats(array: np.ndarray) -> dict[str, float | list[int] | str]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "zero_fraction": float(np.mean(array == 0)),
    }


def main() -> None:
    spike_raw = np.load(DATA_ROOT / "spike/X_train_20_cla_400steps.npy", mmap_mode="r")
    spike_img = np.load(DATA_ROOT / "spike/X_train_20_cla_400steps_img.npy", mmap_mode="r")
    spike_y = np.load(DATA_ROOT / "spike/Y_train_20_cla_400steps.npy")
    uwave_raw = np.load(DATA_ROOT / "uwave/X_train_uwave.npy", mmap_mode="r")
    uwave_img = np.load(DATA_ROOT / "uwave/X_train_uwave_img.npy", mmap_mode="r")
    uwave_y = np.load(DATA_ROOT / "uwave/Y_train_uwave.npy")
    texture_raw = np.load(DATA_ROOT / "texture/X_train_texture.npy", mmap_mode="r")
    texture_img = np.load(DATA_ROOT / "texture/X_train_texture_img.npy", mmap_mode="r")
    texture_y = np.load(DATA_ROOT / "texture/Y_train_texture.npy")

    indices = {
        "spike": _three_classes(spike_y),
        "uwave": _three_classes(uwave_y),
        "texture": _three_classes(texture_y),
    }
    transformed = {
        "spike": [
            np.asarray(spike_raw[index]).reshape(160, 160).astype(np.float32) * 255.0
            for index in indices["spike"]
        ],
        "uwave": [
            np.pad(np.asarray(uwave_raw[index]).reshape(-1), (0, 79)).reshape(32, 32)
            for index in indices["uwave"]
        ],
        "texture": [
            np.pad(np.asarray(texture_raw[index], dtype=np.float32).reshape(-1), (0, 250)).reshape(185, 185)
            for index in indices["texture"]
        ],
    }

    stored_planes = {
        "spike": [np.asarray(spike_img[index, 0], dtype=np.float32) for index in indices["spike"]],
        "uwave": [np.asarray(uwave_img[index, :, :, 0], dtype=np.float32) for index in indices["uwave"]],
        "texture": [np.asarray(texture_img[index, 0], dtype=np.float32) for index in indices["texture"]],
    }
    labels = {"spike": spike_y, "uwave": uwave_y, "texture": texture_y}
    arrays = {"spike": spike_raw, "uwave": uwave_raw, "texture": texture_raw}
    stored = {"spike": spike_img, "uwave": uwave_img, "texture": texture_img}

    report: dict[str, object] = {}
    for dataset in ("spike", "uwave", "texture"):
        selected = indices[dataset]
        report[dataset] = {
            "selected_indices": selected,
            "selected_labels": [int(labels[dataset][index]) for index in selected],
            "raw_samples": [_stats(np.asarray(arrays[dataset][index])) for index in selected],
            "stored_image_shape": list(stored[dataset].shape[1:]),
            "stored_image_dtype": str(stored[dataset].dtype),
            "max_abs_reconstruction_error": float(
                max(
                    np.max(np.abs(generated - persisted))
                    for generated, persisted in zip(transformed[dataset], stored_planes[dataset])
                )
            ),
            "three_channels_identical": bool(
                all(
                    (
                        np.array_equal(np.asarray(stored[dataset][index])[..., 0], np.asarray(stored[dataset][index])[..., 1])
                        and np.array_equal(np.asarray(stored[dataset][index])[..., 0], np.asarray(stored[dataset][index])[..., 2])
                    )
                    if dataset == "uwave"
                    else (
                        np.array_equal(np.asarray(stored[dataset][index])[0], np.asarray(stored[dataset][index])[1])
                        and np.array_equal(np.asarray(stored[dataset][index])[0], np.asarray(stored[dataset][index])[2])
                    )
                    for index in selected
                )
            ),
        }

    figure, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for row, dataset in enumerate(("spike", "uwave", "texture")):
        values = transformed[dataset]
        minimum = min(float(value.min()) for value in values)
        maximum = max(float(value.max()) for value in values)
        norm = None
        if dataset != "spike" and minimum < 0 < maximum:
            norm = TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
        for column, (index, value) in enumerate(zip(indices[dataset], values)):
            axis = axes[row, column]
            image = axis.imshow(
                value,
                cmap="gray" if dataset == "spike" else "coolwarm",
                vmin=0 if dataset == "spike" else None,
                vmax=255 if dataset == "spike" else None,
                norm=norm,
                interpolation="nearest",
            )
            axis.set_title(
                f"{dataset} index={index} label={int(labels[dataset][index])}\n"
                f"shape={value.shape}, range=[{value.min():.3g}, {value.max():.3g}]"
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if dataset != "spike":
                figure.colorbar(image, ax=axis, shrink=0.72)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_ROOT / "input_conversion_examples.png"
    report_path = OUTPUT_ROOT / "input_conversion_report.json"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"figure={figure_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
