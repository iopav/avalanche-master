"""Run ER-2000 and cumulative-joint controls for all three datasets."""

from common import run_joint_report, run_manual_report


SHARED_PARAMETERS = {
    "learning_rate": 0.01,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "train_mb_size": 32,
    "eval_mb_size": 128,
}
ORDER_ID, SEED, EPOCHS, DEVICE = 6, 62, 50, "cuda"


if __name__ == "__main__":
    for dataset in ("spike", "texture", "uwave"):
        run_manual_report(
            dataset=dataset,
            method="er",
            parameters={
                **SHARED_PARAMETERS,
                "memory_size": 2000,
                "batch_size_mem": 200,
            },
            order_id=ORDER_ID,
            seed=SEED,
            epochs=EPOCHS,
            device=DEVICE,
        )
        run_joint_report(
            dataset=dataset,
            parameters=dict(SHARED_PARAMETERS),
            order_id=ORDER_ID,
            seed=SEED,
            epochs=EPOCHS,
            device=DEVICE,
        )
