from common import run_manual_report

PARAMETERS = {"learning_rate": 0.01, "momentum": 0.0, "weight_decay": 0.0, "train_mb_size": 32, "eval_mb_size": 128, "alpha": 5.0, "temperature": 2.0}
ORDER_ID, SEED, EPOCHS, DEVICE = 7, 62, 10, "cuda"

if __name__ == "__main__":
    run_manual_report(dataset="texture", method="lwf", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
