from common import run_manual

PARAMETERS = {"learning_rate": 0.1, "ewc_lambda": 0.4, "mode": "separate"}
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, None

if __name__ == "__main__":
    run_manual(dataset="uwave", method="ewc", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
