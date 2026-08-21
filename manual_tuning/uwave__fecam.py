from common import run_manual

PARAMETERS = {"learning_rate": 0.1, "tukey": False, "shrinkage": True, "shrink1": 1.0, "shrink2": 1.0, "covnorm": True}
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, None

if __name__ == "__main__":
    run_manual(dataset="uwave", method="fecam", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
