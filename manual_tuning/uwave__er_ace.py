from common import run_manual

PARAMETERS = {"learning_rate": 0.1, "memory_size": 200, "batch_size_mem": 10}
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, None

if __name__ == "__main__":
    run_manual(dataset="uwave", method="er_ace", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
