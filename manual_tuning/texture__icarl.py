from common import run_manual

PARAMETERS = {"learning_rate": 0.1, "memory_size": 2000, "fixed_memory": True}
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, None

if __name__ == "__main__":
    run_manual(dataset="texture", method="icarl", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
