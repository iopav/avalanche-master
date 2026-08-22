from common import run_manual

# learning_rate: SGD step size. Larger values learn each task faster but can oscillate;
# smaller values are more stable but may underfit within the fixed epoch budget.
PARAMETERS = {"learning_rate": 0.1}

# ORDER_ID changes the class arrival order and SEED controls all training randomness;
# keep both fixed while comparing candidates. More EPOCHS improves fitting but costs
# more time and may overfit. DEVICE is explicit so a CUDA failure cannot fall back to CPU.
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"

if __name__ == "__main__":
    run_manual(dataset="texture", method="cwr_star", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
