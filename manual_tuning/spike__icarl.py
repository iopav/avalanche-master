from common import run_manual

PARAMETERS = {
    # SGD step size. Larger values adapt quickly but can disturb old features;
    # smaller values are more conservative but may underfit new classes.
    "learning_rate": 0.1,
    # Total exemplar cap shared by all seen classes. Larger values usually improve
    # NCM prototypes and retention, at higher storage, selection and replay cost.
    "memory_size": 2000,
    # Locked protocol flag: True means a fixed total budget divided among seen classes.
    # The local adapter currently implements only this mode, so do not tune this field.
    "fixed_memory": True,
}

# Keep ORDER_ID and SEED fixed across candidates. More EPOCHS can improve representation
# learning but increases replay compute and overfitting risk. CUDA is mandatory here.
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"

if __name__ == "__main__":
    run_manual(dataset="spike", method="icarl", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
