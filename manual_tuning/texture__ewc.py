from common import run_manual

PARAMETERS = {
    # SGD step size. Larger values learn faster but can overpower EWC protection;
    # smaller values are steadier but may leave new tasks underfit.
    "learning_rate": 0.1,
    # Strength of the Fisher-weighted old-parameter penalty. Larger values improve
    # stability but restrict new-task learning; smaller values increase plasticity.
    "ewc_lambda": 0.4,
    # "separate" retains one Fisher/snapshot per past experience. Keep it fixed for
    # the formal protocol; changing mode changes both regularization and storage.
    "mode": "separate",
}

# Keep ORDER_ID and SEED fixed across candidates. More EPOCHS may improve the current
# task but can increase forgetting. Explicit CUDA prevents silent CPU fallback.
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"

if __name__ == "__main__":
    run_manual(dataset="texture", method="ewc", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
