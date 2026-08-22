from common import run_manual

PARAMETERS = {
    # SGD step size for the first experience only. Larger values learn the frozen
    # representation faster but can be unstable; smaller values may underfit it.
    "learning_rate": 0.1,
    # Tukey power transform. Keep False because the shared feature vector may be
    # negative; fractional powers of negative values can create NaN values.
    "tukey": False,
    # Enables covariance regularization. Disabling it uses raw covariance estimates,
    # which may be noisy or singular when a class has few samples.
    "shrinkage": True,
    # Diagonal ridge strength. Larger values improve invertibility but can wash out
    # class-specific variance; smaller values retain variance but may be ill-conditioned.
    "shrink1": 1.0,
    # Off-diagonal shrink strength. Larger values smooth correlations more strongly;
    # excessive values distort class covariance, while small values preserve noise.
    "shrink2": 1.0,
    # Converts covariance to correlation scale. True reduces channel-scale dominance;
    # False preserves absolute feature variance and can favor high-variance dimensions.
    "covnorm": True,
}

# EPOCHS trains only the first experience; later experiences use the frozen backbone
# and one statistics pass. Keep ORDER_ID/SEED fixed. CUDA is mandatory here.
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"

if __name__ == "__main__":
    run_manual(dataset="uwave", method="fecam", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
