from common import run_manual

PARAMETERS = {
    # SGD step size. Larger values adapt faster but can destabilize old classes;
    # smaller values reduce update size but may underfit new classes.
    "learning_rate": 0.1,
    # Total logical replay-sample capacity. Larger values usually improve retention,
    # while increasing persistent storage and exemplar-management work.
    "memory_size": 200,
    # Replay samples paired with each current minibatch. Larger values strengthen
    # replay but raise compute and can reduce emphasis on the current task.
    "batch_size_mem": 10,
}

# Keep ORDER_ID and SEED fixed across candidates. More EPOCHS gives more optimization
# steps but costs more and can overfit. Explicit CUDA prevents silent CPU fallback.
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"

if __name__ == "__main__":
    run_manual(dataset="texture", method="er_ace", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
