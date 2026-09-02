from common import run_manual


PARAMETERS = {
    # Texture 序列较长；显存不足时只减 train_mb_size，不改变回放总容量。
    "learning_rate": 0.1,
    "train_mb_size": 32,
    "memory_size": 2000,
    "contrast_factor": 1.0,
    "contrast_kd_factor": 2.0,
    "aux_factor": 2.0,
    "trans_cls_factor": 0.005,
    "transfer_factor": 1.0,
    "infonce_temp": 0.2,
    "infonce_kd_temp": 0.2,
    "kd_temp": 2.0,
    "proj_hidden_dim": 2048,
    "proj_output_dim": 1024,
    "interpolation_factor": 0.95,
    "attention_heads": 8,
}

ORDER_ID, SEED, EPOCHS, DEVICE = 6, 62, 3, "cuda"

if __name__ == "__main__":
    run_manual(dataset="texture", method="tagfex", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
