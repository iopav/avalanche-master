from common import run_manual


PARAMETERS = {
    # 先调学习率；其余值保持 TagFex 论文实现及本地原生配置的损失比例。
    "learning_rate": 0.03,
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

ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 10, "cuda"

if __name__ == "__main__":
    run_manual(dataset="spike", method="tagfex", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
