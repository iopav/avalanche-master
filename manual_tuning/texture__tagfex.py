from common import run_manual_entry


PARAMETERS = {
    "optimizer": "SGD",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "foreach": False,
    "train_mb_size": 8,
    "eval_mb_size": 8,
    "num_workers": 0,
    "init_epochs": 60,
    "inc_epochs": 40,
    "init_lr": 0.1,
    "inc_lr": 0.1,
    "init_weight_decay": 5e-4,
    "inc_weight_decay": 2e-4,
    "init_milestones": (60, 120, 170),
    "inc_milestones": (80, 120, 150),
    "gamma": 0.1,
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

DATA_MODE = "mini"
BACKBONE_ID = "resnet18_cifar"
ORDER_ID, SEED, EPOCHS, DEVICE = 6, 62, 40, "cuda"

if __name__ == "__main__":
    run_manual_entry(dataset="texture", method="tagfex", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE, data_mode=DATA_MODE, backbone_id=BACKBONE_ID)
