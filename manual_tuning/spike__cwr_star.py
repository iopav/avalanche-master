from common import run_manual_entry

# learning_rate：Adam 学习率。增大可加快每个任务的学习，但过大可能造成震荡；
# 减小通常更稳定，但在固定训练轮数内可能欠拟合。
PARAMETERS = {
    "optimizer": "SGD",
    "learning_rate": 0.05,
    "momentum": 0.9,
    "weight_decay": 0.0005,
}

# ORDER_ID 决定类别到达顺序，SEED 控制训练中的全部随机性；比较候选参数时应保持二者不变。
# 增大 EPOCHS 可提高拟合程度，但会增加耗时和过拟合风险。DEVICE 显式指定 CUDA，
# 因此 GPU 不可用时会直接报错，不会静默回退到 CPU。
DATA_MODE = "mini"
BACKBONE_ID = "resnet18_cifar"
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 30, "cuda"

if __name__ == "__main__":
    run_manual_entry(dataset="spike", method="cwr_star", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE, data_mode=DATA_MODE, backbone_id=BACKBONE_ID)
