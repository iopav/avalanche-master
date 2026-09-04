from common import run_manual_entry

PARAMETERS = {
    "optimizer": "Adam",
    # Adam 学习率。增大可加快新类学习，但可能扩大表征漂移和最新类偏置；
    # 减小通常更稳定，但在固定训练轮数内可能欠拟合。
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    # 旧类知识蒸馏损失的权重。增大可加强旧类输出约束，但过大会妨碍新类学习；
    # 建议先固定 temperature=2，比较 0.5、1.0、2.0、5.0。
    "alpha": 1.0,
    # 蒸馏 softmax 温度。增大可软化教师分布并暴露更多旧类相对关系，
    # 但过高会使分布过平；先使用 2.0，必要时再比较 4.0。
    "temperature": 2.0,
}

# ORDER_ID 决定类别到达顺序，SEED 控制训练随机性；比较候选参数时保持二者不变。
# EPOCHS 与轻量协议一致，DEVICE 固定为 CUDA。手工调参不会计算 FLOPs 或保存 JSON。
DATA_MODE = "mini"
BACKBONE_ID = "resnet18_cifar"
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"

if __name__ == "__main__":
    run_manual_entry(dataset="uwave", method="lwf", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE, data_mode=DATA_MODE, backbone_id=BACKBONE_ID)
