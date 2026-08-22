from common import run_manual


PARAMETERS = {
    # SGD 学习率。余弦分类头消除了权重模长差异，但学习率仍控制骨干和分类方向的更新速度；
    # 增大可能提高新类拟合并加剧旧特征漂移，减小通常更稳定但可能欠拟合。
    "learning_rate": 0.01,
    # Fisher 加权的参数保持强度。增大可加强旧知识保护，但过大会导致新类学不动；
    # 减小可提高可塑性，但可能重新出现“新类接近100%、旧类遗忘”的现象。
    # 建议保持其他参数不变，先比较 0.1、0.3、1.0、3.0、10.0。
    "ewc_lambda": 1.0,
    # separate 为每个历史 Experience 保存独立 Fisher 和参数快照，存储随任务数增长；
    # 当前兼容实现只允许 separate，避免 online 模式还需额外搜索 decay_factor。
    "mode": "separate",
}

# ORDER_ID 和 SEED 决定类别顺序、训练/初始化随机性，比较参数时必须保持不变。
# EPOCHS 增大可改善当前任务拟合，也会增加旧类漂移和EWC Fisher计算时间。
# DEVICE 固定为 CUDA；此入口不统计 FLOPs、不保存 JSON，只打印完整准确率矩阵。
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"


if __name__ == "__main__":
    run_manual(
        dataset="spike",
        method="ewc_cosine",
        parameters=PARAMETERS,
        order_id=ORDER_ID,
        seed=SEED,
        epochs=EPOCHS,
        device=DEVICE,
    )
