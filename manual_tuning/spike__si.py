from common import run_manual

PARAMETERS = {
    # SGD 学习率。增大可加快当前任务学习，但会扩大参数移动并增加旧知识漂移；
    # 减小通常更稳定，但在固定训练轮数内可能欠拟合。
    "learning_rate": 0.01,
    # SI 正则化强度。增大可更强地保护高重要性参数，但会限制新任务学习；
    # 建议按数量级比较，例如 1e-5、1e-4、1e-3、1e-2。
    "si_lambda": 1e-3,
    # 参数重要性分母中的阻尼项。增大将整体压低重要性估计，减小会增强保护但也会
    # 放大小位移参数的敏感性。先保持 Avalanche 默认值 1e-7，不与 lambda 同时搜索。
    "eps": 1e-5,
}

# ORDER_ID 决定类别到达顺序，SEED 控制训练随机性；比较候选参数时保持二者不变。
# EPOCHS 与正式轻量协议一致，DEVICE 固定为 CUDA。手工调参不会计算 FLOPs 或保存 JSON。
ORDER_ID, SEED, EPOCHS, DEVICE = 6, 62, 1, "cuda"

if __name__ == "__main__":
    run_manual(dataset="spike", method="si", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
