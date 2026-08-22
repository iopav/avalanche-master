from common import run_manual

PARAMETERS = {
    # SGD 学习率。增大可快速适应，但可能扰动旧特征；减小更保守，
    # 但可能使新类欠拟合。
    "learning_rate": 0.1,
    # 所有已见类别共享的 exemplar 总上限。增大通常可改善 NCM 原型和旧类保持，
    # 但会增加存储、样本选择和回放计算成本。
    "memory_size": 2000,
    # 锁定协议参数：True 表示固定总预算并在已见类别间分配。
    # 本地适配器只实现此模式，因此不要调节该字段。
    "fixed_memory": True,
}

# 比较候选参数时保持 ORDER_ID 和 SEED 不变。增大 EPOCHS 可改善表征学习，
# 但会增加回放计算量和过拟合风险。此脚本必须使用 CUDA。
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"

if __name__ == "__main__":
    run_manual(dataset="texture", method="icarl", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
