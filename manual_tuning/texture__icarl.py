from common import run_manual_entry

PARAMETERS = {
    "optimizer": "Adam",
    # Adam 学习率。增大可快速适应，但可能扰动旧特征；减小更保守，
    # 但可能使新类欠拟合。
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    # 所有已见类别共享的 exemplar 总上限。增大通常可改善 NCM 原型和旧类保持，
    # 但会增加存储、样本选择和回放计算成本。
    "memory_size": 2000,
    # 锁定协议参数：True 表示固定总预算并在已见类别间分配。
    # 本地适配器只实现此模式，因此不要调节该字段。
    "fixed_memory": True,
}

# 比较候选参数时保持 ORDER_ID 和 SEED 不变。增大 EPOCHS 可改善表征学习，
# 但会增加回放计算量和过拟合风险。此脚本必须使用 CUDA。
DATA_MODE = "mini"
BACKBONE_ID = "resnet18_cifar"
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 30, "cuda"

if __name__ == "__main__":
    run_manual_entry(dataset="texture", method="icarl", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE, data_mode=DATA_MODE, backbone_id=BACKBONE_ID)


# D:\workspace\Avalanche\avalanche-master\avalanche\training\plugins\evaluation.py:94: UserWarning: No loggers specified, metrics will not be logged
#   warnings.warn("No loggers specified, metrics will not be logged")
# dataset=texture method=icarl order_id=1 seed=62 epochs=30 device=cuda:0 device_name=NVIDIA GeForce RTX 4060 Laptop GPU parameters={'learning_rate': 0.1, 'memory_size': 2000, 'fixed_memory': True}

# 准确率矩阵（%，行表示完成训练的任务，列表示测试任务）
# ------------------------------------------------------------------------------------------------------------
#        训练后 |     T01     T02     T03     T04     T05     T06     T07     T08     T09     T10 | 已见任务均值
# ------------------------------------------------------------------------------------------------------------
#        T01 |  100.00       -       -       -       -       -       -       -       -       - |     100.00
#        T02 |  100.00   85.71       -       -       -       -       -       -       -       - |      92.86
#        T03 |  100.00   85.71  100.00       -       -       -       -       -       -       - |      95.24
#        T04 |  100.00   85.71  100.00  100.00       -       -       -       -       -       - |      96.43
#        T05 |  100.00   85.71  100.00  100.00  100.00       -       -       -       -       - |      97.14
#        T06 |  100.00   85.71  100.00  100.00  100.00  100.00       -       -       -       - |      97.62
#        T07 |  100.00   85.71  100.00  100.00  100.00  100.00   94.12       -       -       - |      97.12
#        T08 |  100.00   85.71  100.00  100.00  100.00  100.00   94.12  100.00       -       - |      97.48
#        T09 |  100.00   85.71  100.00  100.00  100.00  100.00   88.24  100.00  100.00       - |      97.11
#        T10 |  100.00   85.71  100.00  100.00  100.00  100.00   88.24  100.00  100.00  100.00 |      97.39
# ------------------------------------------------------------------------------------------------------------
# 最终平均准确率（最后一行）: 97.39%