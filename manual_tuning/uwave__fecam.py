from common import run_manual

PARAMETERS = {
    # 仅用于第一个 Experience 的 SGD 学习率。增大可更快学习即将冻结的表征，
    # 但过大可能不稳定；减小则可能使首任务表征欠拟合。
    "learning_rate": 0.03,
    # Tukey 幂变换开关。共享特征向量可能含负值，负数做分数次幂会产生 NaN，
    # 因此当前保持 False。
    "tukey": False,
    # 协方差正则化开关。关闭后直接使用原始协方差；类别样本较少时可能噪声较大或奇异。
    "shrinkage": True,
    # 对角线岭项强度。增大可改善可逆性，但过大会抹平类别特有方差；
    # 减小可保留方差信息，但矩阵可能病态。
    "shrink1": 0.5,
    # 非对角收缩强度。增大可更强地平滑特征相关性；过大会扭曲类别协方差，
    # 过小则可能保留较多噪声。
    "shrink2": 0.5,
    # 是否把协方差转换到相关系数尺度。True 可降低通道尺度差异的影响；
    # False 保留绝对方差，可能偏向高方差特征维度。
    "covnorm": True,
}

# EPOCHS 只控制第一个 Experience 的训练；后续 Experience 使用冻结的 backbone，
# 并执行一次统计量计算。比较候选参数时保持 ORDER_ID/SEED 不变，此脚本必须使用 CUDA。
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 40, "cuda"

if __name__ == "__main__":
    run_manual(dataset="uwave", method="fecam", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
