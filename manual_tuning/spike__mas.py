from common import run_manual


PARAMETERS = {
    # SGD 学习率。增大可以加快新任务拟合，但也会放大重要参数漂移；
    # 减小通常更稳定，但在固定epoch内可能使当前类欠拟合。
    "learning_rate": 0.01,
    # MAS 参数保持惩罚权重。增大可加强旧知识保护，但过大会阻止新类学习；
    # 减小可提高可塑性，但可能重新出现旧类快速遗忘。建议先比较 0.1、0.3、1、3、10。
    "lambda_reg": 10,
    # 历史重要度的混合权重：new = alpha * old + (1-alpha) * current。
    # 增大更重视历史任务，减小更重视最近任务；必须位于[0,1]，建议比较0.3、0.5、0.7、0.9。
    "alpha": 0.9,
}

# ORDER_ID与SEED决定类别顺序和随机初始化，比较参数时必须保持不变。
# MAS在每个Experience训练后额外遍历一次当前训练数据计算重要度，增加EPOCHS不会改变这一次额外遍历，
# 但会增加常规训练耗时。DEVICE固定为CUDA；此入口不统计FLOPs、不保存JSON，只打印准确率矩阵。
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"


if __name__ == "__main__":
    run_manual(
        dataset="spike",
        method="mas",
        parameters=PARAMETERS,
        order_id=ORDER_ID,
        seed=SEED,
        epochs=EPOCHS,
        device=DEVICE,
    )
