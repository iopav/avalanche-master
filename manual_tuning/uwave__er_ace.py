from common import run_manual

PARAMETERS = {
    # SGD 学习率。增大可加快适应，但可能破坏旧类；减小可缩小参数更新，
    # 但可能使新类欠拟合。
    "learning_rate": 0.1,
    # 逻辑回放样本总容量。增大通常有助于保留旧类，但会增加持久存储和样本管理开销。
    "memory_size": 2000,
    # 每个当前数据 minibatch 配对的回放样本数。增大可强化回放，但会增加计算量，
    # 并可能降低当前任务在一次更新中的权重。
    "batch_size_mem": 200,
}

# 比较候选参数时保持 ORDER_ID 和 SEED 不变。增大 EPOCHS 会增加优化步数，
# 同时增加耗时和过拟合风险。显式指定 CUDA 可防止静默回退到 CPU。
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 10, "cuda"

if __name__ == "__main__":
    run_manual(dataset="uwave", method="er_ace", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
