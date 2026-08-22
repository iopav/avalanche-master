from common import run_manual

PARAMETERS = {
    # SGD 学习率。增大可加快学习，但可能压过 EWC 正则保护；减小更稳，
    # 但可能使新任务欠拟合。
    "learning_rate": 0.1,
    # Fisher 加权旧参数惩罚强度。增大可增强稳定性但限制新任务学习；
    # 减小可提高可塑性，但通常会增加遗忘。
    "ewc_lambda": 0.4,
    # "separate" 为每个历史 Experience 保留一组 Fisher 和参数快照。
    # 正式协议中保持不变；修改 mode 会同时改变正则化行为和持久存储。
    "mode": "separate",
}

# 比较候选参数时保持 ORDER_ID 和 SEED 不变。增大 EPOCHS 可能改善当前任务拟合，
# 但也可能增加遗忘。显式指定 CUDA 可防止静默回退到 CPU。
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 3, "cuda"

if __name__ == "__main__":
    run_manual(dataset="texture", method="ewc", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
