from common import run_manual

# learning_rate：SGD 学习率。增大可加快每个任务的学习，但过大可能造成震荡；
# 减小通常更稳定，但在固定训练轮数内可能欠拟合。
PARAMETERS = {"learning_rate": 0.01}

# ORDER_ID 决定类别到达顺序，SEED 控制训练中的全部随机性；比较候选参数时应保持二者不变。
# 增大 EPOCHS 可提高拟合程度，但会增加耗时和过拟合风险。DEVICE 显式指定 CUDA，
# 因此 GPU 不可用时会直接报错，不会静默回退到 CPU。
ORDER_ID, SEED, EPOCHS, DEVICE = 1, 62, 50, "cuda"

if __name__ == "__main__":
    run_manual(dataset="spike", method="cwr_star", parameters=PARAMETERS, order_id=ORDER_ID, seed=SEED, epochs=EPOCHS, device=DEVICE)
