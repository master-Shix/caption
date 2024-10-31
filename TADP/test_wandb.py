import wandb

def test_wandb():
    try:
        # 初始化一个新的 wandb 运行
        wandb.init(project="test-project", name="test-run")

        # 记录一些数据
        for i in range(10):
            wandb.log({"metric": i})

        # 完成运行
        wandb.finish()
        print("wandb 测试成功!")
    except Exception as e:
        print(f"wandb 测试失败: {e}")

if __name__ == "__main__":
    test_wandb()