import os
import argparse  # 新增：用于解析命令行参数
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def plot_tensorboard_log(log_dir, save_path="loss_curve.png"):
    log_file = None
    # 找到最新的日志文件
    for f in os.listdir(log_dir):
        if "events.out.tfevents" in f:
            log_file = os.path.join(log_dir, f)
            break

    # 容错：如果文件夹路径写错了，或者里面没有日志文件
    if log_file is None:
        print(f"错误：在 '{log_dir}' 目录下没有找到 TensorBoard 日志文件！请检查路径。")
        return

    # 加载日志
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()

    # 提取 Loss/Total
    if 'Loss/Total' in ea.scalars.Keys():
        loss_events = ea.scalars.Items('Loss/Total')
        epochs = [e.step for e in loss_events]
        losses = [e.value for e in loss_events]

        # 设置全局字体样式
        plt.rcParams['font.family'] = 'sans-serif'

        plt.figure(figsize=(8, 5))

        # 画图参数
        plt.plot(epochs, losses, marker='o', linestyle='-', color='b', linewidth=2, markersize=4)

        # 标题和坐标轴
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Total Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存 600 DPI 高清图
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"600 DPI 的高清 Loss 曲线已保存到: {save_path}")
    else:
        print("错误：日志文件中没有找到 'Loss/Total' 数据！")


# ==========================================
# 新增：接收命令行参数的逻辑
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制并导出 600 DPI 的 TensorBoard Loss 曲线")

    # 定义必须传入的输入路径
    parser.add_argument("--log_dir", type=str, required=True, help="TensorBoard 日志所在的文件夹路径")

    # 定义输出图片路径（如果不写，默认叫 high_res_loss.png）
    parser.add_argument("--save_path", type=str, default="high_res_loss.png", help="保存的高清图片路径和名称")

    args = parser.parse_args()

    # 将命令行获取到的路径传给画图函数
    plot_tensorboard_log(args.log_dir, args.save_path)