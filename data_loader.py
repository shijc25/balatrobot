import numpy as np
import os
from tqdm import tqdm # 如果没安装，请 pip install tqdm

data_dir = "/root/autodl-tmp/evaluator_data_3"

def inspect_and_count():
    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if not files:
        print(f"在 {data_dir} 中没找到任何 .npz 文件。")
        return

    print(f"找到 {len(files)} 个数据包。正在统计汇总...")

    total_samples = 0
    total_wins = 0
    
    # 1. 抽样检查第一个文件的格式
    sample_path = os.path.join(data_dir, files[0])
    with np.load(sample_path) as data:
        print("\n" + "="*30)
        print(f"样本文件结构检视: {files[0]}")
        print("-"*30)
        for key in data.files:
            print(f"Key: {key:10} | Shape: {data[key].shape} | Dtype: {data[key].dtype}")
        
        # 假设以 label 的第一维作为样本数
        num_in_file = data['label'].shape[0]
        print(f"\n该文件包含样本数: {num_in_file}")
        print("="*30 + "\n")

    # 2. 遍历所有文件统计总量
    # tqdm 可以显示进度条，1000万数据大概几十秒统计完
    for f in tqdm(files, desc="统计进度"):
        try:
            path = os.path.join(data_dir, f)
            # mmap_mode='r' 可以只读取索引不加载内容，极快
            with np.load(path, mmap_mode='r') as data:
                labels = data['label']
                total_samples += labels.shape[0]
                total_wins += np.sum(labels)
        except Exception as e:
            print(f"文件 {f} 损坏或读取失败: {e}")

    # 3. 最终报告
    print("\n" + "!"*30)
    print("数据工厂运行报告")
    print("-"*30)
    print(f"总文件数:   {len(files)}")
    print(f"总样本行数: {total_samples}")
    print(f"平均轮数: {total_wins/total_samples}" if total_samples > 0 else "0")
    print("!"*30)

if __name__ == "__main__":
    inspect_and_count()