import pandas as pd
import sys


def analyze_csv(file_path: str, output_path: str = None):
    """
    读取 CSV 文件，计算每列的中位数、平均数、标准差。

    参数:
        file_path:   输入 CSV 文件路径
        output_path: 可选，将结果保存为 CSV
    """
    # 读取 CSV（自动识别分隔符；如遇编码问题可加 encoding='gbk'）
    df = pd.read_csv(file_path)

    # 仅对数值列做统计（非数值列会被自动忽略）
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("没有可统计的数值列。")
        return None

    stats = pd.DataFrame({
        "中位数 (median)": numeric_df.median(),
        "平均数 (mean)":   numeric_df.mean(),
        "标准差 (std)":    numeric_df.std(),  # 默认样本标准差 ddof=1
        "非空计数 (count)": numeric_df.count(),
        "缺失值 (missing)": numeric_df.isna().sum(),
    })

    # 格式化显示
    print("\n各列统计结果：")
    print(stats.round(4).to_string())

    if output_path:
        stats.round(6).to_csv(output_path, encoding="utf-8-sig")
        print(f"\n结果已保存到: {output_path}")

    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_csv.py <input.csv> [output.csv]")
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_csv(in_file, out_file)
