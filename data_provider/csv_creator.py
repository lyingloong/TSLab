import pandas as pd
import numpy as np
import datetime

root_path = '../dataset/mine/'

def generate_synthetic_data(
        start_date="2016-07-01 00:00:00",
        periods=96,  # 每天 4 个时间点，共 24 小时
        days=30,  # 生成 30 天的数据
        features=6,  # 特征数量
        feature_type="linear",  # 特征类型：'linear', 'polynomial', 'random'
        output_type="linear",  # 输出类型：'linear', 'random'
        noise_level=0.1  # 噪声水平
):
    # 生成时间序列
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    timestamps = [start + datetime.timedelta(hours=i) for i in range(periods * days)]

    # 初始化数据字典
    data = {"date": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]}

    # 生成特征数据
    for i in range(features):
        if feature_type == "linear":
            # 线性增长
            data[f"feature{i + 1}"] = np.linspace(1, periods * days, periods * days) + np.random.normal(0, noise_level,
                                                                                                        periods * days)
        elif feature_type == "polynomial":
            # 多项式增长
            data[f"feature{i + 1}"] = np.polyval([0.001, -0.05, 10], np.arange(periods * days)) + np.random.normal(0,
                                                                                                                   noise_level,
                                                                                                                   periods * days)
        elif feature_type == "random":
            # 随机数据
            data[f"feature{i + 1}"] = np.random.rand(periods * days) * 10 + np.random.normal(0, noise_level,
                                                                                             periods * days)

    # 生成输出数据
    if output_type == "linear":
        data["OT"] = np.linspace(10, 50, periods * days) + np.random.normal(0, noise_level, periods * days)
    elif output_type == "random":
        data["OT"] = np.random.rand(periods * days) * 40 + 10 + np.random.normal(0, noise_level, periods * days)

    # 转换为 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    return df


# 示例：生成 30 天的数据，6 个特征，特征类型为线性，输出类型为线性
df = generate_synthetic_data(
    start_date="2016-07-01 00:00:00",
    days=30,
    features=6,
    feature_type="linear",
    output_type="linear",
    noise_level=0.1
)

# 保存为 CSV 文件
df.to_csv(root_path+"linear_1.csv", index=False)
print("数据已生成并保存为 linear_1.csv")