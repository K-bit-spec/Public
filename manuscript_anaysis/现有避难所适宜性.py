import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. 配置路径与参数 =====================
CONFIG = {
    "feature_table": r"D:/AICGIS/小文献代码结果/避难所指标提取_最终版.xlsx",
    "output_dir": r"D:/AICGIS/小文献代码结果/",
    # 指标分类：正向(越大越好) vs 负向(越小越好)
    "positive_cols": [
        'A1_有效避难面积(m²)', 'B1_水系距离(m)', 'B2_地质灾害距离(m)',
        'B3_断裂带距离(m)', 'B4_森林距离(m)', 'C1_交叉口数量', 'D1_500m服务人口'
    ],
    "negative_cols": [
        'A2_医院距离(m)', 'A2_消防站距离(m)', 'B1_海拔(m)',
        'B5_坡度(°)', 'C2_市场距离(m)'
    ]
}


# ===================== 2. 核心算法函数 =====================

def entropy_weight(data):
    """熵权法计算客观权重"""
    # 标准化到[0.001, 1] 避免log(0)
    scaler = MinMaxScaler(feature_range=(0.001, 1))
    data_norm = scaler.fit_transform(data)
    # 计算比重
    p = data_norm / data_norm.sum(axis=0)
    # 计算熵值
    e = -1 / np.log(len(data)) * (p * np.log(p)).sum(axis=0)
    # 计算权重
    d = 1 - e
    weights = d / d.sum()
    return dict(zip(data.columns, weights))


def build_proxy_target(df_norm):
    """
    构建代理目标函数 (用于驱动随机森林计算重要性)
    逻辑：综合考虑面积(正)、灾害距离(正)、医院距离(负)、坡度(负)
    """
    target = (
            df_norm['A1_有效避难面积(m²)'] * 0.25 +
            df_norm['B2_地质灾害距离(m)'] * 0.20 +
            (1 - df_norm['A2_医院距离(m)']) * 0.15 +
            (1 - df_norm['B5_坡度(°)']) * 0.15 +
            df_norm['C1_交叉口数量'] * 0.15 +
            df_norm['D1_500m服务人口'] * 0.10
    )
    return target


def calculate_final_scores(df, weights):
    """计算加权适宜性得分"""
    df_norm = df[list(weights.keys())].copy()
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_norm), columns=df_norm.columns)

    # 负向指标反转 (1 - x)
    for col in CONFIG["negative_cols"]:
        if col in df_norm.columns:
            df_norm[col] = 1 - df_norm[col]

    # 加权求和
    score = df_norm.apply(lambda row: sum(row[f] * weights[f] for f in weights), axis=1)
    return score


# ===================== 3. 执行主流程 =====================

def main():
    if not os.path.exists(CONFIG["feature_table"]):
        print(f"错误：找不到特征表 {CONFIG['feature_table']}")
        return

    # 1. 读取数据
    print("正在读取特征数据...")
    df = pd.read_excel(CONFIG["feature_table"])

    # 确定特征列 (排除ID和名称)
    all_features = CONFIG["positive_cols"] + CONFIG["negative_cols"]
    available_features = [f for f in all_features if f in df.columns]

    X = df[available_features].fillna(df[available_features].median())

    # 2. 计算熵权法权重
    print("计算熵权法权重...")
    w_entropy = entropy_weight(X)

    # 3. 计算随机森林权重
    print("计算随机森林重要性权重...")
    scaler = MinMaxScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 构建代理目标
    y_proxy = build_proxy_target(X_norm)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y_proxy)

    w_rf = dict(zip(X.columns, rf.feature_importances_))

    # 4. 组合权重 (简单平均)
    print("生成最终组合权重...")
    final_weights = {}
    for f in X.columns:
        final_weights[f] = (w_entropy[f] + w_rf[f]) / 2

    # 归一化权重总和为1
    total_w = sum(final_weights.values())
    final_weights = {k: v / total_w for k, v in final_weights.items()}

    # 打印权重结果供参考
    print("\n--- 各指标最终权重 ---")
    for k, v in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    # 5. 计算得分与分级
    print("\n正在计算适宜性得分...")
    df['适宜性得分'] = calculate_final_scores(df, final_weights)

    # 分级 (等间距或分位数)
    print("正在进行等级划分...")
    df['适宜性等级'] = pd.qcut(df['适宜性得分'], q=5,
                               labels=['极低适宜性', '较低适宜性', '一般适宜性', '较高适宜性', '极高适宜性'])

    # 6. 保存结果
    output_path = os.path.join(CONFIG["output_dir"], "避难所适宜性评价结果.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\n✅ 处理完成！结果已保存至: {output_path}")


if __name__ == "__main__":
    main()