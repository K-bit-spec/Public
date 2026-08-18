import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. 路径与指标分类配置（按 A/B/C/D 字母分组） =====================
"""
指标分类标准（避难所适宜性评价专用）
A类：应急资源与基础条件
B类：致灾因子（灾害风险）
C类：城市服务与交通设施
D类：承灾体（人口，核心服务对象）
"""
CONFIG = {
    "existing_path": r"D:\AICGIS\小文献代码结果\现有避难所指标提取_最终版.xlsx",
    "candidate_path": r"D:\AICGIS\小文献代码结果\候选避难所指标提取结果_ID136起.xlsx",
    "output_dir": r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\合并的避难所适宜性",

    # ===================== 【按 A/B/C/D 字母分类】正向指标（数值越高越适宜） =====================
    "positive_indicators": [
        # A 类：应急资源
        'A1_有效避难面积(m²)',
        # B 类：风险距离（越远越安全）
        'B1_水系距离(m)', 'B2_地质灾害距离(m)', 'B3_断裂带距离(m)', 'B4_森林距离(m)',
        # C 类：交通与设施
        'C1_交叉口数量',
        # D 类：服务人口（承灾体）
        'D1_500m服务人口'
    ],

    # ===================== 【按 A/B/C/D 字母分类】负向指标（数值越低越适宜） =====================
    "negative_indicators": [
        # A 类：应急资源距离
        'A2_医院距离(m)', 'A3_消防站距离(m)',
        # B 类：地形风险
        'B5_坡度(°)', 'B6_海拔(m)',
        # C 类：服务设施距离
        'C2_市场距离(m)'
    ]
}

# ===================== 2. 核心算法 =====================
def entropy_weight(data):
    """熵权法（EWM）客观赋权"""
    scaler = MinMaxScaler(feature_range=(0.001, 1))
    data_norm = scaler.fit_transform(data)
    p = data_norm / data_norm.sum(axis=0)
    e = -1 / np.log(len(data)) * (p * np.log(p)).sum(axis=0)
    d = 1 - e
    weights = d / d.sum()
    return dict(zip(data.columns, weights))

def build_hazard_target(df_norm, hazard_type):
    """
    【修正】灾害适宜性目标函数（避难所选址专用）
    所有场景均包含：D1人口 + C2市场 + 对应致灾因子 + 应急/交通
    完全匹配你前面确认的场景配置
    """
    if hazard_type == 'seismic':
        # 地震：断裂带(B3) + 面积(A1) + 交叉口(C1) + 人口(D1) + 市场(C2)
        return (
            df_norm['B3_断裂带距离(m)'] * 0.30
            + df_norm['A1_有效避难面积(m²)'] * 0.20
            + df_norm['C1_交叉口数量'] * 0.15
            + df_norm['D1_500m服务人口'] * 0.20
            + (1 - df_norm['C2_市场距离(m)']) * 0.15
        )

    elif hazard_type == 'geohazard':
        # 地质灾害：地质(B2)+坡度(B5)+海拔(B6) + 人口(D1) + 市场(C2)
        return (
            df_norm['B2_地质灾害距离(m)'] * 0.25
            + (1 - df_norm['B5_坡度(°)']) * 0.25
            + (1 - df_norm['B6_海拔(m)']) * 0.20
            + df_norm['D1_500m服务人口'] * 0.15
            + (1 - df_norm['C2_市场距离(m)']) * 0.15
        )

    elif hazard_type == 'flood':
        # 洪水：水系(B1)+海拔(B6) + 面积(A1) + 人口(D1) + 市场(C2)
        return (
            df_norm['B1_水系距离(m)'] * 0.30
            + (1 - df_norm['B6_海拔(m)']) * 0.25
            + df_norm['A1_有效避难面积(m²)'] * 0.15
            + df_norm['D1_500m服务人口'] * 0.15
            + (1 - df_norm['C2_市场距离(m)']) * 0.15
        )

    elif hazard_type == 'social_fire':
        # 火灾：森林(B4)+消防(A3)+医院(A2) + 人口(D1) + 市场(C2)
        return (
            df_norm['B4_森林距离(m)'] * 0.25
            + (1 - df_norm['A3_消防站距离(m)']) * 0.25
            + (1 - df_norm['A2_医院距离(m)']) * 0.20
            + df_norm['D1_500m服务人口'] * 0.15
            + (1 - df_norm['C2_市场距离(m)']) * 0.15
        )

def calculate_suitability(df, weights, neg_cols):
    """适宜性得分计算（标准化 + 负向指标取反）"""
    df_work = df[list(weights.keys())].copy()
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_work), columns=df_work.columns)

    # 负向指标：距离越近越适宜 → 1 - 标准化值
    for col in neg_cols:
        if col in df_norm.columns:
            df_norm[col] = 1 - df_norm[col]

    # 加权求和
    score = df_norm.apply(lambda row: sum(row[f] * weights[f] for f in weights), axis=1)
    return score, df_norm

# ===================== 3. 主程序 =====================
def main():
    # --- 1. 读取数据 ---
    df_exist = pd.read_excel(CONFIG["existing_path"])
    df_exist.columns = [c.strip() for c in df_exist.columns]

    all_features = CONFIG["positive_indicators"] + CONFIG["negative_indicators"]

    # 缺失字段检查
    missing = [f for f in all_features if f not in df_exist.columns]
    if missing:
        print(f"❌ 缺失列：{missing}")
        return

    # 缺失值填充
    df_exist[all_features] = df_exist[all_features].fillna(df_exist[all_features].median())

    # --- 2. 权重计算：EWM + RF + 组合赋权 ---
    hazard_types = ['seismic', 'geohazard', 'flood', 'social_fire']
    hazard_names = {
        'seismic': '地震',
        'geohazard': '地质灾害',
        'flood': '洪涝暴雨',
        'social_fire': '火灾与公卫'
    }

    weight_log = pd.DataFrame(index=all_features)
    print("--- 1. 熵权法权重计算 ---")
    w_entropy = entropy_weight(df_exist[all_features])
    weight_log['熵权法权重'] = weight_log.index.map(w_entropy)

    # 随机森林训练用标准化数据
    scaler = MinMaxScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(df_exist[all_features]), columns=all_features)
    final_weight_dict = {}

    print("--- 2. 随机森林重要性 + 组合权重 ---")
    for h_type in hazard_types:
        y_target = build_hazard_target(X_norm, h_type)
        rf = RandomForestRegressor(n_estimators=150, random_state=42)
        rf.fit(df_exist[all_features], y_target)
        w_rf = dict(zip(all_features, rf.feature_importances_))

        # 记录
        weight_log[f'RF重要性_{hazard_names[h_type]}'] = weight_log.index.map(w_rf)

        # 组合赋权（平均后归一化）
        combined_raw = {f: (w_entropy[f] + w_rf[f]) / 2 for f in all_features}
        total = sum(combined_raw.values())
        final_weight_dict[h_type] = {k: v / total for k, v in combined_raw.items()}
        weight_log[f'最终组合权重_{hazard_names[h_type]}'] = weight_log.index.map(final_weight_dict[h_type])

    # 导出权重表
    weight_log.to_excel(os.path.join(CONFIG["output_dir"], "权重计算结果对照表.xlsx"))

    # --- 3. 适宜性评价 ---
    datasets = [
        (df_exist, "现有避难所综合评价结果.xlsx", True),
        (pd.read_excel(CONFIG["candidate_path"]), "候选避难所综合评价结果.xlsx", False)
    ]

    for df, out_name, is_existing in datasets:
        print(f"\n正在评价：{out_name}")
        df.columns = [c.strip() for c in df.columns]
        df[all_features] = df[all_features].fillna(df_exist[all_features].median())
        detail_data = []

        # 逐个灾害计算
        for h_type in hazard_types:
            h_name = hazard_names[h_type]
            score, norm_df = calculate_suitability(df, final_weight_dict[h_type], CONFIG["negative_indicators"])
            df[f'{h_name}适宜性得分'] = score

            # 等级划分
            df[f'{h_name}适宜性等级'] = pd.qcut(
                df[f'{h_name}适宜性得分'], q=5,
                labels=['低适宜性', '较低适宜性', '一般适宜性', '较高适宜性', '高适宜性']
            )

            # 记录计算过程
            if is_existing:
                tmp = norm_df.head(5).copy()
                tmp['灾害类型'] = h_name
                tmp['适宜性得分'] = score.head(5)
                detail_data.append(tmp)

        # 综合得分（按灾害重要性赋值）
        df['综合适宜性得分'] = (
            df['地震适宜性得分'] * 0.35
            + df['地质灾害适宜性得分'] * 0.35
            + df['洪涝暴雨适宜性得分'] * 0.15
            + df['火灾与公卫适宜性得分'] * 0.15
        )

        df['综合适宜性等级'] = pd.qcut(
            df['综合适宜性得分'], q=5,
            labels=['低适宜性', '较低适宜性', '一般适宜性', '较高适宜性', '高适宜性']
        )

        # 保存结果
        df.to_excel(os.path.join(CONFIG["output_dir"], out_name), index=False)
        if is_existing and detail_data:
            pd.concat(detail_data).to_excel(os.path.join(CONFIG["output_dir"], "计算过程细节抽样.xlsx"))

    print(f"\n✅ 全部完成！文件保存在：{CONFIG['output_dir']}")

if __name__ == "__main__":
    main()