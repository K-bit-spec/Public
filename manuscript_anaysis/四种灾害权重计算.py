import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# ===================== 路径 =====================
INPUT_PATH = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\合并的避难所适宜性\现有避难所综合特征.xlsx"
OUTPUT_PATH = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果\灾害权重结果.xlsx"

# ===================== 四类灾害特征 =====================
HAZARD_INDICATORS = {
    "地震": ['A1', 'A2', 'B3', 'C1', 'C2', 'D1'],
    "地质灾害": ['A1', 'B2', 'B5', 'B6', 'C1', 'C2', 'D1'],
    "洪涝暴雨": ['A3', 'B1', 'B6', 'C1', 'C2', 'D1'],
    "火灾与公卫": ['A2', 'A3', 'B4', 'C1', 'C2', 'D1']
}

KEYWORDS = {
    'A1': '面积', 'A2': '医院', 'A3': '消防',
    'B1': '水系', 'B2': '地质', 'B3': '断裂', 'B4': '森林', 'B5': '坡度', 'B6': '海拔',
    'C1': '交叉口', 'C2': '市场', 'D1': '人口'
}

# ===================== 自动匹配列名 =====================
def find_real_col(df_cols, code):
    kw = KEYWORDS[code]
    for c in df_cols:
        if kw in c:
            return c
    return None

# ===================== 熵权+RF权重计算 =====================
def calc_weights(df, cols):
    # ---------------- 修复核心：只保留数值列 ----------------
    df_sub = df[cols].copy()
    # 强制转数字，无法转的变成 NaN
    for c in df_sub.columns:
        df_sub[c] = pd.to_numeric(df_sub[c], errors='coerce')
    # 删除全为空的行，再填充中位数
    df_sub = df_sub.dropna(how='all').fillna(df_sub.median())
    # ------------------------------------------------------

    scaler = MinMaxScaler((0.001, 1))
    norm = scaler.fit_transform(df_sub)

    # 熵权
    p = norm / norm.sum(axis=0)
    e = -1 / np.log(len(df_sub)) * (p * np.log(p + 1e-10)).sum(axis=0)
    w_entropy = (1 - e) / (1 - e).sum()

    # 随机森林
    rf = RandomForestRegressor(n_estimators=30, random_state=42)
    rf.fit(norm, norm.mean(axis=1))
    w_rf = rf.feature_importances_ / rf.feature_importances_.sum()

    # 综合权重
    w_comb = (w_entropy + w_rf) / 2
    return w_entropy, w_rf, w_comb

# ===================== 主程序 =====================
if __name__ == "__main__":
    df = pd.read_excel(INPUT_PATH)
    all_results = []

    for hazard_name, codes in HAZARD_INDICATORS.items():
        real_cols = [find_real_col(df.columns, c) for c in codes]
        real_cols = [c for c in real_cols if c is not None]

        if len(real_cols) == 0:
            print(f"⚠️ {hazard_name} 无匹配指标，跳过")
            continue

        try:
            we, wr, wc = calc_weights(df, real_cols)
            for i, (code, col, w) in enumerate(zip(codes, real_cols, wc)):
                all_results.append([hazard_name, code, col, we[i], wr[i], wc[i]])
        except Exception as e:
            print(f"❌ {hazard_name} 计算失败：{str(e)}")

    # 保存权重
    res_df = pd.DataFrame(all_results, columns=[
        "灾害类型", "指标代码", "真实列名", "熵权", "RF权重", "综合权重"
    ])
    res_df.to_excel(OUTPUT_PATH, index=False)
    print("✅ 权重计算完成，已保存到：", OUTPUT_PATH)