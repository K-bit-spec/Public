# -*- coding: utf-8 -*-
"""
EWM-RF组合比例敏感性分析
Figure 6: 不同α值下组合权重对比
图例左边界对齐图(c)左边界，右边界对齐图(d)右边界
图例外框长度延长至三倍（内部元素大小不变，间距变宽）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os
import warnings

warnings.filterwarnings('ignore')

# ===================== 路径配置 =====================
INPUT_PATH = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果\优化避难所得分结果\135现有避难所得分.xlsx"
OUTPUT_DIR = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果（修正版）\EWM&RF_Validation\F_Validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 全局字体设置 =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

COLORS_ALPHA = {
    0.3: '#a6bddb',
    0.5: '#1f77b4',
    0.7: '#bdbdbd'
}
EDGE_COLORS = {
    0.3: '#a6bddb',
    0.5: '#d62728',
    0.7: '#bdbdbd'
}
LINE_WIDTHS = {
    0.3: 0.5,
    0.5: 2.5,
    0.7: 0.5
}

HAZARD_CONFIG = {
    "Earthquake": {
        "indicator_cols": ['A1_有效避难面积(m²)', 'A2_医院距离(m)', 'B3_断裂带距离(m)',
                           'C1_交叉口数量', 'C2_市场距离(m)', 'D1_500m服务人口'],
        "score_col": '地震适宜性得分'
    },
    "Geological hazard": {
        "indicator_cols": ['A1_有效避难面积(m²)', 'B2_地质灾害距离(m)', 'B5_坡度(°)',
                           'B6_海拔(m)', 'C1_交叉口数量', 'C2_市场距离(m)', 'D1_500m服务人口'],
        "score_col": '地质灾害适宜性得分'
    },
    "Flood": {
        "indicator_cols": ['A3_消防站距离(m)', 'B1_水系距离(m)', 'B6_海拔(m)',
                           'C1_交叉口数量', 'C2_市场距离(m)', 'D1_500m服务人口'],
        "score_col": '洪涝暴雨适宜性得分'
    },
    "Fire": {
        "indicator_cols": ['A2_医院距离(m)', 'A3_消防站距离(m)', 'B4_森林距离(m)',
                           'C1_交叉口数量', 'C2_市场距离(m)', 'D1_500m服务人口'],
        "score_col": '火灾与公卫适宜性得分'
    }
}


def get_short_name(col):
    mapping = {
        'A1_有效避难面积(m²)': 'A1',
        'A2_医院距离(m)': 'A2',
        'A3_消防站距离(m)': 'A3',
        'B1_水系距离(m)': 'B1',
        'B2_地质灾害距离(m)': 'B2',
        'B3_断裂带距离(m)': 'B3',
        'B4_森林距离(m)': 'B4',
        'B5_坡度(°)': 'B5',
        'B6_海拔(m)': 'B6',
        'C1_交叉口数量': 'C1',
        'C2_市场距离(m)': 'C2',
        'D1_500m服务人口': 'D1'
    }
    return mapping.get(col, col)


def calc_ewm_rf_weights(df, indicator_cols, target_col, alpha):
    X_raw = df[indicator_cols].copy()
    y = df[target_col].copy()

    for c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce')
    X_raw = X_raw.fillna(X_raw.median())
    y = pd.to_numeric(y, errors='coerce').fillna(y.median())

    scaler = MinMaxScaler((0.001, 1))
    X = scaler.fit_transform(X_raw)

    p = X / X.sum(axis=0)
    e = -1 / np.log(len(X)) * (p * np.log(p + 1e-10)).sum(axis=0)
    w_ewm = (1 - e) / (1 - e).sum()

    rf = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=5)
    rf.fit(X, y)
    w_rf = rf.feature_importances_ / rf.feature_importances_.sum()

    w_comb = alpha * w_ewm + (1 - alpha) * w_rf
    return w_comb


def run_combination_sensitivity(df, hazard_name, config):
    indicator_cols = config['indicator_cols']
    short_names = [get_short_name(c) for c in indicator_cols]

    alphas = [0.3, 0.5, 0.7]
    all_weights = []

    for alpha in alphas:
        w_comb = calc_ewm_rf_weights(df, indicator_cols, config['score_col'], alpha)
        all_weights.append(w_comb)

    return {
        'short_names': short_names,
        'all_weights': np.array(all_weights)
    }


def plot_combined_figure(all_results, output_dir):
    hazard_names = list(all_results.keys())

    # 子图行间距稍微缩小（hspace从0.22改为0.16）
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={'hspace': 0.16, 'wspace': 0.18})
    axes = axes.flatten()

    alphas = [0.3, 0.5, 0.7]
    alpha_labels = ['α=0.3', 'α=0.5', 'α=0.7']

    for idx, hazard_name in enumerate(hazard_names):
        ax = axes[idx]
        results = all_results[hazard_name]
        short_names = results['short_names']
        all_weights = results['all_weights']

        x = np.arange(len(short_names))
        width = 0.25

        for i, alpha in enumerate(alphas):
            ax.bar(x + (i - 1) * width, all_weights[i], width,
                   label=alpha_labels[i],
                   color=COLORS_ALPHA[alpha],
                   edgecolor=EDGE_COLORS[alpha],
                   linewidth=LINE_WIDTHS[alpha],
                   alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=15)
        ax.set_xlabel('Indicator', fontsize=16, fontweight='bold')
        ax.set_ylabel('Combined weight', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(all_weights.flatten()) * 1.15)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#444444')

        # 灾害名称在左上角（字号增大两个字号）
        ax.text(0.03, 0.95, f'{hazard_name}',
                transform=ax.transAxes, fontsize=19, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

        # 序号在右上角（字号增大两个字号）
        ax.text(0.97, 0.95, f'({chr(97 + idx)})',
                transform=ax.transAxes, fontsize=19, fontweight='bold',
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

    # ============================================================
    # 图例：左边界对齐图(c)左边界，右边界对齐图(d)右边界
    # 图例外框长度延长至三倍，内部元素大小不变，间距变宽
    # 宽度稍微收窄一点点
    # ============================================================
    fig.canvas.draw()

    # 获取子图位置 (索引2是图c，索引3是图d)
    bbox_c = axes[2].get_position()
    bbox_d = axes[3].get_position()

    # 计算图例边界
    legend_left = bbox_c.x0
    legend_right = bbox_d.x1
    legend_width = legend_right - legend_left
    legend_center = (legend_left + legend_right) / 2

    # 延长图例宽度至三倍，稍微收窄一点点（从3倍改为2.8倍）
    extended_width = legend_width * 2.8
    extended_left = legend_center - extended_width / 2
    extended_right = legend_center + extended_width / 2

    # 创建图例
    legend = fig.legend(
        labels=alpha_labels,
        loc='lower center',
        bbox_to_anchor=(legend_center, 0.025),
        ncol=3,
        fontsize=16,
        frameon=True,
        fancybox=False,
        edgecolor='#333333',
        framealpha=1.0,
        handletextpad=0.8,
        columnspacing=22.0,      # 列间距稍微减小（从22.0改为19.0）
        handlelength=1.5,
        borderpad=0.6
    )

    # 调整子图位置，为图例留出空间
    plt.subplots_adjust(bottom=0.12, left=0.07, right=0.96, top=0.96)

    # 保存
    plt.savefig(os.path.join(output_dir, 'Fig6_weight_combination_sensitivity.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'Fig6_weight_combination_sensitivity.pdf'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Figure 6 已保存: Fig6_weight_combination_sensitivity.png/pdf")


def main():
    print("=" * 60)
    print("Figure 6: EWM-RF组合比例敏感性分析")
    print("=" * 60)

    df = pd.read_excel(INPUT_PATH)
    print(f"✅ 加载数据: {len(df)} 个避难所")

    all_results = {}

    for hazard_name, config in HAZARD_CONFIG.items():
        try:
            all_results[hazard_name] = run_combination_sensitivity(df, hazard_name, config)
            print(f"  ✅ {hazard_name}")
        except Exception as e:
            print(f"  ❌ {hazard_name}: {e}")

    plot_combined_figure(all_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("   - Fig6_weight_combination_sensitivity.png")
    print("   - Fig6_weight_combination_sensitivity.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()