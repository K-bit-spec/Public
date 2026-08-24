# -*- coding: utf-8 -*-
"""
RF重要性验证: 三种方法对比（SCI期刊版）
四个子图都有图例，图例放在序号下方
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
import os
import warnings

warnings.filterwarnings('ignore')

# ===================== 路径配置 =====================
INPUT_PATH = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果\优化避难所得分结果\135现有避难所得分.xlsx"
OUTPUT_DIR = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果（修正版）\RF_Validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 配色 =====================
COLORS = {
    'gini': '#4a7fb5',
    'permutation': '#d97a3a',
    'shap': '#4a9e6a'
}

# ===================== SCI期刊风格设置 =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

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


def get_short_label(col):
    mapping = {
        'A1_有效避难面积(m²)': 'A1', 'A2_医院距离(m)': 'A2', 'A3_消防站距离(m)': 'A3',
        'B1_水系距离(m)': 'B1', 'B2_地质灾害距离(m)': 'B2', 'B3_断裂带距离(m)': 'B3',
        'B4_森林距离(m)': 'B4', 'B5_坡度(°)': 'B5', 'B6_海拔(m)': 'B6',
        'C1_交叉口数量': 'C1', 'C2_市场距离(m)': 'C2', 'D1_500m服务人口': 'D1'
    }
    return mapping.get(col, col[:3])


def run_validation(df, hazard_name, config):
    X_raw = df[config['indicator_cols']].copy()
    y = df[config['score_col']].copy()
    for c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce')
    X_raw = X_raw.fillna(X_raw.median())
    y = pd.to_numeric(y, errors='coerce').fillna(y.median())

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    feature_names = X_raw.columns.tolist()
    short_labels = [get_short_label(c) for c in feature_names]

    rf = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=5)
    rf.fit(X, y)

    gini = rf.feature_importances_ / rf.feature_importances_.sum()

    perm_result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    perm = perm_result.importances_mean / perm_result.importances_mean.sum()

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    shap_imp = np.abs(shap_values).mean(axis=0)
    shap_norm = shap_imp / shap_imp.sum()

    sorted_idx = np.argsort(gini)[::-1]

    return {
        'labels': [short_labels[i] for i in sorted_idx],
        'gini': gini[sorted_idx],
        'permutation': perm[sorted_idx],
        'shap': shap_norm[sorted_idx]
    }


def plot_dot_plot(all_results, output_dir):
    """点图：四个子图都有图例，图例放在序号下方"""
    hazard_names = list(all_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), gridspec_kw={'hspace': 0.18, 'wspace': 0.15})
    axes = axes.flatten()

    markers = ['o', 's', '^']
    colors = [COLORS['gini'], COLORS['permutation'], COLORS['shap']]
    method_labels = ['Gini', 'Permutation', 'SHAP']

    # 图例元素（四个子图共用）
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['gini'],
                   markersize=10, label='Gini', markeredgecolor='white', markeredgewidth=0.8),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['permutation'],
                   markersize=10, label='Permutation', markeredgecolor='white', markeredgewidth=0.8),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['shap'],
                   markersize=10, label='SHAP', markeredgecolor='white', markeredgewidth=0.8),
    ]

    for idx, hazard_name in enumerate(hazard_names):
        ax = axes[idx]
        results = all_results[hazard_name]
        y_labels = results['labels']
        y_pos = np.arange(len(y_labels))

        offset = 0.18

        for j, (method, color, marker) in enumerate(zip(
                method_labels, colors, markers
        )):
            values = results[method.lower()]
            y_offset = (j - 1) * offset
            ax.scatter(values, y_pos + y_offset,
                       label=method,
                       color=color, marker=marker, s=120,
                       alpha=0.85, edgecolors='white', linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=13)
        ax.set_xlabel('Importance', fontsize=14, fontweight='bold')

        ax.grid(axis='x', alpha=0.3, linestyle='--')
        max_val = max(results['gini'].max(), results['permutation'].max(), results['shap'].max())
        ax.set_xlim(0, max_val * 1.18)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#666666')

        # ============================================================
        # 灾害名称：图内部上方居中（字号18）
        # ============================================================
        ax.text(0.5, 0.97, hazard_name, transform=ax.transAxes,
                fontsize=18, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

        # ============================================================
        # 序号：(a)(b)(c)(d) 放在图内部右上角（字号18）
        # ============================================================
        ax.text(0.98, 0.97, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=18, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

        # ============================================================
        # 图例：放在序号下方（右上角偏下位置）
        # ============================================================
        ax.legend(handles=legend_elements,
                  loc='upper right',
                  fontsize=12,
                  frameon=True,
                  fancybox=False,
                  edgecolor='black',
                  handletextpad=0.6,
                  handlelength=1.8,
                  borderpad=0.4,
                  bbox_to_anchor=(0.98, 0.82))  # 放在序号下方

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.96, top=0.96, bottom=0.07)

    # 保存
    plt.savefig(os.path.join(output_dir, 'importance_dot_plot.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'importance_dot_plot.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 点图已保存: importance_dot_plot.png/pdf")


def main():
    print("=" * 60)
    print("RF重要性验证: 三种方法对比")
    print("=" * 60)

    df = pd.read_excel(INPUT_PATH)
    print(f"✅ 加载数据: {len(df)} 个避难所")

    all_results = {}
    for hazard_name, config in HAZARD_CONFIG.items():
        try:
            all_results[hazard_name] = run_validation(df, hazard_name, config)
            print(f"  ✅ {hazard_name}")
        except Exception as e:
            print(f"  ❌ {hazard_name}: {e}")

    plot_dot_plot(all_results, OUTPUT_DIR)

    # 保存汇总表
    summary_rows = []
    for hazard_name, results in all_results.items():
        row = {'灾害': hazard_name}
        for i, label in enumerate(results['labels']):
            row[f'{label}_基尼'] = results['gini'][i]
            row[f'{label}_排列'] = results['permutation'][i]
            row[f'{label}_SHAP'] = results['shap'][i]
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_excel(
        os.path.join(OUTPUT_DIR, 'importance_summary.xlsx'), index=False
    )

    print("\n✅ 完成！输出目录:", OUTPUT_DIR)
    print("   - importance_dot_plot.png (点图)")
    print("   - importance_dot_plot.pdf (矢量图)")
    print("   - importance_summary.xlsx (数据表)")
    print("=" * 60)


if __name__ == "__main__":
    main()