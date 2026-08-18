# -*- coding: utf-8 -*-
"""
SCI高水平期刊风格 - 敏感性分析与鲁棒性可视化
从真实Excel文件读取数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

# ===================== 全部罗马字体 =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

# SCI经典配色
COLORS = {
    'Service Radius': '#1f77b4',
    'Min Distance': '#ff7f0e',
    'Weights': '#2ca02c',
}
HATCHES = {
    'Service Radius': '',
    'Min Distance': '///',
    'Weights': 'xxx',
}
MARKERS = {
    'Service Radius': 'o',
    'Min Distance': 's',
    'Weights': '^',
}

# ===================== 从真实Excel读取数据 =====================
INPUT_EXCEL = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果（修正版）\贪心避难所综合\敏感性\sensitivity_robustness_results_greedy.xlsx"

if not os.path.exists(INPUT_EXCEL):
    print(f"❌ 错误：文件不存在 → {INPUT_EXCEL}")
    exit()

df = pd.read_excel(INPUT_EXCEL)
print(f"✅ 成功读取数据：{len(df)} 行")
print(f"列名：{df.columns.tolist()}")

# ===================== 创建图形 =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.12})

# ================================================================
# 计算各组位置
# ================================================================

param_groups = ['Service Radius', 'Min Distance', 'Weights']
group_indices = {}
for param in param_groups:
    group_indices[param] = df[df['Parameter'] == param].index.tolist()

# 计算每个场景的x位置
x_pos = 0
scenario_x_positions = {}

for param in param_groups:
    idx_list = group_indices[param]
    for i, idx in enumerate(idx_list):
        row = df.loc[idx]
        scenario_x_positions[row['Scenario_ID']] = x_pos
        x_pos += 1
    x_pos += 0.5

# ================================================================
# Panel (a): Jaccard相似度与留存率
# ================================================================

x_pos = 0
x_ticks = []
x_labels = []
bar_width = 0.8
jaccard_width = 0.45

for param in param_groups:
    idx_list = group_indices[param]
    for i, idx in enumerate(idx_list):
        row = df.loc[idx]

        # 先画留存率
        ax1.bar(x_pos, row['Retention_Rate'] * 100,
                width=bar_width, color='white',
                edgecolor=COLORS[param], linewidth=1.5,
                hatch=HATCHES[param], zorder=1)

        # 再画Jaccard相似度
        ax1.bar(x_pos, row['Jaccard_Similarity'] * 100,
                width=jaccard_width, color=COLORS[param],
                edgecolor='black', linewidth=0.8, zorder=2,
                label=param if i == 0 else '')

        # 生成标签
        param_val = row['Parameter_Value']
        if param == 'Weights':
            parts = str(param_val).replace('(', '').replace(')', '').split(',')
            if len(parts) == 3:
                label = f"{int(float(parts[0]) * 100)},{int(float(parts[1]) * 100)},{int(float(parts[2]) * 100)}"
            else:
                label = str(param_val)
        else:
            label = str(param_val)
        x_labels.append(label)
        x_ticks.append(x_pos)
        x_pos += 1
    x_pos += 0.5

ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Similarity / Retention Rate (%)', fontsize=11, labelpad=8)
ax1.set_ylim(0, 120)
ax1.set_yticks([0, 20, 40, 60, 80, 100, 120])
ax1.grid(True, linestyle='--', alpha=0.2, axis='y')

# 添加 (a) 标签（图内部右上角）
ax1.text(0.97, 0.97, '(a)', transform=ax1.transAxes,
         fontsize=14, fontweight='bold', va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

# ================================================================
# Panel (b): 最优N与覆盖率变化 (双轴图)
# ================================================================

x_pos = 0
x_ticks_b = []
x_labels_b = []

for param in param_groups:
    idx_list = group_indices[param]
    for i, idx in enumerate(idx_list):
        row = df.loc[idx]

        # 最优N
        ax2.bar(x_pos, row['Optimal_N'],
                width=0.6, color=COLORS[param], alpha=0.7,
                edgecolor='black', linewidth=0.8,
                label=param if i == 0 else '')

        # 标注N值
        ax2.text(x_pos, row['Optimal_N'] + 2, f"{int(row['Optimal_N'])}",
                 ha='center', va='bottom', fontsize=7, fontweight='bold')

        # 生成标签
        param_val = row['Parameter_Value']
        if param == 'Weights':
            parts = str(param_val).replace('(', '').replace(')', '').split(',')
            if len(parts) == 3:
                label = f"{int(float(parts[0]) * 100)},{int(float(parts[1]) * 100)},{int(float(parts[2]) * 100)}"
            else:
                label = str(param_val)
        else:
            label = str(param_val)
        x_labels_b.append(label)
        x_ticks_b.append(x_pos)
        x_pos += 1
    x_pos += 0.5

ax2.set_xticks(x_ticks_b)
ax2.set_xticklabels(x_labels_b, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Optimal N', fontsize=11, labelpad=8, color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=9)
ax2.set_ylim(50, 120)
ax2.set_yticks([60, 70, 80, 90, 100, 110, 120])

# 第二y轴：覆盖率
ax2b = ax2.twinx()

for param in param_groups:
    idx_list = group_indices[param]
    x_positions = []
    coverage_values = []
    for idx in idx_list:
        row = df.loc[idx]
        x_positions.append(scenario_x_positions[row['Scenario_ID']])
        coverage_values.append(row['Coverage_Rate_Percent'])
    ax2b.plot(x_positions, coverage_values,
              marker=MARKERS[param], color=COLORS[param],
              linewidth=2, markersize=7,
              linestyle='-', label=f'{param} (Coverage)')

ax2b.set_ylabel('Coverage Rate (%)', fontsize=11, labelpad=8, color='black')
ax2b.tick_params(axis='y', labelcolor='black', labelsize=9)
coverage_min = df['Coverage_Rate_Percent'].min() - 2
coverage_max = df['Coverage_Rate_Percent'].max() + 2
ax2b.set_ylim(coverage_min, coverage_max)

# 添加 (b) 标签（图内部右上角）
ax2.text(0.97, 0.97, '(b)', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

# ================================================================
# 图例：2行×6列
# ================================================================

# 获取子图位置（用于图例对齐）
fig.canvas.draw()
bbox1 = ax1.get_position()
bbox2 = ax2.get_position()

legend_left = bbox1.x0 + 0.015
legend_right = bbox2.x1
legend_center = (legend_left + legend_right) / 2

# 图(a)的图例元素
legend_a = [
    Patch(facecolor=COLORS['Service Radius'], edgecolor='black', label='Jaccard (Service Radius)'),
    Patch(facecolor='white', edgecolor=COLORS['Service Radius'], hatch='', label='Retention (Service Radius)'),
    Patch(facecolor=COLORS['Min Distance'], edgecolor='black', label='Jaccard (Min Distance)'),
    Patch(facecolor='white', edgecolor=COLORS['Min Distance'], hatch='///', label='Retention (Min Distance)'),
    Patch(facecolor=COLORS['Weights'], edgecolor='black', label='Jaccard (Weights)'),
    Patch(facecolor='white', edgecolor=COLORS['Weights'], hatch='xxx', label='Retention (Weights)'),
]

# 图(b)的图例元素
legend_b = [
    Patch(facecolor=COLORS['Service Radius'], alpha=0.7, edgecolor='black', label='N (Service Radius)'),
    Patch(facecolor=COLORS['Min Distance'], alpha=0.7, edgecolor='black', label='N (Min Distance)'),
    Patch(facecolor=COLORS['Weights'], alpha=0.7, edgecolor='black', label='N (Weights)'),
    Line2D([0], [0], marker='o', color=COLORS['Service Radius'], linewidth=2, label='Coverage (Service Radius)'),
    Line2D([0], [0], marker='s', color=COLORS['Min Distance'], linewidth=2, label='Coverage (Min Distance)'),
    Line2D([0], [0], marker='^', color=COLORS['Weights'], linewidth=2, label='Coverage (Weights)'),
]

# 合并图例（12个元素 → 2行×6列）
all_legend_elements = legend_a + legend_b

# 创建图例
legend = fig.legend(handles=all_legend_elements,
                    loc='lower center',
                    bbox_to_anchor=(legend_center, 0.02),
                    ncol=6,
                    fontsize=9,
                    frameon=True,
                    fancybox=False,
                    edgecolor='black',
                    framealpha=1.0,
                    columnspacing=4,
                    handlelength=2.4,
                    handletextpad=1.0,
                    labelspacing=0.7)

plt.subplots_adjust(bottom=0.2)

# ================================================================
# 保存图形
# ================================================================

output_dir = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果（修正版）\贪心避难所综合\敏感性"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, 'Fig6_sensitivity_robustness_greedy.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(output_dir, 'Fig6_sensitivity_robustness_greedy.png'),
            dpi=300, bbox_inches='tight', facecolor='white')

print(f"✅ 图表已保存至: {output_dir}")
print("✅ 图表生成完成！")