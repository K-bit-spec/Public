import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from matplotlib.patches import Patch


# ===================== 1. 环境配置 =====================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 字号整体增大
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 18  # 图例字号增大到18

plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


INPUT_PATH = r"D:\AICGIS\小文献代码结果\现有避难所指标提取_最终版.xlsx"
OUTPUT_IMG = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\现有避难所权重可视化\Final_Weights_Distribution.png"


# ===================== 2. 场景与颜色配置 =====================

SCENARIOS = [
    {"name": "Earthquake", "codes": ["A1", "A2", "B3", "C1", "C2", "D1"]},
    {"name": "Geological hazard", "codes": ["A1", "B2", "B5", "B6", "C1", "C2", "D1"]},
    {"name": "Flood", "codes": ["A3", "B1", "B6", "C1", "C2", "D1"]},
    {"name": "Fire", "codes": ["A2", "A3", "B4", "C1", "C2", "D1"]},
]

# 稍微更饱和一点，但仍然保持低饱和度、期刊风
COLORS = {
    "entropy": "#6F97D0",
    "rf": "#DD8A73",
    "combined": "#8FBC66",
    "grid": "#D8D8D8",
}
# 四个子图背景色（变浅版本）
SCENE_BG = {
    "Earthquake": "#FAF5FF",          # 淡紫（更浅）
    "Geological hazard": "#F9FCF5",   # 淡绿（更浅）
    "Flood": "#F5FAFE",               # 淡蓝（更浅）
    "Fire": "#FFF8F2",                # 淡橙（更浅）
}


# ===================== 3. 核心函数 =====================

def find_real_column(df_cols, code):
    """匹配 Excel 列名与指标代码"""
    keywords = {
        "A1": "面积",
        "A2": "医院",
        "A3": "消防",
        "B1": "洪水",
        "B2": "地质",
        "B3": "断裂",
        "B4": "森林",
        "B5": "坡度",
        "B6": "海拔",
        "C1": "交叉口",
        "C2": "市场",
        "D1": "人口",
    }

    kw = keywords.get(code, code)

    for col in df_cols:
        if kw in col:
            return col

    prefix = code + "_"
    for col in df_cols:
        if col.startswith(prefix) or f" {code} " in col or f"_{code}_" in col:
            return col

    return None


def calc_weights(df, cols):
    """轻量化权重计算"""
    df_clean = df[cols].dropna(axis=0, how="any")

    if len(df_clean) < 5:
        n = len(cols)
        return (
            np.array([1 / n] * n),
            np.array([1 / n] * n),
            np.array([1 / n] * n),
        )

    df_clean = df_clean.fillna(df_clean.median())

    scaler = MinMaxScaler(feature_range=(0.001, 1))
    norm = scaler.fit_transform(df_clean)

    p = norm / norm.sum(axis=0)
    e = -1 / np.log(len(df_clean)) * (p * np.log(p + 1e-10)).sum(axis=0)
    w_e = (1 - e) / (1 - e).sum()

    rf = RandomForestRegressor(
        n_estimators=30,
        random_state=42,
        n_jobs=1,
        max_depth=5,
    )
    rf.fit(norm, norm.mean(axis=1))
    w_rf = rf.feature_importances_ / rf.feature_importances_.sum()

    w_c = (w_e + w_rf) / 2

    return w_e, w_rf, w_c


def polish_axis(ax):
    """统一 SCI 风格坐标轴"""
    ax.grid(
        True,
        linestyle=":",
        alpha=0.45,
        axis="y",
        color=COLORS["grid"],
        linewidth=0.75,
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#444444")


def add_panel_label(ax, label):
    """将 (a), (b), (c), (d) 放在子图内部右上角"""
    ax.text(
        0.975,
        0.955,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=20,
        fontweight="bold",
        family="Times New Roman",
        zorder=20,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
            pad=1.2,
        ),
    )


def add_scene_title(ax, scene_name):
    """将灾害名称放在子图内部左上角，字号增大"""
    ax.text(
        0.03,
        0.955,
        scene_name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        family="Times New Roman",
        zorder=20,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
            pad=1.2,
        ),
    )


def plot_vertical_bars(ax, valid_codes, w_e, w_rf, w_c):
    """竖向分组柱状图"""
    x = np.arange(len(valid_codes))
    width = 0.24

    ax.bar(
        x - width,
        w_e,
        width,
        color=COLORS["entropy"],
        edgecolor="#555555",
        linewidth=0.45,
        alpha=0.95,
        label="Entropy",
        zorder=3,
    )

    ax.bar(
        x,
        w_rf,
        width,
        color=COLORS["rf"],
        edgecolor="#555555",
        linewidth=0.45,
        alpha=0.95,
        label="RF",
        zorder=3,
    )

    ax.bar(
        x + width,
        w_c,
        width,
        color=COLORS["combined"],
        edgecolor="#555555",
        linewidth=0.45,
        alpha=0.95,
        label="Combined",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(valid_codes, rotation=0, ha="center", fontsize=16)
    ax.set_ylabel("Weight", fontsize=18, labelpad=8)

    y_max = max(np.max(w_e), np.max(w_rf), np.max(w_c))
    ax.set_ylim(0, y_max * 1.20)


# ===================== 4. 主函数 =====================

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"错误：文件不存在 → {INPUT_PATH}")
        return

    raw_df = pd.read_excel(INPUT_PATH, engine="openpyxl")
    all_cols = raw_df.columns.tolist()

    print(f"成功读取 Excel，共 {len(raw_df)} 行，{len(all_cols)} 列")

    output_dir = os.path.dirname(OUTPUT_IMG)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建目录：{output_dir}")

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.2), dpi=300)

    # 横向子图间距稍微拉开，底部留白略缩小，使图例更贴近横轴
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.185,
        top=0.98,
        wspace=0.18,
        hspace=0.20,
    )

    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

    for idx, sc in enumerate(SCENARIOS):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        ax.set_facecolor(SCENE_BG[sc["name"]])

        # 添加灾害名称在左上角
        add_scene_title(ax, sc["name"])

        codes_original = sc["codes"]
        codes_unique = sorted(set(codes_original))

        valid_codes = []
        real_cols = []

        for c in codes_unique:
            col_real = find_real_column(all_cols, c)

            if col_real is not None:
                valid_codes.append(c)
                real_cols.append(col_real)
            else:
                print(f"警告：{sc['name']} 场景中指标 {c} 未找到，跳过")

        if len(real_cols) == 0:
            print(f"警告：{sc['name']} 场景无匹配列，跳过")
            continue

        w_e, w_rf, w_c = calc_weights(raw_df, real_cols)

        plot_vertical_bars(ax, valid_codes, w_e, w_rf, w_c)

        add_panel_label(ax, subplot_labels[idx])
        polish_axis(ax)
        ax.tick_params(axis="y", labelsize=16)
        ax.tick_params(axis="x", labelsize=16)

    # ===================== 底部统一图例（仅保留三种权重方法，字号增大，位置上移） =====================

    method_handles = [
        Patch(facecolor=COLORS["entropy"], edgecolor="#555555", linewidth=0.6, label="Entropy"),
        Patch(facecolor=COLORS["rf"], edgecolor="#555555", linewidth=0.6, label="RF"),
        Patch(facecolor=COLORS["combined"], edgecolor="#555555", linewidth=0.6, label="Combined"),
    ]

    # 图例稍微上移：y 从 0.070 调整为 0.088
    fig.legend(
        handles=method_handles,
        loc="lower left",
        bbox_to_anchor=(0.08, 0.088, 0.905, 0.090),
        mode="expand",
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=18,  # 图例字号增大到18
        handlelength=1.9,
        handletextpad=0.6,
        columnspacing=1.3,
        borderpad=0.35,
        alignment="center",
    )

    output_img = OUTPUT_IMG.replace(".png", "_Vertical_Bar_Final.png")
    output_pdf = OUTPUT_IMG.replace(".png", "_Vertical_Bar_Final.pdf")

    plt.savefig(
        output_img,
        dpi=300,
        bbox_inches="tight",
        format="png",
        facecolor="white",
    )

    plt.savefig(
        output_pdf,
        dpi=300,
        bbox_inches="tight",
        format="pdf",
        facecolor="white",
    )

    plt.close(fig)

    print(f"图片已保存至：{output_img}")
    print(f"PDF 已保存至：{output_pdf}")


if __name__ == "__main__":
    main()