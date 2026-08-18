# -*- coding: utf-8 -*-
"""
代码1：贪心算法优化 + 结果导出
运行时间：约5-10分钟
输出：Excel文件（含N=135避难所列表 + 对比表格）
"""

import os
import warnings
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import distance_matrix

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")

# ===================== 路径配置 =====================
SHELTER_PATH = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果\优化避难所得分结果\全部避难所得分.xlsx"
POP_TIF_PATH = r"D:\AICGIS\实验\population_shange.tif"
OUTPUT_DIR = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果（修正版）\贪心避难所综合"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_RESULTS_XLSX = os.path.join(OUTPUT_DIR, "shelter_optimization_results_greedy.xlsx")


# ===================== 配置 =====================
@dataclass
class ModelConfig:
    lon_col: str = "lon"
    lat_col: str = "lat"
    score_col: str = "综合适宜性得分"
    id_col: Optional[str] = "避难所ID"
    type_col: str = "type"
    existing_type: str = "existing"
    candidate_type: str = "candidate"
    source_crs: str = "EPSG:4326"
    projected_crs: str = "EPSG:32648"
    service_radius: float = 1000.0
    min_distance: float = 500.0
    n_min: int = 50
    n_max: int = 180
    n_step: int = 10
    w_coverage: float = 0.60
    w_safety: float = 0.30
    w_overlap: float = 0.10
    batch_size: int = 50


# ===================== 优化器 =====================
class ShelterOptimizer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.shelters = None
        self.existing_shelters = None
        self.candidate_shelters = None
        self.pop_points_xy = None
        self.pop_values = None
        self.dist_matrix = None
        self.total_population = 0.0
        self.selected_by_n = {}

    def load_data(self):
        cfg = self.config
        shelters = pd.read_excel(SHELTER_PATH)
        for col in [cfg.lon_col, cfg.lat_col, cfg.score_col]:
            if col not in shelters.columns:
                raise ValueError(f"Missing column: {col}")
        shelters[cfg.lon_col] = pd.to_numeric(shelters[cfg.lon_col], errors="coerce")
        shelters[cfg.lat_col] = pd.to_numeric(shelters[cfg.lat_col], errors="coerce")
        shelters[cfg.score_col] = pd.to_numeric(shelters[cfg.score_col], errors="coerce")
        shelters = shelters.dropna(subset=[cfg.lon_col, cfg.lat_col, cfg.score_col]).copy()
        shelters = shelters.reset_index(drop=True)
        if cfg.id_col is not None and cfg.id_col in shelters.columns:
            shelters["Shelter_ID"] = shelters[cfg.id_col].astype(str)
        else:
            shelters["Shelter_ID"] = [f"S{idx:05d}" for idx in range(len(shelters))]
        transformer = Transformer.from_crs(cfg.source_crs, cfg.projected_crs, always_xy=True)
        x, y = transformer.transform(shelters[cfg.lon_col].values, shelters[cfg.lat_col].values)
        shelters["x_projected"] = x
        shelters["y_projected"] = y
        self.shelters = shelters
        if cfg.type_col in shelters.columns:
            self.existing_shelters = shelters[shelters[cfg.type_col] == cfg.existing_type].copy()
            self.candidate_shelters = shelters[shelters[cfg.type_col] == cfg.candidate_type].copy()
        else:
            self.existing_shelters = shelters.head(135).copy()
            self.candidate_shelters = shelters.iloc[135:].copy()
        print(f"现有: {len(self.existing_shelters)}, 候选: {len(self.candidate_shelters)}, 总计: {len(self.shelters)}")
        self._load_population(transformer)
        self._calc_distance()

    def _load_population(self, transformer):
        with rasterio.open(POP_TIF_PATH) as src:
            pop = src.read(1)
            nodata = src.nodata
            valid = pop > 0
            if nodata is not None:
                valid &= pop != nodata
            rows, cols = np.where(valid)
            xs, ys, vals = [], [], []
            for r, c in zip(rows, cols):
                x, y = src.xy(r, c)
                xs.append(x);
                ys.append(y);
                vals.append(float(pop[r, c]))
            raster_crs = src.crs.to_string() if src.crs is not None else self.config.source_crs
        xs = np.asarray(xs, dtype=float);
        ys = np.asarray(ys, dtype=float);
        vals = np.asarray(vals, dtype=float)
        if raster_crs != self.config.source_crs:
            rt = Transformer.from_crs(raster_crs, self.config.source_crs, always_xy=True)
            xs, ys = rt.transform(xs, ys)
        x_proj, y_proj = transformer.transform(xs, ys)
        self.pop_points_xy = np.column_stack([x_proj, y_proj])
        self.pop_values = vals
        self.total_population = float(vals.sum())
        print(f"人口点: {len(self.pop_points_xy):,}, 总人口: {self.total_population:,.0f}")

    def _calc_distance(self):
        coords = self.shelters[["x_projected", "y_projected"]].values
        n_shelters = len(coords);
        n_pop = len(self.pop_points_xy)
        dist = np.zeros((n_shelters, n_pop), dtype=np.float32)
        print("计算距离矩阵...")
        for i in range(0, n_shelters, self.config.batch_size):
            e = min(i + self.config.batch_size, n_shelters)
            dist[i:e, :] = distance_matrix(coords[i:e], self.pop_points_xy).astype(np.float32)
            print(f"  进度: {e}/{n_shelters}")
        self.dist_matrix = dist
        print(f"距离矩阵完成: {dist.shape}")

    def evaluate_solution(self, selected, service_radius=None):
        if service_radius is None:
            service_radius = self.config.service_radius
        if len(selected) == 0:
            return self._empty()
        d_sub = self.dist_matrix[selected, :]
        min_d = d_sub.min(axis=0)
        covered_mask = min_d <= service_radius
        covered_pop = float(self.pop_values[covered_mask].sum())
        service_count = (d_sub <= service_radius).sum(axis=0)
        overlap_pop = float(np.sum(self.pop_values * np.maximum(service_count - 1, 0)))
        scores = self.shelters.iloc[selected][self.config.score_col].values.astype(float)
        avg_safety = float(scores.mean()) if len(scores) > 0 else 0.0
        coverage_rate = covered_pop / self.total_population if self.total_population > 0 else 0.0
        overlap_rate_covered = overlap_pop / covered_pop if covered_pop > 0 else 0.0
        efficiency = covered_pop / len(selected) if len(selected) > 0 else 0
        return {
            "Selected_Count": len(selected),
            "Covered_Population": covered_pop,
            "Coverage_Rate": coverage_rate,
            "Avg_Safety": avg_safety,
            "Overlap_Population": overlap_pop,
            "Overlap_Rate_Covered": overlap_rate_covered,
            "Efficiency": efficiency,
            "Coverage_Rate_Percent": coverage_rate * 100,
            "Overlap_Rate_Covered_Percent": overlap_rate_covered * 100,
        }

    def _empty(self):
        return {
            "Selected_Count": 0, "Covered_Population": 0.0, "Coverage_Rate": 0.0,
            "Avg_Safety": 0.0, "Overlap_Population": 0.0, "Overlap_Rate_Covered": 0.0,
            "Efficiency": 0.0, "Coverage_Rate_Percent": 0.0, "Overlap_Rate_Covered_Percent": 0.0,
        }

    def calculate_objective(self, metrics):
        cfg = self.config
        score_min = float(self.shelters[cfg.score_col].min())
        score_max = float(self.shelters[cfg.score_col].max())
        safety_score = (metrics["Avg_Safety"] - score_min) / (score_max - score_min) if score_max > score_min else 0
        safety_score = float(np.clip(safety_score, 0, 1))
        return cfg.w_coverage * metrics["Coverage_Rate"] + cfg.w_safety * safety_score - cfg.w_overlap * metrics[
            "Overlap_Rate_Covered"]

    def evaluate_existing(self, service_radius=None):
        if service_radius is None:
            service_radius = self.config.service_radius
        if self.existing_shelters is None or len(self.existing_shelters) == 0:
            return self._empty()
        indices = []
        for idx in self.existing_shelters.index:
            sid = self.existing_shelters.loc[idx, "Shelter_ID"]
            matched = self.shelters[self.shelters["Shelter_ID"] == sid].index
            if len(matched) > 0:
                indices.append(matched[0])
        return self.evaluate_solution(indices, service_radius)

    # ================================================================
    # 贪心选择（替换遗传算法）
    # ================================================================
    def greedy_select(self, N, service_radius=None):
        """
        贪心算法选择N个避难所
        每一步选择使目标函数提升最大的避难所
        """
        if service_radius is None:
            service_radius = self.config.service_radius

        n_shelters = len(self.shelters)
        selected = []
        remaining = list(range(n_shelters))

        # 预计算安全得分
        safety_scores = self.shelters[self.config.score_col].values

        # 当前评估
        current_metrics = self._empty()
        current_obj = self.calculate_objective(current_metrics)

        for i in range(N):
            if not remaining:
                break

            best_gain = -np.inf
            best_idx = None
            best_metrics = None
            best_obj = None

            # 候选集优化：如果剩余太多，只评估前100个高分候选
            candidates = remaining
            if len(remaining) > 100 and len(selected) > 50:
                remaining_scores = [(idx, safety_scores[idx]) for idx in remaining]
                remaining_scores.sort(key=lambda x: x[1], reverse=True)
                candidates = [idx for idx, _ in remaining_scores[:100]]

            for idx in candidates:
                # 检查距离约束
                if selected:
                    coords_selected = self.shelters.iloc[selected][['x_projected', 'y_projected']].values
                    coord_new = self.shelters.iloc[[idx]][['x_projected', 'y_projected']].values
                    distances = np.sqrt(np.sum((coords_selected - coord_new) ** 2, axis=1))
                    if np.any(distances < self.config.min_distance):
                        continue

                # 评估临时方案
                temp_selected = selected + [idx]
                metrics = self.evaluate_solution(temp_selected, service_radius)
                obj = self.calculate_objective(metrics)
                gain = obj - current_obj

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_metrics = metrics
                    best_obj = obj

            if best_idx is None:
                break

            # 选择最佳候选
            selected.append(best_idx)
            remaining.remove(best_idx)
            current_metrics = best_metrics
            current_obj = best_obj

            if (i + 1) % 20 == 0:
                print(f"  已选 {i + 1}/{N}, 当前目标值: {current_obj:.4f}")

        return selected

    def optimize_over_N(self, n_values=None, scenario_id="BASELINE"):
        """对每个N值运行贪心算法"""
        if n_values is None:
            n_values = list(range(self.config.n_min, self.config.n_max + 1, self.config.n_step))

        existing_count = len(self.existing_shelters) if self.existing_shelters is not None else 135
        if existing_count not in n_values:
            n_values.append(existing_count)
            n_values.sort()

        rows = []
        self.selected_by_n = {}

        for N in n_values:
            print(f"\n优化 N={N} (贪心算法)...")
            selected = self.greedy_select(N)
            metrics = self.evaluate_solution(selected)
            obj = self.calculate_objective(metrics)
            self.selected_by_n[N] = selected
            rows.append({"Scenario_ID": scenario_id, "N": N, **metrics, "Objective_Value": obj})

        df = pd.DataFrame(rows)
        df["Coverage_Rate_Percent"] = df["Coverage_Rate"] * 100
        df["Overlap_Rate_Covered_Percent"] = df["Overlap_Rate_Covered"] * 100
        best_idx = df["Objective_Value"].idxmax()
        return df, self.selected_by_n, int(df.loc[best_idx, "N"])

    def selected_shelters_table(self, selected, scenario_id, method, service_radius=None):
        if service_radius is None:
            service_radius = self.config.service_radius
        df = self.shelters.iloc[selected].copy()
        df["Scenario_ID"] = scenario_id
        df["Method"] = method
        df["Rank"] = range(1, len(df) + 1)
        if len(selected) > 0:
            d_sub = self.dist_matrix[selected, :]
            service_mask = d_sub <= service_radius
            nearest = d_sub.argmin(axis=0)
            covered = d_sub.min(axis=0) <= service_radius
            service_pop = []
            unique_pop = []
            for i in range(len(selected)):
                service_pop.append(float(self.pop_values[service_mask[i]].sum()))
                unique_mask = covered & (nearest == i)
                unique_pop.append(float(self.pop_values[unique_mask].sum()))
            df["Service_Population"] = service_pop
            df["Unique_Covered_Population"] = unique_pop
        return df


# ===================== 主程序 =====================
def main():
    cfg = ModelConfig()
    opt = ShelterOptimizer(cfg)
    print("=" * 70)
    print("避难所优化 - 贪心算法")
    print("=" * 70)
    print(f"N范围: {cfg.n_min}~{cfg.n_max}, 步长: {cfg.n_step}")
    print(f"权重: 覆盖={cfg.w_coverage}, 安全={cfg.w_safety}, 重叠={cfg.w_overlap}")
    print("=" * 70)

    opt.load_data()

    # 评估现有
    existing = opt.evaluate_existing()
    print(
        f"\n现有: {existing['Selected_Count']}个, 覆盖率={existing['Coverage_Rate_Percent']:.1f}%, 安全={existing['Avg_Safety']:.4f}")

    # 优化
    df_all, selected_by_n, optimal_N = opt.optimize_over_N()
    print(f"\n✅ 最优N = {optimal_N}")

    # N=135
    N_TARGET = 135
    if N_TARGET in selected_by_n:
        sel_135 = selected_by_n[N_TARGET]
    else:
        closest = min(selected_by_n.keys(), key=lambda x: abs(x - N_TARGET))
        sel_135 = selected_by_n[closest]
        N_TARGET = closest
    shelters_135 = opt.selected_shelters_table(sel_135, f"N={N_TARGET}", "Greedy")

    # 获取结果
    row_135 = df_all[df_all['N'] == N_TARGET].iloc[0]
    row_opt = df_all[df_all['N'] == optimal_N].iloc[0]

    # 保存Excel
    def pct(a, b):
        if a == 0:
            return "N/A"
        return f"{((b - a) / a * 100):+.1f}%"

    with pd.ExcelWriter(ALL_RESULTS_XLSX, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="all_N_results", index=False)
        shelters_135.to_excel(writer, sheet_name=f"N={N_TARGET}_shelters", index=False)
        opt.existing_shelters.to_excel(writer, sheet_name="existing_shelters", index=False)
        opt.candidate_shelters.to_excel(writer, sheet_name="candidate_shelters", index=False)
        comparison = {
            "指标": ["数量", "覆盖率(%)", "覆盖人口", "安全得分", "重叠率(%)", "单位效率"],
            "现有": [
                existing["Selected_Count"],
                f"{existing['Coverage_Rate_Percent']:.1f}",
                f"{existing['Covered_Population']:,.0f}",
                f"{existing['Avg_Safety']:.4f}",
                f"{existing['Overlap_Rate_Covered_Percent']:.1f}",
                f"{existing['Efficiency']:,.0f}"
            ],
            f"优化(N={N_TARGET})": [
                N_TARGET,
                f"{row_135['Coverage_Rate_Percent']:.1f}",
                f"{row_135['Covered_Population']:,.0f}",
                f"{row_135['Avg_Safety']:.4f}",
                f"{row_135['Overlap_Rate_Covered_Percent']:.1f}",
                f"{row_135['Efficiency']:,.0f}"
            ],
            "提升": [
                pct(existing["Selected_Count"], N_TARGET),
                pct(existing["Coverage_Rate_Percent"], row_135["Coverage_Rate_Percent"]),
                pct(existing["Covered_Population"], row_135["Covered_Population"]),
                pct(existing["Avg_Safety"], row_135["Avg_Safety"]),
                pct(existing["Overlap_Rate_Covered_Percent"], row_135["Overlap_Rate_Covered_Percent"]),
                pct(existing["Efficiency"], row_135["Efficiency"])
            ]
        }
        pd.DataFrame(comparison).to_excel(writer, sheet_name="对比", index=False)

    print(f"\n✅ 已保存: {ALL_RESULTS_XLSX}")
    print("\n对比结果:")
    print(
        f"  覆盖率: {existing['Coverage_Rate_Percent']:.1f}% → {row_135['Coverage_Rate_Percent']:.1f}% ({pct(existing['Coverage_Rate_Percent'], row_135['Coverage_Rate_Percent'])})")
    print(
        f"  安全得分: {existing['Avg_Safety']:.4f} → {row_135['Avg_Safety']:.4f} ({pct(existing['Avg_Safety'], row_135['Avg_Safety'])})")
    print(
        f"  单位效率: {existing['Efficiency']:,.0f} → {row_135['Efficiency']:,.0f} ({pct(existing['Efficiency'], row_135['Efficiency'])})")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()