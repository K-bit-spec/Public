# -*- coding: utf-8 -*-
"""
敏感性分析 - 贪心算法完整版
自动运行17个场景，生成完整数据表
包含：服务半径、最小间距、权重 三种参数类型
"""

import os
import warnings
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
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
OUTPUT_DIR = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果（修正版）\贪心避难所综合\敏感性"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SENSITIVITY_EXCEL = os.path.join(OUTPUT_DIR, "sensitivity_robustness_results_greedy.xlsx")
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
        self.baseline_selected_ids = set()  # 用于计算Jaccard和Retention

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

    def greedy_select(self, N, service_radius=None, min_distance=None):
        if service_radius is None:
            service_radius = self.config.service_radius
        if min_distance is None:
            min_distance = self.config.min_distance

        n_shelters = len(self.shelters)
        selected = []
        remaining = list(range(n_shelters))
        safety_scores = self.shelters[self.config.score_col].values

        current_metrics = self._empty()
        current_obj = self.calculate_objective(current_metrics)

        for i in range(N):
            if not remaining:
                break

            best_gain = -np.inf
            best_idx = None
            best_metrics = None
            best_obj = None

            candidates = remaining
            if len(remaining) > 100 and len(selected) > 50:
                remaining_scores = [(idx, safety_scores[idx]) for idx in remaining]
                remaining_scores.sort(key=lambda x: x[1], reverse=True)
                candidates = [idx for idx, _ in remaining_scores[:100]]

            for idx in candidates:
                if selected:
                    coords_selected = self.shelters.iloc[selected][['x_projected', 'y_projected']].values
                    coord_new = self.shelters.iloc[[idx]][['x_projected', 'y_projected']].values
                    distances = np.sqrt(np.sum((coords_selected - coord_new) ** 2, axis=1))
                    if np.any(distances < min_distance):
                        continue

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

            selected.append(best_idx)
            remaining.remove(best_idx)
            current_metrics = best_metrics
            current_obj = best_obj

        return selected

    def optimize_over_N(self, service_radius=None, min_distance=None,
                        w_coverage=None, w_safety=None, w_overlap=None,
                        n_values=None, scenario_id="BASELINE"):
        """对每个N值运行贪心算法"""

        # 临时保存原始权重
        orig_wc = self.config.w_coverage
        orig_ws = self.config.w_safety
        orig_wo = self.config.w_overlap

        if service_radius is not None:
            self.config.service_radius = service_radius
        if min_distance is not None:
            self.config.min_distance = min_distance
        if w_coverage is not None:
            self.config.w_coverage = w_coverage
        if w_safety is not None:
            self.config.w_safety = w_safety
        if w_overlap is not None:
            self.config.w_overlap = w_overlap

        if n_values is None:
            n_values = list(range(self.config.n_min, self.config.n_max + 1, self.config.n_step))

        existing_count = len(self.existing_shelters) if self.existing_shelters is not None else 135
        if existing_count not in n_values:
            n_values.append(existing_count)
            n_values.sort()

        rows = []
        self.selected_by_n = {}

        for N in n_values:
            selected = self.greedy_select(N)
            metrics = self.evaluate_solution(selected)
            obj = self.calculate_objective(metrics)
            self.selected_by_n[N] = selected
            rows.append({"Scenario_ID": scenario_id, "N": N, **metrics, "Objective_Value": obj})

        df = pd.DataFrame(rows)
        df["Coverage_Rate_Percent"] = df["Coverage_Rate"] * 100
        df["Overlap_Rate_Covered_Percent"] = df["Overlap_Rate_Covered"] * 100

        best_idx = df["Objective_Value"].idxmax()
        optimal_N = int(df.loc[best_idx, "N"])

        # 恢复原始权重
        self.config.w_coverage = orig_wc
        self.config.w_safety = orig_ws
        self.config.w_overlap = orig_wo

        return df, self.selected_by_n, optimal_N


# ===================== 运行敏感性分析 =====================

def run_sensitivity_analysis():
    """运行完整的敏感性分析"""

    print("=" * 70)
    print("敏感性分析 - 贪心算法")
    print("=" * 70)

    # 加载基础数据（只加载一次）
    cfg = ModelConfig()
    opt = ShelterOptimizer(cfg)
    opt.load_data()

    # 获取基准选中的避难所ID（用于Jaccard和Retention）
    # 先运行基准场景 (SEN_003 / SEN_008 / SEN_013)
    baseline_df, baseline_selected_by_n, baseline_N = opt.optimize_over_N(
        service_radius=1000, min_distance=500,
        w_coverage=0.60, w_safety=0.30, w_overlap=0.10,
        scenario_id="BASELINE"
    )
    baseline_selected = baseline_selected_by_n[baseline_N]
    baseline_ids = set(opt.shelters.iloc[baseline_selected]["Shelter_ID"].astype(str))

    def similarity_metrics(base_ids, test_ids):
        intersection = len(base_ids & test_ids)
        union = len(base_ids | test_ids)
        jaccard = intersection / union if union > 0 else 0.0
        retention = intersection / len(base_ids) if len(base_ids) > 0 else 0.0
        return jaccard, retention

    all_results = []
    scenario_counter = 0

    # ============================================================
    # 1. 服务半径敏感性 (800, 900, 1000, 1100, 1200)
    # ============================================================
    print("\n" + "=" * 60)
    print("1. 服务半径敏感性分析")
    print("=" * 60)

    radius_scenarios = [800, 900, 1000, 1100, 1200]
    for radius in radius_scenarios:
        scenario_counter += 1
        scenario_id = f"SEN_{scenario_counter:03d}"
        print(f"\n{scenario_id}: 服务半径 = {radius}m")

        start_time = time.time()

        # 运行优化
        df, selected_by_n, optimal_N = opt.optimize_over_N(
            service_radius=float(radius),
            min_distance=500,
            w_coverage=0.60, w_safety=0.30, w_overlap=0.10,
            scenario_id=scenario_id
        )

        # 获取最优行
        row = df[df['N'] == optimal_N].iloc[0]
        selected_ids = set(opt.shelters.iloc[selected_by_n[optimal_N]]["Shelter_ID"].astype(str))
        jaccard, retention = similarity_metrics(baseline_ids, selected_ids)

        all_results.append({
            "Scenario_ID": scenario_id,
            "Parameter": "Service Radius",
            "Parameter_Value": str(radius),
            "Service_Radius": float(radius),
            "Min_Distance": 500.0,
            "W_Coverage": 0.60,
            "W_Safety": 0.30,
            "W_Overlap": 0.10,
            "Optimal_N": optimal_N,
            "Coverage_Rate_Percent": row["Coverage_Rate_Percent"],
            "Avg_Safety": row["Avg_Safety"],
            "Efficiency": row["Efficiency"],
            "Overlap_Rate_Covered_Percent": row["Overlap_Rate_Covered_Percent"],
            "Objective_Value": row["Objective_Value"],
            "Jaccard_Similarity": jaccard,
            "Retention_Rate": retention,
            "Time_s": time.time() - start_time
        })
        print(
            f"  ✅ 最优N={optimal_N}, 覆盖率={row['Coverage_Rate_Percent']:.1f}%, 时间={time.time() - start_time:.1f}s")

    # ============================================================
    # 2. 最小间距敏感性 (300, 400, 500, 600, 700)
    # ============================================================
    print("\n" + "=" * 60)
    print("2. 最小间距敏感性分析")
    print("=" * 60)

    distance_scenarios = [300, 400, 500, 600, 700]
    for distance in distance_scenarios:
        scenario_counter += 1
        scenario_id = f"SEN_{scenario_counter:03d}"
        print(f"\n{scenario_id}: 最小间距 = {distance}m")

        start_time = time.time()

        df, selected_by_n, optimal_N = opt.optimize_over_N(
            service_radius=1000,
            min_distance=float(distance),
            w_coverage=0.60, w_safety=0.30, w_overlap=0.10,
            scenario_id=scenario_id
        )

        row = df[df['N'] == optimal_N].iloc[0]
        selected_ids = set(opt.shelters.iloc[selected_by_n[optimal_N]]["Shelter_ID"].astype(str))
        jaccard, retention = similarity_metrics(baseline_ids, selected_ids)

        all_results.append({
            "Scenario_ID": scenario_id,
            "Parameter": "Min Distance",
            "Parameter_Value": str(distance),
            "Service_Radius": 1000.0,
            "Min_Distance": float(distance),
            "W_Coverage": 0.60,
            "W_Safety": 0.30,
            "W_Overlap": 0.10,
            "Optimal_N": optimal_N,
            "Coverage_Rate_Percent": row["Coverage_Rate_Percent"],
            "Avg_Safety": row["Avg_Safety"],
            "Efficiency": row["Efficiency"],
            "Overlap_Rate_Covered_Percent": row["Overlap_Rate_Covered_Percent"],
            "Objective_Value": row["Objective_Value"],
            "Jaccard_Similarity": jaccard,
            "Retention_Rate": retention,
            "Time_s": time.time() - start_time
        })
        print(
            f"  ✅ 最优N={optimal_N}, 覆盖率={row['Coverage_Rate_Percent']:.1f}%, 时间={time.time() - start_time:.1f}s")

    # ============================================================
    # 3. 权重敏感性
    # ============================================================
    print("\n" + "=" * 60)
    print("3. 权重敏感性分析")
    print("=" * 60)

    weight_scenarios = [
        {"wc": 0.50, "ws": 0.40, "wo": 0.10, "label": "(0.50,0.40,0.10)"},
        {"wc": 0.55, "ws": 0.35, "wo": 0.10, "label": "(0.55,0.35,0.10)"},
        {"wc": 0.60, "ws": 0.30, "wo": 0.10, "label": "(0.60,0.30,0.10)"},
        {"wc": 0.65, "ws": 0.25, "wo": 0.10, "label": "(0.65,0.25,0.10)"},
        {"wc": 0.70, "ws": 0.20, "wo": 0.10, "label": "(0.70,0.20,0.10)"},
        {"wc": 0.40, "ws": 0.50, "wo": 0.10, "label": "(0.40,0.50,0.10)"},
        {"wc": 0.55, "ws": 0.30, "wo": 0.15, "label": "(0.55,0.30,0.15)"},
    ]

    for weight in weight_scenarios:
        scenario_counter += 1
        scenario_id = f"SEN_{scenario_counter:03d}"
        print(f"\n{scenario_id}: 权重 = {weight['label']}")

        start_time = time.time()

        df, selected_by_n, optimal_N = opt.optimize_over_N(
            service_radius=1000,
            min_distance=500,
            w_coverage=weight["wc"],
            w_safety=weight["ws"],
            w_overlap=weight["wo"],
            scenario_id=scenario_id
        )

        row = df[df['N'] == optimal_N].iloc[0]
        selected_ids = set(opt.shelters.iloc[selected_by_n[optimal_N]]["Shelter_ID"].astype(str))
        jaccard, retention = similarity_metrics(baseline_ids, selected_ids)

        all_results.append({
            "Scenario_ID": scenario_id,
            "Parameter": "Weights",
            "Parameter_Value": weight["label"],
            "Service_Radius": 1000.0,
            "Min_Distance": 500.0,
            "W_Coverage": weight["wc"],
            "W_Safety": weight["ws"],
            "W_Overlap": weight["wo"],
            "Optimal_N": optimal_N,
            "Coverage_Rate_Percent": row["Coverage_Rate_Percent"],
            "Avg_Safety": row["Avg_Safety"],
            "Efficiency": row["Efficiency"],
            "Overlap_Rate_Covered_Percent": row["Overlap_Rate_Covered_Percent"],
            "Objective_Value": row["Objective_Value"],
            "Jaccard_Similarity": jaccard,
            "Retention_Rate": retention,
            "Time_s": time.time() - start_time
        })
        print(
            f"  ✅ 最优N={optimal_N}, 覆盖率={row['Coverage_Rate_Percent']:.1f}%, 时间={time.time() - start_time:.1f}s")

    # ============================================================
    # 保存结果
    # ============================================================
    df_results = pd.DataFrame(all_results)

    # 按SEN序号排序
    df_results = df_results.sort_values("Scenario_ID").reset_index(drop=True)

    # 保存到Excel
    df_results.to_excel(SENSITIVITY_EXCEL, index=False)

    print("\n" + "=" * 70)
    print("✅ 敏感性分析完成！")
    print(f"📁 结果已保存: {SENSITIVITY_EXCEL}")
    print(f"📊 共 {len(df_results)} 个场景")
    print("=" * 70)

    # 打印汇总表
    print("\n【数据预览】")
    print(df_results[["Scenario_ID", "Parameter", "Parameter_Value", "Optimal_N",
                      "Coverage_Rate_Percent", "Avg_Safety", "Efficiency"]].to_string(index=False))

    return df_results


if __name__ == "__main__":
    df = run_sensitivity_analysis()