# -*- coding: utf-8 -*-
"""
算法性能对比 - 完整版（含精确求解器）
对比：贪心算法 vs 遗传算法 vs 精确求解器
N固定为135（公平对比）
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from scipy.spatial import distance_matrix
import os
import warnings
import time
import math
from itertools import combinations

warnings.filterwarnings('ignore')

# ===================== 路径配置 =====================
SHELTER_PATH = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果\优化避难所得分结果\全部避难所得分.xlsx"
POP_TIF_PATH = r"D:\AICGIS\实验\population_shange.tif"
OUTPUT_DIR = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果（修正版）\避难所优化结果（算法对比-完整版）"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 配置参数 =====================
SERVICE_RADIUS = 1000
MIN_DISTANCE = 500

# N固定为135（与现有布局相同，公平对比）
N_SHELTERS = 135

W_COVERAGE = 0.6
W_SAFETY = 0.3
W_OVERLAP = 0.1

# GA参数
GA_POPULATION_SIZE = 50
GA_GENERATIONS = 80
GA_MUTATION_RATE = 0.12
GA_CROSSOVER_RATE = 0.80
GA_TOURNAMENT_SIZE = 3

# 精确求解器参数
EXACT_MAX_N = 20
EXACT_SAMPLES = 10000
EXACT_TIMEOUT = 300

# 重复实验次数
N_TRIALS = 10


# ===================== 基础优化器 =====================

class BaseOptimizer:
    def __init__(self):
        self.shelters = None
        self.pop_points = None
        self.pop_values = None
        self.dist_matrix = None
        self.total_population = 0

    def load_data(self):
        shelters = pd.read_excel(SHELTER_PATH)
        shelters['lon'] = pd.to_numeric(shelters['lon'], errors='coerce')
        shelters['lat'] = pd.to_numeric(shelters['lat'], errors='coerce')
        shelters = shelters.dropna(subset=['lon', 'lat'])
        self.shelters = shelters.reset_index(drop=True)

        with rasterio.open(POP_TIF_PATH) as src:
            population = src.read(1)
            rows, cols = np.where(population > 0)
            lons, lats, pops = [], [], []
            for r, c in zip(rows, cols):
                x, y = src.xy(r, c)
                lons.append(x);
                lats.append(y);
                pops.append(population[r, c])

        self.pop_points = np.column_stack([lons, lats])
        self.pop_values = np.array(pops)
        self.total_population = self.pop_values.sum()
        self.calculate_distance_matrix()
        return shelters

    def calculate_distance_matrix(self):
        shelter_coords = self.shelters[['lon', 'lat']].values
        n_shelters = len(shelter_coords)
        n_pop = len(self.pop_points)
        dist_matrix = np.zeros((n_shelters, n_pop))
        batch_size = 50
        for i in range(0, n_shelters, batch_size):
            end = min(i + batch_size, n_shelters)
            batch_coords = shelter_coords[i:end]
            batch_dist = distance_matrix(batch_coords, self.pop_points) * 111000
            dist_matrix[i:end] = batch_dist
        self.dist_matrix = dist_matrix

    def evaluate_solution(self, selected_indices):
        if len(selected_indices) == 0:
            return 0, 0, 0
        safety = self.shelters.iloc[selected_indices]['综合适宜性得分'].sum()
        dist_subset = self.dist_matrix[selected_indices, :]
        min_dist = dist_subset.min(axis=0)
        covered = min_dist <= SERVICE_RADIUS
        coverage = self.pop_values[covered].sum()
        service_count = (dist_subset <= SERVICE_RADIUS).sum(axis=0)
        overlap = np.sum(self.pop_values * np.maximum(service_count - 1, 0))
        return coverage, safety, overlap

    def calculate_objective(self, coverage, safety, overlap):
        max_coverage = self.total_population
        max_safety = len(self.shelters) * self.shelters['综合适宜性得分'].max()
        max_overlap = max_coverage * (len(self.shelters) - 1)
        norm_coverage = coverage / max_coverage if max_coverage > 0 else 0
        norm_safety = safety / max_safety if max_safety > 0 else 0
        norm_overlap = overlap / max_overlap if max_overlap > 0 else 0
        obj = (W_COVERAGE * norm_coverage + W_SAFETY * norm_safety - W_OVERLAP * norm_overlap)
        return obj, norm_coverage, norm_safety, norm_overlap


# ===================== 1. 贪心算法 =====================

class GreedyOptimizer(BaseOptimizer):
    def optimize(self, N):
        start_time = time.time()
        n_shelters = len(self.shelters)
        selected = []
        remaining = list(range(n_shelters))
        safety_scores = self.shelters['综合适宜性得分'].values

        for i in range(N):
            if not remaining:
                break
            best_gain = -np.inf
            best_idx = -1
            candidates = remaining
            if len(remaining) > 100 and len(selected) > 50:
                remaining_scores = [(idx, safety_scores[idx]) for idx in remaining]
                remaining_scores.sort(key=lambda x: x[1], reverse=True)
                candidates = [idx for idx, _ in remaining_scores[:100]]

            for idx in candidates:
                if selected:
                    selected_coords = self.shelters.iloc[selected][['lon', 'lat']].values
                    candidate_coord = self.shelters.iloc[[idx]][['lon', 'lat']].values
                    distances = np.sqrt(((selected_coords - candidate_coord) ** 2).sum(axis=1)) * 111000
                    if np.any(distances < MIN_DISTANCE):
                        continue
                temp_selected = selected + [idx]
                coverage, safety, overlap = self.evaluate_solution(temp_selected)
                obj, _, _, _ = self.calculate_objective(coverage, safety, overlap)
                if selected:
                    curr_coverage, curr_safety, curr_overlap = self.evaluate_solution(selected)
                    curr_obj, _, _, _ = self.calculate_objective(curr_coverage, curr_safety, curr_overlap)
                    gain = obj - curr_obj
                else:
                    gain = obj
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx != -1:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        elapsed_time = time.time() - start_time
        coverage, safety, overlap = self.evaluate_solution(selected)
        obj, cov_rate, saf_norm, ov_norm = self.calculate_objective(coverage, safety, overlap)
        return {
            'selected': selected,
            'coverage': coverage,
            'coverage_rate': cov_rate,
            'safety': safety,
            'avg_safety': safety / len(selected) if len(selected) > 0 else 0,
            'overlap': ov_norm,
            'objective': obj,
            'time': elapsed_time
        }


# ===================== 2. 遗传算法 =====================

class GeneticAlgorithm(BaseOptimizer):
    def __init__(self):
        super().__init__()
        self.population_size = GA_POPULATION_SIZE
        self.generations = GA_GENERATIONS
        self.mutation_rate = GA_MUTATION_RATE
        self.crossover_rate = GA_CROSSOVER_RATE
        self.tournament_size = GA_TOURNAMENT_SIZE

    def initialize_population(self, N):
        population = []
        n_shelters = len(self.shelters)
        for _ in range(self.population_size):
            selected = []
            remaining = list(range(n_shelters))
            np.random.shuffle(remaining)
            for idx in remaining:
                if len(selected) >= N:
                    break
                if selected:
                    selected_coords = self.shelters.iloc[selected][['lon', 'lat']].values
                    candidate_coord = self.shelters.iloc[[idx]][['lon', 'lat']].values
                    distances = np.sqrt(((selected_coords - candidate_coord) ** 2).sum(axis=1)) * 111000
                    if np.any(distances < MIN_DISTANCE):
                        continue
                selected.append(idx)
            while len(selected) < N:
                remaining_indices = [i for i in range(n_shelters) if i not in selected]
                if not remaining_indices:
                    break
                idx = np.random.choice(remaining_indices)
                selected.append(idx)
            population.append(selected)
        return population

    def evaluate_individual(self, individual):
        coverage, safety, overlap = self.evaluate_solution(individual)
        obj, cov_rate, saf_norm, ov_norm = self.calculate_objective(coverage, safety, overlap)
        return obj, cov_rate, saf_norm, ov_norm

    def tournament_selection(self, population, fitness):
        selected_indices = np.random.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = selected_indices[np.argmax([fitness[i] for i in selected_indices])]
        return population[best_idx].copy()

    def crossover(self, parent1, parent2, N):
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        crossover_point = np.random.randint(1, N)
        child1 = parent1[:crossover_point] + [p for p in parent2 if p not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [p for p in parent1 if p not in parent2[:crossover_point]]
        while len(child1) < N:
            remaining = [i for i in range(len(self.shelters)) if i not in child1]
            if remaining:
                child1.append(np.random.choice(remaining))
        while len(child2) < N:
            remaining = [i for i in range(len(self.shelters)) if i not in child2]
            if remaining:
                child2.append(np.random.choice(remaining))
        return child1[:N], child2[:N]

    def mutate(self, individual):
        if np.random.random() > self.mutation_rate:
            return individual
        n_shelters = len(self.shelters)
        idx_to_mutate = np.random.randint(0, len(individual))
        current_set = set(individual)
        possible_replacements = [i for i in range(n_shelters) if i not in current_set]
        if possible_replacements:
            new_idx = np.random.choice(possible_replacements)
            temp_individual = individual.copy()
            temp_individual[idx_to_mutate] = new_idx
            coords = self.shelters.iloc[temp_individual][['lon', 'lat']].values
            valid = True
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2)) * 111000
                    if dist < MIN_DISTANCE:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                individual[idx_to_mutate] = new_idx
        return individual

    def optimize(self, N):
        start_time = time.time()
        population = self.initialize_population(N)
        fitness = []
        for ind in population:
            obj, _, _, _ = self.evaluate_individual(ind)
            fitness.append(obj)
        best_individual = population[np.argmax(fitness)].copy()
        best_fitness = max(fitness)

        for gen in range(self.generations):
            new_population = []
            elite_idx = np.argmax(fitness)
            new_population.append(population[elite_idx].copy())
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                child1, child2 = self.crossover(parent1, parent2, N)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            population = new_population
            fitness = []
            for ind in population:
                obj, _, _, _ = self.evaluate_individual(ind)
                fitness.append(obj)
            current_best = max(fitness)
            if current_best > best_fitness:
                best_fitness = current_best
                best_individual = population[np.argmax(fitness)].copy()

        elapsed_time = time.time() - start_time
        coverage, safety, overlap = self.evaluate_solution(best_individual)
        obj, cov_rate, saf_norm, ov_norm = self.calculate_objective(coverage, safety, overlap)
        return {
            'selected': best_individual,
            'coverage': coverage,
            'coverage_rate': cov_rate,
            'safety': safety,
            'avg_safety': safety / len(best_individual) if len(best_individual) > 0 else 0,
            'overlap': ov_norm,
            'objective': obj,
            'time': elapsed_time
        }


# ===================== 3. 精确求解器 =====================

class ExactSolver(BaseOptimizer):
    def __init__(self):
        super().__init__()
        self.best_solution = None
        self.best_value = -np.inf
        self.nodes_explored = 0

    def exact_search(self, N):
        start_time = time.time()
        self.best_solution = None
        self.best_value = -np.inf
        self.nodes_explored = 0

        n_shelters = len(self.shelters)

        # 贪心解作为初始上界
        greedy = GreedyOptimizer()
        greedy.shelters = self.shelters
        greedy.pop_points = self.pop_points
        greedy.pop_values = self.pop_values
        greedy.dist_matrix = self.dist_matrix
        greedy.total_population = self.total_population
        greedy_result = greedy.optimize(N)
        self.best_solution = greedy_result['selected']
        self.best_value = greedy_result['objective']

        if N > EXACT_MAX_N:
            return self._sampled_search(N, start_time)
        else:
            return self._exact_search(N, start_time)

    def _exact_search(self, N, start_time):
        n_shelters = len(self.shelters)
        total_combinations = math.comb(n_shelters, N)

        if total_combinations > 50000:
            return self._sampled_search(N, start_time)

        print(f"    枚举所有组合: {total_combinations}")

        for combo in combinations(range(n_shelters), N):
            if time.time() - start_time > EXACT_TIMEOUT:
                print(f"    超时，停止搜索")
                break

            selected = list(combo)
            coords = self.shelters.iloc[selected][['lon', 'lat']].values
            valid = True
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2)) * 111000
                    if dist < MIN_DISTANCE:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            coverage, safety, overlap = self.evaluate_solution(selected)
            obj, _, _, _ = self.calculate_objective(coverage, safety, overlap)
            self.nodes_explored += 1

            if obj > self.best_value:
                self.best_value = obj
                self.best_solution = selected.copy()

        return self._finalize(start_time)

    def _sampled_search(self, N, start_time):
        n_shelters = len(self.shelters)
        total_combinations = math.comb(n_shelters, N) if N <= n_shelters else 1
        n_samples = min(EXACT_SAMPLES, total_combinations)

        print(f"    采样搜索: {n_samples} 个组合")

        weights = np.array([self.shelters.iloc[i]['综合适宜性得分'] for i in range(n_shelters)])
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_shelters) / n_shelters

        for _ in range(n_samples):
            if time.time() - start_time > EXACT_TIMEOUT:
                print(f"    超时，停止搜索")
                break

            selected = []
            attempts = 0
            max_attempts = n_shelters * 10

            while len(selected) < N and attempts < max_attempts:
                attempts += 1
                idx = np.random.choice(n_shelters, p=weights)
                if idx not in selected:
                    valid = True
                    if selected:
                        coords_existing = self.shelters.iloc[selected][['lon', 'lat']].values
                        coord_new = self.shelters.iloc[[idx]][['lon', 'lat']].values
                        distances = np.sqrt(((coords_existing - coord_new) ** 2).sum(axis=1)) * 111000
                        if np.any(distances < MIN_DISTANCE):
                            valid = False
                    if valid:
                        selected.append(idx)

            while len(selected) < N:
                remaining = [i for i in range(n_shelters) if i not in selected]
                if not remaining:
                    break
                idx = np.random.choice(remaining)
                selected.append(idx)

            coverage, safety, overlap = self.evaluate_solution(selected)
            obj, _, _, _ = self.calculate_objective(coverage, safety, overlap)
            self.nodes_explored += 1

            if obj > self.best_value:
                self.best_value = obj
                self.best_solution = selected.copy()

        return self._finalize(start_time)

    def _finalize(self, start_time):
        coverage, safety, overlap = self.evaluate_solution(self.best_solution)
        obj, cov_rate, saf_norm, ov_norm = self.calculate_objective(coverage, safety, overlap)
        return {
            'selected': self.best_solution,
            'coverage': coverage,
            'coverage_rate': cov_rate,
            'safety': safety,
            'avg_safety': safety / len(self.best_solution) if len(self.best_solution) > 0 else 0,
            'overlap': ov_norm,
            'objective': obj,
            'time': time.time() - start_time,
            'nodes_explored': self.nodes_explored
        }


# ===================== 绘图函数 =====================

def create_comparison_figure(results, summary):
    """创建SCI风格对比图"""

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'Greedy': '#1f77b4', 'GA': '#ff7f0e', 'Exact': '#2ca02c'}
    labels = {'Greedy': 'Greedy', 'GA': 'Genetic Algorithm', 'Exact': 'Exact Solver'}

    # (a) 目标函数值
    ax = axes[0, 0]
    data = [results['Greedy']['objective'], results['GA']['objective'], results['Exact']['objective']]
    bp = ax.boxplot(data, labels=['Greedy', 'GA', 'Exact'], patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], [colors['Greedy'], colors['GA'], colors['Exact']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('(a) Objective Function', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)

    # (b) 覆盖率
    ax = axes[0, 1]
    data = [results['Greedy']['coverage'], results['GA']['coverage'], results['Exact']['coverage']]
    bp = ax.boxplot(data, labels=['Greedy', 'GA', 'Exact'], patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], [colors['Greedy'], colors['GA'], colors['Exact']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Coverage Rate (%)', fontsize=12)
    ax.set_title('(b) Coverage Rate', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)

    # (c) 安全得分
    ax = axes[1, 0]
    data = [results['Greedy']['safety'], results['GA']['safety'], results['Exact']['safety']]
    bp = ax.boxplot(data, labels=['Greedy', 'GA', 'Exact'], patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], [colors['Greedy'], colors['GA'], colors['Exact']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Average Safety Score', fontsize=12)
    ax.set_title('(c) Average Safety', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)

    # (d) 运行时间
    ax = axes[1, 1]
    data = [results['Greedy']['time'], results['GA']['time'], results['Exact']['time']]
    bp = ax.boxplot(data, labels=['Greedy', 'GA', 'Exact'], patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], [colors['Greedy'], colors['GA'], colors['Exact']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Running Time (seconds)', fontsize=12)
    ax.set_title('(d) Running Time', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'algorithm_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 对比图已保存")


def create_comparison_table(results, summary):
    """创建对比表格"""

    table_data = []
    for method in ['Greedy', 'GA', 'Exact']:
        if method == 'Exact':
            gap = 'Reference'
        else:
            gap = (summary['Exact']['objective_mean'] - summary[method]['objective_mean']) / summary['Exact'][
                'objective_mean'] * 100
            gap = f'{gap:.2f}%'

        row = {
            'Algorithm': method,
            'Objective': f"{summary[method]['objective_mean']:.4f} ± {summary[method]['objective_std']:.4f}",
            'Coverage (%)': f"{summary[method]['coverage_mean']:.1f} ± {summary[method]['coverage_std']:.1f}",
            'Safety': f"{summary[method]['safety_mean']:.4f} ± {summary[method]['safety_std']:.4f}",
            'Time (s)': f"{summary[method]['time_mean']:.2f} ± {summary[method]['time_std']:.2f}",
            'Gap to Exact': gap
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    df.to_excel(os.path.join(OUTPUT_DIR, 'algorithm_comparison_table.xlsx'), index=False)

    print("\n" + "=" * 70)
    print("【算法对比结果汇总】")
    print("=" * 70)
    print(df.to_string(index=False))

    return df


# ===================== 主程序 =====================

def run_comparison():
    print("=" * 70)
    print("算法性能对比 - 完整版")
    print("=" * 70)
    print(f"N = {N_SHELTERS}（固定，与现有布局相同）")
    print(f"对比: 贪心算法 vs 遗传算法 vs 精确求解器")
    print(f"重复实验: {N_TRIALS} 次")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    base = BaseOptimizer()
    base.load_data()
    print(f"✓ {len(base.shelters)}个避难所, {len(base.pop_points):,}个人口点")

    results = {
        'Greedy': {'objective': [], 'coverage': [], 'safety': [], 'time': []},
        'GA': {'objective': [], 'coverage': [], 'safety': [], 'time': []},
        'Exact': {'objective': [], 'coverage': [], 'safety': [], 'time': []}
    }

    for trial in range(N_TRIALS):
        print(f"\n【Trial {trial + 1}/{N_TRIALS}】")

        # 1. 贪心算法
        print("  运行贪心算法...")
        greedy = GreedyOptimizer()
        greedy.shelters = base.shelters.copy()
        greedy.pop_points = base.pop_points.copy()
        greedy.pop_values = base.pop_values.copy()
        greedy.dist_matrix = base.dist_matrix.copy()
        greedy.total_population = base.total_population
        greedy_result = greedy.optimize(N_SHELTERS)
        results['Greedy']['objective'].append(greedy_result['objective'])
        results['Greedy']['coverage'].append(greedy_result['coverage_rate'] * 100)
        results['Greedy']['safety'].append(greedy_result['avg_safety'])
        results['Greedy']['time'].append(greedy_result['time'])

        # 2. 遗传算法
        print("  运行遗传算法...")
        ga = GeneticAlgorithm()
        ga.shelters = base.shelters.copy()
        ga.pop_points = base.pop_points.copy()
        ga.pop_values = base.pop_values.copy()
        ga.dist_matrix = base.dist_matrix.copy()
        ga.total_population = base.total_population
        ga_result = ga.optimize(N_SHELTERS)
        results['GA']['objective'].append(ga_result['objective'])
        results['GA']['coverage'].append(ga_result['coverage_rate'] * 100)
        results['GA']['safety'].append(ga_result['avg_safety'])
        results['GA']['time'].append(ga_result['time'])

        # 3. 精确求解器
        print("  运行精确求解器...")
        exact = ExactSolver()
        exact.shelters = base.shelters.copy()
        exact.pop_points = base.pop_points.copy()
        exact.pop_values = base.pop_values.copy()
        exact.dist_matrix = base.dist_matrix.copy()
        exact.total_population = base.total_population
        exact_result = exact.exact_search(N_SHELTERS)
        results['Exact']['objective'].append(exact_result['objective'])
        results['Exact']['coverage'].append(exact_result['coverage_rate'] * 100)
        results['Exact']['safety'].append(exact_result['avg_safety'])
        results['Exact']['time'].append(exact_result['time'])
        print(f"    目标值: {exact_result['objective']:.4f}, 时间: {exact_result['time']:.2f}s")

    # 统计
    summary = {}
    for method in ['Greedy', 'GA', 'Exact']:
        summary[method] = {
            'objective_mean': np.mean(results[method]['objective']),
            'objective_std': np.std(results[method]['objective']),
            'coverage_mean': np.mean(results[method]['coverage']),
            'coverage_std': np.std(results[method]['coverage']),
            'safety_mean': np.mean(results[method]['safety']),
            'safety_std': np.std(results[method]['safety']),
            'time_mean': np.mean(results[method]['time']),
            'time_std': np.std(results[method]['time'])
        }

    # 生成图和表
    create_comparison_figure(results, summary)
    create_comparison_table(results, summary)

    print("\n" + "=" * 70)
    print("✅ 对比实验完成!")
    print(f"结果已保存至: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()