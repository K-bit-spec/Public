[![DOI](https://zenodo.org/badge/1338028482.svg)](https://doi.org/10.5281/zenodo.21990474)
# Shelter Optimization Framework

This repository contains the code for:  
**"A proximity‑based multi‑indicator framework for optimizing emergency shelter layout in a mountainous city: A case study of Dujiangyan, China"**

## Repository Structure

| File | Description |
|---|---|
| `贪心算法与优化对比.py` | Main greedy optimization and result export |
| `四种灾害权重计算.py` | EWM weight calculation for four hazard types |
| `合并的适宜性.py` | Combined suitability assessment |
| `权重可视化.py` | Weight distribution visualization |
| `RF重要性验证（基尼 vs 排列 vs SHAP）.py` | RF importance comparison |
| `EWM+RF组合比例敏感性.py` | Sensitivity analysis of α combination |
| `目标函数的敏感性分析.py` | Objective weight sensitivity analysis |
| `目标函数的敏感性可视化.py` | Sensitivity visualization |
| `三种算法比较.py` | Algorithm comparison |
| `最后分专业灾害避难所.py` | Hazard-specific shelter selection |
| `现有避难所适宜性.py` | Existing shelter suitability evaluation |

## Requirements

Python 3.8+

```bash
pip install -r requirements.txt# Pulic
