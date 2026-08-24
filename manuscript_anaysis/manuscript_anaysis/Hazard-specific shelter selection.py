import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ===================== 路径配置 =====================
INPUT_PATH = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果\优化避难所得分结果\135优化避难所得分.xlsx"
OUTPUT_DIR = r"D:\AICGIS\小文献代码结果\修改灾害特征分组后的实验\避难所优化结果（修正版）\贪心避难所综合\避难所分级结果"

# ===================== 灾害得分列名 =====================
HAZARD_COLS = ['地震适宜性得分', '地质灾害适宜性得分', '洪涝暴雨适宜性得分', '火灾与公卫适宜性得分']
HAZARD_NAMES = ['Earthquake', 'Geological hazard', 'Flood', 'Fire']


def classify_shelters_by_performance(df):
    """
    根据综合性能和灾害应对能力对避难所进行分级
    """
    df = df.copy()

    # 计算各灾害得分的平均值作为阈值
    thresholds = {}
    for i, hazard in enumerate(HAZARD_NAMES):
        col = HAZARD_COLS[i]
        thresholds[hazard] = df[col].mean()

    # 判断每个避难所的各灾害得分是否高于平均值
    all_above = []
    any_above = []

    for idx, row in df.iterrows():
        above_list = []
        for i, hazard in enumerate(HAZARD_NAMES):
            col = HAZARD_COLS[i]
            is_above = row[col] > thresholds[hazard]
            above_list.append(is_above)

        all_above.append(all(above_list))
        any_above.append(any(above_list))

    df['All_Hazards_Above_Avg'] = all_above
    df['Any_Hazard_Above_Avg'] = any_above

    # 分类
    shelter_types = []
    for idx, row in df.iterrows():
        if row['All_Hazards_Above_Avg']:
            shelter_types.append('Comprehensive Shelter')
        elif row['Any_Hazard_Above_Avg']:
            shelter_types.append('Specialized Shelter')
        else:
            shelter_types.append('Basic Shelter')

    df['Shelter_Type'] = shelter_types

    # 对于Specialized Shelter，识别具体优势灾害
    def get_specialization(row):
        if row['Shelter_Type'] != 'Specialized Shelter':
            return 'None'

        advantages = []
        for i, hazard in enumerate(HAZARD_NAMES):
            col = HAZARD_COLS[i]
            if row[col] > thresholds[hazard]:
                advantages.append(hazard)

        if len(advantages) == 1:
            return f"{advantages[0]}-Specialized"
        elif len(advantages) > 1:
            return "Multi-Hazard Specialized"
        else:
            return "None"

    df['Specialization'] = df.apply(get_specialization, axis=1)

    # 计算综合得分排名
    df['综合得分排名'] = df['综合适宜性得分'].rank(ascending=False, method='min').astype(int)

    return df


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    df = pd.read_excel(INPUT_PATH)
    print(f"✅ 加载数据: {len(df)} 个避难所")

    # 检查必要的列是否存在
    required_cols = ['名称', 'lon', 'lat']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ 警告: 缺少以下列: {missing_cols}")
        # 如果没有这些列，创建空列
        for col in missing_cols:
            df[col] = ''

    # 打印各灾害得分范围
    print("\n灾害得分范围:")
    for i, hazard in enumerate(HAZARD_NAMES):
        col = HAZARD_COLS[i]
        if col in df.columns:
            print(f"  {hazard}: {df[col].min():.3f} - {df[col].max():.3f} (平均: {df[col].mean():.3f})")

    # 分类避难所
    df_classified = classify_shelters_by_performance(df)

    # 打印分类统计
    print("\n" + "=" * 60)
    print("分类统计:")
    print("=" * 60)
    type_counts = df_classified['Shelter_Type'].value_counts()
    for shelter_type, count in type_counts.items():
        percentage = count / len(df_classified) * 100
        print(f"  {shelter_type}: {count} 个 ({percentage:.1f}%)")

    # 打印Specialized Shelters的详细分类
    spec_df = df_classified[df_classified['Shelter_Type'] == 'Specialized Shelter']
    if len(spec_df) > 0:
        print("\nSpecialized Shelters 详细分类:")
        spec_counts = spec_df['Specialization'].value_counts()
        for spec, count in spec_counts.items():
            print(f"  {spec}: {count} 个")

    # 选择要保存的列（添加名称、lon、lat）
    output_columns = ['名称', 'lon', 'lat', '综合适宜性得分', '综合得分排名'] + HAZARD_COLS + ['Shelter_Type',
                                                                                               'Specialization']

    # 确保所有列都存在
    available_columns = [col for col in output_columns if col in df_classified.columns]
    df_output = df_classified[available_columns]

    # 按综合得分排序
    df_output = df_output.sort_values('综合适宜性得分', ascending=False)

    # 保存Excel
    output_path = os.path.join(OUTPUT_DIR, 'shelter_classification_result.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_output.to_excel(writer, sheet_name='避难所分类结果', index=False)

        # 添加统计表
        summary = pd.DataFrame({
            'Shelter Type': type_counts.index,
            'Count': type_counts.values,
            'Percentage (%)': (type_counts.values / len(df_classified) * 100).round(1)
        })
        summary.to_excel(writer, sheet_name='分类统计', index=False)

        if len(spec_df) > 0:
            spec_summary = pd.DataFrame({
                'Specialization': spec_counts.index,
                'Count': spec_counts.values,
                'Percentage (%)': (spec_counts.values / len(spec_df) * 100).round(1)
            })
            spec_summary.to_excel(writer, sheet_name='专业化统计', index=False)

    print(f"\n✅ 分类结果已保存: {output_path}")

    # 打印示例
    print("\n" + "=" * 60)
    print("分类示例 (前10个，按综合得分排序):")
    print("=" * 60)
    # 显示关键列
    display_cols = ['名称', '综合适宜性得分', 'Shelter_Type', 'Specialization']
    display_cols = [col for col in display_cols if col in df_output.columns]
    print(df_output[display_cols].head(10).to_string())

    # 打印各类型的平均得分
    print("\n" + "=" * 60)
    print("各类型避难所平均综合得分:")
    print("=" * 60)
    for shelter_type in type_counts.index:
        mean_score = df_classified[df_classified['Shelter_Type'] == shelter_type]['综合适宜性得分'].mean()
        print(f"  {shelter_type}: {mean_score:.4f}")

    # 打印经纬度范围（确认数据）
    if 'lon' in df_classified.columns and 'lat' in df_classified.columns:
        print("\n经纬度范围:")
        print(f"  经度: {df_classified['lon'].min():.6f} - {df_classified['lon'].max():.6f}")
        print(f"  纬度: {df_classified['lat'].min():.6f} - {df_classified['lat'].max():.6f}")


if __name__ == "__main__":
    main()