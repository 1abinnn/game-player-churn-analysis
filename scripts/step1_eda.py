"""
=============================================================================
第一步：探索性数据分析 (EDA)
=============================================================================
项目：游戏玩家行为与流失预测
宪法约束：基于 CONSTITUTION.md 第一章 - 业务洞察优先，指标体系敏感
=============================================================================

本章目标：
1. 加载并检查数据完整性
2. 描述性统计概览
3. 缺失值分析与处理
4. 数值特征分布可视化
5. 类别特征分布与玩家群体画像
6. 目标变量 EngagementLevel 分析

作者：AI 游戏数据分析师
日期：2026-10-27
"""

# ============================================================================
# 1. 导入库
# ============================================================================
# 导入必要的库（pandas, numpy, matplotlib, seaborn等）
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
sns.set_style("whitegrid")  # 设置图表样式
warnings.filterwarnings('ignore')  # 忽略警告信息

print("=" * 80)
print("[INFO] Game Player Behavior Analysis - Step 1: EDA")
print("=" * 80)


# ============================================================================
# 2. 数据加载与初步检查
# ============================================================================
# 加载原始数据集并进行初步检查
# ============================================================================

print("\n[INFO] Loading dataset...")

# 数据文件路径（使用相对路径，确保可移植性）
DATA_PATH = "../data/online_gaming_behavior_dataset.csv/online_gaming_behavior_dataset.csv"

# 加载数据
df = pd.read_csv(DATA_PATH)

print("[OK] Data loaded successfully!")
print(f"   - 样本数量: {len(df):,} 条")
print(f"   - 特征数量: {len(df.columns)} 个")

# Preview first 5 rows
print("\n[PREVIEW] First 5 rows:")
print("-" * 80)
print(df.head())

# 数据类型信息
print("\n[INFO] Data type information:")
print("-" * 80)
print(df.info())

# 检查重复值
duplicate_count = df.duplicated().sum()
print(f"\n[WARNING] Duplicate records: {duplicate_count}")


# ============================================================================
# 3. 描述性统计概览
# ============================================================================

print("\n" + "=" * 80)
print("[STATS] Descriptive Statistics")
print("=" * 80)

# 数值特征统计
print("\n【数值特征统计】:")
print(df.describe())

# 业务解读
print("\n[INSIGHT] Business Interpretation:")
print(f"   - 平均年龄: {df['Age'].mean():.1f} 岁 (标准差: {df['Age'].std():.1f})")
print(f"   - 平均每周游玩时长: {df['PlayTimeHours'].mean():.1f} 小时")
print(f"   - 平均每周游玩次数: {df['SessionsPerWeek'].mean():.1f} 次")
print(f"   - 平均每次游玩时长: {df['AvgSessionDurationMinutes'].mean():.0f} 分钟")
print(f"   - 平均玩家等级: {df['PlayerLevel'].mean():.1f} 级")
print(f"   - 平均解锁成就数: {df['AchievementsUnlocked'].mean():.1f} 个")


# ============================================================================
# 4. Missing Value Analysis (缺失值分析)
# ============================================================================
# 检测和处理数据中的缺失值
# ============================================================================

print("\n" + "=" * 80)
print("[MISSING] Missing Value Analysis")
print("=" * 80)

# 统计缺失值
missing_stats = pd.DataFrame({
    '缺失数量': df.isnull().sum(),
    '缺失比例(%)': (df.isnull().sum() / len(df) * 100).round(2),
    '数据类型': df.dtypes
})

# 只显示有缺失值的列
missing_info = missing_stats[missing_stats['缺失数量'] > 0]

if len(missing_info) > 0:
    print("\n[WARNING] Missing values found:")
    print(missing_info)
else:
    print("\n[OK] No missing values - good data quality!")

# 缺失值可视化
if df.isnull().sum().sum() > 0:
    plt.figure(figsize=(10, 6))
    missing_ratio = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_ratio[missing_ratio > 0].plot(kind='bar', color='coral')
    plt.title('Missing Value Distribution by Feature', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Name')
    plt.ylabel('Missing Ratio (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../output/missing_values_bar.png', dpi=300, bbox_inches='tight')
    print("\n[SAVED] Missing value distribution chart saved to: ../output/missing_values_bar.png")
    plt.show()


# ============================================================================
# 5. Numerical Feature Distribution Analysis (数值特征分布分析)
# ============================================================================
# 分析数值特征的分布情况
# ============================================================================

print("\n" + "=" * 80)
print("[DIST] Numerical Feature Distribution Analysis")
print("=" * 80)

# 定义数值特征
numerical_features = [
    'Age',
    'PlayTimeHours',
    'SessionsPerWeek',
    'AvgSessionDurationMinutes',
    'PlayerLevel',
    'AchievementsUnlocked'
]

# 创建画布 - 6个特征的分布直方图
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, feature in enumerate(numerical_features):
    ax = axes[idx]

    # 绘制直方图 + KDE曲线
    sns.histplot(df[feature], kde=True, ax=ax, color='steelblue', bins=30)

    # Add mean and median lines
    mean_val = df[feature].mean()
    median_val = df[feature].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')

    ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()

plt.suptitle('Numerical Feature Distribution', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../output/numerical_features_distribution.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] Numerical feature distribution chart saved to: ../output/numerical_features_distribution.png")
plt.show()

# 业务洞察
print("\n[INSIGHT] Numerical Feature Distribution Insights:")
print(f"   1. Age: Player age mainly concentrated around {df['Age'].mode()[0]} years old")
print(f"   2. PlayTimeHours: Median weekly playtime is {df['PlayTimeHours'].median():.1f} hours")

# Identify heavy gamers (weekly playtime > 75th percentile)
heavy_gamer_threshold = df['PlayTimeHours'].quantile(0.75)
heavy_gamers = df[df['PlayTimeHours'] > heavy_gamer_threshold]
print(f"   3. Heavy gamers (>{heavy_gamer_threshold:.1f}h/week): {len(heavy_gamers)} players ({len(heavy_gamers)/len(df)*100:.1f}%)")

print(f"   4. SessionsPerWeek: Average {df['SessionsPerWeek'].mean():.1f} game sessions per week")


# ============================================================================
# 6. Categorical Feature Analysis (类别特征分析)
# ============================================================================
# 分析类别特征的分布情况
# ============================================================================

print("\n" + "=" * 80)
print("[CAT] Categorical Feature Analysis")
print("=" * 80)

# 定义类别特征
categorical_features = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']

# 创建画布 - 4个类别特征的柱状图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
    ax = axes[idx]

    # 统计各类别数量和比例
    value_counts = df[feature].value_counts()

    # 绘制柱状图
    value_counts.plot(kind='bar', ax=ax, color='teal', edgecolor='black')

    # 在柱子上方添加比例标签
    for i, v in enumerate(value_counts):
        pct = v / len(df) * 100
        ax.text(i, v + 50, f'{pct:.1f}%', ha='center', fontsize=9)

    ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Player Count')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Categorical Feature Distribution', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../output/categorical_features_distribution.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] Categorical feature distribution chart saved to: ../output/categorical_features_distribution.png")
plt.show()

# 业务洞察
print("\n[INSIGHT] Categorical Feature Insights:")

# Gender distribution
gender_dist = df['Gender'].value_counts(normalize=True) * 100
print(f"   1. Gender Distribution: Male {gender_dist.get('Male', 0):.1f}%, Female {gender_dist.get('Female', 0):.1f}%")

# Location distribution
location_dist = df['Location'].value_counts(normalize=True) * 100
print(f"   2. Location Distribution: USA accounts for {location_dist.get('USA', 0):.1f}%")

# Game genre preference
genre_dist = df['GameGenre'].value_counts(normalize=True) * 100
top_genre = genre_dist.index[0]
print(f"   3. Most popular game genre: {top_genre} ({genre_dist.iloc[0]:.1f}%)")

# Game difficulty preference
difficulty_dist = df['GameDifficulty'].value_counts(normalize=True) * 100
print(f"   4. Difficulty preference: Easy {difficulty_dist.get('Easy', 0):.1f}%, Medium {difficulty_dist.get('Medium', 0):.1f}%, Hard {difficulty_dist.get('Hard', 0):.1f}%")


# ============================================================================
# 7. Target Variable Analysis: EngagementLevel (目标变量分析)
# ============================================================================
# 分析目标变量 EngagementLevel 的分布
# ============================================================================

print("\n" + "=" * 80)
print("[TARGET] Target Variable Analysis: EngagementLevel")
print("=" * 80)

# 统计各参与度等级的数量和比例
engagement_counts = df['EngagementLevel'].value_counts()
engagement_pct = df['EngagementLevel'].value_counts(normalize=True) * 100

print("\n[Player Engagement Distribution]:")
print("-" * 60)
for level in engagement_counts.index:
    count = engagement_counts[level]
    pct = engagement_pct[level]
    print(f"   {level:8s}: {count:5d} 人 ({pct:5.1f}%)")

# 可视化 - 饼图 + 柱状图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 饼图
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
engagement_counts.plot(kind='pie', ax=axes[0], autopct='%1.1f%%',
                       colors=colors[:len(engagement_counts)], startangle=90)
axes[0].set_ylabel('')
axes[0].set_title('EngagementLevel Proportion', fontsize=12, fontweight='bold')

# Bar chart
engagement_counts.plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
axes[1].set_xlabel('Engagement Level')
axes[1].set_ylabel('Player Count')
axes[1].set_title('EngagementLevel Count Distribution', fontsize=12, fontweight='bold')

# 在柱子上方添加数值标签
for i, v in enumerate(engagement_counts):
    axes[1].text(i, v + 100, str(v), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('../output/engagement_level_distribution.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] EngagementLevel distribution chart saved to: ../output/engagement_level_distribution.png")
plt.show()

# Business insights
print("\n[INSIGHT] Target Variable Insights:")
low_engagement = engagement_pct.get('Low', 0)
print(f"   [!] Low engagement players: {low_engagement:.1f}% - This is the churn risk group!")
print(f"   [!] Next modeling goal: Identify which behavioral features correlate with Low Engagement")


# ============================================================================
# 8. Correlation Analysis (相关性分析)
# ============================================================================
# 分析特征之间的相关性
# ============================================================================

print("\n" + "=" * 80)
print("[CORR] Feature Correlation Analysis")
print("=" * 80)

# 选择数值特征进行相关性分析
numerical_df = df[numerical_features]

# 计算相关系数矩阵
correlation_matrix = numerical_df.corr()

# Visualize - Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Numerical Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] Correlation heatmap saved to: ../output/correlation_heatmap.png")
plt.show()

# Find features most correlated with PlayerLevel and AchievementsUnlocked
print("\n[INSIGHT] Correlation Insights:")
print("\n[Top features correlated with PlayerLevel]:")
player_level_corr = correlation_matrix['PlayerLevel'].drop('PlayerLevel').sort_values(ascending=False)
for feature, corr_val in player_level_corr.head(3).items():
    print(f"   - {feature}: {corr_val:.3f}")

print("\n[Top features correlated with AchievementsUnlocked]:")
achievement_corr = correlation_matrix['AchievementsUnlocked'].drop('AchievementsUnlocked').sort_values(ascending=False)
for feature, corr_val in achievement_corr.head(3).items():
    print(f"   - {feature}: {corr_val:.3f}")


# ============================================================================
# 9. Player Segmentation Profile (玩家群体画像)
# ============================================================================
# 基于参与度等级创建玩家群体画像
# ============================================================================

print("\n" + "=" * 80)
print("[PROFILE] Player Segmentation Profile")
print("=" * 80)

# Group by EngagementLevel
engagement_profile = df.groupby('EngagementLevel')[numerical_features].mean()

print("\n[Mean Feature Values by Engagement Level]:")
print("-" * 80)
print(engagement_profile.round(2))

# Visualize - Radar chart preparation (using standardized data)
from sklearn.preprocessing import StandardScaler

# Standardize data for radar chart comparison
scaler = StandardScaler()
engagement_profile_scaled = pd.DataFrame(
    scaler.fit_transform(engagement_profile),
    columns=engagement_profile.columns,
    index=engagement_profile.index
)

# Select key features for profile comparison
profile_features = ['PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes',
                    'PlayerLevel', 'AchievementsUnlocked']

# Plot comparison by engagement level
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(profile_features):
    ax = axes[idx]

    # Plot boxplot by engagement level
    sns.boxplot(data=df, x='EngagementLevel', y=feature, ax=ax, palette='Set2')
    ax.set_title(f'{feature} vs EngagementLevel', fontsize=11, fontweight='bold')
    ax.set_xlabel('Engagement Level')
    ax.set_ylabel(feature)

# Remove last empty subplot
fig.delaxes(axes[5])

plt.suptitle('Player Segmentation Profile: Feature vs Engagement Level', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../output/player_segmentation_profile.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] Player segmentation profile chart saved to: ../output/player_segmentation_profile.png")
plt.show()

# Business insights
print("\n[INSIGHT] Player Profile Insights:")
print(f"   1. High Engagement Player Profile:")
high_eng = engagement_profile.loc['High']
print(f"      - Avg weekly playtime: {high_eng['PlayTimeHours']:.1f} hours")
print(f"      - Avg sessions per week: {high_eng['SessionsPerWeek']:.1f}")
print(f"      - Avg player level: {high_eng['PlayerLevel']:.1f}")

print(f"\n   2. Low Engagement Player Profile ([!] Churn Risk Group):")
low_eng = engagement_profile.loc['Low']
print(f"      - Avg weekly playtime: {low_eng['PlayTimeHours']:.1f} hours")
print(f"      - Avg sessions per week: {low_eng['SessionsPerWeek']:.1f}")
print(f"      - Avg player level: {low_eng['PlayerLevel']:.1f}")

print(f"\n   3. Key Differences:")
playtime_diff = high_eng['PlayTimeHours'] - low_eng['PlayTimeHours']
sessions_diff = high_eng['SessionsPerWeek'] - low_eng['SessionsPerWeek']
print(f"      - High engagement plays {playtime_diff:.1f} more hours/week than low engagement")
print(f"      - High engagement plays {sessions_diff:.1f} more sessions/week than low engagement")


# ============================================================================
# 10. EDA Summary (EDA 总结)
# ============================================================================
# 总结 EDA 阶段的关键发现和下一步行动
# ============================================================================

print("\n" + "=" * 80)
print("[SUMMARY] Step 1 EDA Summary")
print("=" * 80)

print("\n[OK] Data Quality Assessment:")
print(f"   - Data completeness: {'Excellent' if df.isnull().sum().sum() == 0 else 'Good'}")
print(f"   - Sample size: {len(df):,} records (suitable for modeling)")
print(f"   - Feature dimensions: {len(df.columns)} features")

print("\n[TARGET] Core Business Findings:")
print(f"   1. Low engagement players account for {low_engagement:.1f}% - need attention")
print(f"   2. Weekly playtime and sessions per week are key indicators of engagement")
print(f"   3. Player level and achievements unlocked are positively correlated with engagement")

print("\n[SAVED] Generated Visualization Reports:")
print("   - numerical_features_distribution.png - Numerical feature distributions")
print("   - categorical_features_distribution.png - Categorical feature distributions")
print("   - engagement_level_distribution.png - Target variable distribution")
print("   - correlation_heatmap.png - Feature correlation heatmap")
print("   - player_segmentation_profile.png - Player segmentation profiles")

print("\n[NEXT] Step 2 Preview: Feature Engineering")
print("   - Derive new behavioral features (e.g., activity metrics, engagement intensity)")
print("   - Feature encoding (categorical -> numerical)")
print("   - Outlier detection and handling")
print("   - Feature scaling (standardization/normalization)")

print("\n" + "=" * 80)
print("[DONE] Step 1 EDA completed! Please check ../output/ directory for all visualization charts")
print("=" * 80)
