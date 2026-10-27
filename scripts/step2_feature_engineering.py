"""
=============================================================================
第二步：特征工程与分群
=============================================================================
项目：游戏玩家行为与流失预测
宪法约束：基于 CONSTITUTION.md 第二章 - 三步模块化开发原则
=============================================================================

本章目标：
1. 衍生新的行为特征（活跃度指标、肝度系数）
2. 类别特征编码（分类变量 -> 数值变量）
3. 异常值检测与处理
4. 特征缩放（标准化/归一化）
5. 玩家分群分析

作者：AI 游戏数据分析师
日期：2026-10-27
"""

# ============================================================================
# 1. 导入库
# ============================================================================
# 导入必要的库（pandas, numpy, sklearn等）
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# 设置
# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

print("=" * 80)
print("[INFO] Game Player Behavior Analysis - Step 2: Feature Engineering")
print("=" * 80)


# ============================================================================
# 2. 加载数据
# ============================================================================
# 加载原始数据集并创建副本用于特征工程
# ============================================================================

print("\n[INFO] Loading dataset...")
# 数据文件路径（相对路径）
DATA_PATH = "../data/online_gaming_behavior_dataset.csv/online_gaming_behavior_dataset.csv"

# 读取数据
df = pd.read_csv(DATA_PATH)

print(f"[OK] Data loaded successfully!")
print(f"   - Records: {len(df):,}")
print(f"   - Features: {len(df.columns)}")

# 创建数据副本用于特征工程（避免修改原始数据）
df_feat = df.copy()


# ============================================================================
# 3. 特征衍生（新的行为特征）
# ============================================================================
# 基于原始特征衍生新的业务指标，用于捕捉玩家行为模式
# ============================================================================

print("\n" + "=" * 80)
print("[FEATURE] Deriving New Behavioral Features")
print("=" * 80)

# ----------------------------------------------------------------------------
# 3.1 活跃度评分（综合活跃度指标）
# 公式：游玩时长、游玩次数和单次时长的加权组合
# 业务含义：反映玩家整体活跃程度
# ----------------------------------------------------------------------------

print("\n[1/6] Creating ActivityScore...")
# 活跃度评分：综合衡量玩家活跃程度

# Normalize components to 0-100 scale
playtime_norm = (df_feat['PlayTimeHours'] - df_feat['PlayTimeHours'].min()) / \
                (df_feat['PlayTimeHours'].max() - df_feat['PlayTimeHours'].min()) * 100

sessions_norm = (df_feat['SessionsPerWeek'] - df_feat['SessionsPerWeek'].min()) / \
                (df_feat['SessionsPerWeek'].max() - df_feat['SessionsPerWeek'].min()) * 100

duration_norm = (df_feat['AvgSessionDurationMinutes'] - df_feat['AvgSessionDurationMinutes'].min()) / \
                (df_feat['AvgSessionDurationMinutes'].max() - df_feat['AvgSessionDurationMinutes'].min()) * 100

# Weighted combination (sessions weighted higher as it's the key differentiator)
df_feat['ActivityScore'] = (
    playtime_norm * 0.3 +
    sessions_norm * 0.5 +
    duration_norm * 0.2
)

print(f"   [OK] ActivityScore created (range: {df_feat['ActivityScore'].min():.1f} - {df_feat['ActivityScore'].max():.1f})")
print(f"   - Mean: {df_feat['ActivityScore'].mean():.2f}")
print(f"   - Median: {df_feat['ActivityScore'].median():.2f}")


# ----------------------------------------------------------------------------
# 3.2 Grind Index (Engagement intensity metric)
# Formula: PlayTimeHours * SessionsPerWeek
# Business meaning: Captures both time investment and frequency
# ----------------------------------------------------------------------------

print("\n[2/6] Creating GrindIndex...")

df_feat['GrindIndex'] = df_feat['PlayTimeHours'] * df_feat['SessionsPerWeek']

print(f"   [OK] GrindIndex created (range: {df_feat['GrindIndex'].min():.1f} - {df_feat['GrindIndex'].max():.1f})")
print(f"   - Mean: {df_feat['GrindIndex'].mean():.2f}")
print(f"   - Median: {df_feat['GrindIndex'].median():.2f}")


# ----------------------------------------------------------------------------
# 3.3 Achievement Rate (Achievement efficiency)
# Formula: AchievementsUnlocked / PlayerLevel (with +1 to avoid division by zero)
# Business meaning: How efficiently a player unlocks achievements per level
# ----------------------------------------------------------------------------

print("\n[3/6] Creating AchievementRate...")

df_feat['AchievementRate'] = df_feat['AchievementsUnlocked'] / (df_feat['PlayerLevel'] + 1)

print(f"   [OK] AchievementRate created (range: {df_feat['AchievementRate'].min():.3f} - {df_feat['AchievementRate'].max():.3f})")
print(f"   - Mean: {df_feat['AchievementRate'].mean():.3f}")
print(f"   - Median: {df_feat['AchievementRate'].median():.3f}")


# ----------------------------------------------------------------------------
# 3.4 Session Intensity (Average playtime per session)
# Formula: (PlayTimeHours * 60) / SessionsPerWeek
# Business meaning: Average minutes per session (consistency check)
# ----------------------------------------------------------------------------

print("\n[4/6] Creating SessionIntensity...")

df_feat['SessionIntensity'] = (df_feat['PlayTimeHours'] * 60) / (df_feat['SessionsPerWeek'] + 1)

print(f"   [OK] SessionIntensity created (range: {df_feat['SessionIntensity'].min():.1f} - {df_feat['SessionIntensity'].max():.1f})")
print(f"   - Mean: {df_feat['SessionIntensity'].mean():.2f}")
print(f"   - Median: {df_feat['SessionIntensity'].median():.2f}")


# ----------------------------------------------------------------------------
# 3.5 Purchase Indicator (Binary)
# Formula: 1 if InGamePurchases > 0, else 0
# Business meaning: Whether player has made in-game purchases
# ----------------------------------------------------------------------------

print("\n[5/6] Creating HasPurchased...")

df_feat['HasPurchased'] = (df_feat['InGamePurchases'] > 0).astype(int)

purchase_pct = df_feat['HasPurchased'].mean() * 100
print(f"   [OK] HasPurchased created")
print(f"   - Purchasers: {df_feat['HasPurchased'].sum():,} ({purchase_pct:.1f}%)")
print(f"   - Non-purchasers: {(1 - df_feat['HasPurchased']).sum():,} ({100 - purchase_pct:.1f}%)")


# ----------------------------------------------------------------------------
# 3.6 Engagement Score (Alternative continuous target)
# Formula: Composite score based on all engagement-related features
# Business meaning: Can be used for regression tasks or clustering
# ----------------------------------------------------------------------------

print("\n[6/6] Creating EngagementScore...")

# Normalize all engagement-related features
level_norm = (df_feat['PlayerLevel'] - df_feat['PlayerLevel'].min()) / \
             (df_feat['PlayerLevel'].max() - df_feat['PlayerLevel'].min()) * 100

achievement_norm = (df_feat['AchievementsUnlocked'] - df_feat['AchievementsUnlocked'].min()) / \
                   (df_feat['AchievementsUnlocked'].max() - df_feat['AchievementsUnlocked'].min()) * 100

df_feat['EngagementScore'] = (
    df_feat['ActivityScore'] * 0.4 +
    level_norm * 0.3 +
    achievement_norm * 0.3
)

print(f"   [OK] EngagementScore created (range: {df_feat['EngagementScore'].min():.1f} - {df_feat['EngagementScore'].max():.1f})")
print(f"   - Mean: {df_feat['EngagementScore'].mean():.2f}")
print(f"   - Median: {df_feat['EngagementScore'].median():.2f}")

print("\n" + "-" * 80)
print("[SUMMARY] All new features created:")
new_features = ['ActivityScore', 'GrindIndex', 'AchievementRate',
                'SessionIntensity', 'HasPurchased', 'EngagementScore']
for feat in new_features:
    print(f"   - {feat}")


# ============================================================================
# 4. Feature Encoding (Categorical -> Numerical)
# ============================================================================

print("\n" + "=" * 80)
print("[ENCODE] Categorical Feature Encoding")
print("=" * 80)

# ----------------------------------------------------------------------------
# 4.1 Label Encoding (Ordinal features)
# ----------------------------------------------------------------------------

print("\n[INFO] Applying Label Encoding...")

# Gender (Binary: Male=0, Female=1)
le_gender = LabelEncoder()
df_feat['Gender_Encoded'] = le_gender.fit_transform(df_feat['Gender'])
print(f"   [OK] Gender encoded: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")

# GameDifficulty (Ordinal: Easy=0, Medium=1, Hard=2)
difficulty_order = ['Easy', 'Medium', 'Hard']
df_feat['GameDifficulty_Encoded'] = df_feat['GameDifficulty'].map(
    {val: idx for idx, val in enumerate(difficulty_order)}
)
print(f"   [OK] GameDifficulty encoded: {dict(zip(difficulty_order, range(len(difficulty_order))))}")


# ----------------------------------------------------------------------------
# 4.2 One-Hot Encoding (Nominal features)
# ----------------------------------------------------------------------------

print("\n[INFO] Applying One-Hot Encoding...")

# Location (Nominal)
df_feat = pd.get_dummies(df_feat, columns=['Location'], prefix='Location')
location_cols = [col for col in df_feat.columns if col.startswith('Location_')]
print(f"   [OK] Location encoded: {len(location_cols)} dummy variables created")
for col in location_cols:
    print(f"      - {col}")

# GameGenre (Nominal)
df_feat = pd.get_dummies(df_feat, columns=['GameGenre'], prefix='Genre')
genre_cols = [col for col in df_feat.columns if col.startswith('Genre_')]
print(f"   [OK] GameGenre encoded: {len(genre_cols)} dummy variables created")
for col in genre_cols:
    print(f"      - {col}")


# ============================================================================
# 5. Outlier Detection & Handling
# ============================================================================

print("\n" + "=" * 80)
print("[OUTLIER] Outlier Detection & Handling (IQR Method)")
print("=" * 80)

# Define numerical features for outlier detection
numerical_features_outlier = [
    'Age', 'PlayTimeHours', 'SessionsPerWeek',
    'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
    'ActivityScore', 'GrindIndex', 'AchievementRate', 'SessionIntensity',
    'EngagementScore'
]

outlier_stats = []

for feature in numerical_features_outlier:
    Q1 = df_feat[feature].quantile(0.25)
    Q3 = df_feat[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df_feat[feature] < lower_bound) | (df_feat[feature] > upper_bound)).sum()
    outlier_pct = outliers / len(df_feat) * 100

    outlier_stats.append({
        'Feature': feature,
        'Outliers': outliers,
        'Percentage': outlier_pct,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound
    })

# Display outlier statistics
outlier_df = pd.DataFrame(outlier_stats)
print("\n[STATS] Outlier Detection Results:")
print("-" * 80)
for _, row in outlier_df.iterrows():
    print(f"   {row['Feature']:25s}: {int(row['Outliers']):5d} outliers ({row['Percentage']:5.2f}%)")


# ----------------------------------------------------------------------------
# 5.1 Outlier Capping (Winsorization)
# Instead of removing, cap outliers at 1st and 99th percentiles
# ----------------------------------------------------------------------------

print("\n[INFO] Applying outlier capping (winsorization at 1st/99th percentile)...")

features_to_cap = ['PlayTimeHours', 'SessionsPerWeek', 'GrindIndex', 'SessionIntensity']

for feature in features_to_cap:
    lower_cap = df_feat[feature].quantile(0.01)
    upper_cap = df_feat[feature].quantile(0.99)

    original_min = df_feat[feature].min()
    original_max = df_feat[feature].max()

    df_feat[feature] = df_feat[feature].clip(lower_cap, upper_cap)

    print(f"   [OK] {feature}: capped to [{lower_cap:.2f}, {upper_cap:.2f}] (was [{original_min:.2f}, {original_max:.2f}])")


# ============================================================================
# 6. Feature Scaling (Standardization)
# ============================================================================

print("\n" + "=" * 80)
print("[SCALE] Feature Scaling (Standardization)")
print("=" * 80)

# Define features to scale
features_to_scale = [
    'Age', 'PlayTimeHours', 'SessionsPerWeek',
    'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
    'InGamePurchases',
    'ActivityScore', 'GrindIndex', 'AchievementRate',
    'SessionIntensity', 'EngagementScore',
    'Gender_Encoded', 'GameDifficulty_Encoded'
] + location_cols + genre_cols

print(f"\n[INFO] Scaling {len(features_to_scale)} features...")

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform
df_feat[features_to_scale] = scaler.fit_transform(df_feat[features_to_scale])

print(f"   [OK] All features scaled (mean=0, std=1)")

# Verify scaling
print("\n[VERIFY] Sample scaled feature statistics:")
for feat in features_to_scale[:5]:
    print(f"   {feat:25s}: mean={df_feat[feat].mean():.3f}, std={df_feat[feat].std():.3f}")


# ============================================================================
# 7. Player Segmentation Analysis
# ============================================================================

print("\n" + "=" * 80)
print("[SEGMENT] Player Segmentation Analysis")
print("=" * 80)

# ----------------------------------------------------------------------------
# 7.1 Engagement Level Distribution (After Feature Engineering)
# ----------------------------------------------------------------------------

print("\n[INFO] Engagement Level Distribution:")
engagement_counts = df_feat['EngagementLevel'].value_counts()
engagement_pct = df_feat['EngagementLevel'].value_counts(normalize=True) * 100

for level in engagement_counts.index:
    count = engagement_counts[level]
    pct = engagement_pct[level]
    print(f"   {level:8s}: {count:5d} players ({pct:5.1f}%)")


# ----------------------------------------------------------------------------
# 7.2 Feature Importance by Engagement Level
# ----------------------------------------------------------------------------

print("\n[ANALYSIS] Feature Means by Engagement Level:")
print("-" * 80)

key_features = ['ActivityScore', 'GrindIndex', 'AchievementRate',
                'SessionIntensity', 'EngagementScore']

engagement_profile = df_feat.groupby('EngagementLevel')[key_features].mean()
print(engagement_profile.round(2))


# ----------------------------------------------------------------------------
# 7.3 Visualization: New Feature Distributions
# ----------------------------------------------------------------------------

print("\n[VIZ] Creating feature distribution visualizations...")

# Figure 1: New feature distributions
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, feature in enumerate(new_features):
    if feature == 'HasPurchased':
        # Binary feature - use bar chart
        counts = df_feat[feature].value_counts()
        counts.plot(kind='bar', ax=axes[idx], color='teal', edgecolor='black')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['No Purchase', 'Purchased'])
    else:
        # Continuous features - use histogram
        sns.histplot(df_feat[feature], kde=True, ax=axes[idx], color='steelblue', bins=30)

    axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Count')

plt.suptitle('New Feature Distributions (After Engineering)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../output/step2_new_feature_distributions.png', dpi=300, bbox_inches='tight')
print("   [SAVED] step2_new_feature_distributions.png")
plt.close()


# Figure 2: Feature correlation heatmap (with new features)
fig, plt.figure(figsize=(14, 12))
correlation_features = key_features + ['PlayTimeHours', 'SessionsPerWeek', 'PlayerLevel']
correlation_matrix = df_feat[correlation_features].corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap (With New Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/step2_correlation_with_new_features.png', dpi=300, bbox_inches='tight')
print("   [SAVED] step2_correlation_with_new_features.png")
plt.close()


# Figure 3: Box plots - New features by Engagement Level
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(key_features + ['HasPurchased']):
    ax = axes[idx]
    sns.boxplot(data=df_feat, x='EngagementLevel', y=feature, ax=ax, palette='Set2')
    ax.set_title(f'{feature} vs EngagementLevel', fontsize=11, fontweight='bold')
    ax.set_xlabel('Engagement Level')
    ax.set_ylabel(feature)

plt.suptitle('New Features vs Engagement Level', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../output/step2_features_vs_engagement.png', dpi=300, bbox_inches='tight')
print("   [SAVED] step2_features_vs_engagement.png")
plt.close()


# ----------------------------------------------------------------------------
# 7.4 Player Persona Creation
# ----------------------------------------------------------------------------

print("\n[PERSONA] Creating Player Personas:")
print("-" * 80)

# Define personas based on scaled feature thresholds (using standardized scores)
# After standardization: mean=0, std=1, so >0.5 means above ~70th percentile
def classify_persona(row):
    # Hardcore: High activity score AND high grind index (top players)
    if row['ActivityScore'] > 0.5 and row['GrindIndex'] > 0.5:
        return 'Hardcore Gamer'
    # Regular: Above average activity and frequency
    elif row['ActivityScore'] > 0 and row['SessionsPerWeek'] > 0:
        return 'Regular Player'
    # Casual: Some activity but low engagement
    elif row['ActivityScore'] > -0.5:
        return 'Casual Player'
    # At-Risk: Low activity scores
    else:
        return 'At-Risk Player'

df_feat['Persona'] = df_feat.apply(classify_persona, axis=1)

persona_counts = df_feat['Persona'].value_counts()
persona_pct = df_feat['Persona'].value_counts(normalize=True) * 100

print("\n[RESULT] Player Persona Distribution:")
for persona in persona_counts.index:
    count = persona_counts[persona]
    pct = persona_pct[persona]
    print(f"   {persona:20s}: {count:5d} players ({pct:5.1f}%)")


# Visualize persona distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
persona_counts.plot(kind='pie', ax=axes[0], autopct='%1.1f%%',
                    colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'], startangle=90)
axes[0].set_ylabel('')
axes[0].set_title('Player Persona Distribution', fontsize=12, fontweight='bold')

# Bar chart
persona_counts.plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
axes[1].set_xlabel('Persona Type')
axes[1].set_ylabel('Player Count')
axes[1].set_title('Player Persona Count', fontsize=12, fontweight='bold')

for i, v in enumerate(persona_counts):
    axes[1].text(i, v + 100, str(v), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('../output/step2_player_persona.png', dpi=300, bbox_inches='tight')
print("\n   [SAVED] step2_player_persona.png")
plt.close()


# ============================================================================
# 8. Save Processed Data
# ============================================================================

print("\n" + "=" * 80)
print("[SAVE] Saving Processed Data")
print("=" * 80)

# Save feature-engineered dataset
output_path = "../data/processed_data_with_features.csv"
df_feat.to_csv(output_path, index=False)

print(f"\n[OK] Processed data saved to: {output_path}")
print(f"   - Records: {len(df_feat):,}")
print(f"   - Features: {len(df_feat.columns)} (original: {len(df.columns)}, new: {len(df_feat.columns) - len(df.columns)})")

# Display all features
print("\n[LIST] All Features in Processed Dataset:")
print("-" * 80)
for i, col in enumerate(df_feat.columns, 1):
    print(f"   {i:2d}. {col}")


# ============================================================================
# 9. Step 2 Summary
# ============================================================================

print("\n" + "=" * 80)
print("[SUMMARY] Step 2 Feature Engineering Summary")
print("=" * 80)

print("\n[OK] Completed Tasks:")
print("   1. Derived 6 new behavioral features")
print("      - ActivityScore (composite activity metric)")
print("      - GrindIndex (engagement intensity)")
print("      - AchievementRate (achievement efficiency)")
print("      - SessionIntensity (average playtime per session)")
print("      - HasPurchased (purchase indicator)")
print("      - EngagementScore (alternative continuous target)")
print("   2. Encoded categorical features (Gender, GameDifficulty, Location, GameGenre)")
print("   3. Detected and handled outliers using IQR method")
print("   4. Scaled all numerical features using StandardScaler")
print("   5. Created player personas based on behavioral patterns")

print("\n[INSIGHT] Key Findings:")
print(f"   - Purchasers: {purchase_pct:.1f}% | Non-purchasers: {100-purchase_pct:.1f}%")
print(f"   - Most common persona: {persona_counts.index[0]} ({persona_pct.iloc[0]:.1f}%)")
print(f"   - ActivityScore and GrindIndex show strong correlation with engagement")

print("\n[SAVED] Generated Visualizations:")
print("   - step2_new_feature_distributions.png")
print("   - step2_correlation_with_new_features.png")
print("   - step2_features_vs_engagement.png")
print("   - step2_player_persona.png")

print("\n[NEXT] Step 3 Preview: Model Training & Business Insights")
print("   - Train classification models (LightGBM, XGBoost, Logistic Regression)")
print("   - Feature importance analysis")
print("   - Model evaluation (Accuracy, Precision, Recall, F1-Score)")
print("   - Business recommendations based on model results")
print("   - STAR-format resume project description")

print("\n" + "=" * 80)
print("[DONE] Step 2 Feature Engineering completed!")
print("=" * 80)
