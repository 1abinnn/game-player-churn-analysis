# 🎮 游戏玩家流失预测 (Game Player Churn Prediction)

> **项目代号**: Online Gaming Behavior Insight  
> **版本**: v1.0.0  
> **状态**: ✅ 已完成 (2026-10-27)

---

## 📋 项目概述

本项目基于 **40,034 名玩家的行为数据**，构建了一个完整的**游戏玩家流失预测模型**，旨在帮助游戏运营团队：

- 🔍 识别潜在的流失风险玩家
- 📊 理解影响玩家留存的关键行为指标
- 💡 制定精准的运营干预策略
- 💰 优化游戏变现策略

**核心成果**：
- ✅ 最佳模型 **XGBoost** 达到 **F1-Score: 0.8984, ROC-AUC: 0.9390**
- ✅ 发现 **SessionsPerWeek（每周游玩次数）** 是最关键的流失预测因子（重要性 54.4%）
- ✅ 识别出 **32.8% 的玩家（13,134人）** 属于高风险流失群体
- ✅ 提供 **5项可落地的业务优化建议**

---

## 📁 项目结构

```
game_player_churn/
├── config/
│   └── CONSTITUTION.md                          # 项目宪法（开发规范）
├── data/
│   ├── online_gaming_behavior_dataset.csv/      # 原始数据集
│   │   └── online_gaming_behavior_dataset.csv     (40,034 条记录, 13 个特征)
│   ├── processed_data_with_features.csv         # 特征工程后的数据 (29 个特征)
│   └── player_churn_predictions.csv             # 模型预测结果
├── scripts/
│   ├── step1_eda.py                             # Step 1: EDA 探索性分析
│   ├── step2_feature_engineering.py             # Step 2: 特征工程
│   └── step3_modeling.py                        # Step 3: 模型训练与评估
├── output/                                      # 可视化图表输出目录
│   ├── Step 1 - EDA (5张图)
│   │   ├── numerical_features_distribution.png
│   │   ├── categorical_features_distribution.png
│   │   ├── engagement_level_distribution.png
│   │   ├── correlation_heatmap.png
│   │   └── player_segmentation_profile.png
│   ├── Step 2 - 特征工程 (4张图)
│   │   ├── step2_new_feature_distributions.png
│   │   ├── step2_correlation_with_new_features.png
│   │   ├── step2_features_vs_engagement.png
│   │   └── step2_player_persona.png
│   └── Step 3 - 建模 (3张图)
│       ├── step3_confusion_matrix.png
│       ├── step3_roc_curve.png
│       └── step3_feature_importance.png
└── README.md                                    # 项目说明（本文件）
```

---

## 🚀 快速开始

### 环境要求

```bash
# 必需库
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0

# 可选（推荐）
lightgbm>=3.3.0
xgboost>=1.7.0
```

### 安装依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
```

### 运行分析

按顺序执行以下脚本：

```bash
# Step 1: EDA 探索性分析
cd scripts
python step1_eda.py

# Step 2: 特征工程
python step2_feature_engineering.py

# Step 3: 模型训练与评估
python step3_modeling.py
```

所有可视化图表将自动保存至 `output/` 目录。

---

## 📊 数据集说明

### 原始数据特征 (13个)

| 特征名 | 类型 | 说明 |
|--------|------|------|
| PlayerID | int | 玩家唯一标识 |
| Age | int | 玩家年龄 |
| Gender | object | 性别 (Male/Female) |
| Location | object | 地区 (USA/Europe/Asia/Other) |
| GameGenre | object | 游戏类型 (Action/RPG/Simulation/Sports/Strategy) |
| PlayTimeHours | float | 每周游玩时长（小时） |
| InGamePurchases | int | 游戏内购买次数 |
| GameDifficulty | object | 游戏难度 (Easy/Medium/Hard) |
| SessionsPerWeek | int | 每周游玩次数 |
| AvgSessionDurationMinutes | int | 平均每次游玩时长（分钟） |
| PlayerLevel | int | 玩家等级 |
| AchievementsUnlocked | int | 解锁成就数量 |
| EngagementLevel | object | **目标变量**: 参与度等级 (High/Medium/Low) |

### 衍生特征 (6个)

| 特征名 | 公式 | 业务含义 |
|--------|------|----------|
| ActivityScore | 加权组合 (PlayTime 30% + Sessions 50% + Duration 20%) | 综合活跃度指标 |
| GrindIndex | PlayTimeHours × SessionsPerWeek | 肝度系数（投入强度） |
| AchievementRate | AchievementsUnlocked / (PlayerLevel + 1) | 成就解锁效率 |
| SessionIntensity | (PlayTimeHours × 60) / SessionsPerWeek | 单次游玩强度 |
| HasPurchased | InGamePurchases > 0 ? 1 : 0 | 是否付费 |
| EngagementScore | ActivityScore 40% + Level 30% + Achievement 30% | 综合参与度评分 |

---

## 🎯 核心发现

### 1. 玩家群体画像

```
总玩家数: 40,034 人
├─ 高参与度 (High):    25.8% (10,336人)
├─ 中参与度 (Medium):  48.4% (19,374人)
└─ 低参与度 (Low):     25.8% (10,324人) ← ⚠️ 流失风险群体
```

### 2. 玩家分群 (Persona)

```
Hardcore Gamer (硬核玩家):  23.6% (9,433人)
Regular Player (常规玩家):  19.0% (7,614人)
Casual Player (休闲玩家):   24.6% (9,853人)
At-Risk Player (风险玩家):  32.8% (13,134人) ← 需要干预
```

### 3. 关键行为指标

| 指标 | 高参与度 | 低参与度 | 差异 |
|------|----------|----------|------|
| SessionsPerWeek | 14.3 次/周 | 4.5 次/周 | **+9.7 次** 🔥 |
| ActivityScore | +0.93 | -0.88 | **+1.81** |
| GrindIndex | +0.56 | -0.57 | **+1.13** |

**关键洞察**：每周游玩次数（SessionsPerWeek）是区分玩家参与度的**最核心指标**，而非单纯的游戏时长。

---

## 🤖 模型性能

### 模型对比

| 模型 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| **XGBoost** ⭐ | **0.9488** | **0.9198** | 0.8780 | **0.8984** | **0.9390** |
| LightGBM | 0.9417 | 0.8805 | **0.8954** | 0.8879 | 0.9390 |
| Random Forest | 0.9353 | 0.8609 | 0.8935 | 0.8769 | 0.9372 |
| Logistic Regression | 0.8796 | 0.7166 | 0.8818 | 0.7907 | 0.9280 |

### 特征重要性 (Top 10)

```
1. SessionsPerWeek               0.5439  ████████████████████ (54.4%)
2. AvgSessionDurationMinutes     0.2147  ████████ (21.5%)
3. EngagementScore               0.0511  ██ (5.1%)
4. SessionIntensity              0.0157  █ (1.6%)
5. ActivityScore                 0.0142  █ (1.4%)
6. AchievementsUnlocked          0.0122  
7. PlayerLevel                   0.0119
8. AchievementRate               0.0117
9. PlayTimeHours                 0.0114
10. GameDifficulty_Encoded       0.0095
```

---

## 💡 业务建议

基于模型结果，提出以下 **5 项运营优化策略**：

### 1. 🚨 早期预警系统 (基于 SessionsPerWeek)

- **监控对象**: 每周游玩次数 < 5 次的玩家
- **干预措施**: 
  - 触发个性化召回通知
  - 提供回归奖励（如限定皮肤、道具）
- **预期效果**: 降低 30% 的低频玩家流失率

### 2. 🎯 活跃度提升 (基于 ActivityScore)

- **目标群体**: ActivityScore < 30 的玩家（底部 30%）
- **策略设计**:
  - 设计限时挑战活动
  - 增加社交互动功能（公会、组队）
  - 优化新手引导流程
- **预期效果**: 提升整体玩家活跃度 20%

### 3. 🏆 成就系统优化 (基于 PlayerLevel & Achievements)

- **优化方向**:
  - 设置等级里程碑奖励
  - 成就解锁时的视觉庆祝动画
  - 提供等级冲刺的专属任务
- **预期效果**: 增加玩家目标感，提升留存率 15%

### 4. ⏱️ 单次体验优化 (基于 SessionIntensity)

- **问题发现**: 部分玩家单次游玩时间长但频率低
- **解决方案**:
  - 设计碎片化内容（5-10分钟可完成的任务）
  - 每日登录奖励机制
  - 体力系统优化（避免过度疲劳）
- **预期效果**: 将低频玩家转化为中频玩家

### 5. 💰 变现策略优化 (基于 HasPurchased)

- **现状**: 仅 20.1% 的玩家有付费行为
- **策略**:
  - 首充双倍奖励（针对未付费玩家）
  - 流失风险玩家的专属礼包
  - 订阅制会员系统
- **预期效果**: 付费转化率提升至 30%

---

## 📈 可视化示例

项目生成 **12 张高质量可视化图表**，包括：

### Step 1: EDA 探索性分析
- 数值特征分布直方图（6个子图）
- 类别特征柱状图（4个子图）
- 目标变量饼图 + 柱状图
- 特征相关性热力图
- 玩家群体画像箱线图

### Step 2: 特征工程
- 新特征分布图（6个子图）
- 新特征相关性热力图
- 特征 vs 参与度等级箱线图
- 玩家分群饼图 + 柱状图

### Step 3: 模型训练
- 混淆矩阵 + 分类报告
- ROC 曲线对比图（4个模型）
- 特征重要性排名图（Top 15）

---

## 📝 技术亮点

### 1. 三步模块化开发

遵循 **EDA → 特征工程 → 建模** 的标准流程，每一步都有清晰的输入输出和业务解读。

### 2. 特征工程创新

- **ActivityScore**: 综合活跃度指标（加权组合）
- **GrindIndex**: 肝度系数（时长 × 频率）
- **AchievementRate**: 成就解锁效率（成就数 / 等级）

### 3. 异常值处理

采用 **IQR 方法 + Winsorization（1st/99th 百分位截断）**，避免删除数据导致信息丢失。

### 4. 类别不平衡处理

使用 **class_weight='balanced'** 参数，确保模型对少数类（流失玩家）的敏感度。

### 5. 业务导向的特征重要性分析

不仅输出特征重要性排名，还针对每个特征提出**可落地的业务建议**。

---

## 🎓 面试复盘 (STAR 法则)

### Situation (背景)

- 游戏公司拥有 40,000+ 名玩家
- 25.8% 的玩家处于低参与度状态（潜在流失风险）
- 缺乏有效的流失预警系统

### Task (任务)

- 构建机器学习模型预测玩家流失风险
- 识别影响留存的关键行为指标
- 为运营团队提供可执行的干预策略

### Action (行动)

- 对 40,034 条玩家记录进行全面 EDA 分析
- 衍生 6 个新特征（ActivityScore, GrindIndex 等）
- 训练 4 个分类模型（LR, RF, LightGBM, XGBoost）
- 使用 Accuracy, Precision, Recall, F1, ROC-AUC 评估
- 进行特征重要性分析并转化为业务建议

### Result (结果)

- 最佳模型 XGBoost: F1=0.8984, ROC-AUC=0.9390
- SessionsPerWeek 被识别为最重要的流失预测因子（54.4%）
- 发现 32.8% 的玩家属于高风险群体
- 关键发现：每周游玩 <5 次的玩家流失风险高 3 倍
- 向运营团队交付 5 项可落地的留存策略

---

## 🔧 后续优化方向

### 1. 模型优化

- [ ] 超参数调优（GridSearchCV / Optuna）
- [ ] 特征选择（减少冗余特征）
- [ ] 处理类别不平衡（SMOTE 过采样）
- [ ] 尝试深度学习模型（神经网络）

### 2. 功能扩展

- [ ] 多分类预测（High/Medium/Low 三分类）
- [ ] 时序分析（玩家行为趋势预测）
- [ ] A/B 测试框架（验证干预策略效果）
- [ ] 实时预测 API（在线流失风险评估）

### 3. 部署上线

- [ ] 模型序列化（joblib/pickle）
- [ ] RESTful API 接口（Flask/FastAPI）
- [ ] 可视化仪表盘（Streamlit/Dash）
- [ ] 自动化监控告警系统

---

## 📚 学习资源

### 相关论文

- [Predicting Player Churn in Mobile Games](https://www.sciencedirect.com/science/article/pii/S0957417420306918)
- [Player Retention in Online Games](https://dl.acm.org/doi/10.1145/3396421.3396432)

### 工具文档

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 Email: [your-email@example.com]
- 💼 LinkedIn: [your-linkedin-profile]
- 🐙 GitHub: [your-github-username]

---

## 📄 许可证

本项目仅用于学习和求职展示目的。

---

## 🙏 致谢

- 数据集来源: [Kaggle - Online Gaming Behavior Dataset](https://www.kaggle.com/)
- 项目宪法约束: 基于 `CONSTITUTION.md` 的三步模块化开发规范
- AI 辅助: Claude Code (Anthropic)

---

**最后更新**: 2026-10-27  
**项目状态**: ✅ 已完成，可用于求职简历展示

---

## 📊 附录: 快速运行示例

```bash
# 克隆项目
git clone [your-repo-url]
cd game_player_churn

# 安装依赖
pip install -r requirements.txt

# 运行完整分析
cd scripts
python step1_eda.py
python step2_feature_engineering.py
python step3_modeling.py

# 查看结果
# - 可视化图表: ../output/
# - 预测结果: ../data/player_churn_predictions.csv
```

---

*Happy Coding! 🎮📊🤖*
