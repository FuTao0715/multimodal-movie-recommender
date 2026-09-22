# 🎬 多模态电影推荐系统

> 基于 MovieLens + TMDB 的深度学习电影评分预测与可视化仪表板

---

## 📊 项目概览

本项目构建了一个**多模态神经网络协同过滤（NCF）推荐系统**，融合三种信息源预测用户对电影的评分（0.5～5.0 星）：

| 信息源 | 特征 | 维度 |
|--------|------|------|
| 🖼️ 电影海报 | ResNet50 预训练 CNN 提取 | 2048 维 |
| 📝 剧情简介 | MiniLM  Transformer 提取 | 384 维 |
| 👥 协同过滤 | 用户/电影 Embedding | 各 64 维 |

同时包含一个 **Flask Web 仪表板**，可视化对比多个模型的预测效果。

---

## 🚀 快速开始

### 环境要求

```bash
# Python 3.10+，CUDA 可选（CPU 也能跑）
pip install -r requirements.txt
```

### 启动仪表板

```bash
python app.py
# 浏览器打开 http://localhost:5000
```

仪表板功能：
- 📈 训练过程可视化（Loss / RMSE 曲线）
- 👤 用户推荐探索（任选用户 → AI 推荐 Top-12）
- 📊 用户级 80/20 分层评估（真实 vs 预测对比）
- 🎯 全局准确率统计 + 误差分布饼图
- 🔄 **多模型切换对比**（下拉框切换，实时重算）

---

## 🏗️ 项目结构

```
├── app.py                        # Flask 仪表板（主文件，~1200 行）
├── requirements.txt              # pip 依赖
├── TEAM_MODEL_GUIDE.md           # 组员协作指南（AI 提示词模板）
│
├── enrich_movie_data.py          # ① TMDB 数据富化（海报 + 简介）
├── extract_features.py           # ② 特征提取（ResNet50 + MiniLM）
├── train_multimodal_ncf.py       # ③ 多模态 NCF 训练
├── train_textonly.py             # ④ 纯文本 NCF 训练（消融实验）
│
├── movies_enriched.csv           # 富化后的电影元数据
├── training_history.csv          # 多模态训练曲线
│
├── features/                     # 预提取特征文件
│   ├── image_features.pt         # 2048 维，104 MB
│   ├── text_features.pt          # 384 维，23 MB
│   └── movie_id_map.pt           # ID → 索引映射
│
├── models/                       # 训练好的模型权重
│   ├── multimodal_ncf_best.pt    # 多模态（图像+文本+CF）
│   ├── textonly_ncf.pt           # 纯文本（仅文本+CF，消融）
│   ├── randomforest_recommender.pkl  # 随机森林基线
│   ├── keras_cf_ncf.pt           # CF+Genre（组员 Keras→PyTorch 转换）
│   ├── saved_model.keras         # 组员原始 Keras 模型
│   └── saved_encoders.pkl        # Keras 模型的 LabelEncoder + 类型矩阵

├── posters/                      # 9616 张电影海报
└── ml-latest-small/              # MovieLens 原始数据集
│
├── posters/                      # 9616 张电影海报
└── ml-latest-small/              # MovieLens 原始数据集
```

---

## 📐 数据流水线

### 第一步：TMDB 数据富化

```bash
python enrich_movie_data.py
```

- 输入：`ml-latest-small/movies.csv`（9742 部电影标题）
- 调用 TMDB API 搜索电影 → 下载海报 + 英文简介
- 并发 10 线程，速率限制 5 次/秒
- 输出：`movies_enriched.csv` + `posters/` 目录（9616 张）

### 第二步：特征预提取

```bash
python extract_features.py
```

- 🖼️ **图像特征**：ResNet50（去掉分类头）→ 全局平均池化 → **2048 维**
- 📝 **文本特征**：`all-MiniLM-L6-v2` → **384 维**
- 保存为 `.pt` 文件，后续训练直接加载，**无需重复提取**

### 第三步：模型训练

```bash
# 多模态 NCF（图像 + 文本 + 协同过滤）
python train_multimodal_ncf.py

# 纯文本 NCF（消融实验：去掉图像分支）
python train_textonly.py
```

---

## 🧠 模型架构

### 多模态 NCF（1,491,329 参数）

```
User Embedding (64) ──┐
Movie Embedding (64) ─┤
Image Feature (2048) ─┼── Linear(2048→256) ──┐
Text Feature (384) ───┼── Linear(384→256)  ──┤
                      │                      │
                      ▼                      ▼
              Concat(640) ──► MLP(256→128→64→1) ──► Sigmoid×4+1
```

### 纯文本 NCF（901,249 参数）

与多模态相同，但**去掉图像分支**，用于验证海报对预测的贡献。

### 随机森林基线

4 个手工特征：`user_avg_rating`、`movie_avg_rating`、`log1p(user_count)`、`log1p(movie_count)`，100 棵树，最大深度 12。

### CF + 类型特征 NCF（741,633 参数，组员 Keras 训练）

组员用 Keras 3 训练的模型，特征为 **User Embedding (64) + Movie Embedding (64) + 20 维多热电影类型**，是唯一使用了 MovieLens 类型标签的模型。

```
User Embedding (64) ──┐
Movie Embedding (64) ─┼── Concat(148) ──► MLP(256→128→64→1) ──► 线性输出
Genre Vec (20) ────────┘
```

- 输出为线性（无 sigmoid 约束），可能超出 [0.5, 5.0] 范围
- 使用 L2 正则化 + BatchNorm（Keras eps=0.001）+ Dropout(0.3)
- Keras 模型已通过 h5py 权重提取转换为 PyTorch 格式

---

## 📈 模型性能对比

> 评估方式：全部 610 个用户（≥10 条评分），每人随机 hold out 20% 评分，模型预测后对比真实值

| 模型 | 参数 | RMSE ↓ | Acc@0.5 ↑ | Acc@1.0 ↑ | 说明 |
|------|------|--------|-----------|-----------|------|
| 🧬 **CF + 类型特征** | 74 万 | **0.716** | **55.1%** | **85.6%** | Embedding + 20 维电影类型（组员） |
| 🌲 RandomForest | 100 trees | 0.784 | 51.2% | 82.0% | 简单统计特征 |
| 🎬 多模态 NCF  | 149 万 | 0.788 | 51.5% | 82.1% | ResNet50 + MiniLM + CF |
| 📝 纯文本 NCF  | 90 万 | 0.792 | 51.1% | 81.3% | 仅 MiniLM + CF（消融） |

### 关键发现

- 🧬 **电影类型是最强特征**：仅 20 维的多热编码 + Embedding 超越 2048 维图像和 384 维文本特征
- 🖼️ **海报图像贡献极小**：加上 2048 维图像特征，RMSE 仅改善 0.004（几乎为零）
- 📝 **文本信号已经足够**：MiniLM 384 维语义特征已经捕获了评分相关信息
- 🌲 **简单模型不弱**：随机森林凭借用户/电影统计特征达到 RMSE 0.784
- 📏 **评估是公正的**：使用全部 610 个用户（非只挑活跃用户），代表真实表现

### 评估指标说明

| 指标 | 含义 | 通俗解释 |
|------|------|----------|
| **RMSE** | 均方根误差 | 预测分 vs 真实分的平均偏差，越小越好 |
| **Acc@0.5** | 误差 < 0.5 星的比例 | "几乎猜对"的比例 |
| **Acc@1.0** | 误差 < 1.0 星的比例 | "大致猜对"的比例 |

---

## 🔌 添加新模型

### 方法 1：自己训练（推荐）

1. 复制 `TEAM_MODEL_GUIDE.md` 中的提示词发给 AI
2. 训练完成后将 `.pt` 文件放到 `models/` 目录
3. 在 `app.py` 的 `ALL_MODELS` 注册表中添加你的模型

### 方法 2：Keras 模型集成

组员的 Keras 3 `.keras` 模型已通过以下步骤集成：

1. 用 h5py 从 `.keras` zip 包中提取权重
2. 在 PyTorch 中重写等价模型架构（参数复制，注意 BatchNorm eps=0.001）
3. 转换为 `.pt` 文件（`models/keras_cf_ncf.pt`）
4. 同时保存 LabelEncoder 映射和类型矩阵（`models/saved_encoders.pkl`）

转换脚本见项目根目录（如有需要可重新执行）。

### 方法 3：手动集成

你的 checkpoint 必须包含以下字段：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),   # PyTorch 模型权重
    "model_config": {
        "num_users": 610,                     # 用户总数
        "num_movies": 9734,                   # 电影总数
        "model_type": "your_model_name",      # 自定义类型名
        # ... 你的超参数
    },
    "val_rmse": 0.85,                         # 验证集 RMSE
    "val_mae": 0.65,                          # 验证集 MAE
    "epoch": 10,                              # 最佳 epoch
    "training_history": [...],                # 训练历史（可选，用于画曲线）
}
torch.save(checkpoint, "models/your_model.pt")
```

然后在 `app.py` 中注册：

```python
ALL_MODELS = {
    # ... 已有模型 ...
    "your_model": {
        "name": "你的模型名",
        "icon": "🧪",
        "desc": "简短描述",
        "model": your_model_instance,
        "type": "your_type",
    },
}
```

---

## 📊 仪表板截图说明

仪表板包含以下区域：

```
┌─────────────────────────────────────────────┐
│  🎬 多模态推荐系统    [模型选择 ▼]  [标签]  │ ← 顶部栏：切换模型
├─────────────────────────────────────────────┤
│ [用户数] [电影数] [评分数] [RMSE] [平均分]  │ ← 统计卡片（RMSE 随模型切换更新）
├─────────────────────────────────────────────┤
│  🎯 全局准确率                              │
│  52.4% 误差<0.5星    83.2% 误差<1.0星       │ ← 准确率 + 误差分布饼图
│  RMSE 0.7605              [饼图]            │
├─────────────────────────────────────────────┤
│  📉 Loss 曲线      │  📊 RMSE/MAE 曲线     │ ← 训练曲线（非 DL 模型自动隐藏）
├─────────────────────────────────────────────┤
│  👤 [用户选择 ▼] [查看推荐]                 │
│  [🤖AI推荐] [⭐历史评分] [📊效果评估]       │ ← 三标签切换
│                                              │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐               │
│  │海报│ │海报│ │海报│ │海报│    ...         │ ← 推荐/历史 电影卡片
│  │4.5⭐│ │4.3⭐│ │4.1⭐│ │4.0⭐│            │
│  └────┘ └────┘ └────┘ └────┘               │
└─────────────────────────────────────────────┘
```

---

## ⚠️ 注意事项

- **首次启动慢**：`app.py` 启动时需要加载所有 4 个模型到 GPU（约 8~15 秒），正常现象
- **评估计算**：全量 610 用户评估约 2~30 秒（视模型），之后有缓存秒出
- **CF+Genre 模型**：组员用 Keras 3 训练，已转换为 PyTorch，无需额外安装 Keras
- **PyTorch 版本**：如果用的是 PyTorch ≥ 2.6，`torch.load()` 需要加 `weights_only=False`
- **海报缺失**：约 100 部电影无海报（TMDB 未收录），显示占位图标
- **评分不足**：少于 10 条评分的用户无法做 80/20 评估

---

## 📝 评分标准

| RMSE 范围 | 等级 |
|-----------|------|
| < 0.85 | ⭐ 优秀 |
| 0.85 ~ 0.95 | 👍 良好 |
| 0.95 ~ 1.05 | ✔️ 合格 |
| > 1.05 | 🔧 需要改进 |

---

*项目完成日期：2026-06-22*
