#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
train_multimodal_ncf.py — 多模态神经协同过滤模型训练脚本
===============================================================================
模型架构：
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ User Embed   │  │ Movie Embed  │  │ Image Proj   │  │ Text Proj    │
    │ (64-dim)     │  │ (64-dim)     │  │ 2048→256     │  │ 384→256      │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │                 │
           └─────────────────┴─────────────────┴─────────────────┘
                                     │
                               Concatenate (640-dim)
                                     │
                              MLP: 640→256→128→64→1
                                     │
                              Predicted Rating (1~5)

输入数据：
    - ml-latest-small/ratings.csv       用户评分数据
    - features/image_features.pt        预提取的 ResNet50 图像特征 (2048-dim)
    - features/text_features.pt         预提取的 MiniLM 文本特征 (384-dim)
    - features/movie_id_map.pt          movieId → 连续索引映射

输出：
    - models/multimodal_ncf_best.pt     最佳模型权重
    - training_history.csv              训练历史记录

使用方法：
    python train_multimodal_ncf.py                      # 默认参数训练
    python train_multimodal_ncf.py --epochs 50 --lr 0.001  # 自定义参数

依赖安装：
    pip install torch pandas numpy scikit-learn
    （必须先运行 extract_features.py 生成特征文件）
===============================================================================
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================================
# [配置区] — 所有超参数集中管理
# ============================================================================

# ---- 路径配置 ----
RATINGS_CSV = "ml-latest-small/ratings.csv"
IMAGE_FEATURES_FILE = "features/image_features.pt"
TEXT_FEATURES_FILE = "features/text_features.pt"
MOVIE_ID_MAP_FILE = "features/movie_id_map.pt"
MODEL_DIR = "models"
BEST_MODEL_FILE = "models/multimodal_ncf_best.pt"
HISTORY_FILE = "training_history.csv"

# ---- 模型超参数 ----
USER_EMBEDDING_DIM = 64        # 用户 Embedding 维度
MOVIE_EMBEDDING_DIM = 64       # 电影 Embedding 维度
IMAGE_PROJ_DIM = 256           # 图像特征投影维度（2048 → 256）
TEXT_PROJ_DIM = 256            # 文本特征投影维度（384 → 256）
MLP_HIDDEN_DIMS = [256, 128, 64]  # MLP 各隐藏层维度
DROPOUT_RATE = 0.3             # Dropout 比率

# ---- 训练超参数 ----
BATCH_SIZE = 512               # 批大小
LEARNING_RATE = 0.001          # 学习率
WEIGHT_DECAY = 1e-5            # L2 正则化系数
NUM_EPOCHS = 30                # 训练轮数
PATIENCE = 5                   # 早停耐心值（验证损失不降超过 N 轮则停止）
VAL_RATIO = 0.2                # 验证集比例

# ---- 设备配置 ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# [模块 1] 数据预处理与特征矩阵构建
# ============================================================================

def load_and_prepare_data():
    """
    加载评分数据与预提取特征，构建训练/验证所需的所有数据结构。
    返回:
        ratings       : np.array, shape (N, 3) — [user_idx, movie_idx, rating]
        image_matrix  : np.array, shape (num_movies, 2048) — 按电影索引排列的图像特征
        text_matrix   : np.array, shape (num_movies, 384)  — 按电影索引排列的文本特征
        num_users     : int — 用户总数
        num_movies    : int — 电影总数
    """
    print("\n" + "=" * 60)
    print(">>> 步骤 1：加载数据与特征")
    print("=" * 60)

    # ---- 1.1 加载 movie_id_map ----
    if not os.path.exists(MOVIE_ID_MAP_FILE):
        print(f"[ERROR] 未找到 {MOVIE_ID_MAP_FILE}，请先运行 extract_features.py")
        sys.exit(1)

    movie_id_map = torch.load(MOVIE_ID_MAP_FILE, weights_only=False)       # {movieId: index}
    num_movies = len(movie_id_map)
    print(f"  电影总数: {num_movies}")

    # ---- 1.2 加载预提取的特征，转为矩阵 ----
    image_features = torch.load(IMAGE_FEATURES_FILE, weights_only=False)   # {movieId: np.array(2048,)}
    text_features = torch.load(TEXT_FEATURES_FILE, weights_only=False)     # {movieId: np.array(384,)}

    img_dim = 2048
    txt_dim = 384

    # 初始化为零矩阵（没有特征的电影用零向量填充）
    image_matrix = np.zeros((num_movies, img_dim), dtype=np.float32)
    text_matrix = np.zeros((num_movies, txt_dim), dtype=np.float32)

    img_count, txt_count = 0, 0
    for movie_id, idx in movie_id_map.items():
        if movie_id in image_features:
            image_matrix[idx] = image_features[movie_id]
            img_count += 1
        if movie_id in text_features:
            text_matrix[idx] = text_features[movie_id]
            txt_count += 1

    print(f"  图像特征: {img_count}/{num_movies} 部 ({img_count/num_movies*100:.1f}%)")
    print(f"  文本特征: {txt_count}/{num_movies} 部 ({txt_count/num_movies*100:.1f}%)")

    # ---- 1.3 加载评分数据 ----
    ratings_df = pd.read_csv(RATINGS_CSV)
    print(f"  评分记录: {len(ratings_df)} 条")

    # 建立 userId → 连续索引的映射
    user_ids = sorted(ratings_df["userId"].unique())
    user_id_map = {int(uid): idx for idx, uid in enumerate(user_ids)}
    num_users = len(user_id_map)

    # 只保留 movie 在 movie_id_map 中的评分（排除没有特征映射的电影）
    valid_movie_ids = set(movie_id_map.keys())
    ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movie_ids)]

    # 映射到连续索引
    user_indices = ratings_df["userId"].map(user_id_map).values.astype(np.int64)
    movie_indices = ratings_df["movieId"].map(movie_id_map).values.astype(np.int64)
    ratings = ratings_df["rating"].values.astype(np.float32)

    # 过滤掉映射失败的行
    valid_mask = ~(
        pd.isna(user_indices) | pd.isna(movie_indices)
    )
    user_indices = user_indices[valid_mask]
    movie_indices = movie_indices[valid_mask]
    ratings = ratings[valid_mask]

    print(f"  有效评分记录: {len(ratings)} 条")
    print(f"  用户总数: {num_users}")

    # 合并为 (user_idx, movie_idx, rating) 数组
    ratings_array = np.stack([user_indices, movie_indices, ratings], axis=1)

    return ratings_array, image_matrix, text_matrix, num_users, num_movies


# ============================================================================
# [模块 2] 数据集类
# ============================================================================

class MultimodalRatingDataset(Dataset):
    """
    多模态评分数据集。
    每次 __getitem__ 返回：
        user_idx   : int       — 用户索引
        movie_idx  : int       — 电影索引
        image_feat : Tensor    — 预提取的图像特征 (2048,)
        text_feat  : Tensor    — 预提取的文本特征 (384,)
        rating     : float     — 真实评分
    """
    def __init__(self, ratings_array, image_matrix, text_matrix):
        """
        参数:
            ratings_array : np.array (N, 3) — [user_idx, movie_idx, rating]
            image_matrix  : np.array (M, 2048)
            text_matrix   : np.array (M, 384)
        """
        self.user_indices = ratings_array[:, 0].astype(np.int64)
        self.movie_indices = ratings_array[:, 1].astype(np.int64)
        self.ratings = ratings_array[:, 2].astype(np.float32)
        self.image_matrix = torch.from_numpy(image_matrix)    # 转为 Tensor，无需每次转换
        self.text_matrix = torch.from_numpy(text_matrix)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_idx = self.user_indices[idx]
        movie_idx = self.movie_indices[idx]

        # 从预计算的特征矩阵中直接查表（O(1)，极快）
        image_feat = self.image_matrix[movie_idx]             # (2048,)
        text_feat = self.text_matrix[movie_idx]               # (384,)
        rating = self.ratings[idx]

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(movie_idx, dtype=torch.long),
            image_feat,
            text_feat,
            torch.tensor(rating, dtype=torch.float32),
        )


# ============================================================================
# [模块 3] 多模态神经协同过滤模型
# ============================================================================

class MultimodalNCF(nn.Module):
    """
    多模态神经协同过滤模型 (Multimodal Neural Collaborative Filtering)

    架构说明：
    ┌─────────────────────────────────────────────────────────────┐
    │  输入层                                                      │
    │  ├─ User Embedding    (user_idx) → (64,)                    │
    │  ├─ Movie Embedding   (movie_idx) → (64,)                   │
    │  ├─ Image Projection  (2048,) → FC+ReLU+Drop → (256,)      │
    │  └─ Text Projection   (384,) → FC+ReLU+Drop → (256,)       │
    │                                                             │
    │  融合层: Concat → (640,)                                    │
    │                                                             │
    │  MLP 预测层:                                                │
    │    640 → 256 → BN+ReLU+Drop                                 │
    │    256 → 128 → BN+ReLU+Drop                                 │
    │    128 → 64  → BN+ReLU+Drop                                 │
    │    64  → 1   → Sigmoid*4+1 → 预测评分 (1~5)                │
    └─────────────────────────────────────────────────────────────┘
    """
    def __init__(self, num_users, num_movies,
                 image_feat_dim=2048, text_feat_dim=384,
                 user_embed_dim=64, movie_embed_dim=64,
                 image_proj_dim=256, text_proj_dim=256,
                 mlp_hidden_dims=None, dropout_rate=0.3):
        super(MultimodalNCF, self).__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128, 64]

        # ---- 协同过滤 Embedding 层 ----
        # 让模型学习用户和电影的协同交互模式
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, movie_embed_dim)

        # ---- 图像特征投影层 ----
        # 将高维图像特征 (2048) 压缩到模型可处理的中等维度
        self.image_projection = nn.Sequential(
            nn.Linear(image_feat_dim, image_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ---- 文本特征投影层 ----
        # 将文本句向量 (384) 投影到统一语义空间
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, text_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ---- MLP 预测层 ----
        # 融合四个模态信息，逐层抽象，最终输出预测评分
        fusion_dim = user_embed_dim + movie_embed_dim + image_proj_dim + text_proj_dim
        # 即: 64 + 64 + 256 + 256 = 640

        mlp_layers = []
        input_dim = fusion_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),          # 加速收敛，稳定训练
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_dim = hidden_dim

        # 输出层：最后一层隐藏维度 → 1（预测评分）
        mlp_layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*mlp_layers)

        # ---- 初始化权重 ----
        self._init_weights()

    def _init_weights(self):
        """使用 Xavier 初始化提升训练稳定性。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, user_idx, movie_idx, image_feat, text_feat):
        """
        前向传播。
        参数:
            user_idx   : (B,)   — 用户索引
            movie_idx  : (B,)   — 电影索引
            image_feat : (B, 2048) — 预提取的图像特征
            text_feat  : (B, 384)  — 预提取的文本特征
        返回:
            pred_rating: (B,)   — 预测评分
        """
        # 1. 协同过滤分支
        user_emb = self.user_embedding(user_idx)          # (B, 64)
        movie_emb = self.movie_embedding(movie_idx)       # (B, 64)

        # 2. 图像语义分支
        img_proj = self.image_projection(image_feat)      # (B, 2048) → (B, 256)

        # 3. 文本语义分支
        txt_proj = self.text_projection(text_feat)        # (B, 384) → (B, 256)

        # 4. 多模态融合
        fused = torch.cat([user_emb, movie_emb, img_proj, txt_proj], dim=1)  # (B, 640)

        # 5. MLP 预测
        output = self.mlp(fused).squeeze(-1)              # (B,)

        # 6. 将输出压缩到合理范围 [1.0, 5.0]
        # Sigmoid → [0, 1]，乘以 4 再加 1 → [1, 5]
        output = torch.sigmoid(output) * 4.0 + 1.0

        return output


# ============================================================================
# [模块 4] 训练与验证循环
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion):
    """执行一个训练 epoch。返回平均损失。"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for user_idx, movie_idx, img_feat, txt_feat, rating in dataloader:
        # 数据移动到设备
        user_idx = user_idx.to(DEVICE)
        movie_idx = movie_idx.to(DEVICE)
        img_feat = img_feat.to(DEVICE)
        txt_feat = txt_feat.to(DEVICE)
        rating = rating.to(DEVICE)

        # 前向传播
        pred = model(user_idx, movie_idx, img_feat, txt_feat)

        # 计算损失
        loss = criterion(pred, rating)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion):
    """在验证集上评估模型。返回 (平均损失, RMSE, MAE)。"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    for user_idx, movie_idx, img_feat, txt_feat, rating in dataloader:
        user_idx = user_idx.to(DEVICE)
        movie_idx = movie_idx.to(DEVICE)
        img_feat = img_feat.to(DEVICE)
        txt_feat = txt_feat.to(DEVICE)
        rating = rating.to(DEVICE)

        pred = model(user_idx, movie_idx, img_feat, txt_feat)

        loss = criterion(pred, rating)
        total_loss += loss.item()
        num_batches += 1

        all_preds.append(pred.cpu().numpy())
        all_targets.append(rating.cpu().numpy())

    avg_loss = total_loss / num_batches

    # 计算 RMSE 和 MAE
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))

    return avg_loss, rmse, mae


# ============================================================================
# [模块 5] 主训练流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="多模态 NCF 模型训练")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--no_early_stop", action="store_true",
                        help="禁用早停，强制跑满所有 epoch")
    args = parser.parse_args()

    print("=" * 60)
    print("  多模态神经协同过滤 (Multimodal NCF) 训练")
    print(f"  设备: {DEVICE}")
    print(f"  架构: UserEmb + MovieEmb + ResNet50 + MiniLM → MLP")
    print("=" * 60)

    # ---- 5.1 加载数据 ----
    ratings_array, image_matrix, text_matrix, num_users, num_movies = load_and_prepare_data()

    # ---- 5.2 划分训练集/验证集 ----
    train_arr, val_arr = train_test_split(
        ratings_array, test_size=VAL_RATIO, random_state=42
    )
    print(f"\n  训练集: {len(train_arr)} 条 | 验证集: {len(val_arr)} 条")

    train_dataset = MultimodalRatingDataset(train_arr, image_matrix, text_matrix)
    val_dataset = MultimodalRatingDataset(val_arr, image_matrix, text_matrix)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,              # Windows 安全设置
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ---- 5.3 构建模型 ----
    print("\n" + "=" * 60)
    print(">>> 步骤 2：构建模型")
    print("=" * 60)

    model = MultimodalNCF(
        num_users=num_users,
        num_movies=num_movies,
        image_feat_dim=2048,
        text_feat_dim=384,
        user_embed_dim=USER_EMBEDDING_DIM,
        movie_embed_dim=MOVIE_EMBEDDING_DIM,
        image_proj_dim=IMAGE_PROJ_DIM,
        text_proj_dim=TEXT_PROJ_DIM,
        mlp_hidden_dims=MLP_HIDDEN_DIMS,
        dropout_rate=args.dropout,
    ).to(DEVICE)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型结构:\n{model}")

    # ---- 5.4 损失函数 & 优化器 ----
    criterion = nn.MSELoss()                                # 回归任务用 MSE
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ---- 5.5 训练循环 ----
    print("\n" + "=" * 60)
    print(f">>> 步骤 3：开始训练 (共 {args.epochs} 轮)")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    history = []                # 记录每轮指标
    best_val_rmse = float("inf")
    best_epoch = 0
    no_improve_count = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

        # 验证
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion)

        # 学习率调度
        scheduler.step(val_loss)

        # 记录
        epoch_time = time.time() - epoch_start
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # 打印
        best_marker = ""
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            no_improve_count = 0
            best_marker = " ★ BEST"

            # 保存最佳模型
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "num_users": num_users,
                "num_movies": num_movies,
                "args": vars(args),
            }
            torch.save(checkpoint, BEST_MODEL_FILE)
        else:
            no_improve_count += 1

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"RMSE: {val_rmse:.4f} | "
              f"MAE: {val_mae:.4f} | "
              f"Time: {epoch_time:.1f}s{best_marker}")

        # 早停检查
        if not args.no_early_stop and no_improve_count >= args.patience:
            print(f"\n  [早停] 验证 RMSE 连续 {args.patience} 轮未改善，停止训练。")
            break

    # ---- 5.6 训练报告 ----
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(">>> 训练完成！汇总报告")
    print("=" * 60)
    print(f"  总训练轮数:     {len(history)}")
    print(f"  最佳验证 RMSE:   {best_val_rmse:.4f}  (Epoch {best_epoch})")
    print(f"  总训练时间:      {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  最佳模型保存至:  {BEST_MODEL_FILE}")

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_FILE, index=False)
    print(f"  训练历史保存至:  {HISTORY_FILE}")

    print("\n" + "=" * 60)
    print("  下一步建议：")
    print("  1. 查看 training_history.csv 分析收敛曲线")
    print("  2. 调整 MLP_HIDDEN_DIMS / DROPOUT_RATE / LEARNING_RATE 重新训练")
    print("  3. 试用 --dropout 0.1 或 --lr 0.0005 等不同参数")
    print("=" * 60)


if __name__ == "__main__":
    main()
