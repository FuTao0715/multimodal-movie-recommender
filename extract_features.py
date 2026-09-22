#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
extract_features.py — 多模态特征预提取脚本
===============================================================================
功能说明：
    1. 图像特征提取：使用预训练 ResNet50 提取每张海报的 2048 维特征向量
    2. 文本特征提取：使用 sentence-transformers/all-MiniLM-L6-v2 提取剧情简介的 384 维句向量
    3. 将特征保存为 .pt 文件，供训练脚本直接加载，避免每次训练都重跑 CNN/Transformer

输出文件：
    - features/image_features.pt    : dict {movieId: tensor(2048,)}
    - features/text_features.pt     : dict {movieId: tensor(384,)}
    - features/movie_id_map.pt      : dict {movieId: index}  — movieId 到连续索引的映射

使用方法：
    python extract_features.py                      # 提取所有特征
    python extract_features.py --max_movies 100     # 仅处理前 100 部（测试用）

依赖安装（首次运行前）：
    pip install torch torchvision pandas requests Pillow sentence-transformers
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
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ============================================================================
# [配置区]
# ============================================================================

# 路径配置
MOVIES_CSV = "movies_enriched.csv"          # 补全后的电影数据
POSTER_DIR = "posters"                      # 海报图片目录
FEATURE_DIR = "features"                    # 特征输出目录
IMAGE_FEATURES_FILE = "features/image_features.pt"
TEXT_FEATURES_FILE = "features/text_features.pt"
MOVIE_ID_MAP_FILE = "features/movie_id_map.pt"

# 图像处理配置
IMAGE_SIZE = 224                            # ResNet 标准输入尺寸
BATCH_SIZE_IMAGES = 64                      # 图像特征提取批大小（根据显存调整）

# 文本模型配置
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 轻量级 Sentence Transformer
BATCH_SIZE_TEXTS = 128                      # 文本特征提取批大小

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# [模块 1] 图像特征提取 (Visual Encoder)
# ============================================================================

def build_image_transform():
    """
    构建图像预处理 pipeline。
    ResNet 要求的标准化参数来自 ImageNet 训练集统计值。
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),          # 缩放到 224×224
        transforms.ToTensor(),                                 # 转为张量 [0, 1]
        transforms.Normalize(                                  # ImageNet 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def build_resnet_encoder():
    """
    构建 ResNet50 图像编码器（去掉分类头）。
    返回:
        encoder : nn.Module — 输入 (B, 3, 224, 224) → 输出 (B, 2048)
    """
    print("[INFO] 加载预训练 ResNet50 ...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # 移除最后的全连接分类层 (fc)，保留平均池化后的 2048 维特征
    modules = list(resnet.children())[:-1]       # 去掉 fc 层
    encoder = nn.Sequential(*modules)            # 输出 shape: (B, 2048, 1, 1)
    encoder = encoder.to(DEVICE)
    encoder.eval()                               # 冻结，只做推理
    return encoder


def extract_image_features(movie_df, encoder, transform):
    """
    遍历 posters/ 目录，对每张有效海报提取 2048 维特征向量。
    参数:
        movie_df  : DataFrame，包含 movieId 和 poster_local_path 列
        encoder   : ResNet 编码器
        transform : 图像预处理 pipeline
    返回:
        image_features : dict {movieId: np.ndarray(2048,)}
    """
    print("\n" + "=" * 60)
    print(">>> 阶段 1：图像特征提取 (ResNet50 → 2048-dim)")
    print("=" * 60)

    # 筛选出有海报文件的电影
    valid_records = []
    for _, row in movie_df.iterrows():
        path = row["poster_local_path"]
        if pd.notna(path) and isinstance(path, str) and path.strip():
            full_path = path if os.path.isabs(path) else path
            if os.path.exists(full_path):
                valid_records.append((int(row["movieId"]), full_path))

    print(f"  有效海报数: {len(valid_records)} / {len(movie_df)}")

    if len(valid_records) == 0:
        print("[ERROR] 没有找到任何海报文件，请先运行 enrich_movie_data.py")
        return {}

    # 自定义 Dataset，仅加载图片
    class PosterDataset(Dataset):
        def __init__(self, records, transform):
            self.records = records          # list of (movieId, path)
            self.transform = transform

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            movie_id, path = self.records[idx]
            try:
                img = Image.open(path).convert("RGB")
                tensor = self.transform(img)
                return movie_id, tensor
            except Exception as e:
                print(f"  [警告] 无法加载图片 {path}: {e}")
                # 返回一张空白图作为占位
                blank = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
                return movie_id, blank

    dataset = PosterDataset(valid_records, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_IMAGES,
        shuffle=False,
        num_workers=0,           # Windows 下设为 0 避免多进程问题
        pin_memory=(DEVICE.type == "cuda"),
    )

    image_features = {}
    processed = 0

    with torch.no_grad():
        for movie_ids, images in dataloader:
            images = images.to(DEVICE)                      # (B, 3, 224, 224)
            features = encoder(images)                      # (B, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)     # (B, 2048)
            features = features.cpu().numpy()

            for mid, feat in zip(movie_ids, features):
                image_features[int(mid)] = feat.astype(np.float32)

            processed += len(movie_ids)
            print(f"\r  图像进度: {processed}/{len(valid_records)}", end="", flush=True)

    print()
    print(f"  图像特征提取完成！共 {len(image_features)} 部电影")
    return image_features


# ============================================================================
# [模块 2] 文本特征提取 (Text Encoder)
# ============================================================================

def build_sentence_transformer():
    """
    加载 sentence-transformers/all-MiniLM-L6-v2 模型。
    该模型将英文句子编码为 384 维稠密向量。
    """
    from sentence_transformers import SentenceTransformer
    print("\n[INFO] 加载 Sentence Transformer (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE.type)
    print(f"  模型维度: {model.get_sentence_embedding_dimension()}")
    return model


def extract_text_features(movie_df, model):
    """
    对每部电影的 overview 文本提取 384 维句向量。
    参数:
        movie_df : DataFrame，包含 movieId 和 overview 列
        model    : SentenceTransformer 模型
    返回:
        text_features : dict {movieId: np.ndarray(384,)}
    """
    print("\n" + "=" * 60)
    print(">>> 阶段 2：文本特征提取 (all-MiniLM-L6-v2 → 384-dim)")
    print("=" * 60)

    # 收集所有有效文本
    valid_records = []
    for _, row in movie_df.iterrows():
        overview = row["overview"]
        if pd.notna(overview) and isinstance(overview, str) and overview.strip():
            valid_records.append((int(row["movieId"]), overview.strip()))
        else:
            # 没有剧情简介的电影使用空字符串占位
            valid_records.append((int(row["movieId"]), ""))

    print(f"  有剧情简介: {sum(1 for _, t in valid_records if t)} / {len(valid_records)}")

    texts = [t for _, t in valid_records]
    movie_ids = [mid for mid, _ in valid_records]

    # 批量编码（SentenceTransformer 自带批处理）
    print("  正在编码文本...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE_TEXTS,
        show_progress_bar=True,             # 内置 tqdm 进度条
        convert_to_numpy=True,
        normalize_embeddings=False,         # 不归一化，保留原始幅度
    )  # shape: (N, 384)

    text_features = {}
    for mid, emb in zip(movie_ids, embeddings):
        text_features[mid] = emb.astype(np.float32)

    print(f"  文本特征提取完成！共 {len(text_features)} 部电影")
    return text_features


# ============================================================================
# [模块 3] 数据整合与保存
# ============================================================================

def build_movie_id_map(movie_df):
    """
    建立 movieId → 连续整数索引 (0, 1, 2, ...) 的映射。
    这是后续 Embedding 层需要的。
    """
    movie_ids = sorted(movie_df["movieId"].astype(int).unique())
    return {int(mid): idx for idx, mid in enumerate(movie_ids)}


def save_features(image_features, text_features, movie_id_map):
    """
    将所有特征和映射保存为 .pt 文件。
    """
    print("\n" + "=" * 60)
    print(">>> 阶段 3：保存特征文件")
    print("=" * 60)

    os.makedirs(FEATURE_DIR, exist_ok=True)

    torch.save(image_features, IMAGE_FEATURES_FILE)
    print(f"  图像特征已保存: {IMAGE_FEATURES_FILE}  ({len(image_features)} 条)")

    torch.save(text_features, TEXT_FEATURES_FILE)
    print(f"  文本特征已保存: {TEXT_FEATURES_FILE}  ({len(text_features)} 条)")

    torch.save(movie_id_map, MOVIE_ID_MAP_FILE)
    print(f"  电影ID映射已保存: {MOVIE_ID_MAP_FILE}  ({len(movie_id_map)} 条)")


# ============================================================================
# [主函数]
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="多模态特征预提取")
    parser.add_argument("--max_movies", type=int, default=None,
                        help="仅处理前 N 部电影（测试用）")
    parser.add_argument("--skip_images", action="store_true",
                        help="跳过图像特征提取（仅提取文本）")
    parser.add_argument("--skip_texts", action="store_true",
                        help="跳过文本特征提取（仅提取图像）")
    args = parser.parse_args()

    print("=" * 60)
    print("  多模态特征预提取")
    print(f"  设备: {DEVICE}")
    print(f"  图像编码器: ResNet50 (2048-dim)")
    print(f"  文本编码器: all-MiniLM-L6-v2 (384-dim)")
    print("=" * 60)

    # ---- 读取数据 ----
    movie_df = pd.read_csv(MOVIES_CSV)
    print(f"\n[INFO] 读取 {MOVIES_CSV}: {len(movie_df)} 部电影")

    if args.max_movies is not None:
        movie_df = movie_df.head(args.max_movies)
        print(f"[INFO] 测试模式，仅处理前 {args.max_movies} 部")

    # ---- 图像特征 ----
    if not args.skip_images:
        image_encoder = build_resnet_encoder()
        image_transform = build_image_transform()
        image_features = extract_image_features(movie_df, image_encoder, image_transform)
    else:
        print("\n[SKIP] 跳过图像特征提取")
        image_features = {}

    # ---- 文本特征 ----
    if not args.skip_texts:
        text_encoder = build_sentence_transformer()
        text_features = extract_text_features(movie_df, text_encoder)
    else:
        print("\n[SKIP] 跳过文本特征提取")
        text_features = {}

    # ---- 保存 ----
    movie_id_map = build_movie_id_map(movie_df)
    save_features(image_features, text_features, movie_id_map)

    print("\n" + "=" * 60)
    print("  特征提取全部完成！")
    print(f"  输出目录: {FEATURE_DIR}/")
    print("  下一步: 运行 train_multimodal_ncf.py 进行模型训练")
    print("=" * 60)


if __name__ == "__main__":
    main()
