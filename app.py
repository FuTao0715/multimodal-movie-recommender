#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
app.py — 多模态推荐系统可视化仪表板 (Flask Web App)
===============================================================================
功能:
    1. 训练过程可视化（Loss/RMSE 曲线）
    2. 数据统计面板
    3. 用户选择 → 查看历史高分电影 + 海报
    4. 模型推理 → Top-N 推荐电影 + 海报

启动:
    python app.py
    然后浏览器访问 http://localhost:5000
===============================================================================
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")  # 忽略 sklearn 版本警告
from flask import Flask, jsonify, request, send_from_directory, render_template_string

# ============================================================================
# [配置]
# ============================================================================
MOVIES_CSV = "movies_enriched.csv"
RATINGS_CSV = "ml-latest-small/ratings.csv"
MODEL_FILE = "models/multimodal_ncf_best.pt"
IMAGE_FEATURES_FILE = "features/image_features.pt"
TEXT_FEATURES_FILE = "features/text_features.pt"
MOVIE_ID_MAP_FILE = "features/movie_id_map.pt"
HISTORY_CSV = "training_history.csv"
POSTER_DIR = "posters"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_N = 12  # 推荐展示数量

app = Flask(__name__)

# ============================================================================
# [模型定义] — 与 train_multimodal_ncf.py 保持一致
# ============================================================================

class MultimodalNCF(nn.Module):
    def __init__(self, num_users, num_movies,
                 image_feat_dim=2048, text_feat_dim=384,
                 user_embed_dim=64, movie_embed_dim=64,
                 image_proj_dim=256, text_proj_dim=256,
                 mlp_hidden_dims=None, dropout_rate=0.3):
        super(MultimodalNCF, self).__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128, 64]

        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, movie_embed_dim)

        self.image_projection = nn.Sequential(
            nn.Linear(image_feat_dim, image_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, text_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        fusion_dim = user_embed_dim + movie_embed_dim + image_proj_dim + text_proj_dim
        mlp_layers = []
        input_dim = fusion_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_dim = hidden_dim
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, user_idx, movie_idx, image_feat, text_feat):
        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        img_proj = self.image_projection(image_feat)
        txt_proj = self.text_projection(text_feat)
        fused = torch.cat([user_emb, movie_emb, img_proj, txt_proj], dim=1)
        output = self.mlp(fused).squeeze(-1)
        output = torch.sigmoid(output) * 4.0 + 1.0
        return output


class TextOnlyNCF(nn.Module):
    """
    纯文本 NCF（消融实验：去掉图像，仅文本+CF）
    用于对比验证海报图像对评分预测的贡献。
    输入: 64(user) + 64(movie) + 256(text_proj) = 384
    """
    def __init__(self, num_users, num_movies, text_dim=384, embed_dim=64,
                 text_proj_dim=256, mlp_hidden=None, dropout=0.3):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [256, 128, 64]
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, text_proj_dim), nn.ReLU(), nn.Dropout(dropout))
        fusion_dim = embed_dim * 2 + text_proj_dim
        mlp = []
        d_in = fusion_dim
        for h in mlp_hidden:
            mlp += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        mlp.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, user_idx, movie_idx, text_feat):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        t = self.text_projection(text_feat)
        fused = torch.cat([u, m, t], dim=1)
        out = self.mlp(fused).squeeze(-1)
        return torch.sigmoid(out) * 4.0 + 1.0


class KerasCFModel(nn.Module):
    """
    协同过滤 + 类型特征（Keras 模型导出的 PyTorch 版本）
    组员用 Keras 训练，特征为 User/Movie Embedding + 20 维多热电影类型。
    输入: 64(user) + 64(movie) + 20(genre) = 148，线性输出（无 sigmoid 缩放）。
    BatchNorm 使用 Keras 的 eps=0.001。
    """
    def __init__(self, num_users, num_movies, num_genres=20,
                 embed_dim=64, mlp_hidden=None, dropout=0.3):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [256, 128, 64]
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        fusion_dim = embed_dim * 2 + num_genres
        layers = []
        d_in = fusion_dim
        for h in mlp_hidden:
            layers.append(nn.Linear(d_in, h))
            layers.append(nn.BatchNorm1d(h, eps=0.001))  # Keras BN eps
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx, movie_idx, genre_vec):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        x = torch.cat([u, m, genre_vec], dim=1)
        return self.mlp(x).squeeze(-1)


# ============================================================================
# [全局数据加载] — 启动时一次性加载
# ============================================================================
print("=" * 60)
print("  正在启动推荐系统仪表板...")
print(f"  设备: {DEVICE}")
print("=" * 60)

# ---- 电影数据 ----
movies_df = pd.read_csv(MOVIES_CSV)
print(f"[OK] 电影数据: {len(movies_df)} 部")

# ---- 评分数据 ----
ratings_df = pd.read_csv(RATINGS_CSV)
eligible_eval_users = ratings_df.groupby("userId").size()
eligible_eval_users = int((eligible_eval_users >= 10).sum())
print(f"[OK] 评分数据: {len(ratings_df)} 条 (可评估用户: {eligible_eval_users})")

# ---- ID 映射 ----
movie_id_map = torch.load(MOVIE_ID_MAP_FILE, weights_only=False)
num_movies = len(movie_id_map)

# 反向映射: index → movieId
idx_to_movie_id = {idx: mid for mid, idx in movie_id_map.items()}

# user 映射
user_ids = sorted(ratings_df["userId"].unique())
user_id_map = {int(uid): idx for idx, uid in enumerate(user_ids)}
idx_to_user_id = {idx: uid for uid, idx in user_id_map.items()}
num_users = len(user_id_map)

# 每个用户已评分的电影集合 (用于排除)
user_rated_movies = {}
for _, row in ratings_df.iterrows():
    uid = int(row["userId"])
    mid = int(row["movieId"])
    if uid not in user_rated_movies:
        user_rated_movies[uid] = set()
    user_rated_movies[uid].add(mid)

# ---- 特征矩阵 ----
image_features = torch.load(IMAGE_FEATURES_FILE, weights_only=False)
text_features = torch.load(TEXT_FEATURES_FILE, weights_only=False)

image_matrix = np.zeros((num_movies, 2048), dtype=np.float32)
text_matrix = np.zeros((num_movies, 384), dtype=np.float32)
for mid, idx in movie_id_map.items():
    if mid in image_features:
        image_matrix[idx] = image_features[mid]
    if mid in text_features:
        text_matrix[idx] = text_features[mid]

image_tensor = torch.from_numpy(image_matrix).to(DEVICE)
text_tensor = torch.from_numpy(text_matrix).to(DEVICE)
print(f"[OK] 特征矩阵就绪: image={image_tensor.shape}, text={text_tensor.shape}")

# ---- 模型 ----
checkpoint = torch.load(MODEL_FILE, weights_only=False, map_location=DEVICE)
model = MultimodalNCF(
    num_users=checkpoint["num_users"],
    num_movies=checkpoint["num_movies"],
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"[OK] 模型加载完成 (Epoch {checkpoint['epoch']}, Val RMSE={checkpoint['val_rmse']:.4f})")

# ---- 训练历史（多模态从 CSV 加载） ----
history_df = pd.read_csv(HISTORY_CSV)
print(f"[OK] 训练历史: {len(history_df)} 轮")
MODEL_HISTORY = {
    "multimodal": {
        "epochs": history_df["epoch"].tolist(),
        "train_loss": history_df["train_loss"].tolist(),
        "val_loss": history_df["val_loss"].tolist(),
        "val_rmse": history_df["val_rmse"].tolist(),
        "val_mae": history_df["val_mae"].tolist(),
    },
}

# ---- 构建 movieId → 详细信息查找表 ----
movie_info = {}
for _, row in movies_df.iterrows():
    mid = int(row["movieId"])
    poster = row.get("poster_local_path", "")
    if isinstance(poster, str) and poster.strip():
        # 提取文件名
        poster = poster.replace("\\", "/").split("/")[-1] if "/" in poster else poster
    else:
        poster = ""
    movie_info[mid] = {
        "title": row["title"],
        "overview": str(row.get("overview", ""))[:200],
        "poster": poster,
    }
print(f"[OK] 电影详情: {len(movie_info)} 条\n")

# ---- 加载 RandomForest 模型 ----
RF_MODEL_FILE = "models/randomforest_recommender.pkl"

class RandomForestRecommender:
    pass

sys.modules['__main__'].RandomForestRecommender = RandomForestRecommender
with open(RF_MODEL_FILE, "rb") as f:
    rf_model = pickle.load(f)

# 预计算 RF 特征：user_log_cnt, movie_log_cnt
rf_user_cnt = ratings_df.groupby("userId").size()
rf_movie_cnt = ratings_df.groupby("movieId").size()
rf_user_log_cnt = {uid: np.log1p(cnt) for uid, cnt in rf_user_cnt.items()}
rf_movie_log_cnt = {mid: np.log1p(cnt) for mid, cnt in rf_movie_cnt.items()}
# 填充缺失电影为 0
for mid in movie_id_map:
    if mid not in rf_movie_log_cnt:
        rf_movie_log_cnt[mid] = 0.0

print(f"[OK] RandomForest 模型加载完成 "
      f"(n_estimators={rf_model.model.n_estimators}, "
      f"movies={len(rf_model.movie_avg)})")

# ---- 加载 TextOnly 模型（消融实验）----
TEXTONLY_MODEL_FILE = "models/textonly_ncf.pt"
to_ckpt = torch.load(TEXTONLY_MODEL_FILE, weights_only=False, map_location=DEVICE)
textonly_model = TextOnlyNCF(
    num_users=to_ckpt["model_config"]["num_users"],
    num_movies=to_ckpt["model_config"]["num_movies"],
).to(DEVICE)
textonly_model.load_state_dict(to_ckpt["model_state_dict"])
textonly_model.eval()
to_rmse = to_ckpt.get("val_rmse", "?")
print(f"[OK] TextOnly 模型加载完成 (Val RMSE={to_rmse})")

# 补充 TextOnly 训练历史到 MODEL_HISTORY
th = to_ckpt.get("training_history", [])
if th:
    MODEL_HISTORY["textonly"] = {
        "epochs": [e["epoch"] for e in th],
        "train_loss": [e["train_loss"] for e in th],
        "val_loss": [e["val_loss"] for e in th],
        "val_rmse": [float(e["val_rmse"]) for e in th],
        "val_mae": [float(e["val_mae"]) for e in th],
    }
    print(f"[OK] TextOnly 训练历史: {len(th)} 轮")
else:
    MODEL_HISTORY["textonly"] = None
    print(f"[OK] TextOnly: 无训练历史")

# ---- 加载 Keras CF+Genre 模型（组员用 Keras 训练，已转 PyTorch）----
KERAS_CF_MODEL_FILE = "models/keras_cf_ncf.pt"
GENRE_MATRIX_FILE = "features/genre_matrix.pt"
ENCODERS_FILE = "models/saved_encoders.pkl"

keras_cf_ckpt = torch.load(KERAS_CF_MODEL_FILE, weights_only=False, map_location=DEVICE)
keras_cf_model = KerasCFModel(
    num_users=keras_cf_ckpt["model_config"]["num_users"],
    num_movies=keras_cf_ckpt["model_config"]["num_movies"],
).to(DEVICE)
keras_cf_model.load_state_dict(keras_cf_ckpt["model_state_dict"])
keras_cf_model.eval()

# 加载类型矩阵
_genre_data = torch.load(GENRE_MATRIX_FILE, weights_only=False)
genre_matrix_tensor = (torch.from_numpy(_genre_data).float()
                       if isinstance(_genre_data, np.ndarray)
                       else _genre_data.float()).to(DEVICE)

# 加载 LabelEncoder（Keras 模型用 sorted unique IDs → 0..N-1，与我们的映射一致）
with open(ENCODERS_FILE, "rb") as _f:
    _enc = pickle.load(_f)
# 构建 movie encoder 的反向映射（movieId → index），兼容全部 9724 部电影
keras_movie_enc_classes = _enc["movie_enc"].classes_
keras_movie_id_to_idx = {int(mid): i for i, mid in enumerate(keras_movie_enc_classes)}
# user encoder 同样
keras_user_enc_classes = _enc["user_enc"].classes_
keras_user_id_to_idx = {int(uid): i for i, uid in enumerate(keras_user_enc_classes)}
# 预计算：Keras 模型可用的所有电影索引和原始 movieId
_KERAS_ALL_MOVIE_IDS = np.array(list(keras_movie_id_to_idx.keys()), dtype=np.int64)
_KERAS_ALL_MOVIE_INDICES = np.array([keras_movie_id_to_idx[mid] for mid in _KERAS_ALL_MOVIE_IDS], dtype=np.int64)

print(f"[OK] Keras CF+Genre 模型加载完成 "
      f"(params={sum(p.numel() for p in keras_cf_model.parameters()):,})")

# ---- 预计算：所有电影索引列表（供 predict 用，避免每次扫描） ----
_ALL_MOVIE_INDICES = np.array(list(movie_id_map.values()), dtype=np.int64)
_ALL_MOVIE_IDS = np.array(list(movie_id_map.keys()), dtype=np.int64)

# ---- 模型注册表 ----
MODEL_HISTORY["randomforest"] = None  # RF 无训练历史
MODEL_HISTORY["keras_cf"] = None      # Keras CF+Genre 无训练历史

ALL_MODELS = {
    "multimodal": {
        "name": "多模态 NCF",
        "icon": "🎬",
        "desc": "ResNet50 图像 + MiniLM 文本 + 协同过滤",
        "model": model,
        "type": "multimodal",
    },
    "textonly": {
        "name": "纯文本 NCF（消融）",
        "icon": "📝",
        "desc": "仅 MiniLM 文本 + 协同过滤（去掉图像分支）",
        "model": textonly_model,
        "type": "textonly",
    },
    "randomforest": {
        "name": "RandomForest",
        "icon": "🌲",
        "desc": "用户/电影统计特征 + 随机森林回归（sklearn）",
        "model": rf_model,
        "type": "sklearn",
    },
    "keras_cf": {
        "name": "CF + 类型特征",
        "icon": "🧬",
        "desc": "协同过滤 Embedding + 20 维电影类型标签（组员 Keras 训练）",
        "model": keras_cf_model,
        "type": "keras_cf",
    },
}
# ============================================================================
# [用户版] 自动选择RMSE最优算法
# ============================================================================
BEST_MODEL = "multimodal"  # 默认兜底

def init_best_model():
    """启动时对比4个模型的验证RMSE，选出效果最好的用于用户版推荐"""
    global BEST_MODEL
    model_rmse = {}
    
    # 1. 多模态模型
    try:
        model_rmse["multimodal"] = float(checkpoint["val_rmse"])
    except:
        pass
    
    # 2. 纯文本模型
    try:
        model_rmse["textonly"] = float(to_rmse)
    except:
        pass
    
    # 3. 随机森林模型
    try:
        if hasattr(rf_model, 'val_rmse'):
            model_rmse["randomforest"] = float(rf_model.val_rmse)
    except:
        pass
    
    # 4. CF+流派模型
    try:
        if "val_rmse" in keras_cf_ckpt:
            model_rmse["keras_cf"] = float(keras_cf_ckpt["val_rmse"])
    except:
        pass
    
    # 若部分模型取不到验证RMSE，用全局评估自动补全（仅首次执行，结果会缓存）
    if len(model_rmse) < 4:
        for m_name in ALL_MODELS.keys():
            if m_name not in model_rmse:
                try:
                    res = compute_overall_evaluation(model_name=m_name)
                    if "rmse" in res:
                        model_rmse[m_name] = float(res["rmse"])
                except:
                    pass
    
    # 选出RMSE最小的最优模型
    if model_rmse:
        BEST_MODEL = min(model_rmse, key=model_rmse.get)
        print(f"\n[用户版] 最优算法已自动选择：{ALL_MODELS[BEST_MODEL]['name']}，RMSE={model_rmse[BEST_MODEL]:.4f}")

# 服务启动时自动执行一次
init_best_model()

# ============================================================================
# [辅助函数] 模型推理
# ============================================================================

@torch.no_grad()
def predict_for_user(user_id, top_n=TOP_N, model_name="multimodal"):
    """
    对指定用户，用指定模型预测所有未评分电影的评分，返回 Top-N。
    """
    if user_id not in user_id_map:
        return []

    cfg = ALL_MODELS.get(model_name, ALL_MODELS["multimodal"])
    m = cfg["model"]
    mtype = cfg["type"]

    user_idx = user_id_map.get(user_id) if mtype != "keras_cf" else keras_user_id_to_idx.get(user_id)
    if user_idx is None:
        return []
    rated_set = user_rated_movies.get(user_id, set())

    # 不同模型使用不同的候选电影池
    if mtype == "keras_cf":
        all_mids = _KERAS_ALL_MOVIE_IDS
        all_indices = _KERAS_ALL_MOVIE_INDICES
    else:
        all_mids = _ALL_MOVIE_IDS
        all_indices = _ALL_MOVIE_INDICES

    # 找出未评分的电影
    rated_mask = np.isin(all_mids, list(rated_set), invert=True)
    candidate_mids = all_mids[rated_mask]
    candidate_indices = all_indices[rated_mask]

    if len(candidate_mids) == 0:
        return []

    user_idx_tensor = torch.full((len(candidate_indices),), user_idx,
                                 dtype=torch.long, device=DEVICE)
    movie_idx_tensor = torch.from_numpy(candidate_indices).to(DEVICE)

    if mtype == "multimodal":
        preds = m(user_idx_tensor, movie_idx_tensor,
                  image_tensor[movie_idx_tensor], text_tensor[movie_idx_tensor])
        preds = preds.cpu().numpy()
    elif mtype == "textonly":
        preds = m(user_idx_tensor, movie_idx_tensor, text_tensor[movie_idx_tensor])
        preds = preds.cpu().numpy()
    elif mtype == "keras_cf":
        # 候选电影索引已来自 Keras 映射，直接取类型矩阵
        preds = m(user_idx_tensor, movie_idx_tensor,
                  genre_matrix_tensor[movie_idx_tensor]).cpu().numpy()
        # 线性输出可能超出 [0.5, 5.0]，裁剪到合理范围
        preds = np.clip(preds, 0.5, 5.0)
    elif mtype == "sklearn":
        uid = user_id
        ua = m.user_avg.get(uid, 3.5)
        ulc = rf_user_log_cnt.get(uid, 0.0)
        feats, valid_indices = [], []
        for i, mid in enumerate(candidate_mids):
            if mid in m.movie_avg.index:
                feats.append([ua, m.movie_avg[mid],
                              ulc, rf_movie_log_cnt.get(mid, 0.0)])
                valid_indices.append(i)
        if not feats:
            return []
        preds = np.full(len(candidate_mids), 0.0, dtype=np.float32)
        X = m.scaler.transform(feats)
        preds[valid_indices] = m.model.predict(X)
    # 排序取 Top-N
    top_indices = np.argsort(preds)[::-1][:top_n]

    results = []
    for i in top_indices:
        mid = int(candidate_mids[i])
        info = movie_info.get(mid, {})
        results.append({
            "movieId": mid,
            "title": info.get("title", f"Movie {mid}"),
            "poster": info.get("poster", ""),
            "overview": info.get("overview", ""),
            "predicted_rating": round(float(preds[i]), 2),
        })
    return results


def get_user_history(user_id, top_n=20):
    """
    获取用户的高分评分历史。
    返回: list of dict — 按评分降序排列
    """
    if user_id not in user_id_map:
        return []

    user_ratings = ratings_df[ratings_df["userId"] == user_id].copy()
    user_ratings = user_ratings.sort_values("rating", ascending=False).head(top_n)

    results = []
    for _, row in user_ratings.iterrows():
        mid = int(row["movieId"])
        info = movie_info.get(mid, {})
        results.append({
            "movieId": mid,
            "title": info.get("title", f"Movie {mid}"),
            "poster": info.get("poster", ""),
            "overview": info.get("overview", ""),
            "rating": float(row["rating"]),
        })
    return results


@torch.no_grad()
def evaluate_user(user_id, holdout_ratio=0.2, random_seed=42, model_name="multimodal"):
    """
    【用户分层评估】随机 hold out 20% 评分，用指定模型预测，对比真实值。
    """
    if user_id not in user_id_map:
        return None

    cfg = ALL_MODELS.get(model_name, ALL_MODELS["multimodal"])
    m = cfg["model"]
    mtype = cfg["type"]

    user_mask = ratings_df["userId"] == user_id
    user_ratings = ratings_df[user_mask]

    n_total = len(user_ratings)
    if n_total < 10:
        return None

    rng = np.random.RandomState(random_seed)
    n_holdout = max(1, int(n_total * holdout_ratio))
    holdout_indices = rng.choice(n_total, n_holdout, replace=False)

    user_idx = user_id_map[user_id]

    # ---- 第一阶段：收集有效样本 ----
    valid_rows = []  # (movie_idx, actual_rating, mid)
    for i in holdout_indices:
        row = user_ratings.iloc[i]
        mid = int(row["movieId"])
        if mtype == "keras_cf":
            if mid not in keras_movie_id_to_idx:
                continue
            movie_idx = keras_movie_id_to_idx[mid]
        else:
            if mid not in movie_id_map:
                continue
            movie_idx = movie_id_map[mid]
            if mtype == "sklearn" and (user_id not in m.user_avg.index or mid not in m.movie_avg.index):
                continue
        valid_rows.append((movie_idx, float(row["rating"]), mid))

    if not valid_rows:
        return None
    valid_count = len(valid_rows)

    # ---- 第二阶段：批量预测 ----
    if mtype == "sklearn":
        # RF：构建特征矩阵一次预测
        ua = m.user_avg[user_id]
        ulc = rf_user_log_cnt.get(user_id, 0.0)
        feats = [[ua, m.movie_avg[mid], ulc, rf_movie_log_cnt.get(mid, 0.0)]
                 for _, _, mid in valid_rows]
        preds = m.model.predict(m.scaler.transform(feats))
    elif mtype == "keras_cf":
        # Keras CF+Genre：用户/电影 Embedding + 类型向量
        k_user_idx = keras_user_id_to_idx.get(user_id)
        if k_user_idx is None:
            return None
        k_m_indices = []
        for _, _, mid in valid_rows:
            kidx = keras_movie_id_to_idx.get(mid, -1)
            k_m_indices.append(kidx if kidx >= 0 else 0)
        u_tensor = torch.tensor([k_user_idx] * valid_count, dtype=torch.long, device=DEVICE)
        m_tensor = torch.tensor(k_m_indices, dtype=torch.long, device=DEVICE)
        g_tensor = genre_matrix_tensor[m_tensor]
        with torch.no_grad():
            preds = m(u_tensor, m_tensor, g_tensor).cpu().numpy()
    else:
        # PyTorch：构建 batch tensor 一次前向
        m_indices = [r[0] for r in valid_rows]
        u_tensor = torch.tensor([user_idx] * valid_count, dtype=torch.long, device=DEVICE)
        m_tensor = torch.tensor(m_indices, dtype=torch.long, device=DEVICE)
        if mtype == "multimodal":
            preds = m(u_tensor, m_tensor,
                      image_tensor[m_tensor], text_tensor[m_tensor])
        else:  # textonly
            preds = m(u_tensor, m_tensor, text_tensor[m_tensor])
        preds = preds.cpu().numpy()

    # ---- 第三阶段：组装结果 ----
    results = []
    for k, (movie_idx, actual, mid) in enumerate(valid_rows):
        info = movie_info.get(mid, {})
        pred = float(preds[k])
        results.append({
            "movieId": mid,
            "title": info.get("title", f"Movie {mid}"),
            "poster": info.get("poster", ""),
            "overview": info.get("overview", ""),
            "actual_rating": actual,
            "predicted_rating": round(pred, 2),
            "error": round(pred - actual, 2),
        })

    if valid_count == 0:
        return None

    # ---- 计算该用户的评估指标 ----
    errors = [r["error"] for r in results]
    errors_arr = np.array(errors)
    abs_errors = np.abs(errors_arr)
    rmse = float(np.sqrt(np.mean(errors_arr ** 2)))
    mae = float(np.mean(abs_errors))

    # ---- 误差分布（直观的桶统计） ----
    buckets = {
        "0.0-0.5": int(np.sum(abs_errors < 0.5)),       # 几乎完美
        "0.5-1.0": int(np.sum((abs_errors >= 0.5) & (abs_errors < 1.0))),
        "1.0-1.5": int(np.sum((abs_errors >= 1.0) & (abs_errors < 1.5))),
        "1.5-2.0": int(np.sum((abs_errors >= 1.5) & (abs_errors < 2.0))),
        "2.0+":    int(np.sum(abs_errors >= 2.0)),       # 差很多
    }
    # 准确率（学生最能理解的指标）
    accuracy_05 = round(float(np.sum(abs_errors < 0.5)) / valid_count * 100, 1)   # 误差 < 0.5 星
    accuracy_10 = round(float(np.sum(abs_errors < 1.0)) / valid_count * 100, 1)   # 误差 < 1.0 星

    # 按误差绝对值升序排列（预测最准的排前面）
    results.sort(key=lambda x: abs(x["error"]))

    return {
        "userId": user_id,
        "total_ratings": n_total,
        "train_count": n_total - valid_count,
        "heldout_count": valid_count,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "accuracy_05": accuracy_05,         # "0.5星内准确率"
        "accuracy_10": accuracy_10,         # "1星内准确率"
        "error_distribution": buckets,      # 误差分布
        "results": results,
    }


# 全局评估缓存（按模型分存）
_overall_eval_cache = {}


def compute_overall_evaluation(sample_users=None, random_seed=99, model_name="multimodal"):
    """对所有≥10条评分的用户做整体评估，代表真实的全局表现（结果缓存）。"""
    global _overall_eval_cache
    if model_name in _overall_eval_cache:
        return _overall_eval_cache[model_name]

    # 所有有资格用户（≥10条评分才能做80/20评估）
    user_counts = ratings_df.groupby("userId").size()
    eligible = user_counts[user_counts >= 10].index.tolist()

    rng = np.random.RandomState(random_seed)
    if sample_users and sample_users < len(eligible):
        sampled = list(rng.choice(eligible, sample_users, replace=False))
    else:
        sampled = eligible  # 全部用户，最科学

    all_errors = []
    user_metrics = []

    for uid in sampled:
        result = evaluate_user(int(uid), holdout_ratio=0.2, random_seed=uid, model_name=model_name)
        if result is None:
            continue
        for r in result["results"]:
            all_errors.append(r["error"])
        user_metrics.append({
            "userId": int(uid),
            "rmse": result["rmse"],
            "mae": result["mae"],
            "accuracy_05": result["accuracy_05"],
            "accuracy_10": result["accuracy_10"],
        })

    if not all_errors:
        return {"error": "无法计算全局评估"}

    errors_arr = np.array(all_errors)
    abs_errors = np.abs(errors_arr)

    overall = {
        "sampled_users": len(user_metrics),
        "total_predictions": len(all_errors),
        "rmse": round(float(np.sqrt(np.mean(errors_arr ** 2))), 4),
        "mae": round(float(np.mean(abs_errors)), 4),
        "accuracy_05": round(float(np.sum(abs_errors < 0.5)) / len(abs_errors) * 100, 1),
        "accuracy_10": round(float(np.sum(abs_errors < 1.0)) / len(abs_errors) * 100, 1),
        "error_distribution": {
            "0.0-0.5": int(np.sum(abs_errors < 0.5)),
            "0.5-1.0": int(np.sum((abs_errors >= 0.5) & (abs_errors < 1.0))),
            "1.0-1.5": int(np.sum((abs_errors >= 1.0) & (abs_errors < 1.5))),
            "1.5-2.0": int(np.sum((abs_errors >= 1.5) & (abs_errors < 2.0))),
            "2.0+": int(np.sum(abs_errors >= 2.0)),
        },
    }
    _overall_eval_cache[model_name] = overall
    print(f"[OK] 全局评估完成 ({model_name}): RMSE={overall['rmse']}, 准确率@0.5={overall['accuracy_05']}%")
    return overall


# ============================================================================
# [API 路由]
# ============================================================================

@app.route("/")
def index():
    """主页面"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/stats")
def api_stats():
    """返回整体统计信息"""
    # 将 rating_distribution 的 numpy 键值转为 Python 原生类型
    dist = ratings_df["rating"].value_counts().sort_index()
    rating_dist = {float(k): int(v) for k, v in dist.items()}

    return jsonify({
        "num_users": num_users,
        "num_movies": num_movies,
        "num_ratings": int(len(ratings_df)),
        "model_rmse": round(float(checkpoint["val_rmse"]), 4),
        "model_mae": round(float(checkpoint["val_mae"]), 4),
        "model_epoch": int(checkpoint["epoch"]),
        "avg_rating": round(float(ratings_df["rating"].mean()), 2),
        "rating_distribution": rating_dist,
    })


@app.route("/api/training_history")
def api_training_history():
    """返回训练过程数据（用于绘图），支持 ?model= 切换"""
    model_name = request.args.get("model", "multimodal")
    data = MODEL_HISTORY.get(model_name)
    if data is None:
        return jsonify(None)
    return jsonify(data)


@app.route("/api/users")
def api_users():
    """返回用户列表（用于下拉框）"""
    # 按活跃度排序（评分数量多的在前）
    user_activity = ratings_df.groupby("userId").size().sort_values(ascending=False)
    users = []
    for uid in user_activity.head(200).index:  # 只返回前 200 个活跃用户
        users.append({
            "userId": int(uid),
            "rating_count": int(user_activity[uid]),
        })
    return jsonify(users)


@app.route("/api/user/<int:user_id>/history")
def api_user_history(user_id):
    """返回用户的评分历史"""
    history = get_user_history(user_id, top_n=20)
    return jsonify({
        "userId": user_id,
        "total_ratings": len(user_rated_movies.get(user_id, set())),
        "history": history,
    })


@app.route("/api/models")
def api_models():
    """返回可用模型列表"""
    return jsonify({k: {"name": v["name"], "icon": v["icon"], "desc": v["desc"]}
                    for k, v in ALL_MODELS.items()})


@app.route("/api/user/<int:user_id>/recommendations")
def api_user_recommendations(user_id):
    """返回用户的 Top-N 推荐（支持 ?model=）"""
    model_name = request.args.get("model", "multimodal")
    recs = predict_for_user(user_id, top_n=TOP_N, model_name=model_name)
    return jsonify({"userId": user_id, "model": model_name, "recommendations": recs})


@app.route("/api/user/<int:user_id>/evaluate")
def api_user_evaluate(user_id):
    """【用户分层评估】支持 ?model= 选择模型"""
    model_name = request.args.get("model", "multimodal")
    result = evaluate_user(user_id, holdout_ratio=0.2, model_name=model_name)
    if result is None:
        return jsonify({"error": True, "message": "评分不足（需≥10条）"}), 400
    return jsonify(result)


@app.route("/api/overall_eval")
def api_overall_eval():
    """全局评估指标（支持 ?model=）"""
    model_name = request.args.get("model", "multimodal")
    result = compute_overall_evaluation(model_name=model_name)  # 全部≥10条的用户
    return jsonify(result)


@app.route("/posters/<path:filename>")
def serve_poster(filename):
    """提供海报图片（带浏览器缓存，加快二次加载）"""
    from flask import make_response
    resp = make_response(send_from_directory(POSTER_DIR, filename))
    resp.cache_control.max_age = 86400  # 缓存 24 小时
    return resp
# ===================== 用户版专属接口 =====================


@app.route("/user")
def user_index():
    """用户版页面入口，无任何技术指标"""
    return render_template_string(USER_HTML_TEMPLATE)

@app.route("/api/user/<int:user_id>/best_recommend")
def api_user_best_recommend(user_id):
    """用户版推荐：自动使用最优算法，仅返回电影基础信息"""
    recs = predict_for_user(user_id, top_n=12, model_name=BEST_MODEL)
    clean_recs = []
    for item in recs:
        clean_recs.append({
            "movieId": item["movieId"],
            "title": item["title"],
            "poster": item["poster"],
            "overview": item.get("overview", "")
        })
    return jsonify({
        "userId": user_id,
        "recommendations": clean_recs
    })

def serve_poster(filename):
    """提供海报图片（带浏览器缓存，加快二次加载）"""
    from flask import make_response
    resp = make_response(send_from_directory(POSTER_DIR, filename))
    resp.cache_control.max_age = 86400  # 缓存 24 小时
    return resp



# ============================================================================
# [HTML 模板]
# ============================================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多模态推荐系统 — 可视化仪表板</title>
    <script src="/static/js/chart.umd.min.js"></script>
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --text: #e2e8f0;
            --text2: #94a3b8;
            --accent: #38bdf8;
            --accent2: #818cf8;
            --gold: #fbbf24;
            --green: #34d399;
            --red: #f87171;
        }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
        .header {
            background:linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
            border-bottom:1px solid var(--border); padding:20px 40px;
            display:flex; align-items:center; justify-content:space-between;
        }
        .header h1 { font-size:1.6rem; background:linear-gradient(90deg,var(--accent),var(--accent2)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .header .badge { background:var(--card); border:1px solid var(--border); border-radius:20px; padding:6px 16px; font-size:0.8rem; color:var(--text2); }
        .container { max-width:1400px; margin:0 auto; padding:24px; }
        .stats-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin-bottom:24px; }
        .stat-card {
            background:var(--card); border:1px solid var(--border); border-radius:14px;
            padding:18px 22px; text-align:center;
        }
        .stat-card .num { font-size:2rem; font-weight:700; color:var(--accent); }
        .stat-card .label { font-size:0.85rem; color:var(--text2); margin-top:4px; }
        .panel {
            background:var(--card); border:1px solid var(--border); border-radius:14px;
            padding:22px; margin-bottom:24px;
        }
        .panel h2 { font-size:1.1rem; margin-bottom:16px; color:var(--text); border-left:3px solid var(--accent); padding-left:12px; }
        .charts-row { display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:24px; }
        @media (max-width:900px) { .charts-row { grid-template-columns:1fr; } }
        canvas { width:100% !important; max-height:320px; }
        .selector-area { display:flex; align-items:center; gap:16px; flex-wrap:wrap; margin-bottom:20px; }
        .selector-area select {
            background:var(--bg); color:var(--text); border:1px solid var(--border);
            border-radius:10px; padding:10px 16px; font-size:0.95rem; min-width:260px; cursor:pointer;
        }
        .selector-area button {
            background:var(--accent); color:#0f172a; border:none; border-radius:10px;
            padding:10px 24px; font-size:0.95rem; font-weight:600; cursor:pointer;
            transition:all 0.2s;
        }
        .selector-area button:hover { opacity:0.85; transform:translateY(-1px); }
        .selector-area .info-tag { font-size:0.85rem; color:var(--text2); }
        .movie-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:16px; }
        .movie-card {
            background:var(--bg); border:1px solid var(--border); border-radius:12px;
            overflow:hidden; transition:all 0.2s; cursor:pointer;
        }
        .movie-card:hover { transform:translateY(-4px); box-shadow:0 8px 25px rgba(0,0,0,0.4); border-color:var(--accent); }
        .movie-card .img-wrap {
            width:100%; height:280px; background:#1a1a2e;
            display:flex; align-items:center; justify-content:center; overflow:hidden;
        }
        .movie-card .img-wrap img { width:100%; height:100%; object-fit:cover; }
        .movie-card .img-wrap .no-img { color:var(--text2); font-size:0.8rem; text-align:center; padding:10px; }
        .movie-card .info { padding:12px; }
        .movie-card .info .mtitle { font-size:0.85rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .movie-card .info .mrating {
            font-size:0.9rem; font-weight:700; margin-top:4px;
        }
        .movie-card .info .mrating .star { color:var(--gold); }
        .rating-bar { display:inline-block; height:6px; border-radius:3px; background:var(--border); width:80px; vertical-align:middle; margin-left:6px; }
        .rating-bar .fill { height:100%; border-radius:3px; background:var(--gold); }
        .tabs { display:flex; gap:0; margin-bottom:20px; border-bottom:2px solid var(--border); }
        .tab-btn {
            background:none; border:none; color:var(--text2); padding:10px 24px;
            font-size:0.95rem; cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-2px;
            transition:all 0.2s;
        }
        .tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
        .empty-state { text-align:center; padding:40px; color:var(--text2); }
        .footer { text-align:center; padding:20px; color:var(--text2); font-size:0.8rem; }
        .overview-text { font-size:0.78rem; color:var(--text2); margin-top:6px; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
        .pred-badge { display:inline-block; background:var(--accent); color:#0f172a; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:6px; margin-top:4px; }
        .eval-summary { display:flex; gap:24px; margin-bottom:20px; flex-wrap:wrap; }
        .eval-metric { background:var(--bg); border:1px solid var(--border); border-radius:12px; padding:14px 24px; text-align:center; min-width:120px; }
        .eval-metric .big { font-size:1.6rem; font-weight:700; }
        .eval-metric .big.good { color:var(--green); }
        .eval-metric .big.ok { color:var(--gold); }
        .eval-metric .big.bad { color:var(--red); }
        .eval-metric .sub { font-size:0.78rem; color:var(--text2); margin-top:2px; }
        .eval-card { background:var(--bg); border:1px solid var(--border); border-radius:12px; overflow:hidden; display:flex; height:180px; margin-bottom:12px; transition:all 0.2s; }
        .eval-card:hover { border-color:var(--accent); }
        .eval-card .poster-col { width:120px; min-width:120px; background:#1a1a2e; display:flex; align-items:center; justify-content:center; overflow:hidden; }
        .eval-card .poster-col img { width:100%; height:100%; object-fit:cover; }
        .eval-card .poster-col .no-img { color:var(--text2); font-size:0.7rem; text-align:center; }
        .eval-card .content-col { flex:1; padding:14px 18px; display:flex; flex-direction:column; justify-content:center; min-width:0; }
        .eval-card .content-col .etitle { font-weight:600; font-size:0.95rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .eval-card .content-col .eoverview { font-size:0.75rem; color:var(--text2); margin-top:4px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
        .eval-card .rating-col { display:flex; gap:20px; align-items:center; padding:14px 24px; min-width:260px; }
        .eval-card .rating-col .r-block { text-align:center; }
        .eval-card .rating-col .r-block .r-val { font-size:1.3rem; font-weight:700; }
        .eval-card .rating-col .r-block .r-label { font-size:0.7rem; color:var(--text2); }
        .eval-card .rating-col .r-arrow { font-size:1.2rem; color:var(--text2); }
        .eval-card .error-col { min-width:90px; text-align:center; padding:14px 18px; display:flex; flex-direction:column; justify-content:center; }
        .eval-card .error-col .err-val { font-size:1.1rem; font-weight:700; }
        .eval-card .error-col .err-label { font-size:0.7rem; color:var(--text2); }
        @media (max-width:768px) { .eval-card { flex-wrap:wrap; height:auto; } .eval-card .poster-col { width:100%; height:200px; } }
        .accuracy-row { display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:24px; }
        @media (max-width:900px) { .accuracy-row { grid-template-columns:1fr; } }
        .accuracy-big { font-size:2.8rem; font-weight:800; line-height:1; }
        .accuracy-big.great { color:var(--green); }
        .accuracy-big.good { color:var(--gold); }
        .pie-wrap { display:flex; align-items:center; gap:20px; }
        .pie-wrap canvas { width:180px !important; height:180px !important; max-height:180px; }
        .pie-legend { font-size:0.82rem; }
        .pie-legend .item { display:flex; align-items:center; gap:8px; margin:4px 0; }
        .pie-legend .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
    </style>
</head>
<body>

<div class="header">
    <h1>🎬 多模态电影推荐系统 · 可视化仪表板</h1>
    <div style="display:flex;align-items:center;gap:12px;">
        <select id="modelSelect" onchange="onModelChange()" style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:8px 14px;font-size:0.85rem;cursor:pointer;">
            <option value="multimodal">🎬 多模态 NCF (图像+文本)</option>
            <option value="textonly">📝 纯文本 NCF (仅文本，消融)</option>
            <option value="randomforest">🌲 RandomForest (统计特征)</option>
            <option value="keras_cf">🧬 CF + 类型特征 (Keras)</option>
        </select>
        <span class="badge" id="modelBadge">多模态 NCF</span>
    </div>
</div>

<div class="container">

    <!-- 统计面板 -->
    <div class="stats-grid" id="statsGrid">
        <div class="stat-card"><div class="num" id="statUsers">-</div><div class="label">用户数</div></div>
        <div class="stat-card"><div class="num" id="statMovies">-</div><div class="label">电影数</div></div>
        <div class="stat-card"><div class="num" id="statRatings">-</div><div class="label">评分记录</div></div>
        <div class="stat-card"><div class="num" id="statRMSE">-</div><div class="label">模型 RMSE</div></div>
        <div class="stat-card"><div class="num" id="statAvg">-</div><div class="label">平均评分</div></div>
    </div>

    <!-- 全局预测准确率 -->
    <div class="panel" id="overallAccuracyPanel">
        <h2>🎯 全局预测准确率（全部 {n_users} 个用户 80/20 分层评估）</h2>
        <div style="display:flex;align-items:center;gap:30px;flex-wrap:wrap;justify-content:center;" id="overallAccuracyContent">
            <div style="text-align:center;"><div class="accuracy-big great" id="acc05">-</div><div style="color:var(--text2);">误差 &lt; 0.5 星</div></div>
            <div style="text-align:center;"><div class="accuracy-big good" id="acc10">-</div><div style="color:var(--text2);">误差 &lt; 1.0 星</div></div>
            <div style="text-align:center;"><div style="font-size:1.4rem;font-weight:700;color:var(--accent);" id="accRmse">-</div><div style="color:var(--text2);">RMSE</div></div>
            <div class="pie-wrap">
                <canvas id="overallPieChart"></canvas>
                <div class="pie-legend" id="overallPieLegend"></div>
            </div>
        </div>
    </div>

    <!-- 训练曲线 -->
    <div class="charts-row" id="chartsRow">
        <div class="panel">
            <h2>📉 训练 & 验证 Loss</h2>
            <canvas id="lossChart"></canvas>
        </div>
        <div class="panel">
            <h2>📊 验证 RMSE / MAE</h2>
            <canvas id="metricChart"></canvas>
        </div>
    </div>

    <!-- 交互区 -->
    <div class="panel">
        <h2>👤 用户推荐探索</h2>
        <div class="selector-area">
            <select id="userSelect">
                <option value="">-- 选择一个用户 --</option>
            </select>
            <button onclick="loadUserData()">🔍 查看推荐</button>
            <span class="info-tag" id="userInfo"></span>
        </div>

        <!-- 标签切换 -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('recommendations')">🤖 AI 推荐 (预测评分)</button>
            <button class="tab-btn" onclick="switchTab('history')">⭐ 历史高分评分</button>
            <button class="tab-btn" onclick="switchTab('evaluation')">📊 效果评估 (80/20)</button>
        </div>

        <div id="evalSummary" style="display:none;"></div>
        <div class="movie-grid" id="movieGrid">
            <div class="empty-state">👆 请先选择一个用户，然后点击"查看推荐"</div>
        </div>
        <div id="evalGrid" style="display:none;"></div>
    </div>

</div>

<div class="footer">Multimodal NCF · PyTorch + Flask · {year}</div>

<script>
// ========== 全局状态 ==========
let currentTab = 'recommendations';
let currentModel = 'multimodal';
let currentRecommendations = [];
let currentHistory = [];
let currentEvaluation = null;
let lossChartInst = null;
let metricChartInst = null;

// ========== 初始化 ==========
async function init() {
    await loadStats();
    await loadOverallEval();
    await loadTrainingHistory();
    await loadUsers();
}
init();

// ========== 模型切换 ==========
function onModelChange() {
    currentModel = document.getElementById('modelSelect').value;
    const names = {multimodal:'多模态 NCF', textonly:'纯文本 NCF', randomforest:'RandomForest', keras_cf:'CF+Genre'};
    document.getElementById('modelBadge').textContent = names[currentModel] || currentModel;
    loadOverallEval();
    loadTrainingHistory();  // 切换模型时重绘曲线
    const userId = document.getElementById('userSelect').value;
    if (userId) loadUserData();
}
// ========== 加载全局评估 ==========
let overallEvalData = null;
async function loadOverallEval() {
    try {
        const resp = await fetch('/api/overall_eval?model=' + currentModel);
        overallEvalData = await resp.json();
    } catch(e) { return; }
    if (!overallEvalData || overallEvalData.error) return;

    // 准确率数字
    document.getElementById('acc05').textContent = overallEvalData.accuracy_05 + '%';
    document.getElementById('acc10').textContent = overallEvalData.accuracy_10 + '%';
    document.getElementById('accRmse').textContent = 'RMSE ' + overallEvalData.rmse;

    // 更新顶部统计卡片的模型 RMSE（跟随模型切换）
    document.getElementById('statRMSE').textContent = overallEvalData.rmse;

    // 饼图
    renderPieChart('overallPieChart', overallEvalData.error_distribution, overallEvalData.total_predictions, 'overallPieLegend');
}

// ========== 通用饼图渲染 ==========
function renderPieChart(canvasId, distribution, total, legendId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    // 销毁旧图表
    if (canvas._chart) canvas._chart.destroy();

    const labels = ['0~0.5星','0.5~1.0星','1.0~1.5星','1.5~2.0星','2.0星+'];
    const keys = ['0.0-0.5','0.5-1.0','1.0-1.5','1.5-2.0','2.0+'];
    const colors = ['#34d399','#fbbf24','#fb923c','#f87171','#ef4444'];
    const data = keys.map(k => distribution[k] || 0);

    canvas._chart = new Chart(canvas, {
        type:'doughnut',
        data:{
            labels:labels,
            datasets:[{data:data,backgroundColor:colors,borderColor:'#1e293b',borderWidth:3}]
        },
        options:{
            responsive:true, maintainAspectRatio:true,
            plugins:{
                legend:{display:false},
                tooltip:{callbacks:{label:ctx=>` ${ctx.label}: ${ctx.raw}条 (${(ctx.raw/total*100).toFixed(1)}%)`}}
            },
        }
    });

    // 图例
    const legend = document.getElementById(legendId);
    if (legend) {
        legend.innerHTML = keys.map((k,i)=>`
            <div class="item"><span class="dot" style="background:${colors[i]}"></span>${labels[i]}: <b>${distribution[k]||0}</b> (${((distribution[k]||0)/total*100).toFixed(1)}%)</div>
        `).join('');
    }
}

// ========== 加载统计数据 ==========
async function loadStats() {
    const resp = await fetch('/api/stats');
    const data = await resp.json();
    document.getElementById('statUsers').textContent = data.num_users.toLocaleString();
    document.getElementById('statMovies').textContent = data.num_movies.toLocaleString();
    document.getElementById('statRatings').textContent = (data.num_ratings/1000).toFixed(0)+'k';
    document.getElementById('statRMSE').textContent = data.model_rmse;
    document.getElementById('statAvg').textContent = '⭐'+data.avg_rating;
}

// ========== 加载训练曲线 ==========
async function loadTrainingHistory() {
    const chartsRow = document.getElementById('chartsRow');

    // 销毁旧图表
    if (lossChartInst) { lossChartInst.destroy(); lossChartInst = null; }
    if (metricChartInst) { metricChartInst.destroy(); metricChartInst = null; }

    const resp = await fetch('/api/training_history?model=' + currentModel);
    const data = await resp.json();

    // 无训练历史 → 隐藏曲线区域
    if (!data) {
        chartsRow.style.display = 'none';
        return;
    }

    // 有训练历史 → 显示并绘制
    chartsRow.style.display = '';

    // Loss 图
    lossChartInst = new Chart(document.getElementById('lossChart'), {
        type:'line',
        data:{
            labels:data.epochs.map(e=>`Epoch ${e}`),
            datasets:[
                {label:'训练 Loss',data:data.train_loss,borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,0.1)',fill:true,tension:0.3,pointRadius:2},
                {label:'验证 Loss',data:data.val_loss,borderColor:'#f87171',backgroundColor:'rgba(248,113,113,0.1)',fill:true,tension:0.3,pointRadius:2},
            ]
        },
        options:{
            responsive:true,
            plugins:{legend:{labels:{color:'#94a3b8'}}},
            scales:{
                x:{ticks:{color:'#94a3b8'},grid:{color:'#1e293b'}},
                y:{ticks:{color:'#94a3b8'},grid:{color:'#1e293b'}},
            }
        }
    });

    // RMSE/MAE 图
    metricChartInst = new Chart(document.getElementById('metricChart'), {
        type:'line',
        data:{
            labels:data.epochs.map(e=>`Epoch ${e}`),
            datasets:[
                {label:'RMSE',data:data.val_rmse,borderColor:'#fbbf24',tension:0.3,pointRadius:4,pointBackgroundColor:'#fbbf24'},
                {label:'MAE',data:data.val_mae,borderColor:'#34d399',tension:0.3,pointRadius:4,pointBackgroundColor:'#34d399'},
            ]
        },
        options:{
            responsive:true,
            plugins:{legend:{labels:{color:'#94a3b8'}}},
            scales:{
                x:{ticks:{color:'#94a3b8'},grid:{color:'#1e293b'}},
                y:{ticks:{color:'#94a3b8'},grid:{color:'#1e293b'}},
            }
        }
    });
}

// ========== 加载用户列表 ==========
async function loadUsers() {
    const resp = await fetch('/api/users');
    const users = await resp.json();
    const sel = document.getElementById('userSelect');
    users.forEach(u => {
        const opt = document.createElement('option');
        opt.value = u.userId;
        opt.textContent = `User ${u.userId} (${u.rating_count} ratings)`;
        sel.appendChild(opt);
    });
}

// ========== 切换标签 ==========
function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    event.target.classList.add('active');

    // 切换显示模式
    const movieGrid = document.getElementById('movieGrid');
    const evalGrid = document.getElementById('evalGrid');
    const evalSummary = document.getElementById('evalSummary');

    if (tab === 'evaluation') {
        movieGrid.style.display = 'none';
        evalGrid.style.display = 'block';
        evalSummary.style.display = 'flex';
        renderEvaluation();
    } else {
        movieGrid.style.display = 'grid';
        evalGrid.style.display = 'none';
        evalSummary.style.display = 'none';
        renderMovies();
    }
}

// ========== 加载用户数据 ==========
async function loadUserData() {
    const userId = document.getElementById('userSelect').value;
    if (!userId) return;

    document.getElementById('userInfo').textContent = '加载中...';

    // 并行请求（推荐 + 历史 + 评估，传入 model）
    const [recResp, histResp, evalResp] = await Promise.all([
        fetch(`/api/user/${userId}/recommendations?model=${currentModel}`),
        fetch(`/api/user/${userId}/history`),
        fetch(`/api/user/${userId}/evaluate?model=${currentModel}`),
    ]);
    const recData = await recResp.json();
    const histData = await histResp.json();

    currentRecommendations = recData.recommendations || [];
    currentHistory = histData.history || [];

    // 评估数据（可能因评分不足而失败）
    if (evalResp.ok) {
        currentEvaluation = await evalResp.json();
    } else {
        currentEvaluation = null;
    }

    document.getElementById('userInfo').textContent =
        `该用户已评分 ${histData.total_ratings} 部电影`;

    if (currentTab === 'evaluation') {
        renderEvaluation();
    } else {
        renderMovies();
    }
}

// ========== 渲染电影卡片 ==========
function renderMovies() {
    const grid = document.getElementById('movieGrid');
    const movies = currentTab === 'recommendations' ? currentRecommendations : currentHistory;

    if (!movies.length) {
        grid.innerHTML = '<div class="empty-state">没有数据</div>';
        return;
    }

    grid.innerHTML = movies.map(m => {
        const isRec = currentTab === 'recommendations';
        const rating = isRec ? m.predicted_rating : m.rating;
        const ratingColor = rating >= 4 ? 'var(--green)' : rating >= 3 ? 'var(--gold)' : 'var(--red)';
        const ratingLabel = isRec ? '预测' : '评分';
        const posterHtml = m.poster
            ? `<img src="/posters/${m.poster}" alt="${m.title}" onerror="this.parentElement.innerHTML='<div class=no-img>🎬<br>无海报</div>'">`
            : '<div class="no-img">🎬<br>无海报</div>';

        return `
        <div class="movie-card" title="${m.overview || ''}">
            <div class="img-wrap">${posterHtml}</div>
            <div class="info">
                <div class="mtitle">${m.title}</div>
                <div class="mrating">
                    <span class="star">⭐</span>
                    <span style="color:${ratingColor}">${rating.toFixed(1)}</span>
                    <span style="font-size:0.7rem;color:var(--text2)">${ratingLabel}</span>
                </div>
                ${isRec ? `<span class="pred-badge">🤖 AI 推测</span>` : ''}
                ${m.overview ? `<div class="overview-text">${m.overview}</div>` : ''}
            </div>
        </div>`;
    }).join('');
}

// ========== 渲染评估结果 ==========
function renderEvaluation() {
    const evalSummary = document.getElementById('evalSummary');
    const evalGrid = document.getElementById('evalGrid');

    if (!currentEvaluation) {
        evalSummary.innerHTML = '<div class="empty-state" style="width:100%">⚠️ 该用户评分不足（< 10条），无法进行 80/20 评估。请选择更活跃的用户。</div>';
        evalSummary.style.display = 'flex';
        evalGrid.innerHTML = '';
        return;
    }

    const e = currentEvaluation;

    // 评估指标摘要 + 饼图
    const rmseClass = e.rmse < 0.8 ? 'good' : e.rmse < 1.0 ? 'ok' : 'bad';
    const acc05Class = e.accuracy_05 >= 70 ? 'great' : e.accuracy_05 >= 50 ? 'good' : 'bad';
    evalSummary.innerHTML = `
        <div class="eval-metric"><div class="big ok">${e.total_ratings}</div><div class="sub">总评分</div></div>
        <div class="eval-metric"><div class="big">${e.train_count}</div><div class="sub">训练(80%)</div></div>
        <div class="eval-metric"><div class="big">${e.heldout_count}</div><div class="sub">测试(20%)</div></div>
        <div class="eval-metric"><div class="accuracy-big ${acc05Class}">${e.accuracy_05}%</div><div class="sub">误差 &lt; 0.5 星</div></div>
        <div class="eval-metric"><div class="big ${rmseClass}">${e.rmse}</div><div class="sub">RMSE</div></div>
        <div class="pie-wrap" style="margin-left:auto;">
            <canvas id="userPieChart"></canvas>
            <div class="pie-legend" id="userPieLegend"></div>
        </div>
    `;
    evalSummary.style.display = 'flex';

    // 用户饼图
    setTimeout(() => {
        renderPieChart('userPieChart', e.error_distribution, e.heldout_count, 'userPieLegend');
    }, 100);

    // 评估卡片列表（预测 vs 真实）
    evalGrid.innerHTML = e.results.map((r, i) => {
        const errAbs = Math.abs(r.error);
        const errSign = r.error >= 0 ? '+' : '';
        const errColor = errAbs < 0.5 ? 'var(--green)' : errAbs < 1.0 ? 'var(--gold)' : 'var(--red)';
        const emoji = errAbs < 0.3 ? '✅' : errAbs < 0.7 ? '🟡' : '❌';
        const posterHtml = r.poster
            ? `<img src="/posters/${r.poster}" alt="${r.title}" onerror="this.parentElement.innerHTML='<div class=no-img>🎬<br>N/A</div>'">`
            : '<div class="no-img">🎬<br>无海报</div>';

        return `
        <div class="eval-card">
            <div class="poster-col">${posterHtml}</div>
            <div class="content-col">
                <div class="etitle">#${i+1} ${r.title}</div>
                ${r.overview ? `<div class="eoverview">${r.overview}</div>` : ''}
            </div>
            <div class="rating-col">
                <div class="r-block">
                    <div class="r-val" style="color:var(--gold)">⭐${r.actual_rating.toFixed(1)}</div>
                    <div class="r-label">真实评分</div>
                </div>
                <div class="r-arrow">→</div>
                <div class="r-block">
                    <div class="r-val" style="color:var(--accent)">🤖${r.predicted_rating.toFixed(1)}</div>
                    <div class="r-label">模型预测</div>
                </div>
            </div>
            <div class="error-col">
                <div class="err-val" style="color:${errColor}">${emoji} ${errSign}${r.error.toFixed(2)}</div>
                <div class="err-label">误差</div>
            </div>
        </div>`;
    }).join('');
}
</script>
</body>
</html>
""".replace("{year}", "2026").replace("{n_users}", str(eligible_eval_users))
# ============================================================================
# [用户版 HTML 模板] 简洁无技术内容
# ============================================================================
USER_HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电影推荐系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: "微软雅黑", "PingFang SC", sans-serif; }
        body { background: #f5f7fa; color: #1e293b; }
        
        /* 顶部导航 */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px 40px;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 22px; font-weight: 600; }
        .user-select {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            min-width: 180px;
            cursor: pointer;
            background: rgba(255,255,255,0.95);
            color: #1e293b;
        }

        /* 主体布局 */
        .container {
            max-width: 1300px;
            margin: 30px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 24px;
        }

        /* 卡片通用样式 */
        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .card-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 1px solid #f1f5f9;
            color: #334155;
        }

        /* 历史评分列表 */
        .history-list {
            max-height: 650px;
            overflow-y: auto;
            padding-right: 6px;
        }
        .history-item {
            display: flex;
            gap: 12px;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 8px;
            transition: background 0.2s;
        }
        .history-item:hover { background: #f8fafc; }
        .history-poster {
            width: 46px;
            height: 68px;
            object-fit: cover;
            border-radius: 4px;
            background: #e2e8f0;
            flex-shrink: 0;
        }
        .history-info { flex: 1; min-width: 0; }
        .history-title {
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .star-score {
            color: #f59e0b;
            font-size: 13px;
            font-weight: 600;
        }

        /* 猜你喜欢网格 */
        .recommend-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
        }
        .movie-card {
            border-radius: 10px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.25s;
            background: #f8fafc;
        }
        .movie-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }
        .movie-poster {
            width: 100%;
            aspect-ratio: 2 / 3;
            object-fit: cover;
            background: #e2e8f0;
        }
        .movie-name {
            padding: 10px 12px;
            font-size: 13px;
            line-height: 1.4;
            height: 52px;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        /* 空状态 */
        .empty {
            text-align: center;
            color: #94a3b8;
            padding: 60px 0;
            font-size: 14px;
        }

        /* 滚动条美化 */
        .history-list::-webkit-scrollbar { width: 6px; }
        .history-list::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 电影推荐系统</h1>
        <select class="user-select" id="userSelect">
            <option value="">请选择您的用户账号</option>
        </select>
    </div>

    <div class="container">
        <div class="card">
            <div class="card-title">⭐ 我的历史评分</div>
            <div class="history-list" id="historyList">
                <div class="empty">请先选择用户账号</div>
            </div>
        </div>

        <div class="card">
            <div class="card-title">💡 猜你喜欢</div>
            <div class="recommend-grid" id="recommendGrid">
                <div class="empty" style="grid-column: 1 / -1;">请先选择用户账号，为您生成专属推荐</div>
            </div>
        </div>
    </div>

    <script>
        // 加载用户列表
        fetch('/api/users')
            .then(res => res.json())
            .then(users => {
                const sel = document.getElementById('userSelect');
                users.forEach(u => {
                    const opt = document.createElement('option');
                    opt.value = u.userId;
                    opt.textContent = `用户 ${u.userId}`;
                    sel.appendChild(opt);
                });
            });

        // 用户切换时加载数据
        document.getElementById('userSelect').addEventListener('change', function() {
            const userId = this.value;
            if (!userId) return;

            // 加载历史评分
            fetch(`/api/user/${userId}/history`)
                .then(res => res.json())
                .then(data => {
                    const box = document.getElementById('historyList');
                    const list = data.history || [];
                    if (list.length === 0) {
                        box.innerHTML = '<div class="empty">暂无评分记录</div>';
                        return;
                    }
                    box.innerHTML = list.map(item => `
                        <div class="history-item">
                            <img class="history-poster" src="${item.poster ? '/posters/'+item.poster : ''}" onerror="this.style.background='#e2e8f0';this.src=''">
                            <div class="history-info">
                                <div class="history-title">${item.title}</div>
                                <div class="star-score">⭐ ${item.rating.toFixed(1)} 分</div>
                            </div>
                        </div>
                    `).join('');
                });

            // 加载最优算法推荐
            fetch(`/api/user/${userId}/best_recommend`)
                .then(res => res.json())
                .then(data => {
                    const box = document.getElementById('recommendGrid');
                    const list = data.recommendations || [];
                    if (list.length === 0) {
                        box.innerHTML = '<div class="empty" style="grid-column: 1 / -1;">暂无推荐内容</div>';
                        return;
                    }
                    box.innerHTML = list.map(item => `
                        <div class="movie-card" title="${item.overview || ''}">
                            <img class="movie-poster" src="${item.poster ? '/posters/'+item.poster : ''}" onerror="this.style.background='#e2e8f0';this.src=''">
                            <div class="movie-name">${item.title}</div>
                        </div>
                    `).join('');
                });
        });
    </script>
</body>
</html>
"""

# ============================================================================
# [启动入口]
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  仪表板已启动！")
    print("  请打开浏览器访问: http://localhost:5000")
    print("  请打开浏览器访问: http://localhost:5000/user")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
