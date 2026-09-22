#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
train_textonly.py — 纯文本 NCF（消融实验：去掉图像，只看文本+CF）
===============================================================================
目的：对比多模态 NCF（图像+文本）vs 纯文本 NCF（仅文本），
      验证海报图像对评分预测的贡献。
===============================================================================
"""
import os, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# === 配置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOVIE_ID_MAP_FILE = "features/movie_id_map.pt"
TEXT_FEATURES_FILE = "features/text_features.pt"
RATINGS_CSV = "ml-latest-small/ratings.csv"

BATCH_SIZE = 512
LR = 0.001
EPOCHS = 50
PATIENCE = 7
VAL_RATIO = 0.2
EMBED_DIM = 64
TEXT_PROJ_DIM = 256
MLP_HIDDEN = [256, 128, 64]
DROPOUT = 0.3

# === 模型：只用文本，不用图像 ===
class TextOnlyNCF(nn.Module):
    def __init__(self, num_users, num_movies, text_dim=384):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, EMBED_DIM)
        self.movie_embedding = nn.Embedding(num_movies, EMBED_DIM)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, TEXT_PROJ_DIM), nn.ReLU(), nn.Dropout(DROPOUT),
        )
        fusion_dim = EMBED_DIM * 2 + TEXT_PROJ_DIM  # 64+64+256=384
        mlp = []
        d_in = fusion_dim
        for h in MLP_HIDDEN:
            mlp += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
            d_in = h
        mlp.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*mlp)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, user_idx, movie_idx, text_feat):
        u = self.user_embedding(user_idx)
        m = self.movie_embedding(movie_idx)
        t = self.text_projection(text_feat)
        fused = torch.cat([u, m, t], dim=1)
        out = self.mlp(fused).squeeze(-1)
        return torch.sigmoid(out) * 4.0 + 1.0


# === 数据集 ===
class TextOnlyDataset(Dataset):
    def __init__(self, ratings_array, text_matrix):
        self.u = ratings_array[:, 0].astype(np.int64)
        self.m = ratings_array[:, 1].astype(np.int64)
        self.r = ratings_array[:, 2].astype(np.float32)
        self.txt = torch.from_numpy(text_matrix)

    def __len__(self): return len(self.r)

    def __getitem__(self, i):
        return (torch.tensor(self.u[i], dtype=torch.long),
                torch.tensor(self.m[i], dtype=torch.long),
                self.txt[self.m[i]],
                torch.tensor(self.r[i], dtype=torch.float32))


def main():
    print(f"Device: {DEVICE}")
    # 加载数据
    mid_map = torch.load(MOVIE_ID_MAP_FILE, weights_only=False)
    num_movies = len(mid_map)
    txt_feat = torch.load(TEXT_FEATURES_FILE, weights_only=False)
    txt_matrix = np.zeros((num_movies, 384), dtype=np.float32)
    for mid, idx in mid_map.items():
        if mid in txt_feat:
            txt_matrix[idx] = txt_feat[mid]

    rdf = pd.read_csv(RATINGS_CSV)
    uid_map = {int(u): i for i, u in enumerate(sorted(rdf["userId"].unique()))}
    num_users = len(uid_map)

    valid_mid = set(mid_map.keys())
    rdf = rdf[rdf["movieId"].isin(valid_mid)]
    u_idx = rdf["userId"].map(uid_map).values.astype(np.int64)
    m_idx = rdf["movieId"].map(mid_map).values.astype(np.int64)
    ratings = rdf["rating"].values.astype(np.float32)
    arr = np.stack([u_idx, m_idx, ratings], axis=1)
    print(f"Users:{num_users} Movies:{num_movies} Ratings:{len(arr)}")

    # 划分
    train_arr, val_arr = train_test_split(arr, test_size=VAL_RATIO, random_state=42)
    print(f"Train:{len(train_arr)} Val:{len(val_arr)}")

    train_ds = TextOnlyDataset(train_arr, txt_matrix)
    val_ds = TextOnlyDataset(val_arr, txt_matrix)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type=="cuda"))
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, pin_memory=(DEVICE.type=="cuda"))

    # 模型
    model = TextOnlyNCF(num_users, num_movies).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    crit = nn.MSELoss()

    best_rmse = float("inf")
    best_epoch = 0
    no_imp = 0
    history = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = 0.0
        for u, m, t, r in train_dl:
            u, m, t, r = u.to(DEVICE), m.to(DEVICE), t.to(DEVICE), r.to(DEVICE)
            pred = model(u, m, t)
            loss = crit(pred, r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()

        model.eval()
        vl, preds, targets = 0.0, [], []
        with torch.no_grad():
            for u, m, t, r in val_dl:
                u, m, t, r = u.to(DEVICE), m.to(DEVICE), t.to(DEVICE), r.to(DEVICE)
                pred = model(u, m, t)
                vl += crit(pred, r).item()
                preds.append(pred.cpu().numpy())
                targets.append(r.cpu().numpy())
        yp = np.concatenate(preds)
        yt = np.concatenate(targets)
        rmse = np.sqrt(np.mean((yp - yt) ** 2))
        mae = np.mean(np.abs(yp - yt))
        sch.step(vl / len(val_dl))

        mark = ""
        history.append({"epoch": ep, "train_loss": tl/len(train_dl),
                        "val_loss": vl/len(val_dl), "val_rmse": rmse, "val_mae": mae})

        if rmse < best_rmse:
            best_rmse, best_epoch = rmse, ep
            no_imp = 0
            mark = " *"
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {"num_users": num_users, "num_movies": num_movies,
                                 "text_dim": 384, "embed_dim": EMBED_DIM,
                                 "model_type": "textonly"},
                "val_rmse": float(rmse), "val_mae": float(mae), "epoch": ep,
                "training_history": history,
            }, "models/textonly_ncf.pt")
        else:
            no_imp += 1

        print(f"Epoch {ep:3d} | Loss={tl/len(train_dl):.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}{mark}")

        if no_imp >= PATIENCE:
            print(f"Early stop at epoch {ep}")
            break

    # 训练结束后，用含完整历史的 checkpoint 覆盖（保留最佳权重不变）
    best_ckpt = torch.load("models/textonly_ncf.pt", weights_only=False, map_location="cpu")
    best_ckpt["training_history"] = history
    torch.save(best_ckpt, "models/textonly_ncf.pt")

    print(f"\nDone! Best RMSE={best_rmse:.4f} at epoch {best_epoch}")
    print(f"Model saved: models/textonly_ncf.pt ({len(history)} epochs history)")


if __name__ == "__main__":
    main()
