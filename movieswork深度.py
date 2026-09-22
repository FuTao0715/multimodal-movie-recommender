"""
电影评分预测模型训练脚本 - 多模态融合推荐系统
作者: Recommendation Team
功能: 融合协同过滤、图像特征和文本特征进行电影评分预测
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载模块 ====================

class MovieLensDataset(Dataset):
    """MovieLens数据集类"""
    def __init__(self, ratings_df, movie_indices, user_indices, image_features=None, text_features=None):
        """
        Args:
            ratings_df: DataFrame包含 userId, movieId, rating
            movie_indices: movieId到连续索引的映射
            user_indices: userId到连续索引的映射
            image_features: 图像特征字典 {movieId: feature_vector}
            text_features: 文本特征字典 {movieId: feature_vector}
        """
        self.ratings = ratings_df.values
        self.movie_indices = movie_indices
        self.user_indices = user_indices
        self.image_features = image_features
        self.text_features = text_features

        # 获取所有电影ID列表
        self.movie_ids = list(movie_indices.keys())

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings[idx]
        # ratings.csv 有4列: userId, movieId, rating, timestamp
        user_id = row[0]
        movie_id = row[1]
        rating = row[2]

        # 转换为连续索引
        user_idx = self.user_indices[user_id]
        movie_idx = self.movie_indices[movie_id]

        # 获取特征（如果有）
        img_feat = None
        txt_feat = None

        if self.image_features is not None and movie_id in self.image_features:
            img_feat = torch.FloatTensor(self.image_features[movie_id])

        if self.text_features is not None and movie_id in self.text_features:
            txt_feat = torch.FloatTensor(self.text_features[movie_id])

        return {
            'user_id': torch.LongTensor([user_idx]),
            'movie_id': torch.LongTensor([movie_idx]),
            'rating': torch.FloatTensor([rating]),
            'img_feat': img_feat,
            'txt_feat': txt_feat
        }


def load_data(data_path='ml-latest-small/ratings.csv'):
    """
    加载评分数据，建立用户和电影的连续索引映射

    Returns:
        ratings_df: 原始评分DataFrame
        user_indices: userId -> 连续索引
        movie_indices: movieId -> 连续索引
        num_users: 用户数量
        num_movies: 电影数量
    """
    print("="*60)
    print("加载数据...")
    ratings_df = pd.read_csv(data_path)

    print(f"原始评分数量: {len(ratings_df)}")
    print(f"用户数量: {ratings_df['userId'].nunique()}")
    print(f"电影数量: {ratings_df['movieId'].nunique()}")

    # 建立连续索引映射
    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()

    user_indices = {uid: i for i, uid in enumerate(unique_users)}
    movie_indices = {mid: i for i, mid in enumerate(unique_movies)}

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    print(f"用户索引映射: {num_users} 个用户")
    print(f"电影索引映射: {num_movies} 部电影")
    print("="*60)

    return ratings_df, user_indices, movie_indices, num_users, num_movies


def load_features(feature_dir='features'):
    """
    加载预提取的图像和文本特征

    Returns:
        image_features: dict {movieId: numpy array}
        text_features: dict {movieId: numpy array}
    """
    print("加载预提取特征...")

    image_features = None
    text_features = None

    # 加载图像特征
    img_path = os.path.join(feature_dir, 'image_features.pt')
    if os.path.exists(img_path):
        try:
            image_features = torch.load(img_path, map_location='cpu')
            if image_features:
                sample_key = next(iter(image_features.keys()))
                sample_shape = image_features[sample_key].shape if hasattr(image_features[sample_key], 'shape') else 'unknown'
                print(f"  ✓ 图像特征加载成功: {len(image_features)} 部电影")
            else:
                print(f"  ⚠ 图像特征为空")
        except Exception as e:
            print(f"  ✗ 图像特征加载失败: {e}")

    # 加载文本特征
    txt_path = os.path.join(feature_dir, 'text_features.pt')
    if os.path.exists(txt_path):
        try:
            text_features = torch.load(txt_path, map_location='cpu')
            if text_features:
                print(f"  ✓ 文本特征加载成功: {len(text_features)} 部电影")
            else:
                print(f"  ⚠ 文本特征为空")
        except Exception as e:
            print(f"  ✗ 文本特征加载失败: {e}")

    print("="*60)
    return image_features, text_features


def load_movie_metadata(movies_path='movies_enriched.csv'):
    """
    加载电影元数据，提取年份和体裁信息（作为辅助特征）
    """
    print("加载电影元数据...")
    movies_df = pd.read_csv(movies_path)

    # 从标题提取年份
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)

    print(f"电影元数据加载成功: {len(movies_df)} 部电影")
    print("="*60)
    return movies_df

# ==================== 2. 模型设计模块 ====================

class MultiModalRecommender(nn.Module):
    """
    多模态融合推荐模型

    融合三种模态:
    1. 协同过滤 (用户/物品嵌入)
    2. 图像特征 (ResNet50)
    3. 文本特征 (MiniLM)

    架构: Embedding + MLP + Fusion
    """

    def __init__(self, num_users, num_movies,
                 embed_dim=64,
                 img_feat_dim=2048,
                 txt_feat_dim=384,
                 hidden_dims=[256, 128, 64],
                 dropout=0.2):
        """
        Args:
            num_users: 用户数量
            num_movies: 电影数量
            embed_dim: 嵌入维度
            img_feat_dim: 图像特征维度 (ResNet50: 2048)
            txt_feat_dim: 文本特征维度 (MiniLM: 384)
            hidden_dims: MLP隐藏层维度列表
            dropout: Dropout比率
        """
        super(MultiModalRecommender, self).__init__()

        # 1. 协同过滤部分 - 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)

        # 2. 模态融合层 - 将图像和文本特征投影到统一空间
        self.img_projection = nn.Sequential(
            nn.Linear(img_feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        self.txt_projection = nn.Sequential(
            nn.Linear(txt_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        # 3. 融合层 - 将用户嵌入、物品嵌入和模态特征结合
        # 输入维度: user_embed + movie_embed + img_proj + txt_proj = 4 * embed_dim
        fusion_input_dim = 4 * embed_dim

        # 构建MLP
        layers = []
        prev_dim = fusion_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))

        self.fusion_mlp = nn.Sequential(*layers)

        # 4. 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)

    def forward(self, user_id, movie_id, img_feat=None, txt_feat=None):
        """
        前向传播

        Args:
            user_id: 用户ID索引 [batch_size]
            movie_id: 电影ID索引 [batch_size]
            img_feat: 图像特征 [batch_size, img_feat_dim]
            txt_feat: 文本特征 [batch_size, txt_feat_dim]

        Returns:
            pred: 预测评分 [batch_size, 1]
        """
        # 1. 获取嵌入
        user_emb = self.user_embedding(user_id)  # [batch, embed_dim]
        movie_emb = self.movie_embedding(movie_id)  # [batch, embed_dim]

        # 2. 投影模态特征
        if img_feat is not None:
            img_emb = self.img_projection(img_feat)  # [batch, embed_dim]
        else:
            # 如果没有图像特征，使用零向量
            img_emb = torch.zeros_like(movie_emb)

        if txt_feat is not None:
            txt_emb = self.txt_projection(txt_feat)  # [batch, embed_dim]
        else:
            txt_emb = torch.zeros_like(movie_emb)

        # 3. 拼接所有特征
        concat_feat = torch.cat([user_emb, movie_emb, img_emb, txt_emb], dim=1)

        # 4. 通过MLP预测评分
        pred = self.fusion_mlp(concat_feat)

        return pred.squeeze(-1)


class LightGCNLayer(nn.Module):
    """
    LightGCN单层传播
    """
    def __init__(self):
        super(LightGCNLayer, self).__init__()

    def forward(self, ego_embeddings, adj_matrix):
        """
        Args:
            ego_embeddings: 当前层嵌入 [num_users + num_movies, embed_dim]
            adj_matrix: 归一化邻接矩阵 [num_users + num_movies, num_users + num_movies]
        """
        return torch.sparse.mm(adj_matrix, ego_embeddings)


class LightGCN(nn.Module):
    """
    LightGCN模型

    简化版的GCN，只保留邻居聚合，无特征变换和非线性激活
    """
    def __init__(self, num_users, num_movies, embed_dim=64, n_layers=3, dropout=0.2):
        super(LightGCN, self).__init__()

        self.num_users = num_users
        self.num_movies = num_movies
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)

        # LightGCN层
        self.gcn_layers = nn.ModuleList([LightGCNLayer() for _ in range(n_layers)])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

        # 归一化邻接矩阵将在训练前构建
        self.adj_matrix = None

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.movie_embedding.weight, std=0.1)

    def build_adj_matrix(self, user_movie_pairs):
        """
        构建归一化邻接矩阵

        Args:
            user_movie_pairs: (user_idx, movie_idx) 元组列表
        """
        if len(user_movie_pairs) == 0:
            print("警告: 没有交互数据，无法构建邻接矩阵")
            return

        # 构建图的边列表
        edges = torch.LongTensor(user_movie_pairs).t()
        values = torch.ones(edges.shape[1])

        # 构建稀疏邻接矩阵 [num_users + num_movies, num_users + num_movies]
        n_nodes = self.num_users + self.num_movies
        adj = torch.sparse_coo_tensor(edges, values, (n_nodes, n_nodes))

        # 转换为归一化的邻接矩阵: D^(-1/2) * A * D^(-1/2)
        # 计算度
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

        # 构建归一化矩阵
        indices = adj._indices()
        values = adj._values()
        row_norm = deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]

        self.adj_matrix = torch.sparse_coo_tensor(indices, row_norm, adj.shape).to(self.user_embedding.weight.device)

    def forward(self, user_id, movie_id):
        """
        前向传播

        Args:
            user_id: 用户索引 [batch_size]
            movie_id: 电影索引 [batch_size]

        Returns:
            pred: 预测评分 [batch_size]
        """
        # 获取所有嵌入
        all_embeddings = torch.cat([self.user_embedding.weight, self.movie_embedding.weight], dim=0)

        # 多层GCN传播
        layer_embeddings = [all_embeddings]
        for layer in self.gcn_layers:
            all_embeddings = layer(all_embeddings, self.adj_matrix)
            all_embeddings = self.dropout(all_embeddings)
            layer_embeddings.append(all_embeddings)

        # 平均所有层的嵌入
        final_embeddings = torch.mean(torch.stack(layer_embeddings), dim=0)

        # 分离用户和物品嵌入
        user_emb = final_embeddings[:self.num_users]
        movie_emb = final_embeddings[self.num_users:]

        # 获取当前batch的嵌入
        batch_user_emb = user_emb[user_id]
        batch_movie_emb = movie_emb[movie_id]

        # 预测评分
        pred = torch.sum(batch_user_emb * batch_movie_emb, dim=1)

        return pred


class HybridRecommender(nn.Module):
    """
    混合推荐模型

    结合LightGCN和多模态特征的优点
    """
    def __init__(self, num_users, num_movies,
                 embed_dim=64,
                 img_feat_dim=2048,
                 txt_feat_dim=384,
                 n_layers=2,
                 hidden_dims=[128, 64],
                 dropout=0.2):
        super(HybridRecommender, self).__init__()

        # LightGCN部分
        self.lightgcn = LightGCN(num_users, num_movies, embed_dim, n_layers, dropout)

        # 模态投影层
        self.img_projection = nn.Sequential(
            nn.Linear(img_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        self.txt_projection = nn.Sequential(
            nn.Linear(txt_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        # 融合层
        fusion_input_dim = embed_dim * 2 + 1  # user_emb + modal_emb + gcn_score
        layers = []
        prev_dim = fusion_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.fusion_mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def build_adj_matrix(self, user_movie_pairs):
        """构建LightGCN的邻接矩阵"""
        self.lightgcn.build_adj_matrix(user_movie_pairs)

    def forward(self, user_id, movie_id, img_feat=None, txt_feat=None):
        """
        前向传播
        """
        # 获取LightGCN的预测
        gcn_score = self.lightgcn(user_id, movie_id)

        # 获取模态特征
        if img_feat is not None:
            img_emb = self.img_projection(img_feat)
        else:
            img_emb = torch.zeros(user_id.size(0), self.lightgcn.embed_dim).to(user_id.device)

        if txt_feat is not None:
            txt_emb = self.txt_projection(txt_feat)
        else:
            txt_emb = torch.zeros(user_id.size(0), self.lightgcn.embed_dim).to(user_id.device)

        # 融合模态特征
        modal_emb = (img_emb + txt_emb) / 2

        # 获取LightGCN的用户嵌入
        all_embeddings = torch.cat([self.lightgcn.user_embedding.weight, self.lightgcn.movie_embedding.weight], dim=0)
        user_emb = all_embeddings[user_id]

        # 融合所有特征
        concat_feat = torch.cat([user_emb, modal_emb, gcn_score.unsqueeze(-1)], dim=1)
        pred = self.fusion_mlp(concat_feat)

        return pred.squeeze(-1)

# ==================== 3. 训练与评估模块 ====================

def collate_fn(batch):
    """自定义批处理函数"""
    user_ids = torch.cat([item['user_id'] for item in batch])
    movie_ids = torch.cat([item['movie_id'] for item in batch])
    ratings = torch.cat([item['rating'] for item in batch])

    # 处理图像特征
    img_feats = []
    has_img = False
    for item in batch:
        if item['img_feat'] is not None:
            img_feats.append(item['img_feat'])
            has_img = True
        else:
            # 如果没有特征，使用零向量
            img_feats.append(torch.zeros(2048))

    # 处理文本特征
    txt_feats = []
    has_txt = False
    for item in batch:
        if item['txt_feat'] is not None:
            txt_feats.append(item['txt_feat'])
            has_txt = True
        else:
            txt_feats.append(torch.zeros(384))

    return {
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings,
        'img_feat': torch.stack(img_feats) if has_img else None,
        'txt_feat': torch.stack(txt_feats) if has_txt else None
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc='Training', leave=False):
        user_id = batch['user_id'].squeeze().to(device)
        movie_id = batch['movie_id'].squeeze().to(device)
        rating = batch['rating'].squeeze().to(device)

        img_feat = batch['img_feat'].to(device) if batch['img_feat'] is not None else None
        txt_feat = batch['txt_feat'].to(device) if batch['txt_feat'] is not None else None

        # 前向传播
        pred = model(user_id, movie_id, img_feat, txt_feat)
        loss = criterion(pred, rating)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(pred.detach().cpu().numpy())
        all_targets.extend(rating.detach().cpu().numpy())

    # 计算指标
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))

    return total_loss / len(dataloader), rmse, mae


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            user_id = batch['user_id'].squeeze().to(device)
            movie_id = batch['movie_id'].squeeze().to(device)
            rating = batch['rating'].squeeze().to(device)

            img_feat = batch['img_feat'].to(device) if batch['img_feat'] is not None else None
            txt_feat = batch['txt_feat'].to(device) if batch['txt_feat'] is not None else None

            pred = model(user_id, movie_id, img_feat, txt_feat)
            loss = criterion(pred, rating)

            total_loss += loss.item()
            all_preds.extend(pred.detach().cpu().numpy())
            all_targets.extend(rating.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))

    return total_loss / len(dataloader), rmse, mae


def create_train_val_split(ratings_df, test_size=0.2, random_state=42):
    """
    按用户分层划分训练集和验证集

    确保每个用户的评分都在训练集和验证集中都有体现
    """
    train_list = []
    val_list = []

    for user_id in ratings_df['userId'].unique():
        user_ratings = ratings_df[ratings_df['userId'] == user_id]

        # 如果用户评分少于2条，全部放入训练集
        if len(user_ratings) < 2:
            train_list.append(user_ratings)
            continue

        # 分层采样
        train_user, val_user = train_test_split(
            user_ratings,
            test_size=min(test_size, 0.5),  # 至少保留一条验证
            random_state=random_state
        )
        train_list.append(train_user)
        val_list.append(val_user)

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)

    return train_df, val_df


def build_user_movie_pairs(ratings_df, user_indices, movie_indices):
    """构建用户-物品交互对，用于LightGCN邻接矩阵"""
    pairs = []
    for _, row in ratings_df.iterrows():
        pairs.append((user_indices[row['userId']], movie_indices[row['movieId']]))
    return pairs

# ==================== 4. 主训练函数 ====================

def main():
    """主训练函数"""
    print("\n" + "="*60)
    print("电影评分预测模型训练")
    print("="*60 + "\n")

    # 配置参数
    CONFIG = {
        'embed_dim': 64,
        'n_layers': 2,
        'hidden_dims': [256, 128, 64],
        'dropout': 0.2,
        'batch_size': 512,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 100,
        'patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_seed': 42,
        'model_type': 'hybrid'  # 'hybrid' 或 'lightgcn' 或 'multimodal'
    }

    print(f"使用设备: {CONFIG['device']}")
    print(f"模型类型: {CONFIG['model_type']}")
    print("-"*60)

    # 设置随机种子
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    # 1. 加载数据
    ratings_df, user_indices, movie_indices, num_users, num_movies = load_data()

    # 2. 加载特征
    image_features, text_features = load_features()

    # 打印特征信息
    if image_features:
        print(f"图像特征示例: movieId={list(image_features.keys())[0]}, shape={image_features[list(image_features.keys())[0]].shape}")
    if text_features:
        print(f"文本特征示例: movieId={list(text_features.keys())[0]}, shape={text_features[list(text_features.keys())[0]].shape}")
    print("-"*60)

    # 3. 划分数据集
    print("划分训练集和验证集...")
    train_df, val_df = create_train_val_split(ratings_df, test_size=0.2)
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print("-"*60)

    # 4. 创建数据加载器
    train_dataset = MovieLensDataset(train_df, movie_indices, user_indices, image_features, text_features)
    val_dataset = MovieLensDataset(val_df, movie_indices, user_indices, image_features, text_features)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 5. 创建模型
    device = torch.device(CONFIG['device'])

    if CONFIG['model_type'] == 'lightgcn':
        model = LightGCN(num_users, num_movies, CONFIG['embed_dim'], CONFIG['n_layers'], CONFIG['dropout'])
        # 构建邻接矩阵
        pairs = build_user_movie_pairs(train_df, user_indices, movie_indices)
        model.build_adj_matrix(pairs)
        model_config = {
            'num_users': num_users,
            'num_movies': num_movies,
            'model_type': 'lightgcn',
            'embed_dim': CONFIG['embed_dim'],
            'n_layers': CONFIG['n_layers'],
            'dropout': CONFIG['dropout'],
        }
    elif CONFIG['model_type'] == 'multimodal':
        model = MultiModalRecommender(
            num_users, num_movies,
            embed_dim=CONFIG['embed_dim'],
            img_feat_dim=2048,
            txt_feat_dim=384,
            hidden_dims=CONFIG['hidden_dims'],
            dropout=CONFIG['dropout']
        )
        model_config = {
            'num_users': num_users,
            'num_movies': num_movies,
            'model_type': 'multimodal',
            'embed_dim': CONFIG['embed_dim'],
            'img_feat_dim': 2048,
            'txt_feat_dim': 384,
            'hidden_dims': CONFIG['hidden_dims'],
            'dropout': CONFIG['dropout'],
        }
    else:  # hybrid (默认)
        model = HybridRecommender(
            num_users, num_movies,
            embed_dim=CONFIG['embed_dim'],
            img_feat_dim=2048,
            txt_feat_dim=384,
            n_layers=CONFIG['n_layers'],
            hidden_dims=CONFIG['hidden_dims'],
            dropout=CONFIG['dropout']
        )
        # 构建LightGCN邻接矩阵
        pairs = build_user_movie_pairs(train_df, user_indices, movie_indices)
        model.build_adj_matrix(pairs)
        model_config = {
            'num_users': num_users,
            'num_movies': num_movies,
            'model_type': 'hybrid',
            'embed_dim': CONFIG['embed_dim'],
            'img_feat_dim': 2048,
            'txt_feat_dim': 384,
            'n_layers': CONFIG['n_layers'],
            'hidden_dims': CONFIG['hidden_dims'],
            'dropout': CONFIG['dropout'],
        }

    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 6. 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG['learning_rate'],
                                 weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    # 7. 训练循环
    print("\n开始训练...")
    print("-"*60)

    best_val_rmse = float('inf')
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(CONFIG['epochs']):
        # 训练
        train_loss, train_rmse, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step(val_rmse)

        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'lr': optimizer.param_groups[0]['lr']
        })

        # 打印进度
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")

        # 早停检查
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_mae = val_mae
            best_epoch = epoch + 1
            patience_counter = 0

            # 保存最佳模型
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n早停触发! 最佳Epoch: {best_epoch}, 最佳Val RMSE: {best_val_rmse:.4f}")
                break

    # 8. 保存模型
    print("\n" + "-"*60)
    print("保存模型...")

    # 创建模型目录
    os.makedirs('models', exist_ok=True)

    # 构建checkpoint
    checkpoint = {
        'model_state_dict': best_model_state,
        'model_config': model_config,
        'val_rmse': best_val_rmse,
        'val_mae': best_val_mae,
        'epoch': best_epoch,
        'training_history': history
    }

    # 保存模型
    model_name = f"{CONFIG['model_type']}_recommender"
    model_path = f"models/{model_name}.pt"
    torch.save(checkpoint, model_path)

    print(f"模型已保存到: {model_path}")
    print(f"最佳验证集 RMSE: {best_val_rmse:.4f}")
    print(f"最佳验证集 MAE: {best_val_mae:.4f}")
    print(f"最佳 Epoch: {best_epoch}")

    # 9. 输出最终结果
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"\n模型: {CONFIG['model_type']}")
    print(f"最佳验证 RMSE: {best_val_rmse:.4f}")
    print(f"最佳验证 MAE: {best_val_mae:.4f}")

    # 评估结果
    if best_val_rmse < 0.85:
        print("🎉 表现优秀! (RMSE < 0.85)")
    elif best_val_rmse < 0.95:
        print("👍 表现良好! (RMSE < 0.95)")
    elif best_val_rmse < 1.05:
        print("📊 表现合格! (RMSE < 1.05)")
    else:
        print("⚠️ 需要改进! (RMSE > 1.05)")

    print("="*60)


if __name__ == "__main__":
    main()