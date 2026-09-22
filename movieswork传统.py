"""
电影评分预测模型 - 传统机器学习方法
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

# ==================== 1. 数据加载 ====================

def load_data():
    """加载评分和电影数据"""
    print("加载数据...")

    # 使用完整路径
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    movies = pd.read_csv(r"F:\机器学习\movies\movies\movies_enriched.csv")

    print(f"评分数: {len(ratings)}")
    print(f"用户数: {ratings['userId'].nunique()}")
    print(f"电影数: {ratings['movieId'].nunique()}")

    # 创建连续索引映射
    unique_movies = ratings['movieId'].unique()
    unique_users = ratings['userId'].unique()

    movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}

    # 添加索引列
    ratings['movie_idx'] = ratings['movieId'].map(movie_to_idx)
    ratings['user_idx'] = ratings['userId'].map(user_to_idx)

    return ratings, movies, user_to_idx, movie_to_idx

# ==================== 2. 构建用户-电影评分矩阵 ====================

def build_user_movie_matrix(ratings, n_users, n_movies):
    """构建用户-电影评分矩阵"""
    matrix = np.zeros((n_users, n_movies))
    for row in ratings.itertuples():
        matrix[row.user_idx, row.movie_idx] = row.rating
    return matrix

# ==================== 3. SVD模型 ====================

class SVDRecommender:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.user_factors = None
        self.movie_factors = None
        self.global_mean = 0

    def fit(self, ratings_df, n_users, n_movies):
        print("训练SVD模型...")
        matrix = build_user_movie_matrix(ratings_df, n_users, n_movies)
        self.global_mean = matrix[matrix > 0].mean()
        matrix_centered = matrix.copy()
        matrix_centered[matrix_centered > 0] -= self.global_mean
        U, s, Vt = svds(matrix_centered, k=self.n_factors)
        self.user_factors = U * np.sqrt(s)
        self.movie_factors = Vt.T * np.sqrt(s)

    def predict(self, user_id, movie_id, user_to_idx, movie_to_idx):
        user_idx = user_to_idx[user_id]
        movie_idx = movie_to_idx[movie_id]
        pred = self.global_mean + np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
        return np.clip(pred, 0.5, 5.0)

# ==================== 4. User-KNN模型 ====================

class UserKNNRecommender:
    def __init__(self, k=20):
        self.k = k
        self.user_similarity = None
        self.ratings_matrix = None

    def fit(self, ratings_df, n_users, n_movies):
        print("训练User-KNN模型...")
        self.ratings_matrix = build_user_movie_matrix(ratings_df, n_users, n_movies)
        self.user_similarity = cosine_similarity(self.ratings_matrix)

    def predict(self, user_id, movie_id, user_to_idx, movie_to_idx):
        user_idx = user_to_idx[user_id]
        movie_idx = movie_to_idx[movie_id]
        sim_scores = self.user_similarity[user_idx]
        movie_ratings = self.ratings_matrix[:, movie_idx]
        rated_mask = movie_ratings > 0
        if not rated_mask.any():
            return 3.0
        weights = sim_scores[rated_mask]
        ratings = movie_ratings[rated_mask]
        if weights.sum() == 0:
            return 3.0
        pred = np.dot(weights, ratings) / weights.sum()
        return np.clip(pred, 0.5, 5.0)

# ==================== 5. 随机森林 ====================

class RandomForestRecommender:
    def __init__(self, n_estimators=100, max_depth=12):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.user_avg = None
        self.movie_avg = None

    def fit(self, ratings_df):
        print("训练随机森林模型...")
        self.user_avg = ratings_df.groupby('userId')['rating'].mean()
        self.movie_avg = ratings_df.groupby('movieId')['rating'].mean()
        features = self._extract_features(ratings_df)
        targets = ratings_df['rating'].values
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, targets)

    def _extract_features(self, df):
        features = []
        for _, row in df.iterrows():
            features.append([
                self.user_avg.get(row['userId'], 3.0),
                self.movie_avg.get(row['movieId'], 3.0),
                row['userId'] % 10,
                row['movieId'] % 10,
            ])
        return np.array(features)

    def predict(self, user_id, movie_id, user_to_idx=None, movie_to_idx=None):
        features = np.array([[
            self.user_avg.get(user_id, 3.0),
            self.movie_avg.get(movie_id, 3.0),
            user_id % 10,
            movie_id % 10,
        ]])
        features_scaled = self.scaler.transform(features)
        pred = self.model.predict(features_scaled)[0]
        return np.clip(pred, 0.5, 5.0)

# ==================== 6. 岭回归 ====================

class RidgeRecommender:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha, random_state=42)
        self.scaler = StandardScaler()
        self.user_avg = None
        self.movie_avg = None

    def fit(self, ratings_df):
        print("训练Ridge回归模型...")
        self.user_avg = ratings_df.groupby('userId')['rating'].mean()
        self.movie_avg = ratings_df.groupby('movieId')['rating'].mean()
        features = self._extract_features(ratings_df)
        targets = ratings_df['rating'].values
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, targets)

    def _extract_features(self, df):
        features = []
        for _, row in df.iterrows():
            features.append([
                self.user_avg.get(row['userId'], 3.0),
                self.movie_avg.get(row['movieId'], 3.0),
                row['userId'] % 10,
                row['movieId'] % 10,
            ])
        return np.array(features)

    def predict(self, user_id, movie_id, user_to_idx=None, movie_to_idx=None):
        features = np.array([[
            self.user_avg.get(user_id, 3.0),
            self.movie_avg.get(movie_id, 3.0),
            user_id % 10,
            movie_id % 10,
        ]])
        features_scaled = self.scaler.transform(features)
        pred = self.model.predict(features_scaled)[0]
        return np.clip(pred, 0.5, 5.0)

# ==================== 7. 主函数 ====================

def main():
    print("\n" + "="*60)
    print("传统机器学习推荐系统")
    print("="*60 + "\n")

    # 1. 加载数据
    ratings, movies, user_to_idx, movie_to_idx = load_data()

    # 2. 划分训练集和验证集
    train_df, val_df = train_test_split(ratings, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print("-"*60)

    n_users = ratings['userId'].nunique()
    n_movies = ratings['movieId'].nunique()

    # 3. 训练各模型
    models = []

    # SVD
    try:
        model_svd = SVDRecommender(n_factors=50)
        model_svd.fit(train_df, n_users, n_movies)
        models.append(('SVD', model_svd))
    except Exception as e:
        print(f"SVD训练失败: {e}")

    # User-KNN
    try:
        model_knn = UserKNNRecommender(k=20)
        model_knn.fit(train_df, n_users, n_movies)
        models.append(('UserKNN', model_knn))
    except Exception as e:
        print(f"KNN训练失败: {e}")

    # 随机森林
    try:
        model_rf = RandomForestRecommender(n_estimators=100, max_depth=12)
        model_rf.fit(train_df)
        models.append(('RandomForest', model_rf))
    except Exception as e:
        print(f"随机森林训练失败: {e}")

    # 岭回归
    try:
        model_ridge = RidgeRecommender(alpha=1.0)
        model_ridge.fit(train_df)
        models.append(('Ridge', model_ridge))
    except Exception as e:
        print(f"Ridge训练失败: {e}")

    # 4. 评估各模型
    print("\n" + "-"*60)
    print("评估各模型...")
    print("-"*60)

    results = []
    sample = val_df.sample(min(5000, len(val_df)), random_state=42)

    for name, model in models:
        preds = []
        actuals = []
        for _, row in sample.iterrows():
            pred = model.predict(row['userId'], row['movieId'], user_to_idx, movie_to_idx)
            preds.append(pred)
            actuals.append(row['rating'])
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        results.append((name, rmse, mae))
        print(f"{name:12s} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # 5. 集成
    print("\n" + "-"*60)
    print("集成预测...")

    # 简单平均集成
    ensemble_preds = []
    for _, row in sample.iterrows():
        preds = []
        for _, model in models:
            preds.append(model.predict(row['userId'], row['movieId'], user_to_idx, movie_to_idx))
        ensemble_preds.append(np.mean(preds))

    actuals = sample['rating'].values
    rmse = np.sqrt(mean_squared_error(actuals, ensemble_preds))
    mae = mean_absolute_error(actuals, ensemble_preds)
    print(f"{'Ensemble':12s} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # 6. 结果
    best = min(results, key=lambda x: x[1])
    print("\n" + "="*60)
    print(f"最佳模型: {best[0]}")
    print(f"最佳RMSE: {best[1]:.4f}")
    print(f"最佳MAE: {best[2]:.4f}")
    print("="*60)

    # 7. 保存模型
    os.makedirs('models', exist_ok=True)
    for name, model in models:
        if name == best[0]:
            with open(f'models/{name.lower()}_recommender.pkl', 'wb') as f:
                pickle.dump(model, f)
            print(f"模型已保存: models/{name.lower()}_recommender.pkl")

if __name__ == "__main__":
    main()