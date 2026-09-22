"""
推荐系统可视化分析
展示数据分布、模型对比和预测效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
sns.set_palette('husl')


# ==================== 模型类定义（必须与训练时一致） ====================

class RandomForestRecommender:
    """随机森林推荐器"""
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


# ==================== 1. 数据加载 ====================

def load_data():
    """加载数据"""
    print("加载数据...")
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    movies = pd.read_csv(r"F:\机器学习\movies\movies\movies_enriched.csv")
    return ratings, movies


# ==================== 2. 数据分布可视化 ====================

def plot_rating_distribution(ratings):
    """评分分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 评分分布
    rating_counts = ratings['rating'].value_counts().sort_index()
    colors = ['#FF6B6B', '#FFA94D', '#FFD93D', '#6BCB77', '#4D96FF', '#9B59B6']
    axes[0].bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black', linewidth=1)
    axes[0].set_xlabel('评分', fontsize=12)
    axes[0].set_ylabel('数量', fontsize=12)
    axes[0].set_title('评分分布', fontsize=14, fontweight='bold')
    axes[0].set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    for i, v in enumerate(rating_counts.values):
        axes[0].text(rating_counts.index[i], v + 50, str(v), ha='center', fontsize=10)

    # 评分占比
    rating_pct = ratings['rating'].value_counts(normalize=True).sort_index() * 100
    axes[1].pie(rating_pct.values, labels=rating_pct.index, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=[0.02]*len(rating_pct))
    axes[1].set_title('评分占比', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/rating_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 评分分布图已保存: visualizations/rating_distribution.png")


# ==================== 3. 用户和电影统计 ====================

def plot_user_movie_stats(ratings):
    """用户和电影统计图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 用户评分数量分布
    user_counts = ratings.groupby('userId')['rating'].count()
    axes[0, 0].hist(user_counts, bins=30, color='#4D96FF', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(user_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {user_counts.mean():.1f}')
    axes[0, 0].axvline(user_counts.median(), color='green', linestyle='--', linewidth=2, label=f'中位数: {user_counts.median():.0f}')
    axes[0, 0].set_xlabel('评分数量', fontsize=12)
    axes[0, 0].set_ylabel('用户数', fontsize=12)
    axes[0, 0].set_title('用户评分数量分布', fontsize=14, fontweight='bold')
    axes[0, 0].legend()

    # 电影评分数量分布
    movie_counts = ratings.groupby('movieId')['rating'].count()
    axes[0, 1].hist(movie_counts, bins=50, color='#6BCB77', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(movie_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {movie_counts.mean():.1f}')
    axes[0, 1].axvline(movie_counts.median(), color='green', linestyle='--', linewidth=2, label=f'中位数: {movie_counts.median():.0f}')
    axes[0, 1].set_xlabel('评分数量', fontsize=12)
    axes[0, 1].set_ylabel('电影数', fontsize=12)
    axes[0, 1].set_title('电影评分数量分布', fontsize=14, fontweight='bold')
    axes[0, 1].legend()

    # 用户平均评分分布
    user_avg = ratings.groupby('userId')['rating'].mean()
    axes[1, 0].hist(user_avg, bins=20, color='#FFA94D', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(user_avg.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {user_avg.mean():.2f}')
    axes[1, 0].set_xlabel('平均评分', fontsize=12)
    axes[1, 0].set_ylabel('用户数', fontsize=12)
    axes[1, 0].set_title('用户平均评分分布', fontsize=14, fontweight='bold')
    axes[1, 0].legend()

    # 电影平均评分分布
    movie_avg = ratings.groupby('movieId')['rating'].mean()
    axes[1, 1].hist(movie_avg, bins=20, color='#FF6B6B', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(movie_avg.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {movie_avg.mean():.2f}')
    axes[1, 1].set_xlabel('平均评分', fontsize=12)
    axes[1, 1].set_ylabel('电影数', fontsize=12)
    axes[1, 1].set_title('电影平均评分分布', fontsize=14, fontweight='bold')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('visualizations/user_movie_stats.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 用户电影统计图已保存: visualizations/user_movie_stats.png")


# ==================== 4. 模型性能对比 ====================

def plot_model_comparison(results):
    """模型性能对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = [r[0] for r in results]
    rmse = [r[1] for r in results]
    mae = [r[2] for r in results]

    colors = ['#4D96FF', '#6BCB77', '#FFA94D', '#FF6B6B', '#9B59B6']

    # RMSE对比
    bars1 = axes[0].bar(models, rmse, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=0.85, color='green', linestyle='--', linewidth=2, label='优秀 (<0.85)')
    axes[0].axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='良好 (<0.95)')
    axes[0].axhline(y=1.05, color='red', linestyle='--', linewidth=2, label='合格 (<1.05)')
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].set_title('模型RMSE对比', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0.8, 1.1)

    for bar, val in zip(bars1, rmse):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    # MAE对比
    bars2 = axes[1].bar(models, mae, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('模型MAE对比', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0.6, 0.85)

    for bar, val in zip(bars2, mae):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 模型对比图已保存: visualizations/model_comparison.png")


# ==================== 5. 预测 vs 实际 ====================

def plot_prediction_scatter(y_true, y_pred, model_name):
    """预测值vs实际值散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散点图
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color='#4D96FF')
    axes[0].plot([0.5, 5.0], [0.5, 5.0], 'r--', linewidth=2, label='完美预测')
    axes[0].set_xlabel('实际评分', fontsize=12)
    axes[0].set_ylabel('预测评分', fontsize=12)
    axes[0].set_title(f'{model_name} - 预测 vs 实际', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0.5, 5.0)
    axes[0].set_ylim(0.5, 5.0)
    axes[0].legend()

    # 残差分布
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='green', linestyle='--', linewidth=2, label='零误差')
    axes[1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'均值: {residuals.mean():.3f}')
    axes[1].set_xlabel('预测误差 (预测 - 实际)', fontsize=12)
    axes[1].set_ylabel('频数', fontsize=12)
    axes[1].set_title('预测残差分布', fontsize=14, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name.lower()}_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 预测散点图已保存: visualizations/{model_name.lower()}_prediction.png")


# ==================== 6. 预测误差热力图 ====================

def plot_error_heatmap(y_true, y_pred):
    """误差热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    error_matrix = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            true_val = i * 0.5 + 0.5
            pred_val = j * 0.5 + 0.5
            mask = (y_true == true_val) & (y_pred == pred_val)
            if mask.sum() > 0:
                error_matrix[i, j] = mask.sum()

    row_sums = error_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    error_matrix_pct = error_matrix / row_sums * 100

    sns.heatmap(error_matrix_pct, annot=True, fmt='.1f', cmap='RdYlGn_r',
                xticklabels=[f'{i*0.5+0.5:.1f}' for i in range(9)],
                yticklabels=[f'{i*0.5+0.5:.1f}' for i in range(9)],
                ax=ax, cbar_kws={'label': '占比 (%)'})

    ax.set_xlabel('预测评分', fontsize=12)
    ax.set_ylabel('实际评分', fontsize=12)
    ax.set_title('预测误差热力图\n(每行归一化, 显示实际评分被预测为各评分的比例)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/error_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ 误差热力图已保存: visualizations/error_heatmap.png")


# ==================== 7. 主函数 ====================

def main():
    import os
    os.makedirs('visualizations', exist_ok=True)

    print("\n" + "="*60)
    print("推荐系统可视化分析")
    print("="*60 + "\n")

    # 1. 加载数据
    ratings, movies = load_data()
    print("-"*60)

    # 2. 数据分布可视化
    print("\n📊 绘制数据分布...")
    plot_rating_distribution(ratings)
    plot_user_movie_stats(ratings)

    # 3. 模型预测
    print("\n" + "-"*60)
    print("加载随机森林模型进行预测...")

    try:
        with open('models/randomforest_recommender.pkl', 'rb') as f:
            model = pickle.load(f)

        _, val_df = train_test_split(ratings, test_size=0.2, random_state=42)
        sample = val_df.sample(2000, random_state=42)

        unique_movies = ratings['movieId'].unique()
        unique_users = ratings['userId'].unique()
        movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        user_to_idx = {uid: i for i, uid in enumerate(unique_users)}

        preds = []
        actuals = []
        for _, row in sample.iterrows():
            pred = model.predict(row['userId'], row['movieId'], user_to_idx, movie_to_idx)
            preds.append(pred)
            actuals.append(row['rating'])

        y_true = np.array(actuals)
        y_pred = np.array(preds)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"随机森林 RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        print("\n📊 绘制预测分析图...")
        plot_prediction_scatter(y_true, y_pred, 'RandomForest')
        plot_error_heatmap(y_true, y_pred)

    except FileNotFoundError:
        print("⚠️ 模型文件不存在，请先运行训练脚本")
    except Exception as e:
        print(f"⚠️ 预测失败: {e}")

    # 4. 模型对比
    results = [
        ('SVD', 1.0335, 0.8151),
        ('UserKNN', 0.9957, 0.7737),
        ('RandomForest', 0.9028, 0.6924),
        ('Ridge', 0.9047, 0.6963),
        ('Ensemble', 0.9138, 0.7104),
    ]
    print("\n📊 绘制模型对比图...")
    plot_model_comparison(results)

    print("\n" + "="*60)
    print("可视化完成！")
    print(f"图片保存位置: {os.path.abspath('visualizations')}")
    print("="*60)


if __name__ == "__main__":
    main()