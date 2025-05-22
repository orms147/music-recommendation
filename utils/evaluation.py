import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import LOGS_DIR, RANDOM_STATE
import os

logger = logging.getLogger(__name__)

class RecommenderEvaluator:
    """Đánh giá mô hình đề xuất âm nhạc"""
    
    def __init__(self, user_interactions=None):
        """Khởi tạo evaluator với dữ liệu tương tác người dùng"""
        self.user_interactions = user_interactions
        self.results = {}
    
    def load_interactions(self, file_path):
        """Tải dữ liệu tương tác"""
        self.user_interactions = pd.read_csv(file_path)
        logger.info(f"Loaded {len(self.user_interactions)} user interactions")
        return self
    
    def split_train_test(self, test_size=0.2, by_user=True):
        """Chia dữ liệu thành tập huấn luyện và kiểm tra"""
        if self.user_interactions is None:
            logger.error("No interaction data loaded")
            return None, None
        
        if by_user:
            # Chia theo người dùng - mỗi người dùng xuất hiện ở cả 2 tập
            users = self.user_interactions['user_id'].unique()
            train_data = pd.DataFrame()
            test_data = pd.DataFrame()
            
            for user in users:
                user_data = self.user_interactions[self.user_interactions['user_id'] == user]
                user_train, user_test = train_test_split(
                    user_data, test_size=test_size, random_state=RANDOM_STATE
                )
                train_data = pd.concat([train_data, user_train])
                test_data = pd.concat([test_data, user_test])
                
            logger.info(f"Split data by user: {len(train_data)} train, {len(test_data)} test samples")
        else:
            # Chia ngẫu nhiên
            train_data, test_data = train_test_split(
                self.user_interactions, test_size=test_size, random_state=RANDOM_STATE
            )
            logger.info(f"Split data randomly: {len(train_data)} train, {len(test_data)} test samples")
        
        return train_data, test_data
    
    def evaluate_model(self, model, test_data, k=10, metric='precision'):
        """Đánh giá mô hình với tập kiểm tra"""
        if test_data is None or model is None:
            logger.error("Test data or model is missing")
            return 0
        
        # Đánh giá cho từng người dùng
        users = test_data['user_id'].unique()
        scores = []
        
        for user in users:
            # Lấy bài hát thực tế mà người dùng đã nghe
            actual_tracks = test_data[test_data['user_id'] == user]['track_id'].tolist()
            
            if not actual_tracks:
                continue
                
            # Đề xuất bài hát cho người dùng
            recommendations = model.recommend(user_id=user, n_recommendations=k)
            
            if recommendations.empty:
                continue
                
            # Lấy ID bài hát được đề xuất
            recommended_tracks = recommendations['id'].tolist()
            
            # Tính score
            if metric == 'precision':
                # Precision@k = số lượng đề xuất đúng / số lượng đề xuất
                relevant = len(set(actual_tracks) & set(recommended_tracks))
                score = relevant / min(len(recommended_tracks), k) if recommended_tracks else 0
            elif metric == 'recall':
                # Recall@k = số lượng đề xuất đúng / số lượng thực tế
                relevant = len(set(actual_tracks) & set(recommended_tracks))
                score = relevant / len(actual_tracks) if actual_tracks else 0
            elif metric == 'f1':
                # F1 = 2 * (precision * recall) / (precision + recall)
                relevant = len(set(actual_tracks) & set(recommended_tracks))
                precision = relevant / min(len(recommended_tracks), k) if recommended_tracks else 0
                recall = relevant / len(actual_tracks) if actual_tracks else 0
                score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                score = 0
                
            scores.append(score)
        
        # Tính điểm trung bình
        avg_score = np.mean(scores) if scores else 0
        logger.info(f"Model evaluation - {metric}@{k}: {avg_score:.4f}")
        
        # Lưu kết quả
        self.results[f'{metric}@{k}'] = avg_score
        
        return avg_score
    
    def compare_models(self, models, test_data, k=10, metrics=None):
        """So sánh nhiều mô hình với nhau"""
        if metrics is None:
            metrics = ['precision', 'recall', 'f1']
            
        results = {}
        
        for model_name, model in models.items():
            model_results = {}
            for metric in metrics:
                score = self.evaluate_model(model, test_data, k, metric)
                model_results[metric] = score
            results[model_name] = model_results
        
        # Tạo DataFrame kết quả
        results_df = pd.DataFrame(results).T
        logger.info(f"Model comparison results:\n{results_df}")
        
        return results_df
    
    def plot_comparison(self, results_df, metrics=None, save_path=None):
        """Vẽ biểu đồ so sánh các mô hình"""
        if metrics is None:
            metrics = results_df.columns.tolist()
        
        plt.figure(figsize=(10, 6))
        
        for metric in metrics:
            if metric in results_df.columns:
                ax = results_df[metric].plot(kind='bar', alpha=0.7, label=metric)
        
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved comparison plot to {save_path}")
        
        plt.show()
        
        return plt

def evaluate_recommendation_diversity(recommendations, track_features):
    """Đánh giá tính đa dạng của các đề xuất"""
    if recommendations.empty or track_features.empty:
        return 0
    
    # Kết hợp đề xuất với đặc trưng bài hát
    merged = recommendations.merge(track_features, on='id', how='left')
    
    if merged.empty:
        return 0
    
    # Tính đa dạng dựa trên độ lệch chuẩn của các đặc trưng
    num_features = [col for col in track_features.columns if col not in ['id', 'name', 'artist']]
    diversity_scores = []
    
    for feature in num_features:
        if feature in merged.columns:
            diversity_scores.append(merged[feature].std())
    
    # Tính điểm đa dạng trung bình
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
    
    return avg_diversity