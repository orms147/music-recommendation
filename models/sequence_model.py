import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from collections import defaultdict, Counter
from models.base_model import BaseRecommender
from config.config import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceBasedRecommender(BaseRecommender):
    """Mô hình đề xuất dựa trên chuỗi bài hát"""
    
    def __init__(self, window_size=3):
        super().__init__(name="SequenceBasedRecommender")
        self.window_size = window_size
        self.track_sequences = defaultdict(Counter)
        self.track_id_to_name = {}
        self.track_id_to_artist = {}
    
    def train(self, user_sequences=None, tracks_df=None):
        """Huấn luyện mô hình dựa trên chuỗi nghe nhạc của người dùng"""
        start_time = datetime.now()
        
        # Tải dữ liệu nếu chưa có
        if user_sequences is None:
            # Cố gắng tạo chuỗi từ dữ liệu tương tác người dùng
            interactions_path = os.path.join(PROCESSED_DATA_DIR, 'user_interactions.csv')
            if os.path.exists(interactions_path):
                interactions_df = pd.read_csv(interactions_path)
                user_sequences = self._create_sequences_from_interactions(interactions_df)
                logger.info(f"Created {len(user_sequences)} user sequences from interactions")
            else:
                logger.error(f"User interactions file not found: {interactions_path}")
                return False
        
        if tracks_df is None:
            tracks_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
            if os.path.exists(tracks_path):
                tracks_df = pd.read_csv(tracks_path)
                logger.info(f"Loaded {len(tracks_df)} tracks")
            else:
                logger.warning(f"Track features file not found: {tracks_path}")
                # Có thể tiếp tục mà không cần thông tin chi tiết bài hát
        
        # Tạo ánh xạ từ ID đến tên và nghệ sĩ
        if tracks_df is not None:
            self.track_id_to_name = dict(zip(tracks_df['id'], tracks_df['name']))
            self.track_id_to_artist = dict(zip(tracks_df['id'], tracks_df['artist']))
        
        # Xử lý các chuỗi và tạo mô hình Markov
        for sequence in user_sequences:
            if len(sequence) >= self.window_size + 1:
                for i in range(len(sequence) - self.window_size):
                    # Sử dụng cửa sổ trượt để tạo chuỗi con
                    window = tuple(sequence[i:i+self.window_size])
                    next_track = sequence[i+self.window_size]
                    self.track_sequences[window][next_track] += 1
        
        logger.info(f"Trained sequence model with {len(self.track_sequences)} unique patterns")
        
        self.train_time = datetime.now() - start_time
        self.is_trained = True
        
        return True
    
    def _create_sequences_from_interactions(self, interactions_df):
        """Tạo chuỗi nghe nhạc từ tương tác người dùng"""
        # Sắp xếp theo người dùng và thời gian (nếu có)
        if 'timestamp' in interactions_df.columns:
            interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])
        elif 'added_at' in interactions_df.columns:
            interactions_df = interactions_df.sort_values(['user_id', 'added_at'])
        
        # Tạo chuỗi cho mỗi người dùng
        user_sequences = []
        for user_id in interactions_df['user_id'].unique():
            user_tracks = interactions_df[interactions_df['user_id'] == user_id]['track_id'].tolist()
            if len(user_tracks) >= self.window_size + 1:
                user_sequences.append(user_tracks)
        
        return user_sequences
    
    def recommend(self, recent_tracks=None, user_id=None, track_name=None, n_recommendations=10):
        """Đề xuất bài hát tiếp theo dựa trên chuỗi bài hát gần đây"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        # Nếu không có recent_tracks nhưng có user_id, tìm chuỗi gần đây của user
        if recent_tracks is None and user_id is not None:
            # Cố gắng tìm chuỗi bài hát gần đây của người dùng
            interactions_path = os.path.join(PROCESSED_DATA_DIR, 'user_interactions.csv')
            if os.path.exists(interactions_path):
                interactions_df = pd.read_csv(interactions_path)
                user_tracks = interactions_df[interactions_df['user_id'] == user_id]['track_id'].tolist()
                if len(user_tracks) >= self.window_size:
                    recent_tracks = user_tracks[-self.window_size:]
                    logger.info(f"Using {len(recent_tracks)} recent tracks for user {user_id}")
        
        # Nếu không có recent_tracks nhưng có track_name, trả về DataFrame rỗng
        if recent_tracks is None:
            logger.error("No recent tracks available for recommendation")
            return pd.DataFrame()
            
        n_recommendations = self._validate_n_recommendations(n_recommendations)
        
        # Kiểm tra đầu vào
        if not recent_tracks or len(recent_tracks) < self.window_size:
            logger.error(f"Need at least {self.window_size} recent tracks for recommendation")
            return pd.DataFrame()
        
        # Lấy chuỗi gần đây nhất
        recent_window = tuple(recent_tracks[-self.window_size:])
        
        # Nếu chuỗi này chưa xuất hiện trong dữ liệu huấn luyện, thử các cửa sổ nhỏ hơn
        candidates = Counter()
        
        if recent_window in self.track_sequences:
            candidates = self.track_sequences[recent_window].copy()
        else:
            # Thử với cửa sổ ngắn hơn
            for window_size in range(self.window_size-1, 0, -1):
                smaller_window = tuple(recent_tracks[-window_size:])
                
                # Tìm tất cả các cửa sổ bắt đầu bằng smaller_window
                for window, next_tracks in self.track_sequences.items():
                    if window[:window_size] == smaller_window:
                        # Thêm vào với trọng số giảm dần theo kích thước cửa sổ
                        weight = window_size / self.window_size
                        for track, count in next_tracks.items():
                            candidates[track] += count * weight
                
                if candidates:
                    break
        
        # Loại bỏ các bài hát đã xuất hiện trong chuỗi gần đây
        for track in recent_tracks:
            if track in candidates:
                del candidates[track]
        
        # Nếu không tìm thấy ứng viên, trả về DataFrame rỗng
        if not candidates:
            logger.warning("No recommendations found for the given sequence")
            return pd.DataFrame()
        
        # Lấy top N bài hát với số lần xuất hiện cao nhất
        top_tracks = candidates.most_common(n_recommendations)
        
        # Tạo DataFrame kết quả
        recommendations = pd.DataFrame({
            'id': [track for track, _ in top_tracks],
            'count': [count for _, count in top_tracks]
        })
        
        # Chuẩn hóa điểm số
        recommendations['sequence_score'] = recommendations['count'] / recommendations['count'].max()
        
        # Thêm tên bài hát và nghệ sĩ nếu có
        if self.track_id_to_name:
            recommendations['name'] = recommendations['id'].map(self.track_id_to_name)
            recommendations['artist'] = recommendations['id'].map(self.track_id_to_artist)
        
        # Log metrics
        self._log_recommendation_metrics(recent_tracks, recommendations)
        
        # Trả về kết quả
        return recommendations[['id', 'name', 'artist', 'sequence_score']] if 'name' in recommendations.columns else recommendations
    
    def _log_recommendation_metrics(self, input_sequence, recommendations):
        """Log thông tin về đề xuất"""
        logger.info(f"Generating recommendations for sequence ending with track: {input_sequence[-1]}")
        logger.info(f"Generated {len(recommendations)} recommendations")
        if not recommendations.empty and 'sequence_score' in recommendations.columns:
            logger.info(f"Average sequence score: {recommendations['sequence_score'].mean():.4f}")
            logger.info(f"Max sequence score: {recommendations['sequence_score'].max():.4f}")