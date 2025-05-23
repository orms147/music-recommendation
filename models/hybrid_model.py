import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from models.base_model import BaseRecommender
from models.content_model import ContentBasedRecommender
from config.config import CONTENT_WEIGHT

logger = logging.getLogger(__name__)

class MetadataRecommender(BaseRecommender):
    """Simplified recommendation model based primarily on metadata"""
    
    def __init__(self):
        super().__init__(name="MetadataRecommender")
        self.content_recommender = ContentBasedRecommender()
        self.tracks_df = None
    
    def train(self, tracks_df, user_item_matrix=None):
        """Train model components"""
        start_time = datetime.now()
        
        # Save tracks_df for use during recommendation
        self.tracks_df = tracks_df
        
        logger.info(f"Columns in tracks_df: {self.tracks_df.columns}")
        
        # Train content-based model
        logger.info("Training content-based recommender...")
        self.content_recommender.train(tracks_df)
        
        self.train_time = datetime.now() - start_time
        logger.info(f"Metadata model trained in {self.train_time.total_seconds():.2f} seconds")
        
        self.is_trained = True
        return True
    
    def recommend(self, track_name=None, artist=None, n_recommendations=10):
        """Generate track recommendations"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = min(n_recommendations, len(self.tracks_df) - 1) if self.tracks_df is not None else n_recommendations
        
        # Content-based recommendations
        if track_name is not None:
            try:
                recommendations = self.content_recommender.recommend(
                    track_name=track_name, 
                    artist=artist, 
                    n_recommendations=n_recommendations
                )
                
                if not recommendations.empty:
                    return recommendations
            except Exception as e:
                logger.error(f"Error generating content-based recommendations: {e}")
        
        # Fallback to random recommendations if nothing found
        logger.warning("No recommendations were generated, using fallback")
        if hasattr(self, 'tracks_df') and self.tracks_df is not None:
            sample_size = min(n_recommendations, len(self.tracks_df))
            random_tracks = self.tracks_df.sample(sample_size)
            random_tracks['content_score'] = 0.5  # Medium confidence
            return random_tracks[['id', 'name', 'artist', 'content_score']]
        else:
            return pd.DataFrame()
    
    def generate_playlist_from_seed(self, seed_track, seed_artist="", n_recommendations=10):
        """Generate a playlist from a seed track"""
        try:
            # Tìm kiếm bài hát ban đầu
            track_idx = self.content_recommender._find_track_index(track_name=seed_track, artist=seed_artist)
            
            if track_idx is None:
                logger.warning(f"Seed track '{seed_track}' not found")
                return None, None
                
            seed_id = self.tracks_df.iloc[track_idx]['id']
            
            # Tạo queue với số lượng bài hát đề xuất
            track_ids = self.content_recommender.recommend_queue([seed_id], n_recommendations)
            
            logger.info(f"Columns in tracks_df: {self.tracks_df.columns}")
            if self.tracks_df.empty:
                logger.error("tracks_df is empty!")
                return None, None
            if 'id' not in self.tracks_df.columns:
                logger.error(f"DataFrame columns: {self.tracks_df.columns}")
                raise ValueError("DataFrame không có cột 'id'.")
            
            queue = self.tracks_df[self.tracks_df['id'].isin(track_ids)].copy()
            logger.info(f"Queue shape: {queue.shape}")
            if queue.empty:
                logger.warning("Queue is empty after filtering by track_ids.")
                return None, None
            
            # Đảm bảo thứ tự trong queue
            queue['order'] = queue['id'].apply(lambda x: track_ids.index(x) if x in track_ids else 999)
            queue = queue.sort_values('order').drop('order', axis=1)
            
            # Tạo phân tích - Thay thế bằng thông tin về nghệ sĩ và thể loại
            analysis = self._analyze_artist_genres(queue)
            
            return queue, analysis
        except Exception as e:
            logger.error(f"Error generating playlist: {e}")
            return None, None
            
    def _analyze_artist_genres(self, playlist_df):
        """Tạo phân tích về phân bố thể loại và nghệ sĩ trong playlist"""
        analysis = []
        
        # Kiểm tra nếu playlist rỗng
        if playlist_df is None or playlist_df.empty or len(playlist_df) < 2:
            return pd.DataFrame()
            
        # Tính phân bố nghệ sĩ
        artist_counts = playlist_df['artist'].value_counts()
        total_tracks = len(playlist_df)
        
        # Xây dựng thông tin transition giữa các bài hát
        for i in range(len(playlist_df) - 1):
            current = playlist_df.iloc[i]
            next_track = playlist_df.iloc[i+1]
            
            # Xác định chất lượng transition dựa trên tính đa dạng
            quality = "Good"
            
            # Nếu cùng nghệ sĩ - đánh giá trung bình
            if current['artist'] == next_track['artist']:
                quality = "Fair"
                
            # Nếu quá nhiều bài từ cùng nghệ sĩ trong playlist - đánh giá thấp
            if artist_counts[current['artist']] > total_tracks / 2:
                quality = "Poor"
                
            # Nếu khác nghệ sĩ - đánh giá cao
            if current['artist'] != next_track['artist']:
                quality = "Excellent"
                
            # Tính điểm transition dựa trên metadata
            transition_score = 0.7  # Điểm mặc định
            
            # Tăng điểm nếu có tính đa dạng về nghệ sĩ
            if current['artist'] != next_track['artist']:
                transition_score += 0.2
                
            # Điều chỉnh điểm dựa trên chất lượng
            if quality == "Excellent":
                transition_score = min(transition_score + 0.2, 1.0)
            elif quality == "Poor":
                transition_score = max(transition_score - 0.3, 0.0)
            
            # Thêm vào kết quả
            from_track = f"{current['name']} - {current['artist']}"
            to_track = f"{next_track['name']} - {next_track['artist']}"
            
            analysis.append({
                'from_track': from_track,
                'to_track': to_track,
                'transition_score': transition_score,
                'quality': quality
            })
            
        return pd.DataFrame(analysis)
    
    def discover_by_genre(self, genre, n_recommendations=10):
        """Khám phá bài hát theo thể loại"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
            
        # Kiểm tra nếu có cột thể loại
        genre_col = None
        for col in self.tracks_df.columns:
            if f'genre_{genre.lower().replace(" ", "_")}' == col:
                genre_col = col
                break
        
        # Nếu có cột thể loại, lọc theo giá trị
        if genre_col is not None:
            filtered = self.tracks_df[self.tracks_df[genre_col] > 0]
        else:
            # Tìm thể loại trong cột artist_genres nếu có
            if 'artist_genres' in self.tracks_df.columns:
                genre_keyword = genre.lower()
                filtered = self.tracks_df[self.tracks_df['artist_genres'].str.contains(
                    genre_keyword, case=False, na=False)]
            else:
                # Tìm kiếm trong tên bài hát và nghệ sĩ
                filtered = self.tracks_df[
                    self.tracks_df['name'].str.contains(genre, case=False, na=False) |
                    self.tracks_df['artist'].str.contains(genre, case=False, na=False)
                ]
        
        if filtered.empty:
            logger.warning(f"No tracks found for genre: {genre}")
            return pd.DataFrame()
            
        # Sắp xếp theo độ phổ biến
        if 'popularity' in filtered.columns:
            filtered = filtered.sort_values('popularity', ascending=False)
            
        # Lấy số lượng khuyến nghị
        result = filtered.head(n_recommendations)
        result['content_score'] = 1.0  # Điểm cao vì khớp chính xác thể loại
        
        return result[['id', 'name', 'artist', 'content_score']]