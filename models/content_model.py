import pandas as pd
import numpy as np
import logging
import os
import unicodedata
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from models.base_model import BaseRecommender
from config.config import CONTENT_FEATURES

logger = logging.getLogger(__name__)

class ContentBasedRecommender(BaseRecommender):
    """Content-based recommendation model using metadata features"""
    
    def __init__(self):
        super().__init__()
        self.tracks_df = None
        self.similarity_matrix = None
        self.is_trained = False

    def train(self, tracks_df=None):
        """Train the model using track features"""
        start_time = datetime.now()
        
        if tracks_df is None:
            logger.error("No tracks data provided for training")
            return False
            
        self.tracks_df = tracks_df.copy()
        
        # Metadata-focused features
        metadata_features = [
            # Các đặc trưng cơ bản
            'popularity', 'explicit', 'release_year', 'decade', 
            'duration_min', 'artist_popularity', 'artist_frequency',
            
            # Đặc trưng chuyển đổi 
            'is_vietnamese', 'is_korean', 'is_japanese', 'is_spanish',
            'has_collab', 'is_remix',
            
            # Thể loại
            'genre_pop', 'genre_rock', 'genre_hip_hop', 'genre_rap', 
            'genre_electronic', 'genre_dance', 'genre_r&b', 'genre_indie', 
            'genre_classical', 'genre_jazz', 'genre_country', 'genre_folk', 
            'genre_metal', 'genre_blues'
        ]
        
        # Lọc các đặc trưng có sẵn
        available_features = [f for f in metadata_features if f in self.tracks_df.columns]
        
        # Thêm các đặc trưng audio tổng hợp nếu có
        synthetic_audio = [
            'energy', 'danceability', 'valence', 'acousticness', 
            'instrumentalness', 'liveness', 'speechiness'
        ]
        
        available_features.extend([f for f in synthetic_audio if f in self.tracks_df.columns])
        
        if not available_features:
            logger.error("No valid features found for content-based recommendation")
            return False
            
        logger.info(f"Using features: {available_features}")
        
        # Tạo feature matrix
        feature_matrix = self.tracks_df[available_features].fillna(0).values
        
        # Tính toán ma trận tương đồng
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        # Tạo map từ track ID sang index
        self.track_indices = {track_id: i for i, track_id in enumerate(self.tracks_df['id'])}
        
        self.train_time = datetime.now() - start_time
        logger.info(f"Content-based model trained in {self.train_time.total_seconds():.2f} seconds")
        
        self.is_trained = True
        return True
    
    def _normalize_text(self, text):
        """Normalize text for better matching"""
        if not isinstance(text, str):
            return ""
        return text.lower().strip()
    
    def _find_track_index(self, track_id=None, track_name=None, artist=None):
        """Find track index with stable, consistent behavior"""
        temp_df = self.tracks_df.copy()
        temp_df['name_norm'] = temp_df['name'].apply(self._normalize_text)
        temp_df['artist_norm'] = temp_df['artist'].apply(self._normalize_text)  # Luôn tạo cột này

        # Tìm theo ID
        if track_id is not None and 'id' in temp_df.columns:
            matches = temp_df[temp_df['id'] == track_id].index.tolist()
            if matches:
                return matches[0]

        # Tìm theo tên và nghệ sĩ
        if track_name is not None:
            track_name_norm = self._normalize_text(track_name)
            matches = temp_df[temp_df['name_norm'] == track_name_norm]
            if artist is not None and not matches.empty:
                artist_norm = self._normalize_text(artist)
                matches = matches[matches['artist_norm'] == artist_norm]
            if not matches.empty:
                return matches.index[0]
            # Tìm kiếm một phần
            partial_matches = temp_df[temp_df['name_norm'].str.contains(track_name_norm)]
            if artist is not None and not partial_matches.empty:
                artist_norm = self._normalize_text(artist)
                partial_matches = partial_matches[partial_matches['artist_norm'].str.contains(artist_norm)]
            if not partial_matches.empty:
                return partial_matches.index[0]
        return None
    
    def recommend(self, track_name=None, track_id=None, artist=None, n_recommendations=10):
        """Recommend tracks similar to the input track with enhanced error handling"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return "Model not trained."
        
        n_recommendations = min(n_recommendations, len(self.tracks_df) - 1)
        
        # Find track index
        track_idx = self._find_track_index(track_id, track_name, artist)
        
        # If track not found, return random recommendations as fallback
        if track_idx is None:
            logger.warning(f"Track not found: '{track_name}' by {artist}. Returning random recommendations.")
            # Đặt seed cố định để kết quả fallback luôn giống nhau cho cùng input
            seed = sum(ord(c) for c in (track_name or ""))
            np.random.seed(seed)
            sample_indices = np.random.choice(
                len(self.tracks_df), 
                size=min(n_recommendations, len(self.tracks_df)),
                replace=False
            )
            recommendations = self.tracks_df.iloc[sample_indices][['name', 'artist']].copy()
            recommendations['content_score'] = 0.5
            return recommendations
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[track_idx]))
        
        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: (x[1], -x[0]), reverse=True)
        
        # Exclude the input track
        sim_scores = [s for s in sim_scores if s[0] != track_idx][:n_recommendations]
        
        # Get track indices
        track_indices = [i[0] for i in sim_scores]
        
        # Get recommendations with similarity scores
        recommendations = self.tracks_df.iloc[track_indices][['name', 'artist']].copy()
        recommendations['content_score'] = [i[1] for i in sim_scores]
        
        return recommendations
    
    def recommend_queue(self, seed_tracks, n_total=10):
        """Generate a queue starting from seed tracks"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return []
        
        # Ensure seed_tracks is a list of IDs
        seed_ids = []
        for track in seed_tracks:
            if isinstance(track, dict) and 'id' in track:
                seed_ids.append(track['id'])
            elif isinstance(track, str):
                # Check if it's a track name or ID
                if track in self.track_indices:
                    seed_ids.append(track)
                else:
                    # Try to find by name
                    idx = self._find_track_index(track_name=track)
                    if idx is not None:
                        seed_ids.append(self.tracks_df.iloc[idx]['id'])
        
        if not seed_ids:
            logger.warning("No valid seed tracks found. Returning random tracks.")
            # Return random tracks
            random_indices = np.random.choice(len(self.tracks_df), size=n_total, replace=False)
            random_ids = self.tracks_df.iloc[random_indices]['id'].tolist()
            return random_ids
        
        # How many more tracks we need
        n_needed = max(0, n_total - len(seed_ids))
        
        if n_needed == 0:
            # If we already have enough tracks, return them
            return seed_ids
        
        # Get recommendations for each seed track
        all_recommendations = []
        
        for seed_id in seed_ids:
            # Find similar tracks for this seed
            recs = self.recommend(track_id=seed_id, n_recommendations=n_needed)
            all_recommendations.append(recs)
        
        # Combine recommendations, prioritizing high scores
        if all_recommendations:
            combined = pd.concat(all_recommendations)
            combined = combined.sort_values('content_score', ascending=False)
            combined = combined.drop_duplicates(subset=['id'])
            
            # Remove tracks that are already in seeds
            combined = combined[~combined['id'].isin(seed_ids)]
            
            # Take top n_needed recommendations
            top_recs = combined.head(n_needed)
            
            # Add to seed tracks
            final_ids = seed_ids + top_recs['id'].tolist()
            
            return final_ids
        
        return seed_ids
    
    def __getstate__(self):
        """Customize pickling behavior"""
        state = self.__dict__.copy()
        
        # Lưu thuộc tính cần thiết để phục hồi mô hình
        # Đảm bảo không lưu các thuộc tính quá lớn nếu có thể tái tạo lại
        if 'similarity_matrix' in state and state['similarity_matrix'] is not None:
            # Nếu ma trận tương tự quá lớn, chúng ta có thể lưu ở định dạng tối ưu hơn
            # Chẳng hạn sử dụng scipy.sparse nếu ma trận thưa
            from scipy import sparse
            if sparse.issparse(state['similarity_matrix']):
                pass
            else:
                density = np.count_nonzero(state['similarity_matrix']) / state['similarity_matrix'].size
                if density < 0.5:  # Nếu ma trận thưa (ít hơn 50% phần tử khác 0)
                    state['similarity_matrix'] = sparse.csr_matrix(state['similarity_matrix'])
        
        return state

    def __setstate__(self, state):
        """Customize unpickling behavior"""
        # Khôi phục trạng thái từ pickle
        self.__dict__.update(state)
        
        # Chuyển đổi lại ma trận thưa thành ma trận thường nếu cần
        if 'similarity_matrix' in state and state['similarity_matrix'] is not None:
            from scipy import sparse
            if sparse.issparse(state['similarity_matrix']):
                self.similarity_matrix = state['similarity_matrix'].toarray()