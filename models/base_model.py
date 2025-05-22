import pandas as pd
import numpy as np
import os
import pickle
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from config.config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRecommender(ABC):
    """Base abstract class for all recommendation models"""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.is_trained = False
        self.train_time = None
    
    @abstractmethod
    def train(self, data):
        """Train the model with the given data"""
        pass
    
    def recommend(self, track_name=None, track_id=None, artist=None, n_recommendations=10):
        """Recommend tracks similar to the input track with stable behavior"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = min(n_recommendations, len(self.tracks_df) - 1)
        
        # Find track index
        track_idx = self._find_track_index(track_id, track_name, artist)
        
        # If track not found, return random recommendations as fallback
        if track_idx is None:
            logger.warning(f"Track not found: '{track_name}' by {artist}. Returning random recommendations.")
            # Đặt seed cố định cho random để đảm bảo ổn định giữa các lần gọi với cùng input
            seed = 0
            if track_name:
                # Tạo seed từ tên bài hát để duy trì tính ngẫu nhiên giữa các bài hát khác nhau
                # nhưng đảm bảo ổn định cho cùng 1 bài
                seed = sum(ord(c) for c in track_name)
            np.random.seed(seed)
            
            # Return random tracks as fallback
            sample_indices = np.random.choice(
                len(self.tracks_df), 
                size=min(n_recommendations, len(self.tracks_df)),
                replace=False
            )
            recommendations = self.tracks_df.iloc[sample_indices][['id', 'name', 'artist']].copy()
            recommendations['content_score'] = 0.5  # Baseline score
            return recommendations
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[track_idx]))
        
        # Sort by similarity score first, then by index for stability when scores are equal
        sim_scores = sorted(sim_scores, key=lambda x: (x[1], -x[0]), reverse=True)
        
        # Exclude the input track
        sim_scores = [s for s in sim_scores if s[0] != track_idx][:n_recommendations]
        
        # Get track indices
        track_indices = [i[0] for i in sim_scores]
        
        # Get recommendations with similarity scores
        recommendations = self.tracks_df.iloc[track_indices][['id', 'name', 'artist']].copy()
        recommendations['content_score'] = [i[1] for i in sim_scores]
        
        # Log metrics
        avg_score = recommendations['content_score'].mean()
        logger.info(f"Generated {len(recommendations)} recommendations for '{track_name}' by {artist}")
        logger.info(f"Average content score: {avg_score:.4f}")
        
        return recommendations
    
    def save(self, filepath=None):
        """Save the model to a file"""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, f"{self.name.lower()}.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'wb') as f:
                # Loại bỏ các thuộc tính quá lớn hoặc không thể pickle nếu cần
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, filepath):
        """Load a model from a file"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def _validate_n_recommendations(self, n):
        """Validate and return a valid number of recommendations"""
        try:
            n = int(n)
            return max(1, n)  # At least 1 recommendation
        except (ValueError, TypeError):
            logger.warning(f"Invalid n_recommendations value: {n}. Using default value 10.")
            return 10
            
    def _log_recommendation_metrics(self, input_data, recommendations):
        """Log metrics about the recommendations (to be implemented by child classes)"""
        pass