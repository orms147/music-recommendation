import pandas as pd
import numpy as np
import os
import pickle
import logging
import traceback
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
        """Recommend tracks similar to the input track using real metadata"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = min(n_recommendations, len(self.tracks_df) - 1)
        
        # Find track index
        track_idx = self._find_track_index(track_id, track_name, artist)
        
        # If track not found, return random recommendations as fallback
        if track_idx is None:
            logger.warning(f"Track not found: '{track_name}' by {artist}. Returning random recommendations with real metadata.")
            # Đặt seed cố định cho random để đảm bảo ổn định
            seed = 0
            if track_name:
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
            
            # Thêm real metadata columns nếu có
            for col in ['popularity', 'artist_popularity', 'release_year']:
                if col in self.tracks_df.columns:
                    recommendations[col] = self.tracks_df.iloc[sample_indices][col].values
            
            return recommendations
        
        # Get similarity scores using real metadata features
        sim_scores = list(enumerate(self.similarity_matrix[track_idx]))
        
        # Sort by similarity score first, then by index for stability
        sim_scores = sorted(sim_scores, key=lambda x: (x[1], -x[0]), reverse=True)
        
        # Exclude the input track
        sim_scores = [s for s in sim_scores if s[0] != track_idx][:n_recommendations]
        
        # Get track indices
        track_indices = [i[0] for i in sim_scores]
        
        # Get recommendations with similarity scores
        recommendations = self.tracks_df.iloc[track_indices][['id', 'name', 'artist']].copy()
        recommendations['content_score'] = [i[1] for i in sim_scores]
        
        # Thêm real metadata columns
        for col in ['popularity', 'artist_popularity', 'release_year']:
            if col in self.tracks_df.columns:
                recommendations[col] = self.tracks_df.iloc[track_indices][col].values
        
        # Log metrics
        avg_score = recommendations['content_score'].mean()
        logger.info(f"Generated {len(recommendations)} real metadata-based recommendations for '{track_name}'")
        logger.info(f"Average similarity score: {avg_score:.4f}")
        
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
            logger.error(f"Error saving model to {filepath}: {e}\n{traceback.format_exc()}")
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
            logger.error(f"Error loading model from {filepath}: {e}\n{traceback.format_exc()}")
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