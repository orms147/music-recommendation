import pandas as pd
import numpy as np
import os
import pickle
import logging
import re
import unicodedata
from config.config import MODELS_DIR

logger = logging.getLogger(__name__)

class BaseRecommender:
    """Base class for all recommendation models with common functionality"""
    
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.is_trained = False
        self.train_time = None
        self.tracks_df = None
        self.similarity_matrix = None
    
    def _normalize_text(self, text):
        """Normalize text for better matching"""
        if pd.isna(text) or text is None:
            return ""
        # Cải thiện: Xử lý dấu câu và ký tự đặc biệt
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        # Cải thiện: Xử lý các ký tự Unicode đặc biệt
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        return text
    
    def _find_track_index(self, track_id=None, track_name=None, artist=None):
        """Find track index with consistent behavior across all models"""
        if self.tracks_df is None:
            return None
            
        temp_df = self.tracks_df.copy()
        temp_df['name_norm'] = temp_df['name'].apply(self._normalize_text)
        temp_df['artist_norm'] = temp_df['artist'].apply(self._normalize_text)

        # Search by ID first (if provided)
        if track_id is not None and 'id' in temp_df.columns:
            matches = temp_df[temp_df['id'] == track_id].index.tolist()
            if matches:
                return matches[0]

        # Search by exact name and artist
        if track_name is not None:
            track_name_norm = self._normalize_text(track_name)
            matches = temp_df[temp_df['name_norm'] == track_name_norm]
            
            if artist is not None and not matches.empty:
                artist_norm = self._normalize_text(artist)
                matches = matches[matches['artist_norm'] == artist_norm]
            
            if not matches.empty:
                return matches.index[0]
            
            # Partial search as fallback
            partial_matches = temp_df[temp_df['name_norm'].str.contains(track_name_norm, regex=False, na=False)]
            if artist is not None and not partial_matches.empty:
                artist_norm = self._normalize_text(artist)
                partial_matches = partial_matches[partial_matches['artist_norm'].str.contains(artist_norm, regex=False, na=False)]
            
            if not partial_matches.empty:
                return partial_matches.index[0]
        
        return None
    
    def _create_fallback_recommendations(self, track_name, n_recommendations):
        """Create stable fallback recommendations when track not found"""
        if self.tracks_df is None or len(self.tracks_df) == 0:
            return pd.DataFrame()
        
        # Create deterministic seed from track name for stable results
        seed = sum(ord(c) for c in (track_name or "fallback"))
        np.random.seed(seed)
        
        # Sample tracks
        sample_size = min(n_recommendations, len(self.tracks_df))
        sample_indices = np.random.choice(
            len(self.tracks_df), 
            size=sample_size,
            replace=False
        )
        
        # Create recommendations DataFrame
        base_cols = ['name', 'artist']
        if 'id' in self.tracks_df.columns:
            base_cols.insert(0, 'id')
        
        recommendations = self.tracks_df.iloc[sample_indices][base_cols].copy()
        
        # Use enhanced_score for consistency across all models
        recommendations['enhanced_score'] = 0.5  # Medium confidence for fallback
        
        # Add additional metadata if available
        additional_cols = ['popularity', 'release_year', 'artist_popularity']
        for col in additional_cols:
            if col in self.tracks_df.columns:
                recommendations[col] = self.tracks_df.iloc[sample_indices][col].values
        
        return recommendations
    
    def _log_recommendation_quality(self, recommendations, method="standard"):
        """Log basic quality metrics for recommendations"""
        if recommendations.empty:
            logger.warning(f"Empty recommendations from method: {method}")
            return
        
        # Basic metrics
        total_recs = len(recommendations)
        unique_artists = recommendations['artist'].nunique() if 'artist' in recommendations.columns else 0
        artist_diversity = unique_artists / total_recs if total_recs > 0 else 0
        
        # Score metrics - check both possible score column names
        score_col = None
        if 'enhanced_score' in recommendations.columns:
            score_col = 'enhanced_score'
        elif 'content_score' in recommendations.columns:
            score_col = 'content_score'
        elif 'final_score' in recommendations.columns:
            score_col = 'final_score'
        
        if score_col:
            avg_score = recommendations[score_col].mean()
            min_score = recommendations[score_col].min()
            max_score = recommendations[score_col].max()
            logger.info(f"[{method}] Recommendations: {total_recs}, Artist diversity: {artist_diversity:.3f}, "
                       f"Score range: {min_score:.3f}-{max_score:.3f} (avg: {avg_score:.3f})")
        else:
            logger.info(f"[{method}] Recommendations: {total_recs}, Artist diversity: {artist_diversity:.3f}")
    
    def save(self, filepath=None):
        """Save the model to a file"""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, f"{self.name.lower()}.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {e}")
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
            logger.error(f"Error loading model from {filepath}: {e}")
            return None
    
    def _validate_n_recommendations(self, n):
        """Validate and return a valid number of recommendations"""
        try:
            n = int(n)
            return max(1, min(n, len(self.tracks_df) - 1 if self.tracks_df is not None else 100))
        except (ValueError, TypeError):
            logger.warning(f"Invalid n_recommendations value: {n}. Using default value 10.")
            return 10
    
    def _validate_track_exists(self, track_name, artist=None):
        """Validate if track exists in dataset with better error handling"""
        if self.tracks_df is None:
            return False, "Model not trained with data"
        
        track_idx = self._find_track_index(track_name=track_name, artist=artist)
        if track_idx is not None:
            return True, track_idx
        
        # Cải thiện: Trả về gợi ý bài hát tương tự
        suggestions = self._get_similar_track_names(track_name)
        return False, suggestions
