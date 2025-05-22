import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from models.base_model import BaseRecommender
from config.config import CONTENT_FEATURES, PROCESSED_DATA_DIR, TOP_K_SIMILAR_ITEMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentBasedRecommender(BaseRecommender):
    """Content-based recommendation model using available track features"""
    
    def __init__(self):
        super().__init__(name="ContentBasedRecommender")
        self.tracks_df = None
        self.features_matrix = None
        self.similarity_matrix = None
        self.track_indices = {}  # Map track_id to index in similarity matrix
    
    def train(self, tracks_df=None):
        """Train the model using track features"""
        start_time = datetime.now()
        
        # Load data if not provided
        if tracks_df is None:
            tracks_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
            if os.path.exists(tracks_path):
                tracks_df = pd.read_csv(tracks_path)
                logger.info(f"Loaded {len(tracks_df)} tracks from {tracks_path}")
            else:
                logger.error(f"Track features file not found: {tracks_path}")
                return False
        
        self.tracks_df = tracks_df
        
        # Create track_id to index mapping
        self.track_indices = {track_id: i for i, track_id in enumerate(self.tracks_df['id'])}
        
        # Determine which features to use
        available_features = [f for f in CONTENT_FEATURES if f in self.tracks_df.columns]
        
        if not available_features:
            logger.warning("No audio features found. Using simple one-hot encoding as fallback.")
            # Create a simple one-hot encoding of tracks as fallback
            num_tracks = len(self.tracks_df)
            self.features_matrix = np.eye(num_tracks)
        else:
            # Fill missing values if any
            for feature in available_features:
                if self.tracks_df[feature].isnull().any():
                    self.tracks_df[feature] = self.tracks_df[feature].fillna(self.tracks_df[feature].median())
            
            # Extract features matrix
            self.features_matrix = self.tracks_df[available_features].values
        
        # Compute similarity matrix
        logger.info("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.features_matrix)
        
        self.train_time = datetime.now() - start_time
        self.is_trained = True
        
        logger.info(f"Content-based model trained in {self.train_time}")
        return True
    
    def recommend(self, track_name=None, track_id=None, artist=None, n_recommendations=10):
        """Recommend tracks similar to the input track with enhanced error handling"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = self._validate_n_recommendations(n_recommendations)
        
        # Find track index
        track_idx = self._find_track_index(track_id, track_name, artist)
        
        # If track not found, return random recommendations as fallback
        if track_idx is None:
            logger.warning(f"Track not found: '{track_name}' by {artist}. Returning random recommendations.")
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
        
        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude the input track
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get track indices
        track_indices = [i[0] for i in sim_scores]
        
        # Get recommendations with similarity scores
        recommendations = self.tracks_df.iloc[track_indices][['id', 'name', 'artist']].copy()
        recommendations['content_score'] = [i[1] for i in sim_scores]
        
        # Log metrics
        self._log_recommendation_metrics(track_idx, recommendations)
        
        return recommendations
    
    def _find_track_index(self, track_id=None, track_name=None, artist=None):
        """Find the index of a track by ID, name, or artist"""
        if track_id is not None and track_id in self.track_indices:
            return self.track_indices[track_id]
        
        if track_name is not None:
            # Case-insensitive search
            matches = self.tracks_df[self.tracks_df['name'].str.lower() == track_name.lower()]
            
            # If artist is provided, filter by artist
            if artist is not None and not matches.empty:
                artist_matches = matches[matches['artist'].str.lower() == artist.lower()]
                if not artist_matches.empty:
                    matches = artist_matches
            
            if not matches.empty:
                return self.track_indices[matches.iloc[0]['id']]
            
            # Try partial match if no exact match
            matches = self.tracks_df[self.tracks_df['name'].str.lower().str.contains(track_name.lower())]
            
            # If artist is provided, filter by artist
            if artist is not None and not matches.empty:
                artist_matches = matches[matches['artist'].str.lower() == artist.lower()]
                if not artist_matches.empty:
                    matches = artist_matches
            
            if not matches.empty:
                logger.info(f"No exact match for '{track_name}'. Using closest match: '{matches.iloc[0]['name']}' by {matches.iloc[0]['artist']}")
                return self.track_indices[matches.iloc[0]['id']]
        
        logger.error(f"Track not found with id={track_id}, name={track_name}, artist={artist}")
        return None
    
    def _log_recommendation_metrics(self, track_idx, recommendations):
        """Log metrics about the recommendations"""
        input_track = self.tracks_df.iloc[track_idx]
        logger.info(f"Generated {len(recommendations)} recommendations for '{input_track['name']}' by {input_track['artist']}")
        if 'content_score' in recommendations.columns:
            logger.info(f"Average content score: {recommendations['content_score'].mean():.4f}")