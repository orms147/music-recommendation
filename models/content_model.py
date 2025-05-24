import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from models.base_model import BaseRecommender
from config.config import CONTENT_FEATURES
from scipy import sparse

logger = logging.getLogger(__name__)

class ContentBasedRecommender(BaseRecommender):
    """Baseline Content-based recommendation model using metadata features"""
    
    def __init__(self):
        super().__init__(name="ContentBasedRecommender")
        self.feature_columns = []
        self.track_indices = {}
    
    def train(self, tracks_df=None):
        """Train the model using available real metadata features"""
        start_time = datetime.now()
        
        if tracks_df is None:
            logger.error("No tracks data provided for training")
            return False
            
        self.tracks_df = tracks_df.copy()
        logger.info(f"Training ContentBasedRecommender with {len(self.tracks_df)} tracks")
        
        # Prepare feature matrix
        success = self._prepare_features()
        if not success:
            return False
        
        # Calculate similarity matrix
        self._calculate_similarity_matrix()
        
        # Create track indices mapping
        self._create_track_indices()
        
        self.train_time = datetime.now() - start_time
        logger.info(f"ContentBasedRecommender trained successfully in {self.train_time.total_seconds():.2f} seconds")
        
        self.is_trained = True
        return True
    
    def _prepare_features(self):
        """Prepare features for similarity calculation"""
        try:
            # Define available feature categories
            base_features = [
                'popularity', 'explicit', 'release_year', 'duration_ms',
                'total_tracks', 'track_number', 'disc_number', 'markets_count'
            ]
            
            derived_features = [
                'duration_min', 'artist_frequency', 'name_length',
                'has_collab', 'is_remix', 'is_vietnamese', 'is_korean', 
                'is_japanese', 'is_spanish'
            ]
            
            # Find genre features
            genre_features = [col for col in self.tracks_df.columns if col.startswith('genre_')]
            
            # Combine all potential features
            all_candidate_features = base_features + derived_features + genre_features
            
            # Filter to only existing columns
            self.feature_columns = [f for f in all_candidate_features if f in self.tracks_df.columns]
            
            if len(self.feature_columns) < 3:
                logger.error(f"Insufficient features for training. Only {len(self.feature_columns)} available")
                return False
            
            logger.info(f"Using {len(self.feature_columns)} features: {self.feature_columns}")
            
            # Create feature matrix
            feature_df = self.tracks_df[self.feature_columns].copy()
            
            # Handle missing values intelligently
            for col in feature_df.columns:
                if feature_df[col].dtype in ['int64', 'float64']:
                    if col in ['popularity', 'artist_popularity']:
                        feature_df[col] = feature_df[col].fillna(0)
                    elif col.startswith('genre_') or col.startswith('is_'):
                        feature_df[col] = feature_df[col].fillna(0)
                    else:
                        feature_df[col] = feature_df[col].fillna(feature_df[col].median())
                else:
                    feature_df[col] = feature_df[col].fillna(0)
            
            # Store feature matrix
            self.feature_matrix = feature_df.values
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return False
    
    def _calculate_similarity_matrix(self):
        """Calculate cosine similarity matrix"""
        try:
            logger.info("Calculating similarity matrix...")
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
            logger.info(f"Similarity matrix calculated: {self.similarity_matrix.shape}")
        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {e}")
            self.similarity_matrix = None
    
    def _create_track_indices(self):
        """Create mapping from track ID to DataFrame index"""
        if 'id' in self.tracks_df.columns:
            self.track_indices = {track_id: i for i, track_id in enumerate(self.tracks_df['id'])}
        else:
            self.track_indices = {i: i for i in range(len(self.tracks_df))}
        
        logger.info(f"Created track indices mapping for {len(self.track_indices)} tracks")
    
    def recommend(self, track_name=None, track_id=None, artist=None, n_recommendations=10):
        """Recommend tracks similar to the input track"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = self._validate_n_recommendations(n_recommendations)
        
        # Find track index
        track_idx = self._find_track_index(track_id, track_name, artist)
        
        # If track not found, return fallback recommendations
        if track_idx is None:
            logger.warning(f"Track not found: '{track_name}' by {artist}. Using fallback recommendations.")
            recommendations = self._create_fallback_recommendations(track_name, n_recommendations)
            self._log_recommendation_quality(recommendations, "fallback")
            return recommendations
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[track_idx]))
        
        # Sort by similarity score (desc) then by index (asc) for stability
        sim_scores = sorted(sim_scores, key=lambda x: (x[1], -x[0]), reverse=True)
        
        # Exclude the input track and take top N
        sim_scores = [s for s in sim_scores if s[0] != track_idx][:n_recommendations]
        
        # Get track indices
        track_indices = [i[0] for i in sim_scores]
        
        # Create recommendations DataFrame
        base_cols = ['name', 'artist']
        if 'id' in self.tracks_df.columns:
            base_cols.insert(0, 'id')
        
        recommendations = self.tracks_df.iloc[track_indices][base_cols].copy()
        recommendations['content_score'] = [i[1] for i in sim_scores]
        
        # Add additional metadata columns
        additional_cols = ['popularity', 'release_year', 'artist_popularity']
        for col in additional_cols:
            if col in self.tracks_df.columns:
                recommendations[col] = self.tracks_df.iloc[track_indices][col].values
        
        # Log quality metrics
        seed_track = self.tracks_df.iloc[track_idx]
        logger.info(f"Found recommendations for: '{seed_track['name']}' by {seed_track['artist']}")
        self._log_recommendation_quality(recommendations, "content_similarity")
        
        return recommendations
    
    def recommend_queue(self, seed_tracks, n_total=10):
        """Generate a queue starting from seed tracks"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return []
        
        # Process seed tracks to get IDs
        seed_ids = []
        for track in seed_tracks:
            if isinstance(track, dict) and 'id' in track:
                seed_ids.append(track['id'])
            elif isinstance(track, str):
                # Try to find by ID or name
                if track in self.track_indices:
                    seed_ids.append(track)
                else:
                    idx = self._find_track_index(track_name=track)
                    if idx is not None and 'id' in self.tracks_df.columns:
                        seed_ids.append(self.tracks_df.iloc[idx]['id'])
        
        if not seed_ids:
            logger.warning("No valid seed tracks found. Returning random tracks.")
            # Return random track IDs
            random_indices = np.random.choice(len(self.tracks_df), size=min(n_total, len(self.tracks_df)), replace=False)
            if 'id' in self.tracks_df.columns:
                return self.tracks_df.iloc[random_indices]['id'].tolist()
            else:
                return random_indices.tolist()
        
        # Calculate how many more tracks we need
        n_needed = max(0, n_total - len(seed_ids))
        
        if n_needed == 0:
            return seed_ids
        
        # Get recommendations for each seed track
        all_recommendations = []
        
        for seed_id in seed_ids:
            try:
                recs = self.recommend(track_id=seed_id, n_recommendations=n_needed)
                if not recs.empty:
                    all_recommendations.append(recs)
            except Exception as e:
                logger.warning(f"Failed to get recommendations for seed {seed_id}: {e}")
                continue
        
        # Combine and deduplicate recommendations
        if all_recommendations:
            combined = pd.concat(all_recommendations, ignore_index=True)
            combined = combined.sort_values('content_score', ascending=False)
            
            # Remove duplicates and seeds
            if 'id' in combined.columns:
                combined = combined.drop_duplicates(subset=['id'])
                combined = combined[~combined['id'].isin(seed_ids)]
                
                # Take top recommendations
                top_recs = combined.head(n_needed)
                final_ids = seed_ids + top_recs['id'].tolist()
            else:
                # Fallback if no ID column
                combined = combined.drop_duplicates(subset=['name', 'artist'])
                top_recs = combined.head(n_needed)
                final_ids = seed_ids + top_recs.index.tolist()
            
            return final_ids
        
        return seed_ids
    
    def explore_by_genre(self, genre, n_recommendations=10):
        """Explore tracks by genre"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = self._validate_n_recommendations(n_recommendations)
        
        # Look for genre-specific column
        genre_col = f'genre_{genre.lower().replace(" ", "_")}'
        
        if genre_col in self.tracks_df.columns:
            # Filter by genre column
            filtered = self.tracks_df[self.tracks_df[genre_col] > 0]
            logger.info(f"Found {len(filtered)} tracks with genre column '{genre_col}'")
        elif 'artist_genres' in self.tracks_df.columns:
            # Search in artist_genres text
            filtered = self.tracks_df[self.tracks_df['artist_genres'].str.contains(
                genre, case=False, na=False, regex=False)]
            logger.info(f"Found {len(filtered)} tracks with genre in artist_genres")
        else:
            # Search in track/artist names
            filtered = self.tracks_df[
                self.tracks_df['name'].str.contains(genre, case=False, na=False, regex=False) |
                self.tracks_df['artist'].str.contains(genre, case=False, na=False, regex=False)
            ]
            logger.info(f"Found {len(filtered)} tracks with genre in names")
        
        if filtered.empty:
            logger.warning(f"No tracks found for genre: {genre}")
            return pd.DataFrame()
        
        # Sort by popularity if available, otherwise random
        if 'popularity' in filtered.columns:
            result = filtered.nlargest(n_recommendations, 'popularity')
        else:
            result = filtered.sample(min(n_recommendations, len(filtered)))
        
        # Add content score for consistency
        result = result.copy()
        result['content_score'] = 1.0  # High score for exact genre match
        
        # Log and return
        self._log_recommendation_quality(result, f"genre_exploration_{genre}")
        
        base_cols = ['name', 'artist']
        if 'id' in result.columns:
            base_cols.insert(0, 'id')
        base_cols.append('content_score')
        
        additional_cols = ['popularity', 'release_year']
        for col in additional_cols:
            if col in result.columns:
                base_cols.append(col)
        
        return result[base_cols]
    
    def __getstate__(self):
        """Optimize pickling by handling large matrices"""
        state = self.__dict__.copy()
        
        # Convert similarity matrix to sparse format if beneficial
        if 'similarity_matrix' in state and state['similarity_matrix'] is not None:
            if not sparse.issparse(state['similarity_matrix']):
                density = np.count_nonzero(state['similarity_matrix']) / state['similarity_matrix'].size
                if density < 0.5:  # Convert to sparse if less than 50% density
                    state['similarity_matrix'] = sparse.csr_matrix(state['similarity_matrix'])
        
        return state
    
    def __setstate__(self, state):
        """Handle unpickling of sparse matrices"""
        self.__dict__.update(state)
        
        # Convert sparse matrix back to dense if needed
        if hasattr(self, 'similarity_matrix') and self.similarity_matrix is not None:
            if sparse.issparse(self.similarity_matrix):
                self.similarity_matrix = self.similarity_matrix.toarray()