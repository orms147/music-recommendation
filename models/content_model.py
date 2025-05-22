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
    """Content-based recommendation model using song features"""
    
    def __init__(self):
        super().__init__(name="ContentBasedRecommender")
        self.tracks_df = None
        self.similarity_matrix = None
        self.track_indices = None
    
    def train(self, tracks_df=None):
        """Train the model using track features"""
        start_time = datetime.now()
        
        if tracks_df is None:
            logger.error("No tracks data provided for training")
            return False
            
        self.tracks_df = tracks_df.copy()
        
        # Check which features are available
        available_features = [f for f in CONTENT_FEATURES if f in self.tracks_df.columns]
        
        if not available_features:
            logger.error("No valid features found for content-based recommendation")
            return False
            
        logger.info(f"Using features: {available_features}")
        
        # Create feature matrix
        feature_matrix = self.tracks_df[available_features].values
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        # Create mapping from track ID to index
        self.track_indices = {track_id: i for i, track_id in enumerate(self.tracks_df['id'])}
        
        self.train_time = datetime.now() - start_time
        logger.info(f"Content-based model trained in {self.train_time.total_seconds():.2f} seconds")
        
        self.is_trained = True
        return True
    
    def _normalize_text(self, text):
        """Normalize text for better matching"""
        if not text:
            return ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            
        # Normalize Unicode (NFC) for Vietnamese
        return unicodedata.normalize('NFC', text).lower().strip()
    
    def _find_track_index(self, track_id=None, track_name=None, artist=None):
        """Find track index with enhanced error handling"""
        if track_id is not None and track_id in self.track_indices:
            return self.track_indices[track_id]
        
        if track_name is not None:
            # Normalize track name for searching
            track_name_norm = self._normalize_text(track_name)
            
            # 1. Exact match
            self.tracks_df['name_norm'] = self.tracks_df['name'].apply(self._normalize_text)
            matches = self.tracks_df[self.tracks_df['name_norm'] == track_name_norm]
            
            # 2. Filter by artist if provided
            if artist is not None and not matches.empty:
                artist_norm = self._normalize_text(artist)
                self.tracks_df['artist_norm'] = self.tracks_df['artist'].apply(self._normalize_text)
                artist_matches = matches[matches['artist_norm'] == artist_norm]
                if not artist_matches.empty:
                    matches = artist_matches
            
            if not matches.empty:
                track_idx = self.track_indices[matches.iloc[0]['id']]
                # Clean up temporary columns
                self.tracks_df.drop(columns=['name_norm', 'artist_norm'], inplace=True, errors='ignore')
                return track_idx
            
            # 3. Partial matching
            partial_matches = self.tracks_df[self.tracks_df['name_norm'].str.contains(track_name_norm)]
            if artist is not None and not partial_matches.empty:
                artist_norm = self._normalize_text(artist)
                artist_partial = partial_matches[partial_matches['artist_norm'].str.contains(artist_norm)]
                if not artist_partial.empty:
                    partial_matches = artist_partial
            
            if not partial_matches.empty:
                track_idx = self.track_indices[partial_matches.iloc[0]['id']]
                # Clean up temporary columns
                self.tracks_df.drop(columns=['name_norm', 'artist_norm'], inplace=True, errors='ignore')
                return track_idx
            
            # Clean up temporary columns
            self.tracks_df.drop(columns=['name_norm', 'artist_norm'], inplace=True, errors='ignore')
        
        # 4. Find by artist only if no track_name provided
        if track_name is None and artist is not None:
            artist_norm = self._normalize_text(artist)
            self.tracks_df['artist_norm'] = self.tracks_df['artist'].apply(self._normalize_text)
            artist_matches = self.tracks_df[self.tracks_df['artist_norm'] == artist_norm]
            
            if not artist_matches.empty:
                # Return most popular track by artist
                popular_track = artist_matches.sort_values('popularity', ascending=False).iloc[0]
                track_idx = self.track_indices[popular_track['id']]
                
                # Clean up temporary column
                self.tracks_df.drop(columns=['artist_norm'], inplace=True, errors='ignore')
                return track_idx
                
            # Clean up temporary column
            self.tracks_df.drop(columns=['artist_norm'], inplace=True, errors='ignore')
        
        return None
    
    def recommend(self, track_name=None, track_id=None, artist=None, n_recommendations=10):
        """Recommend tracks similar to the input track with enhanced error handling"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = min(n_recommendations, len(self.tracks_df) - 1)
        
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
        avg_score = recommendations['content_score'].mean()
        logger.info(f"Generated {len(recommendations)} recommendations for '{track_name}' by {artist}")
        logger.info(f"Average content score: {avg_score:.4f}")
        
        return recommendations
    
    def compute_transition_score(self, track1_id, track2_id):
        """Compute transition score between two tracks"""
        # Get track indices
        idx1 = self._find_track_index(track_id=track1_id)
        idx2 = self._find_track_index(track_id=track2_id)
        
        if idx1 is None or idx2 is None:
            return 0.0
        
        # Base transition score is their similarity
        base_score = self.similarity_matrix[idx1][idx2]
        
        # Get detailed scores if audio features are available
        audio_features = [
            'tempo', 'key', 'energy', 'danceability', 'valence',
            'loudness', 'mode'
        ]
        
        available_features = [f for f in audio_features if f in self.tracks_df.columns]
        
        if available_features:
            track1 = self.tracks_df.iloc[idx1]
            track2 = self.tracks_df.iloc[idx2]
            
            detailed_score = 0
            weights = {
                'tempo': 0.3,     # Tempo similarity is very important for smooth transitions
                'key': 0.2,       # Key compatibility
                'energy': 0.2,    # Energy level consistency
                'danceability': 0.15,  # Danceability consistency
                'valence': 0.15,  # Emotional tone consistency
                'loudness': 0.1,  # Volume consistency
                'mode': 0.05      # Musical mode (major/minor)
            }
            
            # Calculate weighted transition score
            total_weight = 0
            
            for feature in available_features:
                feature_weight = weights.get(feature, 0.1)
                total_weight += feature_weight
                
                # Special handling for tempo
                if feature == 'tempo':
                    tempo_diff = abs(track1[feature] - track2[feature])
                    # Lower score for bigger tempo differences
                    tempo_score = max(0, 1 - (tempo_diff / 50))  # 50 BPM difference is threshold
                    detailed_score += feature_weight * tempo_score
                
                # Special handling for key
                elif feature == 'key':
                    # Give bonus for same key or compatible keys
                    if track1[feature] == track2[feature]:  # Same key
                        detailed_score += feature_weight * 1.0
                    elif (abs(track1[feature] - track2[feature]) == 5) or (abs(track1[feature] - track2[feature]) == 7):
                        # Perfect fifth or fourth relation
                        detailed_score += feature_weight * 0.8
                    else:
                        detailed_score += feature_weight * 0.3
                
                # For other features, closer is better
                else:
                    feature_diff = abs(track1[feature] - track2[feature])
                    feature_score = 1 - feature_diff  # Normalized features are in [0,1]
                    detailed_score += feature_weight * feature_score
            
            # Normalize by total weight
            if total_weight > 0:
                detailed_score /= total_weight
                
                # Blend base similarity with detailed audio feature score
                final_score = 0.4 * base_score + 0.6 * detailed_score
                return final_score
        
        # Fall back to just similarity if no audio features
        return base_score
    
    def optimize_queue(self, track_ids, start_fixed=True, end_fixed=False):
        """Optimize a queue of tracks for smooth transitions"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return track_ids
        
        # If we don't have enough tracks, return the original queue
        if len(track_ids) <= 2:
            return track_ids
        
        # Filter to only include tracks in our dataset
        valid_ids = [track_id for track_id in track_ids if track_id in self.track_indices]
        
        if len(valid_ids) != len(track_ids):
            logger.warning(f"{len(track_ids) - len(valid_ids)} tracks not found in dataset")
        
        if len(valid_ids) <= 2:
            return track_ids
        
        # Create transition score matrix
        n = len(valid_ids)
        transition_matrix = np.zeros((n, n))
        
        # Calculate all pairwise transition scores
        for i in range(n):
            for j in range(n):
                if i != j:
                    transition_matrix[i][j] = self.compute_transition_score(valid_ids[i], valid_ids[j])
        
        # Greedy algorithm to find optimal path
        # Start with first track if start_fixed is True
        if start_fixed:
            optimized_indices = [0]
            remaining = set(range(1, n))
        else:
            # Find best starting track if start isn't fixed
            # Sum transition scores to find most compatible track
            compatibility_scores = transition_matrix.sum(axis=1)
            start_idx = np.argmax(compatibility_scores)
            optimized_indices = [start_idx]
            remaining = set(range(n))
            remaining.remove(start_idx)
        
        # If end is fixed and we have more than 2 tracks
        if end_fixed and n > 2:
            end_idx = n - 1
            remaining.remove(end_idx)
            
            # Build the path from start to end-1
            while len(optimized_indices) < n - 1:
                current = optimized_indices[-1]
                # Find best next track
                next_idx = max(remaining, key=lambda x: transition_matrix[current][x])
                optimized_indices.append(next_idx)
                remaining.remove(next_idx)
            
            # Add the end track
            optimized_indices.append(end_idx)
        else:
            # Build the entire path
            while remaining:
                current = optimized_indices[-1]
                # Find best next track
                next_idx = max(remaining, key=lambda x: transition_matrix[current][x])
                optimized_indices.append(next_idx)
                remaining.remove(next_idx)
        
        # Translate indices back to track IDs
        optimized_queue = [valid_ids[i] for i in optimized_indices]
        
        # Add back any tracks that weren't in our dataset at their original positions
        for i, track_id in enumerate(track_ids):
            if track_id not in valid_ids:
                optimized_queue.insert(i, track_id)
        
        return optimized_queue
    
    def recommend_queue(self, seed_tracks, n_total=10):
        """Generate an optimized queue starting from seed tracks"""
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
            # If we already have enough tracks, just optimize the queue
            return self.optimize_queue(seed_ids)
        
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
            
            # Optimize the queue
            return self.optimize_queue(final_ids)
        
        return seed_ids