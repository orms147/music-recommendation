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