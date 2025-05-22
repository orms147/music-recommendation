import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from models.base_model import BaseRecommender
from models.content_model import ContentBasedRecommender
from models.transition_model import TransitionModel
from config.config import CONTENT_WEIGHT, COLLABORATIVE_WEIGHT, SEQUENCE_WEIGHT

logger = logging.getLogger(__name__)

class HybridRecommender(BaseRecommender):
    """Hybrid recommendation model combining content-based and transition-based approaches"""
    
    def __init__(self):
        super().__init__(name="HybridRecommender")
        self.content_recommender = ContentBasedRecommender()
        self.transition_model = TransitionModel()
        self.content_weight = CONTENT_WEIGHT
        self.transition_weight = SEQUENCE_WEIGHT
        self.tracks_df = None
    
    def train(self, tracks_df, user_item_matrix=None, user_sequences=None):
        """Train all component models"""
        start_time = datetime.now()
        
        # Save tracks_df for use during recommendation
        self.tracks_df = tracks_df
        
        # Train content-based model
        logger.info("Training content-based recommender...")
        self.content_recommender.train(tracks_df)
        
        # Train transition model
        logger.info("Training transition model...")
        self.transition_model.train(tracks_df)
        
        self.train_time = datetime.now() - start_time
        logger.info(f"Hybrid model trained in {self.train_time.total_seconds():.2f} seconds")
        logger.info(f"Weights: Content={self.content_weight:.2f}, "
                   f"Transition={self.transition_weight:.2f}")
        
        self.is_trained = True
        return True
    
    def recommend(self, track_name=None, artist=None, user_id=None, n_recommendations=10):
        """Generate track recommendations"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = min(n_recommendations, len(self.tracks_df) - 1) if self.tracks_df is not None else n_recommendations
        all_recommendations = []
        weights = []
        
        # Content-based recommendations
        if track_name is not None and self.content_weight > 0:
            try:
                content_recs = self.content_recommender.recommend(
                    track_name=track_name, 
                    artist=artist, 
                    n_recommendations=n_recommendations*2
                )
                if not content_recs.empty:
                    all_recommendations.append(content_recs)
                    weights.append(self.content_weight)
            except Exception as e:
                logger.error(f"Error generating content-based recommendations: {e}")
        
        # If no recommendations, return fallback
        if not all_recommendations:
            logger.warning("No recommendations were generated")
            # Return random recommendations as fallback
            if hasattr(self, 'tracks_df') and self.tracks_df is not None:
                sample_size = min(n_recommendations, len(self.tracks_df))
                random_tracks = self.tracks_df.sample(sample_size)
                random_tracks['weighted_score'] = 0.5  # Medium confidence
                random_tracks['source'] = 'fallback'
                return random_tracks[['id', 'name', 'artist', 'weighted_score', 'source']]
            else:
                return pd.DataFrame()
        
        # Merge and rank recommendations
        return self._merge_recommendations(all_recommendations, weights, n_recommendations)
    
    def _merge_recommendations(self, all_recommendations, weights, n_recommendations):
        """Merge recommendations from different sources with weighted scoring"""
        if not all_recommendations:
            logger.warning("No recommendations to merge")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['id', 'name', 'artist', 'weighted_score', 'source'])
        
        # Process each recommendation set
        for i, recommendations in enumerate(all_recommendations):
            # Add weight column
            recommendations['weight'] = weights[i]
            
            # Find score column (e.g., content_score, collab_score)
            score_cols = [col for col in recommendations.columns if col.endswith('_score')]
            if score_cols:
                score_col = score_cols[0]
                # Calculate weighted score
                recommendations['weighted_score'] = recommendations[score_col] * weights[i]
                # Set source
                recommendations['source'] = score_col.split('_')[0]
            else:
                # Default values if no score column found
                recommendations['weighted_score'] = weights[i] * 0.5
                recommendations['source'] = 'unknown'
        
        # Combine all recommendations
        combined = pd.concat(all_recommendations)
        
        # Sort by weighted score
        combined = combined.sort_values('weighted_score', ascending=False)
        
        # Remove duplicates
        combined = combined.drop_duplicates(subset=['id'])
        
        # Get top recommendations
        result = combined.head(n_recommendations)
        
        # Ensure name and artist columns exist
        if hasattr(self, 'tracks_df') and self.tracks_df is not None:
            if 'name' not in result.columns or 'artist' not in result.columns:
                # Handle if ID column has different name
                if 'id' not in result.columns and 'track_id' in result.columns:
                    result = result.rename(columns={'track_id': 'id'})
                
                # Merge with tracks_df to get names and artists
                result = result.merge(
                    self.tracks_df[['id', 'name', 'artist']],
                    on='id',
                    how='left'
                )
        
        # Default values for missing columns
        if 'name' not in result.columns:
            result['name'] = 'Unknown'
        if 'artist' not in result.columns:
            result['artist'] = 'Unknown'
        
        return result
    
    def optimize_queue(self, track_ids=None, track_names=None, start_fixed=True, end_fixed=False):
        """Optimize a queue of tracks for smooth transitions"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return track_ids if track_ids else []
        
        # Handle track names if provided
        if track_ids is None and track_names is not None:
            track_ids = []
            for name in track_names:
                # Try to find track by name
                track_idx = self.content_recommender._find_track_index(track_name=name)
                if track_idx is not None:
                    track_ids.append(self.tracks_df.iloc[track_idx]['id'])
        
        if not track_ids:
            logger.warning("No valid track IDs for queue optimization")
            return []
        
        # Use transition model to optimize the queue
        return self.transition_model.optimize_queue(track_ids, start_fixed, end_fixed)
    
    def analyze_queue(self, track_ids=None, track_names=None):
        """Analyze transition quality in a queue"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return None
        
        # Handle track names if provided
        if track_ids is None and track_names is not None:
            track_ids = []
            for name in track_names:
                # Try to find track by name
                track_idx = self.content_recommender._find_track_index(track_name=name)
                if track_idx is not None:
                    track_ids.append(self.tracks_df.iloc[track_idx]['id'])
        
        if not track_ids or len(track_ids) < 2:
            logger.warning("Need at least 2 valid track IDs for queue analysis")
            return None
        
        # Use transition model to analyze the queue
        return self.transition_model.analyze_transitions(track_ids)
    
    def recommend_queue(self, seed_tracks, n_total=10):
        """Generate an optimized queue starting from seed tracks"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return []
        
        # Get recommendations using content model
        queue = self.content_recommender.recommend_queue(seed_tracks, n_total)
        
        # Optimize the queue
        optimized_queue = self.optimize_queue(queue)
        
        return optimized_queue
    
    def generate_playlist_from_seed(self, seed_track, seed_artist="", n_recommendations=10):
        """Generate a playlist from a seed track with transition analysis"""
        try:
            # Tìm kiếm bài hát ban đầu
            track_idx = self.content_recommender._find_track_index(track_name=seed_track, artist=seed_artist)
            
            if track_idx is None:
                logger.warning(f"Seed track '{seed_track}' not found")
                return None, None
                
            seed_id = self.tracks_df.iloc[track_idx]['id']
            
            # Tạo queue với số lượng bài hát đề xuất
            track_ids = self.recommend_queue([seed_id], n_recommendations)
            
            if not track_ids:
                return None, None
                
            # Lấy thông tin bài hát
            queue = self.tracks_df[self.tracks_df['id'].isin(track_ids)].copy()
            
            # Đảm bảo thứ tự trong queue
            queue['order'] = queue['id'].apply(lambda x: track_ids.index(x))
            queue = queue.sort_values('order').drop('order', axis=1)
            
            # Tạo phân tích chuyển tiếp
            analysis = self.analyze_queue(track_ids)
            
            return queue, analysis
        except Exception as e:
            logger.error(f"Error generating playlist: {e}")
            return None, None