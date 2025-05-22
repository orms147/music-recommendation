import pandas as pd
import numpy as np
import logging
from datetime import datetime
from models.base_model import BaseRecommender

logger = logging.getLogger(__name__)

class TransitionModel(BaseRecommender):
    """Model for optimizing track transitions in playlists"""
    
    def __init__(self):
        super().__init__(name="TransitionModel")
        self.tracks_df = None
        self.audio_features = None
    
    def train(self, tracks_df=None):
        """Train the transition model using track audio features"""
        start_time = datetime.now()
        
        if tracks_df is None or tracks_df.empty:
            logger.error("No tracks data provided for training")
            return False
        
        self.tracks_df = tracks_df.copy()
        
        # Check for audio features
        audio_features = [
            'tempo', 'key', 'energy', 'danceability', 'valence',
            'loudness', 'mode', 'acousticness', 'instrumentalness'
        ]
        
        available_features = [f for f in audio_features if f in self.tracks_df.columns]
        
        if not available_features:
            logger.warning("No audio features found for transition modeling")
            
        logger.info(f"Using audio features for transitions: {available_features}")
        
        # Cache audio features by track ID for faster lookups
        self.audio_features = {}
        
        for _, row in self.tracks_df.iterrows():
            feature_dict = {}
            for feature in available_features:
                feature_dict[feature] = row[feature]
            
            self.audio_features[row['id']] = feature_dict
        
        self.train_time = datetime.now() - start_time
        logger.info(f"Transition model trained in {self.train_time.total_seconds():.2f} seconds")
        
        self.is_trained = True
        return True
    
    def compute_transition_score(self, track1_id, track2_id):
        """Calculate transition score between two tracks"""
        if not self.is_trained or not self.audio_features:
            return 0.5  # Default score
            
        # Check if both tracks exist
        if track1_id not in self.audio_features or track2_id not in self.audio_features:
            return 0.5  # Default score
            
        track1 = self.audio_features[track1_id]
        track2 = self.audio_features[track2_id]
        
        # Define weights for different features
        weights = {
            'tempo': 0.30,      # Tempo similarity is critical
            'key': 0.15,        # Key compatibility 
            'energy': 0.15,     # Energy level shift
            'danceability': 0.10,  # Danceability consistency
            'valence': 0.10,    # Emotional tone shift
            'loudness': 0.10,   # Volume consistency
            'mode': 0.05,       # Musical mode (major/minor)
            'acousticness': 0.05
        }
        
        score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in track1 and feature in track2:
                total_weight += weight
                
                # Special handling for tempo
                if feature == 'tempo':
                    tempo_diff = abs(track1[feature] - track2[feature])
                    tempo_score = max(0, 1 - (tempo_diff / 40))  # 40 BPM difference threshold
                    score += weight * tempo_score
                
                # Special handling for key
                elif feature == 'key':
                    # Circle of fifths relationships
                    key_diff = min((track1[feature] - track2[feature]) % 12, 
                                  (track2[feature] - track1[feature]) % 12)
                    
                    if key_diff == 0:  # Same key
                        key_score = 1.0
                    elif key_diff == 7 or key_diff == 5:  # Perfect fifth/fourth
                        key_score = 0.8
                    elif key_diff == 1 or key_diff == 11:  # Semitone (usually harsh)
                        key_score = 0.3
                    else:
                        key_score = 0.5
                    
                    score += weight * key_score
                
                # Special handling for mode
                elif feature == 'mode':
                    # Same mode is better
                    mode_score = 1.0 if track1[feature] == track2[feature] else 0.5
                    score += weight * mode_score
                
                # For normalized features (0-1 range)
                elif feature in ['energy', 'danceability', 'valence', 'acousticness']:
                    feature_diff = abs(track1[feature] - track2[feature])
                    
                    # Small differences are good, but allow some variation
                    # A difference of 0.3 or less is still considered good
                    feature_score = max(0, 1 - (feature_diff / 0.3))
                    score += weight * feature_score
                
                # For loudness (typically negative dB values)
                elif feature == 'loudness':
                    # Convert to positive range for easier comparison
                    loudness1 = min(0, track1[feature]) * -1
                    loudness2 = min(0, track2[feature]) * -1
                    
                    # Normalize to 0-1 range (assuming typical range of 0 to -60 dB)
                    normalized1 = min(1, loudness1 / 60)
                    normalized2 = min(1, loudness2 / 60)
                    
                    loudness_diff = abs(normalized1 - normalized2)
                    loudness_score = max(0, 1 - (loudness_diff / 0.2))  # 20% difference threshold
                    score += weight * loudness_score
        
        # Normalize final score
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0.5  # Default if no features matched
            
        return final_score
    
    def optimize_queue(self, track_ids, start_fixed=True, end_fixed=False):
        """Optimize a queue of tracks for smooth transitions"""
        if not self.is_trained:
            logger.error("Model not trained. Please optimize the queue.")
            return track_ids
            
        # If queue is too short, no need to optimize
        if len(track_ids) <= 2:
            return track_ids
            
        # Filter to only include tracks in our dataset
        valid_ids = [track_id for track_id in track_ids 
                     if track_id in self.audio_features]
                     
        if len(valid_ids) != len(track_ids):
            logger.warning(f"{len(track_ids) - len(valid_ids)} tracks not found in dataset")
            
        if len(valid_ids) <= 2:
            return track_ids
            
        # Create transition score matrix
        n = len(valid_ids)
        transition_matrix = np.zeros((n, n))
        
        # Calculate transition scores for all pairs
        for i in range(n):
            for j in range(n):
                if i != j:
                    transition_matrix[i][j] = self.compute_transition_score(
                        valid_ids[i], valid_ids[j]
                    )
        
        # Apply greedy algorithm to find optimal path
        # Start with first track if start_fixed
        if start_fixed:
            ordered_indices = [0]
            remaining = set(range(1, n))
        else:
            # Find best starting track based on overall compatibility
            compatibility_scores = transition_matrix.sum(axis=1)
            start_idx = np.argmax(compatibility_scores)
            ordered_indices = [start_idx]
            remaining = set(range(n))
            remaining.remove(start_idx)
        
        # If end is fixed and we have more than 2 tracks
        if end_fixed and n > 2:
            end_idx = n - 1
            remaining.remove(end_idx)
            
            # Build path from start to end-1
            while len(ordered_indices) < n - 1:
                current = ordered_indices[-1]
                # Find best next track
                next_idx = max(remaining, key=lambda x: transition_matrix[current][x])
                ordered_indices.append(next_idx)
                remaining.remove(next_idx)
                
            # Add the fixed end track
            ordered_indices.append(end_idx)
        else:
            # Build entire path
            while remaining:
                current = ordered_indices[-1]
                # Find best next track
                next_idx = max(remaining, key=lambda x: transition_matrix[current][x])
                ordered_indices.append(next_idx)
                remaining.remove(next_idx)
        
        # Convert back to track IDs
        optimized_queue = [valid_ids[i] for i in ordered_indices]
        
        # Add back any tracks that weren't in our dataset (at their original positions)
        for i, track_id in enumerate(track_ids):
            if track_id not in valid_ids:
                optimized_queue.insert(min(i, len(optimized_queue)), track_id)
                
        return optimized_queue
    
    def analyze_transitions(self, track_ids):
        """Analyze transition quality between consecutive tracks"""
        if not self.is_trained:
            logger.error("Model not trained")
            return None
            
        if len(track_ids) < 2:
            logger.warning("Need at least 2 tracks to analyze transitions")
            return None
            
        results = []
        
        for i in range(len(track_ids) - 1):
            track1_id = track_ids[i]
            track2_id = track_ids[i+1]
            
            # Get track names if available
            track1_name = "Unknown"
            track2_name = "Unknown"
            
            if hasattr(self, 'tracks_df') and self.tracks_df is not None:
                track1_df = self.tracks_df[self.tracks_df['id'] == track1_id]
                track2_df = self.tracks_df[self.tracks_df['id'] == track2_id]
                
                if not track1_df.empty:
                    track1_name = f"{track1_df.iloc[0]['name']} - {track1_df.iloc[0]['artist']}"
                    
                if not track2_df.empty:
                    track2_name = f"{track2_df.iloc[0]['name']} - {track2_df.iloc[0]['artist']}"
            
            # Calculate transition score
            score = self.compute_transition_score(track1_id, track2_id)
            
            # Categorize transition quality
            if score >= 0.8:
                quality = "Excellent"
            elif score >= 0.6:
                quality = "Good"
            elif score >= 0.4:
                quality = "Average"
            else:
                quality = "Poor"
                
            results.append({
                'from_track': track1_name,
                'to_track': track2_name,
                'transition_score': score,
                'quality': quality
            })
            
        return pd.DataFrame(results)