import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import logging
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CONTENT_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.tracks_df = None
        self.user_interactions_df = None
        self.track_features_df = None
        self.scaler = StandardScaler()
        
    def load_data(self, tracks_path=None, interactions_path=None):
        """Load data from specified paths or use defaults"""
        if tracks_path is None:
            tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
        
        if os.path.exists(tracks_path):
            self.tracks_df = pd.read_csv(tracks_path)
            logger.info(f"Loaded {len(self.tracks_df)} tracks from {tracks_path}")
        else:
            logger.error(f"Tracks file not found: {tracks_path}")
            
        if interactions_path and os.path.exists(interactions_path):
            self.user_interactions_df = pd.read_csv(interactions_path)
            logger.info(f"Loaded {len(self.user_interactions_df)} user interactions from {interactions_path}")
        
        return self
    
    def clean_tracks_data(self):
        """Clean and preprocess the tracks data"""
        if self.tracks_df is None:
            logger.error("No tracks data loaded. Call load_data() first.")
            return self
            
        # Remove duplicates
        self.tracks_df = self.tracks_df.drop_duplicates(subset=['id'])
        
        # Ensure required columns exist
        essential_cols = ['id', 'name', 'artist']
        self.tracks_df = self.tracks_df.dropna(subset=essential_cols)
        
        # Convert release_date to datetime and extract year
        if 'release_date' in self.tracks_df.columns:
            # Convert to datetime
            self.tracks_df['release_date'] = pd.to_datetime(
                self.tracks_df['release_date'], 
                errors='coerce'
            )
            # Extract year as a numerical feature
            self.tracks_df['release_year'] = self.tracks_df['release_date'].dt.year
        
        # Convert explicit to numeric (boolean to 0/1)
        if 'explicit' in self.tracks_df.columns:
            self.tracks_df['explicit'] = self.tracks_df['explicit'].map({True: 1, False: 0})
        
        logger.info(f"Cleaned tracks data. {len(self.tracks_df)} tracks remaining")
        return self
    
    def create_track_features(self, save=True):
        """Create and normalize track features for content-based filtering"""
        if self.tracks_df is None:
            logger.error("No tracks data loaded. Call load_data() first.")
            return None
        
        # Check which features actually exist in our data
        available_features = [f for f in CONTENT_FEATURES if f in self.tracks_df.columns]
        
        if not available_features:
            # If no listed features available, use popularity as fallback feature
            features_df = self.tracks_df[['id', 'name', 'artist', 'popularity']].copy()
            available_features = ['popularity']
            logger.warning("Limited features available. Using 'popularity' as fallback feature.")
        else:
            features_df = self.tracks_df[['id', 'name', 'artist'] + available_features].copy()
        
        # Handle missing values
        for feature in available_features:
            if feature in features_df.columns:
                # Fill missing values with median
                features_df[feature] = features_df[feature].fillna(features_df[feature].median())
        
        # Normalize the numerical features
        if len(available_features) > 0:
            features_to_scale = features_df[available_features].copy()
            scaled_features = self.scaler.fit_transform(features_to_scale)
            features_df[available_features] = scaled_features
        
        self.track_features_df = features_df
        
        if save:
            output_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
            features_df.to_csv(output_path, index=False)
            logger.info(f"Saved track features to {output_path}")
        
        return features_df
    
    def create_genre_features(self, save=True):
        """Chức năng vô hiệu hóa vì không có dữ liệu thể loại"""
        logger.warning("No genre data available - skipping genre features creation")
        return None
    
    def create_user_item_matrix(self, save=True):
        """Create user-item interaction matrix for collaborative filtering"""
        if self.user_interactions_df is None:
            logger.warning("No user interactions loaded. Creating synthetic data.")
            # Create synthetic user interactions if none provided
            np.random.seed(42)
            n_users = 100
            n_interactions_per_user = 20
            
            user_ids = [f"user_{i}" for i in range(n_users)]
            interactions = []
            
            for user_id in user_ids:
                # Sample random tracks
                track_indices = np.random.choice(
                    len(self.tracks_df), 
                    size=n_interactions_per_user, 
                    replace=False
                )
                
                for idx in track_indices:
                    track = self.tracks_df.iloc[idx]
                    # Generate random rating 1-5
                    rating = np.random.randint(1, 6)
                    
                    interactions.append({
                        'user_id': user_id,
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'rating': rating
                    })
            
            self.user_interactions_df = pd.DataFrame(interactions)
            logger.info(f"Created synthetic user interactions: {len(self.user_interactions_df)} entries")
        
        # Create user-item matrix
        user_item = self.user_interactions_df.pivot_table(
            index='user_id',
            columns='track_id',
            values='rating',
            fill_value=0
        )
        
        if save:
            output_path = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')
            user_item.to_csv(output_path)
            
            # Also save the interactions
            interactions_path = os.path.join(PROCESSED_DATA_DIR, 'user_interactions.csv')
            self.user_interactions_df.to_csv(interactions_path, index=False)
            
            logger.info(f"Saved user-item matrix ({user_item.shape[0]} users, {user_item.shape[1]} items) to {output_path}")
        
        return user_item
    
    def process_all(self):
        """Process all data and create all necessary features"""
        self.clean_tracks_data()
        self.create_track_features()
        self.create_genre_features()
        self.create_user_item_matrix()
        
        logger.info("All data processing complete")
        return self

if __name__ == "__main__":
    processor = DataProcessor()
    processor.load_data()
    processor.process_all()