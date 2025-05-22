import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from models.base_model import BaseRecommender
from config.config import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommender(BaseRecommender):
    """Collaborative filtering recommendation model"""
    
    def __init__(self, method='user-based', n_factors=50):
        super().__init__(name=f"CollaborativeFiltering_{method}")
        self.method = method  # 'user-based' or 'item-based'
        self.n_factors = n_factors
        self.user_item_matrix = None
        self.model = None
        self.item_factors = None
        self.user_factors = None
        self.tracks_df = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
    
    def train(self, user_item_matrix=None, tracks_df=None):
        """Train the collaborative filtering model"""
        start_time = datetime.now()
        
        # Load data if not provided
        if user_item_matrix is None:
            matrix_path = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')
            if os.path.exists(matrix_path):
                user_item_matrix = pd.read_csv(matrix_path, index_col=0)
                logger.info(f"Loaded user-item matrix with shape {user_item_matrix.shape}")
            else:
                logger.error(f"User-item matrix file not found: {matrix_path}")
                return False
        
        if tracks_df is None:
            tracks_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
            if os.path.exists(tracks_path):
                tracks_df = pd.read_csv(tracks_path)
                logger.info(f"Loaded {len(tracks_df)} tracks")
            else:
                logger.warning(f"Track features file not found: {tracks_path}")
                # Not critical, can continue without track details
        
        self.tracks_df = tracks_df
        self.user_item_matrix = user_item_matrix
        
        # Create mappings
        self.user_to_idx = {user: i for i, user in enumerate(user_item_matrix.index)}
        self.item_to_idx = {item: i for i, item in enumerate(user_item_matrix.columns)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}
        
        # Choose model based on method
        if self.method == 'matrix-factorization':
            self._train_matrix_factorization()
        elif self.method == 'item-based':
            self._train_item_based()
        else:  # Default to user-based
            self._train_user_based()
        
        self.train_time = datetime.now() - start_time
        self.is_trained = True
        
        logger.info(f"Collaborative filtering model ({self.method}) trained in {self.train_time}")
        return True
    
    def _train_matrix_factorization(self):
        """Train using matrix factorization with SVD"""
        # Convert to numpy array
        matrix = self.user_item_matrix.values
        
        # Apply SVD
        svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.item_factors = svd.fit_transform(matrix.T)  # Items x Factors
        self.user_factors = matrix.dot(self.item_factors)  # Users x Factors
        
        # Store the model
        self.model = svd
        
        logger.info(f"Trained matrix factorization with {self.n_factors} factors")
    
    def _train_item_based(self):
        """Train using item-based collaborative filtering"""
        # Item-item similarity
        matrix = self.user_item_matrix.values.T  # Items x Users
        
        # Use k-nearest neighbors for efficiency
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(matrix)
        
        self.model = model
        
        logger.info("Trained item-based collaborative filtering model")
    
    def _train_user_based(self):
        """Train using user-based collaborative filtering"""
        # User-user similarity
        matrix = self.user_item_matrix.values  # Users x Items
        
        # Use k-nearest neighbors for efficiency
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(matrix)
        
        self.model = model
        
        logger.info("Trained user-based collaborative filtering model")
    
    def recommend(self, user_id=None, track_id=None, n_recommendations=10):
        """Generate recommendations using collaborative filtering"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = self._validate_n_recommendations(n_recommendations)
        
        # Different recommendation approaches based on method and input
        if self.method == 'matrix-factorization':
            return self._recommend_matrix_factorization(user_id, track_id, n_recommendations)
        elif self.method == 'item-based':
            return self._recommend_item_based(user_id, track_id, n_recommendations)
        else:  # user-based
            return self._recommend_user_based(user_id, n_recommendations)
    
    def _recommend_matrix_factorization(self, user_id=None, track_id=None, n_recommendations=10):
        """Generate recommendations using matrix factorization"""
        if user_id is not None and user_id in self.user_to_idx:
            # User-based recommendation
            user_idx = self.user_to_idx[user_id]
            user_vector = self.user_factors[user_idx]
            
            # Calculate predicted ratings for all items
            predicted_ratings = user_vector.dot(self.item_factors.T)
            
            # Mask items already rated by the user
            user_rated_items = self.user_item_matrix.iloc[user_idx].values > 0
            predicted_ratings[user_rated_items] = -1
            
            # Get top recommendations
            top_item_indices = predicted_ratings.argsort()[-n_recommendations:][::-1]
            
        elif track_id is not None and track_id in self.item_to_idx:
            # Item-based recommendation (similar items)
            item_idx = self.item_to_idx[track_id]
            item_vector = self.item_factors[item_idx]
            
            # Calculate similarity to all other items
            similarities = np.dot(self.item_factors, item_vector) / (
                np.linalg.norm(self.item_factors, axis=1) * np.linalg.norm(item_vector)
            )
            
            # Mask the input item
            similarities[item_idx] = -1
            
            # Get top similar items
            top_item_indices = similarities.argsort()[-n_recommendations:][::-1]
            
        else:
            logger.error(f"Invalid input: user_id={user_id}, track_id={track_id}")
            return pd.DataFrame()
        
        # Get item IDs
        recommended_items = [self.idx_to_item[idx] for idx in top_item_indices]
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'id': recommended_items
        })
        
        # Add track details if available
        if self.tracks_df is not None:
            recommendations = recommendations.merge(
                self.tracks_df[['id', 'name', 'artist']], 
                on='id', 
                how='left'
            )
        
        return recommendations
    
    def _recommend_item_based(self, user_id=None, track_id=None, n_recommendations=10):
        """Generate recommendations using item-based collaborative filtering"""
        if track_id is not None and track_id in self.item_to_idx:
            # Get similar items
            item_idx = self.item_to_idx[track_id]
            item_vector = self.user_item_matrix.iloc[:, item_idx].values.reshape(1, -1)
            
            # Find similar items
            distances, indices = self.model.kneighbors(
                item_vector.T,
                n_neighbors=n_recommendations + 1
            )
            
            # First one is the item itself, remove it
            similar_item_indices = indices.flatten()[1:]
            similarity_scores = 1 - distances.flatten()[1:]  # Convert distance to similarity
            
            # Get item IDs
            recommended_items = [self.user_item_matrix.columns[idx] for idx in similar_item_indices]
            
        elif user_id is not None and user_id in self.user_to_idx:
            # Use user's ratings to find similar items
            user_idx = self.user_to_idx[user_id]
            user_ratings = self.user_item_matrix.iloc[user_idx].values
            
            # Get indices of items the user has rated
            rated_item_indices = np.where(user_ratings > 0)[0]
            
            if len(rated_item_indices) == 0:
                logger.warning(f"User {user_id} has no ratings. Cannot generate recommendations.")
                return pd.DataFrame()
            
            # Calculate weighted sum of similar items to find recommendations
            all_similarities = np.zeros(len(self.user_item_matrix.columns))
            
            for idx in rated_item_indices:
                item_vector = self.user_item_matrix.iloc[:, idx].values.reshape(1, -1)
                distances, indices = self.model.kneighbors(
                    item_vector.T,
                    n_neighbors=10  # Find 10 similar items for each rated item
                )
                similarities = 1 - distances.flatten()
                
                for sim_idx, sim_score in zip(indices.flatten(), similarities):
                    # Weight by user's rating
                    all_similarities[sim_idx] += sim_score * user_ratings[idx]
            
            # Mask items already rated by the user
            all_similarities[rated_item_indices] = -1
            
            # Get top recommendations
            top_item_indices = all_similarities.argsort()[-n_recommendations:][::-1]
            
            # Get item IDs
            recommended_items = [self.user_item_matrix.columns[idx] for idx in top_item_indices]
            similarity_scores = all_similarities[top_item_indices]
            
        else:
            logger.error(f"Invalid input: user_id={user_id}, track_id={track_id}")
            return pd.DataFrame()
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'id': recommended_items,
            'similarity_score': similarity_scores
        })
        
        # Add track details if available
        if self.tracks_df is not None:
            recommendations = recommendations.merge(
                self.tracks_df[['id', 'name', 'artist']], 
                on='id', 
                how='left'
            )
        
        return recommendations
    
    def _recommend_user_based(self, user_id, n_recommendations=10):
        """Generate recommendations using user-based collaborative filtering"""
        if user_id not in self.user_to_idx:
            logger.error(f"User {user_id} not found in training data")
            return pd.DataFrame()
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        
        # Find similar users
        distances, indices = self.model.kneighbors(
            user_vector,
            n_neighbors=10  # Find 10 similar users
        )
        
        # First one is the user itself, remove it
        similar_user_indices = indices.flatten()[1:]
        similarity_scores = 1 - distances.flatten()[1:]  # Convert distance to similarity
        
        # Get items already rated by the user
        user_rated_items = self.user_item_matrix.iloc[user_idx].values > 0
        
        # Calculate predicted ratings for all items
        predicted_ratings = np.zeros(len(self.user_item_matrix.columns))
        
        for idx, sim_score in zip(similar_user_indices, similarity_scores):
            # User's ratings
            similar_user_ratings = self.user_item_matrix.iloc[idx].values
            
            # Weight ratings by similarity
            predicted_ratings += similar_user_ratings * sim_score
        
        # Normalize
        predicted_ratings /= np.sum(similarity_scores)
        
        # Mask items already rated by the user
        predicted_ratings[user_rated_items] = -1
        
        # Get top recommendations
        top_item_indices = predicted_ratings.argsort()[-n_recommendations:][::-1]
        
        # Get item IDs
        recommended_items = [self.user_item_matrix.columns[idx] for idx in top_item_indices]
        recommendation_scores = predicted_ratings[top_item_indices]
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'id': recommended_items,
            'predicted_rating': recommendation_scores
        })
        
        # Add track details if available
        if self.tracks_df is not None:
            recommendations = recommendations.merge(
                self.tracks_df[['id', 'name', 'artist']], 
                on='id', 
                how='left'
            )
        
        return recommendations