import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from models.base_model import BaseRecommender
from config.config import CONTENT_FEATURES
import pickle

import logging
logger = logging.getLogger(__name__)

class WeightedContentRecommender(BaseRecommender):
    """
    Content-based recommender with weighted scoring and genre similarity.
    """
    def __init__(self, weights=None):
        super().__init__(name="WeightedContentRecommender")
        self.tracks_df = None
        self.genre_matrix = None
        self.scalers = {}
        self.is_trained = False
        # Default weights
        self.weights = weights or {
            "genre_similarity": 0.4,
            "track_popularity": 0.2,
            "artist_popularity": 0.2,
            "same_language": 0.1,
            "release_recency": 0.1
        }

    def train(self, tracks_df):
        """Chuẩn hóa dữ liệu và chuẩn bị các đặc trưng cần thiết."""
        self.tracks_df = tracks_df.copy()
        
        # Chuẩn hóa popularity
        self.scalers['track_popularity'] = MinMaxScaler()
        self.tracks_df['track_popularity_norm'] = self.scalers['track_popularity'].fit_transform(
            self.tracks_df[['popularity']].fillna(0)
        )
        
        # SỬ DỤNG artist_popularity đã được clean trong DataProcessor
        if 'artist_popularity' in self.tracks_df.columns:
            self.scalers['artist_popularity'] = MinMaxScaler()
            self.tracks_df['artist_popularity_norm'] = self.scalers['artist_popularity'].fit_transform(
                self.tracks_df[['artist_popularity']].fillna(50)
            )
            logger.info("Using cleaned artist_popularity from DataProcessor")
        else:
            logger.warning("artist_popularity not found, using default values")
            self.tracks_df['artist_popularity_norm'] = 0.5

        # Chuẩn hóa release_year (càng mới càng cao)
        if 'release_year' in self.tracks_df.columns:
            self.scalers['release_year'] = MinMaxScaler()
            self.tracks_df['release_recency'] = self.scalers['release_year'].fit_transform(
                self.tracks_df[['release_year']].fillna(0)
            )
        else:
            self.tracks_df['release_recency'] = 0

        # Xác định ngôn ngữ (ví dụ: is_vietnamese)
        if 'is_vietnamese' not in self.tracks_df.columns:
            self.tracks_df['is_vietnamese'] = 0

        # Chuẩn bị ma trận thể loại (genre one-hot)
        genre_cols = [col for col in self.tracks_df.columns if col.startswith('genre_')]
        if genre_cols:
            self.genre_matrix = self.tracks_df[genre_cols].values
        else:
            self.genre_matrix = None

        self.is_trained = True
        logger.info("WeightedContentRecommender trained with %d tracks.", len(self.tracks_df))

    def _genre_similarity(self, idx, genre_vector):
        if self.genre_matrix is None:
            return np.zeros(len(self.tracks_df))
        sim = cosine_similarity([genre_vector], self.genre_matrix)[0]
        return sim

    def recommend(self, track_name=None, artist=None, n_recommendations=10):
        """Đề xuất bài hát dựa trên weighted scoring và genre similarity."""
        if not self.is_trained:
            logger.error("Model not trained.")
            return "Model not trained."

        # Tìm bài hát gốc
        df = self.tracks_df
        if track_name is not None:
            mask = df['name'].str.lower().str.strip() == track_name.lower().strip()
            if artist:
                mask = mask & (df['artist'].str.lower().str.strip() == artist.lower().strip())
            seed_tracks = df[mask]
        else:
            seed_tracks = df.sample(1)

        if seed_tracks.empty:
            logger.warning(f"Track not found: '{track_name}' by '{artist}'. Returning random recommendations.")
            return df.sample(n_recommendations)

        seed = seed_tracks.iloc[0]
        idx = seed.name

        # Genre similarity
        genre_cols = [col for col in df.columns if col.startswith('genre_')]
        if genre_cols:
            genre_vector = seed[genre_cols].values
            genre_sim = self._genre_similarity(idx, genre_vector)
        else:
            genre_sim = np.ones(len(df))

        # Track popularity
        track_pop = df['track_popularity_norm'].values
        # Artist popularity (đã clean)
        artist_pop = df['artist_popularity_norm'].values
        # Same language
        same_lang = (df['is_vietnamese'] == seed['is_vietnamese']).astype(float).values
        # Release recency
        release_rec = df['release_recency'].values

        # Weighted score
        final_score = (
            self.weights["genre_similarity"] * genre_sim +
            self.weights["track_popularity"] * track_pop +
            self.weights["artist_popularity"] * artist_pop +
            self.weights["same_language"] * same_lang +
            self.weights["release_recency"] * release_rec
        )

        # Không đề xuất lại chính bài hát gốc
        df = df.copy()
        df['final_score'] = final_score
        df = df.drop(idx)

        recommendations = df.sort_values('final_score', ascending=False).head(n_recommendations)

        # Tạo DataFrame kết quả với tên cột clean
        result = recommendations[['name', 'artist', 'final_score', 'popularity', 'release_year']].copy()
        
        # Artist popularity đã clean
        if 'artist_popularity' in recommendations.columns:
            result['artist_popularity'] = recommendations['artist_popularity'].values

        return result

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
