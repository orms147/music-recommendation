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
    """Content-based recommender with ISRC-based cultural intelligence"""
    
    def __init__(self, weights=None):
        super().__init__(name="WeightedContentRecommender")
        self.tracks_df = None
        self.genre_matrix = None
        self.is_trained = False
        
        # ✅ SIMPLIFIED WEIGHTS aligned with actual data features
        self.weights = weights or {
            "cultural_similarity": 0.4,    # ISRC-based cultural matching
            "genre_similarity": 0.25,      # Musical style matching
            "popularity": 0.2,             # Track popularity
            "artist_popularity": 0.1,      # Artist reputation
            "duration_similarity": 0.05,   # Track length similarity
        }

    def train(self, tracks_df):
        """Train model with processed data"""
        if tracks_df is None or tracks_df.empty:
            logger.error("Cannot train with empty data")
            return False
            
        self.tracks_df = tracks_df.copy()
        logger.info(f"Training WeightedContentRecommender with {len(self.tracks_df)} tracks")
        
        # Validate essential columns
        required_cols = ['id', 'name', 'artist']
        missing_cols = [col for col in required_cols if col not in self.tracks_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # ✅ Use actual processed features
        self._prepare_features()
        self._prepare_genre_matrix()
        
        self.is_trained = True
        logger.info("WeightedContentRecommender trained successfully!")
        return True

    def _prepare_features(self):
        """Prepare features from actual data processor output"""
        # Normalize popularity
        if 'popularity' in self.tracks_df.columns:
            pop_values = self.tracks_df['popularity'].fillna(50)
            scaler = MinMaxScaler()
            self.tracks_df['popularity_norm'] = scaler.fit_transform(pop_values.values.reshape(-1, 1)).flatten()
        else:
            self.tracks_df['popularity_norm'] = 0.5
        
        # Normalize artist popularity  
        if 'artist_popularity' in self.tracks_df.columns:
            artist_pop = self.tracks_df['artist_popularity'].fillna(50)
            scaler = MinMaxScaler()
            self.tracks_df['artist_popularity_norm'] = scaler.fit_transform(artist_pop.values.reshape(-1, 1)).flatten()
        else:
            self.tracks_df['artist_popularity_norm'] = 0.5

        # Normalize duration
        if 'duration_ms' in self.tracks_df.columns:
            duration_values = self.tracks_df['duration_ms'].fillna(200000)
            scaler = MinMaxScaler()
            self.tracks_df['duration_norm'] = scaler.fit_transform(duration_values.values.reshape(-1, 1)).flatten()
        else:
            self.tracks_df['duration_norm'] = 0.5

        logger.info("Features normalized successfully")

    def _prepare_genre_matrix(self):
        """Prepare genre matrix from actual data"""
        genre_cols = [col for col in self.tracks_df.columns if col.startswith('genre_')]
        if genre_cols:
            self.genre_matrix = self.tracks_df[genre_cols].values
            logger.info(f"Using {len(genre_cols)} genre features")
        else:
            # Create dummy genre matrix
            self.genre_matrix = np.ones((len(self.tracks_df), 1))
            logger.warning("No genre features found, using dummy matrix")

    def recommend(self, track_name=None, artist=None, n_recommendations=10):
        """Generate recommendations using actual data features"""
        if not self.is_trained:
            logger.error("Model not trained")
            return pd.DataFrame()

        n_recommendations = self._validate_n_recommendations(n_recommendations)
        df = self.tracks_df

        # Find seed track
        if track_name is not None:
            mask = df['name'].str.lower().str.strip() == track_name.lower().strip()
            if artist:
                mask = mask & (df['artist'].str.lower().str.strip() == artist.lower().strip())
            seed_tracks = df[mask]
        else:
            seed_tracks = df.sample(1)

        if seed_tracks.empty:
            logger.warning(f"Track not found: '{track_name}' by '{artist}'. Using fallback.")
            return self._create_fallback_recommendations(track_name, n_recommendations)

        seed = seed_tracks.iloc[0]
        idx = seed.name

        # ✅ 1. CULTURAL SIMILARITY using actual features
        cultural_sim = self._calculate_cultural_similarity(seed, df)

        # ✅ 2. GENRE SIMILARITY
        if self.genre_matrix.shape[1] > 1:
            seed_genres = self.genre_matrix[idx].reshape(1, -1)
            genre_sim = cosine_similarity(seed_genres, self.genre_matrix)[0]
        else:
            genre_sim = np.ones(len(df))

        # ✅ 3. OTHER FEATURES
        track_pop = df['popularity_norm'].values
        artist_pop = df['artist_popularity_norm'].values
        
        # Duration similarity
        seed_duration = seed['duration_norm']
        duration_diff = np.abs(df['duration_norm'].values - seed_duration)
        duration_sim = np.exp(-2 * duration_diff)

        # ✅ 4. WEIGHTED COMBINATION
        final_score = (
            self.weights["cultural_similarity"] * cultural_sim +
            self.weights["genre_similarity"] * genre_sim +
            self.weights["popularity"] * track_pop +
            self.weights["artist_popularity"] * artist_pop +
            self.weights["duration_similarity"] * duration_sim
        )

        # Create result
        df = df.copy()
        df['final_score'] = final_score
        df['cultural_similarity'] = cultural_sim
        
        # Remove seed track
        df = df.drop(idx)
        recommendations = df.sort_values('final_score', ascending=False).head(n_recommendations)

        # ✅ 5. CULTURAL ANALYTICS using actual features
        seed_culture = seed.get('music_culture', 'other')
        if 'music_culture' in recommendations.columns:
            same_culture_count = (recommendations['music_culture'] == seed_culture).sum()
            logger.info(f"Cultural recommendation for '{seed['name']}' ({seed_culture}):")
            logger.info(f"  Same culture: {same_culture_count}/{len(recommendations)} ({same_culture_count/len(recommendations)*100:.1f}%)")

        # Return clean result
        result_cols = ['name', 'artist', 'final_score']
        if 'music_culture' in recommendations.columns:
            result_cols.append('music_culture')
        
        meta_cols = ['popularity', 'release_year', 'cultural_similarity']
        for col in meta_cols:
            if col in recommendations.columns:
                result_cols.append(col)

        available_cols = [col for col in result_cols if col in recommendations.columns]
        result = recommendations[available_cols].copy()
        
        # Round scores
        if 'final_score' in result.columns:
            result['final_score'] = result['final_score'].round(3)
        if 'cultural_similarity' in result.columns:
            result['cultural_similarity'] = result['cultural_similarity'].round(3)
        
        return result

    def _calculate_cultural_similarity(self, seed, candidates_df):
        """Calculate cultural similarity using actual data processor features"""
        
        # ✅ Use actual feature: music_culture
        seed_culture = seed.get('music_culture', 'other')
        
        cultural_scores = []
        
        for _, candidate in candidates_df.iterrows():
            candidate_culture = candidate.get('music_culture', 'other')
            
            # 1. Perfect cultural match
            if seed_culture == candidate_culture and seed_culture != 'other':
                base_score = 1.0
            
            # 2. Cross-cultural similarity
            elif seed_culture != 'other' and candidate_culture != 'other':
                # Vietnamese <-> Asian cross-similarity
                if (seed_culture == 'vietnamese' and candidate_culture in ['korean', 'japanese', 'chinese']) or \
                   (candidate_culture == 'vietnamese' and seed_culture in ['korean', 'japanese', 'chinese']):
                    base_score = 0.4
                
                # East Asian cross-similarity
                elif seed_culture in ['korean', 'japanese', 'chinese'] and \
                     candidate_culture in ['korean', 'japanese', 'chinese']:
                    base_score = 0.6
                
                # Western-Spanish similarity  
                elif (seed_culture == 'western' and candidate_culture == 'spanish') or \
                     (seed_culture == 'spanish' and candidate_culture == 'western'):
                    base_score = 0.3
                
                # Different cultures
                else:
                    base_score = 0.2
            
            # 3. Unknown culture
            else:
                base_score = 0.3
            
            # ✅ 4. BONUS from actual features
            bonus = 0.0
            
            # Major label consistency bonus
            if 'is_major_label' in seed and 'is_major_label' in candidate:
                if seed.get('is_major_label', 0) == candidate.get('is_major_label', 0):
                    bonus += 0.1
            
            # Market penetration bonus
            if 'market_penetration' in seed and 'market_penetration' in candidate:
                seed_market = seed.get('market_penetration', 0.5)
                cand_market = candidate.get('market_penetration', 0.5)
                market_sim = 1.0 - abs(seed_market - cand_market)
                bonus += market_sim * 0.1
            
            final_score = min(1.0, base_score + bonus)
            cultural_scores.append(final_score)
        
        return np.array(cultural_scores)

    def save(self, filepath):
        """Save model to file"""
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        try:
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None


# ✅ ALIAS for backward compatibility
ContentBasedRecommender = WeightedContentRecommender