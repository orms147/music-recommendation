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
    Content-based recommender with optimized weighted scoring and genre similarity.
    """
    def __init__(self, weights=None):
        super().__init__(name="WeightedContentRecommender")
        self.tracks_df = None
        self.genre_matrix = None
        self.scalers = {}
        self.is_trained = False
        # Trọng số tối ưu cho EXTREME LANGUAGE-FIRST với mood hierarchy
        self.weights = weights or {
            "same_language": 0.70,         # TĂNG CỰC MẠNH - ngôn ngữ chiếm ưu thế tuyệt đối
            "genre_similarity": 0.20,      # Giảm - nhưng vẫn quan trọng cho mood matching
            "track_popularity": 0.04,      # Giảm rất mạnh
            "artist_popularity": 0.03,     # Giảm rất mạnh
            "duration_similarity": 0.02,   # Giảm rất mạnh
            "artist_same": 0.01,           # Giảm cực mạnh
            "decade_match": 0.00,          # Bỏ hoàn toàn
        }

    def train(self, tracks_df):
        """Enhanced training with artist musical style analysis"""
        self.tracks_df = tracks_df.copy()
        
        # 1. Track popularity - Balanced transformation
        if 'popularity' in self.tracks_df.columns:
            pop_values = self.tracks_df['popularity'].fillna(0)
            # Sử dụng sqrt để giảm skewness thay vì log
            sqrt_pop = np.sqrt(pop_values)
            self.scalers['track_popularity'] = MinMaxScaler()
            self.tracks_df['track_popularity_norm'] = self.scalers['track_popularity'].fit_transform(
                sqrt_pop.values.reshape(-1, 1)
            ).flatten()
        else:
            self.tracks_df['track_popularity_norm'] = 0.5
        
        # 2. Artist popularity - Enhanced processing
        if 'artist_popularity' in self.tracks_df.columns:
            artist_pop = self.tracks_df['artist_popularity'].fillna(50)
            # Smooth transformation để tránh quá bias
            smooth_artist_pop = np.power(artist_pop, 0.7)  # Softer than sqrt
            self.scalers['artist_popularity'] = MinMaxScaler()
            self.tracks_df['artist_popularity_norm'] = self.scalers['artist_popularity'].fit_transform(
                smooth_artist_pop.values.reshape(-1, 1)
            ).flatten()
        else:
            self.tracks_df['artist_popularity_norm'] = 0.5

        # 3. Decade matching - MỚI
        if 'release_year' in self.tracks_df.columns:
            years = self.tracks_df['release_year'].fillna(2000)
            # Tạo decade categories
            self.tracks_df['decade'] = (years // 10) * 10
            logger.info("Created decade categories for matching")
        else:
            self.tracks_df['decade'] = 2000

        # 4. Duration normalization - Enhanced
        if 'duration_ms' in self.tracks_df.columns:
            durations = self.tracks_df['duration_ms'].fillna(200000)
            # Convert to minutes cho dễ hiểu
            duration_min = durations / 60000
            self.tracks_df['duration_min'] = duration_min
            
            # Normalize duration với robust scaling
            from sklearn.preprocessing import RobustScaler
            self.scalers['duration'] = RobustScaler()
            duration_scaled = self.scalers['duration'].fit_transform(duration_min.values.reshape(-1, 1))
            # Convert back to 0-1 range
            self.tracks_df['duration_norm'] = MinMaxScaler().fit_transform(duration_scaled).flatten()
        else:
            self.tracks_df['duration_norm'] = 0.5
            self.tracks_df['duration_min'] = 3.5

        # 5. Language features - Enhanced cultural similarity
        lang_features = ['is_vietnamese', 'is_korean', 'is_japanese', 'is_spanish', 'is_chinese']
        for lang in lang_features:
            if lang not in self.tracks_df.columns:
                self.tracks_df[lang] = 0
        
        # Create language regions for cultural similarity
        self.tracks_df['lang_region'] = 'other'
        self.tracks_df.loc[self.tracks_df['is_vietnamese'] == 1, 'lang_region'] = 'vietnamese'
        self.tracks_df.loc[self.tracks_df['is_korean'] == 1, 'lang_region'] = 'korean'
        self.tracks_df.loc[self.tracks_df['is_japanese'] == 1, 'lang_region'] = 'japanese'
        self.tracks_df.loc[self.tracks_df['is_spanish'] == 1, 'lang_region'] = 'spanish'
        self.tracks_df.loc[self.tracks_df['is_chinese'] == 1, 'lang_region'] = 'chinese'

        # 6. Special features - Enhanced
        for feature in ['has_collab', 'is_remix']:
            if feature not in self.tracks_df.columns:
                self.tracks_df[feature] = 0

        # 7. Genre matrix - Same as before but with logging
        genre_cols = [col for col in self.tracks_df.columns if col.startswith('genre_')]
        if genre_cols:
            self.genre_matrix = self.tracks_df[genre_cols].values
            logger.info(f"Using {len(genre_cols)} genre features for similarity")
        else:
            self.genre_matrix = np.ones((len(self.tracks_df), 1))
            logger.warning("No genre features found, using dummy matrix")

        # 8. Artist musical style analysis - MỚI
        if 'artist_id' in self.tracks_df.columns:
            # Fetch enhanced artist info if not already available
            artist_style_cols = [col for col in self.tracks_df.columns if col.startswith('is_') and 'artist' in col]
            
            if not artist_style_cols:
                logger.info("Analyzing artist musical styles from genres...")
                self._analyze_artist_styles_from_genres()
        
        # 9. Create artist consistency features
        self._create_artist_consistency_features()

        self.is_trained = True
        logger.info(f"WeightedContentRecommender trained with optimized weights on {len(self.tracks_df)} tracks")

    def _analyze_artist_styles_from_genres(self):
        """Analyze artist styles from existing genre data"""
        if 'artist_genres' not in self.tracks_df.columns:
            return
        
        # Extract artist style indicators
        for _, row in self.tracks_df.iterrows():
            genres = str(row.get('artist_genres', '')).lower().split('|')
            
            # Determine primary musical style
            self.tracks_df.loc[self.tracks_df.index == row.name, 'artist_is_pop'] = int(any('pop' in g for g in genres))
            self.tracks_df.loc[self.tracks_df.index == row.name, 'artist_is_rock'] = int(any(r in g for r in ['rock', 'metal'] for g in genres))
            self.tracks_df.loc[self.tracks_df.index == row.name, 'artist_is_hiphop'] = int(any(h in g for h in ['hip hop', 'rap'] for g in genres))
            self.tracks_df.loc[self.tracks_df.index == row.name, 'artist_is_electronic'] = int(any(e in g for e in ['electronic', 'edm'] for g in genres))
            self.tracks_df.loc[self.tracks_df.index == row.name, 'artist_is_asian'] = int(any(a in g for a in ['k-pop', 'j-pop', 'vietnamese'] for g in genres))
        
        logger.info("Analyzed artist musical styles from genre data")

    def _create_artist_consistency_features(self):
        """Create features for artist style consistency"""
        if 'artist' not in self.tracks_df.columns:
            return
        
        # Artist style consistency within dataset
        artist_genre_consistency = {}
        
        for artist in self.tracks_df['artist'].unique():
            artist_tracks = self.tracks_df[self.tracks_df['artist'] == artist]
            
            if len(artist_tracks) > 1:
                # Tính consistency của genre
                genre_cols = [col for col in self.tracks_df.columns if col.startswith('genre_')]
                if genre_cols:
                    genre_variance = artist_tracks[genre_cols].var().mean()
                    artist_genre_consistency[artist] = 1 / (1 + genre_variance)  # Lower variance = higher consistency
        
        # Apply consistency scores
        self.tracks_df['artist_genre_consistency'] = self.tracks_df['artist'].map(
            artist_genre_consistency
        ).fillna(0.5)  # Default for artists with single tracks

    def recommend(self, track_name=None, artist=None, n_recommendations=10):
        """EXTREME LANGUAGE-FIRST with mood hierarchy recommendation"""
        if not self.is_trained:
            logger.error("Model not trained.")
            return "Model not trained."

        # Find seed track
        df = self.tracks_df
        if track_name is not None:
            mask = df['name'].str.lower().str.strip() == track_name.lower().strip()
            if artist:
                mask = mask & (df['artist'].str.lower().str.strip() == artist.lower().strip())
            seed_tracks = df[mask]
        else:
            seed_tracks = df.sample(1)

        if seed_tracks.empty:
            logger.warning(f"Track not found: '{track_name}' by '{artist}'. Using simple fallback.")
            fallback = df.sample(n_recommendations)
            return fallback[['name', 'artist']].assign(final_score=0.5)

        seed = seed_tracks.iloc[0]
        idx = seed.name

        # 1. EXTREME LANGUAGE CONSISTENCY (0.70) - OVERWHELMING PRIORITY
        seed_lang_region = seed['lang_region']
        
        # Hierarchical language scoring
        lang_sim = np.zeros(len(df))
        for i, row_lang in enumerate(df['lang_region']):
            if row_lang == seed_lang_region:
                lang_sim[i] = 1.0  # Perfect match - same language gets full score
            elif (seed_lang_region in ['korean', 'japanese', 'chinese'] and 
                  row_lang in ['korean', 'japanese', 'chinese']):
                lang_sim[i] = 0.08  # Very low cross-Asian bonus
            elif (seed_lang_region in ['vietnamese'] and 
                  row_lang in ['korean', 'japanese', 'chinese']):
                lang_sim[i] = 0.05  # Minimal Vietnamese-Asian bonus
            else:
                lang_sim[i] = 0.02  # Almost zero for completely different regions

        # 2. Enhanced Genre similarity (0.20) - Mood hierarchy
        if self.genre_matrix.shape[1] > 1:
            seed_genres = self.genre_matrix[idx].reshape(1, -1)
            genre_sim = cosine_similarity(seed_genres, self.genre_matrix)[0]
            
            # Apply mood hierarchy - boost similar moods
            genre_sim = self._apply_mood_hierarchy(genre_sim, seed)
        else:
            genre_sim = np.ones(len(df))

        # 3. Popularity scores (0.04 + 0.03) - Minimal impact
        track_pop = df['track_popularity_norm'].values
        artist_pop = df['artist_popularity_norm'].values

        # 4. Duration similarity (0.02) - Very minimal
        seed_duration = seed['duration_norm']
        duration_diff = np.abs(df['duration_norm'].values - seed_duration)
        duration_sim = np.exp(-2 * duration_diff)

        # 5. Artist matching (0.01) - Almost zero to encourage diversity
        artist_match = (df['artist'] == seed['artist']).astype(float)

        # HIERARCHICAL SCORING STRATEGY
        # Ưu tiên 1: Same language + good mood (70% lang + 20% genre = 90%)
        # Ưu tiên 2: Same language + any mood (70% lang + minimal others = ~75%)
        # Ưu tiên 3: Different language + excellent mood + other factors (30% max)
        
        # Language bonus multiplier for mood matching
        language_mood_bonus = np.where(
            (lang_sim >= 0.9),  # Same language
            1.0 + 0.3 * genre_sim,  # Extra bonus for good mood when same language
            1.0  # No bonus for different language
        )

        # Final weighted score with hierarchical approach
        base_score = (
            self.weights["same_language"] * lang_sim +
            self.weights["genre_similarity"] * genre_sim +
            self.weights["track_popularity"] * track_pop +
            self.weights["artist_popularity"] * artist_pop +
            self.weights["duration_similarity"] * duration_sim +
            self.weights["artist_same"] * artist_match
        )
        
        # Apply language-mood hierarchy bonus
        final_score = base_score * language_mood_bonus

        # Create priority tiers for better understanding
        df = df.copy()
        df['language_match'] = lang_sim >= 0.9
        df['mood_score'] = genre_sim
        df['base_score'] = base_score
        df['final_score'] = final_score
        
        # Priority classification
        df['priority_tier'] = 'Tier 4: Other'
        df.loc[(df['language_match'] == True) & (df['mood_score'] >= 0.7), 'priority_tier'] = 'Tier 1: Same Lang + Great Mood'
        df.loc[(df['language_match'] == True) & (df['mood_score'] >= 0.4), 'priority_tier'] = 'Tier 2: Same Lang + Good Mood'
        df.loc[(df['language_match'] == True) & (df['mood_score'] < 0.4), 'priority_tier'] = 'Tier 3: Same Lang + Different Mood'
        
        # Remove seed track
        df = df.drop(idx)

        # Sort by final score (hierarchical)
        recommendations = df.sort_values('final_score', ascending=False).head(n_recommendations)

        # Return result with enhanced metadata
        result_cols = ['name', 'artist', 'final_score', 'priority_tier']
        
        # Add available metadata columns
        meta_cols = ['popularity', 'release_year', 'artist_popularity', 'duration_min', 'lang_region', 'mood_score']
        for col in meta_cols:
            if col in recommendations.columns:
                result_cols.append(col)

        available_cols = [col for col in result_cols if col in recommendations.columns]
        result = recommendations[available_cols].copy()
        
        # Round scores for readability
        if 'final_score' in result.columns:
            result['final_score'] = result['final_score'].round(3)
        if 'mood_score' in result.columns:
            result['mood_score'] = result['mood_score'].round(3)
        
        # Enhanced logging for hierarchical recommendations
        if 'lang_region' in result.columns:
            same_lang_count = (result['lang_region'] == seed['lang_region']).sum()
            logger.info(f"HIERARCHICAL recommendation: {same_lang_count}/{len(result)} same language tracks ({same_lang_count/len(result)*100:.1f}%)")
            
            # Log priority distribution
            if 'priority_tier' in result.columns:
                tier_dist = result['priority_tier'].value_counts()
                logger.info(f"Priority distribution: {dict(tier_dist)}")
            
            # Log language distribution
            lang_dist = result['lang_region'].value_counts()
            logger.info(f"Language distribution: {dict(lang_dist)}")
        
        return result

    def _apply_mood_hierarchy(self, genre_sim, seed):
        """Apply mood hierarchy - similar moods get boost, different moods get penalty"""
        # Determine seed mood category
        seed_mood = self._determine_mood_category(seed)
        
        # Apply mood-based adjustments
        enhanced_genre_sim = np.copy(genre_sim)
        
        for i, row in self.tracks_df.iterrows():
            if i >= len(genre_sim):
                continue
                
            track_mood = self._determine_mood_category(row)
            
            # Mood similarity bonus/penalty
            if track_mood == seed_mood:
                enhanced_genre_sim[i] *= 1.2  # 20% boost for same mood category
            elif self._are_similar_moods(seed_mood, track_mood):
                enhanced_genre_sim[i] *= 1.0  # No change for similar moods
            else:
                enhanced_genre_sim[i] *= 0.7  # 30% penalty for very different moods
        
        return enhanced_genre_sim

    def _determine_mood_category(self, track):
        """Determine mood category based on genre features"""
        genre_cols = [col for col in self.tracks_df.columns if col.startswith('genre_')]
        
        if not genre_cols:
            return 'unknown'
        
        # Simple mood categorization based on common genres
        track_genres = track[genre_cols] if isinstance(track[genre_cols], pd.Series) else track
        
        # Define mood categories
        energetic_genres = ['pop', 'rock', 'electronic', 'dance', 'hip hop', 'rap']
        calm_genres = ['acoustic', 'folk', 'classical', 'ambient', 'jazz', 'ballad']
        emotional_genres = ['r&b', 'soul', 'blues', 'indie', 'alternative']
        
        # Count genre matches for each mood
        energetic_score = sum(1 for g in energetic_genres if any(g in col.lower() for col in genre_cols if track_genres.get(col, 0) > 0.5))
        calm_score = sum(1 for g in calm_genres if any(g in col.lower() for col in genre_cols if track_genres.get(col, 0) > 0.5))
        emotional_score = sum(1 for g in emotional_genres if any(g in col.lower() for col in genre_cols if track_genres.get(col, 0) > 0.5))
        
        # Determine primary mood
        if energetic_score > calm_score and energetic_score > emotional_score:
            return 'energetic'
        elif calm_score > emotional_score:
            return 'calm'
        elif emotional_score > 0:
            return 'emotional'
        else:
            return 'neutral'

    def _are_similar_moods(self, mood1, mood2):
        """Check if two moods are similar"""
        similar_mood_groups = [
            ['energetic', 'neutral'],
            ['calm', 'emotional'],
            ['emotional', 'neutral']
        ]
        
        for group in similar_mood_groups:
            if mood1 in group and mood2 in group:
                return True
        
        return False

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
