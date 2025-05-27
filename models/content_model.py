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
    Content-based recommender with ISRC-based cultural intelligence and optimized weighted scoring.
    """
    def __init__(self, weights=None):
        super().__init__(name="WeightedContentRecommender")
        self.tracks_df = None
        self.genre_matrix = None
        self.scalers = {}
        self.is_trained = False
        # ✅ UPDATED WEIGHTS: ISRC-based cultural intelligence priority
        self.weights = weights or {
            "cultural_similarity": 0.65,    # ISRC + regional + text fallback
            "genre_similarity": 0.20,       # Mood matching remains important
            "professional_quality": 0.08,   # NEW: Major label + market penetration
            "track_popularity": 0.04,       # Reduced 
            "artist_popularity": 0.02,      # Reduced
            "duration_similarity": 0.01,    # Minimal
        }

    def train(self, tracks_df):
        """Enhanced training with ISRC-based cultural intelligence"""
        self.tracks_df = tracks_df.copy()
        
        # 1. Track popularity - Enhanced transformation
        if 'popularity' in self.tracks_df.columns:
            pop_values = self.tracks_df['popularity'].fillna(0)
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
            smooth_artist_pop = np.power(artist_pop, 0.7)
            self.scalers['artist_popularity'] = MinMaxScaler()
            self.tracks_df['artist_popularity_norm'] = self.scalers['artist_popularity'].fit_transform(
                smooth_artist_pop.values.reshape(-1, 1)
            ).flatten()
        else:
            self.tracks_df['artist_popularity_norm'] = 0.5

        # 3. Duration normalization
        if 'duration_ms' in self.tracks_df.columns:
            durations = self.tracks_df['duration_ms'].fillna(200000)
            duration_min = durations / 60000
            self.tracks_df['duration_min'] = duration_min
            
            from sklearn.preprocessing import RobustScaler
            self.scalers['duration'] = RobustScaler()
            duration_scaled = self.scalers['duration'].fit_transform(duration_min.values.reshape(-1, 1))
            self.tracks_df['duration_norm'] = MinMaxScaler().fit_transform(duration_scaled).flatten()
        else:
            self.tracks_df['duration_norm'] = 0.5
            self.tracks_df['duration_min'] = 3.5

        # ✅ 4. CULTURAL INTELLIGENCE - ISRC-based (PRIORITY #1)
        self._prepare_cultural_features()
        
        # ✅ 5. PROFESSIONAL QUALITY - NEW
        self._prepare_professional_quality_features()

        # 6. Genre matrix
        genre_cols = [col for col in self.tracks_df.columns if col.startswith('genre_')]
        if genre_cols:
            self.genre_matrix = self.tracks_df[genre_cols].values
            logger.info(f"Using {len(genre_cols)} genre features for similarity")
        else:
            self.genre_matrix = np.ones((len(self.tracks_df), 1))
            logger.warning("No genre features found, using dummy matrix")

        self.is_trained = True
        logger.info(f"WeightedContentRecommender trained with ISRC-based cultural intelligence on {len(self.tracks_df)} tracks")

    # ✅ NEW METHOD 1: CULTURAL INTELLIGENCE
    def _prepare_cultural_features(self):
        """Prepare ISRC-based cultural intelligence features"""
        
        # Create cultural region mapping from ISRC + markets + text
        cultural_regions = {
            'vietnamese': 0,
            'korean': 1, 
            'japanese': 2,
            'chinese': 3,
            'western': 4,
            'spanish': 5,
            'other': 6
        }
        
        # Primary cultural classification (ISRC > Market > Text)
        def determine_primary_culture(row):
            # ISRC-based (HIGHEST PRIORITY)
            if row.get('isrc_vietnam', 0) == 1:
                return 'vietnamese'
            elif row.get('isrc_south_korea', 0) == 1:
                return 'korean'
            elif row.get('isrc_japan', 0) == 1:
                return 'japanese'
            elif row.get('isrc_china', 0) == 1:
                return 'chinese'
            elif row.get('isrc_united_states', 0) == 1 or row.get('isrc_united_kingdom', 0) == 1:
                return 'western'
            elif row.get('isrc_mexico', 0) == 1 or row.get('isrc_brazil', 0) == 1:
                return 'spanish'
            
            # Market-based (SECONDARY)
            elif row.get('market_southeast_asia', 0) == 1:
                markets = row.get('markets_list', [])
                if isinstance(markets, list) and 'VN' in markets:
                    return 'vietnamese'
            elif row.get('market_east_asia', 0) == 1:
                markets = row.get('markets_list', [])
                if isinstance(markets, list):
                    if 'KR' in markets:
                        return 'korean'
                    elif 'JP' in markets:
                        return 'japanese'
                    elif any(c in markets for c in ['CN', 'TW', 'HK']):
                        return 'chinese'
            elif row.get('market_north_america', 0) == 1:
                return 'western'
            elif row.get('market_latin_america', 0) == 1:
                return 'spanish'
            
            # Text-based fallback (LOWEST PRIORITY)
            elif row.get('text_vietnamese', 0) == 1:
                return 'vietnamese'
            elif row.get('text_korean', 0) == 1:
                return 'korean'
            elif row.get('text_japanese', 0) == 1:
                return 'japanese'
            elif row.get('text_chinese', 0) == 1:
                return 'chinese'
            elif row.get('text_spanish', 0) == 1:
                return 'spanish'
            
            return 'other'
        
        # Apply cultural classification
        self.tracks_df['primary_culture'] = self.tracks_df.apply(determine_primary_culture, axis=1)
        self.tracks_df['culture_code'] = self.tracks_df['primary_culture'].map(cultural_regions).fillna(6)
        
        # Secondary cultural features for cross-cultural similarity
        self.tracks_df['is_asian'] = (
            self.tracks_df['primary_culture'].isin(['vietnamese', 'korean', 'japanese', 'chinese'])
        ).astype(int)
        
        self.tracks_df['is_east_asian'] = (
            self.tracks_df['primary_culture'].isin(['korean', 'japanese', 'chinese'])
        ).astype(int)
        
        # Cultural similarity scores (for cross-cultural recommendations)
        def cultural_openness_score(row):
            # Global releases more likely to cross cultures
            market_penetration = row.get('market_penetration', 0)
            is_global = row.get('is_global_release', 0)
            cultural_diversity = row.get('cultural_diversity_score', 0)
            
            base_openness = market_penetration * 0.5 + is_global * 0.3 + cultural_diversity * 0.2
            return min(1.0, base_openness)
        
        self.tracks_df['cultural_openness'] = self.tracks_df.apply(cultural_openness_score, axis=1)
        
        # Log cultural distribution
        culture_dist = self.tracks_df['primary_culture'].value_counts()
        logger.info(f"Cultural distribution: {dict(culture_dist)}")

    # ✅ NEW METHOD 2: PROFESSIONAL QUALITY
    def _prepare_professional_quality_features(self):
        """Prepare professional quality indicators"""
        
        # Ensure required columns exist
        for col in ['is_major_label', 'market_penetration', 'professional_release_score']:
            if col not in self.tracks_df.columns:
                self.tracks_df[col] = 0.5  # Default medium quality
        
        # Enhanced professional quality score
        def calculate_professional_score(row):
            # Major label indicator (40%)
            major_label = row.get('is_major_label', 0) * 0.4
            
            # ISRC presence (30%) - professional releases have ISRC
            has_isrc = (not pd.isna(row.get('isrc', '')) and row.get('isrc', '').strip() != '') * 0.3
            
            # Market penetration (20%) - wide releases indicate professionalism
            market_score = row.get('market_penetration', 0) * 0.2
            
            # Release precision (10%) - day precision indicates planned release
            precision_bonus = 0.1 if row.get('release_date_precision', 'year') == 'day' else 0.05
            
            return min(1.0, major_label + has_isrc + market_score + precision_bonus)
        
        self.tracks_df['enhanced_professional_score'] = self.tracks_df.apply(calculate_professional_score, axis=1)
        
        # Professional tier classification
        def professional_tier(score):
            if score >= 0.8:
                return 'major_professional'
            elif score >= 0.6:
                return 'professional'
            elif score >= 0.4:
                return 'semi_professional'
            else:
                return 'independent'
        
        self.tracks_df['professional_tier'] = self.tracks_df['enhanced_professional_score'].apply(professional_tier)
        
        # Log professional distribution
        prof_dist = self.tracks_df['professional_tier'].value_counts()
        logger.info(f"Professional quality distribution: {dict(prof_dist)}")

    def recommend(self, track_name=None, artist=None, n_recommendations=10):
        """ISRC-based cultural intelligence recommendation with professional quality matching"""
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
            logger.warning(f"Track not found: '{track_name}' by '{artist}'. Using intelligent fallback.")
            fallback = self._create_intelligent_fallback(track_name, artist, n_recommendations)
            return fallback

        seed = seed_tracks.iloc[0]
        idx = seed.name

        # ✅ 1. CULTURAL SIMILARITY (0.65) - ISRC-based precision
        cultural_sim = self._calculate_cultural_similarity(seed, df)

        # ✅ 2. GENRE SIMILARITY (0.20) - Enhanced mood matching
        if self.genre_matrix.shape[1] > 1:
            seed_genres = self.genre_matrix[idx].reshape(1, -1)
            genre_sim = cosine_similarity(seed_genres, self.genre_matrix)[0]
            genre_sim = self._apply_cultural_genre_boost(genre_sim, seed, df)
        else:
            genre_sim = np.ones(len(df))

        # ✅ 3. PROFESSIONAL QUALITY (0.08) - NEW matching
        professional_sim = self._calculate_professional_similarity(seed, df)

        # 4. Traditional features (reduced weights)
        track_pop = df['track_popularity_norm'].values
        artist_pop = df['artist_popularity_norm'].values
        
        # Duration similarity
        seed_duration = seed['duration_norm']
        duration_diff = np.abs(df['duration_norm'].values - seed_duration)
        duration_sim = np.exp(-2 * duration_diff)

        # ✅ WEIGHTED COMBINATION with cultural-professional priority
        final_score = (
            self.weights["cultural_similarity"] * cultural_sim +
            self.weights["genre_similarity"] * genre_sim +
            self.weights["professional_quality"] * professional_sim +
            self.weights["track_popularity"] * track_pop +
            self.weights["artist_popularity"] * artist_pop +
            self.weights["duration_similarity"] * duration_sim
        )

        # Create enhanced result DataFrame
        df = df.copy()
        df['cultural_similarity'] = cultural_sim
        df['genre_similarity'] = genre_sim
        df['professional_similarity'] = professional_sim
        df['final_score'] = final_score
        
        # Cultural matching analytics
        seed_culture = seed.get('primary_culture', 'other')
        df['cultural_match'] = (df['primary_culture'] == seed_culture).astype(int)
        df['professional_match'] = (df['professional_tier'] == seed.get('professional_tier', 'independent')).astype(int)
        
        # Remove seed track
        df = df.drop(idx)

        # Sort by final score
        recommendations = df.sort_values('final_score', ascending=False).head(n_recommendations)

        # Enhanced result with analytics
        result_cols = ['name', 'artist', 'final_score', 'primary_culture', 'professional_tier']
        
        # Add available metadata
        meta_cols = ['popularity', 'release_year', 'cultural_similarity', 'genre_similarity', 'professional_similarity']
        for col in meta_cols:
            if col in recommendations.columns:
                result_cols.append(col)

        available_cols = [col for col in result_cols if col in recommendations.columns]
        result = recommendations[available_cols].copy()
        
        # Round scores for readability
        score_cols = ['final_score', 'cultural_similarity', 'genre_similarity', 'professional_similarity']
        for col in score_cols:
            if col in result.columns:
                result[col] = result[col].round(3)
        
        # ✅ ENHANCED LOGGING with cultural intelligence metrics
        same_culture_count = (result['primary_culture'] == seed_culture).sum()
        same_professional_count = (result['professional_tier'] == seed.get('professional_tier', 'independent')).sum()
        
        logger.info(f"ISRC-based recommendation for '{seed['name']}' ({seed_culture}):")
        logger.info(f"  Cultural match: {same_culture_count}/{len(result)} ({same_culture_count/len(result)*100:.1f}%)")
        logger.info(f"  Professional match: {same_professional_count}/{len(result)} ({same_professional_count/len(result)*100:.1f}%)")
        
        # Cultural diversity analysis
        if 'primary_culture' in result.columns:
            culture_dist = result['primary_culture'].value_counts()
            logger.info(f"  Cultural distribution: {dict(culture_dist)}")
        
        return result

    # ✅ NEW METHOD 3: CULTURAL SIMILARITY CALCULATION
    def _calculate_cultural_similarity(self, seed, candidates_df):
        """Calculate cultural similarity using ISRC + market + text hierarchy"""
        
        seed_culture = seed.get('primary_culture', 'other')
        seed_openness = seed.get('cultural_openness', 0.5)
        
        cultural_scores = []
        
        for _, candidate in candidates_df.iterrows():
            candidate_culture = candidate.get('primary_culture', 'other')
            candidate_openness = candidate.get('cultural_openness', 0.5)
            
            # Perfect match - same culture
            if seed_culture == candidate_culture:
                base_score = 1.0
            
            # Cross-cultural similarity with hierarchy
            elif seed_culture != 'other' and candidate_culture != 'other':
                # Vietnamese <-> East Asian bonus
                if (seed_culture == 'vietnamese' and candidate_culture in ['korean', 'japanese', 'chinese']) or \
                   (candidate_culture == 'vietnamese' and seed_culture in ['korean', 'japanese', 'chinese']):
                    base_score = 0.35  # Moderate cross-Asian similarity
                
                # East Asian cross-similarity (K-pop, J-pop, C-pop)
                elif seed_culture in ['korean', 'japanese', 'chinese'] and \
                     candidate_culture in ['korean', 'japanese', 'chinese']:
                    base_score = 0.45  # Higher East Asian cross-similarity
                
                # Western languages similarity
                elif seed_culture == 'western' and candidate_culture == 'spanish':
                    base_score = 0.25
                elif seed_culture == 'spanish' and candidate_culture == 'western':
                    base_score = 0.25
                
                # Distant cultures
                else:
                    base_score = 0.15
            
            # One unknown culture
            else:
                base_score = 0.20
            
            # Apply cultural openness bonus for cross-cultural tracks
            if seed_culture != candidate_culture:
                openness_bonus = (seed_openness + candidate_openness) / 2 * 0.3
                base_score = min(1.0, base_score + openness_bonus)
            
            cultural_scores.append(base_score)
        
        return np.array(cultural_scores)

    # ✅ NEW METHOD 4: PROFESSIONAL SIMILARITY
    def _calculate_professional_similarity(self, seed, candidates_df):
        """Calculate professional quality similarity"""
        
        seed_prof_score = seed.get('enhanced_professional_score', 0.5)
        seed_tier = seed.get('professional_tier', 'independent')
        
        professional_scores = []
        
        for _, candidate in candidates_df.iterrows():
            candidate_prof_score = candidate.get('enhanced_professional_score', 0.5)
            candidate_tier = candidate.get('professional_tier', 'independent')
            
            # Exact tier match bonus
            if seed_tier == candidate_tier:
                tier_bonus = 0.3
            elif (seed_tier in ['major_professional', 'professional'] and 
                  candidate_tier in ['major_professional', 'professional']):
                tier_bonus = 0.2  # Both professional
            elif (seed_tier in ['semi_professional', 'independent'] and 
                  candidate_tier in ['semi_professional', 'independent']):
                tier_bonus = 0.15  # Both indie
            else:
                tier_bonus = 0.0
            
            # Score similarity (euclidean distance based)
            score_diff = abs(seed_prof_score - candidate_prof_score)
            score_similarity = max(0.0, 1.0 - score_diff)
            
            # Combined professional similarity
            final_prof_sim = (score_similarity * 0.7) + tier_bonus
            professional_scores.append(min(1.0, final_prof_sim))
        
        return np.array(professional_scores)

    # ✅ NEW METHOD 5: CULTURAL GENRE BOOST
    def _apply_cultural_genre_boost(self, genre_sim, seed, candidates_df):
        """Apply cultural context to genre similarity"""
        
        seed_culture = seed.get('primary_culture', 'other')
        enhanced_genre_sim = np.copy(genre_sim)
        
        # Cultural genre preferences
        cultural_genre_preferences = {
            'vietnamese': ['genre_vpop', 'genre_ballad', 'genre_pop'],
            'korean': ['genre_kpop', 'genre_pop', 'genre_electronic'],
            'japanese': ['genre_jpop', 'genre_pop', 'genre_rock'],
            'chinese': ['genre_cpop', 'genre_pop', 'genre_ballad'],
            'western': ['genre_pop', 'genre_rock', 'genre_hip_hop'],
            'spanish': ['genre_latin', 'genre_reggaeton', 'genre_pop']
        }
        
        preferred_genres = cultural_genre_preferences.get(seed_culture, [])
        
        for i, (_, candidate) in enumerate(candidates_df.iterrows()):
            if i >= len(enhanced_genre_sim):
                continue
                
            # Boost genre similarity if candidate has culturally preferred genres
            genre_boost = 0.0
            for pref_genre in preferred_genres:
                if pref_genre in candidates_df.columns and candidate.get(pref_genre, 0) > 0:
                    genre_boost += 0.1
            
            enhanced_genre_sim[i] = min(1.0, enhanced_genre_sim[i] + genre_boost)
        
        return enhanced_genre_sim

    # ✅ NEW METHOD 6: INTELLIGENT FALLBACK
    def _create_intelligent_fallback(self, track_name, artist, n_recommendations):
        """Create culturally-aware fallback recommendations"""
        
        # Try to detect culture from track name or artist
        detected_culture = self._detect_culture_from_text(track_name, artist)
        
        if detected_culture != 'other':
            # Filter by detected culture
            culture_tracks = self.tracks_df[
                self.tracks_df['primary_culture'] == detected_culture
            ]
            
            if not culture_tracks.empty:
                # Sort by popularity and professional quality
                if 'popularity' in culture_tracks.columns:
                    fallback = culture_tracks.nlargest(n_recommendations, 'popularity')
                else:
                    fallback = culture_tracks.sample(min(n_recommendations, len(culture_tracks)))
                
                fallback = fallback.copy()
                fallback['final_score'] = 0.6  # Medium confidence for cultural fallback
                fallback['fallback_type'] = f'cultural_{detected_culture}'
                
                logger.info(f"Cultural fallback: Found {len(fallback)} {detected_culture} tracks")
                return fallback[['name', 'artist', 'final_score']].round(3)
        
        # Default fallback - popular tracks with cultural diversity
        if 'popularity' in self.tracks_df.columns:
            fallback = self.tracks_df.nlargest(n_recommendations * 2, 'popularity')
            
            # Ensure cultural diversity in fallback
            diverse_fallback = []
            cultures_used = set()
            
            for _, track in fallback.iterrows():
                track_culture = track.get('primary_culture', 'other')
                if track_culture not in cultures_used or len(diverse_fallback) < n_recommendations:
                    diverse_fallback.append(track)
                    cultures_used.add(track_culture)
                    
                if len(diverse_fallback) >= n_recommendations:
                    break
            
            result_df = pd.DataFrame(diverse_fallback)
            result_df['final_score'] = 0.4  # Lower confidence for general fallback
            result_df['fallback_type'] = 'diverse_popular'
            
            return result_df[['name', 'artist', 'final_score']].round(3)
        
        # Last resort - random sampling
        return self._create_fallback_recommendations(track_name, n_recommendations)

    def _detect_culture_from_text(self, track_name, artist):
        """Detect culture from track name and artist text"""
        text = f"{track_name or ''} {artist or ''}".lower()
        
        # Vietnamese detection
        vietnamese_chars = 'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'
        if any(char in text for char in vietnamese_chars):
            return 'vietnamese'
        
        # Korean detection
        korean_chars = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ'
        korean_words = ['kpop', 'k-pop', 'korea', 'seoul', 'bts', 'blackpink']
        if any(char in text for char in korean_chars) or any(word in text for word in korean_words):
            return 'korean'
        
        # Japanese detection  
        japanese_chars = 'あいうえおかきくけこアイウエオカキクケコ'
        japanese_words = ['jpop', 'j-pop', 'japan', 'tokyo', 'anime']
        if any(char in text for char in japanese_chars) or any(word in text for word in japanese_words):
            return 'japanese'
        
        # Spanish detection
        spanish_chars = 'ñáéíóúü'
        spanish_words = ['reggaeton', 'salsa', 'bachata', 'español', 'latino']
        if any(char in text for char in spanish_chars) or any(word in text for word in spanish_words):
            return 'spanish'
        
        return 'other'

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)