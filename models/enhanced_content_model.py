import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from difflib import SequenceMatcher
from models.base_model import BaseRecommender
from models.weighted_content_model import WeightedContentRecommender  # ✅ FIXED IMPORT

logger = logging.getLogger(__name__)

class EnhancedContentRecommender(BaseRecommender):
    """Enhanced Content-based Recommender with smart search and cultural intelligence"""
    
    def __init__(self):
        super().__init__(name="EnhancedContentRecommender")
        self.content_recommender = WeightedContentRecommender()  # ✅ FIXED CLASS NAME
        self.tracks_df = None
        self.feature_quality = {}
    
    def train(self, tracks_df, user_item_matrix=None):
        """Enhanced training with feature analysis"""
        if tracks_df is None or tracks_df.empty:
            logger.error("Cannot train with empty tracks data")
            return False
            
        start_time = datetime.now()
        
        # Save tracks_df for enhanced features
        self.tracks_df = tracks_df
        
        logger.info(f"Training EnhancedContentRecommender with {len(self.tracks_df)} tracks")
        logger.info(f"Available features: {len(self.tracks_df.columns)} columns")
        
        # Analyze feature quality using actual data
        self._analyze_feature_quality()
        
        # Train base content model
        logger.info("Training base WeightedContentRecommender...")
        success = self.content_recommender.train(tracks_df)
        
        if not success:
            logger.error("Failed to train base content recommender")
            return False
        
        self.train_time = datetime.now() - start_time
        logger.info(f"EnhancedContentRecommender trained successfully in {self.train_time.total_seconds():.2f} seconds")
        
        self.is_trained = True
        return True

    def _analyze_feature_quality(self):
        """Analyze quality of available features using actual data"""
        feature_analysis = {}
        
        # ✅ Check actual essential features
        essential_features = ['popularity', 'duration_ms', 'artist_popularity', 'music_culture']
        available_essential = [f for f in essential_features if f in self.tracks_df.columns]
        feature_analysis['essential_features'] = f"{len(available_essential)}/{len(essential_features)}"
        
        # ✅ Check actual cultural intelligence features  
        cultural_features = [col for col in self.tracks_df.columns if col.startswith('is_') or col == 'music_culture']
        feature_analysis['cultural_features'] = len(cultural_features)
        
        # ✅ Check actual genre features
        genre_features = [col for col in self.tracks_df.columns if col.startswith('genre_')]
        feature_analysis['genre_features'] = len(genre_features)
        
        # ✅ Check actual ISRC features
        isrc_features = ['isrc', 'is_major_label', 'market_penetration']
        available_isrc = [f for f in isrc_features if f in self.tracks_df.columns]
        feature_analysis['isrc_features'] = f"{len(available_isrc)}/{len(isrc_features)}"
        
        # ✅ Data completeness using actual columns
        completeness = {}
        for col in ['popularity', 'music_culture', 'isrc']:
            if col in self.tracks_df.columns:
                if col == 'isrc':
                    non_null = (self.tracks_df[col] != '').sum()
                else:
                    non_null = self.tracks_df[col].notna().sum()
                completeness[col] = f"{non_null}/{len(self.tracks_df)} ({non_null/len(self.tracks_df)*100:.1f}%)"
        
        feature_analysis['data_completeness'] = completeness
        self.feature_quality = feature_analysis
        
        logger.info("=== ENHANCED MODEL FEATURE ANALYSIS ===")
        for key, value in feature_analysis.items():
            logger.info(f"{key}: {value}")
        logger.info("=====================================")

    def _find_track_with_fuzzy(self, track_name, artist=None):
        """Enhanced search with fuzzy matching and cultural context"""
        # 1. Try exact match first
        exact_result = self._find_track_index(track_name=track_name, artist=artist)
        if exact_result is not None:
            return exact_result, 1.0  # Perfect confidence
        
        # 2. Enhanced fuzzy matching with cultural context
        best_match = None
        best_score = 0.0
        threshold = 0.7
        
        # Normalize inputs
        track_name_norm = track_name.lower().strip()
        artist_norm = artist.lower().strip() if artist else None
        
        # ✅ Detect culture from input using actual features
        input_culture = self._detect_input_culture(track_name, artist)
        
        for idx, row in self.tracks_df.iterrows():
            # Name similarity
            name_sim = SequenceMatcher(None, track_name_norm, str(row['name']).lower().strip()).ratio()
            
            if artist_norm:
                # Combined name + artist similarity
                artist_sim = SequenceMatcher(None, artist_norm, str(row['artist']).lower().strip()).ratio()
                combined_score = (name_sim * 0.7) + (artist_sim * 0.3)
            else:
                combined_score = name_sim
            
            # ✅ Cultural boost using actual music_culture feature
            if input_culture != 'other' and row.get('music_culture', 'other') == input_culture:
                combined_score *= 1.15  # 15% boost for cultural match
                
            if combined_score > best_score and combined_score >= threshold:
                best_score = combined_score
                best_match = idx
        
        return best_match, best_score

    def _detect_input_culture(self, track_name, artist):
        """Detect culture from user input using simple patterns"""
        text = f"{track_name or ''} {artist or ''}".lower()
        
        # Vietnamese detection
        vietnamese_chars = 'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'
        if any(char in text for char in vietnamese_chars):
            return 'vietnamese'
        
        # Korean detection
        korean_words = ['kpop', 'k-pop', 'korea', 'korean', 'bts', 'blackpink', 'twice']
        if any(word in text for word in korean_words):
            return 'korean'
        
        # Japanese detection
        japanese_words = ['jpop', 'j-pop', 'japan', 'japanese', 'anime']
        if any(word in text for word in japanese_words):
            return 'japanese'
        
        # Spanish detection
        spanish_words = ['spanish', 'latino', 'latin', 'reggaeton']
        if any(word in text for word in spanish_words):
            return 'spanish'
        
        return 'other'

    def recommend(self, track_name=None, artist=None, n_recommendations=10):
        """Enhanced recommendation with cultural intelligence using actual features"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = self._validate_n_recommendations(n_recommendations)
        
        # 1. Enhanced search with cultural context
        track_idx, search_confidence = self._find_track_with_fuzzy(track_name, artist)
        
        if track_idx is None:
            logger.warning(f"Track '{track_name}' not found with enhanced search, using cultural fallback")
            recommendations = self._smart_cultural_fallback(track_name, artist, n_recommendations)
            
            # Return clean result
            result_cols = ['name', 'artist', 'enhanced_score']
            if 'music_culture' in recommendations.columns:
                result_cols.append('music_culture')
            available_cols = [col for col in result_cols if col in recommendations.columns]
            return recommendations[available_cols].round(3)
        
        # 2. Get seed track info with cultural context
        seed_track = self.tracks_df.iloc[track_idx]
        seed_culture = seed_track.get('music_culture', 'other')
        logger.info(f"Found track with {search_confidence:.3f} confidence: '{seed_track['name']}' by {seed_track['artist']} ({seed_culture})")
        
        # 3. Get base recommendations from core model
        try:
            base_recs = self.content_recommender.recommend(
                track_name=track_name, 
                artist=artist, 
                n_recommendations=min(n_recommendations * 2, len(self.tracks_df) - 1)
            )
        except Exception as e:
            logger.error(f"Error getting base recommendations: {e}")
            return self._smart_cultural_fallback(track_name, artist, n_recommendations)
        
        if base_recs.empty:
            return self._smart_cultural_fallback(track_name, artist, n_recommendations)
        
        # 4. ✅ Enhanced scoring using actual features
        enhanced_scores = self._calculate_enhanced_similarity(track_idx, base_recs)
        base_recs['enhanced_score'] = enhanced_scores
        
        # 5. Re-rank by enhanced score with cultural diversity
        final_recommendations = base_recs.sort_values('enhanced_score', ascending=False)
        
        # ✅ Ensure cultural diversity using actual music_culture feature
        if 'music_culture' in final_recommendations.columns:
            same_culture_target = int(n_recommendations * 0.7)
            cross_culture_target = n_recommendations - same_culture_target
            
            same_culture_recs = final_recommendations[
                final_recommendations['music_culture'] == seed_culture
            ].head(same_culture_target)
            
            cross_culture_recs = final_recommendations[
                final_recommendations['music_culture'] != seed_culture
            ].head(cross_culture_target)
            
            final_result = pd.concat([same_culture_recs, cross_culture_recs]).head(n_recommendations)
        else:
            final_result = final_recommendations.head(n_recommendations)
        
        # 6. Log cultural analytics using actual features
        if 'music_culture' in final_result.columns:
            same_culture_count = (final_result['music_culture'] == seed_culture).sum()
            logger.info(f"Enhanced cultural recommendation:")
            logger.info(f"  Same culture ({seed_culture}): {same_culture_count}/{len(final_result)} ({same_culture_count/len(final_result)*100:.1f}%)")
            
            culture_dist = final_result['music_culture'].value_counts()
            logger.info(f"  Cultural distribution: {dict(culture_dist)}")
        
        # Return clean result
        result_cols = ['name', 'artist', 'enhanced_score']
        if 'music_culture' in final_result.columns:
            result_cols.append('music_culture')
        available_cols = [col for col in result_cols if col in final_result.columns]
        
        return final_result[available_cols].round(3)

    def _calculate_enhanced_similarity(self, seed_track_idx, candidates_df):
        """Calculate enhanced similarity using actual data features"""
        enhanced_scores = []
        
        seed_data = self.tracks_df.iloc[seed_track_idx]
        seed_culture = seed_data.get('music_culture', 'other')
        
        for _, candidate in candidates_df.iterrows():
            # 1. Base score from final_score if available
            base_sim = candidate.get('final_score', 0.5)
            
            # 2. ✅ Enhanced cultural similarity using actual features
            candidate_culture = candidate.get('music_culture', 'other')
            if seed_culture == candidate_culture and seed_culture != 'other':
                cultural_boost = 0.3  # Same culture boost
            elif seed_culture == 'vietnamese' and candidate_culture in ['korean', 'japanese', 'chinese']:
                cultural_boost = 0.15  # Vietnamese-Asian cross-similarity
            elif seed_culture in ['korean', 'japanese', 'chinese'] and candidate_culture in ['korean', 'japanese', 'chinese']:
                cultural_boost = 0.2  # East Asian cross-similarity
            else:
                cultural_boost = 0.0
            
            # 3. ✅ Professional quality matching using actual features
            quality_boost = 0.0
            if 'is_major_label' in seed_data and 'is_major_label' in candidate:
                if seed_data.get('is_major_label', 0) == candidate.get('is_major_label', 0):
                    quality_boost = 0.1
            
            # 4. Popularity balancing
            seed_pop = seed_data.get('popularity', 50)
            cand_pop = candidate.get('popularity', 50)
            pop_diff = abs(seed_pop - cand_pop) / 100
            popularity_factor = max(0.8, 1 - pop_diff)
            
            # 5. Artist diversity penalty
            artist_penalty = 0.8 if seed_data.get('artist', '') == candidate.get('artist', '') else 1.0
            
            # ✅ Combine all factors
            enhanced_score = (
                base_sim * 0.5 +      # Base recommendation score
                cultural_boost +       # Cultural intelligence bonus
                quality_boost +        # Professional quality bonus
                0.2                   # Base score
            ) * popularity_factor * artist_penalty
            
            enhanced_scores.append(min(enhanced_score, 1.0))
        
        return enhanced_scores

    def _smart_cultural_fallback(self, track_name, artist, n_recommendations):
        """Smart fallback with cultural intelligence using actual features"""
        fallback_results = []
        
        # 1. Detect culture from input
        detected_culture = self._detect_input_culture(track_name, artist)
        
        # 2. ✅ Cultural priority fallback using actual music_culture feature
        if detected_culture != 'other' and 'music_culture' in self.tracks_df.columns:
            culture_tracks = self.tracks_df[self.tracks_df['music_culture'] == detected_culture]
            if not culture_tracks.empty:
                fallback_results.append(('cultural_match', culture_tracks, detected_culture))
        
        # 3. Artist-based fallback
        if artist:
            artist_tracks = self.tracks_df[
                self.tracks_df['artist'].str.contains(artist, case=False, na=False, regex=False)
            ]
            if not artist_tracks.empty:
                fallback_results.append(('artist_match', artist_tracks, 'artist_based'))
        
        # 4. Partial name match
        if track_name and len(track_name) > 3:
            partial_matches = self.tracks_df[
                self.tracks_df['name'].str.contains(track_name, case=False, na=False, regex=False)
            ]
            if not partial_matches.empty:
                fallback_results.append(('partial_name', partial_matches, 'name_based'))
        
        # 5. Select best fallback
        if fallback_results:
            for fallback_type, fallback_df, fallback_reason in fallback_results:
                if 'popularity' in fallback_df.columns:
                    result = fallback_df.nlargest(n_recommendations, 'popularity')
                else:
                    result = fallback_df.sample(min(n_recommendations, len(fallback_df)))
                
                result = result.copy()
                result['enhanced_score'] = 0.6  # Medium confidence
                result['fallback_type'] = fallback_type
                
                logger.info(f"Cultural fallback used: {fallback_type} ({fallback_reason}) - {len(result)} tracks")
                return result
        
        # 6. Final fallback - popular tracks with cultural diversity
        if 'popularity' in self.tracks_df.columns:
            popular_tracks = self.tracks_df.nlargest(n_recommendations * 2, 'popularity')
            
            # ✅ Ensure cultural diversity using actual music_culture feature
            diverse_result = []
            cultures_used = set()
            
            for _, track in popular_tracks.iterrows():
                track_culture = track.get('music_culture', 'other')
                if track_culture not in cultures_used or len(diverse_result) < n_recommendations:
                    diverse_result.append(track)
                    cultures_used.add(track_culture)
                    
                if len(diverse_result) >= n_recommendations:
                    break
            
            result_df = pd.DataFrame(diverse_result)
            result_df['enhanced_score'] = 0.4  # Lower confidence
            result_df['fallback_type'] = 'diverse_popular'
            
            return result_df
        
        # Last resort
        return self._create_fallback_recommendations(track_name, n_recommendations)

    def explore_by_culture(self, culture, n_recommendations=10):
        """Explore tracks by culture using actual data features"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = self._validate_n_recommendations(n_recommendations)
        
        # ✅ Filter by actual music_culture feature
        if 'music_culture' in self.tracks_df.columns:
            culture_tracks = self.tracks_df[self.tracks_df['music_culture'] == culture.lower()]
        else:
            # Fallback to binary culture features
            culture_col = f'is_{culture.lower()}'
            if culture_col in self.tracks_df.columns:
                culture_tracks = self.tracks_df[self.tracks_df[culture_col] == 1]
            else:
                logger.warning(f"Culture '{culture}' not found in data")
                return pd.DataFrame()
        
        if culture_tracks.empty:
            logger.warning(f"No tracks found for culture: {culture}")
            return pd.DataFrame()
        
        # Rank by enhanced scoring
        result = culture_tracks.copy()
        result['enhanced_score'] = 1.0  # Base for exact culture match
        
        # Boost by popularity and professional quality
        if 'popularity' in result.columns:
            popularity_boost = result['popularity'] / 100 * 0.3
            result['enhanced_score'] += popularity_boost
        
        if 'is_major_label' in result.columns:
            professional_boost = result['is_major_label'] * 0.2
            result['enhanced_score'] += professional_boost
        
        # Sort and select top recommendations
        final_result = result.sort_values(['enhanced_score', 'popularity'], ascending=[False, False]).head(n_recommendations)
        
        logger.info(f"Cultural exploration for '{culture}': {len(final_result)} tracks found")
        
        # Return clean result
        result_cols = ['name', 'artist', 'enhanced_score']
        if 'music_culture' in final_result.columns:
            result_cols.append('music_culture')
        available_cols = [col for col in result_cols if col in final_result.columns]
        
        return final_result[available_cols].round(3)