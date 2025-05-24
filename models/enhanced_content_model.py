import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from difflib import SequenceMatcher
from models.base_model import BaseRecommender
from models.content_model import ContentBasedRecommender
from config.config import CONTENT_WEIGHT

logger = logging.getLogger(__name__)

class EnhancedContentRecommender(BaseRecommender):
    """Enhanced Content-based Recommender with smart features and multi-factor scoring"""
    
    def __init__(self):
        super().__init__(name="EnhancedContentRecommender")
        self.content_recommender = ContentBasedRecommender()
        self.tracks_df = None
        self.feature_quality = {}
    
    def train(self, tracks_df, user_item_matrix=None):
        """Enhanced training with feature analysis"""
        start_time = datetime.now()
        
        # Save tracks_df for use during recommendation
        self.tracks_df = tracks_df
        
        logger.info(f"Training EnhancedContentRecommender with {len(self.tracks_df)} tracks")
        logger.info(f"Available features: {len(self.tracks_df.columns)} columns")
        
        # Analyze feature quality
        self._analyze_feature_quality()
        
        # Train base content model
        logger.info("Training base content recommender...")
        success = self.content_recommender.train(tracks_df)
        
        if not success:
            logger.error("Failed to train base content recommender")
            return False
        
        self.train_time = datetime.now() - start_time
        logger.info(f"EnhancedContentRecommender trained successfully in {self.train_time.total_seconds():.2f} seconds")
        
        self.is_trained = True
        return True

    def _analyze_feature_quality(self):
        """Analyze quality of available features"""
        feature_analysis = {}
        
        # Real Spotify features
        real_features = ['popularity', 'duration_ms', 'explicit', 'release_year', 'artist_popularity']
        real_count = sum(1 for f in real_features if f in self.tracks_df.columns)
        feature_analysis['real_spotify_features'] = f"{real_count}/{len(real_features)}"
        
        # Genre features
        genre_features = [col for col in self.tracks_df.columns if col.startswith('genre_')]
        feature_analysis['genre_features'] = len(genre_features)
        
        # Language features
        lang_features = [col for col in self.tracks_df.columns if col.startswith('is_')]
        feature_analysis['language_features'] = len(lang_features)
        
        # Data completeness
        completeness = {}
        for col in ['popularity', 'release_year', 'artist_popularity']:
            if col in self.tracks_df.columns:
                non_null = self.tracks_df[col].notna().sum()
                completeness[col] = f"{non_null}/{len(self.tracks_df)} ({non_null/len(self.tracks_df)*100:.1f}%)"
        
        feature_analysis['data_completeness'] = completeness
        self.feature_quality = feature_analysis
        
        logger.info("=== FEATURE QUALITY ANALYSIS ===")
        for key, value in feature_analysis.items():
            logger.info(f"{key}: {value}")
        logger.info("===============================")

    def _find_track_with_fuzzy(self, track_name, artist=None):
        """Enhanced search with fuzzy matching"""
        # 1. Try exact match first
        exact_result = self.content_recommender._find_track_index(track_name=track_name, artist=artist)
        if exact_result is not None:
            return exact_result, 1.0  # Perfect confidence
        
        # 2. Fuzzy matching
        best_match = None
        best_score = 0.0
        threshold = 0.75
        
        # Normalize inputs
        track_name_norm = track_name.lower().strip()
        artist_norm = artist.lower().strip() if artist else None
        
        # Create normalized columns for comparison
        temp_df = self.tracks_df.copy()
        temp_df['name_norm'] = temp_df['name'].str.lower().str.strip()
        temp_df['artist_norm'] = temp_df['artist'].str.lower().str.strip()
        
        for idx, row in temp_df.iterrows():
            # Name similarity
            name_sim = SequenceMatcher(None, track_name_norm, row['name_norm']).ratio()
            
            if artist_norm:
                # Combined name + artist similarity
                artist_sim = SequenceMatcher(None, artist_norm, row['artist_norm']).ratio()
                combined_score = (name_sim * 0.7) + (artist_sim * 0.3)
            else:
                combined_score = name_sim
                
            if combined_score > best_score and combined_score >= threshold:
                best_score = combined_score
                best_match = idx
        
        return best_match, best_score

    def _calculate_enhanced_similarity(self, seed_track, candidates_df):
        """Calculate enhanced similarity with multiple factors"""
        enhanced_scores = []
        
        # Get seed track data
        if isinstance(seed_track, (int, np.integer)):
            seed_data = self.tracks_df.iloc[seed_track]
        else:
            seed_data = seed_track
        
        for idx, candidate in candidates_df.iterrows():
            # 1. Base content similarity (from original model)
            base_sim = 0.5  # Default similarity
            if hasattr(self.content_recommender, 'similarity_matrix') and self.content_recommender.similarity_matrix is not None:
                try:
                    if isinstance(seed_track, (int, np.integer)):
                        base_sim = self.content_recommender.similarity_matrix[seed_track][idx]
                    else:
                        # Find seed index
                        seed_idx = self.tracks_df[self.tracks_df['id'] == seed_data.get('id', '')].index
                        if len(seed_idx) > 0:
                            base_sim = self.content_recommender.similarity_matrix[seed_idx[0]][idx]
                except (IndexError, KeyError):
                    base_sim = 0.5
            
            # 2. Popularity factor (prefer similar popularity)
            seed_pop = seed_data.get('popularity', 50)
            cand_pop = candidate.get('popularity', 50)
            pop_diff = abs(seed_pop - cand_pop) / 100
            popularity_factor = max(0.3, 1 - pop_diff)  # Min 30% score
            
            # 3. Artist diversity bonus (avoid same artist)
            artist_penalty = 0.7 if seed_data['artist'] == candidate['artist'] else 1.0
            
            # 4. Release year proximity
            seed_year = seed_data.get('release_year', 2020)
            cand_year = candidate.get('release_year', 2020)
            year_diff = abs(seed_year - cand_year)
            year_factor = max(0.5, 1 - (year_diff / 25))  # 25 years = 50% penalty
            
            # 5. Language consistency bonus
            language_bonus = 1.0
            for lang in ['is_vietnamese', 'is_korean', 'is_japanese', 'is_spanish']:
                if (seed_data.get(lang, 0) == 1 and candidate.get(lang, 0) == 1):
                    language_bonus = 1.3  # 30% bonus for same language
                    break
            
            # Combine all factors
            enhanced_score = (
                base_sim * 0.5 +
                popularity_factor * 0.2 +
                year_factor * 0.2 +
                0.1  # Base score
            ) * artist_penalty * language_bonus
            
            enhanced_scores.append(min(enhanced_score, 1.0))  # Cap at 1.0
        
        return enhanced_scores

    def _smart_fallback(self, track_name, artist, n_recommendations):
        """Intelligent fallback when exact track not found"""
        fallback_results = []
        
        # 1. Artist-based fallback
        if artist:
            artist_tracks = self.tracks_df[
                self.tracks_df['artist'].str.contains(artist, case=False, na=False, regex=False)
            ]
            if not artist_tracks.empty:
                fallback_results.append(('artist_match', artist_tracks))
        
        # 2. Partial name match fallback
        if track_name:
            # Split track name into words for partial matching
            words = track_name.lower().split()
            for word in words:
                if len(word) > 3:  # Only use meaningful words
                    partial_matches = self.tracks_df[
                        self.tracks_df['name'].str.contains(word, case=False, na=False, regex=False)
                    ]
                    if not partial_matches.empty:
                        fallback_results.append(('partial_name', partial_matches))
                        break  # Use first meaningful match
        
        # 3. Language-based fallback
        # Try to detect Vietnamese language from track name
        vietnamese_chars = 'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'
        if any(char in track_name.lower() for char in vietnamese_chars):
            vietnamese_tracks = self.tracks_df[self.tracks_df.get('is_vietnamese', 0) == 1]
            if not vietnamese_tracks.empty:
                fallback_results.append(('language_vietnamese', vietnamese_tracks))
        
        # 4. Genre-based fallback (from track name keywords)
        genre_keywords = {
            'pop': 'genre_pop',
            'rock': 'genre_rock', 
            'rap': 'genre_hip_hop',
            'ballad': 'genre_ballad',
            'jazz': 'genre_jazz'
        }
        
        for keyword, genre_col in genre_keywords.items():
            if keyword in track_name.lower() and genre_col in self.tracks_df.columns:
                genre_tracks = self.tracks_df[self.tracks_df[genre_col] > 0]
                if not genre_tracks.empty:
                    fallback_results.append((f'genre_{keyword}', genre_tracks))
                    break
        
        # 5. Select best fallback
        if fallback_results:
            # Prioritize by fallback type
            priority_order = ['artist_match', 'partial_name', 'language_vietnamese']
            
            for priority_type in priority_order:
                for fallback_type, fallback_df in fallback_results:
                    if fallback_type.startswith(priority_type.split('_')[0]):
                        # Sort by popularity and return top results
                        if 'popularity' in fallback_df.columns:
                            result = fallback_df.nlargest(n_recommendations, 'popularity')
                        else:
                            result = fallback_df.sample(min(n_recommendations, len(fallback_df)))
                        
                        result = result.copy()
                        result['enhanced_score'] = 0.6  # Medium confidence for fallback
                        result['fallback_type'] = fallback_type
                        return result
        
        # 6. Final fallback - popular tracks
        if 'popularity' in self.tracks_df.columns:
            popular_tracks = self.tracks_df.nlargest(n_recommendations, 'popularity').copy()
        else:
            popular_tracks = self.tracks_df.sample(min(n_recommendations, len(self.tracks_df))).copy()
        
        popular_tracks['enhanced_score'] = 0.3  # Low confidence
        popular_tracks['fallback_type'] = 'popular_random'
        return popular_tracks

    def _log_recommendation_quality(self, recommendations, seed_track=None, search_confidence=None, input_method='standard'):
        """Log detailed quality metrics for analysis"""
        if recommendations.empty:
            logger.warning("Empty recommendations provided to quality check")
            return {}
        
        metrics = {}
        
        # 1. Basic metrics
        metrics['total_recommendations'] = len(recommendations)
        metrics['search_confidence'] = search_confidence or 0.0
        metrics['input_method'] = input_method
        
        # 2. Artist diversity
        unique_artists = recommendations['artist'].nunique()
        metrics['artist_diversity'] = unique_artists / len(recommendations)
        
        # 3. Score distribution
        if 'enhanced_score' in recommendations.columns:
            metrics['avg_enhanced_score'] = recommendations['enhanced_score'].mean()
            metrics['min_enhanced_score'] = recommendations['enhanced_score'].min()
            metrics['max_enhanced_score'] = recommendations['enhanced_score'].max()
        elif 'content_score' in recommendations.columns:
            metrics['avg_content_score'] = recommendations['content_score'].mean()
        
        # 4. Popularity distribution
        if 'popularity' in recommendations.columns:
            metrics['avg_popularity'] = recommendations['popularity'].mean()
            metrics['popularity_std'] = recommendations['popularity'].std()
        
        # 5. Year diversity
        if 'release_year' in recommendations.columns:
            year_range = recommendations['release_year'].max() - recommendations['release_year'].min()
            metrics['year_diversity'] = min(1.0, year_range / 30)  # Normalize by 30 years
        
        # 6. Language consistency (if seed provided)
        if seed_track is not None:
            for lang in ['is_vietnamese', 'is_korean', 'is_japanese']:
                if hasattr(seed_track, 'get') and seed_track.get(lang, 0) == 1 and lang in recommendations.columns:
                    same_lang_count = (recommendations[lang] == 1).sum()
                    metrics[f'{lang}_consistency'] = same_lang_count / len(recommendations)
        
        # 7. Log comprehensive metrics
        logger.info("=== RECOMMENDATION QUALITY METRICS ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.3f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=====================================")
        
        return metrics

    def recommend(self, track_name=None, artist=None, n_recommendations=10):
        """Enhanced recommendation with smart search and multi-factor scoring"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        n_recommendations = min(n_recommendations, len(self.tracks_df) - 1)
        
        # 1. Enhanced search with confidence scoring
        track_idx, search_confidence = self._find_track_with_fuzzy(track_name, artist)
        
        if track_idx is None:
            logger.warning(f"Track '{track_name}' not found with fuzzy search, using smart fallback")
            recommendations = self._smart_fallback(track_name, artist, n_recommendations)
            
            # Add quality metrics for fallback
            self._log_recommendation_quality(recommendations, input_method='smart_fallback')
            
            # Return clean result
            result_cols = ['name', 'artist', 'enhanced_score', 'popularity', 'release_year']
            available_cols = [col for col in result_cols if col in recommendations.columns]
            return recommendations[available_cols].round(3)
        
        # 2. Get seed track info
        seed_track = self.tracks_df.iloc[track_idx]
        logger.info(f"Found track with {search_confidence:.3f} confidence: '{seed_track['name']}' by {seed_track['artist']}")
        
        # 3. Get base recommendations from content model
        try:
            base_recs = self.content_recommender.recommend(
                track_name=track_name, 
                artist=artist, 
                n_recommendations=min(n_recommendations * 2, len(self.tracks_df) - 1)  # Get more for filtering
            )
        except Exception as e:
            logger.error(f"Error getting base recommendations: {e}")
            return self._smart_fallback(track_name, artist, n_recommendations)
        
        if base_recs.empty:
            return self._smart_fallback(track_name, artist, n_recommendations)
        
        # 4. Enhanced scoring
        enhanced_scores = self._calculate_enhanced_similarity(track_idx, base_recs)
        base_recs['enhanced_score'] = enhanced_scores
        
        # 5. Re-rank by enhanced score
        final_recommendations = base_recs.sort_values('enhanced_score', ascending=False).head(n_recommendations)
        
        # 6. Add metadata and quality metrics
        final_recommendations['search_confidence'] = search_confidence
        final_recommendations['recommendation_method'] = 'enhanced_content'
        
        # Log quality metrics
        self._log_recommendation_quality(final_recommendations, seed_track, search_confidence)
        
        # Return clean result
        result_cols = ['name', 'artist', 'enhanced_score', 'popularity', 'release_year']
        available_cols = [col for col in result_cols if col in final_recommendations.columns]
        
        return final_recommendations[available_cols].round(3)

    def generate_playlist_from_seed(self, seed_track, seed_artist="", n_recommendations=10):
        """Generate a playlist from a seed track using enhanced recommendations"""
        try:
            # Get enhanced recommendations instead of using content recommender directly
            recommendations = self.recommend(seed_track, seed_artist, n_recommendations)
            
            if recommendations.empty:
                logger.warning(f"No recommendations found for seed track '{seed_track}'")
                return None, None
            
            # Create queue from recommendations
            queue = recommendations.copy()
            
            # Add the seed track at the beginning if found
            track_idx, _ = self._find_track_with_fuzzy(seed_track, seed_artist)
            if track_idx is not None:
                seed_row = self.tracks_df.iloc[track_idx:track_idx+1].copy()
                seed_row['enhanced_score'] = 1.0  # Perfect score for seed
                queue = pd.concat([seed_row, queue], ignore_index=True)
            
            # Generate analysis
            analysis = self._analyze_playlist_quality(queue)
            
            return queue, analysis
        except Exception as e:
            logger.error(f"Error generating playlist: {e}")
            return None, None

    def explore_by_genre(self, genre, n_recommendations=10):
        """Enhanced genre exploration with smart filtering"""
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            return pd.DataFrame()
            
        # 1. Check for specific genre columns
        genre_col = None
        genre_normalized = genre.lower().replace(" ", "_")
        
        for col in self.tracks_df.columns:
            if f'genre_{genre_normalized}' == col:
                genre_col = col
                break
        
        # 2. Filter by genre column if available
        if genre_col is not None:
            filtered = self.tracks_df[self.tracks_df[genre_col] > 0]
            logger.info(f"Found {len(filtered)} tracks with genre column '{genre_col}'")
        else:
            # 3. Search in artist_genres if available
            if 'artist_genres' in self.tracks_df.columns:
                genre_keyword = genre.lower()
                filtered = self.tracks_df[self.tracks_df['artist_genres'].str.contains(
                    genre_keyword, case=False, na=False, regex=False)]
                logger.info(f"Found {len(filtered)} tracks with genre keyword in artist_genres")
            else:
                # 4. Search in track names and artist names
                filtered = self.tracks_df[
                    self.tracks_df['name'].str.contains(genre, case=False, na=False, regex=False) |
                    self.tracks_df['artist'].str.contains(genre, case=False, na=False, regex=False)
                ]
                logger.info(f"Found {len(filtered)} tracks with genre keyword in names")
        
        if filtered.empty:
            logger.warning(f"No tracks found for genre: {genre}")
            return pd.DataFrame()
        
        # 5. Enhanced ranking
        result = filtered.copy()
        
        # Calculate enhanced scores for genre exploration
        result['enhanced_score'] = 1.0  # Base score for exact genre match
        
        # Boost popular tracks slightly
        if 'popularity' in result.columns:
            popularity_boost = result['popularity'] / 100 * 0.2  # Max 20% boost
            result['enhanced_score'] += popularity_boost
        
        # Sort by enhanced score and popularity
        if 'popularity' in result.columns:
            result = result.sort_values(['enhanced_score', 'popularity'], ascending=[False, False])
        else:
            result = result.sort_values('enhanced_score', ascending=False)
            
        # Take top recommendations
        final_result = result.head(n_recommendations)
        
        # Log quality metrics
        self._log_recommendation_quality(final_result, input_method=f'genre_exploration_{genre}')
        
        # Return clean result
        result_cols = ['name', 'artist', 'enhanced_score', 'popularity', 'release_year']
        available_cols = [col for col in result_cols if col in final_result.columns]
        
        return final_result[available_cols].round(3)
    
    def _analyze_playlist_quality(self, playlist_df):
        """Enhanced playlist quality analysis"""
        analysis = []
        
        if playlist_df is None or playlist_df.empty or len(playlist_df) < 2:
            return pd.DataFrame()
            
        # Calculate transitions between consecutive tracks
        for i in range(len(playlist_df) - 1):
            current = playlist_df.iloc[i]
            next_track = playlist_df.iloc[i+1]
            
            # Enhanced transition quality assessment
            quality_score = 0.5  # Base score
            quality_factors = []
            
            # 1. Artist diversity factor
            if current['artist'] != next_track['artist']:
                quality_score += 0.3
                quality_factors.append("artist_diversity")
            else:
                quality_score -= 0.1
                quality_factors.append("same_artist")
            
            # 2. Year proximity factor
            if 'release_year' in current and 'release_year' in next_track:
                year_diff = abs(current.get('release_year', 2020) - next_track.get('release_year', 2020))
                if year_diff <= 5:
                    quality_score += 0.2
                    quality_factors.append("similar_era")
                elif year_diff > 20:
                    quality_score -= 0.1
                    quality_factors.append("different_era")
            
            # 3. Popularity balance factor
            if 'popularity' in current and 'popularity' in next_track:
                pop_diff = abs(current.get('popularity', 50) - next_track.get('popularity', 50))
                if pop_diff <= 20:
                    quality_score += 0.1
                    quality_factors.append("balanced_popularity")
            
            # 4. Enhanced score factor
            if 'enhanced_score' in current and 'enhanced_score' in next_track:
                avg_score = (current.get('enhanced_score', 0.5) + next_track.get('enhanced_score', 0.5)) / 2
                if avg_score >= 0.7:
                    quality_score += 0.2
                    quality_factors.append("high_similarity")
            
            # Normalize quality score
            quality_score = max(0.0, min(1.0, quality_score))
            
            # Determine quality label
            if quality_score >= 0.8:
                quality = "Excellent"
            elif quality_score >= 0.6:
                quality = "Good"
            elif quality_score >= 0.4:
                quality = "Fair"
            else:
                quality = "Poor"
            
            from_track = f"{current['name']} - {current['artist']}"
            to_track = f"{next_track['name']} - {next_track['artist']}"
            
            analysis.append({
                'from_track': from_track,
                'to_track': to_track,
                'transition_score': quality_score,
                'quality': quality,
                'factors': ', '.join(quality_factors)
            })
            
        return pd.DataFrame(analysis)

    def _validate_n_recommendations(self, n):
        """Validate and return a valid number of recommendations"""
        try:
            n = int(n)
            return max(1, min(n, len(self.tracks_df) - 1 if self.tracks_df is not None else 100))
        except (ValueError, TypeError):
            logger.warning(f"Invalid n_recommendations value: {n}. Using default value 10.")
            return 10
