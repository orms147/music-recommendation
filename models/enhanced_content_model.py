import pandas as pd
import numpy as np
import logging
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
        
        # Cải tiến: Thêm cấu hình cho văn hóa âm nhạc
        self.cultural_config = {
            # Ưu tiên theo thứ tự: quốc gia > thể loại > khu vực
            "country_boost": 0.35,      # Tăng điểm cho cùng quốc gia
            "genre_boost": 0.25,        # Tăng điểm cho cùng thể loại
            "region_boost": 0.15,       # Tăng điểm cho cùng khu vực
            
            # Cấu hình đa dạng hóa kết quả
            "same_country_ratio": 0.6,  # 60% kết quả cùng quốc gia
            "same_region_ratio": 0.8,   # 80% kết quả cùng khu vực
            
            # Cấu hình tối ưu hóa hiệu suất
            "use_numba": True,          # Sử dụng Numba để tăng tốc
            "use_eval": True,           # Sử dụng pandas.eval() cho DataFrame lớn
        }
        
        # Cải tiến: Cache kết quả để tăng tốc
        self.recommendation_cache = {}
        self.cache_size_limit = 1000
    
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
        
        # Cải tiến: Phân tích chất lượng đặc trưng
        self._analyze_feature_quality()
        
        # Cải tiến: Tối ưu hóa hiệu suất với Numba
        if self.cultural_config["use_numba"] and self._check_numba_available():
            logger.info("Using Numba for performance optimization")
            self._optimize_with_numba()
        
        # Train base content model
        logger.info("Training base WeightedContentRecommender...")
        success = self.content_recommender.train(tracks_df)
        
        if not success:
            logger.error("Failed to train base content recommender")
            return False
        
        # Cải tiến: Tính toán thời gian huấn luyện
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Enhanced model training completed in {training_time:.2f} seconds")
        
        self.is_trained = True
        return True
    
    def _check_numba_available(self):
        """Check if Numba is available for JIT compilation"""
        try:
            import numba
            return True
        except ImportError:
            logger.warning("Numba not available. Install with 'pip install numba' for performance boost.")
            return False
    
    def _optimize_with_numba(self):
        """Apply Numba optimizations to critical functions"""
        try:
            import numba as nb
            
            # JIT compile similarity calculation function
            @nb.jit(nopython=True, parallel=True)
            def fast_similarity(vec1, matrix):
                """Numba-optimized similarity calculation"""
                result = np.zeros(matrix.shape[0])
                for i in nb.prange(matrix.shape[0]):
                    dot_product = 0.0
                    norm1 = 0.0
                    norm2 = 0.0
                    for j in range(matrix.shape[1]):
                        dot_product += vec1[j] * matrix[i, j]
                        norm1 += vec1[j] * vec1[j]
                        norm2 += matrix[i, j] * matrix[i, j]
                    
                    if norm1 > 0 and norm2 > 0:
                        result[i] = dot_product / (np.sqrt(norm1) * np.sqrt(norm2))
                    else:
                        result[i] = 0.0
                return result
            
            # Lưu hàm đã biên dịch JIT
            self._fast_similarity = fast_similarity
            logger.info("Successfully compiled similarity function with Numba")
            
        except Exception as e:
            logger.error(f"Error optimizing with Numba: {e}")
    
    def _analyze_feature_quality(self):
        """Analyze feature quality for better recommendations"""
        if self.tracks_df is None:
            return
        
        # Cải tiến: Phân tích chất lượng đặc trưng
        feature_quality = {}
        
        # Phân tích độ đầy đủ của dữ liệu
        for col in self.tracks_df.columns:
            non_null_ratio = self.tracks_df[col].notnull().mean()
            feature_quality[col] = {
                'non_null_ratio': non_null_ratio,
                'importance': 0.0  # Sẽ được cập nhật sau
            }
        
        # Đánh giá tầm quan trọng của đặc trưng
        important_features = [
            ('isrc_country', 0.9),  # Quốc gia (ưu tiên cao nhất)
            ('region', 0.8),        # Khu vực (ưu tiên thứ hai)
            ('genre_', 0.7),        # Thể loại (ưu tiên thứ ba)
            ('popularity', 0.6),
            ('artist_popularity', 0.5),
            ('release_year', 0.4),
            ('duration_ms', 0.3)
        ]
        
        for feature_prefix, importance in important_features:
            for col in self.tracks_df.columns:
                if col.startswith(feature_prefix):
                    if col in feature_quality:
                        feature_quality[col]['importance'] = importance
        
        self.feature_quality = feature_quality
        
        # Log kết quả phân tích
        top_features = sorted(
            [(k, v['importance'] * v['non_null_ratio']) 
             for k, v in feature_quality.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        logger.info(f"Top 10 quality features: {top_features}")

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
        
        # Cải tiến: Kiểm tra cache
        cache_key = f"{track_name}|{artist}|{n_recommendations}"
        if cache_key in self.recommendation_cache:
            logger.info(f"Using cached recommendations for '{track_name}'")
            return self.recommendation_cache[cache_key]
        
        # 1. Enhanced search with cultural context
        track_idx, search_confidence = self._find_track_with_fuzzy(track_name, artist)
        
        if track_idx is None:
            logger.warning(f"Track '{track_name}' not found with enhanced search, using cultural fallback")
            recommendations = self._smart_cultural_fallback(track_name, artist, n_recommendations)
            
            # Cải tiến: Lưu vào cache
            self._update_cache(cache_key, recommendations)
            
            # Return clean result
            result_cols = ['name', 'artist', 'enhanced_score']
            if 'music_culture' in recommendations.columns:
                result_cols.append('music_culture')
            if 'isrc_country' in recommendations.columns:
                result_cols.append('isrc_country')
            if 'region' in recommendations.columns:
                result_cols.append('region')
                
            available_cols = [col for col in result_cols if col in recommendations.columns]
            return recommendations[available_cols].round(3)
        
        # 2. Get seed track info with cultural context
        seed_track = self.tracks_df.iloc[track_idx]
        seed_country = seed_track.get('isrc_country', 'XX')
        seed_region = seed_track.get('region', 'other')
        
        logger.info(f"Found track with {search_confidence:.3f} confidence: '{seed_track['name']}' by {seed_track['artist']} (Country: {seed_country}, Region: {seed_region})")
        
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
        
        # 4. Enhanced scoring with cultural priority
        enhanced_scores = self._calculate_enhanced_similarity(track_idx, base_recs)
        base_recs['enhanced_score'] = enhanced_scores
        
        # 5. Re-rank by enhanced score with cultural diversity
        final_recommendations = base_recs.sort_values('enhanced_score', ascending=False)
        
        # Cải tiến: Đảm bảo đa dạng văn hóa với ưu tiên quốc gia > khu vực
        if 'isrc_country' in final_recommendations.columns:
            # Ưu tiên cùng quốc gia
            same_country_target = int(n_recommendations * self.cultural_config["same_country_ratio"])
            other_target = n_recommendations - same_country_target
            
            same_country_recs = final_recommendations[
                final_recommendations['isrc_country'] == seed_country
            ].head(same_country_target)
            
            # Ưu tiên cùng khu vực cho phần còn lại
            if 'region' in final_recommendations.columns:
                same_region_mask = (
                    (final_recommendations['isrc_country'] != seed_country) & 
                    (final_recommendations['region'] == seed_region)
                )
                same_region_recs = final_recommendations[same_region_mask].head(
                    int(other_target * 0.7)  # 70% của phần còn lại từ cùng khu vực
                )
                
                # Phần còn lại từ các khu vực khác
                other_recs = final_recommendations[
                    ~final_recommendations.index.isin(same_country_recs.index) & 
                    ~final_recommendations.index.isin(same_region_recs.index)
                ].head(other_target - len(same_region_recs))
                
                final_result = pd.concat([same_country_recs, same_region_recs, other_recs])
            else:
                # Nếu không có thông tin khu vực
                other_recs = final_recommendations[
                    ~final_recommendations.index.isin(same_country_recs.index)
                ].head(other_target)
                
                final_result = pd.concat([same_country_recs, other_recs])
        else:
            final_result = final_recommendations.head(n_recommendations)
        
        # 6. Log cultural analytics
        if 'isrc_country' in final_result.columns:
            same_country_count = (final_result['isrc_country'] == seed_country).sum()
            logger.info(f"Enhanced cultural recommendation:")
            logger.info(f"  Same country ({seed_country}): {same_country_count}/{len(final_result)} ({same_country_count/len(final_result)*100:.1f}%)")
            
            country_dist = final_result['isrc_country'].value_counts()
            logger.info(f"  Country distribution: {dict(country_dist)}")
        
        # Cải tiến: Lưu vào cache
        self._update_cache(cache_key, final_result)
        
        # Return clean result
        result_cols = ['name', 'artist', 'enhanced_score']
        if 'isrc_country' in final_result.columns:
            result_cols.append('isrc_country')
        if 'region' in final_result.columns:
            result_cols.append('region')
        available_cols = [col for col in result_cols if col in final_result.columns]
        
        return final_result[available_cols].round(3)
    
    def _update_cache(self, key, value):
        """Update recommendation cache with size limit"""
        self.recommendation_cache[key] = value
        
        # Giới hạn kích thước cache
        if len(self.recommendation_cache) > self.cache_size_limit:
            # Xóa mục cũ nhất
            oldest_key = next(iter(self.recommendation_cache))
            del self.recommendation_cache[oldest_key]

    def _calculate_enhanced_similarity(self, seed_track_idx, candidates_df):
        """Calculate enhanced similarity with cultural priority"""
        enhanced_scores = []
        
        seed_data = self.tracks_df.iloc[seed_track_idx]
        seed_country = seed_data.get('isrc_country', 'XX')
        seed_region = seed_data.get('region', 'other')
        seed_artist = seed_data.get('artist', '')
        seed_culture = seed_data.get('music_culture', 'other')
        
        # Cải tiến: Sử dụng pandas.eval() cho DataFrame lớn
        if self.cultural_config["use_eval"] and len(candidates_df) > 10000:
            try:
                # Tạo DataFrame với các đặc trưng cần thiết
                eval_df = candidates_df.copy()
                
                # Thêm các cột tính toán
                if 'isrc_country' in eval_df.columns:
                    eval_df['same_country'] = (eval_df['isrc_country'] == seed_country).astype(float)
                else:
                    eval_df['same_country'] = 0.0
                
                if 'region' in eval_df.columns:
                    eval_df['same_region'] = (eval_df['region'] == seed_region).astype(float)
                else:
                    eval_df['same_region'] = 0.0
                
                if 'music_culture' in eval_df.columns:
                    eval_df['same_culture'] = (eval_df['music_culture'] == seed_culture).astype(float)
                else:
                    eval_df['same_culture'] = 0.0
            
                eval_df['same_artist'] = (eval_df['artist'] == seed_artist).astype(float)
                
                # Tính toán điểm số nâng cao bằng pandas.eval
                expr = (
                    f"final_score * 0.4 + "
                    f"same_country * {self.cultural_config['country_boost']} + "
                    f"same_region * {self.cultural_config['region_boost']} + "
                    f"same_culture * 0.1 + "  # Thêm boost cho cùng culture
                    f"same_artist * 0.2 + "
                    f"0.2"  # Điểm cơ bản
                )
                
                enhanced_scores = pd.eval(expr, engine='numexpr')
                return enhanced_scores.clip(0, 1).tolist()
                
            except Exception as e:
                logger.warning(f"Error using pandas.eval: {e}. Falling back to standard method.")
        
        # Phương pháp tiêu chuẩn nếu không sử dụng pandas.eval
        for _, candidate in candidates_df.iterrows():
            # 1. Base score from final_score if available
            base_sim = candidate.get('final_score', 0.5)
            
            # 2. Cải tiến: Ưu tiên theo thứ tự quốc gia > thể loại > khu vực
            cultural_boost = 0.0
            
            # ✅ SỬA: Safe access to cultural features
            cand_country = candidate.get('isrc_country', 'XX')
            cand_region = candidate.get('region', 'other')
            cand_culture = candidate.get('music_culture', 'other')
            
            # Ưu tiên cao nhất: Cùng quốc gia
            if seed_country != 'XX' and seed_country == cand_country:
                cultural_boost += self.cultural_config['country_boost']
            
            # Ưu tiên thứ hai: Cùng khu vực
            elif seed_region != 'other' and seed_region == cand_region:
                cultural_boost += self.cultural_config['region_boost']
                
            # Ưu tiên thứ ba: Cùng văn hóa âm nhạc
            elif seed_culture != 'other' and seed_culture == cand_culture:
                cultural_boost += 0.1  # Boost nhỏ hơn cho cùng văn hóa
            
            # 3. Professional quality matching
            quality_boost = 0.0
            if 'is_major_label' in seed_data and 'is_major_label' in candidate:
                if seed_data.get('is_major_label', 0) == candidate.get('is_major_label', 0):
                    quality_boost = 0.1
            
            # 4. Popularity balancing
            seed_pop = seed_data.get('popularity', 50)
            cand_pop = candidate.get('popularity', 50)
            pop_diff = abs(seed_pop - cand_pop) / 100
            popularity_factor = max(0.8, 1 - pop_diff)
            
            # 5. Cải thiện: Tăng điểm cho cùng nghệ sĩ
            same_artist_boost = 0.2 if seed_artist == candidate.get('artist', '') else 0.0
            
            # Kết hợp tất cả yếu tố
            enhanced_score = (
                base_sim * 0.4 +           # Base recommendation score
                cultural_boost +           # Cultural intelligence bonus
                quality_boost +            # Professional quality bonus
                same_artist_boost +        # Same artist bonus
                0.2                        # Base score
            ) * popularity_factor
            
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
