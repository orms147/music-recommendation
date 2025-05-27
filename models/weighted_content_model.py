import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from models.base_model import BaseRecommender

logger = logging.getLogger(__name__)

class WeightedContentRecommender(BaseRecommender):
    """Content-based recommender with country-first prioritization"""
    
    def __init__(self, weights=None):
        super().__init__(name="WeightedContentRecommender")
        self.tracks_df = None
        self.genre_matrix = None
        self.is_trained = False
        
        # Điều chỉnh trọng số với ưu tiên cao cho cùng quốc gia
        self.weights = weights or {
            "same_artist": 0.30,           # Ưu tiên cao nhất cho cùng nghệ sĩ
            "genre_similarity": 0.25,       # Ưu tiên thứ hai cho cùng thể loại
            "same_country": 0.20,          # Ưu tiên cao thứ ba cho cùng quốc gia (từ ISRC)
            "same_region": 0.10,           # Ưu tiên thứ tư cho cùng khu vực (từ ISRC)
            "popularity": 0.07,             # Độ phổ biến bài hát
            "artist_popularity": 0.05,      # Độ nổi tiếng nghệ sĩ
            "release_recency": 0.02,        # Độ mới của bài hát
            "duration_similarity": 0.01,    # Độ tương tự về thời lượng
        }
        
        # Cấu hình đa dạng hóa kết quả
        self.diversity_config = {
            "max_same_artist": 6,           # Tối đa 5 bài từ cùng nghệ sĩ
            "country_ratio": 0.6,           # 60% bài hát cùng quốc gia
            "region_ratio": 0.8,            # 80% bài hát cùng khu vực
            "genre_consistency": True,      # Ưu tiên tính nhất quán thể loại
        }
        
        # Định nghĩa bản đồ khu vực từ mã quốc gia ISRC
        self.region_map = {
            # Khu vực Bắc Mỹ
            'US': 'north_america', 'CA': 'north_america',
            
            # Khu vực Châu Âu
            'GB': 'europe', 'DE': 'europe', 'FR': 'europe', 'IT': 'europe', 
            'ES': 'europe', 'NL': 'europe', 'SE': 'europe', 'NO': 'europe',
            'DK': 'europe', 'FI': 'europe', 'PT': 'europe', 'IE': 'europe',
            'BE': 'europe', 'AT': 'europe', 'CH': 'europe', 'PL': 'europe',
            'GR': 'europe', 'CZ': 'europe', 'HU': 'europe', 'RO': 'europe',
            
            # Khu vực Châu Á
            'JP': 'asia', 'KR': 'asia', 'CN': 'asia', 'HK': 'asia', 
            'TW': 'asia', 'VN': 'asia', 'TH': 'asia', 'MY': 'asia',
            'SG': 'asia', 'ID': 'asia', 'PH': 'asia', 'IN': 'asia',
            
            # Khu vực Mỹ Latinh
            'MX': 'latin_america', 'BR': 'latin_america', 'AR': 'latin_america',
            'CO': 'latin_america', 'CL': 'latin_america', 'PE': 'latin_america',
            'VE': 'latin_america', 'EC': 'latin_america', 'UY': 'latin_america',
            
            # Khu vực Châu Úc
            'AU': 'oceania', 'NZ': 'oceania',
            
            # Khu vực Châu Phi
            'ZA': 'africa', 'NG': 'africa', 'EG': 'africa', 'MA': 'africa',
        }

    def train(self, tracks_df):
        """Train model with enhanced features"""
        if tracks_df is None or tracks_df.empty:
            logger.error("Cannot train model: Empty dataframe")
            return False
        
        self.tracks_df = tracks_df.copy()
        
        # Cải thiện: Thêm xử lý đặc trưng mới
        self._preprocess_features()
        self._create_genre_matrix()
        
        self.is_trained = True
        logger.info(f"Model trained with {len(self.tracks_df)} tracks and enhanced features")
        return True

    def _preprocess_features(self):
        """Preprocess and normalize features with enhanced metrics"""
        df = self.tracks_df
        
        # Cải thiện: Thêm xử lý độ mới của bài hát
        if 'release_date' in df.columns:
            current_year = datetime.now().year
            df['release_year'] = df['release_year'].fillna(current_year - 5)
            df['release_recency'] = 1 - ((current_year - df['release_year']) / 50).clip(0, 1)
        
        # Chuẩn hóa các đặc trưng
        for feature in ['popularity', 'artist_popularity']:
            if feature in df.columns:
                df[f'{feature}_norm'] = df[feature].fillna(df[feature].median()) / 100
        
        # Chuẩn hóa thời lượng
        if 'duration_ms' in df.columns:
            mean_duration = df['duration_ms'].mean()
            std_duration = df['duration_ms'].std()
            df['duration_norm'] = (df['duration_ms'] - mean_duration) / (4 * std_duration) + 0.5
            df['duration_norm'] = df['duration_norm'].clip(0, 1)
        
        # Đảm bảo có cột quốc gia và khu vực từ ISRC
        if 'isrc' in df.columns and 'isrc_country' not in df.columns:
            # Trích xuất mã quốc gia từ ISRC (2 ký tự đầu)
            df['isrc_country'] = df['isrc'].str[:2].fillna('XX')
            
            # Ánh xạ mã quốc gia sang khu vực
            def map_to_region(country_code):
                return self.region_map.get(country_code, 'other')
                
            df['region'] = df['isrc_country'].apply(map_to_region)
            
            # Thống kê phân bố quốc gia và khu vực
            country_counts = df['isrc_country'].value_counts().head(10)
            region_counts = df['region'].value_counts()
            logger.info(f"Top 10 countries: {dict(country_counts)}")
            logger.info(f"Region distribution: {dict(region_counts)}")

    def _create_genre_matrix(self):
        """Create genre matrix for similarity calculation"""
        df = self.tracks_df
        
        # Tìm tất cả các cột thể loại
        genre_cols = [col for col in df.columns if col.startswith('genre_')]
        
        if not genre_cols:
            # Nếu không có cột thể loại, tạo ma trận đơn vị
            self.genre_matrix = np.ones((len(df), 1))
            logger.warning("No genre columns found, using unit matrix")
            return
        
        # Tạo ma trận thể loại
        self.genre_matrix = df[genre_cols].fillna(0).values
        logger.info(f"Created genre matrix with {len(genre_cols)} genres")

    def find_track_idx(self, track_name, artist=None):
        """Find track index with improved matching"""
        if self.tracks_df is None or track_name is None:
            return None
            
        # Chuẩn hóa tên bài hát và nghệ sĩ
        track_name_norm = track_name.lower().strip()
        
        # Tìm kiếm chính xác
        mask = self.tracks_df['name'].str.lower().str.strip() == track_name_norm
        if artist:
            artist_norm = artist.lower().strip()
            mask = mask & (self.tracks_df['artist'].str.lower().str.strip() == artist_norm)
            
        matches = self.tracks_df[mask]
        
        if not matches.empty:
            return matches.index[0]
            
        # Tìm kiếm mờ nếu không có kết quả chính xác
        if not artist:
            # Tìm kiếm bài hát có chứa tên
            mask = self.tracks_df['name'].str.lower().str.contains(track_name_norm, regex=False)
            matches = self.tracks_df[mask]
            if not matches.empty:
                # Trả về bài hát phổ biến nhất
                if 'popularity' in matches.columns:
                    return matches.loc[matches['popularity'].idxmax()].name
                else:
                    return matches.index[0]
        
        return None

    def recommend(self, track_name=None, artist=None, n_recommendations=10, popularity_filter=None):
        """Generate recommendations using ISRC region matching"""
        if not self.is_trained:
            logger.error("Model not trained")
            return pd.DataFrame()

        n_recommendations = min(max(n_recommendations, 1), 100)
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
        
        # ✅ Áp dụng bộ lọc độ phổ biến nếu được chỉ định
        filtered_df = df.copy()
        
        # Lấy độ phổ biến của bài hát gốc
        seed_popularity = seed.get('popularity', 50)
        
        # Áp dụng bộ lọc độ phổ biến thông minh
        if popularity_filter == 'similar':
            # Chỉ lấy bài hát có độ phổ biến tương tự (±20)
            pop_min = max(0, seed_popularity - 20)
            pop_max = min(100, seed_popularity + 20)
            filtered_df = filtered_df[(filtered_df['popularity'] >= pop_min) & 
                                     (filtered_df['popularity'] <= pop_max)]
        elif popularity_filter == 'higher':
            # Chỉ lấy bài hát có độ phổ biến cao hơn
            pop_min = seed_popularity
            filtered_df = filtered_df[filtered_df['popularity'] >= pop_min]
        elif popularity_filter == 'lower':
            # Chỉ lấy bài hát có độ phổ biến thấp hơn
            pop_max = seed_popularity
            filtered_df = filtered_df[filtered_df['popularity'] <= pop_max]
        elif popularity_filter == 'top':
            # Chỉ lấy bài hát phổ biến nhất (top 25%)
            pop_threshold = df['popularity'].quantile(0.75)
            filtered_df = filtered_df[filtered_df['popularity'] >= pop_threshold]
        elif popularity_filter == 'niche':
            # Chỉ lấy bài hát ít phổ biến (bottom 50%)
            pop_threshold = df['popularity'].quantile(0.5)
            filtered_df = filtered_df[filtered_df['popularity'] <= pop_threshold]
        
        # Nếu bộ lọc quá nghiêm ngặt và không còn đủ bài hát, quay lại dùng toàn bộ dữ liệu
        if len(filtered_df) < n_recommendations * 2:
            logger.warning(f"Popularity filter too restrictive, using full dataset")
            filtered_df = df.copy()

        # ✅ 1. SAME ARTIST BONUS - Ưu tiên cao nhất
        same_artist_mask = filtered_df['artist'] == seed['artist']
        
        # ✅ 2. REGION SIMILARITY - Ưu tiên cao thứ hai (đơn giản hóa từ ISRC)
        seed_region = seed.get('region', 'other')
        same_region_mask = filtered_df['region'] == seed_region
        
        # ✅ 3. GENRE SIMILARITY - Ưu tiên tính nhất quán thể loại
        if self.genre_matrix.shape[1] > 1:
            # Lấy vector thể loại của bài hát gốc
            seed_genres = self.genre_matrix[idx].reshape(1, -1)
            
            # Lấy các chỉ số của bài hát trong filtered_df
            filtered_indices = filtered_df.index.tolist()
            
            # Tính toán độ tương tự thể loại chỉ cho các bài hát trong filtered_df
            filtered_genre_matrix = self.genre_matrix[filtered_indices]
            genre_sim = cosine_similarity(seed_genres, filtered_genre_matrix)[0]
        else:
            genre_sim = np.ones(len(filtered_df))
        
        # ✅ 4. POPULARITY SIMILARITY
        if 'popularity' in filtered_df.columns:
            # Chuẩn hóa độ chênh lệch về độ phổ biến
            pop_diff = np.abs(filtered_df['popularity'].values - seed_popularity) / 100
            # Chuyển đổi thành độ tương tự (1 = giống nhau hoàn toàn, 0 = khác nhau hoàn toàn)
            popularity_sim = 1 - pop_diff
        else:
            popularity_sim = np.ones(len(filtered_df)) * 0.5
        
        # ✅ 5. OTHER FEATURES
        track_pop = filtered_df['popularity_norm'].values if 'popularity_norm' in filtered_df.columns else np.ones(len(filtered_df)) * 0.5
        artist_pop = filtered_df['artist_popularity_norm'].values if 'artist_popularity_norm' in filtered_df.columns else np.ones(len(filtered_df)) * 0.5
        
        # Duration similarity
        if 'duration_norm' in filtered_df.columns and 'duration_norm' in seed:
            seed_duration = seed['duration_norm']
            duration_diff = np.abs(filtered_df['duration_norm'].values - seed_duration)
            duration_sim = np.exp(-2 * duration_diff)
        else:
            duration_sim = np.ones(len(filtered_df)) * 0.5
        
        # Release recency
        if 'release_recency' in filtered_df.columns:
            recency = filtered_df['release_recency'].values
        else:
            recency = np.ones(len(filtered_df)) * 0.5  # Default value
        
        # ✅ 6. WEIGHTED COMBINATION
        # Base score without same artist bonus
        base_score = (
            self.weights["same_region"] * same_region_mask.astype(float) +
            self.weights["genre_similarity"] * genre_sim +
            self.weights["popularity"] * (track_pop * popularity_sim) +
            self.weights["artist_popularity"] * artist_pop +
            self.weights["release_recency"] * recency +
            self.weights["duration_similarity"] * duration_sim
        )
        
        # Add same artist bonus
        final_score = np.where(same_artist_mask, base_score + self.weights["same_artist"], base_score)
        
        # Tạo kết quả
        df = filtered_df.copy()
        df['final_score'] = final_score
        df['same_artist'] = same_artist_mask
        df['same_region'] = same_region_mask
        
        # Loại bỏ bài hát gốc
        df = df.drop(idx)
        
        # Cải thiện: Đa dạng hóa kết quả
        recommendations = self._diversify_recommendations(
            df, seed, n_recommendations
        )

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

    def _diversify_recommendations(self, candidates_df, seed_track, n_recommendations):
        """Đa dạng hóa kết quả đề xuất"""
        sorted_df = candidates_df.sort_values('final_score', ascending=False)
        
        # ✅ SỬA: Sử dụng music_culture thay vì language
        seed_culture = seed_track.get('music_culture', 'other')
        seed_artist = seed_track.get('artist', '')
        
        # ✅ SỬA: Sử dụng country_ratio
        same_culture_count = int(n_recommendations * self.diversity_config["country_ratio"])
        other_culture_count = n_recommendations - same_culture_count
        
        # ✅ SỬA: Filter by music_culture
        same_culture = sorted_df[sorted_df.get('music_culture', 'other') == seed_culture]
        other_culture = sorted_df[sorted_df.get('music_culture', 'other') != seed_culture]
        
        # Giới hạn số bài hát từ cùng nghệ sĩ
        artist_counts = {}
        
        final_recommendations = []
        
        # Thêm bài hát cùng văn hóa
        for _, track in same_culture.iterrows():
            artist = track['artist']
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            
            # Kiểm tra giới hạn cùng nghệ sĩ
            if artist == seed_artist and artist_counts[artist] > self.diversity_config["max_same_artist"]:
                continue
                
            final_recommendations.append(track)
            if len(final_recommendations) >= same_culture_count:
                break
        
        # Thêm bài hát khác văn hóa
        for _, track in other_culture.iterrows():
            artist = track['artist']
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            
            # Kiểm tra giới hạn cùng nghệ sĩ
            if artist == seed_artist and artist_counts[artist] > self.diversity_config["max_same_artist"]:
                continue
                
            final_recommendations.append(track)
            if len(final_recommendations) >= n_recommendations:
                break
        
        # Nếu không đủ bài hát, lấy thêm từ danh sách đã sắp xếp
        if len(final_recommendations) < n_recommendations:
            remaining = n_recommendations - len(final_recommendations)
            
            # Lấy các bài hát chưa được chọn
            remaining_tracks = sorted_df[~sorted_df.index.isin([r.name for r in final_recommendations])]
            
            for _, track in remaining_tracks.iterrows():
                final_recommendations.append(track)
                if len(final_recommendations) >= n_recommendations:
                    break
        
        return pd.DataFrame(final_recommendations)


# ✅ ALIAS for backward compatibility
ContentBasedRecommender = WeightedContentRecommender
