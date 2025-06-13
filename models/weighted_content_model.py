import logging
import pickle
import numpy as np
import pandas as pd
import time
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
            logger.error("No track data provided for training")
            return False
        
        self.tracks_df = tracks_df.copy()
        
        # Cải thiện: Thêm xử lý đặc trưng mới
        self._preprocess_features()
        self._create_genre_matrix()
        
        # REMOVE pre-computation of similarity matrix - too memory intensive
        # Just validate the genre matrix exists
        if self.genre_matrix is not None and self.genre_matrix.shape[1] > 0:
            logger.info(f"Genre matrix ready with {self.genre_matrix.shape[1]} genres")
        else:
            logger.warning("Genre matrix not created properly")
        
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

    def _find_track_with_fuzzy(self, track_name, artist=None):
        """Tìm kiếm bài hát với fuzzy matching và trả về độ tin cậy"""
        if self.tracks_df is None or track_name is None:
            return None, 0.0
        
        try:
            # Chuẩn hóa tên bài hát và nghệ sĩ
            track_name_norm = track_name.lower().strip()
            
            # Tìm kiếm chính xác
            mask = self.tracks_df['name'].str.lower().str.strip() == track_name_norm
            if artist:
                artist_norm = artist.lower().strip()
                mask = mask & (self.tracks_df['artist'].str.lower().str.strip() == artist_norm)
                
            matches = self.tracks_df[mask]
            
            if not matches.empty:
                # Tìm thấy kết quả chính xác
                return matches.index[0], 1.0
            
            # Tìm kiếm mờ
            from difflib import SequenceMatcher
            
            best_match_idx = None
            best_match_score = 0.0
            
            # Tìm kiếm trong tên bài hát
            for idx, row in self.tracks_df.iterrows():
                name_score = SequenceMatcher(None, track_name_norm, row['name'].lower().strip()).ratio()
                
                # Nếu có artist, tính điểm cho artist
                artist_score = 0.0
                if artist and 'artist' in row:
                    artist_score = SequenceMatcher(None, artist.lower().strip(), row['artist'].lower().strip()).ratio()
                    
                # Tính điểm tổng hợp
                if artist:
                    score = (name_score * 0.7) + (artist_score * 0.3)
                else:
                    score = name_score
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = idx
            
            # Trả về kết quả tốt nhất nếu điểm đủ cao
            if best_match_score >= 0.6:
                return best_match_idx, best_match_score
            
            return None, 0.0
        
        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return None, 0.0

    def recommend(self, track_name=None, artist=None, n_recommendations=10, popularity_filter=None):
        """Generate recommendations based on a seed track"""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return pd.DataFrame()
        
        # Đo thời gian thực hiện
        start_time = time.time()
        
        try:
            # Find the track index
            idx = self.find_track_idx(track_name, artist)
            if idx is None:
                logger.warning(f"Track '{track_name}' by '{artist}' not found")
                return pd.DataFrame()
            
            # Get seed track for reference
            seed_track = self.tracks_df.iloc[idx].to_dict()
            
            # TỐI ƯU: Sử dụng mask thay vì tạo DataFrame mới
            mask = self.tracks_df.index != idx
            
            # Apply popularity filter if specified
            if popularity_filter:
                min_pop = popularity_filter.get('min', 0)
                max_pop = popularity_filter.get('max', 100)
                if 'popularity' in self.tracks_df.columns:
                    mask = mask & (self.tracks_df['popularity'] >= min_pop) & (self.tracks_df['popularity'] <= max_pop)
            
            # TỐI ƯU: Tính toán genre similarity chỉ cho các track thỏa mãn mask
            seed_genres = self.genre_matrix[idx].reshape(1, -1)
            
            # TỐI ƯU: Chỉ tính similarity cho các track cần thiết
            candidate_indices = np.where(mask)[0]
            candidate_genres = self.genre_matrix[candidate_indices]
            genre_sim = cosine_similarity(seed_genres, candidate_genres)[0]
            
            # TỐI ƯU: Tạo DataFrame chỉ với các cột cần thiết
            candidates_df = pd.DataFrame({
                'index': candidate_indices,
                'genre_similarity': genre_sim
            })
            
            # TỐI ƯU: Thêm các cột cần thiết từ tracks_df
            for col in ['artist', 'music_culture', 'is_major_label', 'market_penetration', 'kmeans_cluster', 'hdbscan_cluster']:
                if col in self.tracks_df.columns:
                    candidates_df[col] = self.tracks_df.iloc[candidate_indices][col].values
            
            # TỐI ƯU: Tính cultural similarity với vectorization
            if 'music_culture' in candidates_df.columns and 'music_culture' in seed_track:
                seed_culture = seed_track.get('music_culture', 'other')
                candidates_df['same_culture'] = (candidates_df['music_culture'] == seed_culture).astype(float)
                
                # Áp dụng trọng số văn hóa
                candidates_df['same_culture'] = candidates_df['same_culture'] * self.weights.get('same_country', 0.2)
            else:
                candidates_df['same_culture'] = 0.0
            
            # TỐI ƯU: Tính same_artist với vectorization
            if 'artist' in candidates_df.columns and 'artist' in seed_track:
                candidates_df['same_artist'] = (candidates_df['artist'] == seed_track.get('artist', '')).astype(float)
                candidates_df['same_artist'] = candidates_df['same_artist'] * self.weights.get('same_artist', 0.3)
            else:
                candidates_df['same_artist'] = 0.0
            
            # TỐI ƯU: Tính final_score trực tiếp
            candidates_df['genre_similarity_weighted'] = candidates_df['genre_similarity'] * self.weights.get('genre_similarity', 0.25)
            
            # Tính điểm cuối cùng
            candidates_df['final_score'] = (
                candidates_df['genre_similarity_weighted'] + 
                candidates_df['same_culture'] + 
                candidates_df['same_artist']
            )
            
            # TỐI ƯU: Sắp xếp và lấy top N
            top_candidates = candidates_df.sort_values('final_score', ascending=False).head(n_recommendations * 2)
            
            # TỐI ƯU: Đa dạng hóa kết quả với phương pháp đơn giản hơn
            result = self._fast_diversify(top_candidates, seed_track, n_recommendations)
            
            # Lấy thông tin đầy đủ cho các track được chọn
            result_indices = result['index'].values
            final_result = self.tracks_df.loc[result_indices].copy()
            
            # Thêm điểm final_score
            final_result['final_score'] = result['final_score'].values
            final_result['enhanced_score'] = result['final_score'].values
            
            # Ghi log thời gian thực hiện
            elapsed = time.time() - start_time
            logger.info(f"WeightedContentRecommender recommendation completed in {elapsed:.3f} seconds")
            
            # Ghi log thông tin văn hóa
            if 'music_culture' in final_result.columns:
                seed_culture = seed_track.get('music_culture', 'unknown')
                same_culture_count = (final_result['music_culture'] == seed_culture).sum()
                logger.info(f"Cultural recommendation for '{track_name}' ({seed_culture}):")
                logger.info(f"  Same culture: {same_culture_count}/{len(final_result)} ({same_culture_count/len(final_result)*100:.1f}%)")
            
            return final_result
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return pd.DataFrame()

    def _fast_diversify(self, candidates_df, seed_track, n_recommendations):
        """Phương pháp đa dạng hóa nhanh hơn"""
        # Nếu không đủ bài hát, trả về tất cả
        if len(candidates_df) <= n_recommendations:
            return candidates_df
        
        # Lấy thông tin seed
        seed_artist = seed_track.get('artist', '')
        seed_culture = seed_track.get('music_culture', 'unknown')
        seed_cluster = seed_track.get('kmeans_cluster', -1)
        
        # Số lượng bài hát tối đa từ cùng nghệ sĩ
        max_same_artist = min(self.diversity_config.get("max_same_artist", 3), n_recommendations // 3)
        
        # Số lượng bài hát tối đa từ cùng văn hóa
        same_culture_ratio = self.diversity_config.get("country_ratio", 0.6)
        max_same_culture = int(n_recommendations * same_culture_ratio)
        
        # Sắp xếp theo điểm
        sorted_df = candidates_df.sort_values('final_score', ascending=False)
        
        # Khởi tạo bộ đếm
        artist_counts = {}
        culture_counts = {'same': 0, 'other': 0}
        cluster_counts = {'same': 0, 'other': 0}
        
        # Danh sách kết quả
        result = []
        
        # Duyệt qua các bài hát đã sắp xếp
        for _, track in sorted_df.iterrows():
            artist = track.get('artist', '')
            culture = track.get('music_culture', 'unknown')
            cluster = track.get('kmeans_cluster', -1)
            
            # Kiểm tra giới hạn nghệ sĩ
            artist_counts[artist] = artist_counts.get(artist, 0)
            if artist == seed_artist and artist_counts[artist] >= max_same_artist:
                continue
            
            # Kiểm tra giới hạn văn hóa
            if culture == seed_culture:
                if culture_counts['same'] >= max_same_culture:
                    continue
                culture_counts['same'] += 1
            else:
                if culture_counts['other'] >= (n_recommendations - max_same_culture):
                    continue
                culture_counts['other'] += 1
            
            # Kiểm tra giới hạn cluster
            if cluster == seed_cluster:
                if cluster_counts['same'] >= (n_recommendations // 2):
                    continue
                cluster_counts['same'] += 1
            else:
                if cluster_counts['other'] >= (n_recommendations // 2):
                    continue
                cluster_counts['other'] += 1
            
            # Thêm vào kết quả
            result.append(track)
            artist_counts[artist] += 1
            
            # Kiểm tra đủ số lượng
            if len(result) >= n_recommendations:
                break
        
        # Nếu chưa đủ, thêm các bài hát còn lại
        if len(result) < n_recommendations:
            remaining = sorted_df[~sorted_df.index.isin([r.name for r in result])]
            for _, track in remaining.iterrows():
                result.append(track)
                if len(result) >= n_recommendations:
                    break
        
        return pd.DataFrame(result)

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
        """Diversify recommendations using pre-computed clusters"""
        # Nếu không đủ bài hát, trả về tất cả
        if len(candidates_df) <= n_recommendations:
            return candidates_df
        
        # Sắp xếp theo điểm số
        sorted_df = candidates_df.sort_values('final_score', ascending=False)
        
        # Kiểm tra xem có cột HDBSCAN không
        has_hdbscan = 'hdbscan_cluster' in sorted_df.columns
        has_culture = 'music_culture' in sorted_df.columns and 'music_culture' in seed_track
        
        # Nếu không có clustering hoặc thông tin văn hóa, trả về top N
        if not has_hdbscan and not has_culture:
            return sorted_df.head(n_recommendations)
        
        # Lấy thông tin seed
        seed_artist = seed_track.get('artist', '')
        
        # Khởi tạo danh sách kết quả
        final_recommendations = []
        artist_counts = {}
        
        # Sử dụng thông tin văn hóa nếu có
        if has_culture:
            seed_culture = seed_track.get('music_culture', 'unknown')
            
            # Tỷ lệ bài hát cùng văn hóa
            same_culture_ratio = min(0.7, max(0.3, self.diversity_config.get("country_ratio", 0.6)))
            same_culture_count = int(n_recommendations * same_culture_ratio)
            
            # Lọc bài hát theo văn hóa
            same_culture = sorted_df[sorted_df['music_culture'] == seed_culture]
            other_culture = sorted_df[sorted_df['music_culture'] != seed_culture]
            
            # Thêm bài hát cùng văn hóa
            for _, track in same_culture.iterrows():
                artist = track['artist']
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
                
                # Giới hạn số bài hát từ cùng nghệ sĩ
                if artist == seed_artist and artist_counts[artist] > self.diversity_config.get("max_same_artist", 2):
                    continue
                
                final_recommendations.append(track)
                if len(final_recommendations) >= same_culture_count:
                    break
            
            # Thêm bài hát từ văn hóa khác
            for _, track in other_culture.iterrows():
                artist = track['artist']
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
                
                # Giới hạn số bài hát từ cùng nghệ sĩ
                if artist == seed_artist and artist_counts[artist] > self.diversity_config.get("max_same_artist", 2):
                    continue
                
                final_recommendations.append(track)
                if len(final_recommendations) >= n_recommendations:
                    break
        
        # Sử dụng HDBSCAN clusters nếu có và chưa đủ bài hát
        elif has_hdbscan:
            seed_cluster = seed_track.get('hdbscan_cluster', -1)
            
            # Lấy danh sách các cluster (bỏ qua noise points nếu có nhiều cluster)
            clusters = sorted_df['hdbscan_cluster'].unique()
            if len(clusters) > 2 and -1 in clusters:
                clusters = [c for c in clusters if c != -1]
            
            # Thêm bài hát từ cùng cluster
            same_cluster = sorted_df[sorted_df['hdbscan_cluster'] == seed_cluster]
            for _, track in same_cluster.iterrows():
                artist = track['artist']
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
                
                if artist == seed_artist and artist_counts[artist] > self.diversity_config.get("max_same_artist", 2):
                    continue
                
                final_recommendations.append(track)
                if len(final_recommendations) >= n_recommendations // 2:
                    break
            
            # Thêm bài hát từ các cluster khác
            for cluster in clusters:
                if cluster == seed_cluster:
                    continue
                
                cluster_tracks = sorted_df[sorted_df['hdbscan_cluster'] == cluster]
                for _, track in cluster_tracks.iterrows():
                    artist = track['artist']
                    artist_counts[artist] = artist_counts.get(artist, 0) + 1
                    
                    if artist == seed_artist and artist_counts[artist] > self.diversity_config.get("max_same_artist", 2):
                        continue
                    
                    final_recommendations.append(track)
                    if len(final_recommendations) >= n_recommendations:
                        break
                
                if len(final_recommendations) >= n_recommendations:
                    break
        
        # Nếu vẫn chưa đủ, thêm các bài hát còn lại
        if len(final_recommendations) < n_recommendations:
            remaining = sorted_df[~sorted_df.index.isin([r.name for r in final_recommendations])]
            for _, track in remaining.iterrows():
                final_recommendations.append(track)
                if len(final_recommendations) >= n_recommendations:
                    break
        
        # Chuyển đổi danh sách thành DataFrame
        result_df = pd.DataFrame(final_recommendations)
        
        # Sắp xếp lại theo điểm số
        if not result_df.empty:
            result_df = result_df.sort_values('final_score', ascending=False)
        
        return result_df.head(n_recommendations)


# ✅ ALIAS for backward compatibility
ContentBasedRecommender = WeightedContentRecommender
