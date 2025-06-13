import numpy as np
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity
import logging
from .base_model import BaseRecommender
from functools import lru_cache

logger = logging.getLogger(__name__)

class EnhancedContentRecommender(BaseRecommender):
    """Enhanced content-based recommender with K-Means clustering"""
    
    def __init__(self):
        super().__init__()
        self.name = "EnhancedContentRecommender"
        self.tracks_df = None
        self.feature_matrix = None
        self.is_trained = False
        self.cluster_weights = None  # Thêm trọng số cho các cluster
    
    def _get_feature_columns(self):
        """Get feature columns for similarity calculation"""
        try:
            # Danh sách các cột đặc trưng cần sử dụng
            feature_cols = []
            
            # Thêm các cột thể loại
            genre_cols = [col for col in self.tracks_df.columns if col.startswith('genre_')]
            if genre_cols:
                feature_cols.extend(genre_cols)
            
            # Thêm các cột đặc trưng chuẩn hóa
            norm_cols = [
                'popularity_norm', 'duration_ms_norm', 'artist_popularity_norm',
                'release_year_norm', 'track_age_norm'
            ]
            for col in norm_cols:
                if col in self.tracks_df.columns:
                    feature_cols.append(col)
            
            # Thêm các cột văn hóa
            cultural_cols = [
                'is_vietnamese', 'is_korean', 'is_japanese', 'is_chinese', 'is_western',
                'is_spanish', 'is_brazilian', 'is_indian', 'is_thai',
                'is_asia', 'is_europe', 'is_north_america', 'is_latin_america', 'is_oceania'
            ]
            for col in cultural_cols:
                if col in self.tracks_df.columns:
                    feature_cols.append(col)
            
            # Loại bỏ các cột trùng lặp
            feature_cols = list(set(feature_cols))
            
            if not feature_cols:
                logger.warning("No feature columns found for similarity calculation")
                return []
            
            logger.info(f"Using {len(feature_cols)} feature columns for similarity calculation")
            return feature_cols
        
        except Exception as e:
            logger.error(f"Error getting feature columns: {e}")
            return []
    
    def _compute_cluster_weights(self):
        """Compute weights for clusters based on cultural features"""
        try:
            if 'kmeans_cluster' not in self.tracks_df.columns:
                logger.warning("No kmeans_cluster column found for computing weights")
                self.cluster_weights = {}
                return
            
            # Khởi tạo trọng số mặc định
            self.cluster_weights = {}
            
            # Lấy số lượng cluster
            clusters = self.tracks_df['kmeans_cluster'].unique()
            
            # Tính toán trọng số cho từng cluster
            for cluster in clusters:
                # Lấy các track trong cluster
                cluster_tracks = self.tracks_df[self.tracks_df['kmeans_cluster'] == cluster]
                
                # Tính toán trọng số dựa trên đặc điểm văn hóa
                cultural_weight = 1.0  # Trọng số mặc định
                
                # Nếu cluster có nhiều bài hát Việt Nam, tăng trọng số
                if 'is_vietnamese' in cluster_tracks.columns:
                    vn_ratio = cluster_tracks['is_vietnamese'].mean()
                    if vn_ratio > 0.5:
                        cultural_weight = 1.2  # Tăng 20% cho cluster có nhiều bài hát Việt Nam
                
                # Nếu cluster có nhiều bài hát Hàn Quốc, tăng trọng số
                if 'is_korean' in cluster_tracks.columns:
                    kr_ratio = cluster_tracks['is_korean'].mean()
                    if kr_ratio > 0.5:
                        cultural_weight = 1.15  # Tăng 15% cho cluster có nhiều bài hát Hàn Quốc
                
                # Lưu trọng số
                self.cluster_weights[cluster] = cultural_weight
            
            logger.info(f"Computed cultural-focused weights for {len(self.cluster_weights)} clusters")
            
        except Exception as e:
            logger.error(f"Error computing cluster weights: {e}")
            self.cluster_weights = {}
    
    def train(self, tracks_df):
        """Train the model with the given tracks dataframe"""
        try:
            logger.info(f"Training {self.name} with {len(tracks_df)} tracks")
            self.tracks_df = tracks_df.copy()
            
            # Check if K-Means clusters exist and set up cluster weights if available
            has_clusters = 'kmeans_cluster' in self.tracks_df.columns
            if has_clusters:
                logger.info(f"Using pre-computed K-Means clusters")
                # Gọi phương thức tính toán trọng số cluster dựa trên đặc điểm văn hóa
                self._compute_cluster_weights()
                logger.info(f"Computed cultural-based cluster weights for {len(self.cluster_weights or {})} clusters")
            else:
                logger.warning("No K-Means clusters found in data. Clustering features will not be used.")
                self.cluster_weights = None
            
            # Prepare feature matrix for content-based similarity
            feature_cols = self._get_feature_columns()
            
            if not feature_cols:
                logger.error("No valid feature columns found for training")
                return False
            
            # Create feature matrix
            self.feature_matrix = self.tracks_df[feature_cols].values
            
            # Mark as trained
            self.is_trained = True
            self.train_time = time.time()
            
            logger.info(f"{self.name} trained successfully with {len(feature_cols)} features")
            return True
        
        except Exception as e:
            logger.error(f"Error training {self.name}: {e}")
            self.is_trained = False
            return False

    def recommend(self, track_name=None, artist=None, n_recommendations=10):
        """Generate recommendations based on a seed track with improved performance"""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return pd.DataFrame()
        
        # Đo thời gian thực hiện
        start_time = time.time()
        
        try:
            # Tìm track index
            track_idx = self.find_track_idx(track_name, artist)
            
            if track_idx is None:
                logger.warning(f"Track '{track_name}' by '{artist}' not found")
                return pd.DataFrame()
            
            # TỐI ƯU: Chỉ tính similarity cho một batch tracks thay vì toàn bộ
            batch_size = 10000  # Xử lý theo batch để tránh tràn bộ nhớ
            total_tracks = len(self.tracks_df)
            similarities = np.zeros(total_tracks)
            
            # Lấy vector đặc trưng của track gốc
            seed_vector = self.feature_matrix[track_idx].reshape(1, -1)
            
            # Tính toán similarity theo batch
            for i in range(0, total_tracks, batch_size):
                end_idx = min(i + batch_size, total_tracks)
                batch_similarities = cosine_similarity(
                    seed_vector,
                    self.feature_matrix[i:end_idx]
                )[0]
                similarities[i:end_idx] = batch_similarities
            
            # Tạo DataFrame với similarity scores
            sim_df = pd.DataFrame({
                'index': range(len(similarities)),
                'similarity': similarities
            })
            
            # Loại bỏ track gốc
            sim_df = sim_df[sim_df['index'] != track_idx]
            
            # Áp dụng trọng số cluster nếu có (vectorized)
            if self.cluster_weights is not None and 'kmeans_cluster' in self.tracks_df.columns:
                # Lấy cluster của track gốc
                seed_cluster = self.tracks_df.iloc[track_idx]['kmeans_cluster']
                
                # Lấy cluster của tất cả tracks
                all_clusters = self.tracks_df['kmeans_cluster'].values
                
                # Tạo mảng trọng số (mặc định là 1.0)
                cluster_boost = np.ones(len(sim_df))
                
                # Áp dụng trọng số cho tracks cùng cluster (vectorized)
                same_cluster_mask = all_clusters[sim_df['index']] == seed_cluster
                cluster_boost[same_cluster_mask] = self.cluster_weights.get(seed_cluster, 1.0)
                
                # Áp dụng trọng số
                sim_df['similarity'] = sim_df['similarity'] * cluster_boost
            
            # Sắp xếp theo similarity
            sim_df = sim_df.sort_values('similarity', ascending=False)
            
            # Lấy top N*2 recommendations
            top_indices = sim_df.head(n_recommendations * 2)['index'].values
            
            # Lấy thông tin đầy đủ cho các track được chọn
            recommendations = self.tracks_df.iloc[top_indices].copy()
            
            # Thêm điểm similarity
            recommendations['enhanced_score'] = sim_df.head(n_recommendations * 2)['similarity'].values
            
            # Đa dạng hóa kết quả (tối ưu hóa)
            recommendations = self._fast_diversify(recommendations, track_idx, n_recommendations)
            
            # Ghi log thời gian thực hiện
            elapsed = time.time() - start_time
            logger.info(f"EnhancedContentRecommender recommendation completed in {elapsed:.3f} seconds")
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error in EnhancedContentRecommender.recommend: {e}")
            return pd.DataFrame()

    def _fast_diversify(self, candidates_df, seed_idx, n_recommendations):
        """Phương pháp đa dạng hóa nhanh hơn"""
        # Nếu không đủ bài hát, trả về tất cả
        if len(candidates_df) <= n_recommendations:
            return candidates_df
        
        # Lấy thông tin bài hát gốc
        seed_track = self.tracks_df.iloc[seed_idx]
        
        # Sắp xếp theo điểm số
        sorted_df = candidates_df.sort_values('enhanced_score', ascending=False)
        
        # Khởi tạo các biến
        has_culture = 'music_culture' in sorted_df.columns and 'music_culture' in seed_track
        seed_culture = seed_track.get('music_culture', 'unknown') if has_culture else None
        seed_artist = seed_track.get('artist', '')
        
        # Tạo các mask để lọc nhanh
        if has_culture:
            same_culture_mask = sorted_df['music_culture'] == seed_culture
            same_culture = sorted_df[same_culture_mask]
            different_culture = sorted_df[~same_culture_mask]
            
            # Số lượng bài hát cho mỗi nhóm
            same_culture_count = min(int(n_recommendations * 0.6), len(same_culture))
            different_culture_count = n_recommendations - same_culture_count
            
            # Đa dạng hóa nghệ sĩ trong cùng văn hóa
            same_culture_diverse = self._diversify_artists(same_culture, seed_artist, same_culture_count)
            
            # Đa dạng hóa nghệ sĩ trong văn hóa khác
            different_culture_diverse = self._diversify_artists(different_culture, seed_artist, different_culture_count)
            
            # Kết hợp kết quả
            result = pd.concat([same_culture_diverse, different_culture_diverse])
        else:
            # Không có thông tin văn hóa, chỉ đa dạng hóa nghệ sĩ
            result = self._diversify_artists(sorted_df, seed_artist, n_recommendations)
        
        # Sắp xếp lại theo điểm số
        result = result.sort_values('enhanced_score', ascending=False).head(n_recommendations)
        
        return result

    def _diversify_artists(self, df, seed_artist, n_items):
        """Đa dạng hóa nghệ sĩ trong DataFrame"""
        if len(df) <= n_items:
            return df
        
        # Số lượng bài hát tối đa từ cùng nghệ sĩ
        max_same_artist = min(3, n_items // 3)
        
        # Đếm số lượng bài hát từ mỗi nghệ sĩ
        artist_counts = {}
        
        # Kết quả
        result = []
        
        # Duyệt qua từng bài hát
        for _, track in df.iterrows():
            artist = track['artist']
            
            # Đếm số lượng bài hát từ nghệ sĩ này
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            
            # Nếu đã đủ số lượng bài hát từ nghệ sĩ này, bỏ qua
            if artist == seed_artist and artist_counts[artist] > max_same_artist:
                continue
            elif artist != seed_artist and artist_counts[artist] > 2:
                continue
            
            # Thêm vào kết quả
            result.append(track)
            
            # Nếu đã đủ số lượng bài hát, dừng lại
            if len(result) >= n_items:
                break
        
        # Nếu chưa đủ, thêm các bài hát còn lại
        if len(result) < n_items:
            remaining = df[~df.index.isin([r.name for r in result])]
            for _, track in remaining.iterrows():
                result.append(track)
                if len(result) >= n_items:
                    break
        
        # Chuyển kết quả thành DataFrame
        return pd.DataFrame(result)

    @lru_cache(maxsize=100)
    def find_track_idx(self, track_name, artist=None):
        """Find track index with improved matching (cached)"""
        if self.tracks_df is None or track_name is None:
            return None
        
        # Chuẩn hóa tên bài hát và nghệ sĩ
        track_name_norm = track_name.lower().strip()
        artist_norm = artist.lower().strip() if artist else None
     
        # Tìm kiếm chính xác
        mask = self.tracks_df['name'].str.lower().str.strip() == track_name_norm
        if artist_norm:
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
                    return matches.sort_values('popularity', ascending=False).index[0]
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
