import os
import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import hdbscan
from sklearn.decomposition import PCA
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CONTENT_FEATURES, CLUSTERING_CONFIG

# Bỏ qua cảnh báo cụ thể về force_all_finite
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process raw data into ML-ready features - NO FETCHING"""
    
    def __init__(self):
        self.tracks_df = None
        self.audio_features_df = None
        self.artists_df = None

    def load_data(self):
        """Load existing data from processed directory"""
        try:
            processed_path = os.path.join(PROCESSED_DATA_DIR, "processed_tracks.csv")
            
            if os.path.exists(processed_path):
                self.tracks_df = pd.read_csv(processed_path, encoding='utf-8')
                logger.info(f"✅ Loaded {len(self.tracks_df)} tracks from processed data")
                return True
            else:
                # Try loading from raw data
                return self.load_raw_data()
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def load_raw_data(self):
        """Load raw data from files"""
        try:
            # Load tracks
            tracks_file = os.path.join(RAW_DATA_DIR, "tracks.csv")
            if os.path.exists(tracks_file):
                self.tracks_df = pd.read_csv(tracks_file, encoding='utf-8')
                logger.info(f"✅ Loaded {len(self.tracks_df)} raw tracks")
            else:
                logger.error(f"No data found in {RAW_DATA_DIR}")
                return False
            
            # Load audio features if available
            features_file = os.path.join(RAW_DATA_DIR, "audio_features.csv")
            if os.path.exists(features_file):
                self.audio_features_df = pd.read_csv(features_file, encoding='utf-8')
                logger.info(f"✅ Loaded {len(self.audio_features_df)} audio features")
            
            # Load artist info if available
            artists_file = os.path.join(RAW_DATA_DIR, "artist_genres.csv")
            if os.path.exists(artists_file):
                self.artists_df = pd.read_csv(artists_file, encoding='utf-8')
                logger.info(f"✅ Loaded {len(self.artists_df)} artists")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return False

    def process_all(self):
        """Run full data processing pipeline"""
        try:
            logger.info("Starting full data processing pipeline")
            
            # Load raw data
            if not self.load_raw_data():
                logger.error("Failed to load raw data")
                return False
            
            # Clean data
            if not self.clean_data():
                logger.error("Failed to clean data")
                return False
            
            # Extract features
            if not self.extract_features():
                logger.error("Failed to extract features")
                return False
            
            # Extract cultural features from ISRC
            if not self.extract_cultural_features():
                logger.warning("Failed to extract cultural features")
            
            # Apply clustering
            if not self.apply_clustering():
                logger.warning("Failed to apply clustering")
            
            # Remove duplicate columns
            if not self.remove_duplicate_columns():
                logger.warning("Failed to remove duplicate columns")
            
            # Save processed data
            if not self.save_processed_data():
                logger.error("Failed to save processed data")
                return False
            
            logger.info("Full data processing pipeline completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error in process_all: {e}")
            return False

    def merge_audio_features(self):
        """Merge audio features with tracks"""
        if self.audio_features_df is None:
            logger.info("No audio features to merge")
            return True
        
        try:
            initial_cols = len(self.tracks_df.columns)
            self.tracks_df = self.tracks_df.merge(
                self.audio_features_df, 
                on='id', 
                how='left'
            )
            new_cols = len(self.tracks_df.columns)
            logger.info(f"✅ Merged audio features: {new_cols - initial_cols} new columns")
            return True
            
        except Exception as e:
            logger.error(f"Error merging audio features: {e}")
            return False

    def merge_artist_data(self):
        """Merge artist information"""
        if self.artists_df is None:
            logger.info("No artist data to merge")
            return True
        
        try:
            # Clean artist data
            artist_data = self.artists_df.copy()
            
            # Log cấu trúc dữ liệu để debug
            logger.info(f"Artist DataFrame columns: {artist_data.columns.tolist()}")
            logger.info(f"Tracks DataFrame columns: {self.tracks_df.columns.tolist()}")
            
            # Merge dựa trên artist_id (chính xác nhất)
            self.tracks_df = self.tracks_df.merge(
                artist_data[['artist_id', 'genres']], 
                on='artist_id',
                how='left'
            )
            
            # Đổi tên cột genres thành artist_genres
            if 'genres' in self.tracks_df.columns:
                self.tracks_df = self.tracks_df.rename(columns={'genres': 'artist_genres'})
            
            # Fill missing genres
            self.tracks_df['artist_genres'] = self.tracks_df['artist_genres'].fillna('')
            
            logger.info(f"✅ Merged artist data - {self.tracks_df.shape[1]} columns now available")
            return True
            
        except Exception as e:
            logger.error(f"Error merging artist data: {e}")
            return False

    def extract_release_year(self):
        """Extract release year from release_date"""
        if self.tracks_df is None:
            return False

        try:
            current_year = datetime.now().year
            
            # Extract year from release_date
            def extract_year(date_str):
                if pd.isna(date_str) or date_str == '':
                    return current_year
                try:
                    return int(str(date_str)[:4])
                except:
                    return current_year
            
            self.tracks_df['release_year'] = self.tracks_df['release_date'].apply(extract_year)
            
            # Calculate track age
            self.tracks_df['track_age'] = current_year - self.tracks_df['release_year']
            self.tracks_df['track_age'] = self.tracks_df['track_age'].clip(lower=0)
            
            logger.info("✅ Extracted release year and track age")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting release year: {e}")
            return False

    def extract_cultural_features(self):
        """Extract cultural features from ISRC ONLY - NO MARKET DATA"""
        if self.tracks_df is None:
            return False

        try:
            # Region mapping
            region_map = {
                # Asia
                'VN': 'asia', 'KR': 'asia', 'JP': 'asia', 'CN': 'asia', 
                'HK': 'asia', 'TW': 'asia', 'TH': 'asia', 'MY': 'asia', 
                'ID': 'asia', 'PH': 'asia', 'SG': 'asia', 'IN': 'asia',
                
                # Europe
                'GB': 'europe', 'DE': 'europe', 'FR': 'europe', 'IT': 'europe', 
                'ES': 'europe', 'NL': 'europe', 'SE': 'europe', 'NO': 'europe', 
                'DK': 'europe', 'FI': 'europe', 'PL': 'europe', 'RU': 'europe',
                
                # North America
                'US': 'north_america', 'CA': 'north_america',
                
                # Latin America
                'MX': 'latin_america', 'BR': 'latin_america', 'AR': 'latin_america', 
                'CO': 'latin_america', 'CL': 'latin_america', 'PE': 'latin_america',
                
                # Oceania
                'AU': 'oceania', 'NZ': 'oceania'
            }
            
            # Extract country from ISRC
            self.tracks_df['isrc_country'] = self.tracks_df['isrc'].str[:2].fillna('XX')
            
            # Map to region
            self.tracks_df['region'] = self.tracks_df['isrc_country'].map(region_map).fillna('other')
            
            # Cultural mapping
            cultural_mapping = {
                'VN': 'vietnamese',
                'KR': 'korean', 
                'JP': 'japanese',
                'CN': 'chinese', 'HK': 'chinese', 'TW': 'chinese',
                'US': 'western', 'GB': 'western', 'CA': 'western', 'AU': 'western',
                'ES': 'spanish', 'MX': 'spanish', 'AR': 'spanish', 'CO': 'spanish',
                'BR': 'brazilian',
                'DE': 'western', 'FR': 'western', 'IT': 'western',
                'IN': 'indian',
                'TH': 'thai', 'MY': 'malaysian', 'ID': 'indonesian'
            }
            
            self.tracks_df['music_culture'] = self.tracks_df['isrc_country'].map(cultural_mapping).fillna('other')
            
            # Cultural confidence
            self.tracks_df['cultural_confidence'] = np.where(
                self.tracks_df['isrc'].notna() & (self.tracks_df['isrc'] != ''), 
                0.9, 0.3
            )
            
            # Binary cultural features
            cultures = ['vietnamese', 'korean', 'japanese', 'chinese', 'western', 'spanish', 'brazilian', 'indian', 'thai']
            for culture in cultures:
                self.tracks_df[f'is_{culture}'] = (self.tracks_df['music_culture'] == culture).astype(int)
            
            # Binary regional features
            regions = ['asia', 'europe', 'north_america', 'latin_america', 'oceania']
            for region in regions:
                self.tracks_df[f'is_{region}'] = (self.tracks_df['region'] == region).astype(int)
            
            logger.info("✅ Extracted ISRC-based cultural features")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting cultural features: {e}")
            return False

    def create_genre_features(self):
        """Create genre features based purely on frequency"""
        if self.tracks_df is None:
            return False

        try:
            # Use artist_genres if available, otherwise skip
            genre_column = 'artist_genres' if 'artist_genres' in self.tracks_df.columns else None
            
            if genre_column is None or self.tracks_df[genre_column].isna().all():
                logger.info("No genre data available, skipping genre features")
                return True

            # Count all genres
            genre_counts = {}
            for genres_str in self.tracks_df[genre_column].dropna():
                if isinstance(genres_str, str) and genres_str.strip():
                    genres = [g.strip().lower().replace(' ', '_').replace('-', '_') 
                             for g in genres_str.split(',')]
                    for genre in genres:
                        if genre and len(genre) > 1:
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1

            # Filter by frequency
            significant_genres = {
                genre: count for genre, count in genre_counts.items() 
                if count >= 5  # At least 5 tracks
            }

            # Take top genres
            sorted_genres = sorted(significant_genres.items(), key=lambda x: x[1], reverse=True)
            final_genres = [genre for genre, count in sorted_genres[:100]]

            # Create genre matrix
            if final_genres:
                genre_data = {}
                for genre in final_genres:
                    genre_col = f'genre_{genre}'
                    genre_mask = self.tracks_df[genre_column].str.contains(
                        genre, case=False, na=False, regex=False
                    )
                    genre_data[genre_col] = genre_mask.astype(int)

                # Add to DataFrame
                genre_df = pd.DataFrame(genre_data, index=self.tracks_df.index)
                self.tracks_df = pd.concat([self.tracks_df, genre_df], axis=1)
                
                logger.info(f"✅ Created {len(final_genres)} genre features")
            else:
                logger.info("No significant genres found")

            return True

        except Exception as e:
            logger.error(f"Error creating genre features: {e}")
            return False

    def clean_tracks_data(self):
        """Clean tracks data"""
        if self.tracks_df is None or self.tracks_df.empty:
            return False
        
        initial_count = len(self.tracks_df)
        
        try:
            # Remove duplicates
            if 'id' in self.tracks_df.columns:
                before_dedup = len(self.tracks_df)
                self.tracks_df = self.tracks_df.drop_duplicates(subset=['id'], keep='first')
                after_dedup = len(self.tracks_df)
                logger.info(f"Removed {before_dedup - after_dedup} duplicates")
            
            # Fill numeric columns
            numeric_columns = {
                'popularity': 50,
                'duration_ms': 200000,
                'release_year': 2020,
                'track_age': 4
            }
            
            for col, default_value in numeric_columns.items():
                if col in self.tracks_df.columns:
                    self.tracks_df[col] = self.tracks_df[col].fillna(default_value)
            
            # Fill string columns
            string_columns = {
                'isrc': '',
                'artist_genres': '',
                'album': 'Unknown Album'
            }
            
            for col, default_value in string_columns.items():
                if col in self.tracks_df.columns:
                    self.tracks_df[col] = self.tracks_df[col].fillna(default_value)
            
            # Remove rows with missing essential data
            essential_columns = ['id', 'name', 'artist']
            for col in essential_columns:
                if col in self.tracks_df.columns:
                    mask = self.tracks_df[col].notna() & (self.tracks_df[col] != '')
                    self.tracks_df = self.tracks_df[mask]
            
            clean_count = len(self.tracks_df)
            logger.info(f"Cleaned: {initial_count} -> {clean_count} tracks")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return False

    def clean_data(self):
        """Clean all data"""
        try:
            logger.info("Cleaning data...")
            
            # Clean tracks data
            if not self.clean_tracks_data():
                logger.error("Failed to clean tracks data")
                return False
            
            # Merge audio features if available
            if self.audio_features_df is not None:
                if not self.merge_audio_features():
                    logger.warning("Failed to merge audio features")
            
            # Merge artist data if available
            if self.artists_df is not None:
                if not self.merge_artist_data():
                    logger.warning("Failed to merge artist data")
            
            logger.info("Data cleaning completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error in clean_data: {e}")
            return False

    def normalize_features(self):
        """Normalize numerical features based on available data"""
        if self.tracks_df is None:
            return False

        try:
            # Xác định các đặc trưng số dựa trên dữ liệu thực tế có sẵn
            numerical_features = [
                'popularity', 'duration_ms', 'markets_count', 
                'artist_popularity', 'disc_number', 'track_number'
            ]
            
            # Lọc các cột thực sự tồn tại
            existing_features = [feat for feat in numerical_features if feat in self.tracks_df.columns]
            
            if not existing_features:
                logger.info("No numerical features to normalize")
                return True

            # Tạo các cột chuẩn hóa
            normalized_data = {}
            scaler = MinMaxScaler()
            
            for feature in existing_features:
                feature_data = self.tracks_df[feature].fillna(self.tracks_df[feature].median())
                normalized_values = scaler.fit_transform(feature_data.values.reshape(-1, 1))
                normalized_data[f'{feature}_norm'] = normalized_values.flatten()

            # Thêm vào DataFrame
            if normalized_data:
                normalized_df = pd.DataFrame(normalized_data, index=self.tracks_df.index)
                self.tracks_df = pd.concat([self.tracks_df, normalized_df], axis=1)
            
            # Tạo thêm đặc trưng từ release_date nếu có
            if 'release_date' in self.tracks_df.columns:
                try:
                    # Trích xuất năm từ release_date
                    self.tracks_df['release_year'] = self.tracks_df['release_date'].apply(
                        lambda x: int(str(x)[:4]) if pd.notna(x) and str(x).strip() and str(x)[:4].isdigit() else 2020
                    )
                    
                    # Chuẩn hóa năm phát hành
                    year_data = self.tracks_df['release_year'].fillna(2020)
                    min_year = max(1900, year_data.min())  # Tránh năm quá nhỏ
                    max_year = min(2030, year_data.max())  # Tránh năm quá lớn
                    
                    year_norm = (year_data - min_year) / (max_year - min_year + 1e-8)
                    self.tracks_df['release_year_norm'] = year_norm
                    
                    # Tính tuổi bài hát
                    current_year = datetime.now().year
                    self.tracks_df['track_age'] = current_year - self.tracks_df['release_year']
                    self.tracks_df['track_age_norm'] = 1 - year_norm  # Đảo ngược năm phát hành
                    
                    logger.info("✅ Created release year and track age features")
                except Exception as e:
                    logger.warning(f"Error processing release dates: {e}")
            
            # Tạo đặc trưng từ tên bài hát
            if 'name' in self.tracks_df.columns:
                name_length = self.tracks_df['name'].fillna('').apply(len)
                name_length_norm = (name_length - name_length.min()) / (name_length.max() - name_length.min() + 1e-8)
                self.tracks_df['name_length_norm'] = name_length_norm
            
            # Tạo đặc trưng từ is_playable và explicit nếu có
            if 'is_playable' in self.tracks_df.columns:
                self.tracks_df['is_playable_norm'] = self.tracks_df['is_playable'].astype(float)
            
            if 'explicit' in self.tracks_df.columns:
                self.tracks_df['explicit_norm'] = self.tracks_df['explicit'].astype(float)
            
            # Tạo đặc trưng từ available_markets
            if 'available_markets' in self.tracks_df.columns:
                # Nếu available_markets là chuỗi, đếm số thị trường
                if isinstance(self.tracks_df['available_markets'].iloc[0], str):
                    self.tracks_df['markets_count'] = self.tracks_df['available_markets'].apply(
                        lambda x: len(x.split('|')) if pd.notna(x) else 0
                    )
                
                # Chuẩn hóa markets_count
                if 'markets_count' in self.tracks_df.columns:
                    markets_data = self.tracks_df['markets_count'].fillna(0)
                    markets_norm = (markets_data - markets_data.min()) / (markets_data.max() - markets_data.min() + 1e-8)
                    self.tracks_df['markets_count_norm'] = markets_norm
            
            norm_count = len([col for col in self.tracks_df.columns if col.endswith('_norm')])
            logger.info(f"✅ Normalized {norm_count} features")
            return True

        except Exception as e:
            logger.error(f"Error normalizing: {e}")
            return False

    def extract_features(self):
        """Extract all features from data"""
        try:
            logger.info("Extracting features...")
            
            # Extract release year
            if not self.extract_release_year():
                logger.warning("Failed to extract release year")
            
            # Create genre features
            if not self.create_genre_features():
                logger.warning("Failed to create genre features")
            
            # Normalize features
            if not self.normalize_features():
                logger.warning("Failed to normalize features")
            
            logger.info("Feature extraction completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return False

    def apply_clustering(self):
        """Apply K-Means and HDBSCAN clustering to the dataset"""
        if self.tracks_df is None or len(self.tracks_df) < 10:
            logger.error("Not enough data for clustering")
            return False
        
        try:
            logger.info("Applying clustering algorithms to dataset")
            
            # Lấy tất cả các đặc trưng đã chuẩn hóa
            normalized_features = [col for col in self.tracks_df.columns if col.endswith('_norm')]
            
            # Nếu không đủ đặc trưng, thử sử dụng các đặc trưng văn hóa
            cultural_features = [col for col in self.tracks_df.columns if col.startswith('is_')]
            
            # Kết hợp các đặc trưng
            available_features = normalized_features + cultural_features
            
            # Lọc các đặc trưng thực sự tồn tại và có giá trị
            valid_features = []
            for feature in available_features:
                if feature in self.tracks_df.columns:
                    # Kiểm tra xem cột có chứa dữ liệu hợp lệ không
                    if self.tracks_df[feature].notna().sum() > len(self.tracks_df) * 0.5:  # Ít nhất 50% giá trị hợp lệ
                        valid_features.append(feature)
            
            # Giới hạn số lượng đặc trưng để tránh quá nhiều chiều
            if len(valid_features) > 20:
                valid_features = valid_features[:20]
            
            logger.info(f"Using {len(valid_features)} features for clustering: {valid_features}")
            
            if len(valid_features) < 3:
                logger.warning(f"Not enough features for clustering. Found only: {valid_features}")
                logger.warning("Creating dummy clustering columns. Clustering NOT applied.")
                self.tracks_df['kmeans_cluster'] = -1
                self.tracks_df['hdbscan_cluster'] = -1
                self.tracks_df['hdbscan_outlier_score'] = 0.0
                return True
            
            # Chuẩn bị dữ liệu cho clustering
            cluster_data = self.tracks_df[valid_features].copy()
            
            # Xử lý giá trị thiếu
            for col in cluster_data.columns:
                if cluster_data[col].dtype.kind in 'ifc':  # integer, float, complex
                    cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())
                else:
                    cluster_data[col] = cluster_data[col].fillna(0)
            
            # Chuẩn hóa dữ liệu
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # 1. K-Means Clustering
            logger.info("Applying K-Means clustering")
            from sklearn.cluster import KMeans
            n_clusters = 8  # Số lượng cluster phù hợp
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(scaled_data)
            self.tracks_df['kmeans_cluster'] = kmeans_labels
            
            # 2. HDBSCAN Clustering
            logger.info("Applying HDBSCAN clustering")
            min_cluster_size = 5
            min_samples = 3
            
            # Điều chỉnh min_cluster_size dựa trên kích thước dataset
            if len(self.tracks_df) > 1000:
                min_cluster_size = max(min_cluster_size, len(self.tracks_df) // 200)
            
            try:
                import hdbscan
                hdbscan_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=0.5,
                    metric='euclidean',
                    prediction_data=True
                )
                hdbscan_labels = hdbscan_clusterer.fit_predict(scaled_data)
                self.tracks_df['hdbscan_cluster'] = hdbscan_labels
                self.tracks_df['hdbscan_outlier_score'] = hdbscan_clusterer.outlier_scores_
                
                # Ghi log thông tin về clusters
                unique_clusters = np.unique(hdbscan_labels[hdbscan_labels >= 0])
                noise_count = np.sum(hdbscan_labels == -1)
                noise_pct = noise_count / len(self.tracks_df) * 100
                
                logger.info(f"HDBSCAN found {len(unique_clusters)} clusters and {noise_count} noise points ({noise_pct:.1f}%)")
                
            except Exception as e:
                logger.error(f"HDBSCAN clustering failed: {e}")
                self.tracks_df['hdbscan_cluster'] = -1
                self.tracks_df['hdbscan_outlier_score'] = 0.0
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying clustering: {e}")
            # Tạo các cột giả để tránh lỗi
            self.tracks_df['kmeans_cluster'] = -1
            self.tracks_df['hdbscan_cluster'] = -1
            self.tracks_df['hdbscan_outlier_score'] = 0.0
            return False

    def save_processed_data(self):
        """Save processed data"""
        try:
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            
            # Loại bỏ các cột trùng lặp
            columns = self.tracks_df.columns
            duplicate_cols = []
            
            # Tìm các cột có hậu tố .1, .2, v.v.
            for col in columns:
                if '.' in col and col.split('.')[-1].isdigit():
                    base_col = col.split('.')[0]
                    if base_col in columns:
                        duplicate_cols.append(col)
            
            if duplicate_cols:
                logger.info(f"Removing {len(duplicate_cols)} duplicate columns")
                self.tracks_df = self.tracks_df.drop(columns=duplicate_cols)
            
            # Save main processed file
            processed_file = os.path.join(PROCESSED_DATA_DIR, "processed_tracks.csv")
            self.tracks_df.to_csv(processed_file, index=False, encoding='utf-8')
            
            # Calculate summary
            feature_summary = {
                'total_tracks': len(self.tracks_df),
                'total_features': len(self.tracks_df.columns),
                'isrc_coverage': float((self.tracks_df['isrc'] != '').sum() / len(self.tracks_df)) if 'isrc' in self.tracks_df.columns else 0.0,
                'cultural_confidence': float(self.tracks_df['cultural_confidence'].mean()) if 'cultural_confidence' in self.tracks_df.columns else 0.0
            }
            
            # Count features
            genre_features = [col for col in self.tracks_df.columns if col.startswith('genre_')]
            cultural_features = [col for col in self.tracks_df.columns if col.startswith('is_')]
            normalized_features = [col for col in self.tracks_df.columns if col.endswith('_norm')]
            
            feature_summary['genre_features_count'] = len(genre_features)
            feature_summary['cultural_features_count'] = len(cultural_features)
            feature_summary['normalized_features_count'] = len(normalized_features)
            
            logger.info(f"✅ Saved to {processed_file}")
            logger.info(f"📊 Summary: {feature_summary}")
            
            return True

        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False

    def remove_duplicate_columns(self):
        """Remove duplicate columns from DataFrame"""
        if self.tracks_df is None:
            return False
        
        try:
            # Lấy danh sách cột ban đầu
            original_columns = list(self.tracks_df.columns)
            
            # Tạo từ điển để theo dõi các cột đã thấy
            seen_columns = {}
            duplicate_columns = []
            
            for col in original_columns:
                # Loại bỏ hậu tố .1, .2, v.v. nếu có
                base_col = col.split('.')[0] if '.' in col and col.split('.')[-1].isdigit() else col
                
                # Nếu đã thấy cột gốc, đánh dấu cột hiện tại là trùng lặp
                if base_col in seen_columns:
                    duplicate_columns.append(col)
                else:
                    seen_columns[base_col] = col
            
            # Loại bỏ các cột trùng lặp
            if duplicate_columns:
                logger.info(f"Removing {len(duplicate_columns)} duplicate columns")
                self.tracks_df = self.tracks_df.drop(columns=duplicate_columns)
                
                # Kiểm tra xem có cột nào bị trùng tên không
                if len(self.tracks_df.columns) != len(set(self.tracks_df.columns)):
                    # Tìm các cột trùng tên
                    cols_count = {}
                    for col in self.tracks_df.columns:
                        cols_count[col] = cols_count.get(col, 0) + 1
                    
                    duplicate_names = [col for col, count in cols_count.items() if count > 1]
                    logger.warning(f"Found columns with identical names: {duplicate_names}")
                    
                    # Đổi tên các cột trùng lặp
                    for col in duplicate_names:
                        # Tìm tất cả các vị trí của cột trùng lặp
                        indices = [i for i, x in enumerate(self.tracks_df.columns) if x == col]
                        
                        # Đổi tên các cột trùng lặp (trừ cột đầu tiên)
                        for i, idx in enumerate(indices[1:], 1):
                            new_name = f"{col}_dup{i}"
                            self.tracks_df.columns.values[idx] = new_name
                            logger.info(f"Renamed duplicate column '{col}' to '{new_name}'")
            
            logger.info(f"Final dataset has {len(self.tracks_df.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error removing duplicate columns: {e}")
            return False

# ✅ FIXED STANDALONE EXECUTION
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔄 STEP 2: PROCESSING RAW DATA")
    print("=" * 50)
    
    processor = DataProcessor()
    success = processor.process_all()
    
    if success:
        print("\n🎉 ✅ Data processing completed successfully!")
        print(f"📂 Processed data saved to: {PROCESSED_DATA_DIR}")
        print("\n🤖 Next step: Run main.py to train models")
    else:
        print("❌ Processing failed")
