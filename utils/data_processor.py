import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CONTENT_FEATURES

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to process music data"""
    
    def __init__(self):
        """Initialize data processor"""
        self.tracks_df = None
        self.audio_features_df = None
        self.artist_genres_df = None
        self.user_item_matrix = None
    
    def load_data(self):
        """Load raw data from files - focus on real metadata only"""
        # Đọc dữ liệu bài hát
        tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
        enriched_path = os.path.join(RAW_DATA_DIR, 'enriched_tracks.csv')
        
        # Ưu tiên dùng enriched_tracks nếu có
        if os.path.exists(enriched_path):
            self.tracks_df = pd.read_csv(enriched_path)
            logger.info(f"Loaded {len(self.tracks_df)} enriched tracks from {enriched_path}")
        elif os.path.exists(tracks_path):
            self.tracks_df = pd.read_csv(tracks_path)
            logger.info(f"Loaded {len(self.tracks_df)} tracks from {tracks_path}")
        else:
            logger.warning(f"Track data files not found")
            self.tracks_df = pd.DataFrame()
        
        # LOẠI BỎ audio_features loading vì Spotify đã chặn
        # self.audio_features_df = None  # Không sử dụng
        
        # Đọc thông tin thể loại nghệ sĩ (vẫn còn available)
        genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
        if os.path.exists(genres_path):
            self.artist_genres_df = pd.read_csv(genres_path)
            logger.info(f"Loaded genre data for {len(self.artist_genres_df)} artists")
        else:
            logger.warning(f"Artist genres file not found: {genres_path}")
            self.artist_genres_df = pd.DataFrame()
        
        # Log available real features
        if not self.tracks_df.empty:
            real_features = ['popularity', 'duration_ms', 'explicit', 'release_year', 
                           'album_type', 'total_tracks', 'track_number', 'markets_count']
            available_features = [f for f in real_features if f in self.tracks_df.columns]
            logger.info(f"Available real Spotify features: {available_features}")
        
        return True
    
    def clean_tracks_data(self):
        """Clean tracks data"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to clean")
            return False
        
        # Lưu số lượng ban đầu
        initial_count = len(self.tracks_df)
        
        # Đảm bảo các cột bắt buộc tồn tại
        essential_cols = ['id', 'name', 'artist']
        self.tracks_df = self.tracks_df.dropna(subset=essential_cols)
        
        # Loại bỏ trùng lặp
        self.tracks_df = self.tracks_df.drop_duplicates(subset=['id'])
        
        # Điền missing values
        if 'popularity' in self.tracks_df.columns:
            self.tracks_df['popularity'] = self.tracks_df['popularity'].fillna(0)
        
        if 'explicit' in self.tracks_df.columns:
            self.tracks_df['explicit'] = self.tracks_df['explicit'].fillna(0).astype(int)
        
        if 'duration_ms' in self.tracks_df.columns:
            self.tracks_df['duration_ms'] = self.tracks_df['duration_ms'].fillna(self.tracks_df['duration_ms'].median())
        
        if 'artist_popularity' not in self.tracks_df.columns:
            self.tracks_df['artist_popularity'] = 50
        self.tracks_df['artist_popularity'] = self.tracks_df['artist_popularity'].fillna(50)
        
        # Log kết quả
        clean_count = len(self.tracks_df)
        logger.info(f"Cleaned tracks data: {initial_count} -> {clean_count} tracks")
        
        return True
    
    def create_synthetic_audio_features(self):
        """Skip creating synthetic audio features - focus on real metadata only"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to check features for")
            return False
        
        logger.info("Skipping synthetic audio features - focusing on real Spotify metadata only")
        
        # Chỉ đảm bảo các features thiết yếu từ Spotify API có sẵn
        essential_features = ['popularity', 'duration_ms', 'explicit']
        
        for feature in essential_features:
            if feature not in self.tracks_df.columns:
                logger.warning(f"Missing essential feature: {feature}")
                if feature == 'popularity':
                    self.tracks_df[feature] = 0
                elif feature == 'duration_ms':
                    self.tracks_df[feature] = 200000  # 3min 20sec default
                elif feature == 'explicit':
                    self.tracks_df[feature] = 0
        
        # Log real features từ Spotify metadata
        real_spotify_features = ['popularity', 'duration_ms', 'explicit', 'release_date', 
                               'album_type', 'total_tracks', 'track_number', 'disc_number']
        existing_real_features = [f for f in real_spotify_features if f in self.tracks_df.columns]
        
        logger.info(f"Using {len(existing_real_features)} real Spotify metadata features: {existing_real_features}")
        
        return True
    
    def merge_artist_genres(self):
        """Merge artist genres to tracks data và clean artist_popularity columns"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to merge genres with")
            return False
        
        if self.artist_genres_df is None or self.artist_genres_df.empty:
            logger.warning("No artist genres to merge")
            return False
        
        # Kiểm tra cột artist_id tồn tại
        if 'artist_id' not in self.tracks_df.columns:
            logger.warning("No artist_id column in tracks data for genre merging")
            return False
        
        # Lưu số lượng ban đầu
        initial_count = len(self.tracks_df)
        
        # Đổi tên cột genres để tránh xung đột
        if 'genres' in self.artist_genres_df.columns:
            self.artist_genres_df = self.artist_genres_df.rename(columns={'genres': 'artist_genres'})
        
        # Kết hợp với dữ liệu bài hát
        genres_columns = ['artist_id', 'artist_genres']
        for col in ['artist_popularity', 'artist_followers']:
            if col in self.artist_genres_df.columns:
                genres_columns.append(col)
        
        merged_df = self.tracks_df.merge(
            self.artist_genres_df[genres_columns], 
            left_on='artist_id', 
            right_on='artist_id', 
            how='left'
        )
        
        # **CLEAN ARTIST_POPULARITY COLUMNS NGAY TẠI ĐÂY**
        self._clean_artist_popularity_columns(merged_df)
        
        # Điền missing values
        merged_df['artist_genres'] = merged_df['artist_genres'].fillna('')
        if 'artist_followers' in merged_df.columns:
            merged_df['artist_followers'] = merged_df['artist_followers'].fillna(0)
        
        self.tracks_df = merged_df
        
        # Log kết quả
        merged_count = len(self.tracks_df)
        logger.info(f"Merged artist genres: {initial_count} tracks -> {merged_count} tracks")
        
        return True

    def _clean_artist_popularity_columns(self, df):
        """Clean artist_popularity columns - chỉ giữ data có nghĩa từ artist_genres.csv"""
        
        # Step 1: Xác định cột artist_popularity có nghĩa
        meaningful_col = None
        if 'artist_popularity_y' in df.columns:
            meaningful_col = 'artist_popularity_y'
            logger.info("Found artist_popularity_y - using as primary artist_popularity")
        elif 'artist_popularity' in df.columns:
            meaningful_col = 'artist_popularity'
            logger.info("Found artist_popularity - using as primary")
        
        # Step 2: Tạo cột artist_popularity cuối cùng
        if meaningful_col:
            df['artist_popularity'] = df[meaningful_col].fillna(50)  # Default value
            logger.info(f"Set artist_popularity from {meaningful_col}")
        else:
            df['artist_popularity'] = 50  # Default for all
            logger.warning("No meaningful artist_popularity found, using default value 50")
        
        # Step 3: Xóa TẤT CẢ các variant columns
        cols_to_drop = [col for col in df.columns 
                       if col.startswith('artist_popularity_')]
        
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Dropped artist_popularity variants: {cols_to_drop}")
        
        logger.info("Artist popularity columns cleaned successfully")
        return df
    
    def extract_release_year(self):
        """Extract release year from release date"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to extract release year from")
            return False
        
        if 'release_date' not in self.tracks_df.columns:
            logger.warning("No release_date column in tracks data")
            return False
        
        # Nếu đã có release_year, bỏ qua
        if 'release_year' in self.tracks_df.columns:
            logger.info("Release year already exists in data")
            return True
        
        # Extract year from release_date
        def extract_year(date_str):
            if not date_str or pd.isna(date_str):
                return datetime.now().year  # Default to current year
            
            try:
                # Handle different date formats
                if '-' in str(date_str):
                    parts = str(date_str).split('-')
                    if len(parts) > 0 and len(parts[0]) == 4:
                        return int(parts[0])
                
                # Try direct conversion for year-only values
                year = int(date_str)
                if 1900 <= year <= datetime.now().year:
                    return year
            except:
                pass
            
            # Default
            return datetime.now().year
        
        self.tracks_df['release_year'] = self.tracks_df['release_date'].apply(extract_year)
        
        # Tính decade
        self.tracks_df['decade'] = (self.tracks_df['release_year'] // 10) * 10
        
        logger.info("Extracted release year and decade from release dates")
        
        return True
    
    def create_genre_features(self):
        """Create genre features from artist_genres column"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create genre features")
            return False
        
        if 'artist_genres' not in self.tracks_df.columns:
            logger.warning("No artist_genres column in tracks data")
            return False
        
        # Extract all unique genres
        all_genres = []
        for genres in self.tracks_df['artist_genres'].dropna():
            if genres:
                genre_list = genres.split('|')
                all_genres.extend(genre_list)
        
        # Count occurrences and get top N genres
        from collections import Counter
        genre_counts = Counter(all_genres)
        top_n = 20  # Top 20 genres
        top_genres = [genre for genre, count in genre_counts.most_common(top_n)]
        
        logger.info(f"Top {top_n} genres: {', '.join(top_genres)}")
        
        # Create binary features for top genres
        for genre in top_genres:
            col_name = f'genre_{genre.replace(" ", "_").lower()}'
            self.tracks_df[col_name] = self.tracks_df['artist_genres'].apply(
                lambda x: 1 if pd.notna(x) and genre in str(x).split('|') else 0
            )
        
        logger.info(f"Created {top_n} genre binary features")
        
        return True
    
    def create_language_features(self):
        """Create features for detecting language and region"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create language features")
            return False
        
        # Check if name column exists
        if 'name' not in self.tracks_df.columns:
            logger.warning("No name column in tracks data for language detection")
            return False
        
        # Detect Vietnamese content
        viet_chars = ['Đ', 'đ', 'Ư', 'ư', 'Ơ', 'ơ', 'Ă', 'ă', 'Â', 'â', 'Ê', 'ê', 'Ô', 'ô']
        viet_words = ['việt', 'tình', 'yêu', 'anh', 'em', 'trời', 'đất', 'người']
        
        # Check track name for Vietnamese
        viet_pattern = '|'.join(viet_chars + viet_words)
        self.tracks_df['is_vietnamese'] = (
            self.tracks_df['name'].str.contains(viet_pattern, case=False, na=False) | 
            self.tracks_df['artist'].str.contains(viet_pattern, case=False, na=False)
        ).astype(int)
        
        # Detect other languages/regions
        language_patterns = {
            'is_korean': ['korea', 'k-pop', 'kpop', 'seoul', '(', ')', '아', '이', '으'],
            'is_japanese': ['japan', 'j-pop', 'jpop', 'tokyo', 'anime', '月', '日', 'の', 'は'],
            'is_spanish': ['latino', 'spanish', 'españa', 'méxico', 'cuba', 'latin']
        }
        
        for lang, patterns in language_patterns.items():
            pattern = '|'.join(patterns)
            self.tracks_df[lang] = (
                self.tracks_df['name'].str.contains(pattern, case=False, na=False) | 
                self.tracks_df['artist'].str.contains(pattern, case=False, na=False) |
                (self.tracks_df['artist_genres'].str.contains(pattern, case=False, na=False) 
                 if 'artist_genres' in self.tracks_df.columns else False)
            ).astype(int)
        
        logger.info("Created language and region detection features")
        
        return True
    
    def create_additional_features(self):
        """Create meaningful additional features from metadata"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create additional features")
            return False
        
        # 1. Duration features - cải thiện
        if 'duration_ms' in self.tracks_df.columns:
            self.tracks_df['duration_min'] = self.tracks_df['duration_ms'] / 60000
            
            # Tạo duration categories có ý nghĩa
            self.tracks_df['duration_category'] = pd.cut(
                self.tracks_df['duration_min'],
                bins=[0, 2.5, 3.5, 4.5, 6, 100],
                labels=['short', 'normal', 'extended', 'long', 'very_long']
            )
            
            # Duration score (bài hát 3-4 phút được ưa thích nhất)
            optimal_duration = 3.5
            self.tracks_df['duration_score'] = np.exp(-0.5 * ((self.tracks_df['duration_min'] - optimal_duration) / 1.5) ** 2)
        
        # 2. Popularity features - cải thiện
        if 'popularity' in self.tracks_df.columns:
            # Tạo popularity tiers có ý nghĩa
            self.tracks_df['popularity_tier'] = pd.cut(
                self.tracks_df['popularity'],
                bins=[0, 30, 50, 70, 85, 100],
                labels=['niche', 'emerging', 'popular', 'hit', 'viral']
            )
            
            # Popularity momentum (cho tracks mới)
            current_year = datetime.now().year
            if 'release_year' in self.tracks_df.columns:
                years_since_release = current_year - self.tracks_df['release_year'].fillna(current_year)
                # Tracks mới với popularity cao = momentum tốt
                self.tracks_df['popularity_momentum'] = self.tracks_df['popularity'] / (1 + 0.1 * years_since_release)
        
        # 3. Track name features - cải thiện
        if 'name' in self.tracks_df.columns:
            # Collab detection với regex tốt hơn
            collab_pattern = r'(feat\.?|ft\.?|featuring|with|vs\.?|x\s+|\&|\+)'
            self.tracks_df['has_collab'] = self.tracks_df['name'].str.contains(
                collab_pattern, case=False, regex=True, na=False
            ).astype(int)
            
            # Remix/version detection
            remix_pattern = r'(remix|edit|version|mix|remaster|acoustic|live|demo|instrumental)'
            self.tracks_df['is_remix'] = self.tracks_df['name'].str.contains(
                remix_pattern, case=False, regex=True, na=False
            ).astype(int)
            
            # Track name complexity
            self.tracks_df['name_length'] = self.tracks_df['name'].str.len()
            self.tracks_df['name_words'] = self.tracks_df['name'].str.split().str.len()
            
            # Special characters (có thể là non-English)
            self.tracks_df['has_special_chars'] = self.tracks_df['name'].str.contains(
                r'[^\w\s\-\(\)\[\]\.,:;!?\'"&]', regex=True, na=False
            ).astype(int)
        
        # 4. Artist features - cải thiện
        if 'artist' in self.tracks_df.columns:
            # Artist frequency với log scaling
            artist_counts = self.tracks_df['artist'].value_counts()
            self.tracks_df['artist_frequency'] = self.tracks_df['artist'].map(artist_counts)
            self.tracks_df['artist_frequency_log'] = np.log1p(self.tracks_df['artist_frequency'])
            
            # Normalize artist frequency
            max_freq_log = self.tracks_df['artist_frequency_log'].max()
            if max_freq_log > 0:
                self.tracks_df['artist_frequency_norm'] = self.tracks_df['artist_frequency_log'] / max_freq_log
            else:
                self.tracks_df['artist_frequency_norm'] = 0
            
            # Multi-artist tracks
            self.tracks_df['is_multi_artist'] = self.tracks_df['artist'].str.contains(
                r'[,&\+]|feat|ft\.?', case=False, regex=True, na=False
            ).astype(int)
        
        # 5. Album features - nếu có
        if 'album_type' in self.tracks_df.columns:
            # Encode album types
            album_type_mapping = {'album': 1.0, 'single': 0.8, 'compilation': 0.6}
            self.tracks_df['album_type_score'] = self.tracks_df['album_type'].map(
                album_type_mapping
            ).fillna(0.5)
        
        if 'total_tracks' in self.tracks_df.columns:
            # Album size categories
            self.tracks_df['album_size_category'] = pd.cut(
                self.tracks_df['total_tracks'].fillna(1),
                bins=[0, 1, 5, 12, 20, 100],
                labels=['single', 'ep', 'album', 'long_album', 'compilation']
            )
        
        logger.info("Created improved additional features from metadata")
        return True

    def normalize_features(self):
        """Improved feature normalization focusing on real metadata"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to normalize")
            return False
        
        # Định nghĩa features cần normalize
        numeric_features = {
            # Core Spotify features
            'popularity': {'method': 'minmax', 'fill': 0},
            'duration_ms': {'method': 'robust', 'fill': 'median'},  # Robust để handle outliers
            'artist_popularity': {'method': 'minmax', 'fill': 50},
            'release_year': {'method': 'minmax', 'fill': 2000},
            
            # Derived features
            'duration_min': {'method': 'robust', 'fill': 'median'},
            'name_length': {'method': 'robust', 'fill': 'median'},
            'name_words': {'method': 'robust', 'fill': 'median'},
            'artist_frequency': {'method': 'log_minmax', 'fill': 1},
            'total_tracks': {'method': 'log_minmax', 'fill': 1},
            'track_number': {'method': 'minmax', 'fill': 1},
            'disc_number': {'method': 'minmax', 'fill': 1},
            'markets_count': {'method': 'robust', 'fill': 0},
            
            # Computed scores
            'duration_score': {'method': 'minmax', 'fill': 0.5},
            'popularity_momentum': {'method': 'robust', 'fill': 'median'},
            'album_type_score': {'method': 'minmax', 'fill': 0.5}
        }
        
        # Apply normalization
        for feature, config in numeric_features.items():
            if feature in self.tracks_df.columns:
                # Handle missing values
                if config['fill'] == 'median':
                    fill_value = self.tracks_df[feature].median()
                else:
                    fill_value = config['fill']
                
                self.tracks_df[feature] = self.tracks_df[feature].fillna(fill_value)
                
                # Apply normalization
                values = self.tracks_df[feature].values.reshape(-1, 1)
                
                if config['method'] == 'minmax':
                    scaler = MinMaxScaler()
                    self.tracks_df[feature] = scaler.fit_transform(values).flatten()
                elif config['method'] == 'robust':
                    scaler = RobustScaler()
                    self.tracks_df[feature] = scaler.fit_transform(values).flatten()
                    # Clip to [0, 1] range after robust scaling
                    self.tracks_df[feature] = np.clip(self.tracks_df[feature], 0, 1)
                elif config['method'] == 'log_minmax':
                    log_values = np.log1p(values)
                    scaler = MinMaxScaler()
                    self.tracks_df[feature] = scaler.fit_transform(log_values).flatten()
        
        logger.info(f"Normalized {len([f for f in numeric_features.keys() if f in self.tracks_df.columns])} features with improved methods")
        return True
    
    def create_user_item_matrix(self, output_path=None):
        """Skip creating synthetic user-item matrix - focus on content-based only"""
        logger.info("Skipping synthetic user-item matrix - focusing on real data content-based approach")
        
        # Tạo ma trận rỗng minimal để tương thích
        if output_path:
            minimal_df = pd.DataFrame({'info': ['No synthetic user data - using content-based approach only']})
            minimal_df.to_csv(output_path, index=False)
            logger.info("Created placeholder user-item file for compatibility")
        
        return True
    
    def process_all(self):
        """Process all data with focus on metadata"""
        # Tạo thư mục processed nếu chưa tồn tại
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # 1. Load data
        self.load_data()
        
        # 2. Clean tracks data
        self.clean_tracks_data()
        
        # 3. Create synthetic audio features (since real ones aren't available)
        self.create_synthetic_audio_features()
        
        # 4. Merge artist genres
        self.merge_artist_genres()
        
        # 5. Extract release year
        self.extract_release_year()
        
        # 6. Create genre features
        self.create_genre_features()
        
        # 7. Create language features
        self.create_language_features()
        
        # 8. Create additional metadata features
        self.create_additional_features()
        
        # 9. Normalize features
        self.normalize_features()
        
        # 10. Lưu dữ liệu đã xử lý
        track_features_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
        self.tracks_df.to_csv(track_features_path, index=False)
        logger.info(f"Processed tracks data saved to {track_features_path}")
        
        # 11. Create user-item matrix
        user_matrix_path = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')
        self.create_user_item_matrix(output_path=user_matrix_path)
        
        logger.info("All data processing complete")
        return self.tracks_df

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()