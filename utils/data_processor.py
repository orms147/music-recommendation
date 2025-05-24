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
            logger.info(f"Loading enriched tracks from {enriched_path}")
            self.tracks_df = pd.read_csv(enriched_path)
        elif os.path.exists(tracks_path):
            logger.info(f"Loading basic tracks from {tracks_path}")
            self.tracks_df = pd.read_csv(tracks_path)
        else:
            logger.error("No track data files found!")
            return False
        
        # Đọc thông tin thể loại nghệ sĩ
        artist_genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
        if os.path.exists(artist_genres_path):
            logger.info(f"Loading artist genres from {artist_genres_path}")
            self.artist_genres_df = pd.read_csv(artist_genres_path)
        else:
            logger.warning("No artist genres file found - will attempt to fetch from Spotify")
            # CẢI THIỆN: Tự động fetch artist genres nếu thiếu
            if self._fetch_missing_artist_genres():
                self.artist_genres_df = pd.read_csv(artist_genres_path)
            else:
                logger.warning("Failed to fetch artist genres, will use fallback genre features")
                self.artist_genres_df = None
        
        logger.info(f"Loaded {len(self.tracks_df)} tracks")
        if self.artist_genres_df is not None:
            logger.info(f"Loaded {len(self.artist_genres_df)} artist genre records")
        
        return True

    def _fetch_missing_artist_genres(self):
        """Fetch missing artist genres using SpotifyDataFetcher"""
        try:
            from utils.data_fetcher import SpotifyDataFetcher
            
            logger.info("Attempting to fetch missing artist genres from Spotify...")
            fetcher = SpotifyDataFetcher()
            return fetcher.fetch_all_missing_artist_genres()
            
        except Exception as e:
            logger.error(f"Failed to fetch artist genres: {e}")
            return False

    def merge_artist_genres(self):
        """Merge artist genres with tracks data"""
        if self.artist_genres_df is None:
            logger.warning("No artist genres data available, skipping merge")
            return
        
        if 'artist_id' not in self.tracks_df.columns:
            logger.warning("No artist_id column in tracks data")
            return
        
        try:
            # Merge tracks với artist genres
            before_count = len(self.tracks_df)
            merged_df = self.tracks_df.merge(
                self.artist_genres_df[['artist_id', 'genres', 'artist_popularity', 'artist_followers']], 
                on='artist_id', 
                how='left'
            )
            
            # Clean artist popularity columns để tránh conflicts
            merged_df = self._clean_artist_popularity_columns(merged_df)
            
            self.tracks_df = merged_df
            
            # Log merge results
            genre_coverage = (self.tracks_df['genres'].notna().sum() / len(self.tracks_df)) * 100
            logger.info(f"Merged artist genres: {before_count} -> {len(self.tracks_df)} tracks")
            logger.info(f"Genre coverage: {genre_coverage:.1f}% of tracks have genre data")
            
        except Exception as e:
            logger.error(f"Error merging artist genres: {e}")

    def _clean_artist_popularity_columns(self, df):
        """Clean conflicting artist popularity columns"""
        # Nếu có cả artist_popularity từ tracks và từ genres, ưu tiên từ genres
        if 'artist_popularity_x' in df.columns and 'artist_popularity_y' in df.columns:
            # Ưu tiên giá trị từ artist_genres (y), fallback về tracks (x)
            df['artist_popularity'] = df['artist_popularity_y'].fillna(df['artist_popularity_x'])
            df = df.drop(['artist_popularity_x', 'artist_popularity_y'], axis=1)
        
        return df

    def create_genre_features(self):
        """Create genre features from artist genres data"""
        if 'genres' not in self.tracks_df.columns:
            logger.warning("No genres column found, creating fallback genre features")
            self._create_fallback_genre_features()
            return
        
        try:
            # Parse genres và tạo binary features
            all_genres = set()
            
            # Thu thập tất cả genres
            for genre_str in self.tracks_df['genres'].dropna():
                if isinstance(genre_str, str) and genre_str.strip():
                    genres = [g.strip().lower() for g in genre_str.split('|') if g.strip()]
                    all_genres.update(genres)
            
            logger.info(f"Found {len(all_genres)} unique genres")
            
            # Tạo features cho top genres (có ít nhất 10 tracks)
            genre_counts = {}
            for genre in all_genres:
                count = self.tracks_df['genres'].str.contains(genre, case=False, na=False).sum()
                if count >= 10:  # Threshold để loại bỏ genres quá ít
                    genre_counts[genre] = count
            
            # Sort và lấy top 25 genres
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:25]
            
            logger.info(f"Creating features for top {len(top_genres)} genres:")
            for genre, count in top_genres[:10]:  # Log top 10
                logger.info(f"  {genre}: {count} tracks")
            
            # Tạo binary features
            for genre, count in top_genres:
                # Clean genre name để tạo column name
                clean_genre = genre.replace(' ', '_').replace('-', '_').replace('&', 'and')
                clean_genre = ''.join(c for c in clean_genre if c.isalnum() or c == '_')
                col_name = f'genre_{clean_genre}'
                
                # Tạo binary feature
                self.tracks_df[col_name] = self.tracks_df['genres'].str.contains(
                    genre, case=False, na=False
                ).astype(int)
            
            # Tạo thêm compound genre features
            self._create_compound_genre_features()
            
            genre_feature_count = len([col for col in self.tracks_df.columns if col.startswith('genre_')])
            logger.info(f"Created {genre_feature_count} genre features")
            
        except Exception as e:
            logger.error(f"Error creating genre features: {e}")
            self._create_fallback_genre_features()

    def _create_fallback_genre_features(self):
        """Create fallback genre features when real genre data is unavailable"""
        logger.info("Creating fallback genre features from available data...")
        
        # Basic genre mapping từ language và popularity patterns
        genre_mapping = {
            'genre_pop': lambda row: 1 if row.get('popularity', 0) > 60 else 0,
            'genre_kpop': lambda row: row.get('is_korean', 0) if 'is_korean' in self.tracks_df.columns else 0,
            'genre_jpop': lambda row: row.get('is_japanese', 0) if 'is_japanese' in self.tracks_df.columns else 0,
            'genre_vpop': lambda row: row.get('is_vietnamese', 0) if 'is_vietnamese' in self.tracks_df.columns else 0,
            'genre_latin': lambda row: row.get('is_spanish', 0) if 'is_spanish' in self.tracks_df.columns else 0,
            'genre_mainstream': lambda row: 1 if row.get('popularity', 0) > 70 else 0,
            'genre_underground': lambda row: 1 if row.get('popularity', 0) < 30 else 0,
        }
        
        # Advanced pattern detection từ tên bài hát và artist
        pattern_genres = {
            'genre_electronic': ['electronic', 'edm', 'house', 'techno', 'trance', 'dubstep'],
            'genre_hip_hop': ['rap', 'hip hop', 'trap', 'drill'],
            'genre_rock': ['rock', 'metal', 'punk', 'grunge'],
            'genre_indie': ['indie', 'alternative', 'art'],
            'genre_rnb': ['r&b', 'soul', 'neo soul'],
            'genre_country': ['country', 'folk', 'bluegrass'],
            'genre_jazz': ['jazz', 'blues', 'swing'],
            'genre_classical': ['classical', 'orchestra', 'symphony'],
        }
        
        # Apply basic mapping
        for genre_col, func in genre_mapping.items():
            try:
                self.tracks_df[genre_col] = self.tracks_df.apply(func, axis=1)
            except Exception as e:
                logger.warning(f"Error creating {genre_col}: {e}")
                self.tracks_df[genre_col] = 0
        
        # Apply pattern detection
        for genre_col, patterns in pattern_genres.items():
            try:
                genre_pattern = '|'.join(patterns)
                
                # Check trong tên bài hát
                name_match = self.tracks_df['name'].str.contains(
                    genre_pattern, case=False, na=False
                ).astype(int)
                
                # Check trong tên artist
                artist_match = self.tracks_df['artist'].str.contains(
                    genre_pattern, case=False, na=False
                ).astype(int)
                
                # Combine matches
                self.tracks_df[genre_col] = (name_match | artist_match).astype(int)
                
            except Exception as e:
                logger.warning(f"Error creating {genre_col}: {e}")
                self.tracks_df[genre_col] = 0
        
        fallback_count = len([col for col in self.tracks_df.columns if col.startswith('genre_')])
        logger.info(f"Created {fallback_count} fallback genre features")

    def _create_compound_genre_features(self):
        """Create compound genre features from existing ones"""
        try:
            # Asian music categories
            asian_genres = ['genre_kpop', 'genre_jpop', 'genre_cpop']
            available_asian = [g for g in asian_genres if g in self.tracks_df.columns]
            if available_asian:
                self.tracks_df['genre_asian'] = self.tracks_df[available_asian].max(axis=1)
            
            # Electronic music umbrella
            electronic_patterns = ['house', 'techno', 'trance', 'dubstep', 'electronic']
            electronic_cols = [f'genre_{pattern}' for pattern in electronic_patterns 
                             if f'genre_{pattern}' in self.tracks_df.columns]
            if electronic_cols:
                self.tracks_df['genre_electronic_umbrella'] = self.tracks_df[electronic_cols].max(axis=1)
            
            # Popularity tiers as genre-like features
            if 'popularity' in self.tracks_df.columns:
                self.tracks_df['genre_viral'] = (self.tracks_df['popularity'] > 80).astype(int)
                self.tracks_df['genre_hit'] = ((self.tracks_df['popularity'] > 60) & 
                                             (self.tracks_df['popularity'] <= 80)).astype(int)
                self.tracks_df['genre_niche'] = (self.tracks_df['popularity'] < 30).astype(int)
            
        except Exception as e:
            logger.warning(f"Error creating compound genre features: {e}")

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
     
    def process_all(self):
        """Process all data steps"""
        logger.info("Starting complete data processing pipeline...")
        
        try:
            # Step 1: Load data
            if not self.load_data():
                logger.error("Failed to load data")
                return False
            
            # Step 2: Clean tracks data
            self.clean_tracks_data()
            
            # Step 3: Merge artist genres (với auto-fetch nếu thiếu)
            self.merge_artist_genres()
            
            # Step 4: Extract release year
            self.extract_release_year()
            
            # Step 5: Create genre features (real hoặc fallback)
            self.create_genre_features()
            
            # Step 6: Create language features
            self.create_language_features()
            
            # Step 7: Create additional features
            self.create_additional_features()
            
            # Step 8: Normalize features
            self.normalize_features()
            
            # Step 9: Save processed data
            output_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            self.tracks_df.to_csv(output_path, index=False)
            
            feature_count = len(self.tracks_df.columns)
            genre_feature_count = len([col for col in self.tracks_df.columns if col.startswith('genre_')])
            
            logger.info(f"✅ Processing complete!")
            logger.info(f"   📊 {len(self.tracks_df)} tracks processed")
            logger.info(f"   🏷️ {feature_count} total features")
            logger.info(f"   🎨 {genre_feature_count} genre features")
            logger.info(f"   💾 Saved to: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            return False

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()