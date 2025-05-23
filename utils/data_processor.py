import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
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
        """Load raw data from files"""
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
        
        # Đọc thông tin thể loại nghệ sĩ
        genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
        if os.path.exists(genres_path):
            self.artist_genres_df = pd.read_csv(genres_path)
            logger.info(f"Loaded genre data for {len(self.artist_genres_df)} artists")
        else:
            logger.warning(f"Artist genres file not found: {genres_path}")
            self.artist_genres_df = pd.DataFrame()
        
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
        """Create synthetic audio features when real ones aren't available"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create synthetic features for")
            return False
        
        logger.info("Creating synthetic audio features")
        
        # Define the audio features and their default values
        audio_features = {
            'danceability': 0.5,
            'energy': 0.5,
            'key': 0,
            'loudness': -10,
            'mode': 1,
            'speechiness': 0.1,
            'acousticness': 0.5,
            'instrumentalness': 0.01,
            'liveness': 0.1,
            'valence': 0.5,
            'tempo': 120,
            'time_signature': 4
        }
        
        # Check which features are already present
        existing_features = [f for f in audio_features.keys() if f in self.tracks_df.columns]
        missing_features = [f for f in audio_features.keys() if f not in self.tracks_df.columns]
        
        if existing_features:
            logger.info(f"Found existing audio features: {', '.join(existing_features)}")
        
        if missing_features:
            logger.info(f"Adding synthetic audio features: {', '.join(missing_features)}")
            
            # Add missing features with default values
            for feature in missing_features:
                # Set default value
                self.tracks_df[feature] = audio_features[feature]
                
                # Modulate based on available data to make synthetic features more realistic
                if feature == 'energy' and 'popularity' in self.tracks_df.columns:
                    # Higher popularity often correlates with higher energy
                    self.tracks_df[feature] = 0.3 + (self.tracks_df['popularity'] / 100) * 0.5
                
                elif feature == 'danceability' and 'duration_ms' in self.tracks_df.columns:
                    # Shorter songs tend to be more danceable
                    avg_duration = 3.5 * 60 * 1000  # 3.5 minutes in ms
                    self.tracks_df[feature] = 0.5 + 0.3 * (1 - np.minimum(self.tracks_df['duration_ms'] / avg_duration, 1))
                
                elif feature == 'valence' and 'name' in self.tracks_df.columns:
                    # Try to infer mood from track name
                    happy_keywords = ['happy', 'joy', 'fun', 'party', 'love', 'dance']
                    sad_keywords = ['sad', 'blue', 'cry', 'tear', 'alone', 'broken', 'lost']
                    
                    # Default valence
                    self.tracks_df[feature] = 0.5
                    
                    # Check for happy keywords
                    for keyword in happy_keywords:
                        self.tracks_df.loc[
                            self.tracks_df['name'].str.contains(keyword, case=False, na=False),
                            feature
                        ] = np.maximum(self.tracks_df[feature], 0.7)
                    
                    # Check for sad keywords
                    for keyword in sad_keywords:
                        self.tracks_df.loc[
                            self.tracks_df['name'].str.contains(keyword, case=False, na=False),
                            feature
                        ] = np.minimum(self.tracks_df[feature], 0.3)
        
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
        """Create additional features from metadata"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create additional features")
            return False
        
        # 1. Calculate duration in minutes
        if 'duration_ms' in self.tracks_df.columns and 'duration_min' not in self.tracks_df.columns:
            self.tracks_df['duration_min'] = self.tracks_df['duration_ms'] / 60000
            
            # Create duration category
            self.tracks_df['duration_category'] = pd.cut(
                self.tracks_df['duration_min'],
                bins=[0, 2, 3, 4, 6, 100],
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            )
        
        # 2. Create popularity categories
        if 'popularity' in self.tracks_df.columns and 'popularity_category' not in self.tracks_df.columns:
            self.tracks_df['popularity_category'] = pd.cut(
                self.tracks_df['popularity'],
                bins=[0, 20, 40, 60, 80, 100],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        # 3. Extract features from track name
        if 'name' in self.tracks_df.columns:
            # Has feature/collab indicated by 'feat.', 'ft.', 'with'
            has_collab_pattern = r'(feat\.?|ft\.?|with)\s'
            self.tracks_df['has_collab'] = self.tracks_df['name'].str.contains(
                has_collab_pattern, case=False, regex=True, na=False
            ).astype(int)
            
            # Is a remix/edit
            self.tracks_df['is_remix'] = self.tracks_df['name'].str.contains(
                r'(remix|edit|version|mix|dub|remaster)', case=False, na=False
            ).astype(int)
            
            # Length of track name (can indicate complexity)
            self.tracks_df['name_length'] = self.tracks_df['name'].str.len()
        
        # 4. Create artist frequency features
        if 'artist' in self.tracks_df.columns:
            # Count frequency of each artist
            artist_counts = self.tracks_df['artist'].value_counts()
            
            # Map counts back to tracks
            self.tracks_df['artist_frequency'] = self.tracks_df['artist'].map(artist_counts)
            
            # Normalize to 0-1
            max_freq = self.tracks_df['artist_frequency'].max()
            if max_freq > 0:
                self.tracks_df['artist_frequency_norm'] = self.tracks_df['artist_frequency'] / max_freq
        
        logger.info("Created additional metadata-based features")
        
        return True
    
    def normalize_features(self):
        """Normalize numerical features"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to normalize")
            return False
        
        # Các đặc trưng số cần chuẩn hóa (bao gồm cả đặc trưng từ metadata)
        numeric_features = [
            'popularity', 'duration_ms', 'duration_min',
            'artist_popularity', 'artist_frequency', 'release_year',
            'name_length', 'artist_followers'
        ]
        
        # Thêm đặc trưng audio nếu có
        audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        numeric_features.extend([f for f in audio_features if f in self.tracks_df.columns])
        
        # Lọc các đặc trưng có trong DataFrame
        features_to_normalize = [f for f in numeric_features if f in self.tracks_df.columns]
        
        if not features_to_normalize:
            logger.warning("No numeric features to normalize")
            return False
        
        # Sử dụng MinMaxScaler để chuẩn hóa về khoảng [0,1]
        scaler = MinMaxScaler()
        
        # Tiền xử lý - thay thế giá trị NaN bằng trung bình
        for feature in features_to_normalize:
            if self.tracks_df[feature].isna().any():
                self.tracks_df[feature] = self.tracks_df[feature].fillna(self.tracks_df[feature].mean())
        
        # Chuẩn hóa
        self.tracks_df[features_to_normalize] = scaler.fit_transform(self.tracks_df[features_to_normalize])
        
        logger.info(f"Normalized {len(features_to_normalize)} features to range [0,1]")
        
        return True
    
    def create_user_item_matrix(self, output_path=None):
        """Create user-item matrix for collaborative filtering"""
        # Nếu không có dữ liệu tương tác người dùng thực sự, tạo dữ liệu giả
        logger.info("Creating synthetic user-item matrix")
        
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create user-item matrix")
            return False
        
        # Số lượng người dùng giả
        num_users = 100
        # Số lượng tương tác trung bình mỗi người dùng
        interactions_per_user = 20
        
        # Tạo dữ liệu tương tác giả
        user_data = []
        
        for user_id in range(1, num_users + 1):
            # Tạo sở thích người dùng giả lập
            # Mỗi người dùng thích một số thể loại và nghệ sĩ cụ thể
            
            # Lựa chọn nghệ sĩ yêu thích cho mỗi người dùng
            if 'artist' in self.tracks_df.columns:
                favorite_artists = np.random.choice(
                    self.tracks_df['artist'].unique(),
                    size=min(5, len(self.tracks_df['artist'].unique())),
                    replace=False
                )
                
                # Ưu tiên bài hát từ nghệ sĩ yêu thích
                favorite_tracks = self.tracks_df[self.tracks_df['artist'].isin(favorite_artists)]
                other_tracks = self.tracks_df[~self.tracks_df['artist'].isin(favorite_artists)]
                
                # Số lượng bài favorite vs other
                fav_count = min(int(interactions_per_user * 0.7), len(favorite_tracks))
                other_count = min(interactions_per_user - fav_count, len(other_tracks))
                
                # Lấy mẫu
                if fav_count > 0:
                    fav_sample = favorite_tracks.sample(n=fav_count)
                else:
                    fav_sample = pd.DataFrame()
                    
                if other_count > 0:
                    other_sample = other_tracks.sample(n=other_count)
                else:
                    other_sample = pd.DataFrame()
                
                user_tracks = pd.concat([fav_sample, other_sample])
            else:
                # Nếu không có cột artist, lấy mẫu ngẫu nhiên
                sample_size = min(interactions_per_user, len(self.tracks_df))
                user_tracks = self.tracks_df.sample(n=sample_size)
            
            # Tạo điểm số - cao hơn cho bài hát từ nghệ sĩ yêu thích
            for _, track in user_tracks.iterrows():
                is_favorite = 'artist' in self.tracks_df.columns and track['artist'] in favorite_artists
                rating = np.random.randint(4, 6) if is_favorite else np.random.randint(1, 6)
                
                user_data.append({
                    'user_id': f"user_{user_id}",
                    'track_id': track['id'],
                    'rating': rating
                })
        
        # Tạo DataFrame
        interactions_df = pd.DataFrame(user_data)
        
        # Tạo ma trận user-item
        user_item_df = interactions_df.pivot(
            index='user_id',
            columns='track_id',
            values='rating'
        ).fillna(0)
        
        # Lưu ma trận nếu có đường dẫn
        if output_path:
            user_item_df.to_csv(output_path)
            logger.info(f"Saved user-item matrix with {len(user_item_df)} users and {len(user_item_df.columns)} tracks")
        
        self.user_item_matrix = user_item_df
        
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