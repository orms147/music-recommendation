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
        if os.path.exists(tracks_path):
            self.tracks_df = pd.read_csv(tracks_path)
            logger.info(f"Loaded {len(self.tracks_df)} tracks from {tracks_path}")
        else:
            logger.warning(f"Track data file not found: {tracks_path}")
            self.tracks_df = pd.DataFrame()
        
        # Đọc đặc trưng âm thanh
        audio_path = os.path.join(RAW_DATA_DIR, 'audio_features.csv')
        if os.path.exists(audio_path):
            self.audio_features_df = pd.read_csv(audio_path)
            logger.info(f"Loaded audio features for {len(self.audio_features_df)} tracks")
        else:
            logger.warning(f"Audio features file not found: {audio_path}")
            self.audio_features_df = pd.DataFrame()
        
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
        
        # Log kết quả
        clean_count = len(self.tracks_df)
        logger.info(f"Cleaned tracks data: {initial_count} -> {clean_count} tracks")
        
        return True
    
    def merge_audio_features(self):
        """Merge audio features to tracks data"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to merge audio features with")
            return False
        
        if self.audio_features_df is None or self.audio_features_df.empty:
            logger.warning("No audio features to merge")
            return False
        
        # Lưu số lượng ban đầu
        initial_count = len(self.tracks_df)
        
        # Kết hợp với dữ liệu bài hát
        self.tracks_df = self.tracks_df.merge(self.audio_features_df, on='id', how='left')
        
        # Điền missing values cho các đặc trưng âm thanh
        audio_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'time_signature'
        ]
        
        for feature in audio_features:
            if feature in self.tracks_df.columns:
                # Sử dụng giá trị trung bình cho dữ liệu bị thiếu
                mean_value = self.tracks_df[feature].mean()
                self.tracks_df[feature] = self.tracks_df[feature].fillna(mean_value)
        
        # Log kết quả
        merged_count = len(self.tracks_df)
        logger.info(f"Merged audio features: {initial_count} tracks -> {merged_count} tracks")
        
        return True
    
    def merge_artist_genres(self):
        """Merge artist genres to tracks data"""
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
        
        # Kết hợp với dữ liệu bài hát
        merged_df = self.tracks_df.merge(
            self.artist_genres_df[['artist_id', 'artist_genres', 'artist_popularity']], 
            left_on='artist_id', 
            right_on='artist_id', 
            how='left'
        )
        
        # Điền missing values
        merged_df['artist_genres'] = merged_df['artist_genres'].fillna('')
        merged_df['artist_popularity'] = merged_df['artist_popularity'].fillna(
            merged_df['popularity'] if 'popularity' in merged_df.columns else 50
        )
        
        self.tracks_df = merged_df
        
        # Log kết quả
        merged_count = len(self.tracks_df)
        logger.info(f"Merged artist genres: {initial_count} tracks -> {merged_count} tracks")
        
        return True
    
    def extract_release_year(self):
        """Extract release year from release date"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to extract release year from")
            return False
        
        if 'release_date' not in self.tracks_df.columns:
            logger.warning("No release_date column in tracks data")
            return False
        
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
        logger.info("Extracted release year from release dates")
        
        return True
    
    def create_genre_features(self):
        """Create genre features from artist_genres column"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create genre features")
            return False
        
        if 'artist_genres' not in self.tracks_df.columns:
            logger.warning("No artist_genres column in tracks data")
            return False
        
        # Extract main genres
        top_genres = []
        
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
            col_name = f'genre_{genre.replace(" ", "_")}'
            self.tracks_df[col_name] = self.tracks_df['artist_genres'].apply(
                lambda x: 1 if genre in str(x).split('|') else 0
            )
        
        logger.info(f"Created {top_n} genre binary features")
        
        return True
    
    def normalize_features(self):
        """Normalize numerical features"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to normalize")
            return False
        
        # Các đặc trưng số cần chuẩn hóa
        numeric_features = [
            'popularity', 'duration_ms', 'danceability', 'energy', 
            'loudness', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo'
        ]
        
        # Lọc các đặc trưng có trong DataFrame
        features_to_normalize = [f for f in numeric_features if f in self.tracks_df.columns]
        
        if not features_to_normalize:
            logger.warning("No numeric features to normalize")
            return False
        
        # Sử dụng MinMaxScaler để chuẩn hóa về khoảng [0,1]
        scaler = MinMaxScaler()
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
        """Process all data"""
        # Tạo thư mục processed nếu chưa tồn tại
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # 1. Clean tracks data
        self.clean_tracks_data()
        
        # 2. Merge audio features
        self.merge_audio_features()
        
        # 3. Merge artist genres
        self.merge_artist_genres()
        
        # 4. Extract release year
        self.extract_release_year()
        
        # 5. Create genre features
        self.create_genre_features()
        
        # 6. Normalize features
        self.normalize_features()
        
        # 7. Lưu dữ liệu đã xử lý
        track_features_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
        self.tracks_df.to_csv(track_features_path, index=False)
        logger.info(f"Processed tracks data saved to {track_features_path}")
        
        # 8. Create user-item matrix
        user_matrix_path = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')
        self.create_user_item_matrix(output_path=user_matrix_path)
        
        logger.info("All data processing complete")
        return True

if __name__ == "__main__":
    processor = DataProcessor()
    processor.load_data()
    processor.process_all()