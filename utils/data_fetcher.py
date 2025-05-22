import os
import time
import logging
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
from config.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, RAW_DATA_DIR

logger = logging.getLogger(__name__)

class SpotifyDataFetcher:
    """Class to fetch data from Spotify API"""
    
    def __init__(self):
        """Initialize Spotify API client"""
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET
            ))
            logger.info("Spotify API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spotify API client: {e}")
            raise
    
    def _extract_track_data(self, track):
        """Extract relevant data from a track object"""
        try:
            return {
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'] if track['artists'] else "Unknown",
                'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                'album': track['album']['name'] if 'album' in track else None,
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'explicit': int(track['explicit']),
                'release_date': track['album']['release_date'] if 'album' in track else None
            }
        except Exception as e:
            logger.warning(f"Error extracting track data: {e}")
            return None
    
    def fetch_tracks_by_search(self, queries, tracks_per_query=50, save_path=None, combine_with_existing=True):
        """Fetch tracks by search queries with improved error handling"""
        all_tracks = []
        
        for query in tqdm(queries, desc="Fetching tracks"):
            tracks_fetched = 0
            offset = 0
            max_retries = 3
            
            while tracks_fetched < tracks_per_query:
                retries = 0
                while retries < max_retries:
                    try:
                        # API call with exponential backoff
                        results = self.sp.search(q=query, type='track', limit=50, offset=offset)
                        break  # Success, exit retry loop
                    except Exception as e:
                        retries += 1
                        logger.warning(f"Error fetching query {query} (retry {retries}/{max_retries}): {e}")
                        if retries >= max_retries:
                            logger.error(f"Failed to fetch query {query} after {max_retries} retries")
                            break  # Max retries reached, skip this query
                        time.sleep(2 ** retries)  # Exponential backoff
                
                if retries >= max_retries:
                    break  # Skip this query after max retries
                    
                # Process results
                tracks = results['tracks']['items']
                if not tracks:
                    break  # No more tracks for this query
                    
                for track in tracks:
                    # Process track data
                    track_data = self._extract_track_data(track)
                    if track_data:
                        all_tracks.append(track_data)
                        tracks_fetched += 1
                        
                    if tracks_fetched >= tracks_per_query:
                        break
                        
                offset += len(tracks)
                time.sleep(0.5)  # Rate limiting
        
        # Tạo DataFrame và lưu
        if all_tracks:
            tracks_df = pd.DataFrame(all_tracks)
            
            # Loại bỏ trùng lặp
            tracks_df = tracks_df.drop_duplicates(subset=['id'])
            
            logger.info(f"Fetched {len(tracks_df)} unique tracks from {len(queries)} queries")
            
            if save_path:
                # Backup và merge nếu file đã tồn tại
                if combine_with_existing and os.path.exists(save_path):
                    existing_df = pd.read_csv(save_path)
                    logger.info(f"Found existing data with {len(existing_df)} tracks")
                    
                    # Merge và loại bỏ trùng lặp
                    combined_df = pd.concat([existing_df, tracks_df]).drop_duplicates(subset=['id'])
                    combined_df.to_csv(save_path, index=False)
                    logger.info(f"Combined with existing data. Total: {len(combined_df)} tracks")
                    return combined_df
                else:
                    tracks_df.to_csv(save_path, index=False)
                    logger.info(f"Saved {len(tracks_df)} tracks to {save_path}")
            
            return tracks_df
        else:
            logger.warning("No tracks fetched")
            return pd.DataFrame()
    
    def fetch_audio_features(self, track_ids, save_path=None):
        """
        Lấy đặc trưng âm thanh từ Spotify API
        
        Args:
            track_ids (list): Danh sách ID bài hát
            save_path (str, optional): Đường dẫn file để lưu
            
        Returns:
            pd.DataFrame: DataFrame chứa đặc trưng âm thanh
        """
        audio_features = []
        
        # Xử lý theo lô 100 bài hát (giới hạn API)
        for i in tqdm(range(0, len(track_ids), 100), desc="Fetching audio features"):
            batch = track_ids[i:i+100]
            
            # Thêm xử lý retry
            max_retries = 3
            retries = 0
            
            while retries < max_retries:
                try:
                    # Gọi API để lấy đặc trưng âm thanh cho lô
                    features_batch = self.sp.audio_features(batch)
                    
                    # Thêm vào danh sách kết quả
                    for feature in features_batch:
                        if feature:  # Kiểm tra không None
                            audio_features.append(feature)
                    
                    # Thành công, thoát vòng lặp retry
                    break
                except Exception as e:
                    retries += 1
                    logger.warning(f"Error fetching audio features (retry {retries}/{max_retries}): {e}")
                    if retries >= max_retries:
                        logger.error(f"Failed to fetch audio features after {max_retries} retries")
                        break
                    time.sleep(2 ** retries)  # Exponential backoff
            
            # Rate limiting - tránh bị hạn chế API
            time.sleep(1)
        
        # Chuyển thành DataFrame
        features_df = pd.DataFrame(audio_features)
        
        # Lọc chỉ giữ các cột cần thiết
        if not features_df.empty:
            features_df = features_df[[
                'id', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo', 'time_signature'
            ]]
        
        # Lưu vào file nếu có đường dẫn
        if save_path and not features_df.empty:
            features_df.to_csv(save_path, index=False)
            logger.info(f"Saved audio features for {len(features_df)} tracks to {save_path}")
        
        return features_df
    
    def fetch_artist_genres(self, artist_ids, save_path=None):
        """
        Lấy thông tin thể loại của nghệ sĩ từ Spotify API
        
        Args:
            artist_ids (list): Danh sách ID nghệ sĩ
            save_path (str, optional): Đường dẫn file để lưu
            
        Returns:
            pd.DataFrame: DataFrame chứa thông tin thể loại
        """
        artists_data = []
        
        # Xử lý theo lô 50 nghệ sĩ (giới hạn API)
        for i in tqdm(range(0, len(artist_ids), 50), desc="Fetching artist genres"):
            batch = artist_ids[i:i+50]
            
            # Thêm xử lý retry
            max_retries = 3
            retries = 0
            
            while retries < max_retries:
                try:
                    # Gọi API để lấy thông tin nghệ sĩ
                    artists_batch = self.sp.artists(batch)
                    
                    # Thêm vào danh sách kết quả
                    for artist in artists_batch['artists']:
                        if artist:  # Kiểm tra không None
                            artists_data.append({
                                'artist_id': artist['id'],
                                'artist_name': artist['name'],
                                'artist_popularity': artist['popularity'],
                                'artist_followers': artist['followers']['total'],
                                'artist_genres': '|'.join(artist['genres'])
                            })
                    
                    # Thành công, thoát vòng lặp retry
                    break
                except Exception as e:
                    retries += 1
                    logger.warning(f"Error fetching artist data (retry {retries}/{max_retries}): {e}")
                    if retries >= max_retries:
                        logger.error(f"Failed to fetch artist data after {max_retries} retries")
                        break
                    time.sleep(2 ** retries)  # Exponential backoff
            
            # Rate limiting - tránh bị hạn chế API
            time.sleep(1)
        
        # Chuyển thành DataFrame
        artists_df = pd.DataFrame(artists_data)
        
        # Lưu vào file nếu có đường dẫn
        if save_path and not artists_df.empty:
            artists_df.to_csv(save_path, index=False)
            logger.info(f"Saved genre data for {len(artists_df)} artists to {save_path}")
        
        return artists_df

# Hàm cấp cao để lấy dữ liệu ban đầu
def fetch_initial_dataset(tracks_per_query=50):
    """Fetch an initial dataset for the recommendation system"""
    # Create fetcher
    fetcher = SpotifyDataFetcher()
    
    # Define diverse search queries
    queries = [
        'pop 2023', 'rock 2023', 'hip hop 2023', 'rap 2023', 
        'electronic 2023', 'dance 2023', 'r&b 2023', 'indie 2023', 
        'classical', 'jazz', 'country', 'folk', 'metal', 'blues',
        'vietnamese music', 'vpop', 'vietnamese songs', 
        'top hits', 'billboard hot 100', 'new releases'
    ]
    
    # 1. Fetch basic track data
    tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
    tracks_df = fetcher.fetch_tracks_by_search(queries, tracks_per_query=tracks_per_query, save_path=tracks_path)
    
    if tracks_df.empty:
        logger.error("Failed to fetch tracks data")
        return None
    
    # 2. Fetch audio features
    if not tracks_df.empty:
        track_ids = tracks_df['id'].tolist()
        audio_features_path = os.path.join(RAW_DATA_DIR, 'audio_features.csv')
        fetcher.fetch_audio_features(track_ids, save_path=audio_features_path)
    
    # 3. Fetch artist genres
    if 'artist_id' in tracks_df.columns:
        artist_ids = tracks_df['artist_id'].dropna().unique().tolist()
        artist_genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
        fetcher.fetch_artist_genres(artist_ids, save_path=artist_genres_path)
    
    logger.info("Initial dataset fetched successfully")
    return tracks_df

if __name__ == "__main__":
    # Run if script is executed directly
    fetch_initial_dataset()