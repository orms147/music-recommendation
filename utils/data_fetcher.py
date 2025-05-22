import os
import time
import logging
import pandas as pd
import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyOAuth
import webbrowser
from config.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, RAW_DATA_DIR
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Cập nhật Redirect URI để khớp với ứng dụng Gradio
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:7860/callback"

class SpotifyDataFetcher:
    """Class to fetch data from Spotify API"""
    
    def __init__(self, client_id=None, client_secret=None):
        """Initialize Spotify API client with simplified authentication"""
        self.sp = None

        # Use provided credentials or environment variables
        client_id = client_id or SPOTIFY_CLIENT_ID
        client_secret = client_secret or SPOTIFY_CLIENT_SECRET

        try:
            # Sử dụng Client Credentials Flow thay vì OAuth
            auth_manager = spotipy.oauth2.SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )

            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            # Test kết nối
            self.sp.search(q="test", limit=1)
            logger.info("Kết nối Spotify API thành công")
        except Exception as e:
            logger.error(f"Lỗi kết nối Spotify API: {e}")
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
    
    def fetch_audio_features(self, track_ids):
        """Fetch audio features for a list of track IDs with improved error handling"""
        if not track_ids:
            return pd.DataFrame()
        
        # Đảm bảo track_ids là list
        if isinstance(track_ids, pd.Series):
            track_ids = track_ids.tolist()
        
        max_retries = 5
        retry_count = 0
        backoff_time = 1
        
        while retry_count < max_retries:
            try:
                # Lấy audio features
                audio_features = self.sp.audio_features(track_ids)
                
                # Kiểm tra kết quả có hợp lệ không
                if not audio_features or all(af is None for af in audio_features):
                    logger.warning("Không tìm thấy audio features cho các track ID đã cho")
                    return pd.DataFrame()
                
                # Loại bỏ các kết quả None
                audio_features = [af for af in audio_features if af is not None]
                
                # Chuyển đổi sang DataFrame
                audio_df = pd.DataFrame(audio_features)
                
                return audio_df
                
            except Exception as e:
                retry_count += 1
                error_message = str(e)
                
                if "rate limiting" in error_message.lower() or "429" in error_message:
                    logger.warning(f"Rate limit exceeded. Retrying in {backoff_time} seconds...")
                elif "403" in error_message:
                    logger.warning(f"Forbidden error (403). Client credentials may be expired. Retrying...")
                    # Thử khởi tạo lại client
                    try:
                        auth_manager = spotipy.oauth2.SpotifyClientCredentials(
                            client_id=SPOTIFY_CLIENT_ID,
                            client_secret=SPOTIFY_CLIENT_SECRET
                        )
                        self.sp = spotipy.Spotify(auth_manager=auth_manager)
                    except Exception as auth_error:
                        logger.error(f"Failed to reinitialize Spotify client: {auth_error}")
                else:
                    logger.warning(f"Error fetching audio features: {e}. Retrying {retry_count}/{max_retries}...")
                
                if retry_count < max_retries:
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Tăng thời gian chờ theo cấp số nhân
                else:
                    logger.error(f"Failed after {max_retries} retries: {e}")
                    return pd.DataFrame()
            
    def _reinitialize_client(self):
        """Reinitialize Spotify client to refresh token"""
        try:
            logger.info("Reinitializing Spotify client...")
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope="user-library-read user-top-read",
                cache_path=".spotifycache"
            ))
            # Test connection
            self.sp.search(q="test", limit=1)
            logger.info("Spotify client reinitialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reinitialize Spotify client: {e}")
            return False

    def fetch_artist_genres(self, artist_ids, save_path=None, batch_size=20):
        """Fetches genres for a list of artist IDs"""
        if not artist_ids:
            logger.warning("No artist IDs provided to fetch_artist_genres")
            return pd.DataFrame()
        
        all_genres = []
        
        # Process in smaller batches
        for i in range(0, len(artist_ids), batch_size):
            batch = artist_ids[i:i+batch_size]
            try:
                artists_data = self.sp.artists(batch)['artists']
                
                for artist in artists_data:
                    if artist and 'id' in artist and 'genres' in artist:
                        all_genres.append({
                            'artist_id': artist['id'],
                            'genres': ','.join(artist['genres']) if artist['genres'] else ''
                        })
                
                # Wait to avoid rate limiting
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Error fetching artist data (batch {i//batch_size + 1}): {e}")
        
        genres_df = pd.DataFrame(all_genres)
        
        if save_path and not genres_df.empty:
            genres_df.to_csv(save_path, index=False)
            logger.info(f"Saved {len(genres_df)} artist genres to {save_path}")
        
        return genres_df

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