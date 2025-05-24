import os
import time
import logging
import pandas as pd
import spotipy
import numpy as np
from config.config import (
    SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, RAW_DATA_DIR,
    DEFAULT_TRACKS_PER_QUERY, LARGE_DATASET_BATCH_SIZE, LARGE_DATASET_SAVE_INTERVAL
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SpotifyDataFetcher:
    """Class to fetch data from Spotify API focusing on metadata only"""
    
    def __init__(self, client_id=None, client_secret=None):
        """Initialize Spotify API client with simplified authentication"""
        self.sp = None
        self.request_count = 0
        self.last_request_time = 0

        # Use provided or environment variables
        client_id = client_id or SPOTIFY_CLIENT_ID
        client_secret = client_secret or SPOTIFY_CLIENT_SECRET

        try:
            # Sử dụng Client Credentials Flow
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

    def _smart_delay(self, operation_type="search", batch_size=1, error_count=0):
        """Intelligent delay calculation based on Spotify rate limits"""
        current_time = time.time()
        
        # Spotify rate limit: 100 requests/minute = 1.67 requests/second
        # OPTIMIZE: Reduce delays significantly while staying under limits
        
        base_delays = {
            "search": 0.6,      # Giảm từ 1.5 xuống 0.6 (100 requests/60s = 0.6s/request)
            "artist": 0.8,      # Giảm từ 2.0 xuống 0.8  
            "batch": 0.4        # Cho batch operations
        }
        
        base_delay = base_delays.get(operation_type, 0.6)
        
        # Add error penalty (nhưng giảm)
        error_penalty = error_count * 1.0  # Giảm từ 2.0 xuống 1.0
        
        # Add small jitter to avoid synchronized requests
        jitter = np.random.uniform(0, 0.2)  # Giảm từ 0.5 xuống 0.2
        
        total_delay = base_delay + error_penalty + jitter
        
        # Cap maximum delay (giảm limits)
        max_delays = {
            "search": 2.0,      # Giảm từ 5.0 xuống 2.0
            "artist": 3.0,      # Giảm từ 8.0 xuống 3.0  
            "batch": 1.5
        }
        
        max_delay = max_delays.get(operation_type, 2.0)
        final_delay = min(total_delay, max_delay)
        
        # OPTIMIZE: Adaptive delay based on request frequency
        time_since_last = current_time - self.last_request_time
        if time_since_last > 2.0:  # If we've been idle, reduce delay
            final_delay *= 0.5
        
        self.last_request_time = current_time
        return final_delay

    def _extract_track_data(self, track):
        """Extract relevant data from track object"""
        try:
            artists = track.get('artists', [])
            artist_names = [artist['name'] for artist in artists]
            artist_ids = [artist['id'] for artist in artists]
            
            album = track.get('album', {})
            
            return {
                'id': track['id'],
                'name': track['name'],
                'artist': ', '.join(artist_names),
                'artist_id': artist_ids[0] if artist_ids else None,
                'album': album.get('name'),
                'album_id': album.get('id'),
                'album_type': album.get('album_type'),
                'total_tracks': album.get('total_tracks'),
                'popularity': track.get('popularity', 0),
                'duration_ms': track.get('duration_ms', 0),
                'explicit': track.get('explicit', False),
                'release_date': album.get('release_date'),
                'release_year': int(album.get('release_date', '1900')[:4]) if album.get('release_date') else None,
                'track_number': track.get('track_number'),
                'disc_number': track.get('disc_number'),
                'preview_url': track.get('preview_url'),
                'external_urls': track.get('external_urls', {}).get('spotify'),
                'is_local': track.get('is_local', False),
                'is_playable': track.get('is_playable', True),
                'markets_count': len(track.get('available_markets', []))
            }
        except Exception as e:
            logger.warning(f"Error extracting track data: {e}")
            return None

    def _filter_unwanted_tracks(self, track_data):
        """Filter out unwanted tracks (local, karaoke, etc.)"""
        if not track_data:
            return False
            
        name = track_data.get('name', '').lower()
        artist = track_data.get('artist', '').lower()
        
        # Skip local tracks
        if track_data.get('is_local', False):
            return False
            
        # Skip if not playable
        if not track_data.get('is_playable', True):
            return False
            
        # Skip very short tracks (likely interludes)
        if track_data.get('duration_ms', 0) < 30000:  # 30 seconds
            return False
            
        # Skip karaoke versions
        unwanted_keywords = ['karaoke', 'instrumental', 'backing track', 'cover version']
        if any(keyword in name for keyword in unwanted_keywords):
            return False
            
        return True

    def fetch_tracks_by_search(self, queries, tracks_per_query=50, save_path=None, combine_with_existing=True):
        """Fetch tracks by search queries with optimized timing"""
        all_tracks = []
        
        for query in tqdm(queries, desc="Fetching tracks"):
            tracks_fetched = 0
            offset = 0
            max_retries = 2  # Giảm từ 3 xuống 2
            tracks_per_query = min(tracks_per_query, 1000)
            target_tracks = tracks_per_query

            while tracks_fetched < target_tracks:
                retries = 0
                while retries < max_retries:
                    try:
                        results = self.sp.search(q=query, type='track', limit=50, offset=offset)
                        break
                    except Exception as e:
                        retries += 1
                        logger.warning(f"Error fetching query {query} (retry {retries}/{max_retries}): {e}")
                        if retries >= max_retries:
                            logger.error(f"Failed to fetch query {query} after {max_retries} retries")
                            break
                        # OPTIMIZE: Faster backoff
                        backoff_time = min(2 ** retries, 4.0)  # Cap at 4 seconds
                        time.sleep(backoff_time)
                
                if retries >= max_retries:
                    break
                    
                tracks = results['tracks']['items']
                if not tracks:
                    break
                    
                for track in tracks:
                    track_data = self._extract_track_data(track)
                    if track_data and self._filter_unwanted_tracks(track_data):
                        all_tracks.append(track_data)
                        tracks_fetched += 1
                        
                    if tracks_fetched >= target_tracks:
                        break
                        
                offset += len(tracks)
                
                # OPTIMIZE: Smart delay based on response
                if len(tracks) < 50:  # API returning fewer results
                    delay = self._smart_delay("search", error_count=1)
                else:
                    delay = self._smart_delay("search")
                
                time.sleep(delay)
        
        # Convert to DataFrame and save
        tracks_df = pd.DataFrame(all_tracks)
        
        if save_path:
            if combine_with_existing and os.path.exists(save_path):
                try:
                    existing_df = pd.read_csv(save_path)
                    combined_df = pd.concat([existing_df, tracks_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['id'])
                    combined_df.to_csv(save_path, index=False)
                    logger.info(f"Combined and saved {len(combined_df)} tracks to {save_path}")
                except Exception as e:
                    logger.warning(f"Error combining with existing data: {e}")
                    tracks_df.to_csv(save_path, index=False)
                    logger.info(f"Saved {len(tracks_df)} new tracks to {save_path}")
            else:
                tracks_df.to_csv(save_path, index=False)
                logger.info(f"Saved {len(tracks_df)} tracks to {save_path}")
        
        return tracks_df

    def fetch_artist_genres(self, artist_ids, save_path=None, batch_size=20):
        """Fetches genres for a list of artist IDs with optimized timing"""
        if not artist_ids:
            logger.warning("No artist IDs provided to fetch_artist_genres")
            return pd.DataFrame()
        
        all_genres = []
        
        # OPTIMIZE: Increase batch size and reduce delays
        batch_size = min(batch_size, 25)  # Tăng từ 10 lên 25
        total_batches = (len(artist_ids) + batch_size - 1) // batch_size
        
        # FIX: Remove emoji that causes Unicode error in Windows
        print(f"Fetching genres for {len(artist_ids)} artists in {total_batches} batches...")
        
        for i in tqdm(range(0, len(artist_ids), batch_size), desc="Fetching artist genres"):
            batch = artist_ids[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # OPTIMIZE: Faster retry logic
                max_retries = 2  # Giảm từ 3 xuống 2
                for retry in range(max_retries):
                    try:
                        artists_data = self.sp.artists(batch)['artists']
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            wait_time = min(2 ** retry, 3.0)  # Cap at 3 seconds
                            logger.warning(f"Retry {retry + 1}/{max_retries} for batch {batch_num}, waiting {wait_time:.1f}s")
                            time.sleep(wait_time)
                        else:
                            raise e
                
                # Process successful response
                batch_genres = []
                for artist in artists_data:
                    if artist and 'id' in artist and 'genres' in artist:
                        genre_data = {
                            'artist_id': artist['id'],
                            'genres': '|'.join(artist['genres']) if artist['genres'] else '',
                            'artist_popularity': artist.get('popularity', 0),
                            'artist_followers': artist.get('followers', {}).get('total', 0) if 'followers' in artist else 0,
                        }
                        batch_genres.append(genre_data)
                        all_genres.append(genre_data)
                
                # Log progress less frequently
                if batch_num % 20 == 0 or batch_num == total_batches:
                    logger.info(f"Progress: {batch_num}/{total_batches} batches, {len(all_genres)} artists processed")
                
                # OPTIMIZE: Much shorter delays
                delay = self._smart_delay("artist", batch_size=len(batch))
                time.sleep(delay)
                
            except Exception as e:
                logger.warning(f"Error fetching artist data (batch {batch_num}): {e}")
                # OPTIMIZE: Shorter error delay
                time.sleep(2.0)  # Giảm từ 10.0 xuống 2.0
                continue
        
        # Create and save DataFrame
        genres_df = pd.DataFrame(all_genres)
        
        if save_path and not genres_df.empty:
            # Merge with existing data if available
            if os.path.exists(save_path):
                try:
                    existing_df = pd.read_csv(save_path)
                    combined_df = pd.concat([existing_df, genres_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['artist_id'])
                    combined_df.to_csv(save_path, index=False)
                    logger.info(f"Merged and saved {len(combined_df)} artist genres to {save_path}")
                except Exception as e:
                    logger.warning(f"Error merging with existing data: {e}")
                    genres_df.to_csv(save_path, index=False)
                    logger.info(f"Saved {len(genres_df)} new artist genres to {save_path}")
            else:
                genres_df.to_csv(save_path, index=False)
                logger.info(f"Saved {len(genres_df)} artist genres to {save_path}")
        
        return genres_df

    def fetch_all_missing_artist_genres(self):
        """Fetch all missing artist genres with optimized timing"""
        print("FETCHING ALL MISSING ARTIST GENRES (OPTIMIZED)...")
        
        # Load tracks data
        tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
        if not os.path.exists(tracks_path):
            logger.error("No spotify_tracks.csv found!")
            return False
        
        tracks_df = pd.read_csv(tracks_path)
        logger.info(f"Found {len(tracks_df)} tracks")
        
        # Get unique artist IDs
        if 'artist_id' not in tracks_df.columns:
            logger.error("No artist_id column found!")
            return False
        
        artist_ids = tracks_df['artist_id'].dropna().unique().tolist()
        logger.info(f"Found {len(artist_ids)} unique artists")
        
        # Check existing genres
        genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
        if os.path.exists(genres_path):
            existing_genres = pd.read_csv(genres_path)
            existing_ids = set(existing_genres['artist_id'].tolist())
            missing_ids = [aid for aid in artist_ids if aid not in existing_ids]
            logger.info(f"Found {len(existing_ids)} existing, need {len(missing_ids)} more")
        else:
            missing_ids = artist_ids
            logger.info(f"No existing genres, fetching all {len(missing_ids)} artists")
        
        if not missing_ids:
            logger.info("All artist genres already available!")
            return True
        
        # OPTIMIZE: Estimate time more accurately
        estimated_batches = (len(missing_ids) + 24) // 25  # 25 per batch
        estimated_time = estimated_batches * 1.0  # ~1 second per batch
        logger.info(f"Estimated time: {estimated_time/60:.1f} minutes for {estimated_batches} batches")
        
        # Fetch missing genres with optimized settings
        try:
            result_df = self.fetch_artist_genres(
                missing_ids, 
                save_path=genres_path,
                batch_size=25  # Tăng batch size
            )
            
            if not result_df.empty:
                logger.info(f"Successfully fetched {len(result_df)} artist genres!")
                return True
            else:
                logger.warning("No genres were fetched")
                return False
                
        except Exception as e:
            logger.error(f"Error fetching artist genres: {e}")
            return False

# Hàm cấp cao để lấy dữ liệu ban đầu
def fetch_initial_dataset(tracks_per_query=50):
    """Fetch initial dataset with popular music queries"""
    
    popular_queries = [
        # Pop & Mainstream
        "top hits 2024", "billboard hot 100", "pop music 2024", "viral songs",
        "trending music", "popular songs", "chart toppers", "radio hits",
        
        # Genres
        "indie pop", "alternative rock", "electronic music", "hip hop 2024",
        "r&b music", "country music", "latin music", "k-pop",
        
        # Moods & Vibes  
        "chill music", "upbeat songs", "sad songs", "party music",
        "workout music", "study music", "relaxing music", "happy songs",
        
        # Decades
        "2020s music", "2010s hits", "2000s nostalgia", "90s music",
        "80s classics", "throwback songs",
        
        # Artists & Playlists
        "taylor swift", "billie eilish", "the weeknd", "dua lipa",
        "ed sheeran", "ariana grande", "post malone", "drake",
        
        # International
        "vietnamese music", "vpop", "japanese music", "spanish songs",
        "latin pop", "korean music", "chinese songs", "thai music",
        
        # Specific themes
        "love songs", "breakup songs", "motivational music", "acoustic",
        "piano music", "guitar songs", "jazz music", "blues",
        
        # Albums & EPs
        "new albums 2024", "best albums", "debut albums", "ep releases",
        
        # Seasonal
        "summer songs", "winter music", "holiday songs", "festival music",
        
        # Underground & Indie
        "indie rock", "underground hip hop", "bedroom pop", "lo-fi",
        "dream pop", "shoegaze", "post rock", "experimental",
        
        # Dance & Electronic
        "edm", "house music", "techno", "trance", "dubstep",
        "electronic dance", "club music", "rave music",
        
        # Rock & Metal
        "rock music", "metal songs", "punk rock", "alternative metal",
        "hard rock", "classic rock", "indie rock",
        
        # Regional scenes
        "uk music", "australian music", "canadian artists", "european pop",
        "african music", "middle eastern music", "scandinavian pop",
        
        # Awards & Recognition
        "grammy winners 2024", "grammy nominees", "award winning songs",
        "critically acclaimed", "music awards",
        
        # Collaborations
        "featured artists", "duets", "collaborations", "remix",
        
        # Rising artists
        "new artists 2024", "rising stars", "breakthrough artists",
        "emerging talent", "next big thing"
    ]
    
    logger.info(f"Fetching {len(popular_queries)} queries with {tracks_per_query} tracks each (target: {len(popular_queries) * tracks_per_query} total)")
    
    try:
        fetcher = SpotifyDataFetcher()
        tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
        
        # Fetch tracks with new optimized timing
        tracks_df = fetcher.fetch_tracks_by_search(
            popular_queries, 
            tracks_per_query=tracks_per_query,
            save_path=tracks_path,
            combine_with_existing=True
        )
        
        if tracks_df.empty:
            logger.error("No tracks were fetched")
            return None
        
        logger.info(f"Successfully fetched {len(tracks_df)} tracks")
        
        # Get unique artists for genre fetching
        if 'artist_id' in tracks_df.columns:
            unique_artists = tracks_df['artist_id'].dropna().nunique()
            logger.info(f"Found {unique_artists} unique artists")
            
            # Fetch artist genres automatically
            artist_ids = tracks_df['artist_id'].dropna().unique().tolist()
            artist_genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
            
            logger.info("Fetching artist genres...")
            fetcher.fetch_artist_genres(artist_ids, save_path=artist_genres_path)
        
        return tracks_df
        
    except Exception as e:
        logger.error(f"Error in fetch_initial_dataset: {e}")
        return None

# ...existing methods unchanged...