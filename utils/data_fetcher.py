import os
import time
import logging
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from config.config import (
    SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, RAW_DATA_DIR,
    DEFAULT_TRACKS_PER_QUERY, LARGE_DATASET_BATCH_SIZE, LARGE_DATASET_SAVE_INTERVAL
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SpotifyDataFetcher:
    """Class to fetch data from Spotify API focusing on ISRC and metadata"""
    
    def __init__(self, client_id=None, client_secret=None):
        """Initialize Spotify client with credentials"""
        try:
            self.client_id = client_id or SPOTIFY_CLIENT_ID
            self.client_secret = client_secret or SPOTIFY_CLIENT_SECRET
            
            if not self.client_id or not self.client_secret:
                raise ValueError("Spotify client ID and secret are required")
            
            # Initialize Spotify client
            client_credentials_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            
            # Test connection
            self.sp.search('test', limit=1)
            logger.info("Spotify API connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            raise

    def _extract_track_data(self, track):
        """Extract enhanced track data from Spotify API"""
        try:
            # Basic track info
            track_id = track['id']
            name = track['name']
            
            # Artists info
            artists = track.get('artists', [])
            artist_names = [artist['name'] for artist in artists]
            artist_ids = [artist['id'] for artist in artists]
            
            # Album info
            album = track.get('album', {})
            
            # Cải thiện: Thu thập thêm thông tin từ API
            return {
                'id': track_id,
                'name': name,
                'artist': ', '.join(artist_names),
                'artist_id': artist_ids[0] if artist_ids else None,
                'album': album.get('name', ''),
                'album_type': album.get('album_type', ''),
                'popularity': track.get('popularity', 0),
                'duration_ms': track.get('duration_ms', 0),
                'release_date': album.get('release_date', ''),
                'release_date_precision': album.get('release_date_precision', ''),
                # ISRC và thông tin thị trường
                'isrc': track.get('external_ids', {}).get('isrc', ''),
                'available_markets': '|'.join(track.get('available_markets', [])),
                'markets_count': len(track.get('available_markets', [])),
                # Cải thiện: Thêm các trường mới
                'is_playable': track.get('is_playable', True),
                'explicit': track.get('explicit', False),
                'disc_number': track.get('disc_number', 1),
                'track_number': track.get('track_number', 1),
                'preview_url': track.get('preview_url', ''),
                'has_preview': track.get('preview_url', '') != '',
            }
            
        except Exception as e:
            logger.warning(f"Error extracting track data: {e}")
            return None

    def search_tracks_by_query(self, query, limit=50):
        """Search tracks by query"""
        all_tracks = []
        
        try:
            offset = 0
            max_requests = min(20, (limit + 49) // 50)
            
            for _ in range(max_requests):
                search_limit = min(50, limit - len(all_tracks))
                if search_limit <= 0:
                    break
                
                results = self.sp.search(
                    q=query, 
                    type='track', 
                    limit=search_limit, 
                    offset=offset
                )
                
                tracks = results['tracks']['items']
                
                for track in tracks:
                    track_data = self._extract_track_data(track)
                    if track_data:
                        all_tracks.append(track_data)
                
                if len(tracks) < search_limit:
                    break
                    
                offset += search_limit
                time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
        
        logger.info(f"Found {len(all_tracks)} tracks for query: '{query}'")
        return all_tracks

    def fetch_artist_details(self, artist_ids, batch_size=50):
        """Enhanced artist details fetching with followers"""
        if not artist_ids:
            return {}
        
        artist_details = {}
        unique_ids = list(dict.fromkeys(artist_ids))
        
        try:
            for i in range(0, len(unique_ids), batch_size):
                batch = unique_ids[i:i+batch_size]
                
                artists_data = self.sp.artists(batch)['artists']
                
                for artist in artists_data:
                    if artist:
                        # Cải thiện: Thu thập thêm thông tin followers
                        artist_details[artist['id']] = {
                            'name': artist.get('name', ''),
                            'genres': artist.get('genres', []),
                            'popularity': artist.get('popularity', 0),
                            'followers': artist.get('followers', {}).get('total', 0),
                            'images': artist.get('images', []),
                            'has_image': len(artist.get('images', [])) > 0
                        }
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error fetching artist details: {e}")
        
        return artist_details

    def create_diverse_dataset(self, tracks_per_category=100):
        """Create diverse dataset"""
        base_queries = [
            'Billboard Hot 100',
            'Grammy Award winners',
            'top usuk songs',
            'UK Top 40', 'American Top 40',
            'Brit Awards',

            'J-Pop top chart',
            'Oricon Chart',
            'Japanese music awards',

            'Vietnamese music',
            'top Vietnamese songs',
            'Vietnamese indie',
            'Zing Chart',
            'Làn Sóng Xanh',
            'Vietnamese rap',

            'US rap', 'UK rap', 'Japanese rap',

            'C-Pop', 'Mandarin pop', 'Chinese top songs',

            'K-Pop', 'MAMA Awards', 'Melon Top 100',
            'Golden Disc Awards', 'Gaon Chart',

            'top EDM songs', 'EDM festival anthems',
            'Ultra Music Festival', 'Tomorrowland hits'
        ]
        

        years = list(range(2010, 2026))  # 2010 ~ 2025

        # Kết hợp từng query với từng năm
        search_queries = [f"{query} {year}" for query in base_queries for year in years]


        
        all_tracks = []
        
        logger.info(f"Fetching diverse dataset with {len(search_queries)} categories...")
        
        for query in tqdm(search_queries, desc="Fetching categories"):
            tracks = self.search_tracks_by_query(query, limit=tracks_per_category)
            all_tracks.extend(tracks)
            time.sleep(0.5)
        
        # Remove duplicates
        unique_tracks = {}
        for track in all_tracks:
            if track['id'] not in unique_tracks:
                unique_tracks[track['id']] = track
        
        final_tracks = list(unique_tracks.values())
        logger.info(f"Created diverse dataset: {len(final_tracks)} unique tracks")
        
        return final_tracks

    def fetch_and_save_dataset(self, filename='tracks.csv', tracks_per_category=100):
        """Fetch and save complete dataset"""
        try:
            # Create diverse dataset
            tracks_data = self.create_diverse_dataset(tracks_per_category)
            
            if not tracks_data:
                logger.error("No tracks data fetched")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(tracks_data)
            
            # Get artist IDs for genre fetching
            all_artist_ids = df['artist_id'].dropna().unique().tolist()
            
            logger.info(f"Fetching details for {len(all_artist_ids)} unique artists...")
            
            # Fetch artist details
            artist_details = self.fetch_artist_details(all_artist_ids)
            
            # Add artist popularity
            def get_artist_popularity(artist_id):
                return artist_details.get(artist_id, {}).get('popularity', 50)
            
            df['artist_popularity'] = df['artist_id'].apply(get_artist_popularity)
            
            # Save tracks data
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            tracks_path = os.path.join(RAW_DATA_DIR, filename)
            df.to_csv(tracks_path, index=False)
            
            # Save artist genres
            artist_genres = []
            for artist_id, details in artist_details.items():
                artist_genres.append({
                    'artist_id': artist_id,
                    'artist': details['name'],
                    'genres': ', '.join(details['genres']),
                    'artist_popularity': details['popularity'],
                    'artist_followers': details['followers']
                })
            
            if artist_genres:
                genres_df = pd.DataFrame(artist_genres)
                genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
                genres_df.to_csv(genres_path, index=False)
                logger.info(f"Saved {len(artist_genres)} artist genres")
            
            # Log summary
            logger.info(f"Dataset saved successfully:")
            logger.info(f"  Tracks: {len(df)}")
            logger.info(f"  ISRC coverage: {(df['isrc'] != '').sum()}/{len(df)} ({(df['isrc'] != '').sum()/len(df)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching and saving dataset: {e}")
            return False


def fetch_initial_dataset(tracks_per_query=50):
    """Convenience function to fetch initial dataset"""
    try:
        fetcher = SpotifyDataFetcher()
        return fetcher.fetch_and_save_dataset(tracks_per_category=tracks_per_query)
    except Exception as e:
        logger.error(f"Error in fetch_initial_dataset: {e}")
        return False


if __name__ == "__main__":
    success = fetch_initial_dataset(tracks_per_query=100)
    if success:
        print("✅ Dataset fetched successfully!")
    else:
        print("❌ Failed to fetch dataset")
