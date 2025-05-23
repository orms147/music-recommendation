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

        # Use provided credentials or environment variables
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

    def _extract_track_data(self, track):
        """Extract relevant metadata from a track object"""
        try:
            # Thêm nhiều thông tin hơn từ metadata có sẵn
            album_type = track['album']['album_type'] if 'album' in track else None
            total_tracks = track['album']['total_tracks'] if 'album' in track else None
            
            # Lấy thông tin tất cả nghệ sĩ, không chỉ nghệ sĩ đầu tiên
            artists = ", ".join([artist['name'] for artist in track['artists']]) if track['artists'] else "Unknown"
            
            # Trích xuất năm phát hành
            release_date = track['album']['release_date'] if 'album' in track else None
            release_year = None
            if release_date:
                try:
                    # Xử lý cả ngày đầy đủ và chỉ năm
                    if len(release_date) >= 4:
                        release_year = int(release_date[:4])
                except:
                    pass
            
            return {
                'id': track['id'],
                'name': track['name'],
                'artist': artists,
                'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                'album': track['album']['name'] if 'album' in track else None,
                'album_type': album_type,
                'total_tracks': total_tracks,
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'explicit': int(track['explicit']),
                'release_date': release_date,
                'release_year': release_year,
                'track_number': track.get('track_number'),
                'disc_number': track.get('disc_number')
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
            # Spotify API chỉ cho phép offset + limit <= 1000
            tracks_per_query = min(tracks_per_query, 1000)

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
    
    def _reinitialize_client(self):
        """Reinitialize Spotify client to refresh token"""
        try:
            logger.info("Reinitializing Spotify client...")
            auth_manager = spotipy.oauth2.SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
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
                            'genres': '|'.join(artist['genres']) if artist['genres'] else '',
                            'artist_popularity': artist.get('popularity', 0),
                            'artist_followers': artist.get('followers', {}).get('total', 0) if 'followers' in artist else 0,
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

    def enrich_track_data(self, tracks_df, save_path=None):
        """Tự động tạo đặc trưng bổ sung từ dữ liệu bài hát hiện có"""
        if tracks_df is None or tracks_df.empty:
            logger.warning("No tracks data to enrich")
            return tracks_df
            
        enriched_df = tracks_df.copy()
        
        # 1. Trích xuất năm phát hành nếu chưa có
        if 'release_year' not in enriched_df.columns and 'release_date' in enriched_df.columns:
            enriched_df['release_year'] = enriched_df['release_date'].apply(
                lambda x: int(str(x)[:4]) if pd.notna(x) and len(str(x)) >= 4 else None
            )
        
        # 2. Trích xuất thập kỷ
        if 'release_year' in enriched_df.columns:
            enriched_df['decade'] = enriched_df['release_year'].apply(
                lambda x: (x // 10) * 10 if pd.notna(x) else None
            )
        
        # 3. Tính độ dài bài hát (phút)
        if 'duration_ms' in enriched_df.columns:
            enriched_df['duration_min'] = enriched_df['duration_ms'] / 60000
            
            # Phân loại độ dài bài hát
            enriched_df['duration_category'] = pd.cut(
                enriched_df['duration_min'], 
                bins=[0, 2, 4, 6, 10, 100],
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            )
        
        # 4. Phân loại độ phổ biến
        if 'popularity' in enriched_df.columns:
            enriched_df['popularity_category'] = pd.cut(
                enriched_df['popularity'],
                bins=[0, 20, 40, 60, 80, 100],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        # 5. Phát hiện ngôn ngữ (dựa trên các ký tự đặc biệt)
        enriched_df['has_vietnamese'] = enriched_df['name'].str.contains(
            '|'.join(['Vietnam', 'việt', 'Đ', 'Ư', 'Ơ', 'ă', 'â', 'ê', 'ô', 'ơ', 'ư']), 
            case=False, regex=True, na=False
        ).astype(int)
        
        if save_path:
            enriched_df.to_csv(save_path, index=False)
            logger.info(f"Saved {len(enriched_df)} enriched tracks to {save_path}")
            
        return enriched_df

# Hàm cấp cao để lấy dữ liệu ban đầu
def fetch_initial_dataset(tracks_per_query=DEFAULT_TRACKS_PER_QUERY):
    """Fetch an initial dataset focusing on metadata only"""
    # Create fetcher
    fetcher = SpotifyDataFetcher()
    
    # Define diverse search queries
    queries = [
    # 🎵 Grammy & Billboard (Top-quality music)
    'grammy winners 2023', 'grammy winners 2022', 'grammy winners 2021',
    'billboard hot 100 2024', 'billboard hot 100 2023', 'billboard hot 100 2022',
    'top global songs 2024', 'top global songs 2023',

    # 🇯🇵 Japan (J-Pop, Anime, Official)
    'j-pop top hits 2024', 'official japanese music 2024', 'j-pop 2023', 'japan top songs',

    # 🇻🇳 Vietnam (V-Pop)
    'v-pop top hits 2024', 'vietnamese chart songs 2023', 'vietnamese pop official',

    # 🇰🇷 Korea (K-Pop)
    'k-pop top hits 2024', 'korean top charts 2023', 'k-pop 2022', 'korean music official',

    # 🇺🇸 US-UK (Mainstream)
    'us top 100 songs 2024', 'uk top 100 songs 2024', 'us pop 2023', 'uk chart 2023',

    # 🇨🇳 China (C-Pop, Mandopop)
    'c-pop top songs 2024', 'mandopop 2023', 'chinese pop official', 'chinese music charts',

    # 🎤 Rap (mainstream, award-winning)
    'top rap songs 2024', 'rap billboard 2023', 'famous hip hop tracks', 'grammy rap songs',

    # 🎧 EDM (non-remix, mainstream)
    'top edm songs 2024', 'famous edm tracks', 'edm billboard 2023', 'edm festival anthems',

    # 🎷 Other high-quality genres
    'jazz standards', 'blues originals', 'country top 2023', 'indie top songs 2023'
    ]


    
    # 1. Fetch basic track data
    tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
    tracks_df = fetcher.fetch_tracks_by_search(queries, tracks_per_query=tracks_per_query, save_path=tracks_path)
    
    if tracks_df.empty:
        logger.error("Failed to fetch tracks data")
        return None
    
    # 2. Fetch artist genres
    if 'artist_id' in tracks_df.columns:
        artist_ids = tracks_df['artist_id'].dropna().unique().tolist()
        artist_genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
        fetcher.fetch_artist_genres(artist_ids, save_path=artist_genres_path)
    
    # 3. Enrich track data with additional features
    enriched_tracks_path = os.path.join(RAW_DATA_DIR, 'enriched_tracks.csv')
    fetcher.enrich_track_data(tracks_df, save_path=enriched_tracks_path)
    
    logger.info("Initial dataset fetched successfully")
    return tracks_df

def fetch_large_dataset(target_size=100000, batch_size=LARGE_DATASET_BATCH_SIZE, save_interval=LARGE_DATASET_SAVE_INTERVAL):
    """Thu thập một tập dữ liệu rất lớn theo lô"""
    fetcher = SpotifyDataFetcher()
    
    # Tạo các truy vấn đa dạng bao gồm nhiều thể loại và năm
    base_genres = ['pop', 'rock', 'hip hop', 'rap', 'electronic', 'dance', 'r&b', 
                  'indie', 'classical', 'jazz', 'country', 'folk', 'metal', 'blues',
                  'vietnamese', 'vpop', 'korean', 'k-pop', 'japanese', 'j-pop',
                  'latin', 'spanish', 'reggaeton', 'bollywood', 'afrobeats']
    
    years = range(2010, 2025)
    
    # Tạo kết hợp của thể loại và năm
    queries = []
    for genre in base_genres:
        for year in years:
            queries.append(f"{genre} {year}")
    
    # Thêm các truy vấn cụ thể bổ sung
    additional_queries = [
        'top hits', 'billboard hot 100', 'new releases', 'chart toppers',
        'grammy winners', 'viral hits', 'trending music', 'spotify viral',
        'tiktok songs', 'youtube hits', 'movie soundtrack', 'tv soundtrack'
    ]
    queries.extend(additional_queries)
    
    tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks_large.csv')
    artist_genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres_large.csv')
    
    all_tracks_df = pd.DataFrame()
    processed_artists = set()
    
    # Xử lý truy vấn theo lô cho đến khi đạt được kích thước mục tiêu
    import random
    random.shuffle(queries)  # Ngẫu nhiên hóa thứ tự truy vấn
    
    for i in range(0, len(queries), batch_size):
        if len(all_tracks_df) >= target_size:
            break
            
        batch_queries = queries[i:i+batch_size]
        logger.info(f"Đang xử lý lô {i//batch_size + 1}: {len(batch_queries)} truy vấn")
        
        # Lấy bài hát cho lô này
        batch_df = fetcher.fetch_tracks_by_search(
            batch_queries, 
            tracks_per_query=50,  # Giữ ở mức khiêm tốn mỗi truy vấn để tránh giới hạn tỷ lệ
            save_path=None  # Không lưu kết quả trung gian
        )
        
        # Kết hợp với dữ liệu hiện có
        if not batch_df.empty:
            all_tracks_df = pd.concat([all_tracks_df, batch_df]).drop_duplicates(subset=['id'])
            
            # Xử lý thể loại nghệ sĩ theo lô
            new_artist_ids = batch_df['artist_id'].dropna().unique().tolist()
            new_artist_ids = [aid for aid in new_artist_ids if aid not in processed_artists]
            
            if new_artist_ids:
                fetcher.fetch_artist_genres(new_artist_ids, save_path=None)
                processed_artists.update(new_artist_ids)
            
            # Lưu theo khoảng thời gian
            if len(all_tracks_df) % save_interval < 100:
                all_tracks_df.to_csv(tracks_path, index=False)
                logger.info(f"Tiến độ: {len(all_tracks_df)}/{target_size} bài hát đã thu thập")
                
                # Cho API nghỉ ngơi
                time.sleep(5)
                fetcher._reinitialize_client()  # Làm mới token
    
    # Lưu cuối cùng
    all_tracks_df.to_csv(tracks_path, index=False)
    
    # Lấy tất cả ID nghệ sĩ duy nhất
    all_artist_ids = all_tracks_df['artist_id'].dropna().unique().tolist()
    
    # Lấy và lưu tất cả thể loại nghệ sĩ
    all_genres_df = fetcher.fetch_artist_genres(all_artist_ids, save_path=artist_genres_path)
    
    # Làm phong phú và lưu tập dữ liệu cuối cùng
    enriched_tracks_path = os.path.join(RAW_DATA_DIR, 'enriched_tracks_large.csv')
    enriched_df = fetcher.enrich_track_data(all_tracks_df, save_path=enriched_tracks_path)
    
    logger.info(f"Thu thập tập dữ liệu lớn hoàn tất: {len(enriched_df)} bài hát")
    return enriched_df

if __name__ == "__main__":
    # Run if script is executed directly
    fetch_initial_dataset()