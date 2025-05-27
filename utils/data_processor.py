import os
import logging
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CONTENT_FEATURES

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to process music data with ISRC-based cultural intelligence"""
    
    def __init__(self):
        """Initialize data processor"""
        self.tracks_df = None
        self.artist_genres_df = None

    def load_data(self):
        """Load all data files"""
        try:
            # Load tracks data
            tracks_file = os.path.join(RAW_DATA_DIR, "tracks.csv")
            if os.path.exists(tracks_file):
                self.tracks_df = pd.read_csv(tracks_file, encoding='utf-8')
                logger.info(f"Loaded tracks data: {len(self.tracks_df)} tracks")
                logger.info(f"Columns: {list(self.tracks_df.columns)}")
            else:
                logger.error(f"Tracks file not found: {tracks_file}")
                return False

            # Load artist genres
            genres_file = os.path.join(RAW_DATA_DIR, "artist_genres.csv")
            if os.path.exists(genres_file):
                self.artist_genres_df = pd.read_csv(genres_file, encoding='utf-8')
                logger.info(f"Loaded artist genres: {len(self.artist_genres_df)} artists")
            else:
                logger.warning("Artist genres file not found")

            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def extract_release_year(self):
        """Extract release year from release date"""
        if self.tracks_df is None:
            return False

        try:
            def extract_year(date_str):
                if pd.isna(date_str) or date_str == '':
                    return 2020
                
                year_match = re.search(r'(\d{4})', str(date_str))
                if year_match:
                    year = int(year_match.group(1))
                    if 1900 <= year <= 2025:
                        return year
                
                return 2020

            self.tracks_df['release_year'] = self.tracks_df['release_date'].apply(extract_year)
            
            current_year = datetime.now().year
            self.tracks_df['track_age'] = current_year - self.tracks_df['release_year']
            
            logger.info(f"Release year extracted - Range: {self.tracks_df['release_year'].min()}-{self.tracks_df['release_year'].max()}")
            return True

        except Exception as e:
            logger.error(f"Error extracting release year: {e}")
            self.tracks_df['release_year'] = 2020
            self.tracks_df['track_age'] = 4
            return False

    def extract_cultural_features(self):
        """Extract country and region information from ISRC codes"""
        if self.tracks_df is None:
            return False

        try:
            # Định nghĩa bản đồ khu vực từ mã quốc gia ISRC
            region_map = {
                # Khu vực Bắc Mỹ
                'US': 'north_america', 'CA': 'north_america',
                
                # Khu vực Châu Âu
                'GB': 'europe', 'DE': 'europe', 'FR': 'europe', 'IT': 'europe', 
                'ES': 'europe', 'NL': 'europe', 'SE': 'europe', 'NO': 'europe',
                'DK': 'europe', 'FI': 'europe', 'PT': 'europe', 'IE': 'europe',
                'CH': 'europe', 'AT': 'europe', 'BE': 'europe', 'GR': 'europe',
                
                # Khu vực Châu Á
                'JP': 'asia', 'KR': 'asia', 'CN': 'asia', 'HK': 'asia', 
                'TW': 'asia', 'VN': 'asia', 'TH': 'asia', 'MY': 'asia',
                'ID': 'asia', 'PH': 'asia', 'SG': 'asia', 'IN': 'asia',
                
                # Khu vực Mỹ Latinh
                'MX': 'latin_america', 'BR': 'latin_america', 'AR': 'latin_america',
                'CO': 'latin_america', 'CL': 'latin_america', 'PE': 'latin_america',
                
                # Khu vực Châu Đại Dương
                'AU': 'oceania', 'NZ': 'oceania'
            }
            
            # Trích xuất mã quốc gia từ ISRC (2 ký tự đầu)
            self.tracks_df['isrc_country'] = self.tracks_df['isrc'].str[:2].fillna('XX')
            
            # Ánh xạ mã quốc gia sang khu vực
            def map_to_region(country_code):
                return region_map.get(country_code, 'other')
                
            self.tracks_df['region'] = self.tracks_df['isrc_country'].apply(map_to_region)
            
            # ✅ THÊM: Tạo music_culture từ ISRC
            def determine_music_culture(country_code):
                """Determine music culture from ISRC country code"""
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
                return cultural_mapping.get(country_code, 'other')
            
            # Tạo music_culture từ isrc_country
            self.tracks_df['music_culture'] = self.tracks_df['isrc_country'].apply(determine_music_culture)
            
            # Tạo cultural_confidence dựa trên ISRC availability
            self.tracks_df['cultural_confidence'] = np.where(
                self.tracks_df['isrc'].notna() & (self.tracks_df['isrc'] != ''), 
                0.9,  # High confidence với ISRC
                0.3   # Low confidence không có ISRC
            )
            
            # Tạo các cột binary cho từng văn hóa âm nhạc
            cultures = ['vietnamese', 'korean', 'japanese', 'chinese', 'western', 'spanish', 'brazilian', 'indian', 'thai']
            for culture in cultures:
                col_name = f'is_{culture}'
                self.tracks_df[col_name] = (self.tracks_df['music_culture'] == culture).astype(int)
            
            # Log cultural distribution
            culture_dist = self.tracks_df['music_culture'].value_counts()
            logger.info(f"Music culture distribution: {dict(culture_dist)}")
            
            # Tạo các cột binary cho từng khu vực
            regions = ['asia', 'europe', 'north_america', 'latin_america', 'oceania']
            for region in regions:
                col_name = f'is_{region}'
                self.tracks_df[col_name] = (self.tracks_df['region'] == region).astype(int)
            
            # Tạo các cột phân loại phát hành
            self.tracks_df['is_global_release'] = (self.tracks_df['markets_count'] > 100).astype(int)
            self.tracks_df['is_regional_release'] = ((self.tracks_df['markets_count'] > 20) & 
                                                    (self.tracks_df['markets_count'] <= 100)).astype(int)
            self.tracks_df['is_local_release'] = (self.tracks_df['markets_count'] <= 20).astype(int)
            
            # Tính market_penetration (tỷ lệ thị trường so với tổng số thị trường)
            max_markets = 200  # Ước tính tổng số thị trường Spotify
            self.tracks_df['market_penetration'] = self.tracks_df['markets_count'] / max_markets
            
            logger.info(f"Extracted cultural features from {len(self.tracks_df)} tracks")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting cultural features: {e}")
            return False

    def merge_artist_genres(self):
        """Merge artist genres with tracks data"""
        if self.tracks_df is None:
            return False

        if self.artist_genres_df is None:
            logger.warning("No artist genres data")
            return True

        try:
            # Merge artist data
            merge_columns = ['artist_id']
            available_columns = [col for col in ['genres', 'artist_followers'] if col in self.artist_genres_df.columns]
            merge_columns.extend(available_columns)
            
            before_count = len(self.tracks_df)
            self.tracks_df = self.tracks_df.merge(
                self.artist_genres_df[merge_columns], 
                on='artist_id', 
                how='left'
            )

            genres_matched = self.tracks_df['genres'].notna().sum() if 'genres' in self.tracks_df.columns else 0
            logger.info(f"Merged artist genres: {genres_matched}/{len(self.tracks_df)} tracks")
            
            return True

        except Exception as e:
            logger.error(f"Error merging artist genres: {e}")
            return False

    def create_genre_features(self):
        """Create genre features with optimized performance"""
        if self.tracks_df is None:
            return False

        try:
            if 'genres' not in self.tracks_df.columns or self.tracks_df['genres'].isna().all():
                self._create_fallback_genre_features()
                return True

            # Extract all unique genres
            all_genres = set()
            for genres_str in self.tracks_df['genres'].dropna():
                if isinstance(genres_str, str) and genres_str.strip():
                    # Clean and split genres
                    genres = [g.strip().lower().replace(' ', '_').replace('-', '_') for g in genres_str.split(',')]
                    all_genres.update([g for g in genres if g and len(g) > 1])

            # Filter for meaningful genres
            common_genre_keywords = [
                'pop', 'rock', 'hip_hop', 'rap', 'electronic', 'dance', 'edm', 
                'folk', 'jazz', 'blues', 'country', 'classical', 'ballad', 
                'indie', 'alternative', 'metal', 'punk', 'reggae', 'soul',
                'r&b', 'funk', 'disco', 'house', 'techno', 'trance',
                'acoustic', 'vocal', 'instrumental'
            ]
            
            target_genres = []
            for genre in all_genres:
                if any(keyword in genre for keyword in common_genre_keywords):
                    target_genres.append(genre)
            
            # Limit to most common genres to avoid too many features
            genre_counts = {}
            for genre in target_genres:
                count = self.tracks_df['genres'].str.contains(genre, case=False, na=False).sum()
                if count >= 5:  # Only include genres with at least 5 tracks
                    genre_counts[genre] = count
            
            # Sort by frequency and take top genres
            sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
            final_genres = [genre for genre, count in sorted_genres[:100]]  # Limit to top 100 genres
            
            logger.info(f"Creating {len(final_genres)} genre features")

            # ✅ OPTIMIZED: Create all genre columns at once using pd.concat
            genre_data = {}
            
            for genre in final_genres:
                genre_col = f'genre_{genre}'
                # Create boolean mask for this genre
                genre_mask = self.tracks_df['genres'].str.contains(genre, case=False, na=False)
                genre_data[genre_col] = genre_mask.astype(int)
            
            # ✅ Create DataFrame from genre data and concatenate
            if genre_data:
                genre_df = pd.DataFrame(genre_data, index=self.tracks_df.index)
                self.tracks_df = pd.concat([self.tracks_df, genre_df], axis=1)
            
            logger.info(f"✅ Genre features created efficiently: {len(final_genres)} features")
            return True

        except Exception as e:
            logger.error(f"Error creating genre features: {e}")
            self._create_fallback_genre_features()
            return False

    def _create_fallback_genre_features(self):
        """Create basic genre features efficiently"""
        fallback_genres = ['pop', 'rock', 'hip_hop', 'electronic', 'ballad', 'dance', 'indie', 'folk']
        
        # ✅ OPTIMIZED: Create all fallback columns at once
        fallback_data = {}
        for genre in fallback_genres:
            fallback_data[f'genre_{genre}'] = 0
        
        # Default assignment based on cultural features if available
        if 'music_culture' in self.tracks_df.columns:
            # Assign genres based on cultural intelligence
            fallback_data['genre_pop'] = 1  # Default to pop
            
            # Cultural-specific genre preferences
            korean_mask = self.tracks_df.get('is_korean', pd.Series([0] * len(self.tracks_df))) == 1
            if korean_mask.any():
                fallback_data['genre_pop'] = np.where(korean_mask, 1, fallback_data['genre_pop'])
                fallback_data['genre_dance'] = np.where(korean_mask, 1, 0)
            
            electronic_cultures = ['japanese', 'western']
            for culture in electronic_cultures:
                culture_mask = self.tracks_df.get(f'is_{culture}', pd.Series([0] * len(self.tracks_df))) == 1
                if culture_mask.any():
                    fallback_data['genre_electronic'] = np.where(culture_mask, 1, fallback_data.get('genre_electronic', 0))
        else:
            # Simple fallback
            fallback_data['genre_pop'] = 1
        
        fallback_df = pd.DataFrame(fallback_data, index=self.tracks_df.index)
        self.tracks_df = pd.concat([self.tracks_df, fallback_df], axis=1)
        
        logger.info("✅ Fallback genre features created efficiently")

    def clean_tracks_data(self):
        """Clean tracks data"""
        if self.tracks_df is None or self.tracks_df.empty:
            return False
        
        initial_count = len(self.tracks_df)
        
        try:
            # Remove duplicates based on track ID (most reliable)
            if 'id' in self.tracks_df.columns:
                before_dedup = len(self.tracks_df)
                self.tracks_df = self.tracks_df.drop_duplicates(subset=['id'], keep='first')
                after_dedup = len(self.tracks_df)
                logger.info(f"Removed {before_dedup - after_dedup} duplicates by ID")
            
            # Secondary deduplication by name + artist
            elif 'name' in self.tracks_df.columns and 'artist' in self.tracks_df.columns:
                before_dedup = len(self.tracks_df)
                self.tracks_df = self.tracks_df.drop_duplicates(subset=['name', 'artist'], keep='first')
                after_dedup = len(self.tracks_df)
                logger.info(f"Removed {before_dedup - after_dedup} duplicates by name+artist")
            
            # Fill missing values with reasonable defaults
            numeric_columns = {
                'popularity': 50,
                'artist_popularity': 50,
                'duration_ms': 200000,  # ~3:20 minutes
                'release_year': 2020,
                'track_age': 4,
                'markets_count': 1
            }
            
            for col, default_value in numeric_columns.items():
                if col in self.tracks_df.columns:
                    self.tracks_df[col] = self.tracks_df[col].fillna(default_value)
            
            # Fill string columns
            string_columns = {
                'isrc': '',
                'available_markets': '',
                'genres': '',
                'album': 'Unknown Album'
            }
            
            for col, default_value in string_columns.items():
                if col in self.tracks_df.columns:
                    self.tracks_df[col] = self.tracks_df[col].fillna(default_value)
            
            # Remove rows with missing essential data
            essential_columns = ['id', 'name', 'artist']
            for col in essential_columns:
                if col in self.tracks_df.columns:
                    self.tracks_df = self.tracks_df[self.tracks_df[col].notna()]
                    self.tracks_df = self.tracks_df[self.tracks_df[col] != '']
            
            clean_count = len(self.tracks_df)
            logger.info(f"Cleaned: {initial_count} -> {clean_count} tracks")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return False

    def normalize_features(self):
        """Normalize features with optimized performance"""
        if self.tracks_df is None:
            return False

        try:
            # Define features to normalize
            numerical_features = [
                'popularity', 'duration_ms', 'artist_popularity', 
                'track_age', 'markets_count'
            ]
            
            # Filter for existing columns
            existing_features = [feat for feat in numerical_features if feat in self.tracks_df.columns]
            
            if not existing_features:
                logger.warning("No numerical features found to normalize")
                return True

            # ✅ OPTIMIZED: Create all normalized columns at once
            normalized_data = {}
            scaler = MinMaxScaler()
            
            for feature in existing_features:
                # Handle missing values
                feature_data = self.tracks_df[feature].fillna(self.tracks_df[feature].median())
                
                # Normalize
                normalized_values = scaler.fit_transform(feature_data.values.reshape(-1, 1))
                normalized_data[f'{feature}_norm'] = normalized_values.flatten()

            # ✅ Create DataFrame and concatenate once
            if normalized_data:
                normalized_df = pd.DataFrame(normalized_data, index=self.tracks_df.index)
                self.tracks_df = pd.concat([self.tracks_df, normalized_df], axis=1)
            
            # Bật chế độ Copy-on-Write để tối ưu hiệu suất
            pd.options.mode.copy_on_write = True

            logger.info(f"✅ Normalized {len(existing_features)} features efficiently")
            return True

        except Exception as e:
            logger.error(f"Error normalizing: {e}")
            return False

    def save_processed_data(self):
        """Save processed data with comprehensive summary"""
        if self.tracks_df is None:
            return False

        try:
            # Ensure output directory exists
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            
            # Save main processed file
            processed_file = os.path.join(PROCESSED_DATA_DIR, "processed_tracks.csv")
            self.tracks_df.to_csv(processed_file, index=False, encoding='utf-8')
            
            # Calculate comprehensive feature summary
            feature_summary = {
                'total_tracks': len(self.tracks_df),
                'total_features': len(self.tracks_df.columns),
                'isrc_coverage': float((self.tracks_df['isrc'] != '').sum() / len(self.tracks_df)) if 'isrc' in self.tracks_df.columns else 0.0,
                'cultural_confidence': float(self.tracks_df['cultural_confidence'].mean()) if 'cultural_confidence' in self.tracks_df.columns else 0.0
            }
            
            # Additional analytics
            if 'music_culture' in self.tracks_df.columns:
                culture_dist = self.tracks_df['music_culture'].value_counts()
                feature_summary['culture_distribution'] = dict(culture_dist)
            
            # Count genre features
            genre_features = [col for col in self.tracks_df.columns if col.startswith('genre_')]
            feature_summary['genre_features_count'] = len(genre_features)
            
            # Count cultural features
            cultural_features = [col for col in self.tracks_df.columns if col.startswith('is_')]
            feature_summary['cultural_features_count'] = len(cultural_features)
            
            # Count normalized features
            normalized_features = [col for col in self.tracks_df.columns if col.endswith('_norm')]
            feature_summary['normalized_features_count'] = len(normalized_features)
            
            logger.info(f"✅ Saved to {processed_file}")
            logger.info(f"📊 Summary: {feature_summary}")
            
            return True

        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False

    def process_all(self):
        """Complete processing pipeline with enhanced error handling"""
        logger.info("🚀 Starting data processing...")
        
        # Define processing steps
        steps = [
            ("🔄 Loading data", self.load_data),
            ("🔄 Extracting release year", self.extract_release_year), 
            ("🔄 Extracting country and region from ISRC", self.extract_cultural_features),
            ("🔄 Merging artist genres", self.merge_artist_genres),
            ("🔄 Creating genre features", self.create_genre_features),
            ("🔄 Cleaning data", self.clean_tracks_data),
            ("🔄 Normalizing features", self.normalize_features),
            ("🔄 Saving processed data", self.save_processed_data)
        ]
        
        successful_steps = 0
        
        for step_name, step_func in steps:
            logger.info(step_name)
            try:
                if step_func():
                    successful_steps += 1
                    logger.info(f"✅ Completed: {step_name.replace('🔄 ', '')}")
                else:
                    logger.error(f"❌ Failed: {step_name.replace('🔄 ', '')}")
                    # Continue with other steps for robustness
            except Exception as e:
                logger.error(f"❌ Exception in {step_name}: {e}")
        
        # Final assessment
        success_rate = successful_steps / len(steps)
        
        if success_rate >= 0.75:  # 75% success rate
            logger.info(f"✅ Data processing completed! ({successful_steps}/{len(steps)} steps successful)")
            
            # Final data quality check
            if self.tracks_df is not None:
                # Kiểm tra tỷ lệ dữ liệu quốc gia và khu vực
                if 'isrc_country' in self.tracks_df.columns:
                    valid_country = (self.tracks_df['isrc_country'] != 'XX').mean() * 100
                    logger.info(f"📊 Valid country data: {valid_country:.1f}%")
                
                if 'region' in self.tracks_df.columns:
                    valid_region = (self.tracks_df['region'] != 'other').mean() * 100
                    logger.info(f"📊 Valid region data: {valid_region:.1f}%")
                
                logger.info(f"📈 Final dataset: {len(self.tracks_df):,} tracks, {len(self.tracks_df.columns)} features")
                return True
            else:
                logger.error("❌ No final dataset available")
                return False
        else:
            logger.error(f"❌ Data processing failed! ({successful_steps}/{len(steps)} steps successful)")
            return False

    def enrich_with_spotify_api(self):
        """Enrich existing data với Spotify API"""
        if self.tracks_df is None:
            logger.error("No data loaded for enrichment")
            return False
        
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            from config.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
            import time
            from tqdm import tqdm
            
            # Initialize Spotify client
            sp = spotipy.Spotify(
                client_credentials_manager=SpotifyClientCredentials(
                    client_id=SPOTIFY_CLIENT_ID,
                    client_secret=SPOTIFY_CLIENT_SECRET
                )
            )
            
            # Tìm tracks thiếu ISRC
            missing_isrc = self.tracks_df[
                (self.tracks_df['isrc'].isna()) | (self.tracks_df['isrc'] == '')
            ].copy()
            
            logger.info(f"Found {len(missing_isrc)} tracks with missing ISRC")
            
            if len(missing_isrc) == 0:
                logger.info("No tracks need ISRC enrichment")
                return True
            
            # Giới hạn để tránh rate limit
            max_enrich = min(len(missing_isrc), 1000)
            missing_isrc = missing_isrc.head(max_enrich)
            
            enriched_count = 0
            batch_size = 50
            
            for i in tqdm(range(0, len(missing_isrc), batch_size), desc="Enriching with API"):
                batch = missing_isrc.iloc[i:i+batch_size]
                track_ids = batch['id'].tolist()
                
                try:
                    # Get tracks data from API
                    tracks_data = sp.tracks(track_ids)['tracks']
                    
                    for j, track in enumerate(tracks_data):
                        if track:
                            idx = batch.iloc[j].name
                            
                            # Update ISRC
                            new_isrc = track.get('external_ids', {}).get('isrc', '')
                            if new_isrc:
                                self.tracks_df.at[idx, 'isrc'] = new_isrc
                                enriched_count += 1
                            
                            # Update markets
                            new_markets = track.get('available_markets', [])
                            if new_markets:
                                self.tracks_df.at[idx, 'available_markets'] = '|'.join(new_markets)
                                self.tracks_df.at[idx, 'markets_count'] = len(new_markets)
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error enriching batch {i}: {e}")
                    continue
            
            logger.info(f"✅ Enriched {enriched_count} tracks with ISRC data")
            
            # Re-extract cultural features if ISRC updated
            if enriched_count > 0:
                logger.info("Reprocessing cultural features...")
                self.extract_cultural_features()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in API enrichment: {e}")
            return False


# ✅ STANDALONE EXECUTION
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run processor
    processor = DataProcessor()
    success = processor.process_all()
    
    if success:
        print("\n🎉 ✅ Data processing completed successfully!")
        print("🚀 Ready for model training!")
    else:
        print("\n❌ Data processing failed!")
        print("🔧 Check logs for details and retry")
