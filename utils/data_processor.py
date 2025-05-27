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
                self.tracks_df = pd.read_csv(tracks_file)
                logger.info(f"Loaded tracks data: {len(self.tracks_df)} tracks")
                logger.info(f"Columns: {list(self.tracks_df.columns)}")
            else:
                logger.error(f"Tracks file not found: {tracks_file}")
                return False

            # Load artist genres
            genres_file = os.path.join(RAW_DATA_DIR, "artist_genres.csv")
            if os.path.exists(genres_file):
                self.artist_genres_df = pd.read_csv(genres_file)
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

    def create_language_features(self):
        """✅ ISRC-based cultural intelligence ONLY - NO TEXT DETECTION"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create language features")
            return False
        
        logger.info("Creating ISRC-based cultural intelligence (NO TEXT DETECTION)...")
        
        # ✅ ONLY ISRC COUNTRY EXTRACTION - NO FALLBACK
        if 'isrc' in self.tracks_df.columns:
            # Extract country code (first 2 chars of ISRC)
            self.tracks_df['isrc_country'] = self.tracks_df['isrc'].str[:2].fillna('XX')
            
            # ✅ SIMPLE country to culture mapping
            country_culture_map = {
                'US': 'western', 'GB': 'western', 'CA': 'western', 'AU': 'western',
                'KR': 'korean', 'JP': 'japanese', 
                'CN': 'chinese', 'TW': 'chinese', 'HK': 'chinese',
                'VN': 'vietnamese', 
                'MX': 'spanish', 'ES': 'spanish', 'BR': 'spanish', 'AR': 'spanish'
            }
            
            self.tracks_df['music_culture'] = self.tracks_df['isrc_country'].map(country_culture_map).fillna('other')
            
            # Create binary cultural features
            cultures = ['vietnamese', 'korean', 'japanese', 'chinese', 'western', 'spanish']
            for culture in cultures:
                self.tracks_df[f'is_{culture}'] = (self.tracks_df['music_culture'] == culture).astype(int)
            
            # Major label detection
            major_registrants = ['UMG', 'SME', 'WEA', 'SON', 'CAP', 'COL', 'ATL', 'RCA']
            self.tracks_df['isrc_registrant'] = self.tracks_df['isrc'].str[2:5].fillna('IND')
            self.tracks_df['is_major_label'] = self.tracks_df['isrc_registrant'].isin(major_registrants).astype(int)
            
            logger.info(f"ISRC-based culture distribution: {self.tracks_df['music_culture'].value_counts().to_dict()}")
            
        else:
            logger.warning("No ISRC column found")
            self.tracks_df['music_culture'] = 'other'
            self.tracks_df['isrc_country'] = 'XX'
            for culture in ['vietnamese', 'korean', 'japanese', 'chinese', 'western', 'spanish']:
                self.tracks_df[f'is_{culture}'] = 0
            self.tracks_df['is_major_label'] = 0
        
        # ✅ MARKET PENETRATION
        if 'available_markets' in self.tracks_df.columns:
            def count_markets(markets_str):
                if pd.isna(markets_str) or markets_str == '':
                    return 0
                return len(markets_str.split('|'))
            
            self.tracks_df['markets_count'] = self.tracks_df['available_markets'].apply(count_markets)
            self.tracks_df['market_penetration'] = (self.tracks_df['markets_count'] / 170).clip(0, 1)
            
            self.tracks_df['is_global_release'] = (self.tracks_df['markets_count'] > 100).astype(int)
            self.tracks_df['is_regional_release'] = ((self.tracks_df['markets_count'] > 20) & (self.tracks_df['markets_count'] <= 100)).astype(int)
            self.tracks_df['is_local_release'] = (self.tracks_df['markets_count'] <= 20).astype(int)
            
        else:
            self.tracks_df['markets_count'] = 1
            self.tracks_df['market_penetration'] = 0.1
            self.tracks_df['is_global_release'] = 0
            self.tracks_df['is_regional_release'] = 1
            self.tracks_df['is_local_release'] = 0
        
        # ✅ CULTURAL CONFIDENCE (based only on ISRC)
        def calculate_cultural_confidence(row):
            if row.get('isrc', '').strip() and row.get('isrc_country', 'XX') != 'XX':
                return 1.0  # Perfect confidence with ISRC
            else:
                return 0.1  # Low confidence without ISRC
        
        self.tracks_df['cultural_confidence'] = self.tracks_df.apply(calculate_cultural_confidence, axis=1)
        
        culture_dist = self.tracks_df['music_culture'].value_counts()
        confidence_avg = self.tracks_df['cultural_confidence'].mean()
        
        logger.info("✅ ISRC-based cultural intelligence created!")
        logger.info(f"  Culture distribution: {dict(culture_dist)}")
        logger.info(f"  Cultural confidence: {confidence_avg:.3f}")
        
        return True

    def merge_artist_genres(self):
        """Merge artist genres with tracks data"""
        if self.tracks_df is None:
            return False

        if self.artist_genres_df is None:
            logger.warning("No artist genres data")
            return True

        try:
            before_count = len(self.tracks_df)
            self.tracks_df = self.tracks_df.merge(
                self.artist_genres_df[['artist_id', 'genres', 'artist_followers']], 
                on='artist_id', 
                how='left'
            )

            genres_matched = self.tracks_df['genres'].notna().sum()
            logger.info(f"Merged artist genres: {genres_matched}/{len(self.tracks_df)} tracks")
            
            return True

        except Exception as e:
            logger.error(f"Error merging artist genres: {e}")
            return False

    def create_genre_features(self):
        """Create genre features"""
        if self.tracks_df is None:
            return False

        try:
            if 'genres' not in self.tracks_df.columns or self.tracks_df['genres'].isna().all():
                self._create_fallback_genre_features()
                return True

            # Extract genres
            all_genres = set()
            for genres_str in self.tracks_df['genres'].dropna():
                if isinstance(genres_str, str) and genres_str.strip():
                    genres = [g.strip().lower().replace(' ', '_') for g in genres_str.split(',')]
                    all_genres.update([g for g in genres if g and len(g) > 1])

            # Filter common genres
            common_genres = ['pop', 'rock', 'hip_hop', 'electronic', 'folk', 'jazz', 'ballad', 'dance']
            target_genres = [g for g in all_genres if any(cg in g for cg in common_genres)]
            
            logger.info(f"Creating {len(target_genres)} genre features")

            for genre in target_genres:
                genre_col = f'genre_{genre}'
                self.tracks_df[genre_col] = 0
                
                genre_mask = self.tracks_df['genres'].str.contains(genre, case=False, na=False)
                self.tracks_df.loc[genre_mask, genre_col] = 1

            return True

        except Exception as e:
            logger.error(f"Error creating genre features: {e}")
            self._create_fallback_genre_features()
            return False

    def _create_fallback_genre_features(self):
        """Create basic genre features"""
        fallback_genres = ['pop', 'rock', 'hip_hop', 'electronic', 'ballad']
        
        for genre in fallback_genres:
            self.tracks_df[f'genre_{genre}'] = 0
        
        # Default to pop
        self.tracks_df['genre_pop'] = 1
        
        logger.info("Fallback genre features created")

    def clean_tracks_data(self):
        """Clean tracks data"""
        if self.tracks_df is None or self.tracks_df.empty:
            return False
        
        initial_count = len(self.tracks_df)
        
        # Remove duplicates
        if 'name' in self.tracks_df.columns and 'artist' in self.tracks_df.columns:
            before_dedup = len(self.tracks_df)
            self.tracks_df = self.tracks_df.drop_duplicates(subset=['name', 'artist'], keep='first')
            after_dedup = len(self.tracks_df)
            logger.info(f"Removed {before_dedup - after_dedup} duplicates")
        
        # Fill missing values
        self.tracks_df['popularity'] = self.tracks_df['popularity'].fillna(50)
        self.tracks_df['artist_popularity'] = self.tracks_df['artist_popularity'].fillna(50)
        self.tracks_df['duration_ms'] = self.tracks_df['duration_ms'].fillna(200000)
        
        if 'release_year' not in self.tracks_df.columns:
            self.tracks_df['release_year'] = 2020
        
        clean_count = len(self.tracks_df)
        logger.info(f"Cleaned: {initial_count} -> {clean_count} tracks")
        
        return True

    def normalize_features(self):
        """Normalize features"""
        if self.tracks_df is None:
            return False

        try:
            numerical_features = ['popularity', 'duration_ms', 'artist_popularity', 'track_age', 'markets_count']
            
            scaler = MinMaxScaler()
            
            for feature in numerical_features:
                if feature in self.tracks_df.columns:
                    feature_data = self.tracks_df[feature].fillna(self.tracks_df[feature].median())
                    normalized_values = scaler.fit_transform(feature_data.values.reshape(-1, 1))
                    self.tracks_df[f'{feature}_norm'] = normalized_values.flatten()

            logger.info(f"Normalized {len(numerical_features)} features")
            return True

        except Exception as e:
            logger.error(f"Error normalizing: {e}")
            return False

    def save_processed_data(self):
        """Save processed data"""
        if self.tracks_df is None:
            return False

        try:
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            
            processed_file = os.path.join(PROCESSED_DATA_DIR, "processed_tracks.csv")
            self.tracks_df.to_csv(processed_file, index=False)
            
            feature_summary = {
                'total_tracks': len(self.tracks_df),
                'total_features': len(self.tracks_df.columns),
                'isrc_coverage': (self.tracks_df['isrc'] != '').sum() / len(self.tracks_df),
                'cultural_confidence': self.tracks_df['cultural_confidence'].mean()
            }
            
            logger.info(f"✅ Saved to {processed_file}")
            logger.info(f"📊 Summary: {feature_summary}")
            
            return True

        except Exception as e:
            logger.error(f"Error saving: {e}")
            return False

    def process_all(self):
        """Complete processing pipeline"""
        logger.info("🚀 Starting data processing...")
        
        steps = [
            ("Loading data", self.load_data),
            ("Extracting release year", self.extract_release_year), 
            ("Creating ISRC cultural features", self.create_language_features),
            ("Merging artist genres", self.merge_artist_genres),
            ("Creating genre features", self.create_genre_features),
            ("Cleaning data", self.clean_tracks_data),
            ("Normalizing features", self.normalize_features),
            ("Saving processed data", self.save_processed_data)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"🔄 {step_name}...")
            if not step_func():
                logger.error(f"❌ Failed: {step_name}")
                return False
        
        logger.info("✅ Data processing completed!")
        return True


if __name__ == "__main__":
    processor = DataProcessor()
    success = processor.process_all()
    if success:
        print("✅ Data processing completed successfully!")
    else:
        print("❌ Data processing failed!")