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
                logger.warning("Artist genres file not found - will create fallback genres")

            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def extract_release_year(self):
        """Extract release year from release date"""
        if self.tracks_df is None:
            logger.error("No tracks data to extract release year")
            return False

        try:
            def extract_year(date_str):
                if pd.isna(date_str) or date_str == '':
                    return 2020
                
                # Extract 4-digit year
                year_match = re.search(r'(\d{4})', str(date_str))
                if year_match:
                    year = int(year_match.group(1))
                    if 1900 <= year <= 2025:
                        return year
                
                return 2020

            self.tracks_df['release_year'] = self.tracks_df['release_date'].apply(extract_year)
            
            # Track age
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
        """✅ ISRC-based cultural intelligence for 2 models"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create language features")
            return False
        
        logger.info("Creating ISRC-based cultural intelligence for recommendation models...")
        
        # 1. ✅ ISRC COUNTRY EXTRACTION (Primary source)
        if 'isrc' in self.tracks_df.columns:
            # Extract country code (first 2 chars of ISRC)
            self.tracks_df['isrc_country'] = self.tracks_df['isrc'].str[:2].fillna('XX')
            
            # Enhanced country to culture mapping
            country_culture_map = {
                # Western countries
                'US': 'western', 'GB': 'western', 'CA': 'western', 'AU': 'western', 
                'DE': 'western', 'FR': 'western', 'NL': 'western', 'SE': 'western',
                
                # East Asian countries
                'KR': 'korean', 'JP': 'japanese', 
                'CN': 'chinese', 'TW': 'chinese', 'HK': 'chinese', 'SG': 'chinese',
                
                # Southeast Asian countries (Vietnamese cultural sphere)
                'VN': 'vietnamese', 'TH': 'vietnamese', 'PH': 'vietnamese',
                
                # Spanish/Latin countries
                'MX': 'spanish', 'ES': 'spanish', 'BR': 'spanish', 'AR': 'spanish', 
                'CO': 'spanish', 'CL': 'spanish', 'PE': 'spanish'
            }
            
            self.tracks_df['music_culture'] = self.tracks_df['isrc_country'].map(country_culture_map).fillna('other')
            
            # Create binary cultural features for models
            cultures = ['vietnamese', 'korean', 'japanese', 'chinese', 'western', 'spanish']
            for culture in cultures:
                self.tracks_df[f'is_{culture}'] = (self.tracks_df['music_culture'] == culture).astype(int)
            
            # Major label detection from ISRC registrant code
            major_registrants = [
                'UMG', 'SME', 'WEA', 'SON', 'CAP', 'COL', 'ATL', 'RCA', 
                'INT', 'WMG', 'EMI', 'DEF', 'REP', 'MAV', 'INS'
            ]
            self.tracks_df['isrc_registrant'] = self.tracks_df['isrc'].str[2:5].fillna('IND')
            self.tracks_df['is_major_label'] = self.tracks_df['isrc_registrant'].isin(major_registrants).astype(int)
            
            logger.info(f"ISRC-based culture distribution: {self.tracks_df['music_culture'].value_counts().to_dict()}")
            
        else:
            logger.warning("No ISRC column found, creating fallback features")
            self.tracks_df['music_culture'] = 'other'
            self.tracks_df['isrc_country'] = 'XX'
            for culture in ['vietnamese', 'korean', 'japanese', 'chinese', 'western', 'spanish']:
                self.tracks_df[f'is_{culture}'] = 0
            self.tracks_df['is_major_label'] = 0
        
        # 2. ✅ MARKET PENETRATION ANALYSIS (Secondary source)
        if 'available_markets' in self.tracks_df.columns:
            # Count markets
            def count_markets(markets_str):
                if pd.isna(markets_str) or markets_str == '':
                    return 0
                return len(markets_str.split('|'))
            
            self.tracks_df['markets_count'] = self.tracks_df['available_markets'].apply(count_markets)
            
            # Market penetration score (normalized to 0-1)
            max_possible_markets = 170  # Approximate total countries on Spotify
            self.tracks_df['market_penetration'] = (self.tracks_df['markets_count'] / max_possible_markets).clip(0, 1)
            
            # Release tier classification
            self.tracks_df['is_global_release'] = (self.tracks_df['markets_count'] > 100).astype(int)
            self.tracks_df['is_regional_release'] = ((self.tracks_df['markets_count'] > 20) & (self.tracks_df['markets_count'] <= 100)).astype(int)
            self.tracks_df['is_local_release'] = (self.tracks_df['markets_count'] <= 20).astype(int)
            
            logger.info(f"Market stats - Avg: {self.tracks_df['market_penetration'].mean():.3f}, Global releases: {self.tracks_df['is_global_release'].sum()}")
            
        else:
            logger.warning("No available_markets column found")
            self.tracks_df['markets_count'] = 1
            self.tracks_df['market_penetration'] = 0.1
            self.tracks_df['is_global_release'] = 0
            self.tracks_df['is_regional_release'] = 1
            self.tracks_df['is_local_release'] = 0
        
        # 3. ✅ TEXT-BASED FALLBACK (Tertiary source for missing ISRC)
        missing_culture = self.tracks_df['music_culture'] == 'other'
        if missing_culture.sum() > 0:
            logger.info(f"Applying text analysis fallback to {missing_culture.sum()} tracks")
            
            # Vietnamese detection
            vietnamese_patterns = {
                'chars': 'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ',
                'words': ['việt nam', 'vietnam', 'vpop', 'sài gòn', 'hà nội', 'đàm vĩnh hưng', 'mỹ tâm']
            }
            
            vietnamese_mask = self.tracks_df['name'].str.contains('|'.join(vietnamese_patterns['chars']), na=False)
            vietnamese_mask |= self.tracks_df['artist'].str.contains('|'.join(vietnamese_patterns['chars']), na=False)
            for word in vietnamese_patterns['words']:
                vietnamese_mask |= self.tracks_df['name'].str.contains(word, case=False, na=False)
                vietnamese_mask |= self.tracks_df['artist'].str.contains(word, case=False, na=False)
            
            self.tracks_df.loc[missing_culture & vietnamese_mask, 'is_vietnamese'] = 1
            self.tracks_df.loc[missing_culture & vietnamese_mask, 'music_culture'] = 'vietnamese'
            
            # Korean detection
            korean_patterns = {
                'chars': 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ가나다라마바사아자차카타파하',
                'words': ['kpop', 'k-pop', 'bts', 'blackpink', 'twice', 'itzy', 'aespa', 'stray kids', '방탄소년단', 'korea']
            }
            
            korean_mask = self.tracks_df['name'].str.contains('|'.join(korean_patterns['chars']), na=False)
            korean_mask |= self.tracks_df['artist'].str.contains('|'.join(korean_patterns['chars']), na=False)
            for word in korean_patterns['words']:
                korean_mask |= self.tracks_df['name'].str.contains(word, case=False, na=False)
                korean_mask |= self.tracks_df['artist'].str.contains(word, case=False, na=False)
            
            self.tracks_df.loc[missing_culture & korean_mask, 'is_korean'] = 1
            self.tracks_df.loc[missing_culture & korean_mask, 'music_culture'] = 'korean'
            
            # Japanese detection
            japanese_patterns = {
                'chars': 'あいうえおかきくけこアイウエオカキクケコひらがなカタカナ',
                'words': ['jpop', 'j-pop', 'anime', 'japan', 'tokyo', 'osaka']
            }
            
            japanese_mask = self.tracks_df['name'].str.contains('|'.join(japanese_patterns['chars']), na=False)
            japanese_mask |= self.tracks_df['artist'].str.contains('|'.join(japanese_patterns['chars']), na=False)
            for word in japanese_patterns['words']:
                japanese_mask |= self.tracks_df['name'].str.contains(word, case=False, na=False)
                japanese_mask |= self.tracks_df['artist'].str.contains(word, case=False, na=False)
            
            self.tracks_df.loc[missing_culture & japanese_mask, 'is_japanese'] = 1
            self.tracks_df.loc[missing_culture & japanese_mask, 'music_culture'] = 'japanese'
        
        # 4. ✅ CULTURAL CONFIDENCE SCORE (For model quality assessment)
        def calculate_cultural_confidence(row):
            confidence = 0.0
            
            # ISRC-based (highest confidence)
            if row.get('isrc', '').strip() and row.get('isrc_country', 'XX') != 'XX':
                confidence += 0.7
            
            # Market penetration consistency
            if row.get('markets_count', 0) > 20:
                confidence += 0.2
            
            # Text evidence
            if row.get('music_culture', 'other') != 'other':
                confidence += 0.1
            
            return min(1.0, confidence)
        
        self.tracks_df['cultural_confidence'] = self.tracks_df.apply(calculate_cultural_confidence, axis=1)
        
        # Log final cultural intelligence stats
        culture_dist = self.tracks_df['music_culture'].value_counts()
        confidence_avg = self.tracks_df['cultural_confidence'].mean()
        
        logger.info("✅ Cultural intelligence features created successfully!")
        logger.info(f"  Culture distribution: {dict(culture_dist)}")
        logger.info(f"  Average cultural confidence: {confidence_avg:.3f}")
        
        return True

    def merge_artist_genres(self):
        """Merge artist genres with tracks data"""
        if self.tracks_df is None:
            logger.error("No tracks data to merge genres")
            return False

        if self.artist_genres_df is None:
            logger.warning("No artist genres data - will create fallback genres")
            return True

        try:
            # Merge on artist_id
            before_count = len(self.tracks_df)
            self.tracks_df = self.tracks_df.merge(
                self.artist_genres_df[['artist_id', 'genres', 'artist_followers']], 
                on='artist_id', 
                how='left'
            )
            
            # Update artist_popularity if we have better data
            popularity_mask = self.tracks_df['artist_popularity'].isna() | (self.tracks_df['artist_popularity'] == 0)
            if 'artist_popularity' in self.artist_genres_df.columns:
                popularity_updates = self.artist_genres_df.set_index('artist_id')['artist_popularity']
                self.tracks_df.loc[popularity_mask, 'artist_popularity'] = self.tracks_df.loc[popularity_mask, 'artist_id'].map(popularity_updates)

            genres_matched = self.tracks_df['genres'].notna().sum()
            logger.info(f"Merged artist genres: {genres_matched}/{len(self.tracks_df)} tracks have genre data")
            
            return True

        except Exception as e:
            logger.error(f"Error merging artist genres: {e}")
            return False

    def create_genre_features(self):
        """Create genre features for recommendation models"""
        if self.tracks_df is None:
            logger.error("No tracks data to create genre features")
            return False

        try:
            # If no genres column, create fallback
            if 'genres' not in self.tracks_df.columns or self.tracks_df['genres'].isna().all():
                logger.warning("No genres data found, creating fallback genre features")
                self._create_fallback_genre_features()
                return True

            # Extract all unique genres
            all_genres = set()
            for genres_str in self.tracks_df['genres'].dropna():
                if isinstance(genres_str, str) and genres_str.strip():
                    genres = [g.strip().lower().replace(' ', '_').replace('-', '_') for g in genres_str.split(',')]
                    all_genres.update([g for g in genres if g and len(g) > 1])

            # Filter meaningful genres
            common_genres = ['pop', 'rock', 'hip_hop', 'electronic', 'folk', 'country', 'jazz', 'classical', 'r&b', 'soul']
            cultural_genres = ['kpop', 'jpop', 'cpop', 'vpop', 'latin', 'reggaeton', 'bollywood']
            mood_genres = ['ballad', 'chill', 'dance', 'party', 'sad', 'happy', 'energetic']
            
            target_genres = set(common_genres + cultural_genres + mood_genres)
            target_genres.update([g for g in all_genres if any(tg in g for tg in target_genres)])
            
            logger.info(f"Creating binary features for {len(target_genres)} genres")

            # Create binary genre features
            for genre in target_genres:
                genre_col = f'genre_{genre}'
                self.tracks_df[genre_col] = 0
                
                # Check if track has this genre
                genre_mask = self.tracks_df['genres'].str.contains(genre, case=False, na=False)
                self.tracks_df.loc[genre_mask, genre_col] = 1

            # Create compound genre features for better recommendations
            self._create_compound_genre_features()

            genre_cols = [col for col in self.tracks_df.columns if col.startswith('genre_')]
            logger.info(f"Created {len(genre_cols)} genre features")
            
            return True

        except Exception as e:
            logger.error(f"Error creating genre features: {e}")
            self._create_fallback_genre_features()
            return False

    def _create_fallback_genre_features(self):
        """Create fallback genre features when real genre data is unavailable"""
        logger.info("Creating fallback genre features for recommendation models")
        
        # Basic genre categories
        fallback_genres = ['pop', 'rock', 'hip_hop', 'electronic', 'ballad', 'folk', 'dance']
        
        for genre in fallback_genres:
            self.tracks_df[f'genre_{genre}'] = 0
        
        # Simple pattern-based assignment
        if 'name' in self.tracks_df.columns:
            # Pop patterns
            pop_patterns = ['love', 'heart', 'baby', 'girl', 'boy', 'feel', 'good', 'happy']
            for pattern in pop_patterns:
                mask = self.tracks_df['name'].str.contains(pattern, case=False, na=False)
                self.tracks_df.loc[mask, 'genre_pop'] = 1
            
            # Ballad patterns
            ballad_patterns = ['sad', 'cry', 'tear', 'miss', 'goodbye', 'alone', 'lonely']
            for pattern in ballad_patterns:
                mask = self.tracks_df['name'].str.contains(pattern, case=False, na=False)
                self.tracks_df.loc[mask, 'genre_ballad'] = 1
        
        # Default assignment for tracks without genres
        no_genre_mask = self.tracks_df[[f'genre_{g}' for g in fallback_genres]].sum(axis=1) == 0
        self.tracks_df.loc[no_genre_mask, 'genre_pop'] = 1
        
        logger.info("Fallback genre features created")

    def _create_compound_genre_features(self):
        """Create compound genre features for better recommendation"""
        try:
            # Mood-based compounds
            mood_mappings = {
                'genre_upbeat': ['genre_pop', 'genre_dance', 'genre_electronic', 'genre_party'],
                'genre_chill': ['genre_ballad', 'genre_folk', 'genre_ambient', 'genre_acoustic'],
                'genre_energetic': ['genre_rock', 'genre_hip_hop', 'genre_electronic', 'genre_dance'],
                'genre_emotional': ['genre_ballad', 'genre_soul', 'genre_r&b', 'genre_sad']
            }

            for compound_genre, component_genres in mood_mappings.items():
                available_components = [g for g in component_genres if g in self.tracks_df.columns]
                if available_components:
                    self.tracks_df[compound_genre] = self.tracks_df[available_components].max(axis=1)

            logger.info(f"Created {len(mood_mappings)} compound genre features")

        except Exception as e:
            logger.warning(f"Error creating compound genre features: {e}")

    def clean_tracks_data(self):
        """Clean and prepare tracks data for models"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to clean")
            return False
        
        initial_count = len(self.tracks_df)
        logger.info(f"Cleaning tracks data: {initial_count} tracks")
        
        # Remove duplicates
        if 'name' in self.tracks_df.columns and 'artist' in self.tracks_df.columns:
            before_dedup = len(self.tracks_df)
            self.tracks_df = self.tracks_df.drop_duplicates(subset=['name', 'artist'], keep='first')
            after_dedup = len(self.tracks_df)
            logger.info(f"Removed {before_dedup - after_dedup} duplicate tracks")
        
        # Fill missing essential values
        self.tracks_df['popularity'] = self.tracks_df['popularity'].fillna(50)
        self.tracks_df['artist_popularity'] = self.tracks_df['artist_popularity'].fillna(50)
        self.tracks_df['duration_ms'] = self.tracks_df['duration_ms'].fillna(200000)
        
        # Ensure required columns exist
        if 'release_year' not in self.tracks_df.columns:
            self.tracks_df['release_year'] = 2020
        
        clean_count = len(self.tracks_df)
        logger.info(f"Data cleaned: {initial_count} -> {clean_count} tracks")
        
        return True

    def normalize_features(self):
        """Normalize numerical features for model training"""
        if self.tracks_df is None:
            logger.error("No tracks data to normalize")
            return False

        try:
            # Features to normalize for models
            numerical_features = ['popularity', 'duration_ms', 'artist_popularity', 'track_age', 'markets_count', 'market_penetration']
            
            scaler = MinMaxScaler()
            normalized_count = 0
            
            for feature in numerical_features:
                if feature in self.tracks_df.columns:
                    # Fill missing values with median
                    feature_data = self.tracks_df[feature].fillna(self.tracks_df[feature].median())
                    
                    # Normalize to 0-1 range
                    normalized_values = scaler.fit_transform(feature_data.values.reshape(-1, 1))
                    self.tracks_df[f'{feature}_norm'] = normalized_values.flatten()
                    normalized_count += 1

            logger.info(f"Normalized {normalized_count} features for model training")
            return True

        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return False

    def save_processed_data(self):
        """Save processed data for model training"""
        if self.tracks_df is None:
            logger.error("No processed data to save")
            return False

        try:
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            
            # Save processed tracks
            processed_file = os.path.join(PROCESSED_DATA_DIR, "processed_tracks.csv")
            self.tracks_df.to_csv(processed_file, index=False)
            
            # Create feature summary for models
            feature_summary = {
                'total_tracks': len(self.tracks_df),
                'total_features': len(self.tracks_df.columns),
                'cultural_features': len([col for col in self.tracks_df.columns if col.startswith('is_') or col == 'music_culture']),
                'genre_features': len([col for col in self.tracks_df.columns if col.startswith('genre_')]),
                'normalized_features': len([col for col in self.tracks_df.columns if col.endswith('_norm')]),
                'isrc_coverage': (self.tracks_df['isrc'] != '').sum() / len(self.tracks_df),
                'cultural_confidence': self.tracks_df['cultural_confidence'].mean()
            }
            
            logger.info(f"✅ Processed data saved to {processed_file}")
            logger.info(f"📊 Feature summary for models: {feature_summary}")
            
            return True

        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False

    def process_all(self):
        """Complete data processing pipeline for recommendation models"""
        logger.info("🚀 Starting complete data processing for recommendation models...")
        
        steps = [
            ("Loading raw data", self.load_data),
            ("Extracting release year", self.extract_release_year), 
            ("Creating ISRC-based cultural intelligence", self.create_language_features),
            ("Merging artist genres", self.merge_artist_genres),
            ("Creating genre features", self.create_genre_features),
            ("Cleaning tracks data", self.clean_tracks_data),
            ("Normalizing features", self.normalize_features),
            ("Saving processed data", self.save_processed_data)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"🔄 {step_name}...")
            if not step_func():
                logger.error(f"❌ Failed at step: {step_name}")
                return False
        
        logger.info("✅ Data processing pipeline completed successfully!")
        logger.info("📊 Data is ready for training WeightedContentRecommender and EnhancedContentRecommender!")
        return True


if __name__ == "__main__":
    processor = DataProcessor()
    success = processor.process_all()
    if success:
        print("✅ Data processing completed successfully!")
    else:
        print("❌ Data processing failed!")