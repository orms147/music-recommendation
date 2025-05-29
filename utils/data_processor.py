import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CONTENT_FEATURES

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process raw data into ML-ready features - NO FETCHING"""
    
    def __init__(self):
        self.tracks_df = None
        self.audio_features_df = None
        self.artists_df = None

    def load_data(self):
        """Load existing data from processed directory"""
        try:
            processed_path = os.path.join(PROCESSED_DATA_DIR, "processed_tracks.csv")
            
            if os.path.exists(processed_path):
                self.tracks_df = pd.read_csv(processed_path, encoding='utf-8')
                logger.info(f"✅ Loaded {len(self.tracks_df)} tracks from processed data")
                return True
            else:
                # Try loading from raw data
                return self.load_raw_data()
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def load_raw_data(self):
        """Load raw data from files"""
        try:
            # Load tracks
            tracks_file = os.path.join(RAW_DATA_DIR, "tracks.csv")
            if os.path.exists(tracks_file):
                self.tracks_df = pd.read_csv(tracks_file, encoding='utf-8')
                logger.info(f"✅ Loaded {len(self.tracks_df)} raw tracks")
            else:
                logger.error(f"No data found in {RAW_DATA_DIR}")
                return False
            
            # Load audio features if available
            features_file = os.path.join(RAW_DATA_DIR, "audio_features.csv")
            if os.path.exists(features_file):
                self.audio_features_df = pd.read_csv(features_file, encoding='utf-8')
                logger.info(f"✅ Loaded {len(self.audio_features_df)} audio features")
            
            # Load artist info if available
            artists_file = os.path.join(RAW_DATA_DIR, "artist_genres.csv")
            if os.path.exists(artists_file):
                self.artists_df = pd.read_csv(artists_file, encoding='utf-8')
                logger.info(f"✅ Loaded {len(self.artists_df)} artists")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return False

    def process_all(self):
        """Complete processing pipeline"""
        logger.info("🔄 STARTING DATA PROCESSING PIPELINE")
        
        # Load data
        if not self.load_data():
            logger.error("❌ Failed to load data")
            return False
        
        # Processing steps
        steps = [
            ("Merge audio features", self.merge_audio_features),
            ("Merge artist data", self.merge_artist_data),
            ("Extract release year", self.extract_release_year),
            ("Extract cultural features", self.extract_cultural_features),
            ("Create genre features", self.create_genre_features),
            ("Clean data", self.clean_tracks_data),
            ("Normalize features", self.normalize_features),
            ("Save processed data", self.save_processed_data)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"🔄 {step_name}...")
            try:
                if not step_func():
                    logger.error(f"❌ Failed at: {step_name}")
                    return False
                logger.info(f"✅ {step_name} completed")
            except Exception as e:
                logger.error(f"❌ Error in {step_name}: {e}")
                return False
        
        logger.info("🎉 ALL PROCESSING COMPLETED SUCCESSFULLY!")
        return True

    def merge_audio_features(self):
        """Merge audio features with tracks"""
        if self.audio_features_df is None:
            logger.info("No audio features to merge")
            return True
        
        try:
            initial_cols = len(self.tracks_df.columns)
            self.tracks_df = self.tracks_df.merge(
                self.audio_features_df, 
                on='id', 
                how='left'
            )
            new_cols = len(self.tracks_df.columns)
            logger.info(f"✅ Merged audio features: {new_cols - initial_cols} new columns")
            return True
            
        except Exception as e:
            logger.error(f"Error merging audio features: {e}")
            return False

    def merge_artist_data(self):
        """Merge artist information"""
        if self.artists_df is None:
            logger.info("No artist data to merge")
            return True
        
        try:
            # Clean artist data
            artist_data = self.artists_df.copy()
            
            # Log cấu trúc dữ liệu để debug
            logger.info(f"Artist DataFrame columns: {artist_data.columns.tolist()}")
            logger.info(f"Tracks DataFrame columns: {self.tracks_df.columns.tolist()}")
            
            # Merge dựa trên artist_id (chính xác nhất)
            self.tracks_df = self.tracks_df.merge(
                artist_data[['artist_id', 'genres']], 
                on='artist_id',
                how='left'
            )
            
            # Đổi tên cột genres thành artist_genres
            if 'genres' in self.tracks_df.columns:
                self.tracks_df = self.tracks_df.rename(columns={'genres': 'artist_genres'})
            
            # Fill missing genres
            self.tracks_df['artist_genres'] = self.tracks_df['artist_genres'].fillna('')
            
            logger.info(f"✅ Merged artist data - {self.tracks_df.shape[1]} columns now available")
            return True
            
        except Exception as e:
            logger.error(f"Error merging artist data: {e}")
            return False

    def extract_release_year(self):
        """Extract release year from release_date"""
        if self.tracks_df is None:
            return False

        try:
            current_year = datetime.now().year
            
            # Extract year from release_date
            def extract_year(date_str):
                if pd.isna(date_str) or date_str == '':
                    return current_year
                try:
                    return int(str(date_str)[:4])
                except:
                    return current_year
            
            self.tracks_df['release_year'] = self.tracks_df['release_date'].apply(extract_year)
            
            # Calculate track age
            self.tracks_df['track_age'] = current_year - self.tracks_df['release_year']
            self.tracks_df['track_age'] = self.tracks_df['track_age'].clip(lower=0)
            
            logger.info("✅ Extracted release year and track age")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting release year: {e}")
            return False

    def extract_cultural_features(self):
        """Extract cultural features from ISRC ONLY - NO MARKET DATA"""
        if self.tracks_df is None:
            return False

        try:
            # Region mapping
            region_map = {
                # Asia
                'VN': 'asia', 'KR': 'asia', 'JP': 'asia', 'CN': 'asia', 
                'HK': 'asia', 'TW': 'asia', 'TH': 'asia', 'MY': 'asia', 
                'ID': 'asia', 'PH': 'asia', 'SG': 'asia', 'IN': 'asia',
                
                # Europe
                'GB': 'europe', 'DE': 'europe', 'FR': 'europe', 'IT': 'europe', 
                'ES': 'europe', 'NL': 'europe', 'SE': 'europe', 'NO': 'europe', 
                'DK': 'europe', 'FI': 'europe', 'PL': 'europe', 'RU': 'europe',
                
                # North America
                'US': 'north_america', 'CA': 'north_america',
                
                # Latin America
                'MX': 'latin_america', 'BR': 'latin_america', 'AR': 'latin_america', 
                'CO': 'latin_america', 'CL': 'latin_america', 'PE': 'latin_america',
                
                # Oceania
                'AU': 'oceania', 'NZ': 'oceania'
            }
            
            # Extract country from ISRC
            self.tracks_df['isrc_country'] = self.tracks_df['isrc'].str[:2].fillna('XX')
            
            # Map to region
            self.tracks_df['region'] = self.tracks_df['isrc_country'].map(region_map).fillna('other')
            
            # Cultural mapping
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
            
            self.tracks_df['music_culture'] = self.tracks_df['isrc_country'].map(cultural_mapping).fillna('other')
            
            # Cultural confidence
            self.tracks_df['cultural_confidence'] = np.where(
                self.tracks_df['isrc'].notna() & (self.tracks_df['isrc'] != ''), 
                0.9, 0.3
            )
            
            # Binary cultural features
            cultures = ['vietnamese', 'korean', 'japanese', 'chinese', 'western', 'spanish', 'brazilian', 'indian', 'thai']
            for culture in cultures:
                self.tracks_df[f'is_{culture}'] = (self.tracks_df['music_culture'] == culture).astype(int)
            
            # Binary regional features
            regions = ['asia', 'europe', 'north_america', 'latin_america', 'oceania']
            for region in regions:
                self.tracks_df[f'is_{region}'] = (self.tracks_df['region'] == region).astype(int)
            
            logger.info("✅ Extracted ISRC-based cultural features")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting cultural features: {e}")
            return False

    def create_genre_features(self):
        """Create genre features based purely on frequency"""
        if self.tracks_df is None:
            return False

        try:
            # Use artist_genres if available, otherwise skip
            genre_column = 'artist_genres' if 'artist_genres' in self.tracks_df.columns else None
            
            if genre_column is None or self.tracks_df[genre_column].isna().all():
                logger.info("No genre data available, skipping genre features")
                return True

            # Count all genres
            genre_counts = {}
            for genres_str in self.tracks_df[genre_column].dropna():
                if isinstance(genres_str, str) and genres_str.strip():
                    genres = [g.strip().lower().replace(' ', '_').replace('-', '_') 
                             for g in genres_str.split(',')]
                    for genre in genres:
                        if genre and len(genre) > 1:
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1

            # Filter by frequency
            significant_genres = {
                genre: count for genre, count in genre_counts.items() 
                if count >= 5  # At least 5 tracks
            }

            # Take top genres
            sorted_genres = sorted(significant_genres.items(), key=lambda x: x[1], reverse=True)
            final_genres = [genre for genre, count in sorted_genres[:100]]

            # Create genre matrix
            if final_genres:
                genre_data = {}
                for genre in final_genres:
                    genre_col = f'genre_{genre}'
                    genre_mask = self.tracks_df[genre_column].str.contains(
                        genre, case=False, na=False, regex=False
                    )
                    genre_data[genre_col] = genre_mask.astype(int)

                # Add to DataFrame
                genre_df = pd.DataFrame(genre_data, index=self.tracks_df.index)
                self.tracks_df = pd.concat([self.tracks_df, genre_df], axis=1)
                
                logger.info(f"✅ Created {len(final_genres)} genre features")
            else:
                logger.info("No significant genres found")

            return True

        except Exception as e:
            logger.error(f"Error creating genre features: {e}")
            return False

    def clean_tracks_data(self):
        """Clean tracks data"""
        if self.tracks_df is None or self.tracks_df.empty:
            return False
        
        initial_count = len(self.tracks_df)
        
        try:
            # Remove duplicates
            if 'id' in self.tracks_df.columns:
                before_dedup = len(self.tracks_df)
                self.tracks_df = self.tracks_df.drop_duplicates(subset=['id'], keep='first')
                after_dedup = len(self.tracks_df)
                logger.info(f"Removed {before_dedup - after_dedup} duplicates")
            
            # Fill numeric columns
            numeric_columns = {
                'popularity': 50,
                'duration_ms': 200000,
                'release_year': 2020,
                'track_age': 4
            }
            
            for col, default_value in numeric_columns.items():
                if col in self.tracks_df.columns:
                    self.tracks_df[col] = self.tracks_df[col].fillna(default_value)
            
            # Fill string columns
            string_columns = {
                'isrc': '',
                'artist_genres': '',
                'album': 'Unknown Album'
            }
            
            for col, default_value in string_columns.items():
                if col in self.tracks_df.columns:
                    self.tracks_df[col] = self.tracks_df[col].fillna(default_value)
            
            # Remove rows with missing essential data
            essential_columns = ['id', 'name', 'artist']
            for col in essential_columns:
                if col in self.tracks_df.columns:
                    mask = self.tracks_df[col].notna() & (self.tracks_df[col] != '')
                    self.tracks_df = self.tracks_df[mask]
            
            clean_count = len(self.tracks_df)
            logger.info(f"Cleaned: {initial_count} -> {clean_count} tracks")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return False

    def normalize_features(self):
        """Normalize numerical features"""
        if self.tracks_df is None:
            return False

        try:
            # Define numerical features to normalize
            numerical_features = ['popularity', 'duration_ms', 'track_age']
            
            # Filter for existing columns
            existing_features = [feat for feat in numerical_features if feat in self.tracks_df.columns]
            
            if not existing_features:
                logger.info("No numerical features to normalize")
                return True

            # Create normalized columns
            normalized_data = {}
            scaler = MinMaxScaler()
            
            for feature in existing_features:
                feature_data = self.tracks_df[feature].fillna(self.tracks_df[feature].median())
                normalized_values = scaler.fit_transform(feature_data.values.reshape(-1, 1))
                normalized_data[f'{feature}_norm'] = normalized_values.flatten()

            # Add to DataFrame
            if normalized_data:
                normalized_df = pd.DataFrame(normalized_data, index=self.tracks_df.index)
                self.tracks_df = pd.concat([self.tracks_df, normalized_df], axis=1)
            
            logger.info(f"✅ Normalized {len(existing_features)} features")
            return True

        except Exception as e:
            logger.error(f"Error normalizing: {e}")
            return False

    def save_processed_data(self):
        """Save processed data"""
        try:
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            
            # Save main processed file
            processed_file = os.path.join(PROCESSED_DATA_DIR, "processed_tracks.csv")
            self.tracks_df.to_csv(processed_file, index=False, encoding='utf-8')
            
            # Calculate summary
            feature_summary = {
                'total_tracks': len(self.tracks_df),
                'total_features': len(self.tracks_df.columns),
                'isrc_coverage': float((self.tracks_df['isrc'] != '').sum() / len(self.tracks_df)) if 'isrc' in self.tracks_df.columns else 0.0,
                'cultural_confidence': float(self.tracks_df['cultural_confidence'].mean()) if 'cultural_confidence' in self.tracks_df.columns else 0.0
            }
            
            # Count features
            genre_features = [col for col in self.tracks_df.columns if col.startswith('genre_')]
            cultural_features = [col for col in self.tracks_df.columns if col.startswith('is_')]
            normalized_features = [col for col in self.tracks_df.columns if col.endswith('_norm')]
            
            feature_summary['genre_features_count'] = len(genre_features)
            feature_summary['cultural_features_count'] = len(cultural_features)
            feature_summary['normalized_features_count'] = len(normalized_features)
            
            logger.info(f"✅ Saved to {processed_file}")
            logger.info(f"📊 Summary: {feature_summary}")
            
            return True

        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False

# ✅ FIXED STANDALONE EXECUTION
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔄 STEP 2: PROCESSING RAW DATA")
    print("=" * 50)
    
    processor = DataProcessor()
    success = processor.process_all()
    
    if success:
        print("\n🎉 ✅ Data processing completed successfully!")
        print(f"📂 Processed data saved to: {PROCESSED_DATA_DIR}")
        print("\n🤖 Next step: Run main.py to train models")
    else:
        print("❌ Processing failed")
