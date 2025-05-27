import os
import logging
import pandas as pd
import numpy as np
import re  # ✅ THÊM IMPORT NÀY
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, RobustScaler
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
        """Load raw data from files - focus on real metadata only"""
        tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
        enriched_path = os.path.join(RAW_DATA_DIR, 'enriched_tracks.csv')
        
        if os.path.exists(enriched_path):
            logger.info(f"Loading enriched tracks from {enriched_path}")
            self.tracks_df = pd.read_csv(enriched_path)
        elif os.path.exists(tracks_path):
            logger.info(f"Loading basic tracks from {tracks_path}")
            self.tracks_df = pd.read_csv(tracks_path)
        else:
            logger.error("No track data files found!")
            return False
        
        artist_genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
        if os.path.exists(artist_genres_path):
            logger.info(f"Loading artist genres from {artist_genres_path}")
            self.artist_genres_df = pd.read_csv(artist_genres_path)
        else:
            logger.warning("No artist genres file found - will attempt to fetch from Spotify")
            if self._fetch_missing_artist_genres():
                self.artist_genres_df = pd.read_csv(artist_genres_path)
            else:
                logger.warning("Failed to fetch artist genres, will use fallback genre features")
                self.artist_genres_df = None
        
        logger.info(f"Loaded {len(self.tracks_df)} tracks")
        if self.artist_genres_df is not None:
            logger.info(f"Loaded {len(self.artist_genres_df)} artist genre records")
        
        return True

    def _fetch_missing_artist_genres(self):
        """Fetch missing artist genres using SpotifyDataFetcher"""
        try:
            from utils.data_fetcher import SpotifyDataFetcher
            
            logger.info("Attempting to fetch missing artist genres from Spotify...")
            fetcher = SpotifyDataFetcher()
            return fetcher.fetch_all_missing_artist_genres()
            
        except Exception as e:
            logger.error(f"Failed to fetch artist genres: {e}")
            return False

    def merge_artist_genres(self):
        """Merge artist genres with tracks data"""
        if self.artist_genres_df is None:
            logger.warning("No artist genres data available, skipping merge")
            return
        
        if 'artist_id' not in self.tracks_df.columns:
            logger.warning("No artist_id column in tracks data")
            return
        
        try:
            before_count = len(self.tracks_df)
            merged_df = self.tracks_df.merge(
                self.artist_genres_df[['artist_id', 'genres', 'artist_popularity', 'artist_followers']], 
                on='artist_id', 
                how='left'
            )
            
            merged_df = self._clean_artist_popularity_columns(merged_df)
            self.tracks_df = merged_df
            
            genre_coverage = (self.tracks_df['genres'].notna().sum() / len(self.tracks_df)) * 100
            logger.info(f"Merged artist genres: {before_count} -> {len(self.tracks_df)} tracks")
            logger.info(f"Genre coverage: {genre_coverage:.1f}% of tracks have genre data")
            
        except Exception as e:
            logger.error(f"Error merging artist genres: {e}")

    def _clean_artist_popularity_columns(self, df):
        """Clean conflicting artist popularity columns"""
        if 'artist_popularity_x' in df.columns and 'artist_popularity_y' in df.columns:
            df['artist_popularity'] = df['artist_popularity_y'].fillna(df['artist_popularity_x'])
            df = df.drop(['artist_popularity_x', 'artist_popularity_y'], axis=1)
        
        return df

    def create_genre_features(self):
        """Create genre features from artist genres data"""
        if 'genres' not in self.tracks_df.columns:
            logger.warning("No genres column found, creating fallback genre features")
            self._create_fallback_genre_features()
            return
        
        try:
            all_genres = set()
            
            for genre_str in self.tracks_df['genres'].dropna():
                if isinstance(genre_str, str) and genre_str.strip():
                    genres = [g.strip().lower() for g in genre_str.split('|') if g.strip()]
                    all_genres.update(genres)
            
            logger.info(f"Found {len(all_genres)} unique genres")
            
            genre_counts = {}
            for genre in all_genres:
                count = self.tracks_df['genres'].str.contains(genre, case=False, na=False).sum()
                if count >= 10:
                    genre_counts[genre] = count
            
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:25]
            
            logger.info(f"Creating features for top {len(top_genres)} genres:")
            for genre, count in top_genres[:10]:
                logger.info(f"  {genre}: {count} tracks")
            
            for genre, count in top_genres:
                clean_genre = genre.replace(' ', '_').replace('-', '_').replace('&', 'and')
                clean_genre = ''.join(c for c in clean_genre if c.isalnum() or c == '_')
                col_name = f'genre_{clean_genre}'
                
                self.tracks_df[col_name] = self.tracks_df['genres'].str.contains(
                    genre, case=False, na=False
                ).astype(int)
            
            self._create_compound_genre_features()
            
            genre_feature_count = len([col for col in self.tracks_df.columns if col.startswith('genre_')])
            logger.info(f"Created {genre_feature_count} genre features")
            
        except Exception as e:
            logger.error(f"Error creating genre features: {e}")
            self._create_fallback_genre_features()

    def _create_fallback_genre_features(self):
        """Create fallback genre features when real genre data is unavailable"""
        logger.info("Creating fallback genre features from available data...")
        
        genre_mapping = {
            'genre_pop': lambda row: 1 if row.get('popularity', 0) > 60 else 0,
            'genre_kpop': lambda row: row.get('is_korean', 0) if 'is_korean' in self.tracks_df.columns else 0,
            'genre_jpop': lambda row: row.get('is_japanese', 0) if 'is_japanese' in self.tracks_df.columns else 0,
            'genre_vpop': lambda row: row.get('is_vietnamese', 0) if 'is_vietnamese' in self.tracks_df.columns else 0,
            'genre_latin': lambda row: row.get('is_spanish', 0) if 'is_spanish' in self.tracks_df.columns else 0,
            'genre_mainstream': lambda row: 1 if row.get('popularity', 0) > 70 else 0,
            'genre_underground': lambda row: 1 if row.get('popularity', 0) < 30 else 0,
        }
        
        pattern_genres = {
            'genre_electronic': ['electronic', 'edm', 'house', 'techno', 'trance', 'dubstep'],
            'genre_hip_hop': ['rap', 'hip hop', 'trap', 'drill'],
            'genre_rock': ['rock', 'metal', 'punk', 'grunge'],
            'genre_indie': ['indie', 'alternative', 'art'],
            'genre_rnb': ['r&b', 'soul', 'neo soul'],
            'genre_country': ['country', 'folk', 'bluegrass'],
            'genre_jazz': ['jazz', 'blues', 'swing'],
            'genre_classical': ['classical', 'orchestra', 'symphony'],
        }
        
        for genre_col, func in genre_mapping.items():
            try:
                self.tracks_df[genre_col] = self.tracks_df.apply(func, axis=1)
            except Exception as e:
                logger.warning(f"Error creating {genre_col}: {e}")
                self.tracks_df[genre_col] = 0
        
        for genre_col, patterns in pattern_genres.items():
            try:
                genre_pattern = '|'.join(patterns)
                
                name_match = self.tracks_df['name'].str.contains(
                    genre_pattern, case=False, na=False
                ).astype(int)
                
                artist_match = self.tracks_df['artist'].str.contains(
                    genre_pattern, case=False, na=False
                ).astype(int)
                
                self.tracks_df[genre_col] = (name_match | artist_match).astype(int)
                
            except Exception as e:
                logger.warning(f"Error creating {genre_col}: {e}")
                self.tracks_df[genre_col] = 0
        
        fallback_count = len([col for col in self.tracks_df.columns if col.startswith('genre_')])
        logger.info(f"Created {fallback_count} fallback genre features")

    def _create_compound_genre_features(self):
        """Create compound genre features from existing ones"""
        try:
            asian_genres = ['genre_kpop', 'genre_jpop', 'genre_cpop']
            available_asian = [g for g in asian_genres if g in self.tracks_df.columns]
            if available_asian:
                self.tracks_df['genre_asian'] = self.tracks_df[available_asian].max(axis=1)
            
            electronic_patterns = ['house', 'techno', 'trance', 'dubstep', 'electronic']
            electronic_cols = [f'genre_{pattern}' for pattern in electronic_patterns 
                             if f'genre_{pattern}' in self.tracks_df.columns]
            if electronic_cols:
                self.tracks_df['genre_electronic_umbrella'] = self.tracks_df[electronic_cols].max(axis=1)
            
            if 'popularity' in self.tracks_df.columns:
                self.tracks_df['genre_viral'] = (self.tracks_df['popularity'] > 80).astype(int)
                self.tracks_df['genre_hit'] = ((self.tracks_df['popularity'] > 60) & 
                                             (self.tracks_df['popularity'] <= 80)).astype(int)
                self.tracks_df['genre_niche'] = (self.tracks_df['popularity'] < 30).astype(int)
            
        except Exception as e:
            logger.warning(f"Error creating compound genre features: {e}")

    def clean_tracks_data(self):
        """Clean tracks data"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to clean")
            return False
        
        initial_count = len(self.tracks_df)
        
        essential_cols = ['id', 'name', 'artist']
        self.tracks_df = self.tracks_df.dropna(subset=essential_cols)
        
        self.tracks_df = self.tracks_df.drop_duplicates(subset=['id'])
        
        if 'popularity' in self.tracks_df.columns:
            self.tracks_df['popularity'] = self.tracks_df['popularity'].fillna(0)
        
        if 'explicit' in self.tracks_df.columns:
            self.tracks_df['explicit'] = self.tracks_df['explicit'].fillna(0).astype(int)
        
        if 'duration_ms' in self.tracks_df.columns:
            self.tracks_df['duration_ms'] = self.tracks_df['duration_ms'].fillna(self.tracks_df['duration_ms'].median())
        
        if 'artist_popularity' not in self.tracks_df.columns:
            self.tracks_df['artist_popularity'] = 50
        self.tracks_df['artist_popularity'] = self.tracks_df['artist_popularity'].fillna(50)
        
        clean_count = len(self.tracks_df)
        logger.info(f"Cleaned tracks data: {initial_count} -> {clean_count} tracks")
        
        return True
    
    def create_synthetic_audio_features(self):
        """Skip creating synthetic audio features - focus on real metadata only"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to check features for")
            return False
        
        logger.info("Skipping synthetic audio features - focusing on real Spotify metadata only")
        
        essential_features = ['popularity', 'duration_ms', 'explicit']
        
        for feature in essential_features:
            if feature not in self.tracks_df.columns:
                logger.warning(f"Missing essential feature: {feature}")
                if feature == 'popularity':
                    self.tracks_df[feature] = 0
                elif feature == 'duration_ms':
                    self.tracks_df[feature] = 200000
                elif feature == 'explicit':
                    self.tracks_df[feature] = 0
        
        real_spotify_features = ['popularity', 'duration_ms', 'explicit', 'release_date', 
                               'album_type', 'total_tracks', 'track_number', 'disc_number']
        existing_real_features = [f for f in real_spotify_features if f in self.tracks_df.columns]
        
        logger.info(f"Using {len(existing_real_features)} real Spotify metadata features: {existing_real_features}")
        
        return True
    
    def extract_release_year(self):
        """Extract release year from release date"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to extract release year from")
            return False
        
        if 'release_date' not in self.tracks_df.columns:
            logger.warning("No release_date column in tracks data")
            return False
        
        if 'release_year' in self.tracks_df.columns:
            logger.info("Release year already exists in data")
            return True
        
        def extract_year(date_str):
            if not date_str or pd.isna(date_str):
                return datetime.now().year
            
            try:
                if '-' in str(date_str):
                    parts = str(date_str).split('-')
                    if len(parts) > 0 and len(parts[0]) == 4:
                        return int(parts[0])
                
                year = int(date_str)
                if 1900 <= year <= datetime.now().year:
                    return year
            except:
                pass
            
            return datetime.now().year
        
        self.tracks_df['release_year'] = self.tracks_df['release_date'].apply(extract_year)
        self.tracks_df['decade'] = (self.tracks_df['release_year'] // 10) * 10
        
        logger.info("Extracted release year and decade from release dates")
        
        return True
    
    # ✅ THAY THẾ HOÀN TOÀN METHOD NÀY
    def create_language_features(self):
        """Enhanced cultural intelligence using ISRC + available_markets + text analysis"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create language features")
            return False
        
        logger.info("Creating ISRC-based cultural intelligence features...")
        
        # 1. ISRC-based country detection (PRIORITY #1 - MOST ACCURATE)
        if 'isrc' in self.tracks_df.columns:
            self._create_isrc_based_features()
        else:
            logger.warning("No ISRC column found - creating default country column")
            self.tracks_df['isrc_country'] = 'Unknown'
        
        # 2. Market-based regional features (SECONDARY)
        if 'available_markets' in self.tracks_df.columns:
            self._create_market_based_features()
        else:
            logger.warning("No available_markets column - skipping market analysis")
        
        # 3. Text-based detection (FALLBACK for missing ISRC)
        self._create_text_based_language_features()
        
        # 4. Create unified cultural features (FINAL)
        self._create_unified_cultural_features()
        
        logger.info("Enhanced cultural intelligence features created")
        return True

    # ✅ THÊM METHOD MỚI 1
    def _create_isrc_based_features(self):
        """Create accurate country/cultural features from ISRC codes"""
        logger.info("Analyzing ISRC codes for precise country detection...")
        
        def extract_isrc_country(isrc):
            if pd.isna(isrc) or len(str(isrc)) < 2:
                return 'Unknown'
            return str(isrc)[:2].upper()
        
        def extract_isrc_registrant(isrc):
            if pd.isna(isrc) or len(str(isrc)) < 5:
                return 'Unknown'
            return str(isrc)[2:5].upper()
        
        # Extract ISRC components
        self.tracks_df['isrc_country'] = self.tracks_df['isrc'].apply(extract_isrc_country)
        self.tracks_df['isrc_registrant'] = self.tracks_df['isrc'].apply(extract_isrc_registrant)
        
        # DEFINITIVE COUNTRY MAPPING from ISRC standards
        isrc_country_mapping = {
            'US': 'united_states', 'GB': 'united_kingdom', 'KR': 'south_korea', 'JP': 'japan',
            'VN': 'vietnam', 'CN': 'china', 'DE': 'germany', 'FR': 'france', 'BR': 'brazil',
            'MX': 'mexico', 'CA': 'canada', 'AU': 'australia', 'SE': 'sweden', 'NO': 'norway',
            'IT': 'italy', 'ES': 'spain', 'NL': 'netherlands', 'AT': 'austria', 'CH': 'switzerland',
            'TW': 'taiwan', 'HK': 'hong_kong', 'SG': 'singapore', 'MY': 'malaysia', 'TH': 'thailand',
            'ID': 'indonesia', 'PH': 'philippines', 'IN': 'india', 'RU': 'russia', 'AR': 'argentina'
        }
        
        self.tracks_df['isrc_country_name'] = self.tracks_df['isrc_country'].map(
            isrc_country_mapping
        ).fillna('unknown')
        
        # Create binary features for major countries
        major_countries = ['united_states', 'united_kingdom', 'south_korea', 'japan', 
                          'vietnam', 'china', 'germany', 'france', 'brazil', 'mexico']
        
        for country in major_countries:
            col_name = f'isrc_{country}'
            self.tracks_df[col_name] = (self.tracks_df['isrc_country_name'] == country).astype(int)
        
        # Major label detection from ISRC registrant codes
        major_label_registrants = {
            'UMG': 'universal', 'UNI': 'universal', 'MCA': 'universal', 'DEF': 'universal',
            'CAP': 'universal', 'VIR': 'universal', 'REP': 'universal', 'POL': 'universal',
            'SME': 'sony', 'SON': 'sony', 'COL': 'sony', 'RCA': 'sony', 'ARI': 'sony',
            'EPC': 'sony', 'REL': 'sony', 'LEG': 'sony',
            'WEA': 'warner', 'WMG': 'warner', 'ATL': 'warner', 'ELE': 'warner', 
            'NON': 'warner', 'WBR': 'warner', 'ASY': 'warner',
            'BMG': 'bmg', 'EMI': 'emi', 'IND': 'independent'
        }
        
        self.tracks_df['record_label_type'] = self.tracks_df['isrc_registrant'].map(
            major_label_registrants
        ).fillna('independent')
        
        self.tracks_df['is_major_label'] = (
            self.tracks_df['record_label_type'].isin(['universal', 'sony', 'warner'])
        ).astype(int)
        
        # Log results
        country_counts = self.tracks_df['isrc_country_name'].value_counts()
        logger.info(f"ISRC country distribution: {dict(country_counts.head(10))}")
        
        label_counts = self.tracks_df['record_label_type'].value_counts()
        logger.info(f"Label distribution: {dict(label_counts)}")

    # ✅ THÊM METHOD MỚI 2
    def _create_market_based_features(self):
        """Create regional features based on available_markets"""
        
        def parse_markets(markets_str):
            if pd.isna(markets_str) or not markets_str:
                return []
            return [market.strip() for market in str(markets_str).split('|') if market.strip()]
        
        self.tracks_df['markets_list'] = self.tracks_df['available_markets'].apply(parse_markets)
        
        # Update markets_count if not accurate
        if 'markets_count' not in self.tracks_df.columns:
            self.tracks_df['markets_count'] = self.tracks_df['markets_list'].apply(len)
        
        # Market penetration score
        max_markets = self.tracks_df['markets_count'].max()
        if max_markets > 0:
            self.tracks_df['market_penetration'] = self.tracks_df['markets_count'] / max_markets
        else:
            self.tracks_df['market_penetration'] = 0.0
        
        # Regional market presence for music recommendation
        market_regions = {
            'market_east_asia': ['KR', 'JP', 'CN', 'TW', 'HK', 'MO'],
            'market_southeast_asia': ['VN', 'TH', 'MY', 'SG', 'ID', 'PH', 'KH', 'LA', 'MM', 'BN'],
            'market_north_america': ['US', 'CA', 'MX'],
            'market_europe': ['GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'SE', 'NO', 'DK', 'FI', 'AT', 'CH', 'BE'],
            'market_latin_america': ['BR', 'AR', 'CO', 'CL', 'PE', 'VE', 'EC', 'UY', 'PY', 'BO'],
            'market_oceania': ['AU', 'NZ', 'FJ', 'PG'],
            'market_south_asia': ['IN', 'PK', 'BD', 'LK', 'NP'],
            'market_middle_east': ['AE', 'SA', 'IL', 'TR', 'EG', 'LB', 'JO'],
            'market_africa': ['ZA', 'NG', 'KE', 'GH', 'MA', 'TN'],
        }
        
        for region_name, countries in market_regions.items():
            self.tracks_df[region_name] = self.tracks_df['markets_list'].apply(
                lambda markets: 1 if any(country in markets for country in countries) else 0
            )
        
        # Global release indicator
        self.tracks_df['is_global_release'] = (self.tracks_df['markets_count'] > 100).astype(int)
        
        # Cultural diversity score
        region_cols = list(market_regions.keys())
        self.tracks_df['cultural_diversity_score'] = self.tracks_df[region_cols].sum(axis=1) / len(region_cols)
        
        logger.info("Created market-based regional features for music recommendation")

    # ✅ THÊM METHOD MỚI 3
    def _create_text_based_language_features(self):
        """Fallback text-based language detection (only for tracks without ISRC)"""
        
        # Only for tracks missing ISRC country info
        missing_isrc = (self.tracks_df['isrc_country_name'] == 'unknown') | (self.tracks_df['isrc_country'].isna())
        
        if missing_isrc.sum() == 0:
            logger.info("All tracks have ISRC data, skipping text analysis")
            return
        
        logger.info(f"Using text analysis for {missing_isrc.sum():,} tracks without ISRC")
        
        # Refined language patterns for major music markets
        language_patterns = {
            'vietnamese': {
                'chars': ['Đ', 'đ', 'Ư', 'ư', 'Ơ', 'ơ', 'Ă', 'ă', 'Â', 'â', 'Ê', 'ê', 'Ô', 'ô'],
                'words': ['việt', 'tình', 'yêu', 'anh', 'em', 'sài gòn', 'hà nội'],
            },
            'korean': {
                'chars': ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'],
                'words': ['korea', 'k-pop', 'kpop', 'seoul', '한국', 'bts', 'blackpink'],
            },
            'japanese': {
                'chars': ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'ア', 'イ', 'ウ', 'エ', 'オ'],
                'words': ['japan', 'j-pop', 'jpop', 'tokyo', 'anime'],
            },
            'chinese': {
                'chars': ['中', '国', '华', '文', '歌', '曲', '音', '乐', '爱', '你', '我'],
                'words': ['china', 'chinese', 'mandarin', 'taiwan', 'hong kong', 'c-pop'],
            },
            'spanish': {
                'chars': ['ñ', 'Ñ', 'á', 'é', 'í', 'ó', 'ú'],
                'words': ['español', 'latino', 'reggaeton', 'salsa', 'bachata'],
            }
        }
        
        # Apply text detection only to missing ISRC tracks
        for lang, patterns in language_patterns.items():
            all_patterns = patterns.get('chars', []) + patterns.get('words', [])
            if all_patterns:
                pattern = '|'.join([re.escape(p) for p in all_patterns])
                
                text_detected = (
                    self.tracks_df['name'].str.contains(pattern, case=False, na=False, regex=True) |
                    self.tracks_df['artist'].str.contains(pattern, case=False, na=False, regex=True)
                )
                
                # Only apply to tracks missing ISRC
                text_lang_col = f'text_{lang}'
                self.tracks_df[text_lang_col] = (missing_isrc & text_detected).astype(int)

    # ✅ THÊM METHOD MỚI 4
    def _create_unified_cultural_features(self):
        """Create unified cultural features combining ISRC + market + text analysis"""
        
        # Priority: ISRC > Market > Text analysis
        
        # 1. Vietnamese tracks
        vietnamese_isrc = self.tracks_df.get('isrc_vietnam', 0)
        vietnamese_market = self.tracks_df.get('market_southeast_asia', 0) & (
            self.tracks_df['markets_list'].apply(lambda x: 'VN' in x if isinstance(x, list) else False).astype(int)
        )
        vietnamese_text = self.tracks_df.get('text_vietnamese', 0)
        self.tracks_df['is_vietnamese'] = (vietnamese_isrc | vietnamese_market | vietnamese_text).astype(int)
        
        # 2. Korean tracks (K-pop)
        korean_isrc = self.tracks_df.get('isrc_south_korea', 0)
        korean_market = self.tracks_df.get('market_east_asia', 0) & (
            self.tracks_df['markets_list'].apply(lambda x: 'KR' in x if isinstance(x, list) else False).astype(int)
        )
        korean_text = self.tracks_df.get('text_korean', 0)
        self.tracks_df['is_korean'] = (korean_isrc | korean_market | korean_text).astype(int)
        
        # 3. Japanese tracks
        japanese_isrc = self.tracks_df.get('isrc_japan', 0)
        japanese_market = self.tracks_df.get('market_east_asia', 0) & (
            self.tracks_df['markets_list'].apply(lambda x: 'JP' in x if isinstance(x, list) else False).astype(int)
        )
        japanese_text = self.tracks_df.get('text_japanese', 0)
        self.tracks_df['is_japanese'] = (japanese_isrc | japanese_market | japanese_text).astype(int)
        
        # 4. Chinese tracks
        chinese_isrc = self.tracks_df.get('isrc_china', 0)
        chinese_market = self.tracks_df.get('market_east_asia', 0) & (
            self.tracks_df['markets_list'].apply(lambda x: any(c in x for c in ['CN', 'TW', 'HK']) if isinstance(x, list) else False).astype(int)
        )
        chinese_text = self.tracks_df.get('text_chinese', 0)
        self.tracks_df['is_chinese'] = (chinese_isrc | chinese_market | chinese_text).astype(int)
        
        # 5. Spanish tracks
        spanish_isrc = (
            self.tracks_df.get('isrc_mexico', 0) |
            self.tracks_df.get('isrc_brazil', 0) |
            self.tracks_df['isrc_country_name'].isin(['spain', 'argentina', 'colombia']).astype(int)
        )
        spanish_market = self.tracks_df.get('market_latin_america', 0)
        spanish_text = self.tracks_df.get('text_spanish', 0)
        self.tracks_df['is_spanish'] = (spanish_isrc | spanish_market | spanish_text).astype(int)
        
        # 6. Western tracks
        western_isrc = (
            self.tracks_df.get('isrc_united_states', 0) |
            self.tracks_df.get('isrc_united_kingdom', 0) |
            self.tracks_df['isrc_country_name'].isin(['canada', 'australia']).astype(int)
        )
        western_market = self.tracks_df.get('market_north_america', 0)
        self.tracks_df['is_western'] = (western_isrc | western_market).astype(int)
        
        # Professional release quality score
        self.tracks_df['professional_release_score'] = (
            self.tracks_df['is_major_label'] * 0.4 +                    # Major label
            (~self.tracks_df['isrc'].isna()).astype(float) * 0.3 +      # Has ISRC
            self.tracks_df.get('market_penetration', 0) * 0.2 +         # Market reach
            (self.tracks_df.get('markets_count', 0) > 50).astype(float) * 0.1  # Wide release
        )
        
        # Log unified results
        language_summary = {}
        for lang in ['vietnamese', 'korean', 'japanese', 'chinese', 'spanish', 'western']:
            count = self.tracks_df[f'is_{lang}'].sum()
            if count > 0:
                language_summary[lang] = count
        
        logger.info(f"Unified cultural features: {language_summary}")
    
    def create_additional_features(self):
        """Create meaningful additional features from metadata"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to create additional features")
            return False
        
        # 1. Duration features
        if 'duration_ms' in self.tracks_df.columns:
            self.tracks_df['duration_min'] = self.tracks_df['duration_ms'] / 60000
            
            self.tracks_df['duration_category'] = pd.cut(
                self.tracks_df['duration_min'],
                bins=[0, 2.5, 3.5, 4.5, 6, 100],
                labels=['short', 'normal', 'extended', 'long', 'very_long']
            )
            
            optimal_duration = 3.5
            self.tracks_df['duration_score'] = np.exp(-0.5 * ((self.tracks_df['duration_min'] - optimal_duration) / 1.5) ** 2)
        
        # 2. Popularity features
        if 'popularity' in self.tracks_df.columns:
            self.tracks_df['popularity_tier'] = pd.cut(
                self.tracks_df['popularity'],
                bins=[0, 30, 50, 70, 85, 100],
                labels=['niche', 'emerging', 'popular', 'hit', 'viral']
            )
            
            current_year = datetime.now().year
            if 'release_year' in self.tracks_df.columns:
                years_since_release = current_year - self.tracks_df['release_year'].fillna(current_year)
                self.tracks_df['popularity_momentum'] = self.tracks_df['popularity'] / (1 + 0.1 * years_since_release)
        
        # 3. Track name features
        if 'name' in self.tracks_df.columns:
            collab_pattern = r'(feat\.?|ft\.?|featuring|with|vs\.?|x\s+|\&|\+)'
            self.tracks_df['has_collab'] = self.tracks_df['name'].str.contains(
                collab_pattern, case=False, regex=True, na=False
            ).astype(int)
            
            remix_pattern = r'(remix|edit|version|mix|remaster|acoustic|live|demo|instrumental)'
            self.tracks_df['is_remix'] = self.tracks_df['name'].str.contains(
                remix_pattern, case=False, regex=True, na=False
            ).astype(int)
            
            self.tracks_df['name_length'] = self.tracks_df['name'].str.len()
            self.tracks_df['name_words'] = self.tracks_df['name'].str.split().str.len()
            
            self.tracks_df['has_special_chars'] = self.tracks_df['name'].str.contains(
                r'[^\w\s\-\(\)\[\]\.,:;!?\'"&]', regex=True, na=False
            ).astype(int)
        
        # 4. Artist features
        if 'artist' in self.tracks_df.columns:
            artist_counts = self.tracks_df['artist'].value_counts()
            self.tracks_df['artist_frequency'] = self.tracks_df['artist'].map(artist_counts)
            self.tracks_df['artist_frequency_log'] = np.log1p(self.tracks_df['artist_frequency'])
            
            max_freq_log = self.tracks_df['artist_frequency_log'].max()
            if max_freq_log > 0:
                self.tracks_df['artist_frequency_norm'] = self.tracks_df['artist_frequency_log'] / max_freq_log
            else:
                self.tracks_df['artist_frequency_norm'] = 0
            
            self.tracks_df['is_multi_artist'] = self.tracks_df['artist'].str.contains(
                r'[,&\+]|feat|ft\.?', case=False, regex=True, na=False
            ).astype(int)
        
        # 5. Album features
        if 'album_type' in self.tracks_df.columns:
            album_type_mapping = {'album': 1.0, 'single': 0.8, 'compilation': 0.6}
            self.tracks_df['album_type_score'] = self.tracks_df['album_type'].map(
                album_type_mapping
            ).fillna(0.5)
        
        if 'total_tracks' in self.tracks_df.columns:
            self.tracks_df['album_size_category'] = pd.cut(
                self.tracks_df['total_tracks'].fillna(1),
                bins=[0, 1, 5, 12, 20, 100],
                labels=['single', 'ep', 'album', 'long_album', 'compilation']
            )
        
        logger.info("Created improved additional features from metadata")
        return True

    def normalize_features(self):
        """Improved feature normalization focusing on real metadata"""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("No tracks data to normalize")
            return False
        
        # ✅ BỔ SUNG FEATURES MỚI VÀO NORMALIZATION
        numeric_features = {
            # Core Spotify features
            'popularity': {'method': 'minmax', 'fill': 0},
            'duration_ms': {'method': 'robust', 'fill': 'median'},
            'artist_popularity': {'method': 'minmax', 'fill': 50},
            'release_year': {'method': 'minmax', 'fill': 2000},
            
            # NEW: ISRC + Market features
            'market_penetration': {'method': 'minmax', 'fill': 0},
            'cultural_diversity_score': {'method': 'minmax', 'fill': 0},
            'professional_release_score': {'method': 'minmax', 'fill': 0},
            'markets_count': {'method': 'robust', 'fill': 0},
            
            # Derived features
            'duration_min': {'method': 'robust', 'fill': 'median'},
            'name_length': {'method': 'robust', 'fill': 'median'},
            'name_words': {'method': 'robust', 'fill': 'median'},
            'artist_frequency': {'method': 'log_minmax', 'fill': 1},
            'total_tracks': {'method': 'log_minmax', 'fill': 1},
            'track_number': {'method': 'minmax', 'fill': 1},
            'disc_number': {'method': 'minmax', 'fill': 1},
            
            # Computed scores
            'duration_score': {'method': 'minmax', 'fill': 0.5},
            'popularity_momentum': {'method': 'robust', 'fill': 'median'},
            'album_type_score': {'method': 'minmax', 'fill': 0.5}
        }
        
        # Apply normalization
        for feature, config in numeric_features.items():
            if feature in self.tracks_df.columns:
                # Handle missing values
                if config['fill'] == 'median':
                    fill_value = self.tracks_df[feature].median()
                else:
                    fill_value = config['fill']
                
                self.tracks_df[feature] = self.tracks_df[feature].fillna(fill_value)
                
                # Apply normalization
                values = self.tracks_df[feature].values.reshape(-1, 1)
                
                if config['method'] == 'minmax':
                    scaler = MinMaxScaler()
                    self.tracks_df[feature] = scaler.fit_transform(values).flatten()
                elif config['method'] == 'robust':
                    scaler = RobustScaler()
                    self.tracks_df[feature] = scaler.fit_transform(values).flatten()
                    self.tracks_df[feature] = np.clip(self.tracks_df[feature], 0, 1)
                elif config['method'] == 'log_minmax':
                    log_values = np.log1p(values)
                    scaler = MinMaxScaler()
                    self.tracks_df[feature] = scaler.fit_transform(log_values).flatten()
        
        logger.info(f"Normalized {len([f for f in numeric_features.keys() if f in self.tracks_df.columns])} features with ISRC + market features")
        return True
     
    def process_all(self):
        """Process all data steps"""
        logger.info("Starting complete data processing pipeline...")
        
        try:
            # Step 1: Load data
            if not self.load_data():
                logger.error("Failed to load data")
                return False
            
            # Step 2: Clean tracks data
            self.clean_tracks_data()
            
            # Step 3: Merge artist genres
            self.merge_artist_genres()
            
            # Step 4: Extract release year
            self.extract_release_year()
            
            # Step 5: Create genre features
            self.create_genre_features()
            
            # Step 6: ✅ Create ENHANCED language features (ISRC-based)
            self.create_language_features()
            
            # Step 7: Create additional features
            self.create_additional_features()
            
            # Step 8: Normalize features (including new ones)
            self.normalize_features()
            
            # Step 9: Save processed data
            output_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
            self.tracks_df.to_csv(output_path, index=False)
            
            feature_count = len(self.tracks_df.columns)
            genre_feature_count = len([col for col in self.tracks_df.columns if col.startswith('genre_')])
            cultural_feature_count = len([col for col in self.tracks_df.columns if col.startswith('is_') or col.startswith('market_') or col.startswith('isrc_')])
            
            logger.info(f"✅ Processing complete with ISRC-based cultural intelligence!")
            logger.info(f"   📊 {len(self.tracks_df)} tracks processed")
            logger.info(f"   🏷️ {feature_count} total features")
            logger.info(f"   🎨 {genre_feature_count} genre features")
            logger.info(f"   🌍 {cultural_feature_count} cultural intelligence features")
            logger.info(f"   💾 Saved to: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            return False

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()