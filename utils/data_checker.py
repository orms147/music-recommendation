import os
import pandas as pd
import logging
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

def check_data_completeness():
    """Comprehensive data completeness check for recommendation system"""
    print("CHECKING DATA COMPLETENESS FOR RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # 1. Check raw data files
    print("\nRAW DATA FILES:")
    raw_files = {
        'spotify_tracks.csv': 'Track metadata from Spotify API',
        'enriched_tracks.csv': 'Enhanced track data with derived features', 
        'artist_genres.csv': 'Artist genres and popularity data'
    }
    
    raw_data_status = {}
    
    for filename, description in raw_files.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                size = len(df)
                raw_data_status[filename] = {'exists': True, 'size': size, 'df': df}
                print(f"  [OK] {filename}: {size:,} records - {description}")
            except Exception as e:
                raw_data_status[filename] = {'exists': False, 'error': str(e)}
                print(f"  [ERROR] {filename}: {str(e)}")
        else:
            raw_data_status[filename] = {'exists': False, 'error': 'File not found'}
            print(f"  [MISSING] {filename}: NOT FOUND - {description}")
    
    # 2. Check processed data files
    print("\nPROCESSED DATA FILES:")
    processed_files = {
        'track_features.csv': 'Final processed features for ML models'
    }
    
    processed_data_status = {}
    
    for filename, description in processed_files.items():
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                size = len(df)
                processed_data_status[filename] = {'exists': True, 'size': size, 'df': df}
                print(f"  [OK] {filename}: {size:,} records - {description}")
            except Exception as e:
                processed_data_status[filename] = {'exists': False, 'error': str(e)}
                print(f"  [ERROR] {filename}: {str(e)}")
        else:
            processed_data_status[filename] = {'exists': False, 'error': 'File not found'}
            print(f"  [MISSING] {filename}: NOT FOUND - {description}")
    
    # 3. Analyze track data quality
    print("\nTRACK DATA ANALYSIS:")
    
    # Use the best available tracks data
    tracks_df = None
    if processed_data_status.get('track_features.csv', {}).get('exists'):
        tracks_df = processed_data_status['track_features.csv']['df']
        print(f"  Using processed data: {len(tracks_df):,} tracks")
    elif raw_data_status.get('enriched_tracks.csv', {}).get('exists'):
        tracks_df = raw_data_status['enriched_tracks.csv']['df']
        print(f"  Using enriched raw data: {len(tracks_df):,} tracks")
    elif raw_data_status.get('spotify_tracks.csv', {}).get('exists'):
        tracks_df = raw_data_status['spotify_tracks.csv']['df']
        print(f"  Using basic raw data: {len(tracks_df):,} tracks")
    
    if tracks_df is not None:
        # Essential columns check
        essential_cols = ['id', 'name', 'artist']
        missing_essential = [col for col in essential_cols if col not in tracks_df.columns]
        
        if missing_essential:
            print(f"  [ERROR] Missing essential columns: {missing_essential}")
        else:
            print(f"  [OK] All essential columns present")
        
        # Core Spotify metadata check
        spotify_metadata = {
            'popularity': 'Track popularity (0-100)',
            'duration_ms': 'Track duration in milliseconds',
            'explicit': 'Explicit content flag',
            'release_year': 'Release year',
            'artist_popularity': 'Artist popularity (0-100)'
        }
        
        print(f"\n  CORE SPOTIFY METADATA AVAILABILITY:")
        available_metadata = 0
        for col, desc in spotify_metadata.items():
            if col in tracks_df.columns:
                non_null_count = tracks_df[col].notna().sum()
                coverage = (non_null_count / len(tracks_df)) * 100
                print(f"    [OK] {col}: {non_null_count:,}/{len(tracks_df):,} ({coverage:.1f}%) - {desc}")
                available_metadata += 1
            else:
                print(f"    [MISSING] {col}: NOT FOUND - {desc}")
        
        print(f"  Metadata coverage: {available_metadata}/{len(spotify_metadata)} ({available_metadata/len(spotify_metadata)*100:.1f}%)")
        
        # Language features check
        language_features = ['is_vietnamese', 'is_korean', 'is_japanese', 'is_spanish', 'is_chinese']
        print(f"\n  LANGUAGE FEATURES:")
        available_languages = 0
        for lang in language_features:
            if lang in tracks_df.columns:
                count = tracks_df[lang].sum()
                print(f"    [OK] {lang}: {count:,} tracks")
                available_languages += 1
            else:
                print(f"    [MISSING] {lang}: NOT FOUND")
        
        # Genre features check
        genre_cols = [col for col in tracks_df.columns if col.startswith('genre_')]
        print(f"\n  GENRE FEATURES:")
        print(f"    Total genre features: {len(genre_cols)}")
        if len(genre_cols) > 0:
            # Show top genres by track count
            genre_counts = {}
            for col in genre_cols[:10]:  # Top 10
                count = tracks_df[col].sum()
                genre_name = col.replace('genre_', '').replace('_', ' ').title()
                genre_counts[genre_name] = count
            
            sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
            for genre, count in sorted_genres:
                print(f"      {genre}: {count:,} tracks")
        
        # Data quality metrics
        print(f"\n  DATA QUALITY METRICS:")
        total_tracks = len(tracks_df)
        unique_tracks = tracks_df['id'].nunique() if 'id' in tracks_df.columns else 0
        unique_artists = tracks_df['artist'].nunique() if 'artist' in tracks_df.columns else 0
        
        print(f"    Total tracks: {total_tracks:,}")
        print(f"    Unique track IDs: {unique_tracks:,}")
        print(f"    Unique artists: {unique_artists:,}")
        print(f"    Avg tracks per artist: {total_tracks/unique_artists:.1f}" if unique_artists > 0 else "    Avg tracks per artist: N/A")
        
        # Duplicates check
        if 'id' in tracks_df.columns:
            duplicates = total_tracks - unique_tracks
            print(f"    Duplicate tracks: {duplicates:,} ({duplicates/total_tracks*100:.1f}%)")
        
        # Missing data analysis
        print(f"\n  MISSING DATA ANALYSIS:")
        missing_data = tracks_df.isnull().sum()
        critical_missing = missing_data[missing_data > 0]
        
        if len(critical_missing) > 0:
            for col, missing_count in critical_missing.head(10).items():
                missing_pct = (missing_count / len(tracks_df)) * 100
                print(f"    {col}: {missing_count:,} missing ({missing_pct:.1f}%)")
        else:
            print(f"    [OK] No missing data detected")
    
    # 4. Artist data analysis
    print("\nARTIST DATA ANALYSIS:")
    if raw_data_status.get('artist_genres.csv', {}).get('exists'):
        artist_df = raw_data_status['artist_genres.csv']['df']
        print(f"  Artist records: {len(artist_df):,}")
        
        if 'genres' in artist_df.columns or 'artist_genres' in artist_df.columns:
            genre_col = 'artist_genres' if 'artist_genres' in artist_df.columns else 'genres'
            artists_with_genres = artist_df[genre_col].notna().sum()
            print(f"  Artists with genres: {artists_with_genres:,} ({artists_with_genres/len(artist_df)*100:.1f}%)")
        
        if 'artist_popularity' in artist_df.columns:
            pop_available = artist_df['artist_popularity'].notna().sum()
            print(f"  Artists with popularity data: {pop_available:,} ({pop_available/len(artist_df)*100:.1f}%)")
    else:
        print(f"  [ERROR] No artist data available")
    
    # 5. Recommendation readiness assessment
    print("\nRECOMMENDATION SYSTEM READINESS:")
    
    readiness_score = 0
    max_score = 7
    
    # Check 1: Basic track data
    if tracks_df is not None and len(tracks_df) >= 1000:
        print(f"  [OK] Sufficient track data: {len(tracks_df):,} tracks (minimum: 1,000)")
        readiness_score += 1
    else:
        track_count = len(tracks_df) if tracks_df is not None else 0
        print(f"  [FAIL] Insufficient track data: {track_count:,} tracks (minimum: 1,000)")
    
    # Check 2: Essential metadata
    if tracks_df is not None and all(col in tracks_df.columns for col in ['id', 'name', 'artist']):
        print(f"  [OK] Essential metadata available")
        readiness_score += 1
    else:
        print(f"  [FAIL] Missing essential metadata")
    
    # Check 3: Popularity data
    if tracks_df is not None and 'popularity' in tracks_df.columns:
        pop_coverage = tracks_df['popularity'].notna().sum() / len(tracks_df)
        if pop_coverage >= 0.8:
            print(f"  [OK] Good popularity data coverage: {pop_coverage*100:.1f}%")
            readiness_score += 1
        else:
            print(f"  [WARN] Low popularity data coverage: {pop_coverage*100:.1f}%")
    else:
        print(f"  [FAIL] No popularity data")
    
    # Check 4: Genre features
    genre_count = len([col for col in tracks_df.columns if col.startswith('genre_')]) if tracks_df is not None else 0
    if genre_count >= 10:
        print(f"  [OK] Good genre feature coverage: {genre_count} genre features")
        readiness_score += 1
    else:
        print(f"  [FAIL] Insufficient genre features: {genre_count} (minimum: 10)")
    
    # Check 5: Language features
    lang_count = len([col for col in tracks_df.columns if col.startswith('is_')]) if tracks_df is not None else 0
    if lang_count >= 3:
        print(f"  [OK] Language features available: {lang_count} language features")
        readiness_score += 1
    else:
        print(f"  [FAIL] Limited language features: {lang_count}")
    
    # Check 6: Artist diversity
    if tracks_df is not None and 'artist' in tracks_df.columns:
        unique_artists = tracks_df['artist'].nunique()
        if unique_artists >= 500:
            print(f"  [OK] Good artist diversity: {unique_artists:,} unique artists")
            readiness_score += 1
        else:
            print(f"  [FAIL] Limited artist diversity: {unique_artists:,} (minimum: 500)")
    
    # Check 7: Data quality
    if tracks_df is not None:
        missing_pct = (tracks_df.isnull().sum().sum() / (len(tracks_df) * len(tracks_df.columns))) * 100
        if missing_pct < 20:
            print(f"  [OK] Good data quality: {missing_pct:.1f}% missing data")
            readiness_score += 1
        else:
            print(f"  [WARN] Data quality concerns: {missing_pct:.1f}% missing data")
    
    # Final assessment
    print(f"\nOVERALL READINESS SCORE: {readiness_score}/{max_score} ({readiness_score/max_score*100:.1f}%)")
    
    if readiness_score >= 6:
        print(f"  EXCELLENT: Ready for production recommendation system!")
    elif readiness_score >= 4:
        print(f"  GOOD: Ready for recommendation system with minor improvements needed")
    elif readiness_score >= 2:
        print(f"  FAIR: Basic recommendation possible, significant improvements recommended")
    else:
        print(f"  POOR: Need more data collection before building recommendation system")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if readiness_score < max_score:
        if tracks_df is None or len(tracks_df) < 1000:
            print(f"  - Collect more track data (target: 5,000+ tracks)")
        
        if genre_count < 10:
            print(f"  - Improve genre feature extraction from artist data")
            
        if tracks_df is not None and tracks_df['artist'].nunique() < 500:
            print(f"  - Increase artist diversity in dataset")
            
        print(f"  - Run data processing pipeline to create missing features")
        print(f"  - Consider fetching additional data using large dataset mode")
    
    return {
        'readiness_score': readiness_score,
        'max_score': max_score,
        'tracks_count': len(tracks_df) if tracks_df is not None else 0,
        'raw_data_status': raw_data_status,
        'processed_data_status': processed_data_status
    }

if __name__ == "__main__":
    check_data_completeness()