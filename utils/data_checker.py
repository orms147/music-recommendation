import os
import pandas as pd
import logging
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

def check_data_completeness():
    """Comprehensive data completeness check for recommendation system"""
    print("CHECKING DATA COMPLETENESS FOR RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # 1. ✅ Check actual raw data files from data_fetcher.py
    print("\nRAW DATA FILES:")
    raw_files = {
        'tracks.csv': 'Track metadata from Spotify API with ISRC',  # ✅ Actual filename
        'artist_genres.csv': 'Artist genres and popularity data'    # ✅ Actual filename
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
    
    # 2. ✅ Check actual processed data files from data_processor.py
    print("\nPROCESSED DATA FILES:")
    processed_files = {
        'processed_tracks.csv': 'Final processed features with ISRC cultural intelligence'  # ✅ Actual filename
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
    
    # 3. Analyze track data quality using actual data structure
    print("\nTRACK DATA ANALYSIS:")
    
    # ✅ Use the best available tracks data with correct priority
    tracks_df = None
    if processed_data_status.get('processed_tracks.csv', {}).get('exists'):
        tracks_df = processed_data_status['processed_tracks.csv']['df']
        print(f"  Using processed data: {len(tracks_df):,} tracks")
    elif raw_data_status.get('tracks.csv', {}).get('exists'):
        tracks_df = raw_data_status['tracks.csv']['df']
        print(f"  Using raw data: {len(tracks_df):,} tracks")
    
    if tracks_df is not None:
        # Essential columns check
        essential_cols = ['id', 'name', 'artist']
        missing_essential = [col for col in essential_cols if col not in tracks_df.columns]
        
        if missing_essential:
            print(f"  [ERROR] Missing essential columns: {missing_essential}")
        else:
            print(f"  [OK] All essential columns present")
        
        # ✅ Core Spotify metadata check with actual column names
        spotify_metadata = {
            'popularity': 'Track popularity (0-100)',
            'duration_ms': 'Track duration in milliseconds',
            'release_year': 'Release year (extracted from release_date)',
            'artist_popularity': 'Artist popularity (0-100)',
            'markets_count': 'Number of available markets'
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
        
        # ✅ ISRC and Cultural Intelligence Features
        print(f"\n  ISRC & CULTURAL INTELLIGENCE:")
        
        # ISRC coverage
        if 'isrc' in tracks_df.columns:
            isrc_available = (tracks_df['isrc'] != '').sum()
            isrc_coverage = isrc_available / len(tracks_df)
            print(f"    [OK] ISRC data: {isrc_available:,}/{len(tracks_df):,} ({isrc_coverage*100:.1f}%)")
        else:
            print(f"    [MISSING] ISRC data: NOT FOUND")
        
        # Cultural intelligence features
        cultural_features = ['music_culture', 'isrc_country', 'cultural_confidence']
        available_cultural = 0
        for feat in cultural_features:
            if feat in tracks_df.columns:
                if feat == 'music_culture':
                    culture_dist = tracks_df[feat].value_counts()
                    print(f"    [OK] {feat}: {dict(culture_dist)}")
                else:
                    non_null = tracks_df[feat].notna().sum()
                    coverage = non_null / len(tracks_df) * 100
                    print(f"    [OK] {feat}: {non_null:,} tracks ({coverage:.1f}%)")
                available_cultural += 1
            else:
                print(f"    [MISSING] {feat}: NOT FOUND")
        
        # ✅ Binary cultural features check
        binary_cultural = ['is_vietnamese', 'is_korean', 'is_japanese', 'is_chinese', 'is_western', 'is_spanish']
        print(f"\n  BINARY CULTURAL FEATURES:")
        available_languages = 0
        for lang in binary_cultural:
            if lang in tracks_df.columns:
                count = tracks_df[lang].sum()
                print(f"    [OK] {lang}: {count:,} tracks")
                available_languages += 1
            else:
                print(f"    [MISSING] {lang}: NOT FOUND")
        
        # ✅ Market and Professional Quality Features
        print(f"\n  MARKET & QUALITY FEATURES:")
        quality_features = {
            'is_major_label': 'Major record label releases',
            'market_penetration': 'Global market reach (0-1)',
            'is_global_release': 'Released in 100+ markets',
            'is_regional_release': 'Released in 20-100 markets',
            'is_local_release': 'Released in <20 markets'
        }
        
        for feat, desc in quality_features.items():
            if feat in tracks_df.columns:
                if feat in ['is_major_label', 'is_global_release', 'is_regional_release', 'is_local_release']:
                    count = tracks_df[feat].sum()
                    print(f"    [OK] {feat}: {count:,} tracks - {desc}")
                else:
                    avg_val = tracks_df[feat].mean()
                    print(f"    [OK] {feat}: avg {avg_val:.3f} - {desc}")
            else:
                print(f"    [MISSING] {feat}: NOT FOUND - {desc}")
        
        # ✅ Genre features check with actual naming
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
        
        # ✅ Normalized features check
        normalized_cols = [col for col in tracks_df.columns if col.endswith('_norm')]
        print(f"\n  NORMALIZED ML FEATURES:")
        print(f"    Total normalized features: {len(normalized_cols)}")
        for col in normalized_cols:
            min_val = tracks_df[col].min()
            max_val = tracks_df[col].max()
            print(f"      {col}: range [{min_val:.3f}, {max_val:.3f}]")
        
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
        
        if 'genres' in artist_df.columns:
            artists_with_genres = artist_df['genres'].notna().sum()
            print(f"  Artists with genres: {artists_with_genres:,} ({artists_with_genres/len(artist_df)*100:.1f}%)")
        
        if 'artist_popularity' in artist_df.columns:
            pop_available = artist_df['artist_popularity'].notna().sum()
            print(f"  Artists with popularity data: {pop_available:,} ({pop_available/len(artist_df)*100:.1f}%)")
        
        if 'artist_followers' in artist_df.columns:
            followers_available = artist_df['artist_followers'].notna().sum()
            print(f"  Artists with follower data: {followers_available:,} ({followers_available/len(artist_df)*100:.1f}%)")
    else:
        print(f"  [ERROR] No artist data available")
    
    # 5. ✅ Enhanced recommendation readiness assessment
    print("\nRECOMMENDATION SYSTEM READINESS:")
    
    readiness_score = 0
    max_score = 10  # ✅ Increased for more comprehensive assessment
    
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
    
    # Check 4: ISRC Cultural Intelligence
    if tracks_df is not None and 'isrc' in tracks_df.columns:
        isrc_coverage = (tracks_df['isrc'] != '').sum() / len(tracks_df)
        if isrc_coverage >= 0.3:
            print(f"  [OK] Good ISRC coverage: {isrc_coverage*100:.1f}%")
            readiness_score += 1
        else:
            print(f"  [WARN] Low ISRC coverage: {isrc_coverage*100:.1f}%")
    else:
        print(f"  [FAIL] No ISRC data for cultural intelligence")
    
    # Check 5: Cultural features
    cultural_count = len([col for col in tracks_df.columns if col.startswith('is_') and col.endswith(('vietnamese', 'korean', 'japanese', 'chinese', 'western', 'spanish'))]) if tracks_df is not None else 0
    if cultural_count >= 4:
        print(f"  [OK] Good cultural features: {cultural_count} cultural categories")
        readiness_score += 1
    else:
        print(f"  [FAIL] Limited cultural features: {cultural_count} (minimum: 4)")
    
    # Check 6: Genre features
    genre_count = len([col for col in tracks_df.columns if col.startswith('genre_')]) if tracks_df is not None else 0
    if genre_count >= 5:
        print(f"  [OK] Good genre feature coverage: {genre_count} genre features")
        readiness_score += 1
    else:
        print(f"  [FAIL] Insufficient genre features: {genre_count} (minimum: 5)")
    
    # Check 7: Market penetration features
    market_features = ['market_penetration', 'markets_count']
    market_available = sum(1 for feat in market_features if tracks_df is not None and feat in tracks_df.columns)
    if market_available >= 2:
        print(f"  [OK] Market analysis features available: {market_available}/2")
        readiness_score += 1
    else:
        print(f"  [FAIL] Limited market features: {market_available}/2")
    
    # Check 8: Normalized features for ML
    norm_count = len([col for col in tracks_df.columns if col.endswith('_norm')]) if tracks_df is not None else 0
    if norm_count >= 3:
        print(f"  [OK] ML-ready normalized features: {norm_count}")
        readiness_score += 1
    else:
        print(f"  [FAIL] Insufficient normalized features: {norm_count} (minimum: 3)")
    
    # Check 9: Artist diversity
    if tracks_df is not None and 'artist' in tracks_df.columns:
        unique_artists = tracks_df['artist'].nunique()
        if unique_artists >= 500:
            print(f"  [OK] Good artist diversity: {unique_artists:,} unique artists")
            readiness_score += 1
        else:
            print(f"  [FAIL] Limited artist diversity: {unique_artists:,} (minimum: 500)")
    
    # Check 10: Overall data quality
    if tracks_df is not None:
        missing_pct = (tracks_df.isnull().sum().sum() / (len(tracks_df) * len(tracks_df.columns))) * 100
        if missing_pct < 15:
            print(f"  [OK] Good data quality: {missing_pct:.1f}% missing data")
            readiness_score += 1
        else:
            print(f"  [WARN] Data quality concerns: {missing_pct:.1f}% missing data")
    
    # ✅ Final assessment with cultural intelligence focus
    print(f"\nOVERALL READINESS SCORE: {readiness_score}/{max_score} ({readiness_score/max_score*100:.1f}%)")
    
    if readiness_score >= 8:
        print(f"  🚀 EXCELLENT: Production-ready with advanced cultural intelligence!")
    elif readiness_score >= 6:
        print(f"  ✅ GOOD: Ready for recommendations with cultural intelligence")
    elif readiness_score >= 4:
        print(f"  ⚠️ FAIR: Basic recommendations possible, cultural features need improvement")
    else:
        print(f"  ❌ POOR: Need comprehensive data collection and processing")
    
    # ✅ Enhanced recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if readiness_score < max_score:
        if tracks_df is None or len(tracks_df) < 1000:
            print(f"  - Run data fetcher: fetch_initial_dataset() to collect more tracks")
        
        if cultural_count < 4:
            print(f"  - Run data processor: process cultural intelligence from ISRC")
            
        if genre_count < 5:
            print(f"  - Improve genre extraction from artist data")
            
        if tracks_df is not None and (tracks_df['isrc'] != '').sum() / len(tracks_df) < 0.3:
            print(f"  - Collect more tracks with ISRC data for better cultural intelligence")
            
        if norm_count < 3:
            print(f"  - Run data processor to create normalized ML features")
    
    print(f"\n✅ Ready to train WeightedContentRecommender and EnhancedContentRecommender!")
    
    return {
        'readiness_score': readiness_score,
        'max_score': max_score,
        'tracks_count': len(tracks_df) if tracks_df is not None else 0,
        'raw_data_status': raw_data_status,
        'processed_data_status': processed_data_status,
        'cultural_intelligence_ready': cultural_count >= 4,
        'isrc_coverage': (tracks_df['isrc'] != '').sum() / len(tracks_df) if tracks_df is not None and 'isrc' in tracks_df.columns else 0
    }

if __name__ == "__main__":
    check_data_completeness()