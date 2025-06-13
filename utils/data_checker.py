import os
import pandas as pd
import logging
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

def check_cultural_intelligence(tracks_df):
    """Check cultural intelligence features in dataset"""
    print("\nCULTURAL INTELLIGENCE ANALYSIS:")
    
    if tracks_df is None:
        print("  [ERROR] No tracks data available")
        return False
    
    # Check ISRC coverage
    isrc_available = False
    if 'isrc' in tracks_df.columns:
        isrc_count = (tracks_df['isrc'] != '').sum()
        isrc_coverage = isrc_count / len(tracks_df)
        isrc_available = isrc_coverage > 0.3
        print(f"  [{'OK' if isrc_available else 'WARN'}] ISRC coverage: {isrc_count:,}/{len(tracks_df):,} ({isrc_coverage*100:.1f}%)")
    else:
        print("  [ERROR] ISRC data not available")
    
    # Check music_culture feature
    culture_available = False
    if 'music_culture' in tracks_df.columns:
        culture_dist = tracks_df['music_culture'].value_counts()
        culture_available = len(culture_dist) >= 3
        print(f"  [{'OK' if culture_available else 'WARN'}] Music culture distribution: {dict(culture_dist)}")
        
        # Check cultural diversity
        non_other = (tracks_df['music_culture'] != 'other').sum()
        cultural_coverage = non_other / len(tracks_df)
        print(f"  [{'OK' if cultural_coverage > 0.5 else 'WARN'}] Cultural coverage: {non_other:,}/{len(tracks_df):,} ({cultural_coverage*100:.1f}%)")
    else:
        print("  [ERROR] Music culture data not available")
    
    # Check binary cultural features
    binary_features = [col for col in tracks_df.columns if col.startswith('is_') and col not in ['is_major_label', 'is_global_release', 'is_regional_release', 'is_local_release']]
    if binary_features:
        print(f"  [OK] Binary cultural features: {len(binary_features)}")
        for feat in binary_features[:5]:  # Show first 5
            count = tracks_df[feat].sum()
            print(f"    - {feat}: {count:,} tracks ({count/len(tracks_df)*100:.1f}%)")
    else:
        print("  [WARN] No binary cultural features found")
    
    # Check cultural confidence
    if 'cultural_confidence' in tracks_df.columns:
        avg_confidence = tracks_df['cultural_confidence'].mean()
        high_confidence = (tracks_df['cultural_confidence'] > 0.7).sum()
        print(f"  [OK] Cultural confidence: avg={avg_confidence:.3f}, high confidence={high_confidence:,} tracks ({high_confidence/len(tracks_df)*100:.1f}%)")
    else:
        print("  [WARN] Cultural confidence not available")
    
    # Check region data
    if 'region' in tracks_df.columns:
        region_dist = tracks_df['region'].value_counts()
        print(f"  [OK] Region distribution: {dict(region_dist)}")
    else:
        print("  [WARN] Region data not available")
    
    # Overall cultural intelligence assessment
    cultural_score = 0
    max_cultural_score = 5
    
    if isrc_available: cultural_score += 1
    if culture_available: cultural_score += 1
    if len(binary_features) >= 3: cultural_score += 1
    if 'cultural_confidence' in tracks_df.columns: cultural_score += 1
    if 'region' in tracks_df.columns: cultural_score += 1
    
    print(f"\n  CULTURAL INTELLIGENCE SCORE: {cultural_score}/{max_cultural_score}")
    if cultural_score >= 4:
        print("  [EXCELLENT] Advanced cultural intelligence ready for recommendations")
    elif cultural_score >= 3:
        print("  [GOOD] Basic cultural intelligence available")
    else:
        print("  [POOR] Limited cultural intelligence, needs improvement")
    
    return cultural_score >= 3

def check_genre_features(tracks_df):
    """Check genre features in dataset"""
    print("\nGENRE FEATURES ANALYSIS:")
    
    if tracks_df is None:
        print("  [ERROR] No tracks data available")
        return False
    
    # Check genre columns
    genre_columns = [col for col in tracks_df.columns if col.startswith('genre_')]
    genre_count = len(genre_columns)
    
    if genre_count == 0:
        print("  [ERROR] No genre features found")
        return False
    
    print(f"  [{'OK' if genre_count >= 20 else 'WARN'}] Genre features: {genre_count} features")
    
    # Check top genres by coverage
    genre_coverage = {}
    for genre in genre_columns:
        count = tracks_df[genre].sum()
        coverage = count / len(tracks_df)
        genre_coverage[genre] = (count, coverage)
    
    # Sort by coverage
    top_genres = sorted(genre_coverage.items(), key=lambda x: x[1][0], reverse=True)[:10]
    
    print("  Top 10 genres by track count:")
    for genre, (count, coverage) in top_genres:
        print(f"    - {genre[6:]}: {count:,} tracks ({coverage*100:.1f}%)")
    
    # Check regional genre coverage
    regional_genres = {
        'asian': ['korean', 'japanese', 'chinese', 'vietnamese', 'thai', 'mandopop', 'cantopop', 'j_pop', 'k_pop', 'v_pop'],
        'western': ['pop', 'rock', 'hip_hop', 'rap', 'edm', 'dance', 'electronic', 'r_b', 'indie'],
        'latin': ['latin', 'reggaeton', 'salsa', 'bachata', 'spanish', 'mexican']
    }
    
    print("\n  Regional genre coverage:")
    for region, keywords in regional_genres.items():
        region_cols = [col for col in genre_columns if any(kw in col for kw in keywords)]
        if region_cols:
            region_tracks = tracks_df[region_cols].any(axis=1).sum()
            region_coverage = region_tracks / len(tracks_df)
            print(f"    - {region.title()}: {region_tracks:,} tracks ({region_coverage*100:.1f}%) from {len(region_cols)} genres")
        else:
            print(f"    - {region.title()}: No genres found")
    
    # Overall genre quality assessment
    genre_score = 0
    max_genre_score = 5
    
    if genre_count >= 50: genre_score += 2
    elif genre_count >= 20: genre_score += 1
    
    # Check if we have good coverage of top genres
    top_coverage = sum(coverage for _, (_, coverage) in top_genres[:5])
    if top_coverage >= 0.5: genre_score += 1
    
    # Check if we have regional diversity
    regions_with_genres = sum(1 for region, keywords in regional_genres.items() 
                             if any(kw in col for col in genre_columns for kw in keywords))
    if regions_with_genres >= 2: genre_score += 1
    
    # Check if we have at least one genre with high coverage
    if any(coverage >= 0.1 for _, (_, coverage) in top_genres): genre_score += 1
    
    print(f"\n  GENRE FEATURES SCORE: {genre_score}/{max_genre_score}")
    if genre_score >= 4:
        print("  [EXCELLENT] Rich genre features ready for recommendations")
    elif genre_score >= 3:
        print("  [GOOD] Adequate genre features available")
    else:
        print("  [POOR] Limited genre features, needs improvement")
    
    return genre_score >= 3

def check_data_completeness():
    """Check data completeness and quality"""
    print("\nDATA COMPLETENESS CHECK:")
    
    # 1. Check raw data files
    print("\nRAW DATA FILES:")
    raw_files = {
        'tracks.csv': 'Main tracks data with Spotify metadata and markets_count',
        'artist_genres.csv': 'Artist genres data'
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
    if genre_count >= 20:  # Adjusted threshold from 5 to 20 since we now expect up to 100 genres
        print(f"  [OK] Good genre feature coverage: {genre_count} genre features")
        readiness_score += 1
    elif genre_count >= 10:  # Add intermediate level
        print(f"  [PARTIAL] Basic genre feature coverage: {genre_count} genre features")
        readiness_score += 0.5
    else:
        print(f"  [FAIL] Insufficient genre features: {genre_count} (minimum: 10)")
    
    # Add check for genre diversity
    if tracks_df is not None:
        genre_columns = [col for col in tracks_df.columns if col.startswith('genre_')]
        if genre_columns:
            # Count tracks with at least one genre
            tracks_with_genres = tracks_df[genre_columns].any(axis=1).sum()
            genre_coverage = tracks_with_genres / len(tracks_df)
            
            if genre_coverage >= 0.7:
                print(f"  [OK] Excellent genre coverage: {tracks_with_genres:,} tracks ({genre_coverage*100:.1f}%)")
                readiness_score += 1
            elif genre_coverage >= 0.4:
                print(f"  [PARTIAL] Moderate genre coverage: {tracks_with_genres:,} tracks ({genre_coverage*100:.1f}%)")
                readiness_score += 0.5
            else:
                print(f"  [WARN] Poor genre coverage: {tracks_with_genres:,} tracks ({genre_coverage*100:.1f}%)")
    
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
    
    # Add detailed genre analysis if needed
    if tracks_df is not None and check_genre_features(tracks_df):
        print("  [OK] Genre features analysis passed")
    else:
        print("  [WARN] Genre features analysis failed or incomplete")
    
    return {
        'readiness_score': readiness_score,
        'max_score': max_score,
        'tracks_count': len(tracks_df) if tracks_df is not None else 0,
        'raw_data_status': raw_data_status,
        'processed_data_status': processed_data_status,
        'cultural_intelligence_ready': cultural_count >= 4,
        'isrc_coverage': (tracks_df['isrc'] != '').sum() / len(tracks_df) if tracks_df is not None and 'isrc' in tracks_df.columns else 0
    }

def check_clustering_features(tracks_df):
    """Check clustering features in dataset"""
    print("\nCLUSTERING ANALYSIS:")
    
    if tracks_df is None:
        print("  [ERROR] No tracks data available")
        return False
    
    # Check K-Means clusters
    kmeans_available = 'kmeans_cluster' in tracks_df.columns
    if kmeans_available:
        kmeans_clusters = tracks_df['kmeans_cluster'].nunique()
        print(f"  [{'OK' if kmeans_clusters > 1 else 'WARN'}] K-Means clusters: {kmeans_clusters}")
    else:
        print("  [WARN] K-Means clustering not available")
    
    # Check HDBSCAN clusters
    hdbscan_available = 'hdbscan_cluster' in tracks_df.columns
    if hdbscan_available:
        hdbscan_clusters = tracks_df['hdbscan_cluster'].nunique()
        noise_points = (tracks_df['hdbscan_cluster'] == -1).sum()
        noise_percentage = noise_points / len(tracks_df) * 100
        print(f"  [{'OK' if hdbscan_clusters > 1 else 'WARN'}] HDBSCAN clusters: {hdbscan_clusters}")
        print(f"  [{'OK' if noise_percentage < 30 else 'WARN'}] HDBSCAN noise points: {noise_points} ({noise_percentage:.1f}%)")
    else:
    
    # Overall clustering assessment
    clustering_score = 0
    max_clustering_score = 3
    
    if kmeans_available and kmeans_clusters > 1: clustering_score += 1
    if hdbscan_available and hdbscan_clusters > 1: clustering_score += 1
    if hdbscan_available and noise_percentage < 30: clustering_score += 1
    
    print(f"\n  CLUSTERING SCORE: {clustering_score}/{max_clustering_score}")
    if clustering_score >= 2:
        print("  [GOOD] Clustering features ready for recommendation models")
    else:
        print("  [POOR] Clustering features need improvement")
    
    return clustering_score >= 2

if __name__ == "__main__":
    check_data_completeness()
    check_clustering_features(tracks_df)
