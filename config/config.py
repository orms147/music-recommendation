import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Spotify API
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models', 'saved')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ✅ ALIGNED với actual data processor output
CONTENT_FEATURES = [
    # ✅ Basic Spotify metadata
    'popularity', 'duration_ms', 'explicit', 'release_year', 'track_age',
    'artist_popularity', 'markets_count', 'market_penetration',
    
    # ✅ ISRC-based cultural intelligence  
    'music_culture', 'isrc_country', 'cultural_confidence',
    'is_vietnamese', 'is_korean', 'is_japanese', 'is_chinese', 'is_western', 'is_spanish',
    'is_brazilian', 'is_indian', 'is_thai',
    'is_asia', 'is_europe', 'is_north_america', 'is_latin_america', 'is_oceania',
    'is_major_label', 'is_global_release', 'is_regional_release', 'is_local_release',
    
    # ✅ Genre features (dynamic based on actual data)
    'genre_pop', 'genre_rock', 'genre_hip_hop', 'genre_electronic', 'genre_ballad',
    'genre_v_pop', 'genre_k_pop', 'genre_j_pop', 'genre_c_pop', 'genre_mandopop',
    'genre_cantopop', 'genre_vinahouse', 'genre_anime', 'genre_trance',
    
    # ✅ Normalized features for ML
    'popularity_norm', 'artist_popularity_norm', 'duration_norm', 'markets_count_norm',
    'release_year_norm', 'track_age_norm', 'name_length_norm', 'is_playable_norm', 'explicit_norm',
    
    # Thêm các feature cho clustering
    'kmeans_cluster', 'hdbscan_cluster', 'hdbscan_outlier_score'
]

# Data collection settings
DEFAULT_TRACKS_PER_QUERY = 100
MAX_TRACKS_PER_QUERY = 500     # ✅ Reduced for realistic API limits
MIN_TRACKS_PER_QUERY = 50      # ✅ Minimum for decent diversity
TRACKS_QUERY_STEP = 25

# Large dataset settings
LARGE_DATASET_DEFAULT_SIZE = 5000   # ✅ More realistic target
LARGE_DATASET_BATCH_SIZE = 100      # ✅ Reasonable batch size
LARGE_DATASET_SAVE_INTERVAL = 1000  # ✅ Save more frequently

# Cultural Intelligence Configuration
CULTURAL_FEATURES = {
    'ISRC_COUNTRIES': {
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
    },
    'PRIORITY_ORDER': [
        'country',      # Highest priority: same country (from ISRC)
        'genre',        # Second priority: same genre
        'region',       # Third priority: same region
        'popularity',   # Fourth priority: similar popularity
        'recency'       # Fifth priority: similar release date
    ],
    'REGION_MAP': {
        # Khu vực Bắc Mỹ
        'US': 'north_america', 'CA': 'north_america',
        
        # Khu vực Châu Âu
        'GB': 'europe', 'DE': 'europe', 'FR': 'europe', 'IT': 'europe', 
        'ES': 'europe', 'NL': 'europe', 'SE': 'europe', 'NO': 'europe',
        'DK': 'europe', 'FI': 'europe', 'PT': 'europe', 'IE': 'europe',
        'BE': 'europe', 'AT': 'europe', 'CH': 'europe', 'PL': 'europe',
        
        # Khu vực Châu Á
        'JP': 'asia', 'KR': 'asia', 'CN': 'asia', 'HK': 'asia', 
        'TW': 'asia', 'VN': 'asia', 'TH': 'asia', 'MY': 'asia',
        'SG': 'asia', 'ID': 'asia', 'PH': 'asia', 'IN': 'asia',
        
        # Khu vực Mỹ Latinh
        'MX': 'latin_america', 'BR': 'latin_america', 'AR': 'latin_america',
        'CO': 'latin_america', 'CL': 'latin_america', 'PE': 'latin_america',
        
        # Khu vực Châu Đại Dương
        'AU': 'oceania', 'NZ': 'oceania'
    }
}

# Recommendation System Configuration
RECOMMENDATION_CONFIG = {
    'DIVERSITY': {
        'max_same_artist': 5,       # Maximum tracks from same artist
        'country_ratio': 0.6,       # 60% tracks from same country
        'region_ratio': 0.8,        # 80% tracks from same region
        'genre_consistency': True,  # Maintain genre consistency
    },
    'WEIGHTS': {
        'same_artist': 0.85,        # Same artist weight
        'same_country': 0.75,       # Same country weight
        'genre_similarity': 0.60,   # Genre similarity weight
        'same_region': 0.40,        # Same region weight
        'popularity': 0.30,         # Popularity weight
        'artist_popularity': 0.20,  # Artist popularity weight
        'release_recency': 0.15,    # Release recency weight
        'duration_similarity': 0.05 # Duration similarity weight
    }
}

# Cấu hình clustering
CLUSTERING_CONFIG = {
    'CLUSTERING_FEATURES': [
        'popularity_norm', 'artist_popularity_norm', 'duration_norm', 'markets_count_norm',
        'is_vietnamese', 'is_korean', 'is_japanese', 'is_chinese', 'is_western',
        'is_spanish', 'is_brazilian', 'is_indian', 'is_thai',
        'genre_v_pop', 'genre_k_pop', 'genre_j_pop', 'genre_c_pop', 'genre_mandopop',
        'genre_cantopop', 'genre_vinahouse', 'genre_anime', 'genre_trance'
    ],
    'KMEANS_N_CLUSTERS': 8,
    'KMEANS_RANDOM_STATE': 42,
    'HDBSCAN_MIN_CLUSTER_SIZE': 100,
    'HDBSCAN_MIN_SAMPLES': 10,
    'HDBSCAN_CLUSTER_SELECTION_EPSILON': 0.5
}

# Cấu hình hiệu suất
PERFORMANCE_CONFIG = {
    # Tối ưu hóa bộ nhớ
    'MAX_MEMORY_USAGE_GB': 4,           # Giới hạn bộ nhớ tối đa (GB)
    'USE_SPARSE_MATRICES': True,        # Sử dụng ma trận thưa cho dữ liệu lớn
    
    # Tối ưu hóa tốc độ
    'ENABLE_CACHING': True,             # Bật cache kết quả
    'CACHE_SIZE': 100,                  # Số lượng kết quả lưu trong cache
    'CACHE_EXPIRY_MINUTES': 60,         # Thời gian hết hạn cache
    
    # Tối ưu hóa tính toán
    'BATCH_SIZE': 1000,                 # Kích thước batch khi xử lý dataset lớn
    'USE_VECTORIZED_OPERATIONS': True,  # Sử dụng vectorized operations
    'SIMILARITY_THRESHOLD': 0.1,        # Ngưỡng similarity tối thiểu để xem xét
    
    # Tối ưu hóa clustering
    'FAST_CLUSTERING': True,            # Sử dụng phương pháp clustering nhanh
    'MAX_CLUSTERS': 20,                 # Số lượng cluster tối đa
}
