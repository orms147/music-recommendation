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
    'is_major_label', 'is_global_release', 'is_regional_release', 'is_local_release',
    
    # ✅ Genre features (dynamic based on actual data)
    'genre_pop', 'genre_rock', 'genre_hip_hop', 'genre_electronic', 'genre_ballad',
    
    # ✅ Normalized features for ML
    'popularity_norm', 'artist_popularity_norm', 'duration_norm'
]

# Data collection settings
DEFAULT_TRACKS_PER_QUERY = 100
MAX_TRACKS_PER_QUERY = 300     # ✅ Reduced for realistic API limits
MIN_TRACKS_PER_QUERY = 50      # ✅ Minimum for decent diversity
TRACKS_QUERY_STEP = 25

# Large dataset settings
LARGE_DATASET_DEFAULT_SIZE = 5000   # ✅ More realistic target
LARGE_DATASET_BATCH_SIZE = 100      # ✅ Reasonable batch size
LARGE_DATASET_SAVE_INTERVAL = 1000  # ✅ Save more frequently