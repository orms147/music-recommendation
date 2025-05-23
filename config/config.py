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

# Model weights
CONTENT_WEIGHT = 1.0
COLLABORATIVE_WEIGHT = 0.0  # Không sử dụng collaborative filtering
SEQUENCE_WEIGHT = 0.0  # Không sử dụng sequence/transition model

# Content features - CHỈ SỬ DỤNG REAL METADATA
CONTENT_FEATURES = [
    # Real Spotify metadata features only
    'popularity', 'duration_ms', 'explicit', 'release_year',
    'artist_popularity', 'total_tracks', 'track_number', 'disc_number',
    'markets_count', 'album_type_encoded', 'duration_category_encoded',
    'popularity_category_encoded', 'is_vietnamese', 'is_korean', 
    'is_japanese', 'is_spanish', 'has_collab', 'is_remix',
    'name_length', 'artist_frequency_norm'
    # LOẠI BỎ: 'danceability', 'energy', 'valence', 'tempo', etc.
]

# System settings
TRACKS_PER_QUERY = 30  # Số lượng bài hát thu thập cho mỗi query
MAX_RECOMMENDATIONS = 20  # Số lượng khuyến nghị tối đa

# Cài đặt thu thập dữ liệu
DEFAULT_TRACKS_PER_QUERY = 100  # Số lượng bài hát mặc định cho mỗi truy vấn
MAX_TRACKS_PER_QUERY = 800     # Giới hạn tối đa số bài hát mỗi truy vấn
MIN_TRACKS_PER_QUERY = 5       # Giới hạn tối thiểu số bài hát mỗi truy vấn
TRACKS_QUERY_STEP = 5          # Bước nhảy cho thanh trượt

# Cài đặt cho bộ dữ liệu lớn
LARGE_DATASET_DEFAULT_SIZE = 20000  # Kích thước mặc định cho tập dữ liệu lớn
LARGE_DATASET_BATCH_SIZE = 200      # Số lượng truy vấn mỗi lô
LARGE_DATASET_SAVE_INTERVAL = 2500  # Lưu sau mỗi bao nhiêu bài hát