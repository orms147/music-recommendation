import os
from pathlib import Path
import logging

# Đường dẫn thư mục
BASE_DIR = Path(__file__).parent.parent.absolute()
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Đảm bảo các thư mục tồn tại
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Cấu hình Spotify API
# Đọc từ môi trường hoặc sử dụng giá trị mặc định cho môi trường dev
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID', '')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', '')

# Function to check credentials are set
def check_spotify_credentials():
    """Check if Spotify credentials are set and valid"""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logging.error("SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not set in environment")
        return False
    return True

# API configuration
API_RATE_LIMIT = 0.5  # seconds between API calls
API_MAX_RETRIES = 5   # maximum number of retries for API calls
API_BATCH_SIZE = 50   # items per batch for API calls

# Cấu hình trọng số cho mô hình kết hợp
CONTENT_WEIGHT = 0.7  # Content-based recommendation weight
COLLABORATIVE_WEIGHT = 0.0  # Not using collaborative filtering
SEQUENCE_WEIGHT = 0.3  # Transition-based recommendation weight

# Danh sách đặc trưng cho mô hình content-based
CONTENT_FEATURES = [
    # Đặc trưng cơ bản
    'popularity', 'duration_ms', 'explicit', 'release_year',
    
    # Đặc trưng âm thanh
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Fallback mode - if True, system will try to work even with missing data
FALLBACK_MODE = True

# Model configuration
MODEL_PATH = os.path.join(MODELS_DIR, 'hybrid_recommender.pkl')