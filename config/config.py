import os
from pathlib import Path

# Đường dẫn thư mục
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Tạo thư mục nếu chưa tồn tại
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Cấu hình Spotify API
SPOTIFY_CLIENT_ID = ""  
SPOTIFY_CLIENT_SECRET = ""  

# Tham số mô hình
N_RECOMMENDATIONS = 10  # Số lượng bài hát gợi ý mặc định
TOP_K_SIMILAR_ITEMS = 100  # Số lượng bài hát tương tự để xem xét

# Danh sách đặc trưng có sẵn trong dữ liệu của bạn để sử dụng cho mô hình content-based
CONTENT_FEATURES = [
    'popularity', 'duration_ms', 'explicit', 'release_year'
]  # Loại bỏ các genre features

# Weights for hybrid model - tăng trọng số cho content do không có audio features
CONTENT_WEIGHT = 0.7
COLLABORATIVE_WEIGHT = 0.3
SEQUENCE_WEIGHT = 0.0  # Vô hiệu hóa vì không có đặc trưng âm thanh

# Tham số khác
RANDOM_STATE = 42  # Random seed for reproducibility

# Tham số huấn luyện
TRAIN_TEST_SPLIT = 0.8
N_EPOCHS = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 64