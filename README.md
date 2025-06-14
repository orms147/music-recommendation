# Hệ thống Đề xuất Âm nhạc

Hệ thống đề xuất âm nhạc thông minh sử dụng dữ liệu từ Spotify API, tập trung vào phương pháp dựa trên metadata và ISRC Cultural Intelligence để đưa ra các gợi ý bài hát phù hợp với sở thích người dùng.

## Tính năng chính

- Thu thập dữ liệu âm nhạc từ Spotify API
- Hỗ trợ tập dữ liệu lớn (đã thử nghiệm với ~100,000 bài hát)
- Xử lý và phân tích metadata của bài hát
- Phát hiện văn hóa âm nhạc từ mã ISRC và metadata
- Đề xuất bài hát dựa trên đặc trưng văn hóa và metadata
- Giao diện người dùng trực quan với Gradio
- **So sánh hai mô hình đề xuất:** EnhancedContentRecommender và WeightedContentRecommender
- **Phân tích trực quan** với biểu đồ so sánh mô hình
- Tạo danh sách phát (playlist) từ bài hát gợi ý
- Khám phá bài hát theo thể loại và văn hóa âm nhạc
- **Thông báo rõ ràng khi không tìm thấy bài hát trong dữ liệu**
- **Clustering thông minh** với K-Means và HDBSCAN

## Thiết lập dự án

### Yêu cầu

- Python 3.8+ 
- Các thư viện cần thiết (được liệt kê trong requirements.txt)
- Tài khoản nhà phát triển Spotify với Client ID và Client Secret

### Cài đặt

1. Clone repository:
```bash
git clone https://github.com/orms147/music-recommendation.git
cd music-recommendation
```

2. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

3. Tạo file .env trong thư mục gốc của dự án và thêm thông tin xác thực Spotify:
```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

## Cơ chế hoạt động

### Thu thập dữ liệu

Hệ thống sử dụng Spotify API để thu thập dữ liệu bài hát với hai phương thức chính:

1. **Thu thập dữ liệu cơ bản**: Lấy dữ liệu từ một số lượng truy vấn hạn chế, phù hợp cho việc phát triển và thử nghiệm.
   ```bash
   python main.py
   ```
   - Sử dụng giao diện web để thiết lập số lượng bài hát mỗi truy vấn

2. **Thu thập dữ liệu lớn**: Tự động thu thập hàng chục nghìn bài hát thông qua các truy vấn đa dạng.
   - Hỗ trợ lên tới 100,000 bài hát (đã thử nghiệm thành công)
   - Tự động xử lý theo lô để tránh vượt quá giới hạn API
   - Lưu dữ liệu theo định kỳ để tránh mất dữ liệu

### Xử lý dữ liệu và ISRC Cultural Intelligence

Sau khi thu thập, dữ liệu được xử lý thông qua `DataProcessor` để tạo ra các đặc trưng giá trị:

1. **Làm sạch dữ liệu**: Loại bỏ bài hát trùng lặp và dữ liệu không hợp lệ
2. **Trích xuất thông tin văn hóa từ ISRC**:
   - Phát hiện quốc gia từ mã ISRC (VN, KR, JP, CN, US, v.v.)
   - Xử lý thông minh với các mã ISRC không chuẩn (QZ, QM, QH, FX, v.v.)
   - Phát hiện văn hóa âm nhạc từ metadata khi ISRC không rõ ràng
3. **Tạo đặc trưng từ metadata**:
   - Năm phát hành, thập kỷ, độ phổ biến, độ dài bài hát
   - Phân tích tên bài hát để phát hiện remix, collab
   - Phát hiện ngôn ngữ/khu vực (Việt Nam, Hàn Quốc, Nhật Bản, Tây Ban Nha)
   - Phân loại thể loại dựa trên thông tin nghệ sĩ
4. **Clustering thông minh**:
   - K-Means clustering để phân nhóm bài hát theo đặc trưng
   - HDBSCAN clustering để phát hiện các nhóm tự nhiên và outliers
   - Sử dụng thông tin cluster để cải thiện đề xuất

### Mô hình đề xuất

Hệ thống sử dụng hai mô hình đề xuất chính:

1. **EnhancedContentRecommender**:
   - Sử dụng cosine similarity trên các đặc trưng metadata
   - Tăng cường với thông tin văn hóa từ ISRC
   - Tận dụng thông tin cluster từ K-Means

2. **WeightedContentRecommender**:
   - Sử dụng hệ thống trọng số cho các đặc trưng khác nhau
   - Tối ưu hóa cho sự tương đồng văn hóa
   - Đa dạng hóa kết quả dựa trên HDBSCAN clusters

### Phân tích trực quan

Hệ thống cung cấp các biểu đồ so sánh trực quan:
- So sánh kết quả từ hai mô hình đề xuất
- Phân tích phân bố văn hóa trong kết quả đề xuất
- Đánh giá độ đa dạng của các đề xuất

## Cấu trúc dự án

```
music-recommendation/
├── config/
│   └── config.py                # Cấu hình tập trung cho toàn bộ hệ thống
├── data/
│   ├── raw/                     # Dữ liệu thô từ Spotify API
│   └── processed/               # Dữ liệu đã qua xử lý
├── models/
│   ├── base_model.py            # Lớp cơ sở cho các mô hình đề xuất
│   ├── enhanced_content_model.py# Mô hình đề xuất nâng cao với ISRC Cultural Intelligence
│   ├── weighted_content_model.py# Mô hình weighted scoring + ISRC cultural similarity
│   └── visualization.py         # Mô-đun phân tích và trực quan hóa kết quả
├── utils/
│   ├── data_fetcher.py          # Thu thập dữ liệu từ Spotify API
│   ├── data_processor.py        # Xử lý và làm giàu dữ liệu với ISRC Cultural Intelligence
│   └── data_checker.py          # Kiểm tra chất lượng dữ liệu
├── outputs/
│   └── charts/                  # Biểu đồ so sánh mô hình được lưu ở đây
├── main.py                      # Điểm vào chính và giao diện người dùng
├── requirements.txt             # Các thư viện cần thiết
└── .env                         # Biến môi trường (không được đưa lên Git)
```

## Tùy chỉnh hệ thống

Tất cả các thông số có thể tùy chỉnh đều nằm trong config.py:

- `DEFAULT_TRACKS_PER_QUERY`: Số lượng bài hát mặc định cho mỗi truy vấn (mặc định: 100)
- `MAX_TRACKS_PER_QUERY`: Giới hạn tối đa số bài hát mỗi truy vấn (mặc định: 500)
- `MIN_TRACKS_PER_QUERY`: Giới hạn tối thiểu số bài hát mỗi truy vấn (mặc định: 50)
- `TRACKS_QUERY_STEP`: Bước nhảy cho thanh trượt (mặc định: 25)
- `LARGE_DATASET_DEFAULT_SIZE`: Kích thước mặc định cho tập dữ liệu lớn (mặc định: 5,000)
- `LARGE_DATASET_BATCH_SIZE`: Số lượng truy vấn mỗi lô khi lấy dữ liệu lớn (mặc định: 100)
- `LARGE_DATASET_SAVE_INTERVAL`: Lưu sau mỗi bao nhiêu bài hát (mặc định: 1,000)
- `CONTENT_FEATURES`: Danh sách các đặc trưng metadata được sử dụng
- `RECOMMENDATION_CONFIG`: Cấu hình cho hệ thống đề xuất
- `CLUSTERING_CONFIG`: Cấu hình cho các thuật toán clustering

### Cấu hình ISRC Cultural Intelligence

Hệ thống có các cấu hình đặc biệt cho ISRC Cultural Intelligence:

```python
# Cấu hình văn hóa âm nhạc
cultural_config = {
    # Ưu tiên theo thứ tự: quốc gia > thể loại > khu vực
    "country_boost": 0.35,      # Tăng điểm cho cùng quốc gia
    "genre_boost": 0.25,        # Tăng điểm cho cùng thể loại
    "region_boost": 0.15,       # Tăng điểm cho cùng khu vực
    
    # Cấu hình đa dạng hóa kết quả
    "same_country_ratio": 0.6,  # 60% kết quả cùng quốc gia
    "same_region_ratio": 0.8,   # 80% kết quả cùng khu vực
}
```

### Cấu hình Clustering

```python
# Cấu hình clustering
CLUSTERING_CONFIG = {
    'CLUSTERING_FEATURES': [
        'popularity_norm', 'artist_popularity_norm', 'duration_norm',
        'is_vietnamese', 'is_korean', 'is_japanese', 'is_chinese', 'is_western',
        'genre_pop', 'genre_rock', 'genre_hip_hop', 'genre_electronic'
    ],
    'KMEANS_N_CLUSTERS': 8,
    'KMEANS_RANDOM_STATE': 42,
    'HDBSCAN_MIN_CLUSTER_SIZE': 100,
    'HDBSCAN_MIN_SAMPLES': 10,
    'HDBSCAN_CLUSTER_SELECTION_EPSILON': 0.5
}
```

## Sử dụng thông qua giao diện

1. **Khởi động giao diện người dùng**:
   ```bash
   python main.py
   ```

2. **Thiết lập dữ liệu**: Sử dụng tab "Thiết lập dữ liệu" để thu thập dữ liệu ban đầu

3. **Huấn luyện mô hình**: Nhấn nút "Huấn luyện mô hình" để xử lý dữ liệu và xây dựng cả hai mô hình

4. **Gợi ý bài hát**: Nhập tên bài hát và nghệ sĩ để nhận gợi ý từ cả hai mô hình (so sánh trực tiếp)

5. **So sánh mô hình**: Sử dụng tab "So sánh mô hình" để xem biểu đồ so sánh trực quan giữa hai mô hình

6. **Tạo danh sách phát**: Sử dụng tab "Tạo danh sách phát" để tạo queue bài hát từ một bài hát gốc

7. **Khám phá theo thể loại**: Sử dụng tab "Khám phá theo thể loại" để tìm bài hát từ một thể loại cụ thể

**Lưu ý:** Nếu bài hát không có trong dữ liệu, giao diện sẽ thông báo rõ ràng cho người dùng.

## Xử lý lỗi và cải tiến

Hệ thống đã được cải tiến để xử lý các trường hợp lỗi phổ biến:

- **Kiểm tra tồn tại của cột clustering**: Tự động áp dụng clustering nếu cột không tồn tại
- **Xử lý thiếu dữ liệu**: Tạo đề xuất dự phòng khi không tìm thấy bài hát
- **Bỏ qua cảnh báo không cần thiết**: Lọc các cảnh báo từ thư viện bên thứ ba
- **Tự động làm mới token**: Xử lý hết hạn token Spotify API

## Lưu ý về giới hạn API

Spotify API có các giới hạn cần lưu ý:
- Mỗi truy vấn tìm kiếm chỉ trả về tối đa 1,000 kết quả
- Giới hạn tốc độ: khoảng 30 yêu cầu/giây hoặc 3,600 yêu cầu/giờ
- Xác thực token hết hạn sau 1 giờ (hệ thống tự động làm mới)

Khi thu thập bộ dữ liệu lớn, hệ thống đã cài đặt các cơ chế xử lý lỗi và thử lại để đảm bảo thu thập dữ liệu ổn định.

## Đóng góp

Dự án hệ thống đề xuất âm nhạc này được phát triển dựa trên kiến thức và tham khảo từ nhiều nguồn khác nhau.

## Tài liệu tham khảo chính

1. **Tài liệu API Spotify**: 
   - [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api/)
   - [Spotify API Endpoints](https://developer.spotify.com/documentation/web-api/reference/#/)
   - Hướng dẫn xác thực và giới hạn API

2. **Nghiên cứu về hệ thống đề xuất**:
   - "Recommender Systems Handbook" của Francesco Ricci, Lior Rokach, Bracha Shapira
   - Các bài báo về Content-based Recommendation Systems
   - Nghiên cứu về ứng dụng của hệ thống đề xuất trong lĩnh vực âm nhạc

3. **Thư viện Python**:
   - Tài liệu của [Spotipy](https://spotipy.readthedocs.io/) - thư viện Python cho Spotify API
   - Tài liệu của scikit-learn cho xử lý cosine similarity và các kỹ thuật học máy
   - Tài liệu của Pandas và NumPy cho xử lý dữ liệu
   - Tài liệu của [Gradio](https://gradio.app/docs/) để xây dựng giao diện
   - Tài liệu của [HDBSCAN](https://hdbscan.readthedocs.io/) cho clustering

4. **Các bài viết kỹ thuật về phân tích đặc trưng âm nhạc**:
   - Phương pháp trích xuất đặc trưng từ metadata
   - Kỹ thuật phân loại thể loại âm nhạc
   - Phát hiện ngôn ngữ và đặc trưng khu vực trong âm nhạc
   - Phân tích mã ISRC và ứng dụng trong phân loại văn hóa âm nhạc

5. **Các nguồn mở tương tự**:
   - Các dự án mã nguồn mở trên GitHub về hệ thống đề xuất âm nhạc
   - Các hướng dẫn về xây dựng hệ thống đề xuất dựa trên nội dung

