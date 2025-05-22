# Hệ thống Đề xuất Âm nhạc

Hệ thống đề xuất âm nhạc thông minh sử dụng dữ liệu từ Spotify API, tập trung vào phương pháp dựa trên metadata để đưa ra các gợi ý bài hát phù hợp với sở thích người dùng.

## Tính năng chính

- Thu thập dữ liệu âm nhạc từ Spotify API
- Hỗ trợ tập dữ liệu lớn 
- Xử lý và phân tích metadata của bài hát
- Khả năng tùy chỉnh số lượng dữ liệu thu thập
- Đề xuất bài hát dựa trên các đặc trưng metadata
- Giao diện người dùng trực quan với Gradio
- Gợi ý âm nhạc tương tự bài hát đang nghe
- Tạo danh sách phát từ bài hát gợi ý
- Khám phá bài hát theo thể loại

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

2. **Thu thập dữ liệu lớn**: Tự động thu thập hàng trăm nghìn bài hát thông qua các truy vấn đa dạng.
   - Hỗ trợ lên tới 400,000 bài hát (có thể lớn hơn)
   - Tự động xử lý theo lô để tránh vượt quá giới hạn API
   - Lưu dữ liệu theo định kỳ để tránh mất dữ liệu

### Xử lý dữ liệu

Sau khi thu thập, dữ liệu được xử lý thông qua `DataProcessor` để tạo ra các đặc trưng giá trị:

1. **Làm sạch dữ liệu**: Loại bỏ bài hát trùng lặp và dữ liệu không hợp lệ
2. **Tạo đặc trưng từ metadata**:
   - Năm phát hành, thập kỷ, độ phổ biến, độ dài bài hát
   - Phân tích tên bài hát để phát hiện remix, collab
   - Phát hiện ngôn ngữ/khu vực (Việt Nam, Hàn Quốc, Nhật Bản, Tây Ban Nha)
   - Phân loại thể loại dựa trên thông tin nghệ sĩ
3. **Tạo đặc trưng audio tổng hợp**: Khi không có dữ liệu audio thực, hệ thống tạo ra các đặc trưng tổng hợp
4. **Chuẩn hóa**: Đảm bảo tất cả các đặc trưng số nằm trong khoảng [0,1]

### Mô hình đề xuất

Hệ thống sử dụng phương pháp đề xuất dựa trên nội dung (Content-based Recommendation):

1. **ContentBasedRecommender**: Mô hình cốt lõi tính toán độ tương tự giữa các bài hát dựa trên metadata
   - Sử dụng độ tương tự cosine giữa các vector đặc trưng
   - Hỗ trợ tìm kiếm bài hát theo tên và nghệ sĩ
   - Xử lý thông minh với các trường hợp không tìm thấy bài hát chính xác

2. **MetadataRecommender**: Lớp cao cấp hơn bao bọc ContentBasedRecommender và thêm các tính năng:
   - Tạo danh sách phát từ một bài hát gốc
   - Phân tích sự chuyển tiếp giữa các bài hát trong danh sách phát
   - Khám phá bài hát theo thể loại
   - Xử lý lỗi và fallback khi không tìm thấy kết quả

## Cấu trúc dự án

```
music-recommendation/
├── config/
│   └── config.py          # Cấu hình tập trung cho toàn bộ hệ thống
├── data/
│   ├── raw/               # Dữ liệu thô từ Spotify API
│   └── processed/         # Dữ liệu đã qua xử lý
├── models/
│   ├── base_model.py      # Lớp cơ sở cho các mô hình đề xuất
│   ├── content_model.py   # Mô hình đề xuất dựa trên nội dung
│   └── hybrid_model.py    # Mô hình kết hợp (MetadataRecommender)
├── utils/
│   ├── data_fetcher.py    # Thu thập dữ liệu từ Spotify API
│   └── data_processor.py  # Xử lý và làm giàu dữ liệu
├── main.py                # Điểm vào chính và giao diện người dùng
├── requirements.txt       # Các thư viện cần thiết
└── .env                   # Biến môi trường (không được đưa lên Git)
```

## Tùy chỉnh hệ thống

Tất cả các thông số có thể tùy chỉnh đều nằm trong config.py:

- `DEFAULT_TRACKS_PER_QUERY`: Số lượng bài hát mặc định cho mỗi truy vấn (mặc định: 100)
- `MAX_TRACKS_PER_QUERY`: Giới hạn tối đa số bài hát mỗi truy vấn (mặc định: 500)
- `MIN_TRACKS_PER_QUERY`: Giới hạn tối thiểu số bài hát mỗi truy vấn (mặc định: 5)
- `TRACKS_QUERY_STEP`: Bước nhảy cho thanh trượt (mặc định: 5)
- `LARGE_DATASET_DEFAULT_SIZE`: Kích thước mặc định cho tập dữ liệu lớn (mặc định: 100,000)
- `LARGE_DATASET_BATCH_SIZE`: Số lượng truy vấn mỗi lô khi lấy dữ liệu lớn (mặc định: 500)
- `LARGE_DATASET_SAVE_INTERVAL`: Lưu sau mỗi bao nhiêu bài hát (mặc định: 5,000)
- `CONTENT_FEATURES`: Danh sách các đặc trưng metadata được sử dụng

## Sử dụng thông qua giao diện

1. **Khởi động giao diện người dùng**:
   ```bash
   python main.py
   ```

2. **Thiết lập dữ liệu**: Sử dụng tab "Thiết lập dữ liệu" để thu thập dữ liệu ban đầu

3. **Huấn luyện mô hình**: Nhấn nút "Huấn luyện mô hình" để xử lý dữ liệu và xây dựng mô hình

4. **Gợi ý bài hát**: Nhập tên bài hát và nghệ sĩ để nhận gợi ý bài hát tương tự

5. **Tạo danh sách phát**: Sử dụng tab "Tạo danh sách phát" để tạo queue bài hát từ một bài hạt gốc

6. **Khám phá theo thể loại**: Sử dụng tab "Khám phá theo thể loại" để tìm bài hát từ một thể loại cụ thể

## Lưu ý về giới hạn API

Spotify API có các giới hạn cần lưu ý:
- Mỗi truy vấn tìm kiếm chỉ trả về tối đa 1,000 kết quả
- Giới hạn tốc độ: khoảng 30 yêu cầu/giây hoặc 3,600 yêu cầu/giờ
- Xác thực token hết hạn sau 1 giờ (hệ thống tự động làm mới)

Khi thu thập bộ dữ liệu lớn, hệ thống đã cài đặt các cơ chế xử lý lỗi và thử lại để đảm bảo thu thập dữ liệu ổn định.

## Đóng góp

Dự án hệ thống đề xuất âm nhạc này được tôi phát triển dựa trên kiến thức và tham khảo từ nhiều nguồn khác nhau:

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

4. **Các bài viết kỹ thuật về phân tích đặc trưng âm nhạc**:
   - Phương pháp trích xuất đặc trưng từ metadata
   - Kỹ thuật phân loại thể loại âm nhạc
   - Phát hiện ngôn ngữ và đặc trưng khu vực trong âm nhạc

5. **Các nguồn mở tương tự**:
   - Các dự án mã nguồn mở trên GitHub về hệ thống đề xuất âm nhạc
   - Các hướng dẫn về xây dựng hệ thống đề xuất dựa trên nội dung

