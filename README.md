# Music Recommendation System

Hệ thống đề xuất âm nhạc thông minh sử dụng dữ liệu từ Spotify API, kết hợp nhiều phương pháp học máy để đề xuất bài hát phù hợp.

## Tính năng

- Tìm nạp dữ liệu âm nhạc từ Spotify API
- Xử lý và phân tích đặc trưng âm thanh
- Đề xuất bài hát dựa trên nhiều phương pháp:
  - Dựa trên nội dung (Content-based)
  - Lọc cộng tác (Collaborative Filtering)
  - Dựa trên chuỗi (Sequence-based)
  - Kết hợp (Hybrid)
- Giao diện dòng lệnh tương tác
- Trực quan hóa dữ liệu âm nhạc

## Cài đặt

1. Clone repository:
```
git clone https://github.com/yourusername/music-recommendation-system.git
cd music-recommendation-system
```

2. Cài đặt các gói phụ thuộc:
```
pip install -r requirements.txt
```

3. Đăng ký và lấy API keys từ [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)

## Thiết lập Spotify API Credentials

Để sử dụng hệ thống đề xuất âm nhạc này, bạn cần có thông tin xác thực từ Spotify API. Hãy làm theo các bước sau:

### Bước 1: Tạo file .env
Tạo một file có tên `.env` trong thư mục gốc của dự án.

### Bước 2: Thêm thông tin xác thực 
Thêm các dòng sau vào file `.env`:

Thay thế `your_client_id_here` và `your_client_secret_here` bằng thông tin xác thực thực tế của bạn.

### Bước 3: Lưu file và khởi động lại ứng dụng
Sau khi lưu file, hãy khởi động lại ứng dụng để các thay đổi có hiệu lực.

### Cách lấy thông tin xác thực Spotify API:

1. Truy cập [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Đăng nhập với tài khoản Spotify của bạn
3. Tạo một ứng dụng mới
4. Sao chép Client ID và Client Secret từ trang chi tiết ứng dụng

> **Lưu ý**: Không chia sẻ thông tin xác thực Spotify API của bạn với người khác hoặc đưa chúng lên GitHub.

## Sử dụng

### Cài đặt hệ thống
```
python main.py --setup
```

### Chế độ tương tác
```
python main.py
```

### Lấy đề xuất cho một bài hát cụ thể
```
python main.py --recommend --track "Shape of You" --artist "Ed Sheeran"
```

## Cấu trúc dự án

- `config/`: Cấu hình và các tham số
- `data/`: Thư mục dữ liệu
  - `raw/`: Dữ liệu thô từ Spotify API
  - `processed/`: Dữ liệu đã xử lý
- `models/`: Các mô hình đề xuất
- `utils/`: Tiện ích và công cụ
- `main.py`: Đầu vào chính của chương trình

## Đóng góp

Đóng góp luôn được chào đón! Vui lòng xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết thêm chi tiết.

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem tệp [LICENSE](LICENSE) để biết thêm chi tiết.