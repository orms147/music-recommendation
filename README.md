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