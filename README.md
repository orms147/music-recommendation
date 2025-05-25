# Hệ thống Đề xuất Âm nhạc

Hệ thống đề xuất âm nhạc sử dụng dữ liệu thực từ Spotify API với hai mô hình content-based filtering tiên tiến.

## Tính năng chính

- Thu thập dữ liệu thực từ Spotify API (chỉ metadata, không synthetic data)
- Hai mô hình đề xuất:
  - **EnhancedContentRecommender:** Fuzzy search + multi-factor scoring
  - **WeightedContentRecommender:** Language-first + mood hierarchy
- Tối ưu hóa cho âm nhạc đa ngôn ngữ (Việt Nam, Hàn Quốc, Nhật Bản, Tây Ban Nha)
- Fuzzy search thông minh - tìm bài hát ngay cả khi tên không chính xác
- Giao diện web với Gradio để so sánh trực tiếp hai mô hình
- Tự động fetch artist genres với API optimization

## Cấu trúc dự án

```
music-recommendation/
├── config/config.py             # Cấu hình hệ thống
├── data/                        # Dữ liệu raw và processed
├── models/                      # Các mô hình đề xuất
│   ├── base_model.py
│   ├── content_model.py
│   ├── enhanced_content_model.py
│   └── weighted_content_model.py
├── utils/                       # Utilities
│   ├── data_fetcher.py
│   ├── data_processor.py
│   └── data_checker.py
├── main.py                      # Entry point
└── requirements.txt
```

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/your-username/music-recommendation.git
cd music-recommendation
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Tạo file `.env`:
```env
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

4. Chạy ứng dụng:
```bash
python main.py
```

## Hai mô hình đề xuất

### EnhancedContentRecommender
- **Fuzzy search** với SequenceMatcher (75% threshold)
- **Multi-factor scoring** từ 6 yếu tố
- **Artist diversity** cao với penalty cho cùng nghệ sĩ
- **Balanced popularity** weighting
- Thích hợp cho: general users, music discovery

### WeightedContentRecommender  
- **Language-first approach** (70% weight cho cùng ngôn ngữ)
- **Mood hierarchy** với 4 priority tiers
- **Cultural similarity** awareness
- Minimal popularity impact
- Thích hợp cho: language-specific preferences, Asian music

## Features sử dụng

Chỉ sử dụng real Spotify metadata:

```python
CONTENT_FEATURES = [
    # Core Spotify metadata
    'popularity', 'duration_ms', 'explicit', 'release_year',
    'artist_popularity', 'total_tracks', 'track_number',
    
    # Language detection  
    'is_vietnamese', 'is_korean', 'is_japanese', 'is_spanish',
    
    # Track analysis
    'has_collab', 'is_remix', 'name_length', 'artist_frequency_norm'
]
```

## Sử dụng

1. **Kiểm tra dữ liệu**: Đánh giá quality và completeness
2. **Thiết lập dữ liệu**: Thu thập từ Spotify API (50-800 tracks/query)
3. **Huấn luyện mô hình**: Train cả hai models
4. **Đề xuất bài hát**: So sánh trực tiếp kết quả từ hai mô hình
5. **Tạo playlist**: Generate queue từ seed tracks

## Troubleshooting

**Model not trained:** Chạy "Huấn luyện mô hình" trước khi đề xuất

**Track not found:** Sử dụng fuzzy search hoặc kiểm tra tên bài hát

**API limits:** Hệ thống tự động handle rate limiting và retry

## Dependencies chính

- spotipy: Spotify API wrapper
- scikit-learn: ML algorithms  
- gradio: Web interface
- pandas/numpy: Data processing

## Tài liệu tham khảo

### API Documentation
- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api/)
- [Spotify API Reference](https://developer.spotify.com/documentation/web-api/reference/)
- [Spotipy Documentation](https://spotipy.readthedocs.io/)

### Research Papers & Books
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.

### Technical References
- [Content-Based Recommendation Systems](https://developers.google.com/machine-learning/recommendation/content-based/basics)
- [Cosine Similarity in Recommendation Systems](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Fuzzy String Matching with Python](https://docs.python.org/3/library/difflib.html)

### Libraries Documentation
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Gradio Documentation](https://gradio.app/docs/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

### Music Information Retrieval
- Schedl, M., Gómez, E., & Urbano, J. (2014). Music information retrieval: Recent developments and applications. *Foundations and Trends in Information Retrieval*, 8(2-3), 127-261.
- Celma, Ò. (2010). *Music Recommendation and Discovery*. Springer.

### Multi-cultural Music Analysis
- Hu, X., & Lee, J. H. (2012). A cross-cultural study of music mood perception between American and Chinese listeners. *ISMIR*, 19-24.
- Laplante, A., & Downie, J. S. (2006). Everyday life music information-seeking behaviour of young adults. *ISMIR*, 381-382.