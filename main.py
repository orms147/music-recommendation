import os
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import pandas as pd

from config.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    DEFAULT_TRACKS_PER_QUERY, MAX_TRACKS_PER_QUERY, 
    MIN_TRACKS_PER_QUERY, TRACKS_QUERY_STEP,
    LARGE_DATASET_DEFAULT_SIZE, LARGE_DATASET_BATCH_SIZE, LARGE_DATASET_SAVE_INTERVAL
)
from utils.data_fetcher import fetch_initial_dataset
from utils.data_processor import DataProcessor
from models.hybrid_model import MetadataRecommender

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load biến môi trường
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# Biến toàn cục cho model
model = None

def check_spotify_credentials():
    from config.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
    return bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)

def setup_initial_dataset(progress=gr.Progress(), tracks_per_query=DEFAULT_TRACKS_PER_QUERY):
    """Thiết lập bộ dữ liệu ban đầu với progress bar"""
    if not check_spotify_credentials():
        return "⚠️ Thiếu thông tin xác thực Spotify. Vui lòng thiết lập file .env."
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    progress(0.1, desc="Đang thu thập dữ liệu từ Spotify...")
    try:
        tracks_df = fetch_initial_dataset(tracks_per_query=tracks_per_query)
        if tracks_df is None or tracks_df.empty:
            return "❌ Không thể lấy dữ liệu bài hát từ Spotify."
        progress(0.6, desc="Đang xử lý dữ liệu...")
        processor = DataProcessor()
        processor.process_all()
        progress(1.0, desc="Hoàn tất!")
        return f"✅ Đã thiết lập dữ liệu với {len(tracks_df)} bài hát."
    except Exception as e:
        logger.error(f"Lỗi thiết lập dữ liệu: {e}\n{traceback.format_exc()}")
        return f"❌ Lỗi thiết lập dữ liệu: {e}"

def train_model():
    """Huấn luyện mô hình đề xuất dựa trên metadata"""
    global model
    try:
        processor = DataProcessor()
        processor.load_data()
        processor.clean_tracks_data()
        processor.create_synthetic_audio_features()
        processor.merge_artist_genres()
        processor.extract_release_year()
        processor.create_genre_features()
        processor.create_language_features()
        processor.create_additional_features()
        processor.normalize_features()
        tracks_df = processor.tracks_df
        model = MetadataRecommender()
        model.train(tracks_df)
        return f"✅ Đã huấn luyện mô hình với {len(tracks_df)} bài hát."
    except Exception as e:
        logger.error(f"Lỗi huấn luyện mô hình: {e}\n{traceback.format_exc()}")
        return f"❌ Lỗi huấn luyện mô hình: {e}"

def recommend_similar(song_name, artist_name="", n=10):
    global model
    if model is None or not model.is_trained:
        return "⚠️ Vui lòng huấn luyện mô hình trước."
    try:
        recs = model.recommend(track_name=song_name, artist=artist_name, n_recommendations=n)
        if recs is None or recs.empty:
            return "Không tìm thấy bài hát hoặc không có đề xuất phù hợp."
        result = "| # | Bài hát | Nghệ sĩ |\n|---|---|---|\n"
        for i, row in enumerate(recs.itertuples(), 1):
            result += f"| {i} | {row.name} | {row.artist} |\n"
        return result
    except Exception as e:
        logger.error(f"Lỗi đề xuất: {e}\n{traceback.format_exc()}")
        return f"❌ Lỗi đề xuất: {e}"

def generate_playlist(seed_track, seed_artist="", queue_length=10):
    global model
    if model is None or not model.is_trained:
        return "⚠️ Vui lòng huấn luyện mô hình trước."
    try:
        playlist, analysis = model.generate_playlist_from_seed(seed_track, seed_artist, queue_length)
        if playlist is None or playlist.empty:
            return "Không thể tạo playlist."
        result = "## Playlist đề xuất:\n"
        for i, row in enumerate(playlist.itertuples(), 1):
            result += f"{i}. **{row.name}** - {row.artist}\n"
        if analysis is not None and not analysis.empty:
            result += "\n### Phân tích chuyển tiếp:\n"
            for i, row in analysis.iterrows():
                result += f"{i+1}. {row['from_track']} → {row['to_track']}: {row['quality']} ({row['transition_score']:.2f})\n"
        return result
    except Exception as e:
        logger.error(f"Lỗi tạo playlist: {e}\n{traceback.format_exc()}")
        return f"❌ Lỗi tạo playlist: {e}"

def discover_by_genre(genre, n=10):
    global model
    if model is None or not model.is_trained:
        return "⚠️ Vui lòng huấn luyện mô hình trước."
    try:
        recs = model.discover_by_genre(genre, n)
        if recs is None or recs.empty:
            return f"Không tìm thấy bài hát thuộc thể loại {genre}."
        result = f"## Top {n} bài hát thể loại {genre}\n"
        for i, row in enumerate(recs.itertuples(), 1):
            result += f"{i}. **{row.name}** - {row.artist}\n"
        return result
    except Exception as e:
        logger.error(f"Lỗi khám phá thể loại: {e}\n{traceback.format_exc()}")
        return f"❌ Lỗi khám phá thể loại: {e}"

def create_ui():
    with gr.Blocks(title="Music Recommender (Metadata)") as app:
        with gr.Tab("Thiết lập dữ liệu"):
            gr.Markdown("### Thiết lập dữ liệu ban đầu từ Spotify")
            tracks_per_query = gr.Slider(
                MIN_TRACKS_PER_QUERY, 
                MAX_TRACKS_PER_QUERY, 
                value=DEFAULT_TRACKS_PER_QUERY, 
                step=TRACKS_QUERY_STEP, 
                label="Số bài hát mỗi truy vấn"
            )
            setup_btn = gr.Button("Thiết lập dữ liệu")
            setup_output = gr.Markdown()
            setup_btn.click(fn=setup_initial_dataset, inputs=[tracks_per_query], outputs=setup_output)

        with gr.Tab("Huấn luyện mô hình"):
            gr.Markdown("### Huấn luyện mô hình đề xuất")
            train_btn = gr.Button("Huấn luyện mô hình")
            train_output = gr.Markdown()
            train_btn.click(fn=train_model, outputs=train_output)

        with gr.Tab("Đề xuất tương tự"):
            gr.Markdown("### Đề xuất bài hát tương tự")
            song_input = gr.Textbox(label="Tên bài hát")
            artist_input = gr.Textbox(label="Tên nghệ sĩ (tùy chọn)")
            n_similar = gr.Slider(5, 20, value=10, step=1, label="Số lượng đề xuất")
            rec_btn = gr.Button("Đề xuất")
            rec_output = gr.Markdown()
            rec_btn.click(fn=recommend_similar, inputs=[song_input, artist_input, n_similar], outputs=rec_output)

        with gr.Tab("Tạo playlist"):
            gr.Markdown("### Tạo playlist từ bài hát gốc")
            seed_song = gr.Textbox(label="Tên bài hát gốc")
            seed_artist = gr.Textbox(label="Tên nghệ sĩ (tùy chọn)")
            playlist_len = gr.Slider(5, 20, value=10, step=1, label="Độ dài playlist")
            playlist_btn = gr.Button("Tạo playlist")
            playlist_output = gr.Markdown()
            playlist_btn.click(fn=generate_playlist, inputs=[seed_song, seed_artist, playlist_len], outputs=playlist_output)

        with gr.Tab("Khám phá theo thể loại"):
            gr.Markdown("### Khám phá âm nhạc theo thể loại")
            genre_input = gr.Textbox(label="Thể loại (ví dụ: Pop, Rock, Hip-hop, Vietnamese, ...)")
            genre_n = gr.Slider(5, 20, value=10, step=1, label="Số lượng bài hát")
            genre_btn = gr.Button("Khám phá")
            genre_output = gr.Markdown()
            genre_btn.click(fn=discover_by_genre, inputs=[genre_input, genre_n], outputs=genre_output)
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hệ thống đề xuất âm nhạc")
    parser.add_argument("--fetch-large", action="store_true", help="Thu thập tập dữ liệu lớn (100,000+ bài hát)")
    parser.add_argument("--size", type=int, default=LARGE_DATASET_DEFAULT_SIZE, help="Kích thước tập dữ liệu mục tiêu")
    args = parser.parse_args()
    
    if args.fetch_large:
        from utils.data_fetcher import fetch_large_dataset
        fetch_large_dataset(
            target_size=args.size,
            batch_size=LARGE_DATASET_BATCH_SIZE,
            save_interval=LARGE_DATASET_SAVE_INTERVAL
        )
    else:
        demo = create_ui()
        demo.launch()