import os
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import pandas as pd
import numpy as np
import pickle

from config.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    DEFAULT_TRACKS_PER_QUERY, MAX_TRACKS_PER_QUERY, 
    MIN_TRACKS_PER_QUERY, TRACKS_QUERY_STEP,
    LARGE_DATASET_DEFAULT_SIZE, LARGE_DATASET_BATCH_SIZE, LARGE_DATASET_SAVE_INTERVAL
)
from utils.data_fetcher import fetch_initial_dataset
from utils.data_processor import DataProcessor
from models.hybrid_model import MetadataRecommender
from models.weighted_content_model import WeightedContentRecommender

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load biến môi trường
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# Biến toàn cục cho model
model = None
weighted_model = None

def initialize_model():
    """Khởi tạo model khi khởi động ứng dụng"""
    global model, weighted_model
    model_path = os.path.join(MODELS_DIR, 'metadata_recommender.pkl')
    weighted_model_path = os.path.join(MODELS_DIR, 'weighted_content_recommender.pkl')

    # MetadataRecommender
    if os.path.exists(model_path):
        try:
            model = MetadataRecommender.load(model_path)
            logging.info(f"Đã nạp model từ {model_path}")
            logging.info(f"Model được huấn luyện vào: {model.train_time}")
        except Exception as e:
            logging.error(f"Lỗi khi nạp model: {e}")
            model = None

    # WeightedContentRecommender
    if os.path.exists(weighted_model_path):
        try:
            weighted_model = WeightedContentRecommender.load(weighted_model_path)
            logging.info(f"Đã nạp weighted model từ {weighted_model_path}")
        except Exception as e:
            logging.error(f"Lỗi khi nạp weighted model: {e}")
            weighted_model = None

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
    """Huấn luyện hoặc nạp lại mô hình đề xuất dựa trên metadata"""
    global model, weighted_model
    try:
        # Đường dẫn để lưu/nạp mô hình
        model_path = os.path.join(MODELS_DIR, 'metadata_recommender.pkl')
        weighted_model_path = os.path.join(MODELS_DIR, 'weighted_content_recommender.pkl')

        # Kiểm tra xem mô hình đã tồn tại chưa
        if os.path.exists(model_path):
            logging.info("Tìm thấy mô hình đã huấn luyện, đang nạp...")
            model = MetadataRecommender.load(model_path)
            logging.info(f"Đã nạp mô hình thành công (được huấn luyện vào: {model.train_time})")
        else:
            logging.info("Không tìm thấy mô hình đã lưu, đang huấn luyện mới...")
            processed_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
            if not os.path.exists(processed_path):
                processor = DataProcessor()
                processor.process_all()
            tracks_df = pd.read_csv(processed_path)
            model = MetadataRecommender()
            model.train(tracks_df)
            model.save(model_path)
            logging.info(f"Đã huấn luyện và lưu mô hình MetadataRecommender thành công!")

        # WeightedContentRecommender
        if os.path.exists(weighted_model_path):
            logging.info("Tìm thấy weighted model đã huấn luyện, đang nạp...")
            weighted_model = WeightedContentRecommender.load(weighted_model_path)
            logging.info("Đã nạp weighted model thành công!")
        else:
            logging.info("Không tìm thấy weighted model đã lưu, đang huấn luyện mới...")
            processed_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
            if not os.path.exists(processed_path):
                processor = DataProcessor()
                processor.process_all()
            tracks_df = pd.read_csv(processed_path)
            weighted_model = WeightedContentRecommender()
            weighted_model.train(tracks_df)
            weighted_model.save(weighted_model_path)
            logging.info("Đã huấn luyện và lưu weighted model thành công!")

        return "Đã huấn luyện cả hai mô hình thành công!"
    except Exception as e:
        logging.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        return f"Lỗi khi huấn luyện mô hình: {str(e)}"

def recommend_similar(song_name, artist_name="", n=10):
    """Enhanced recommendation with better debugging"""
    global model, weighted_model
    if model is None or not model.is_trained:
        return "⚠️ Mô hình Metadata chưa được huấn luyện. Vui lòng huấn luyện mô hình trước."
    if weighted_model is None or not weighted_model.is_trained:
        return "⚠️ Mô hình Weighted chưa được huấn luyện. Vui lòng huấn luyện mô hình trước."
    
    try:
        # Load processed data để kiểm tra
        processed_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
        if os.path.exists(processed_path):
            tracks_df = pd.read_csv(processed_path)
            
            # Debug: Log available features
            logger.info(f"Available features: {tracks_df.columns.tolist()}")
            logger.info(f"Dataset shape: {tracks_df.shape}")
            
            # Tìm bài hát gốc
            mask = tracks_df['name'].str.lower().str.strip() == song_name.lower().strip()
            if artist_name:
                mask = mask & (tracks_df['artist'].str.lower().str.strip() == artist_name.lower().strip())
            
            found_tracks = tracks_df[mask]
            
            if found_tracks.empty:
                # Enhanced fallback với suggestions
                available_tracks_sample = tracks_df[['name', 'artist']].head(10)
                
                # Tìm tracks tương tự bằng fuzzy matching
                from difflib import get_close_matches
                track_names = tracks_df['name'].tolist()
                close_matches = get_close_matches(song_name, track_names, n=5, cutoff=0.6)
                
                suggestion_text = ""
                if close_matches:
                    suggestion_text = f"\n**Gợi ý tương tự:** {', '.join(close_matches[:3])}"
                
                return f"""❌ Không tìm thấy bài hát **{song_name}** (nghệ sĩ: {artist_name}) trong dữ liệu.
{suggestion_text}

**Một số bài hát có sẵn:**
{available_tracks_sample.to_markdown(index=False)}

Vui lòng kiểm tra lại tên bài hát và nghệ sĩ!"""
            
            # Hiển thị thông tin bài hát gốc với nhiều thông tin hơn
            original_info = found_tracks.iloc[0]
            seed_info = f"""**🎵 Bài hát gốc:** {original_info['name']} - {original_info['artist']}
**Popularity:** {original_info.get('popularity', 'N/A')} | **Năm phát hành:** {original_info.get('release_year', 'N/A')}
**Duration:** {original_info.get('duration_min', 0):.1f} phút | **Album:** {original_info.get('album', 'N/A')}

---
"""
        else:
            seed_info = "**Dữ liệu bài hát gốc không có sẵn**\n\n---\n"
        
        # Thực hiện đề xuất với error handling tốt hơn
        logger.info(f"Generating recommendations for '{song_name}' by {artist_name}")
        
        try:
            rec1 = model.recommend(track_name=song_name, artist=artist_name, n_recommendations=n)
            model_1_success = True
        except Exception as e:
            logger.error(f"Model 1 failed: {e}")
            rec1 = pd.DataFrame()
            model_1_success = False
        
        try:
            rec2 = weighted_model.recommend(track_name=song_name, artist=artist_name, n_recommendations=n)
            model_2_success = True
        except Exception as e:
            logger.error(f"Model 2 failed: {e}")
            rec2 = pd.DataFrame()
            model_2_success = False
        
        # Tạo kết quả với debug info
        result = seed_info
        
        # Model 1 results
        result += "### 📊 MetadataRecommender (Content-Based):\n"
        if model_1_success and not rec1.empty:
            display_cols = ['name', 'artist', 'content_score', 'popularity', 'release_year']
            available_cols = [col for col in display_cols if col in rec1.columns]
            result += rec1[available_cols].round(3).to_markdown(index=False) + "\n"
            
            # Add quality metrics
            avg_score = rec1['content_score'].mean() if 'content_score' in rec1.columns else 0
            result += f"\n*Avg similarity: {avg_score:.3f}*\n"
        else:
            result += "❌ Model failed to generate recommendations\n"
        
        result += "\n---\n"
        
        # Model 2 results
        result += "### ⚖️ WeightedContentRecommender (Advanced Scoring):\n"
        if model_2_success and not rec2.empty:
            display_cols = ['name', 'artist', 'final_score', 'popularity', 'release_year']
            available_cols = [col for col in display_cols if col in rec2.columns]
            result += rec2[available_cols].round(3).to_markdown(index=False)
            
            # Add quality metrics
            avg_score = rec2['final_score'].mean() if 'final_score' in rec2.columns else 0
            result += f"\n\n*Avg weighted score: {avg_score:.3f}*"
        else:
            result += "❌ Model failed to generate recommendations"
        
        return result
        
    except Exception as e:
        logger.error(f"Lỗi khi đề xuất: {e}\n{traceback.format_exc()}")
        return f"❌ Lỗi hệ thống khi đề xuất: {str(e)}"

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

def check_data_status():
    """Check data completeness and quality for recommendation system"""
    try:
        from utils.data_checker import check_data_completeness
        result = check_data_completeness()
        
        # Format result for Gradio display
        score = result['readiness_score']
        max_score = result['max_score']
        tracks_count = result['tracks_count']
        
        # Use text-based status indicators
        if score >= 6:
            status_emoji = "[EXCELLENT]"
            status_text = "Production Ready!"
        elif score >= 4:
            status_emoji = "[GOOD]"
            status_text = "Ready with minor improvements"
        elif score >= 2:
            status_emoji = "[FAIR]"
            status_text = "Basic functionality available"
        else:
            status_emoji = "[POOR]"
            status_text = "More data needed"
        
        summary = f"""
## {status_emoji} Data Status Report

**Overall Readiness:** {score}/{max_score} ({score/max_score*100:.1f}%)

**Dataset Size:** {tracks_count:,} tracks

**Status:** {status_text}

**Detailed analysis logged to console.**

**Next Steps:**
- Check console output for detailed breakdown
- If score < 6, consider collecting more data
- Run data processing if files are missing
        """
        
        return summary
        
    except Exception as e:
        logger.error(f"Error checking data status: {e}")
        return f"[ERROR] Error checking data status: {str(e)}"

def create_ui():
    with gr.Blocks(title="Music Recommender (Real Spotify Metadata)", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎵 Music Recommender - Real Spotify Data Only")
        gr.Markdown("*Hệ thống đề xuất âm nhạc dựa trên metadata thực từ Spotify API*")
        
        with gr.Tab("🔧 Thiết lập dữ liệu"):
            gr.Markdown("### Thiết lập dữ liệu ban đầu từ Spotify API")
            gr.Markdown("*Thu thập metadata thực từ Spotify, không sử dụng synthetic data*")
            
            # Data status check
            with gr.Row():
                with gr.Column():
                    check_data_btn = gr.Button("🔍 Kiểm tra dữ liệu hiện tại", variant="secondary")
                    data_status_output = gr.Markdown()
                    check_data_btn.click(fn=check_data_status, outputs=data_status_output)
                
                with gr.Column():
                    tracks_per_query = gr.Slider(
                        MIN_TRACKS_PER_QUERY, 
                        MAX_TRACKS_PER_QUERY, 
                        value=DEFAULT_TRACKS_PER_QUERY, 
                        step=TRACKS_QUERY_STEP, 
                        label="Số bài hát mỗi truy vấn"
                    )
                    setup_btn = gr.Button("🚀 Thiết lập dữ liệu", variant="primary")
                    setup_output = gr.Markdown()
                    setup_btn.click(fn=setup_initial_dataset, inputs=[tracks_per_query], outputs=setup_output)

        with gr.Tab("🤖 Huấn luyện mô hình"):
            gr.Markdown("### Huấn luyện mô hình đề xuất")
            gr.Markdown("*Huấn luyện với real metadata từ Spotify (popularity, duration, genre, release_year, v.v.)*")
            train_btn = gr.Button("🏋️ Huấn luyện mô hình", variant="primary")
            train_output = gr.Markdown()
            train_btn.click(fn=train_model, outputs=train_output)

        with gr.Tab("🎯 Đề xuất tương tự"):
            gr.Markdown("### Đề xuất bài hát tương tự")
            gr.Markdown("*Dựa trên real Spotify metadata: popularity, genre, artist, release year, duration...*")
            with gr.Row():
                song_input = gr.Textbox(label="🎵 Tên bài hát", placeholder="Ví dụ: Shape of You")
                artist_input = gr.Textbox(label="👤 Tên nghệ sĩ (tùy chọn)", placeholder="Ví dụ: Ed Sheeran")
            n_similar = gr.Slider(5, 20, value=10, step=1, label="📊 Số lượng đề xuất")
            rec_btn = gr.Button("🔍 Đề xuất", variant="primary")
            rec_output = gr.Markdown()
            rec_btn.click(fn=recommend_similar, inputs=[song_input, artist_input, n_similar], outputs=rec_output)
    
    return app

if __name__ == "__main__":
    import argparse
    
    # Khởi tạo model nếu có
    initialize_model()
    
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