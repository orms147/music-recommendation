import os
import logging
import pandas as pd
import numpy as np
import gradio as gr
import sys
import io
import traceback
from pathlib import Path
import time
from dotenv import load_dotenv

# Đảm bảo load file .env từ thư mục gốc của dự án
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# Kiểm tra và cấu hình truy cập API
if not os.getenv('SPOTIFY_CLIENT_ID') or not os.getenv('SPOTIFY_CLIENT_SECRET'):
    print("WARNING: Spotify API credentials not found in .env file.")
    print("Please create a .env file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")

from config.config import (
    SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, RAW_DATA_DIR,
    PROCESSED_DATA_DIR, MODELS_DIR
)
from utils.data_fetcher import SpotifyDataFetcher, fetch_initial_dataset
from utils.data_processor import DataProcessor
from models.content_model import ContentBasedRecommender
from models.transition_model import TransitionModel
from models.hybrid_model import HybridRecommender

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Biến toàn cục cho mô hình
model = None

def check_spotify_credentials():
    """Kiểm tra thông tin đăng nhập Spotify API từ .env"""
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        logger.warning("Không tìm thấy thông tin Spotify API trong file .env")
        return False
    
    logger.info("Tìm thấy thông tin Spotify API")
    return True

def test_spotify_connection():
    """Kiểm tra kết nối với Spotify API"""
    try:
        from utils.data_fetcher import SpotifyDataFetcher
        
        # Tạo instance mới của SpotifyDataFetcher
        fetcher = SpotifyDataFetcher()
        
        # Thử tìm kiếm một bài hát để kiểm tra kết nối
        results = fetcher.sp.search(q="test", limit=1)
        
        if results and 'tracks' in results and 'items' in results['tracks']:
            track_name = results['tracks']['items'][0]['name'] if results['tracks']['items'] else "Unknown"
            return f"✅ Kết nối Spotify API thành công! Tìm thấy bài hát: {track_name}"
        else:
            return "❌ Lỗi: Kết nối thành công nhưng định dạng phản hồi không đúng"
    except Exception as e:
        return f"❌ Lỗi kết nối: {str(e)}"

def load_model():
    """Nạp mô hình nếu đã tồn tại"""
    global model
    
    model_path = os.path.join(MODELS_DIR, 'hybrid_recommender.pkl')
    
    if os.path.exists(model_path):
        try:
            model = HybridRecommender()
            model.load(model_path)
            
            # Nạp tracks_df nếu cần
            tracks_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
            if os.path.exists(tracks_path):
                model.tracks_df = pd.read_csv(tracks_path)
            
            return "✅ Đã nạp mô hình thành công!"
        except Exception as e:
            return f"⚠️ Lỗi khi nạp mô hình: {str(e)}"
    
    return "⚠️ Chưa tìm thấy mô hình. Vui lòng thiết lập hệ thống trước."

def get_song_recommendations(song_name, artist_name="", num_recommendations=10):
    """Tìm đề xuất bài hát tương tự"""
    global model
    
    # Kiểm tra mô hình đã nạp chưa
    if model is None:
        result = load_model()
        if "⚠️" in result:
            return None, result
    
    # Kiểm tra input
    if not song_name:
        return None, "⚠️ Vui lòng nhập tên bài hát."
    
    try:
        # Tìm đề xuất
        recommendations = model.recommend(
            track_name=song_name,
            artist=artist_name,
            n_recommendations=num_recommendations
        )
        
        # Format kết quả
        output_text = f"### Đề xuất cho: '{song_name}'{' - ' + artist_name if artist_name else ''}\n\n"
        
        if recommendations is not None and not recommendations.empty:
            # Hiển thị từng đề xuất
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                # Format điểm số
                score = round(row.get('weighted_score', 0) * 100, 2)
                
                # Tạo thông tin bài hát
                track_info = f"**{row['name']}** - {row['artist']} (Độ tương tự: {score}%)"
                    
                output_text += f"{i}. {track_info}\n"
        else:
            output_text += "Không tìm thấy đề xuất nào 😢"
        
        return recommendations, output_text
    
    except Exception as e:
        logger.error(f"Lỗi tìm đề xuất: {str(e)}")
        return None, f"⚠️ Lỗi: {str(e)}"

def optimize_music_queue(queue_text):
    """Tối ưu hóa hàng đợi phát nhạc"""
    global model
    
    # Kiểm tra mô hình đã nạp chưa
    if model is None:
        result = load_model()
        if "⚠️" in result:
            return result
    
    # Phân tích danh sách bài hát
    if not queue_text:
        return "⚠️ Vui lòng nhập danh sách bài hát, cách nhau bởi dấu phẩy."
    
    tracks = [track.strip() for track in queue_text.split(',')]
    tracks = [track for track in tracks if track]
    
    parsed_tracks = []
    for track in tracks:
        parts = track.split('-', 1)
        if len(parts) > 1:
            track_name = parts[0].strip()
            artist_name = parts[1].strip()
            parsed_tracks.append((track_name, artist_name))
        else:
            track_name = parts[0].strip()
            parsed_tracks.append((track_name, None))
    
    # Tìm ID của các bài hát
    track_ids = []
    not_found = []
    
    for track_name, artist_name in parsed_tracks:
        track_idx = model.content_recommender._find_track_index(track_name=track_name, artist=artist_name)
        if track_idx is not None:
            track_id = model.tracks_df.iloc[track_idx]['id']
            track_ids.append(track_id)
        else:
            not_found.append(f"{track_name}{' - ' + artist_name if artist_name else ''}")
    
    if not track_ids:
        return "⚠️ Không tìm thấy bài hát nào. Vui lòng kiểm tra lại tên bài hát."
    
    # Tối ưu hóa hàng đợi
    try:
        optimized_queue = model.optimize_queue(track_ids)
        analysis = model.analyze_queue(optimized_queue)
        
        # Format kết quả
        output_text = f"### Hàng đợi đã tối ưu ({len(optimized_queue)} bài hát)\n\n"
        
        if not_found:
            output_text += f"**Lưu ý**: Không tìm thấy {len(not_found)} bài hát: {', '.join(not_found)}\n\n"
        
        tracks_df = model.tracks_df
        for i, track_id in enumerate(optimized_queue, 1):
            track = tracks_df[tracks_df['id'] == track_id]
            if not track.empty:
                output_text += f"{i}. **{track.iloc[0]['name']}** - {track.iloc[0]['artist']}\n"
        
        if analysis is not None and not analysis.empty:
            output_text += "\n### Phân tích chất lượng chuyển tiếp\n\n"
            quality_emoji = {
                "Excellent": "🟢 Tuyệt vời",
                "Good": "🟡 Tốt",
                "Average": "🟠 Trung bình",
                "Poor": "🔴 Không tốt"
            }
            
            for i, row in analysis.iterrows():
                from_parts = row['from_track'].split(' - ', 1)
                to_parts = row['to_track'].split(' - ', 1)
                from_name = from_parts[0]
                to_name = to_parts[0]
                output_text += f"{i+1}. **{from_name}** → **{to_name}**: {quality_emoji.get(row['quality'], row['quality'])} ({row['transition_score']:.2f})\n"
        
        return output_text
        
    except Exception as e:
        logger.error(f"Lỗi tối ưu hàng đợi: {str(e)}")
        return f"⚠️ Lỗi: {str(e)}"

def generate_queue(seed_track, seed_artist="", queue_length=10):
    """Tạo danh sách phát từ bài hát đầu vào"""
    global model
    try:
        # Đảm bảo mô hình đã được nạp
        if model is None:
            result = load_model()
            if "⚠️" in result:
                return result

        # Tạo queue
        queue, analysis = model.generate_playlist_from_seed(
            seed_track=seed_track,
            seed_artist=seed_artist,
            n_recommendations=int(queue_length)
        )
        
        if queue is None or queue.empty:
            return "Không thể tạo đề xuất. Vui lòng thử với bài hát khác."
        
        # Tạo kết quả dưới dạng markdown
        output_text = f"## Đề xuất từ '{seed_track}'\n\n"
        
        # Emoji cho chất lượng chuyển đổi
        quality_emoji = {
            "Excellent": "🌟",
            "Good": "👍",
            "Fair": "👌",
            "Poor": "👎"
        }
        
        # Tạo danh sách bài hát
        output_text += "### Danh sách bài hát:\n"
        for i, track in enumerate(queue['name']):
            artist = queue['artist'].iloc[i]
            output_text += f"{i+1}. **{track}** - *{artist}*\n"
        
        # Nếu có phân tích chuyển đổi
        if analysis is not None and not analysis.empty:
            output_text += "\n### Phân tích chuyển đổi:\n"
            for i, row in analysis.iterrows():
                from_parts = row['from_track'].split(' - ', 1)
                to_parts = row['to_track'].split(' - ', 1)
                from_name = from_parts[0]
                to_name = to_parts[0]
                output_text += f"{i+1}. **{from_name}** → **{to_name}**: {quality_emoji.get(row['quality'], row['quality'])} ({row['transition_score']:.2f})\n"
        
        return output_text
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Lỗi khi tạo queue: {str(e)}\n{error_trace}")
        return f"Lỗi khi tạo đề xuất: {str(e)}"
    
def setup_initial_dataset(progress=gr.Progress(), tracks_per_query=20):
    """Thiết lập bộ dữ liệu ban đầu với progress bar"""
    # Kiểm tra thông tin đăng nhập
    if not check_spotify_credentials():
        return "⚠️ Thiếu thông tin xác thực Spotify. Vui lòng thiết lập file .env."
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    progress(0.1, desc="Kiểm tra dữ liệu hiện có...")
    
    try:
        # Khởi tạo fetcher
        fetcher = SpotifyDataFetcher()
        
        # Tải bài hát
        progress(0.3, desc=f"Tải danh sách bài hát (tối đa {tracks_per_query} bài/query)...")
        tracks_df = fetch_initial_dataset(tracks_per_query=tracks_per_query)
        
        if tracks_df is None or tracks_df.empty:
            return "⚠️ Không thể tải bài hát từ Spotify API."
            
        # Lưu danh sách bài hát
        tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
        tracks_df.to_csv(tracks_path, index=False)
        
        progress(0.5, desc="Tải đặc trưng âm thanh...")
        
        # Tải đặc trưng âm thanh (giới hạn để tránh lỗi API)
        audio_features_path = os.path.join(RAW_DATA_DIR, 'audio_features.csv')
        max_tracks = min(200, len(tracks_df))  # Giảm số lượng tracks tối đa
        track_ids = tracks_df['id'].head(max_tracks).tolist()
        
        # Tải từng batch nhỏ hơn
        batch_size = 20  # Giảm kích thước batch từ 50 xuống 20
        all_features = []
        
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i+batch_size]
            progress_value = 0.5 + (i / len(track_ids)) * 0.3
            batch_desc = f"Tải đặc trưng âm thanh batch {i//batch_size + 1}/{len(track_ids)//batch_size + 1}..."
            progress(progress_value, desc=batch_desc)
            
            try:
                batch_features = fetcher.fetch_audio_features(batch)
                if not batch_features.empty:
                    all_features.append(batch_features)
            except Exception as e:
                logger.warning(f"Lỗi khi tải batch {i//batch_size + 1}: {str(e)}")
                # Tiếp tục với batch tiếp theo
            
            # Tạm dừng để tránh rate limit - tăng thời gian chờ
            time.sleep(2)  # Tăng từ 1s lên 2s
        
        # Kết hợp và lưu
        if all_features:
            features_df = pd.concat(all_features)
            features_df.to_csv(audio_features_path, index=False)
            
            progress(0.8, desc=f"Đã tải đặc trưng cho {len(features_df)} bài hát")
        else:
            progress(0.8, desc="Không thể tải đặc trưng âm thanh")
            logger.warning("Không thể tải đặc trưng âm thanh")
        
        # Tải thông tin thể loại nghệ sĩ với batch size nhỏ hơn
        genres_path = os.path.join(RAW_DATA_DIR, 'artist_genres.csv')
        
        if 'artist_id' in tracks_df.columns:
            artist_ids = tracks_df['artist_id'].dropna().unique().tolist()
            max_artists = min(50, len(artist_ids))  # Giảm từ 100 xuống 50
            artist_ids = artist_ids[:max_artists]
            
            # Tải thông tin nghệ sĩ theo batch
            try:
                progress(0.9, desc="Tải thông tin thể loại...")
                fetcher.fetch_artist_genres(artist_ids, save_path=genres_path, batch_size=20)  # Thêm batch_size=20
            except Exception as e:
                logger.warning(f"Lỗi khi tải thông tin thể loại: {str(e)}")
        
        progress(1.0, desc="Hoàn tất thiết lập dữ liệu ban đầu!")
        return f"✅ Dữ liệu ban đầu đã được thiết lập! Đã tải {len(tracks_df)} bài hát."
        
    except Exception as e:
        logger.error(f"Lỗi thiết lập dữ liệu ban đầu: {traceback.format_exc()}")
        return f"⚠️ Lỗi: {str(e)}"
    
def process_data(progress=gr.Progress()):
    """Xử lý dữ liệu thô thành đặc trưng cho mô hình"""
    # Kiểm tra dữ liệu thô
    tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
    audio_path = os.path.join(RAW_DATA_DIR, 'audio_features.csv')
    
    if not os.path.exists(tracks_path) or not os.path.exists(audio_path):
        return "⚠️ Không tìm thấy dữ liệu thô. Vui lòng thiết lập dữ liệu ban đầu trước."
    
    progress(0.1, desc="Khởi tạo bộ xử lý dữ liệu...")
    
    try:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Khởi tạo processor
        processor = DataProcessor()
        
        # Nạp dữ liệu
        progress(0.2, desc="Đọc dữ liệu thô...")
        processor.load_data()
        
        # Xử lý dữ liệu
        progress(0.4, desc="Làm sạch dữ liệu...")
        processor.clean_tracks_data()
        
        progress(0.5, desc="Kết hợp đặc trưng âm thanh...")
        processor.merge_audio_features()
        
        progress(0.6, desc="Kết hợp thông tin thể loại...")
        processor.merge_artist_genres()
        
        progress(0.7, desc="Trích xuất năm phát hành...")
        processor.extract_release_year()
        
        progress(0.8, desc="Tạo đặc trưng thể loại...")
        processor.create_genre_features()
        
        progress(0.9, desc="Chuẩn hóa đặc trưng...")
        processor.normalize_features()
        
        # Lưu dữ liệu đã xử lý
        track_features_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
        processor.tracks_df.to_csv(track_features_path, index=False)
        
        # Tạo ma trận user-item (tùy chọn)
        progress(0.95, desc="Tạo ma trận user-item...")
        user_matrix_path = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')
        processor.create_user_item_matrix(output_path=user_matrix_path)
        
        progress(1.0, desc="Hoàn tất xử lý dữ liệu!")
        return f"✅ Xử lý dữ liệu hoàn tất! Đã xử lý {len(processor.tracks_df)} bài hát."
        
    except Exception as e:
        logger.error(f"Lỗi xử lý dữ liệu: {traceback.format_exc()}")
        return f"⚠️ Lỗi: {str(e)}"

def train_models(progress=gr.Progress()):
    """Huấn luyện các mô hình đề xuất"""
    # Kiểm tra dữ liệu đã xử lý
    track_features_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
    
    if not os.path.exists(track_features_path):
        return "⚠️ Không tìm thấy dữ liệu đã xử lý. Vui lòng xử lý dữ liệu trước."
    
    progress(0.1, desc="Đọc dữ liệu đã xử lý...")
    
    try:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Nạp dữ liệu đã xử lý
        tracks_df = pd.read_csv(track_features_path)
        
        # Nạp ma trận user-item (nếu có)
        user_matrix_path = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')
        user_item_matrix = None
        
        if os.path.exists(user_matrix_path):
            try:
                user_item_matrix = pd.read_csv(user_matrix_path, index_col=0)
            except:
                logger.warning("Không thể đọc ma trận user-item, tiếp tục mà không có nó")
        
        # Huấn luyện Content-Based Model
        progress(0.3, desc="Huấn luyện mô hình content-based...")
        content_model = ContentBasedRecommender()
        content_model.train(tracks_df)
        content_model.save(os.path.join(MODELS_DIR, 'content_recommender.pkl'))
        
        # Huấn luyện Transition Model
        progress(0.5, desc="Huấn luyện mô hình transition...")
        transition_model = TransitionModel()
        transition_model.train(tracks_df)
        transition_model.save(os.path.join(MODELS_DIR, 'transition_model.pkl'))
        
        # Huấn luyện Hybrid Model
        progress(0.7, desc="Huấn luyện mô hình hybrid...")
        hybrid_model = HybridRecommender()
        hybrid_model.train(tracks_df, user_item_matrix)
        hybrid_model.save(os.path.join(MODELS_DIR, 'hybrid_recommender.pkl'))
        
        # Cập nhật biến toàn cục
        global model
        model = hybrid_model
        
        progress(1.0, desc="Hoàn tất huấn luyện mô hình!")
        return "✅ Các mô hình đã được huấn luyện thành công! Hệ thống đã sẵn sàng sử dụng."
        
    except Exception as e:
        logger.error(f"Lỗi huấn luyện mô hình: {traceback.format_exc()}")
        return f"⚠️ Lỗi: {str(e)}"

def setup_full_system(progress=gr.Progress()):
    """Thiết lập toàn bộ hệ thống trong một bước"""
    # Kiểm tra thông tin đăng nhập
    if not check_spotify_credentials():
        return "⚠️ Thiếu thông tin xác thực Spotify. Vui lòng thiết lập file .env."
    
    progress(0.05, desc="Bắt đầu thiết lập hệ thống...")
    
    try:
        # 1. Thiết lập dữ liệu ban đầu
        progress(0.1, desc="Đang thiết lập dữ liệu ban đầu...")
        dataset_result = setup_initial_dataset(progress, tracks_per_query=15)
        
        if "⚠️" in dataset_result:
            return dataset_result
        
        # 2. Xử lý dữ liệu
        progress(0.4, desc="Đang xử lý dữ liệu...")
        process_result = process_data(progress)
        
        if "⚠️" in process_result:
            return process_result
        
        # 3. Huấn luyện mô hình
        progress(0.7, desc="Đang huấn luyện mô hình...")
        training_result = train_models(progress)
        
        if "⚠️" in training_result:
            return training_result
        
        progress(1.0, desc="Thiết lập toàn bộ hệ thống hoàn tất!")
        return "✅ Thiết lập toàn bộ hệ thống hoàn tất! Hệ thống đề xuất đã sẵn sàng sử dụng."
        
    except Exception as e:
        logger.error(f"Lỗi thiết lập hệ thống: {traceback.format_exc()}")
        return f"⚠️ Lỗi: {str(e)}"

def create_ui():
    """Tạo giao diện người dùng Gradio"""
    with gr.Blocks(title="Hệ thống Đề xuất Âm nhạc") as app:
        # Tab Trang chủ
        with gr.Tab("Trang chủ"):
            gr.Markdown("# Hệ thống Đề xuất Âm nhạc")
            gr.Markdown("Chào mừng đến với hệ thống đề xuất âm nhạc thông minh!")
            
            # Trạng thái hệ thống
            gr.Markdown("## Trạng thái hệ thống")
            
            with gr.Row():
                with gr.Column():
                    api_status = gr.Markdown("Trạng thái kết nối API: Chưa kiểm tra")
                    model_status = gr.Markdown("Trạng thái mô hình: Chưa kiểm tra")
                
                with gr.Column():
                    check_api_button = gr.Button("Kiểm tra kết nối Spotify API")
                    check_model_button = gr.Button("Kiểm tra trạng thái mô hình")
            
            check_api_button.click(fn=test_spotify_connection, inputs=[], outputs=api_status)
            check_model_button.click(fn=load_model, inputs=[], outputs=model_status)
        
        # Tab Thiết lập hệ thống
        with gr.Tab("Thiết lập hệ thống"):
            gr.Markdown("# Thiết lập hệ thống")
            gr.Markdown("Thiết lập dữ liệu và huấn luyện mô hình trước khi sử dụng")
            
            with gr.Row():
                with gr.Column():
                    setup_data_btn = gr.Button("1. Thiết lập dữ liệu ban đầu")
                    process_data_btn = gr.Button("2. Xử lý dữ liệu")
                    train_models_btn = gr.Button("3. Huấn luyện mô hình")
                    setup_all_btn = gr.Button("Thiết lập toàn bộ hệ thống (1+2+3)", variant="primary")
                    
                    tracks_per_query = gr.Slider(minimum=5, maximum=50, value=15, step=5,
                                              label="Số bài hát mỗi truy vấn")
                
                with gr.Column():
                    setup_output = gr.Markdown("Kết quả sẽ hiển thị ở đây...")
            
            # Kết nối các nút với hàm tương ứng
            setup_data_btn.click(fn=setup_initial_dataset, inputs=[tracks_per_query], outputs=setup_output)
            process_data_btn.click(fn=process_data, inputs=[], outputs=setup_output)
            train_models_btn.click(fn=train_models, inputs=[], outputs=setup_output)
            setup_all_btn.click(fn=setup_full_system, inputs=[], outputs=setup_output)
            
        # Tab Đề xuất
        with gr.Tab("Đề xuất"):
            with gr.Row():
                with gr.Column():
                    song_input = gr.Textbox(label="Nhập tên bài hát")
                    artist_input = gr.Textbox(label="Nhập tên nghệ sĩ (tùy chọn)")
                    queue_length = gr.Slider(minimum=5, maximum=20, value=10, step=1, 
                                           label="Số lượng bài hát đề xuất")
                    recommend_btn = gr.Button("Đề xuất bài hát")
                
                with gr.Column():
                    result_output = gr.Markdown("Kết quả đề xuất sẽ hiển thị ở đây")
            
            recommend_btn.click(
                fn=generate_queue,
                inputs=[song_input, artist_input, queue_length],
                outputs=result_output
            )
        
        # Tab Tối ưu hóa hàng đợi
        with gr.Tab("Tối ưu hóa hàng đợi"):
            gr.Markdown("# Tối ưu hóa hàng đợi phát nhạc")
            gr.Markdown("Nhập danh sách bài hát để tối ưu thứ tự phát tạo trải nghiệm nghe tốt nhất")
            
            with gr.Row():
                with gr.Column():
                    queue_input = gr.Textbox(
                        label="Danh sách bài hát (cách nhau bởi dấu phẩy)",
                        placeholder="Last Christmas - Wham, Shape of You - Ed Sheeran, Blinding Lights - The Weeknd",
                        lines=5
                    )
                    optimize_btn = gr.Button("Tối ưu hóa hàng đợi")
                
                with gr.Column():
                    optimize_output = gr.Markdown("Kết quả tối ưu hóa sẽ hiển thị ở đây")
            
            optimize_btn.click(
                fn=optimize_music_queue,
                inputs=[queue_input],
                outputs=optimize_output
            )
            
        # Tab Tìm kiếm bài hát tương tự
        with gr.Tab("Tìm kiếm tương tự"):
            gr.Markdown("# Tìm kiếm bài hát tương tự")
            gr.Markdown("Tìm các bài hát tương tự với bài hát bạn yêu thích")
            
            with gr.Row():
                with gr.Column():
                    similar_song_input = gr.Textbox(label="Nhập tên bài hát")
                    similar_artist_input = gr.Textbox(label="Nhập tên nghệ sĩ (tùy chọn)")
                    similar_count = gr.Slider(minimum=5, maximum=20, value=10, step=1, 
                                            label="Số lượng bài hát tương tự")
                    similar_btn = gr.Button("Tìm bài hát tương tự")
                
                with gr.Column():
                    similar_output = gr.Markdown("Kết quả tìm kiếm sẽ hiển thị ở đây")
            
            similar_btn.click(
                fn=lambda song, artist, count: get_song_recommendations(song, artist, count)[1],  # Lấy phần văn bản
                inputs=[similar_song_input, similar_artist_input, similar_count],
                outputs=similar_output
            )
            
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)