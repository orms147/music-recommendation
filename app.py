import gradio as gr
import pandas as pd
import os
import logging
import time
import plotly.express as px
import matplotlib.pyplot as plt
from config.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.data_fetcher import SpotifyDataFetcher, fetch_initial_dataset
from utils.data_processor import DataProcessor
from utils.visualization import MusicVisualizer
from models.content_model import ContentBasedRecommender
from models.collaborative_model import CollaborativeFilteringRecommender
from models.sequence_model import SequenceBasedRecommender
from models.hybrid_model import HybridRecommender
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Đường dẫn đến model đã huấn luyện
MODEL_PATH = os.path.join("models", "hybrid_recommender.pkl")

# Kiểm tra và tạo thư mục nếu chưa tồn tại
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, "models", "logs"]:
    os.makedirs(dir_path, exist_ok=True)

def check_setup():
    """Kiểm tra xem hệ thống đã được cài đặt hay chưa"""
    raw_data_exists = os.path.exists(os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv'))
    processed_data_exists = os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'track_features.csv'))
    model_exists = os.path.exists(MODEL_PATH)
    
    return raw_data_exists, processed_data_exists, model_exists

def setup_system(progress=gr.Progress()):
    """Cài đặt toàn bộ hệ thống"""
    progress(0, desc="Kiểm tra dữ liệu...")
    
    # Kiểm tra xem đã có dữ liệu chưa
    raw_data_exists = os.path.exists(os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv'))
    
    # Tải dữ liệu nếu chưa có
    if not raw_data_exists:
        progress(0.1, desc="Tải dữ liệu từ Spotify API...")
        fetch_initial_dataset()
    
    progress(0.3, desc="Xử lý dữ liệu...")
    # Xử lý dữ liệu
    processor = DataProcessor()
    processor.load_data()
    processor.process_all()
    
    progress(0.6, desc="Huấn luyện mô hình...")
    # Huấn luyện mô hình
    tracks_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'track_features.csv'))
    
    # Tải ma trận người dùng-bài hát nếu có
    user_item_matrix = None
    matrix_path = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')
    if os.path.exists(matrix_path):
        user_item_matrix = pd.read_csv(matrix_path, index_col=0)
    
    # Huấn luyện mô hình hybrid
    hybrid_model = HybridRecommender()
    hybrid_model.train(tracks_df, user_item_matrix)
    
    # Lưu mô hình
    progress(0.9, desc="Lưu mô hình...")
    hybrid_model.save(MODEL_PATH)
    
    progress(1.0, desc="Hoàn tất!")
    return "Hệ thống đã được cài đặt thành công! Bạn có thể bắt đầu sử dụng các tính năng đề xuất."

def get_song_recommendations(song_name, artist_name="", num_recommendations=10):
    """Lấy đề xuất bài hát dựa trên tên bài hát"""
    if not song_name:
        return pd.DataFrame(), "Vui lòng nhập tên bài hát"
    
    # Kiểm tra mô hình đã huấn luyện chưa
    if not os.path.exists(MODEL_PATH):
        return pd.DataFrame(), "Mô hình chưa được huấn luyện. Vui lòng cài đặt hệ thống trước."
    
    # Tải mô hình
    model = HybridRecommender.load(MODEL_PATH)
    
    # Lấy đề xuất
    recommendations = model.recommend(
        track_name=song_name,
        artist=artist_name,
        n_recommendations=num_recommendations
    )
    
    if recommendations.empty:
        return None, f"Không tìm thấy bài hát '{song_name}'. Vui lòng thử lại với bài hát khác."
    
    # Format kết quả
    output_text = f"### Đề xuất cho bài hát: '{song_name}'{' - ' + artist_name if artist_name else ''}\n\n"
    
    # Kiểm tra các cột tồn tại
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        track_info = ""
        if 'name' in row and 'artist' in row:
            track_info = f"**{row['name']}** - {row['artist']}"
        elif 'id' in row:
            track_info = f"Track ID: {row['id']}"
        else:
            track_info = f"Đề xuất #{i}"
            
        output_text += f"{i}. {track_info}\n"
    
    # Trước khi trả về, kiểm tra và hiển thị cảnh báo nếu có bài hát "Unknown"
    if 'name' in recommendations.columns and recommendations['name'].isin(['Unknown']).any():
        logger.warning("Some recommendations have unknown track information")
    
    return recommendations, output_text

def get_sequence_recommendations(recent_tracks_input, num_recommendations=10):
    """Lấy đề xuất dựa trên chuỗi bài hát gần đây"""
    if not recent_tracks_input:
        return pd.DataFrame(), "Vui lòng nhập các bài hát gần đây"
    
    # Kiểm tra mô hình đã huấn luyện chưa
    if not os.path.exists(MODEL_PATH):
        return pd.DataFrame(), "Mô hình chưa được huấn luyện. Vui lòng cài đặt hệ thống trước."
    
    # Tách danh sách bài hát
    recent_tracks_list = [track.strip() for track in recent_tracks_input.split(',')]
    
    if len(recent_tracks_list) < 2:
        return pd.DataFrame(), "Vui lòng nhập ít nhất 2 bài hát, cách nhau bởi dấu phẩy"
    
    # Tải track features để tìm track IDs
    tracks_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'track_features.csv'))
    
    # Tìm track IDs
    track_ids = []
    not_found = []
    
    for track_name in recent_tracks_list:
        # Tìm kiếm không phân biệt chữ hoa/thường
        matches = tracks_df[tracks_df['name'].str.lower() == track_name.lower()]
        
        if matches.empty:
            # Thử tìm kiếm một phần
            matches = tracks_df[tracks_df['name'].str.lower().str.contains(track_name.lower())]
            
        if not matches.empty:
            track_ids.append(matches.iloc[0]['id'])
        else:
            not_found.append(track_name)
    
    if not track_ids:
        return pd.DataFrame(), "Không tìm thấy bài hát nào trong danh sách. Vui lòng thử lại."
    
    # Tải mô hình
    model = HybridRecommender.load(MODEL_PATH)
    
    # Lấy đề xuất
    recommendations = model.recommend(
        recent_tracks=track_ids,
        n_recommendations=num_recommendations
    )
    
    if recommendations.empty:
        return None, "Không thể tạo đề xuất cho chuỗi bài hát này."
    
    # Format kết quả
    output_text = f"### Đề xuất dựa trên {len(track_ids)} bài hát gần đây\n\n"
    
    if not_found:
        output_text += f"*Không tìm thấy: {', '.join(not_found)}*\n\n"
    
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        output_text += f"{i}. **{row['name']}** - {row['artist']}\n"
    
    return recommendations, output_text

def get_track_recommendations(selected_track_index, catalog_df=None):
    """Lấy đề xuất cho bài hát được chọn từ danh mục"""
    # Kiểm tra catalog_df có tồn tại hay không
    if catalog_df is None or isinstance(catalog_df, type(None)):
        return pd.DataFrame(), "Không có dữ liệu danh mục bài hát"
    
    if catalog_df.empty or selected_track_index >= len(catalog_df):
        return pd.DataFrame(), f"Không tìm thấy bài hát với chỉ số {selected_track_index}"
    
    # Lấy thông tin bài hát đã chọn
    selected_track = catalog_df.iloc[selected_track_index]
    
    # Lấy đề xuất
    return get_song_recommendations(
        selected_track.get('name', ''), 
        selected_track.get('artist', ''), 
        10
    )

def visualize_recommendations(recommendations_df):
    """Trực quan hóa các đề xuất"""
    if recommendations_df is None or recommendations_df.empty:
        return None
    
    # Tải dữ liệu đặc trưng
    tracks_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
    if not os.path.exists(tracks_path):
        return None
    
    tracks_df = pd.read_csv(tracks_path)
    
    # Tạo visualizer
    visualizer = MusicVisualizer(tracks_df)
    
    # Lấy bài hát đầu vào (giả sử là bài hát đầu tiên trong recommendations)
    input_track_data = {"id": recommendations_df.iloc[0]['id']}
    
    # Trực quan hóa đề xuất
    fig = visualizer.visualize_recommendations(input_track_data, recommendations_df)
    
    return fig

def fetch_new_data(query_input, num_tracks=50):
    """Tải thêm dữ liệu từ Spotify API"""
    if not query_input:
        return "Vui lòng nhập từ khóa tìm kiếm"
    
    # Kiểm tra credentials
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return "Thiếu thông tin xác thực Spotify API. Vui lòng kiểm tra file config."
    
    # Thay vì genres, sử dụng các query tìm kiếm
    queries = [q.strip() for q in query_input.split(',')]
    
    # Tạo fetcher
    fetcher = SpotifyDataFetcher()
    
    # Tải dữ liệu - sửa hàm fetch_tracks_by_genres thành một hàm tìm kiếm chung
    tracks_df = fetcher.fetch_tracks_by_search(
        queries,
        tracks_per_query=num_tracks,
        save_path=os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
    )
    
    if tracks_df.empty:
        return "Không thể tải dữ liệu. Vui lòng kiểm tra logs để biết thêm chi tiết."
    
    # Xử lý dữ liệu
    processor = DataProcessor()
    processor.load_data()
    processor.process_all()
    
    # Huấn luyện lại mô hình
    tracks_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'track_features.csv'))
    
    user_item_matrix = None
    matrix_path = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')
    if os.path.exists(matrix_path):
        user_item_matrix = pd.read_csv(matrix_path, index_col=0)
    
    # Huấn luyện mô hình hybrid
    hybrid_model = HybridRecommender()
    hybrid_model.train(tracks_df, user_item_matrix)
    
    # Lưu mô hình
    hybrid_model.save(MODEL_PATH)
    
    return f"Đã tải thành công dữ liệu cho {len(queries)} từ khóa tìm kiếm. Tổng số bài hát: {len(tracks_df)}"

def analyze_music_data():
    """Phân tích dữ liệu âm nhạc"""
    # Kiểm tra dữ liệu đã xử lý
    tracks_path = os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')
    if not os.path.exists(tracks_path):
        return None, None, "Dữ liệu chưa được xử lý. Vui lòng cài đặt hệ thống trước."
    
    # Tải dữ liệu
    tracks_df = pd.read_csv(tracks_path)
    
    # Tạo visualizer
    visualizer = MusicVisualizer(tracks_df)
    
    # Tạo biểu đồ phân phối đặc trưng
    feature_dist_fig = visualizer.visualize_feature_distributions()
    
    # Nhúng bài hát trong không gian 2D
    embedding_fig = visualizer.visualize_track_embedding(method='tsne')
    
    # Tạo dashboard
    dashboard_fig = visualizer.create_dashboard()
    
    return feature_dist_fig, embedding_fig, dashboard_fig

def recommend_and_visualize(song_name, artist_name, num_recommendations):
    """Lấy và hiển thị đề xuất bài hát"""
    # Chuẩn hóa tên bài hát và nghệ sĩ để tìm kiếm tốt hơn
    if song_name:
        song_name = song_name.strip()
    if artist_name:
        artist_name = artist_name.strip()
        
    logger.info(f"Finding recommendations for: '{song_name}' by '{artist_name}'")
    
    # Kiểm tra bài hát có tồn tại không
    try:
        tracks_path = os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
        if os.path.exists(tracks_path):
            tracks_df = pd.read_csv(tracks_path)
            matches = tracks_df[tracks_df['name'].str.lower() == song_name.lower()]
            
            if artist_name and not matches.empty:
                artist_matches = matches[matches['artist'].str.lower() == artist_name.lower()]
                if not artist_matches.empty:
                    matches = artist_matches
                    
            if not matches.empty:
                logger.info(f"Found match: {matches.iloc[0]['name']} by {matches.iloc[0]['artist']}")
                
                # Sử dụng tên và nghệ sĩ chính xác từ dữ liệu
                song_name = matches.iloc[0]['name']
                artist_name = matches.iloc[0]['artist']
    except Exception as e:
        logger.error(f"Error checking song existence: {e}")
    
    # Lấy đề xuất
    df, text = get_song_recommendations(song_name, artist_name, num_recommendations)
    
    # Không cần trực quan hóa vì thiếu đặc trưng âm thanh
    fig = None
    
    return df, text, fig

def get_recommendations_for_selected(selected_index, catalog_df=None):
    """Lấy đề xuất cho bài hát được chọn từ bảng dữ liệu"""
    # Xử lý trường hợp không có dữ liệu
    if catalog_df is None:
        return pd.DataFrame(), "Không có dữ liệu danh mục bài hát"
    
    # Chuyển về số nguyên
    try:
        adjusted_idx = int(selected_index) if selected_index is not None else 0
    except (ValueError, TypeError):
        adjusted_idx = 0
    
    # Kiểm tra giới hạn
    if not catalog_df.empty and 0 <= adjusted_idx < len(catalog_df):
        return get_track_recommendations(adjusted_idx, catalog_df)
    else:
        return pd.DataFrame(), "Vui lòng chọn một bài hát từ danh sách"

# Khởi tạo giao diện Gradio
with gr.Blocks(title="Hệ thống Đề xuất Âm nhạc") as app:
    gr.Markdown("# 🎵 Hệ thống Đề xuất Âm nhạc AI")
    
    with gr.Tab("Thiết lập Hệ thống"):
        raw_data_exists, processed_data_exists, model_exists = check_setup()
        
        gr.Markdown(f"""
        ### Trạng thái hệ thống:
        - Dữ liệu thô: {"✅ Đã có" if raw_data_exists else "❌ Chưa có"}
        - Dữ liệu đã xử lý: {"✅ Đã có" if processed_data_exists else "❌ Chưa có"}
        - Mô hình đã huấn luyện: {"✅ Đã có" if model_exists else "❌ Chưa có"}
        
        Nếu bạn chưa thiết lập hệ thống, vui lòng nhấn nút "Thiết lập Hệ thống" bên dưới.
        """)
        
        setup_btn = gr.Button("Thiết lập Hệ thống")
        setup_output = gr.Textbox(label="Trạng thái")
        
        setup_btn.click(setup_system, inputs=[], outputs=setup_output)
    
    with gr.Tab("Đề xuất Bài hát"):
        with gr.Row():
            with gr.Column():
                song_name = gr.Textbox(label="Tên bài hát", placeholder="Shape of You")
                artist_name = gr.Textbox(label="Tên nghệ sĩ (tùy chọn)", placeholder="Ed Sheeran")
                num_recommendations = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Số lượng đề xuất")
                recommend_btn = gr.Button("Tìm bài hát tương tự")
            
            with gr.Column():
                recommendation_output = gr.Markdown(label="Kết quả đề xuất")
        
        # Biến ẩn để lưu DataFrame kết quả
        recommendation_df = gr.State()
        
        recommend_btn.click(
            recommend_and_visualize, 
            inputs=[song_name, artist_name, num_recommendations], 
            outputs=[recommendation_df, recommendation_output]
        )
    
    with gr.Tab("Đề xuất Dựa trên Trình tự"):
        with gr.Column():
            recent_tracks = gr.Textbox(
                label="Các bài hát gần đây", 
                placeholder="Shape of You, Perfect, Thinking Out Loud",
                info="Nhập danh sách bài hát, cách nhau bởi dấu phẩy"
            )
            sequence_num_recommendations = gr.Slider(
                minimum=1, maximum=20, value=10, step=1, 
                label="Số lượng đề xuất"
            )
            sequence_recommend_btn = gr.Button("Tìm bài hát tiếp theo")
            sequence_output = gr.Markdown(label="Kết quả đề xuất")
        
        sequence_recommend_btn.click(
            get_sequence_recommendations, 
            inputs=[recent_tracks, sequence_num_recommendations], 
            outputs=[gr.State(), sequence_output]
        )
    
    with gr.Tab("Quản lý Dữ liệu"):
        with gr.Column():
            gr.Markdown("""
            ### Tải thêm dữ liệu từ Spotify API
            
            Nhập danh sách từ khóa tìm kiếm bạn muốn tải, cách nhau bởi dấu phẩy.
            
            *Ví dụ: pop 2023, rock hits, vietnamese music*
            """)
            
            search_input = gr.Textbox(
                label="Từ khóa tìm kiếm", 
                placeholder="pop 2023, rock hits, vietnamese music"
            )
            tracks_per_query = gr.Slider(
                minimum=10, maximum=100, value=50, step=10, 
                label="Số lượng bài hát mỗi từ khóa"
            )
            fetch_btn = gr.Button("Tải dữ liệu")
            fetch_output = gr.Textbox(label="Trạng thái")
        
        fetch_btn.click(
            fetch_new_data, 
            inputs=[search_input, tracks_per_query], 
            outputs=fetch_output
        )

# Chạy ứng dụng
if __name__ == "__main__":
    # Kiểm tra xem đã có dữ liệu chưa, nếu chưa thì chuẩn bị thông báo
    raw_exists, processed_exists, model_exists = check_setup()
    
    if not raw_exists or not processed_exists or not model_exists:
        try:
            print("CẢNH BÁO: Hệ thống chưa được thiết lập đầy đủ!")
            print("Vui lòng chuyển đến tab 'Thiết lập Hệ thống' và nhấn nút 'Thiết lập Hệ thống'.")
        except UnicodeEncodeError:
            print("WARNING: System not fully set up!")
            print("Please go to the 'System Setup' tab and click the 'Setup System' button.")
    
    # Khởi chạy ứng dụng Gradio
    app.launch(share=True)