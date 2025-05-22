import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from config.config import CONTENT_FEATURES

logger = logging.getLogger(__name__)

class MusicVisualizer:
    """Công cụ trực quan hóa dữ liệu âm nhạc và kết quả đề xuất"""
    
    def __init__(self, tracks_df=None):
        """Khởi tạo visualizer với DataFrame bài hát"""
        self.tracks_df = tracks_df
    
    def set_data(self, tracks_df):
        """Cài đặt dữ liệu bài hát"""
        self.tracks_df = tracks_df
        return self
    
    def visualize_feature_distributions(self, features=None, save_path=None):
        """Vẽ phân phối của các đặc trưng âm nhạc cơ bản"""
        if self.tracks_df is None:
            logger.error("No tracks data set")
            return None
        
        # Chỉ sử dụng đặc trưng cơ bản
        features = ['popularity', 'duration_ms']
        if 'release_year' in self.tracks_df.columns:
            features.append('release_year')
        if 'explicit' in self.tracks_df.columns:
            features.append('explicit')
        
        # Chỉ lấy các đặc trưng có sẵn trong dữ liệu
        available_features = [f for f in features if f in self.tracks_df.columns]
        
        if not available_features:
            logger.error("No valid features to visualize")
            return None
        
        # Tạo biểu đồ phân phối
        n_features = len(available_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features):
            ax = axes[i]
            sns.histplot(self.tracks_df[feature], kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            ax.grid(True, alpha=0.3)
        
        # Ẩn các biểu đồ thừa
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved feature distributions to {save_path}")
        
        return fig
    
    def visualize_genre_distribution(self, save_path=None):
        """Vô hiệu hóa vì không có dữ liệu thể loại"""
        logger.warning("No genre data available - skipping genre visualization")
        return None
    
    def visualize_track_embedding(self, n_components=2, method='pca', save_path=None):
        """Trực quan hóa bài hát trong không gian 2D sử dụng PCA hoặc t-SNE"""
        if self.tracks_df is None:
            logger.error("No tracks data set")
            return None
            
        # Chọn đặc trưng để giảm chiều
        features = [f for f in CONTENT_FEATURES if f in self.tracks_df.columns]
        
        if not features:
            logger.error("No valid features for dimensionality reduction")
            return None
        
        # Lấy dữ liệu đặc trưng
        X = self.tracks_df[features].values
        
        # Giảm chiều
        if method.lower() == 'pca':
            model = PCA(n_components=n_components, random_state=42)
            logger.info("Using PCA for dimensionality reduction")
        else:  # t-SNE
            model = TSNE(n_components=n_components, random_state=42, perplexity=30, learning_rate=200)
            logger.info("Using t-SNE for dimensionality reduction")
            
        # Áp dụng giảm chiều
        X_reduced = model.fit_transform(X)
        
        # Tạo DataFrame mới với kết quả
        reduced_df = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1] if n_components > 1 else np.zeros(len(X_reduced))
        })
        
        # Thêm thông tin bài hát và thể loại nếu có
        reduced_df['name'] = self.tracks_df['name'].values
        reduced_df['artist'] = self.tracks_df['artist'].values
        
        if 'genre' in self.tracks_df.columns:
            reduced_df['genre'] = self.tracks_df['genre'].values
            
            # Vẽ biểu đồ tương tác với Plotly
            fig = px.scatter(
                reduced_df, x='x', y='y', color='genre',
                hover_data=['name', 'artist'],
                title=f'Track Embedding using {method.upper()}',
                labels={'x': f'{method.upper()} Dimension 1', 'y': f'{method.upper()} Dimension 2'}
            )
        else:
            # Vẽ biểu đồ không có thể loại
            fig = px.scatter(
                reduced_df, x='x', y='y',
                hover_data=['name', 'artist'],
                title=f'Track Embedding using {method.upper()}',
                labels={'x': f'{method.upper()} Dimension 1', 'y': f'{method.upper()} Dimension 2'}
            )
        
        # Cải thiện layout
        fig.update_layout(
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            width=900, height=700
        )
        
        # Lưu biểu đồ nếu có đường dẫn
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved track embedding visualization to {save_path}")
        
        return fig
    
    def visualize_recommendations(self, input_track, recommendations, save_path=None):
        """Trực quan hóa mối quan hệ giữa bài hát đầu vào và các đề xuất"""
        if self.tracks_df is None or recommendations is None or recommendations.empty:
            logger.error("Missing data for recommendation visualization")
            return None
        
        # Tìm bài hát đầu vào trong DataFrame
        input_track_data = None
        if isinstance(input_track, str):
            # Tìm theo tên
            input_track_data = self.tracks_df[self.tracks_df['name'] == input_track]
        elif isinstance(input_track, dict) and 'id' in input_track:
            # Tìm theo ID
            input_track_data = self.tracks_df[self.tracks_df['id'] == input_track['id']]
            
        if input_track_data is None or input_track_data.empty:
            logger.error("Input track not found in dataset")
            return None
        
        # Kết hợp bài hát đầu vào và các đề xuất
        recommendations['type'] = 'Recommendation'
        input_track_data['type'] = 'Input Track'
        all_tracks = pd.concat([input_track_data, recommendations[input_track_data.columns.tolist() + ['type']]])
        
        # Vì không có đặc trưng âm thanh, chúng ta sẽ sử dụng các đặc trưng khác
        features = ['popularity', 'duration_ms']
        if 'explicit' in all_tracks.columns:
            features.append('explicit')
        if 'release_year' in all_tracks.columns:
            features.append('release_year')
        
        features = [f for f in features if f in all_tracks.columns]
        
        if len(features) < 2:
            logger.error("Not enough features for visualization")
            return None
        
        # Thay vì radar chart, sử dụng scatter plot
        fig = px.scatter(
            all_tracks, x=features[0], y=features[1], 
            color='type', hover_data=['name', 'artist'],
            title=f"Recommendations for {input_track_data.iloc[0]['name']} by {input_track_data.iloc[0]['artist']}",
            labels={features[0]: features[0].capitalize(), features[1]: features[1].capitalize()}
        )
        
        # Cải thiện layout
        fig.update_layout(
            width=800,
            height=600,
            legend_title_text="Track Type"
        )
        
        # Lưu biểu đồ nếu có đường dẫn
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved recommendation visualization to {save_path}")
        
        return fig
    
    def create_dashboard(self, recommendations=None, save_path=None):
        """Tạo dashboard tương tác với nhiều biểu đồ"""
        if self.tracks_df is None:
            logger.error("No tracks data set")
            return None
            
        # Tạo biểu đồ với nhiều cửa sổ
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Audio Feature Distributions',
                'Genre Distribution',
                'Feature Correlations',
                'Popularity vs Energy'
            ),
            specs=[
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Box plot cho các đặc trưng âm thanh
        features = ['danceability', 'energy', 'valence', 'acousticness']
        features = [f for f in features if f in self.tracks_df.columns]
        
        for i, feature in enumerate(features):
            fig.add_trace(
                go.Box(
                    y=self.tracks_df[feature],
                    name=feature,
                    boxmean=True
                ),
                row=1, col=1
            )
        
        # 2. Biểu đồ cột cho thể loại
        if 'genre' in self.tracks_df.columns:
            genre_counts = self.tracks_df['genre'].value_counts().head(10)
            fig.add_trace(
                go.Bar(
                    x=genre_counts.index,
                    y=genre_counts.values,
                    text=genre_counts.values,
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        # 3. Bản đồ nhiệt cho mối tương quan
        corr_features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo', 'loudness']
        corr_features = [f for f in corr_features if f in self.tracks_df.columns]
        
        if len(corr_features) > 1:
            corr = self.tracks_df[corr_features].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale='Viridis',
                    hoverongaps=False
                ),
                row=2, col=1
            )
        
        # 4. Biểu đồ phân tán cho popularity vs energy
        if 'popularity' in self.tracks_df.columns and 'energy' in self.tracks_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.tracks_df['energy'],
                    y=self.tracks_df['popularity'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        opacity=0.6,
                        color='steelblue'
                    ),
                    hovertext=self.tracks_df['name'] + ' by ' + self.tracks_df['artist']
                ),
                row=2, col=2
            )
        
        # Cập nhật layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Music Dataset Dashboard",
            showlegend=False
        )
        
        # Điều chỉnh các trục
        fig.update_xaxes(title_text="Genre", row=1, col=2)
        fig.update_xaxes(title_text="Energy", row=2, col=2)
        fig.update_yaxes(title_text="Popularity", row=2, col=2)
        
        # Lưu dashboard nếu có đường dẫn
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved dashboard to {save_path}")
        
        return fig