import os
import logging
import traceback
import pandas as pd
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

from config.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    DEFAULT_TRACKS_PER_QUERY, MAX_TRACKS_PER_QUERY, 
    MIN_TRACKS_PER_QUERY, TRACKS_QUERY_STEP,
    LARGE_DATASET_DEFAULT_SIZE, LARGE_DATASET_BATCH_SIZE, LARGE_DATASET_SAVE_INTERVAL
)
from utils.data_fetcher import fetch_initial_dataset
from utils.data_processor import DataProcessor
from models.enhanced_content_model import EnhancedContentRecommender
from models.weighted_content_model import WeightedContentRecommender

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# Global model variables
model = None
weighted_model = None

def initialize_model():
    """Initialize models on app startup"""
    global model, weighted_model
    
    # ✅ Fixed file paths aligned with actual data structure
    model_path = os.path.join(MODELS_DIR, 'enhanced_content_recommender.pkl')
    weighted_model_path = os.path.join(MODELS_DIR, 'weighted_content_recommender.pkl')

    # EnhancedContentRecommender
    if os.path.exists(model_path):
        try:
            model = EnhancedContentRecommender.load(model_path)
            logger.info(f"Loaded EnhancedContentRecommender from {model_path}")
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")
            model = None

    # WeightedContentRecommender
    if os.path.exists(weighted_model_path):
        try:
            weighted_model = WeightedContentRecommender.load(weighted_model_path)
            logger.info(f"Loaded WeightedContentRecommender from {weighted_model_path}")
        except Exception as e:
            logger.error(f"Error loading weighted model: {e}")
            weighted_model = None

def check_spotify_credentials():
    """Check if Spotify credentials are configured"""
    from config.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
    return bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)

def setup_initial_dataset(progress=gr.Progress(), tracks_per_query=DEFAULT_TRACKS_PER_QUERY):
    """Setup initial dataset with progress tracking"""
    if not check_spotify_credentials():
        return "⚠️ Missing Spotify credentials. Please setup .env file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    progress(0.1, desc="Checking existing data...")
    
    try:
        # ✅ Check actual file paths from data processor
        processed_path = os.path.join(PROCESSED_DATA_DIR, 'processed_tracks.csv')
        
        if os.path.exists(processed_path):
            existing_df = pd.read_csv(processed_path)
            progress(0.3, desc=f"Found {len(existing_df)} existing tracks...")
            
            if len(existing_df) >= 1000:  # Already have sufficient data
                # ✅ THÊM: Check ISRC coverage
                isrc_coverage = (existing_df.get('isrc', pd.Series()) != '').sum() / len(existing_df)
                
                if isrc_coverage < 0.5:  # Low ISRC coverage
                    progress(0.4, desc="Enhancing data with Spotify API...")
                    processor = DataProcessor()
                    processor.load_data()  # Load existing data
                    processor.enrich_with_spotify_api()  # Enrich with API
                    processor.extract_cultural_features()  # Re-extract with new ISRC
                    processor.save_processed_data()
                    progress(0.7, desc="ISRC data enhanced!")
                
                progress(1.0, desc="Dataset ready!")
                
                # ✅ Check cultural intelligence features
                cultural_features = ['music_culture', 'is_vietnamese', 'is_korean', 'is_japanese']
                available_cultural = [f for f in cultural_features if f in existing_df.columns]
                
                if len(available_cultural) >= 3:
                    return f"✅ Dataset ready with {len(existing_df)} tracks and {len(available_cultural)} cultural intelligence features!"
                else:
                    progress(0.5, desc="Reprocessing for cultural features...")
                    processor = DataProcessor()
                    processor.process_all()
                    progress(1.0, desc="Cultural features updated!")
                    return f"✅ Updated dataset with {len(existing_df)} tracks and enhanced cultural intelligence!"
        
        # Fetch new data
        progress(0.2, desc="Fetching tracks from Spotify...")
        success = fetch_initial_dataset(tracks_per_query=tracks_per_query)
        
        if not success:
            return "❌ Failed to fetch tracks from Spotify API."
        
        progress(0.6, desc="Processing data with cultural intelligence...")
        processor = DataProcessor()
        success = processor.process_all()
        
        if not success:
            return "❌ Failed to process data."
        
        # ✅ Check final result
        if os.path.exists(processed_path):
            final_df = pd.read_csv(processed_path)
            progress(1.0, desc="Setup complete!")
            
            # ✅ Report on cultural intelligence
            cultural_dist = final_df.get('music_culture', pd.Series()).value_counts().to_dict()
            isrc_coverage = (final_df.get('isrc', pd.Series()) != '').sum() if 'isrc' in final_df.columns else 0
            
            return f"""✅ Dataset setup successful!
            
**📊 Dataset Stats:**
- **Total tracks:** {len(final_df):,}
- **ISRC coverage:** {isrc_coverage}/{len(final_df)} ({isrc_coverage/len(final_df)*100:.1f}%)
- **Cultural distribution:** {cultural_dist}

**🧠 Cultural Intelligence Features:**
- ISRC-based culture classification ✅
- Market penetration analysis ✅  
- Cross-cultural similarity ✅
"""
        else:
            return "❌ Data processing completed but no output file found."
        
    except Exception as e:
        logger.error(f"Error in setup_initial_dataset: {e}\n{traceback.format_exc()}")
        return f"❌ Setup error: {str(e)}"

def train_model():
    """Train or load recommendation models"""
    global model, weighted_model
    
    try:
        # ✅ Fixed file paths
        enhanced_model_path = os.path.join(MODELS_DIR, 'enhanced_content_recommender.pkl')
        weighted_model_path = os.path.join(MODELS_DIR, 'weighted_content_recommender.pkl')
        processed_path = os.path.join(PROCESSED_DATA_DIR, 'processed_tracks.csv')

        # Check data availability
        if not os.path.exists(processed_path):
            return "❌ No processed data found. Please setup dataset first."
        
        tracks_df = pd.read_csv(processed_path)
        logger.info(f"Training with {len(tracks_df)} tracks and {len(tracks_df.columns)} features")
        
        # ✅ Validate required features
        required_features = ['id', 'name', 'artist', 'popularity']
        missing_features = [f for f in required_features if f not in tracks_df.columns]
        if missing_features:
            return f"❌ Missing required features: {missing_features}. Please reprocess data."
        
        results = []
        
        # Train EnhancedContentRecommender
        if os.path.exists(enhanced_model_path):
            logger.info("Loading existing EnhancedContentRecommender...")
            model = EnhancedContentRecommender.load(enhanced_model_path)
            results.append("✅ EnhancedContentRecommender loaded")
        else:
            logger.info("Training new EnhancedContentRecommender...")
            model = EnhancedContentRecommender()
            success = model.train(tracks_df)
            if success:
                model.save(enhanced_model_path)
                results.append("✅ EnhancedContentRecommender trained and saved")
            else:
                results.append("❌ EnhancedContentRecommender training failed")

        # Train WeightedContentRecommender  
        if os.path.exists(weighted_model_path):
            logger.info("Loading existing WeightedContentRecommender...")
            weighted_model = WeightedContentRecommender.load(weighted_model_path)
            results.append("✅ WeightedContentRecommender loaded")
        else:
            logger.info("Training new WeightedContentRecommender...")
            weighted_model = WeightedContentRecommender()
            success = weighted_model.train(tracks_df)
            if success:
                weighted_model.save(weighted_model_path)
                results.append("✅ WeightedContentRecommender trained and saved")
            else:
                results.append("❌ WeightedContentRecommender training failed")

        # ✅ Feature quality analysis
        feature_analysis = []
        cultural_features = [col for col in tracks_df.columns if col.startswith('is_') or col == 'music_culture']
        genre_features = [col for col in tracks_df.columns if col.startswith('genre_')]
        
        feature_analysis.append(f"📊 **Cultural features:** {len(cultural_features)}")
        feature_analysis.append(f"🎵 **Genre features:** {len(genre_features)}")
        
        if 'cultural_confidence' in tracks_df.columns:
            avg_confidence = tracks_df['cultural_confidence'].mean()
            feature_analysis.append(f"🧠 **Cultural confidence:** {avg_confidence:.3f}")

        return f"""**🤖 Model Training Results:**

{chr(10).join(results)}

**📈 Feature Quality:**
{chr(10).join(feature_analysis)}

**🚀 Ready for recommendations!**"""

    except Exception as e:
        logger.error(f"Error training models: {e}\n{traceback.format_exc()}")
        return f"❌ Training error: {str(e)}"

def recommend_similar(song_name, artist_name="", n=10):
    """Generate recommendations with both models"""
    global model, weighted_model
    
    if model is None or not model.is_trained:
        return "⚠️ EnhancedContentRecommender not trained. Please train models first."
    if weighted_model is None or not weighted_model.is_trained:
        return "⚠️ WeightedContentRecommender not trained. Please train models first."
    
    try:
        # ✅ Load data for context
        processed_path = os.path.join(PROCESSED_DATA_DIR, 'processed_tracks.csv')
        if os.path.exists(processed_path):
            tracks_df = pd.read_csv(processed_path)
            
            # Find seed track for context
            mask = tracks_df['name'].str.lower().str.strip() == song_name.lower().strip()
            if artist_name:
                mask = mask & (tracks_df['artist'].str.lower().str.strip() == artist_name.lower().strip())
            
            found_tracks = tracks_df[mask]
            
            if found_tracks.empty:
                # Try partial matching
                mask = tracks_df['name'].str.lower().str.contains(song_name.lower(), na=False)
                if artist_name:
                    mask = mask & tracks_df['artist'].str.lower().str.contains(artist_name.lower(), na=False)
                found_tracks = tracks_df[mask]
            
            # ✅ Show seed track info with cultural context
            if not found_tracks.empty:
                seed_track = found_tracks.iloc[0]
                seed_culture = seed_track.get('music_culture', 'other')
                seed_confidence = seed_track.get('cultural_confidence', 0)
                
                seed_info = f"""**🎵 Seed Track:** {seed_track['name']} - {seed_track['artist']}
**🌍 Culture:** {seed_culture} | **📊 Popularity:** {seed_track.get('popularity', 'N/A')}
**📅 Year:** {seed_track.get('release_year', 'N/A')} | **🧠 Cultural Confidence:** {seed_confidence:.3f}

---
"""
            else:
                seed_info = f"**⚠️ Track '{song_name}' not found in database. Showing similar recommendations...**\n\n---\n"
        else:
            seed_info = "**🎵 Generating recommendations...**\n\n---\n"

        # ✅ Generate recommendations with error handling
        logger.info(f"Generating recommendations for '{song_name}' by '{artist_name}'")
        
        results = []
        
        # EnhancedContentRecommender
        try:
            enhanced_recs = model.recommend(track_name=song_name, artist=artist_name, n_recommendations=n)
            if not enhanced_recs.empty:
                results.append("## 🔍 EnhancedContentRecommender:")
                
                display_cols = ['name', 'artist', 'enhanced_score']
                if 'music_culture' in enhanced_recs.columns:
                    display_cols.insert(-1, 'music_culture')
                if 'popularity' in enhanced_recs.columns:
                    display_cols.insert(-1, 'popularity')
                
                available_cols = [col for col in display_cols if col in enhanced_recs.columns]
                results.append(enhanced_recs[available_cols].round(3).to_markdown(index=False))
                
                # ✅ Cultural analytics
                if 'music_culture' in enhanced_recs.columns:
                    culture_dist = enhanced_recs['music_culture'].value_counts()
                    results.append(f"**🌍 Cultural diversity:** {dict(culture_dist)}")
                
                avg_score = enhanced_recs['enhanced_score'].mean() if 'enhanced_score' in enhanced_recs.columns else 0
                results.append(f"**📈 Avg enhanced score:** {avg_score:.3f}")
            else:
                results.append("## 🔍 EnhancedContentRecommender:\n❌ No recommendations generated")
        except Exception as e:
            logger.error(f"EnhancedContentRecommender failed: {e}")
            results.append("## 🔍 EnhancedContentRecommender:\n❌ Model error")

        results.append("\n---\n")

        # WeightedContentRecommender
        try:
            weighted_recs = weighted_model.recommend(track_name=song_name, artist=artist_name, n_recommendations=n)
            if not weighted_recs.empty:
                results.append("## 🎯 WeightedContentRecommender (ISRC Cultural + Genre Weights):")
                
                display_cols = ['name', 'artist', 'final_score']
                if 'music_culture' in weighted_recs.columns:
                    display_cols.insert(-1, 'music_culture')
                if 'popularity' in weighted_recs.columns:
                    display_cols.insert(-1, 'popularity')
                
                available_cols = [col for col in display_cols if col in weighted_recs.columns]
                results.append(weighted_recs[available_cols].round(3).to_markdown(index=False))
                
                # ✅ Cultural analytics
                if 'music_culture' in weighted_recs.columns:
                    culture_dist = weighted_recs['music_culture'].value_counts()
                    results.append(f"**🌍 Cultural diversity:** {dict(culture_dist)}")
                
                avg_score = weighted_recs['final_score'].mean() if 'final_score' in weighted_recs.columns else 0
                results.append(f"**📈 Avg weighted score:** {avg_score:.3f}")
            else:
                results.append("## 🎯 WeightedContentRecommender:\n❌ No recommendations generated")
        except Exception as e:
            logger.error(f"WeightedContentRecommender failed: {e}")
            results.append("## 🎯 WeightedContentRecommender:\n❌ Model error")

        return seed_info + "\n".join(results)
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}\n{traceback.format_exc()}")
        return f"❌ System error: {str(e)}"

def check_data_status():
    """Check data completeness and quality"""
    try:
        # ✅ Simple data status check
        raw_tracks = os.path.join(RAW_DATA_DIR, 'tracks.csv')
        processed_tracks = os.path.join(PROCESSED_DATA_DIR, 'processed_tracks.csv')
        
        status_lines = []
        score = 0
        max_score = 5
        
        # Check raw data
        if os.path.exists(raw_tracks):
            raw_df = pd.read_csv(raw_tracks)
            status_lines.append(f"**Raw data:** {len(raw_df):,} tracks")
            score += 1
        else:
            status_lines.append("**Raw data:** Not found")
        
        # Check processed data
        if os.path.exists(processed_tracks):
            processed_df = pd.read_csv(processed_tracks)
            status_lines.append(f"✅ **Processed data:** {len(processed_df):,} tracks")
            score += 1
            
            # Check cultural features
            cultural_features = [col for col in processed_df.columns if col.startswith('is_') or col == 'music_culture']
            if len(cultural_features) >= 3:
                status_lines.append(f"✅ **Cultural intelligence:** {len(cultural_features)} features")
                score += 1
            else:
                status_lines.append("⚠️ **Cultural intelligence:** Limited features")
            
            # Check ISRC coverage
            if 'isrc' in processed_df.columns:
                isrc_coverage = (processed_df['isrc'] != '').sum() / len(processed_df)
                if isrc_coverage > 0.5:
                    status_lines.append(f"✅ **ISRC coverage:** {isrc_coverage*100:.1f}%")
                    score += 1
                else:
                    status_lines.append(f"⚠️ **ISRC coverage:** {isrc_coverage*100:.1f}% (low)")
            
            # Check genre features
            genre_features = [col for col in processed_df.columns if col.startswith('genre_')]
            if len(genre_features) >= 20:  # Adjusted threshold from 3 to 20
                status_lines.append(f"✅ **Genre features:** {len(genre_features)} types")
                score += 1
                
                # Add top genres information
                if len(genre_features) > 0:
                    # Get top 5 genres by count
                    genre_counts = {}
                    for genre in genre_features:
                        genre_counts[genre[6:]] = processed_df[genre].sum()  # Remove 'genre_' prefix
                    
                    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    top_genres_str = ", ".join([f"{genre} ({count})" for genre, count in top_genres])
                    status_lines.append(f"📊 **Top genres:** {top_genres_str}")
            else:
                status_lines.append(f"⚠️ **Genre features:** Limited ({len(genre_features)} types)")
        else:
            status_lines.append("❌ **Processed data:** Not found")
        
        # Check models
        enhanced_model_path = os.path.join(MODELS_DIR, 'enhanced_content_recommender.pkl')
        weighted_model_path = os.path.join(MODELS_DIR, 'weighted_content_recommender.pkl')
        
        if os.path.exists(enhanced_model_path) and os.path.exists(weighted_model_path):
            status_lines.append("✅ **Models:** Both models available")
        elif os.path.exists(enhanced_model_path) or os.path.exists(weighted_model_path):
            status_lines.append("⚠️ **Models:** Partial availability")
        else:
            status_lines.append("❌ **Models:** Not trained")

        # Overall status
        if score >= 4:
            overall_status = "🚀 **EXCELLENT** - Production ready!"
        elif score >= 3:
            overall_status = "✅ **GOOD** - Ready for recommendations"
        elif score >= 2:
            overall_status = "⚠️ **FAIR** - Basic functionality"
        else:
            overall_status = "❌ **POOR** - Need more data"

        return f"""# 📊 Data Status Report

{overall_status}

**Readiness Score:** {score}/{max_score} ({score/max_score*100:.0f}%)

## Detailed Status:
{chr(10).join(status_lines)}

## Next Steps:
- Score < 3: Run data setup
- Score < 4: Train models
- Score ≥ 4: Ready for recommendations!
"""
        
    except Exception as e:
        logger.error(f"Error checking data status: {e}")
        return f"❌ **Error checking data status:** {str(e)}"

def create_ui():
    """Create Gradio interface"""
    with gr.Blocks(title="Music Recommender - ISRC Cultural Intelligence", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎵 Music Recommender - ISRC Cultural Intelligence")
        gr.Markdown("*Advanced recommendation system with ISRC-based cultural intelligence and Spotify metadata*")
        
        with gr.Tab("🔧 Data Setup"):
            gr.Markdown("### Setup Dataset from Spotify API")
            gr.Markdown("*Collect real metadata with ISRC cultural intelligence*")
            
            with gr.Row():
                with gr.Column():
                    check_data_btn = gr.Button("🔍 Check Current Data", variant="secondary")
                    data_status_output = gr.Markdown()
                    check_data_btn.click(fn=check_data_status, outputs=data_status_output)
                
                with gr.Column():
                    tracks_per_query = gr.Slider(
                        MIN_TRACKS_PER_QUERY, 
                        MAX_TRACKS_PER_QUERY, 
                        value=DEFAULT_TRACKS_PER_QUERY, 
                        step=TRACKS_QUERY_STEP, 
                        label="Tracks per Query"
                    )
                    setup_btn = gr.Button("🚀 Setup Dataset", variant="primary")
                    setup_output = gr.Markdown()
                    setup_btn.click(fn=setup_initial_dataset, inputs=[tracks_per_query], outputs=setup_output)

        with gr.Tab("🤖 Model Training"):
            gr.Markdown("### Train Recommendation Models")
            gr.Markdown("*Train with ISRC cultural intelligence and real Spotify features*")
            train_btn = gr.Button("🏋️ Train Models", variant="primary")
            train_output = gr.Markdown()
            train_btn.click(fn=train_model, outputs=train_output)

        with gr.Tab("🎯 Recommendations"):
            gr.Markdown("### Smart Music Recommendations")
            gr.Markdown("*Powered by ISRC cultural intelligence and weighted content features*")
            
            with gr.Row():
                song_input = gr.Textbox(label="🎵 Song Name", placeholder="e.g., Dynamite")
                artist_input = gr.Textbox(label="👤 Artist Name (optional)", placeholder="e.g., BTS")
            
            n_similar = gr.Slider(5, 20, value=10, step=1, label="📊 Number of Recommendations")
            rec_btn = gr.Button("🔍 Get Recommendations", variant="primary")
            rec_output = gr.Markdown()
            rec_btn.click(fn=recommend_similar, inputs=[song_input, artist_input, n_similar], outputs=rec_output)
    
    return app

if __name__ == "__main__":
    # Initialize models if available
    initialize_model()
    
    # Create and launch UI
    demo = create_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
