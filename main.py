import argparse
import pandas as pd
import os
import logging
from config.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.data_fetcher import SpotifyDataFetcher, fetch_initial_dataset
from utils.data_processor import DataProcessor
from models.content_model import ContentBasedRecommender
from models.collaborative_model import CollaborativeFilteringRecommender
from models.hybrid_model import HybridRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_spotify_credentials():
    """Setup Spotify API credentials"""
    global SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
    
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        SPOTIFY_CLIENT_ID = input("Enter your Spotify Client ID: ")
        SPOTIFY_CLIENT_SECRET = input("Enter your Spotify Client Secret: ")
        
        # Update config file
        config_path = os.path.join("config", "config.py")
        with open(config_path, 'r') as f:
            config_data = f.read()
            
        config_data = config_data.replace('SPOTIFY_CLIENT_ID = ""', f'SPOTIFY_CLIENT_ID = "{SPOTIFY_CLIENT_ID}"')
        config_data = config_data.replace('SPOTIFY_CLIENT_SECRET = ""', f'SPOTIFY_CLIENT_SECRET = "{SPOTIFY_CLIENT_SECRET}"')
        
        with open(config_path, 'w') as f:
            f.write(config_data)
            
        logger.info("Spotify credentials updated in config file")
    
    return SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

def fetch_data(query=None):
    """Fetch music data from Spotify API"""
    # Sửa để sử dụng queries thay vì genres
    if query:
        queries = [q.strip() for q in query.split(',')]
        logger.info(f"Fetching data for queries: {queries}")
        
        fetcher = SpotifyDataFetcher()
        tracks_df = fetcher.fetch_tracks_by_search(
            queries=queries,
            tracks_per_query=50,
            save_path=os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
        )
    else:
        logger.info("Fetching initial dataset")
        from utils.data_fetcher import fetch_initial_dataset
        tracks_df = fetch_initial_dataset()
    
    return tracks_df

def process_data():
    """Process the raw data into features for recommendation models"""
    if not os.path.exists(os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')):
        logger.error("Raw data not found. Please fetch data first.")
        return False
    
    logger.info("Processing raw data...")
    processor = DataProcessor()
    processor.load_data()
    processor.process_all()
    return True

def train_models():
    """Train all recommendation models"""
    if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'track_features.csv')):
        logger.error("Processed data not found. Please process data first.")
        return False
    
    logger.info("Training recommendation models...")
    
    # Load processed data
    tracks_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'track_features.csv'))
    
    # Try to load user-item matrix if it exists
    user_item_matrix = None
    matrix_path = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')
    if os.path.exists(matrix_path):
        user_item_matrix = pd.read_csv(matrix_path, index_col=0)
    
    # Train hybrid model (which trains component models)
    hybrid_model = HybridRecommender()
    hybrid_model.train(tracks_df, user_item_matrix)
    
    # Save trained model
    hybrid_model.save(os.path.join("models", "hybrid_recommender.pkl"))
    
    logger.info("Models trained and saved successfully")
    return hybrid_model

def recommend_tracks(track_name=None, artist=None, user_id=None, recent_tracks=None, n_recommendations=10):
    """Generate track recommendations based on input"""
    model_path = os.path.join("models", "hybrid_recommender.pkl")
    
    if os.path.exists(model_path):
        logger.info("Loading trained model...")
        model = HybridRecommender.load(model_path)
    else:
        logger.info("No trained model found. Training now...")
        model = train_models()
        if not model:
            logger.error("Failed to train models.")
            return []
    
    input_type = "track" if track_name else "user" if user_id else "sequence" if recent_tracks else "unknown"
    input_value = track_name or user_id or "recent tracks" if recent_tracks else "unknown"
    logger.info(f"Generating recommendations for {input_type}: {input_value}")
    
    recommendations = model.recommend(
        track_name=track_name,
        artist=artist,
        user_id=user_id,
        recent_tracks=recent_tracks,
        n_recommendations=n_recommendations
    )
    
    if recommendations.empty:
        logger.info("No recommendations generated")
        return []
    
    return recommendations

def interactive_mode():
    """Interactive command-line interface for the recommendation system"""
    print("\n===== Music Recommendation System =====\n")
    
    # Check if we have models trained
    model_path = os.path.join("models", "hybrid_recommender.pkl")
    if not os.path.exists(model_path):
        print("No trained models found. Setting up the system...")
        
        # Check if we have data
        if not os.path.exists(os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')):
            fetch_data()
        
        process_data()
        train_models()
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Get song recommendations by song name")
        # Xóa tùy chọn duyệt theo thể loại
        print("2. Fetch more data from Spotify")
        print("3. Retrain recommendation models")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            song_name = input("Enter song name: ")
            artist = input("Enter artist name (optional): ")
            n_recommendations = int(input("Number of recommendations (default 10): ") or "10")
            
            print(f"\nGetting recommendations for '{song_name}'...")
            recommendations = recommend_tracks(song_name, artist, n_recommendations=n_recommendations)
            
            if recommendations is not None and not recommendations.empty:
                print("\nRecommended songs:")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    print(f"{i}. {row['name']} by {row['artist']}")
            else:
                print("No recommendations found")
        
        elif choice == '2':  # Đổi từ 3 thành 2
            # Sửa để sử dụng queries thay vì genres
            queries = input("Enter search queries (comma separated, e.g., 'pop 2023, rock hits'): ")
            num_tracks = int(input("Number of tracks per query (default 50): ") or "50")
            
            print(f"\nFetching tracks from Spotify...")
            queries_list = [q.strip() for q in queries.split(',')]
            
            fetcher = SpotifyDataFetcher()
            tracks_df = fetcher.fetch_tracks_by_search(
                queries=queries_list,
                tracks_per_query=num_tracks,
                save_path=os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
            )
            
            print(f"Fetched {len(tracks_df)} tracks!")
            
            retrain = input("\nWould you like to retrain the models with the new data? (y/n): ")
            if retrain.lower() == 'y':
                process_data()
                train_models()
        
        elif choice == '3':  # Đổi từ 4 thành 3
            print("\nRetraining recommendation models...")
            process_data()
            train_models()
            print("Model training complete!")
        
        elif choice == '4':  # Đổi từ 5 thành 4
            print("Thank you for using the Music Recommendation System!")
            break

def main():
    """Main function for the Music Recommendation Engine"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Music Recommendation Engine")
    
    # Setup flags
    parser.add_argument("--setup", action="store_true", help="Setup the system (fetch data, process, train)")
    parser.add_argument("--fetch", action="store_true", help="Fetch data from Spotify API")
    # Xóa tham số --genres vì không còn sử dụng thể loại
    parser.add_argument("--query", type=str, help="Search query for fetching tracks (e.g., 'pop 2023, rock hits')")
    parser.add_argument("--process", action="store_true", help="Process the raw data")
    parser.add_argument("--train", action="store_true", help="Train the recommendation models")
    parser.add_argument("--recommend", action="store_true", help="Get music recommendations")
    parser.add_argument("--song", type=str, help="Song name for recommendation")
    parser.add_argument("--artist", type=str, help="Artist name for recommendation")
    parser.add_argument("--user", type=str, help="User ID for recommendation")
    parser.add_argument("--n", type=int, default=10, help="Number of recommendations")
    
    args = parser.parse_args()
    
    if args.setup:
        print("Setting up the system...")
        # Fetch initial data if needed
        if not os.path.exists(os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')):
            fetch_data(args.query)
        
        # Process data
        process_data()
        
        # Train models
        train_models()
        
        print("System setup complete!")
    
    elif args.fetch:
        print("Fetching data from Spotify API...")
        # Sửa để sử dụng queries thay vì genres
        queries = args.query.split(',') if args.query else ["pop 2023", "rock hits", "new releases"]
        fetcher = SpotifyDataFetcher()
        tracks_df = fetcher.fetch_tracks_by_search(
            queries=queries,
            tracks_per_query=50,
            save_path=os.path.join(RAW_DATA_DIR, 'spotify_tracks.csv')
        )
        print(f"Fetched {len(tracks_df)} tracks!")
    
    elif args.process:
        print("Processing data...")
        process_data()
        print("Data processing complete!")
    
    elif args.train:
        print("Training recommendation models...")
        train_models()
        print("Model training complete!")
    
    elif args.recommend:
        # Giữ nguyên phần recommend vì không liên quan đến thể loại
        if args.song:
            print(f"Getting recommendations for song: {args.song}")
            
            recommendations = recommend_tracks(
                track_name=args.song,
                artist=args.artist,
                n_recommendations=args.n
            )
            
            if recommendations is not None and not recommendations.empty:
                print("\nRecommended songs:")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    print(f"{i}. {row['name']} by {row['artist']}")
            else:
                print("No recommendations found")
        else:
            print("Please specify a song name with --song")
    
    else:
        interactive_mode()

if __name__ == "__main__":
    main()