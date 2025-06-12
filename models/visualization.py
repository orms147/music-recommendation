import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import logging
from matplotlib.ticker import PercentFormatter

logger = logging.getLogger(__name__)

def compare_recommendation_models(enhanced_model, weighted_model, track_name, artist=None, n_recommendations=10):
    """
    Create comprehensive visualization comparing EnhancedContentRecommender and WeightedContentRecommender
    
    Args:
        enhanced_model: EnhancedContentRecommender instance
        weighted_model: WeightedContentRecommender instance
        track_name: Name of the seed track
        artist: Artist name (optional)
        n_recommendations: Number of recommendations to compare
        
    Returns:
        matplotlib figure object
    """
    if not enhanced_model.is_trained or not weighted_model.is_trained:
        logger.error("Models not trained. Please train both models first.")
        return None
    
    # Get recommendations from both models with timing
    start_time = time.perf_counter() # Sử dụng perf_counter
    enhanced_recs = enhanced_model.recommend(track_name, artist, n_recommendations)
    enhanced_time = time.perf_counter() - start_time # Sử dụng perf_counter
    
    start_time = time.perf_counter() # Sử dụng perf_counter
    weighted_recs = weighted_model.recommend(track_name, artist, n_recommendations)
    weighted_time = time.perf_counter() - start_time # Sử dụng perf_counter
    
    if enhanced_recs.empty or weighted_recs.empty:
        logger.error(f"Could not get recommendations for '{track_name}'")
        return None
    
    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f"Model Comparison: '{track_name}' by {artist or 'Unknown'}", fontsize=16)
    
    # 1. Recommendation Accuracy (top left)
    plot_recommendation_overlap(axes[0, 0], enhanced_recs, weighted_recs)
    
    # 2. Cultural Diversity (top right)
    plot_cultural_diversity(axes[0, 1], enhanced_recs, weighted_recs)
    
    # 3. Search Performance (middle left)
    plot_search_performance(axes[1, 0], enhanced_model, weighted_model, track_name, artist, enhanced_time, weighted_time)
    
    # 4. Popularity Distribution (middle right)
    plot_popularity_distribution(axes[1, 1], enhanced_recs, weighted_recs)
    
    # 5. Popularity vs Relevance Balance (bottom left)
    plot_popularity_relevance_balance(axes[2, 0], enhanced_recs, weighted_recs)
    
    # 6. Summary Statistics (bottom right)
    plot_summary_statistics(axes[2, 1], enhanced_recs, weighted_recs, enhanced_time, weighted_time)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    return fig

def plot_recommendation_overlap(ax, enhanced_recs, weighted_recs):
    """Plot overlap between recommendations from both models"""
    # Get track names from both models
    enhanced_tracks = set(enhanced_recs['name'].str.lower())
    weighted_tracks = set(weighted_recs['name'].str.lower())
    
    # Calculate overlap
    common_tracks = enhanced_tracks.intersection(weighted_tracks)
    only_enhanced = enhanced_tracks - weighted_tracks
    only_weighted = weighted_tracks - enhanced_tracks
    
    # Calculate percentages
    total = len(enhanced_tracks.union(weighted_tracks))
    common_pct = len(common_tracks) / total * 100
    enhanced_pct = len(only_enhanced) / total * 100
    weighted_pct = len(only_weighted) / total * 100
    
    # Create pie chart
    labels = ['Common', 'Only Enhanced', 'Only Weighted']
    sizes = [common_pct, enhanced_pct, weighted_pct]
    colors = ['#66b3ff', '#ff9999', '#99ff99']
    explode = (0.1, 0, 0)  # explode the 1st slice (Common)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title('Recommendation Overlap')
    
    # Add text with actual counts
    ax.text(0, -1.2, f"Common: {len(common_tracks)}, Only Enhanced: {len(only_enhanced)}, Only Weighted: {len(only_weighted)}")

def plot_cultural_diversity(ax, enhanced_recs, weighted_recs):
    """Plot cultural diversity in recommendations"""
    if 'music_culture' not in enhanced_recs.columns or 'music_culture' not in weighted_recs.columns:
        ax.text(0.5, 0.5, 'Cultural data not available', ha='center', va='center')
        return
    
    # Get culture counts
    enhanced_cultures = enhanced_recs['music_culture'].value_counts().to_dict()
    weighted_cultures = weighted_recs['music_culture'].value_counts().to_dict()
    
    # Get all unique cultures
    all_cultures = sorted(set(enhanced_cultures.keys()) | set(weighted_cultures.keys()))
    
    # Create data for plotting
    enhanced_data = [enhanced_cultures.get(culture, 0) for culture in all_cultures]
    weighted_data = [weighted_cultures.get(culture, 0) for culture in all_cultures]
    
    # Create bar chart
    x = np.arange(len(all_cultures))
    width = 0.35
    
    ax.bar(x - width/2, enhanced_data, width, label='Enhanced Model')
    ax.bar(x + width/2, weighted_data, width, label='Weighted Model')
    
    ax.set_title('Cultural Diversity')
    ax.set_xlabel('Music Culture')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(all_cultures, rotation=45, ha='right')
    ax.legend()

def plot_search_performance(ax, enhanced_model, weighted_model, track_name, artist, enhanced_time, weighted_time):
    """Plot search performance metrics"""
    # Create bar chart for processing time
    labels = ['Enhanced Model', 'Weighted Model']
    times = [enhanced_time, weighted_time]
    
    ax.bar(labels, times, color=['#ff9999', '#99ff99'])
    ax.set_title('Search Performance')
    ax.set_ylabel('Processing Time (seconds)')
    
    # Add confidence score if available
    enhanced_conf_text = "N/A"
    weighted_conf_text = "N/A"

    try:
        if hasattr(enhanced_model, '_find_track_with_fuzzy'):
            # Assuming _find_track_with_fuzzy returns (index, confidence_score)
            # or (None, None) if not found or error
            _, enhanced_conf_val = enhanced_model._find_track_with_fuzzy(track_name, artist)
            if enhanced_conf_val is not None and isinstance(enhanced_conf_val, (int, float)):
                enhanced_conf_text = f"{enhanced_conf_val:.2f}"
            elif enhanced_conf_val is not None:
                logger.warning(f"Enhanced model _find_track_with_fuzzy returned non-numeric confidence: {enhanced_conf_val}")
        else:
            logger.info("Enhanced model does not have '_find_track_with_fuzzy' method for confidence score.")
            
        if hasattr(weighted_model, '_find_track_with_fuzzy'):
            _, weighted_conf_val = weighted_model._find_track_with_fuzzy(track_name, artist)
            if weighted_conf_val is not None and isinstance(weighted_conf_val, (int, float)):
                weighted_conf_text = f"{weighted_conf_val:.2f}"
            elif weighted_conf_val is not None:
                logger.warning(f"Weighted model _find_track_with_fuzzy returned non-numeric confidence: {weighted_conf_val}")
        else:
            logger.info("Weighted model does not have '_find_track_with_fuzzy' method for confidence score.")
            
        ax.text(0, times[0] * 1.05 if times[0] > 0 else 0.01, f"Confidence: {enhanced_conf_text}", ha='center', va='bottom')
        ax.text(1, times[1] * 1.05 if times[1] > 0 else 0.01, f"Confidence: {weighted_conf_text}", ha='center', va='bottom')

    except Exception as e:
        logger.error(f"Error getting search confidence for visualization: {e}", exc_info=True)
        # Display error on the chart itself if something goes wrong
        ax.text(0, times[0] * 1.05 if times[0] > 0 else 0.01, "Confidence: Error", ha='center', va='bottom')
        ax.text(1, times[1] * 1.05 if times[1] > 0 else 0.01, "Confidence: Error", ha='center', va='bottom')
    
    # Add time values
    for i, v_time in enumerate(times): # Renamed v to v_time to avoid conflict if v is used above
        ax.text(i, v_time / 2, f"{v_time:.3f}s", ha='center', va='center', color='black', fontweight='bold')

def plot_popularity_distribution(ax, enhanced_recs, weighted_recs):
    """Plot popularity distribution of recommended tracks"""
    if 'popularity' not in enhanced_recs.columns or 'popularity' not in weighted_recs.columns:
        ax.text(0.5, 0.5, 'Popularity data not available', ha='center', va='center')
        return
    
    # Create histogram
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    
    enhanced_hist = np.histogram(enhanced_recs['popularity'], bins=bins)[0]
    weighted_hist = np.histogram(weighted_recs['popularity'], bins=bins)[0]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, enhanced_hist, width, label='Enhanced Model')
    ax.bar(x + width/2, weighted_hist, width, label='Weighted Model')
    
    ax.set_title('Popularity Distribution')
    ax.set_xlabel('Popularity Range')
    ax.set_ylabel('Number of Tracks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add average popularity
    enhanced_avg = enhanced_recs['popularity'].mean()
    weighted_avg = weighted_recs['popularity'].mean()
    ax.text(0.05, 0.95, f"Enhanced Avg: {enhanced_avg:.1f}", transform=ax.transAxes, va='top')
    ax.text(0.05, 0.90, f"Weighted Avg: {weighted_avg:.1f}", transform=ax.transAxes, va='top')

def plot_popularity_relevance_balance(ax, enhanced_recs, weighted_recs):
    """Plot balance between popularity and relevance"""
    if 'popularity' not in enhanced_recs.columns or 'popularity' not in weighted_recs.columns:
        ax.text(0.5, 0.5, 'Popularity data not available', ha='center', va='center')
        return
    
    # Get scores and popularity
    enhanced_scores = enhanced_recs['enhanced_score'].values
    weighted_scores = weighted_recs['final_score'].values if 'final_score' in weighted_recs.columns else weighted_recs.index.map(lambda x: 1 - x/len(weighted_recs))
    
    enhanced_pop = enhanced_recs['popularity'].values
    weighted_pop = weighted_recs['popularity'].values
    
    # Create scatter plot
    ax.scatter(enhanced_scores, enhanced_pop, alpha=0.7, label='Enhanced Model')
    ax.scatter(weighted_scores, weighted_pop, alpha=0.7, label='Weighted Model')
    
    ax.set_title('Popularity vs Relevance Balance')
    ax.set_xlabel('Recommendation Score')
    ax.set_ylabel('Track Popularity')
    ax.legend()
    
    # Add trend lines
    if len(enhanced_scores) > 1:
        z1 = np.polyfit(enhanced_scores, enhanced_pop, 1)
        p1 = np.poly1d(z1)
        ax.plot(enhanced_scores, p1(enhanced_scores), "r--", alpha=0.5)
    
    if len(weighted_scores) > 1:
        z2 = np.polyfit(weighted_scores, weighted_pop, 1)
        p2 = np.poly1d(z2)
        ax.plot(weighted_scores, p2(weighted_scores), "g--", alpha=0.5)

def plot_summary_statistics(ax, enhanced_recs, weighted_recs, enhanced_time, weighted_time):
    """Plot summary statistics for both models"""
    # Create a table with summary statistics
    data = []
    columns = ['Metric', 'Enhanced Model', 'Weighted Model']
    
    # Add basic stats
    data.append(['Number of recommendations', len(enhanced_recs), len(weighted_recs)])
    data.append(['Processing time (s)', f"{enhanced_time:.3f}", f"{weighted_time:.3f}"])
    
    # Add popularity stats if available
    if 'popularity' in enhanced_recs.columns and 'popularity' in weighted_recs.columns:
        data.append(['Avg. popularity', f"{enhanced_recs['popularity'].mean():.1f}", f"{weighted_recs['popularity'].mean():.1f}"])
        data.append(['Min. popularity', f"{enhanced_recs['popularity'].min():.1f}", f"{weighted_recs['popularity'].min():.1f}"])
        data.append(['Max. popularity', f"{enhanced_recs['popularity'].max():.1f}", f"{weighted_recs['popularity'].max():.1f}"])
    
    # Add unique artists count
    data.append(['Unique artists', enhanced_recs['artist'].nunique(), weighted_recs['artist'].nunique()])
    
    # Add cultural diversity if available
    if 'music_culture' in enhanced_recs.columns and 'music_culture' in weighted_recs.columns:
        data.append(['Cultural diversity', enhanced_recs['music_culture'].nunique(), weighted_recs['music_culture'].nunique()])
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    ax.set_title('Summary Statistics')

def save_comparison_visualization(enhanced_model, weighted_model, track_name, artist=None, 
                                 n_recommendations=10, output_path="model_comparison.png"):
    """Generate and save model comparison visualization
    
    Args:
        enhanced_model: EnhancedContentRecommender instance
        weighted_model: WeightedContentRecommender instance
        track_name: Name of the seed track
        artist: Artist name (optional)
        n_recommendations: Number of recommendations to compare
        output_path: Path to save the visualization
        
    Returns:
        Path to saved image or None if failed
    """
    fig = None # Initialize fig to None
    try:
        fig = compare_recommendation_models(
            enhanced_model, weighted_model, track_name, artist, n_recommendations
        )
        
        if fig is None:
            logger.warning(f"Figure generation failed for '{track_name}' by '{artist or 'Unknown'}' in compare_recommendation_models.")
            return None
            
        # Ensure the output directory exists
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not an empty string
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory: {output_dir}")

        fig.savefig(output_path, dpi=150, bbox_inches='tight') # Increased DPI for potentially better quality
        logger.info(f"Saved model comparison visualization to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error during visualization generation or saving for '{track_name}': {e}", exc_info=True)
        return None
    finally:
        if fig is not None:
            try:
                plt.close(fig) # Close the figure to free memory
                logger.debug(f"Closed figure for '{track_name}'")
            except Exception as e_close:
                logger.error(f"Error closing figure for '{track_name}': {e_close}", exc_info=True)