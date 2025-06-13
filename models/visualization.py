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
    
    # Measure recommendation time
    start_time = time.time()
    enhanced_recs = enhanced_model.recommend(track_name, artist, n_recommendations)
    enhanced_time = time.time() - start_time
    
    start_time = time.time()
    weighted_recs = weighted_model.recommend(track_name, artist, n_recommendations)
    weighted_time = time.time() - start_time
    
    # Check if recommendations were generated
    if enhanced_recs.empty and weighted_recs.empty:
        logger.error(f"Both models failed to generate recommendations for '{track_name}'")
        return None
    
    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f"Model Comparison: '{track_name}' by {artist or 'Unknown'}", fontsize=16)
    
    # 1. Recommendation Overlap (top left)
    plot_recommendation_overlap(axes[0, 0], enhanced_recs, weighted_recs)
    
    # 2. Cultural Diversity (top right)
    plot_cultural_diversity(axes[0, 1], enhanced_recs, weighted_recs)
    
    # 3. Search Performance (middle left)
    plot_search_performance(axes[1, 0], enhanced_model, weighted_model, track_name, artist, enhanced_time, weighted_time)
    
    # 4. Popularity Distribution (middle right)
    plot_popularity_distribution(axes[1, 1], enhanced_recs, weighted_recs)
    
    # 5. Cluster Analysis (bottom left) - NEW
    plot_cluster_analysis(axes[2, 0], enhanced_recs, weighted_recs, enhanced_model, weighted_model, track_name, artist)
    
    # 6. Cultural Similarity Heatmap (bottom right) - NEW
    plot_cultural_similarity_heatmap(axes[2, 1], enhanced_recs, weighted_recs)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
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
            # Gọi phương thức và kiểm tra kết quả
            result = enhanced_model._find_track_with_fuzzy(track_name, artist)
            if result is not None and isinstance(result, tuple) and len(result) == 2:
                _, enhanced_conf_val = result
                if enhanced_conf_val is not None and isinstance(enhanced_conf_val, (int, float)):
                    enhanced_conf_text = f"{enhanced_conf_val:.2f}"
            else:
                logger.warning(f"Enhanced model _find_track_with_fuzzy returned invalid result: {result}")
        else:
            logger.info("Enhanced model does not have '_find_track_with_fuzzy' method for confidence score.")
            
        if hasattr(weighted_model, '_find_track_with_fuzzy'):
            # Gọi phương thức và kiểm tra kết quả
            result = weighted_model._find_track_with_fuzzy(track_name, artist)
            if result is not None and isinstance(result, tuple) and len(result) == 2:
                _, weighted_conf_val = result
                if weighted_conf_val is not None and isinstance(weighted_conf_val, (int, float)):
                    weighted_conf_text = f"{weighted_conf_val:.2f}"
            else:
                logger.warning(f"Weighted model _find_track_with_fuzzy returned invalid result: {result}")
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

def plot_cluster_analysis(ax, enhanced_recs, weighted_recs, enhanced_model, weighted_model, track_name, artist=None):
    """Plot cluster analysis for both recommendation models"""
    ax.set_title("Cluster Distribution Analysis", fontsize=12)
    
    # Find seed track
    seed_idx_enhanced = None
    seed_idx_weighted = None
    
    # Kiểm tra phương thức find_track_idx
    if hasattr(enhanced_model, 'find_track_idx'):
        seed_idx_enhanced = enhanced_model.find_track_idx(track_name, artist)
    else:
        logger.warning("Enhanced model does not have find_track_idx method")
    
    if hasattr(weighted_model, 'find_track_idx'):
        seed_idx_weighted = weighted_model.find_track_idx(track_name, artist)
    else:
        logger.warning("Weighted model does not have find_track_idx method")
    
    # Check if we have cluster information
    has_kmeans = False
    has_hdbscan = False
    
    if seed_idx_enhanced is not None and 'kmeans_cluster' in enhanced_model.tracks_df.columns:
        seed_kmeans = enhanced_model.tracks_df.iloc[seed_idx_enhanced]['kmeans_cluster']
        has_kmeans = True
    elif seed_idx_weighted is not None and 'kmeans_cluster' in weighted_model.tracks_df.columns:
        seed_kmeans = weighted_model.tracks_df.iloc[seed_idx_weighted]['kmeans_cluster']
        has_kmeans = True
    else:
        seed_kmeans = -1
    
    if seed_idx_enhanced is not None and 'hdbscan_cluster' in enhanced_model.tracks_df.columns:
        seed_hdbscan = enhanced_model.tracks_df.iloc[seed_idx_enhanced]['hdbscan_cluster']
        has_hdbscan = True
    elif seed_idx_weighted is not None and 'hdbscan_cluster' in weighted_model.tracks_df.columns:
        seed_hdbscan = weighted_model.tracks_df.iloc[seed_idx_weighted]['hdbscan_cluster']
        has_hdbscan = True
    else:
        seed_hdbscan = -1
    
    # If no cluster information, show message
    if not has_kmeans and not has_hdbscan:
        ax.text(0.5, 0.5, "No clustering information available", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    # Prepare data for plotting
    data = []
    labels = []
    
    # K-Means analysis
    if has_kmeans:
        if 'kmeans_cluster' in enhanced_recs.columns:
            same_kmeans_enhanced = (enhanced_recs['kmeans_cluster'] == seed_kmeans).sum()
            data.append(same_kmeans_enhanced)
            labels.append('Enhanced\nSame K-Means')
            
            diff_kmeans_enhanced = len(enhanced_recs) - same_kmeans_enhanced
            data.append(diff_kmeans_enhanced)
            labels.append('Enhanced\nDiff K-Means')
        
        if 'kmeans_cluster' in weighted_recs.columns:
            same_kmeans_weighted = (weighted_recs['kmeans_cluster'] == seed_kmeans).sum()
            data.append(same_kmeans_weighted)
            labels.append('Weighted\nSame K-Means')
            
            diff_kmeans_weighted = len(weighted_recs) - same_kmeans_weighted
            data.append(diff_kmeans_weighted)
            labels.append('Weighted\nDiff K-Means')
    
    # HDBSCAN analysis
    if has_hdbscan:
        if 'hdbscan_cluster' in enhanced_recs.columns:
            same_hdbscan_enhanced = (enhanced_recs['hdbscan_cluster'] == seed_hdbscan).sum()
            data.append(same_hdbscan_enhanced)
            labels.append('Enhanced\nSame HDBSCAN')
            
            diff_hdbscan_enhanced = len(enhanced_recs) - same_hdbscan_enhanced
            data.append(diff_hdbscan_enhanced)
            labels.append('Enhanced\nDiff HDBSCAN')
        
        if 'hdbscan_cluster' in weighted_recs.columns:
            same_hdbscan_weighted = (weighted_recs['hdbscan_cluster'] == seed_hdbscan).sum()
            data.append(same_hdbscan_weighted)
            labels.append('Weighted\nSame HDBSCAN')
            
            diff_hdbscan_weighted = len(weighted_recs) - same_hdbscan_weighted
            data.append(diff_hdbscan_weighted)
            labels.append('Weighted\nDiff HDBSCAN')
    
    # Plot the data
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    ax.bar(labels, data, color=colors)
    ax.set_ylabel('Number of Tracks')
    ax.set_ylim(0, max(data) * 1.2)
    
    # Add seed cluster info
    if has_kmeans:
        ax.text(0.02, 0.98, f"Seed K-Means Cluster: {seed_kmeans}", 
                transform=ax.transAxes, va='top', fontsize=10)
    if has_hdbscan:
        ax.text(0.02, 0.93, f"Seed HDBSCAN Cluster: {seed_hdbscan}", 
                transform=ax.transAxes, va='top', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def plot_cultural_similarity_heatmap(ax, enhanced_recs, weighted_recs):
    """Plot cultural similarity heatmap between recommendations"""
    ax.set_title("Cultural Similarity Heatmap", fontsize=12)
    
    # Check if we have cultural information
    if 'music_culture' not in enhanced_recs.columns or 'music_culture' not in weighted_recs.columns:
        ax.text(0.5, 0.5, "No cultural information available", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    # Get unique cultures from both recommendation sets
    enhanced_cultures = enhanced_recs['music_culture'].unique()
    weighted_cultures = weighted_recs['music_culture'].unique()
    all_cultures = np.unique(np.concatenate([enhanced_cultures, weighted_cultures]))
    
    # Create matrix for heatmap
    matrix_size = len(all_cultures)
    similarity_matrix = np.zeros((matrix_size, matrix_size))
    
    # Count occurrences of each culture in each recommendation set
    enhanced_counts = enhanced_recs['music_culture'].value_counts()
    weighted_counts = weighted_recs['music_culture'].value_counts()
    
    # Fill the matrix
    for i, culture1 in enumerate(all_cultures):
        for j, culture2 in enumerate(all_cultures):
            # Diagonal: percentage of this culture in both sets
            if i == j:
                enhanced_pct = enhanced_counts.get(culture1, 0) / len(enhanced_recs) if len(enhanced_recs) > 0 else 0
                weighted_pct = weighted_counts.get(culture1, 0) / len(weighted_recs) if len(weighted_recs) > 0 else 0
                similarity_matrix[i, j] = (enhanced_pct + weighted_pct) / 2 * 100
            # Off-diagonal: co-occurrence in recommendations
            else:
                enhanced_co = ((enhanced_recs['music_culture'] == culture1) & 
                              (enhanced_recs['music_culture'].shift(1) == culture2)).sum()
                weighted_co = ((weighted_recs['music_culture'] == culture1) & 
                              (weighted_recs['music_culture'].shift(1) == culture2)).sum()
                similarity_matrix[i, j] = enhanced_co + weighted_co
    
    # Plot heatmap
    im = ax.imshow(similarity_matrix, cmap='YlOrRd')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Cultural Co-occurrence / Percentage', rotation=-90, va="bottom")
    
    # Add labels
    ax.set_xticks(np.arange(matrix_size))
    ax.set_yticks(np.arange(matrix_size))
    ax.set_xticklabels(all_cultures)
    ax.set_yticklabels(all_cultures)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i == j:
                text = ax.text(j, i, f"{similarity_matrix[i, j]:.1f}%",
                              ha="center", va="center", color="black" if similarity_matrix[i, j] < 50 else "white")
            else:
                if similarity_matrix[i, j] > 0:
                    text = ax.text(j, i, f"{int(similarity_matrix[i, j])}",
                                  ha="center", va="center", color="black" if similarity_matrix[i, j] < 5 else "white")
    
    ax.set_title("Cultural Distribution & Co-occurrence")

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
