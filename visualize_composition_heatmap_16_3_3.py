import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import FixedLocator

def visualize_composition_heatmap(results_df, significant_features, tick_interval=1):
    """
    Visualize contributions (coefficient * average count) as a heatmap.
    Positive values are red, negative values are blue, and zero values are white with gradual intensity.

    Parameters:
        results_df (DataFrame): Results DataFrame containing 'Name' as the index and significant feature columns.
        significant_features (list): List of significant feature names.
        tick_interval (int): Interval for major ticks on the x-axis.
    """
    print("Starting visualize_composition_heatmap...")

    # Filter out the Intercept from significant_features
    significant_features = [f for f in significant_features if f != "Intercept"]

    # Extract contributions matrix (Features × Names)
    try:
        contributions_matrix = results_df[significant_features]
    except KeyError as e:
        print(f"Error extracting contributions matrix: {e}")
        return

    # Preserve original order of names
    sorted_names = results_df.index.tolist()

    # Transpose to Names × Features (for x-axis as Names, y-axis as Features)
    contributions_matrix = contributions_matrix.T

    # Replace non-finite values with zero (critical to avoid heatmap errors)
    contributions_matrix = contributions_matrix.replace([np.inf, -np.inf, np.nan], 0)

    # Verify that all values are finite
    if not np.all(np.isfinite(contributions_matrix.values)):
        print("Warning: Contributions matrix contains non-finite values. Replacing with zeros.")
        contributions_matrix = contributions_matrix.replace([np.inf, -np.inf, np.nan], 0)

    # Calculate major ticks at midpoints of intervals for x-axis
    num_names = len(sorted_names)
    major_indices = np.arange(0, num_names, tick_interval).astype(int)
    major_ticks = [(i + (i + tick_interval - 1)) / 2 for i in major_indices]  # Midpoint of each interval

    # Calculate major ticks at midpoints of intervals for y-axis
    num_features = len(significant_features)
    y_major_indices = np.arange(0, num_features, tick_interval).astype(int)
    y_major_ticks = [(i + (i + tick_interval - 1)) / 2 for i in y_major_indices]  # Midpoint of each interval

    # Set figure size
    fig_width = 7
    fig_height = 5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Diverging colormap (red-blue with gradual intensity)
    cmap = sns.color_palette("RdBu_r", as_cmap=True)

    # Handle edge cases for vmin, vmax
    min_val = contributions_matrix.values.min()
    max_val = contributions_matrix.values.max()

    # Use SymLogNorm for diverging colors centered at zero with adjusted linthresh and linscale
    linthresh = 0.1  # Narrower linear region
    linscale = 1     # Smaller linear region
    norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=min_val, vmax=max_val)

    # Plot heatmap
    try:
        heatmap = ax.imshow(contributions_matrix, cmap=cmap, norm=norm, aspect='auto')
    except Exception as e:
        print(f"Error plotting heatmap: {e}")
        return

    # Configure major ticks and labels for x-axis
    ax.set_xticks(major_ticks)  # Ticks at midpoints of intervals
    ax.set_xticklabels([sorted_names[i] for i in major_indices],  # Labels at interval starts
                       rotation=90,
                       ha='center',
                       fontsize=10)

    # Configure major ticks and labels for y-axis
    ax.set_yticks(y_major_ticks)  # Ticks at midpoints of intervals
    ax.set_yticklabels([str(i + 1) for i in y_major_indices])  # Labels at interval starts

    # Remove vertical grid lines
    ax.grid(False)

    # Minor ticks at every column (for precise alignment)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(num_names)))
    ax.tick_params(axis='x', which='minor', bottom=True, length=2, color='gray')

    # Format labels and title
    ax.set_title("Contributions Heatmap", fontsize=12, pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("Features", fontsize=10)
    ax.tick_params(axis='y', labelsize=10, rotation=0)

    # Adjust colorbar ticks and labels
    cbar = fig.colorbar(heatmap, ax=ax, label="Contribution")
    if cbar:
        # Calculate specific natural values for colorbar ticks
        num_ticks = 5  # Number of ticks on the colorbar
        cbar_ticks = np.linspace(min_val, max_val, num_ticks)
        cbar_labels = [f"{val:.2f}" for val in cbar_ticks]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_labels)

    # Final adjustments
    plt.tight_layout()
    try:
        plt.savefig("contributions_heatmap.png", dpi=300, bbox_inches='tight')
        print("Heatmap saved to contributions_heatmap.png")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
    plt.show()
