from typing import Optional, Dict, Any, Callable
import pandas as pd
import jscatter
import numpy as np
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import VBox, HBox
from sklearn.neighbors import NearestNeighbors, KDTree, KernelDensity


def kde(bandwidth=1):
    def calculate_kde_density(points):
        """Calculate density using Gaussian kernel"""
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(points)
        log_density = kde.score_samples(points)
        return np.exp(log_density)  # Convert from log density
    return calculate_kde_density


def knn(k=100):
    def calculate_knn_density(points):
        """Calculate density based on distance to k-th nearest neighbor"""
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)  # +1 to exclude self
        distances, _ = nbrs.kneighbors(points)
        # Use inverse of average distance to k neighbors as density measure
        density = 1 / (distances[:, 1:].mean(axis=1) + 1e-10)  # avoid division by zero
        return density

    return calculate_knn_density


def radius(radius=1):
    def calculate_radius_density(points):
        """Calculate density based on points within radius"""
        tree = KDTree(points)
        counts = tree.query_radius(points, r=radius, count_only=True)
        return np.array(counts, dtype=float)
    return calculate_radius_density


def scatterplot(
    x: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    join_column: str,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    colormap: Callable | None = None
) -> Dict[str, Any]:
    """
    Create a JScatter scatterplot with two datasets and interactive selection.

    Parameters:
    -----------
    x : pd.DataFrame
        Dataset for Y-axis values
    y : pd.DataFrame
        Dataset for X-axis values
    metadata : pd.DataFrame
        Metadata describing the cCREs with columns: rDHS, cCRE, chr, start, end, class
    join_column : str
        Column name to join the datasets on
    x_label : str, optional
        Custom label for X-axis
    y_label : str, optional
        Custom label for Y-axis
    title : str, optional
        Custom title for the plot
    colormap : Callable, optional
        Function for density-based coloring. Use kde(), knn(), or radius() functions.
        - kde(bandwidth=1): Kernel density estimation
        - knn(k=100): K-nearest neighbors density
        - radius(radius=1): Points within radius density
        - None: No density coloring (default)

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'scatter': The jscatter plot object
        - 'merged_data': The merged dataset used for plotting
        - 'container': The plot container widget
        - 'selection': Function that returns DataFrame of currently selected points
    """

    # Validate inputs
    if join_column not in x.columns:
        raise ValueError(f"Column '{join_column}' not found in x")
    if join_column not in y.columns:
        raise ValueError(f"Column '{join_column}' not found in y")
    if join_column not in metadata.columns:
        raise ValueError(f"Column '{join_column}' not found in metadata")

    # Validate metadata columns
    required_metadata_cols = ["rDHS", "cCRE", "chr", "start", "end", "class"]
    missing_cols = [
        col for col in required_metadata_cols if col not in metadata.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required metadata columns: {missing_cols}")

    # Merge datasets
    # First merge x and y
    merged_xy = pd.merge(x, y, on=join_column, how="inner", suffixes=("_x", "_y"))

    # Then merge with metadata
    merged_data = pd.merge(merged_xy, metadata, on=join_column, how="inner")

    if len(merged_data) == 0:
        raise ValueError("No matching records found between datasets")

    # Prepare data for plotting
    # Assume we want to plot the first numeric column from each dataset
    # (excluding the join column)
    x_cols = [
        col
        for col in x.columns
        if col != join_column and pd.api.types.is_numeric_dtype(x[col])
    ]
    y_cols = [
        col
        for col in y.columns
        if col != join_column and pd.api.types.is_numeric_dtype(y[col])
    ]

    if not x_cols:
        raise ValueError("No numeric columns found in x for plotting")
    if not y_cols:
        raise ValueError("No numeric columns found in y for plotting")

    # Use the first numeric column from each dataset, with suffix handling
    y_col = x_cols[0]
    if y_col + "_x" in merged_data.columns:
        y_col = y_col + "_x"

    x_col = y_cols[0]
    if x_col + "_y" in merged_data.columns:
        x_col = x_col + "_y"

    # Extract coordinates and ensure they are proper 1D arrays
    x_coords = merged_data[x_col].values.flatten()
    y_coords = merged_data[y_col].values.flatten()

    # Ensure we have valid numeric data
    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("No data points to plot")

    # Remove any NaN values
    valid_mask = ~(pd.isna(x_coords) | pd.isna(y_coords))
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    merged_data = merged_data[valid_mask]

    # Set default labels
    if x_label is None:
        x_label = f"{x_col}"
    if y_label is None:
        y_label = f"{y_col}"

    # Create selection tracking for selected points
    selected_data = pd.DataFrame()  # Will hold selected points data

    # Calculate shared axis limits for equal coordinate systems
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Use the same range for both axes to create shared coordinate system
    overall_min = min(x_min, y_min)
    overall_max = max(x_max, y_max)

    # Add a small padding
    padding = (overall_max - overall_min) * 0.05
    axis_min = overall_min - padding
    axis_max = overall_max + padding

    print(f"Data ranges: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    print(f"Shared axis range: [{axis_min:.2f}, {axis_max:.2f}]")

    # Calculate density if requested
    colors = None
    if colormap is not None:
        points = np.column_stack([x_coords, y_coords])
        colors = colormap(points)

    # Create scatter plot using jscatter.Scatter (not jscatter.plot)
    try:
        # Prepare data for jscatter.Scatter - it expects a DataFrame
        plot_df = pd.DataFrame({"x_data": x_coords, "y_data": y_coords})
        if colors is not None:
            plot_df["colormap"] = colors

        # Create scatter plot using the correct API with square aspect ratio
        scatter = jscatter.Scatter(
            data=plot_df,
            x="x_data",
            y="y_data",
            x_label=x_label,
            y_label=y_label,
            width=500,
            height=500,
            aspect_ratio=1.0,  # Square aspect ratio
            axes=True,  # Enable axes
            axes_grid=True,  # Show grid lines
        )

        # Set identical axis ranges using the scale parameter
        shared_range = (axis_min, axis_max)
        scatter.x("x_data", scale=shared_range)
        scatter.y("y_data", scale=shared_range)

        # Configure density-based coloring if requested
        if colors is not None:
            # Use color mapping for density while keeping the built-in density opacity
            scatter.color(by="colormap", map="viridis")
        else:
            # Enable density-based opacity for better visualization of overlapping points
            scatter.opacity(by="density")

        # Workaround for jupyter-scatter Button widget bug
        # Add missing _dblclick_handler attribute to prevent AttributeError
        try:
            if hasattr(scatter, "widget") and hasattr(scatter.widget, "children"):
                for widget in scatter.widget.children:
                    if hasattr(widget, "children"):
                        for child in widget.children:
                            if hasattr(child, "_click_handler") and not hasattr(
                                child, "_dblclick_handler"
                            ):
                                child._dblclick_handler = None
        except Exception as e:
            print(f"Warning: Could not apply Button widget fix: {e}")

    except Exception as e:
        print(f"Error creating plot: {e}")
        # Fallback: try basic version
        plot_df = pd.DataFrame({"x_data": x_coords, "y_data": y_coords})
        scatter = jscatter.Scatter(data=plot_df, x="x_data", y="y_data")

    # Set up selection callback for lasso selection
    def on_selection_change(change):
        """Callback for when selection changes in the scatter plot."""
        nonlocal selected_data
        # Get the new selection indices
        selected_indices = change.get("new", [])
        if selected_indices is not None and len(selected_indices) > 0:
            # Convert to list if it's a numpy array
            if hasattr(selected_indices, "tolist"):
                selected_indices = selected_indices.tolist()
            # Update selected_data with the selected points
            selected_data = merged_data.iloc[selected_indices].copy()
            print(f"Selected {len(selected_indices)} points")
        else:
            # No selection - empty DataFrame
            selected_data = pd.DataFrame()
            print("No points selected")

    # Connect selection callback for jupyter-scatter
    try:
        # Based on jupyter-scatter documentation, use scatter.widget.selection
        if hasattr(scatter, "widget") and hasattr(scatter.widget, "selection"):
            print("Connecting to scatter.widget.selection trait")
            scatter.widget.observe(on_selection_change, names=["selection"])
        elif hasattr(scatter, "selection"):
            print("Connecting to scatter.selection trait")
            scatter.observe(on_selection_change, names=["selection"])
        else:
            # Fallback: check if scatter itself has the selection trait
            print(
                f"Available traits on scatter: {scatter.trait_names() if hasattr(scatter, 'trait_names') else 'no trait_names method'}"
            )
            print("Could not find selection trait - selection updates may not work")

    except Exception as e:
        print(f"Warning: Could not connect selection callback: {e}")
        print("Lasso selection updates may not work automatically")

    # Create layout with just the plot
    plot_widget = scatter.show()

    if title:
        # Create container widget with title
        title_widget = widgets.HTML(f"<h3>{title}</h3>")
        container = VBox([title_widget, plot_widget])
    elif x_label and y_label and not title:
        title_widget = widgets.HTML(f"<h3>{y_label} vs. {x_label}</h3>")
        container = VBox([title_widget, plot_widget])
    else:
        container = plot_widget

    # Make the main container responsive
    if hasattr(container, 'layout'):
        container.layout.width = "100%"
        container.layout.padding = "18px"  # Add padding to container

    # Display the container
    display(container)

    # Create selection handle that returns currently selected data
    def get_selection():
        """Return DataFrame of currently selected points, empty if none selected"""
        return selected_data.copy() if not selected_data.empty else pd.DataFrame()

    return {
        "scatter": scatter,
        "merged_data": merged_data,
        "container": container,
        "selection": get_selection,
    }
