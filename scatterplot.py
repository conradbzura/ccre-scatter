from typing import Any, Callable, NamedTuple

import ipywidgets as widgets
import jscatter
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from ipywidgets import HBox, VBox
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors


class ScatterplotResult(NamedTuple):
    """Named tuple for scatterplot function return value."""
    scatter: Any
    merged_data: pl.DataFrame
    container: Any
    selection: Callable[[], pl.DataFrame]
    class_dropdown: Any


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
    x: pl.DataFrame,
    y: pl.DataFrame,
    metadata: pl.DataFrame,
    join_column: str,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    colormap: Callable | None = None,
    default_class: str = "All",
) -> ScatterplotResult:
    """
    Create a JScatter scatterplot with two datasets and interactive selection.

    Note: This function uses Polars DataFrames for efficient data processing, but
    internally converts to Pandas DataFrames where needed for jscatter compatibility.

    Parameters:
    -----------
    x : pl.DataFrame
        Dataset for Y-axis values
    y : pl.DataFrame
        Dataset for X-axis values
    metadata : pl.DataFrame
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
        - None: Use class-based coloring (default)
    default_class : str, default "All"
        Default class to filter by on initial display. Use "All" to show all classes.
        A dropdown will allow changing the class filter.

    Returns:
    --------
    ScatterplotResult
        Named tuple containing:
        - scatter: The jscatter plot object
        - merged_data: The merged dataset used for plotting
        - container: The plot container widget
        - selection: Function that returns DataFrame of currently selected points
        - class_dropdown: The class filter dropdown widget
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
    merged_xy = x.join(y, on=join_column, how="inner", suffix="_y")

    # Then merge with metadata
    full_merged_data = merged_xy.join(metadata, on=join_column, how="inner")

    if len(full_merged_data) == 0:
        raise ValueError("No matching records found between datasets")

    # Get unique classes for dropdown
    unique_classes = sorted(full_merged_data["class"].unique().to_list())
    available_classes = ["All"] + unique_classes

    # Create static color mapping for all classes using custom color palette
    color_palette = [
        "#8f2be7",  # Purple
        "#fb4fd9",  # Pink
        "#e9162d",  # Red
        "#f28200",  # Orange
        "#ffdb28",  # Yellow
        "#1fb819",  # Green
        "#00e1da",  # Cyan
        "#007bd8",  # Blue
    ]

    # Create mapping of class to color
    class_color_map = {}
    for i, cls in enumerate(unique_classes):
        class_color_map[cls] = color_palette[i % len(color_palette)]

    # Validate default_class
    if default_class not in available_classes:
        print(
            f"Warning: default_class '{default_class}' not found in data. Available classes: {available_classes}"
        )
        if available_classes:
            default_class = available_classes[0]
        else:
            raise ValueError("No class data available for filtering")

    # Initially filter by default class
    if default_class == "All":
        merged_data = full_merged_data
    else:
        merged_data = full_merged_data.filter(pl.col("class") == default_class)

    # Prepare data for plotting
    # Assume we want to plot the first numeric column from each dataset
    # (excluding the join column)
    x_cols = [
        col for col in x.columns if col != join_column and x[col].dtype.is_numeric()
    ]
    y_cols = [
        col for col in y.columns if col != join_column and y[col].dtype.is_numeric()
    ]

    if not x_cols:
        raise ValueError("No numeric columns found in x for plotting")
    if not y_cols:
        raise ValueError("No numeric columns found in y for plotting")

    # Use the first numeric column from each dataset, with suffix handling
    y_col = x_cols[0]
    if y_col in merged_data.columns and y_col + "_y" in merged_data.columns:
        y_col = y_col  # Use the original column from x
    elif y_col + "_y" in merged_data.columns:
        # Column was renamed during join, but we want the one from x (no suffix in Polars join)
        pass

    x_col = y_cols[0]
    if x_col + "_y" in merged_data.columns:
        x_col = x_col + "_y"

    # Extract coordinates and ensure they are proper 1D arrays
    x_coords = merged_data[x_col].to_numpy().ravel()
    y_coords = merged_data[y_col].to_numpy().ravel()

    # Ensure we have valid numeric data
    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("No data points to plot")

    # Remove any NaN values
    valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    # Convert boolean mask to row indices for Polars filtering
    valid_indices = np.where(valid_mask)[0]
    merged_data = merged_data[valid_indices]

    # Create complete plot DataFrame once for all cCREs with their colors
    # Filter out NaN values from the complete dataset
    full_merged_clean = full_merged_data.filter(
        (~pl.col(x_col).is_nan()) & (~pl.col(y_col).is_nan())
    )

    # Extract all coordinates for the complete dataset
    all_x_coords = full_merged_clean[x_col].to_numpy().ravel()
    all_y_coords = full_merged_clean[y_col].to_numpy().ravel()
    all_class_data = full_merged_clean["class"].to_numpy()

    # Create complete plot DataFrame with colors for all cCREs
    complete_plot_df = pd.DataFrame({
        "x_data": all_x_coords,
        "y_data": all_y_coords,
        "class": all_class_data
    })

    if colormap is not None:
        # Use density-based coloring for all points
        points = np.column_stack([all_x_coords, all_y_coords])
        complete_plot_df["colormap"] = colormap(points)
    else:
        # Use class-based static coloring for all points
        complete_plot_df["color"] = [class_color_map[cls] for cls in all_class_data]

    # Set default labels
    if x_label is None:
        x_label = f"{x_col}"
    if y_label is None:
        y_label = f"{y_col}"

    # Create selection tracking for selected points
    selected_data = pl.DataFrame()  # Will hold selected points data

    # Storage for current plot components
    current_scatter = None
    current_plot_widget = None

    def create_plot(selected_class="All"):
        """Create a scatter plot for the selected class by filtering complete_plot_df"""
        nonlocal current_scatter, current_plot_widget, selected_data

        # Filter the complete plot DataFrame based on selected class (no copy needed)
        if selected_class == "All":
            plot_df = complete_plot_df
        else:
            plot_df = complete_plot_df[complete_plot_df["class"] == selected_class]

        if len(plot_df) == 0:
            print(f"No valid data points for class: {selected_class}")
            return None

        # Calculate shared axis limits for equal coordinate systems
        filter_x_coords = plot_df["x_data"].to_numpy()
        filter_y_coords = plot_df["y_data"].to_numpy()

        x_min, x_max = filter_x_coords.min(), filter_x_coords.max()
        y_min, y_max = filter_y_coords.min(), filter_y_coords.max()

        # Use the same range for both axes to create shared coordinate system
        overall_min = min(x_min, y_min)
        overall_max = max(x_max, y_max)

        # Add a small padding
        padding = (overall_max - overall_min) * 0.05
        axis_min = overall_min - padding
        axis_max = overall_max + padding

        print(
            f"Data ranges: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]"
        )
        print(f"Shared axis range: [{axis_min:.2f}, {axis_max:.2f}]")

        return plot_df, axis_min, axis_max

    # Initial plot creation
    plot_result = create_plot(default_class)
    if plot_result is None:
        raise ValueError("No valid data points to plot")

    plot_df, axis_min, axis_max = plot_result

    # Create scatter plot using jscatter.Scatter (not jscatter.plot)
    try:
        # plot_df is already prepared with all necessary data and colors

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

        # Configure coloring
        if colormap is not None and "colormap" in plot_df.columns:
            # Use color mapping for density
            scatter.color(by="colormap", map="viridis")
        elif "color" in plot_df.columns:
            # Use static class-based coloring with fixed color mapping
            # Create a fixed categorical mapping for all possible classes
            scatter.color(by="class", map=class_color_map)
        else:
            # Fallback to density-based opacity for better visualization of overlapping points
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
            selected_data = merged_data[selected_indices]
            print(f"Selected {len(selected_indices)} points")
        else:
            # No selection - empty DataFrame
            selected_data = pl.DataFrame()
            print("No points selected")

    # Connect selection callback for jupyter-scatter
    if hasattr(scatter, "widget") and hasattr(scatter.widget, "selection"):
        # Based on jupyter-scatter documentation, use scatter.widget.selection
        scatter.widget.observe(on_selection_change, names=["selection"])
    elif hasattr(scatter, "selection"):
        scatter.observe(on_selection_change, names=["selection"])
    else:
        # Fallback: check if scatter itself has the selection trait
        print("Could not find selection trait - selection updates may not work")

    # Create class filter dropdown
    class_dropdown = widgets.Dropdown(
        options=available_classes,
        value=default_class,
        description="Class:",
        disabled=False,
    )

    def update_plot(change):
        """Update the plot when class filter changes"""
        nonlocal current_scatter, current_plot_widget, merged_data, scatter

        selected_class = change["new"]
        print(f"Filtering by class: {selected_class}")


        # Create new plot with filtered data
        plot_result = create_plot(selected_class)
        if plot_result is None:
            return

        new_plot_df, new_axis_min, new_axis_max = plot_result

        # Create new scatter plot
        try:
            # new_plot_df already has all the data and colors prepared

            # Create new scatter plot
            new_scatter = jscatter.Scatter(
                data=new_plot_df,
                x="x_data",
                y="y_data",
                x_label=x_label,
                y_label=y_label,
                width=500,
                height=500,
                aspect_ratio=1.0,
                axes=True,
                axes_grid=True,
            )

            # Set axis ranges
            shared_range = (new_axis_min, new_axis_max)
            new_scatter.x("x_data", scale=shared_range)
            new_scatter.y("y_data", scale=shared_range)

            # Configure coloring
            if colormap is not None and "colormap" in new_plot_df.columns:
                # Use color mapping for density
                new_scatter.color(by="colormap", map="viridis")
            elif "color" in new_plot_df.columns:
                # Use static class-based coloring with fixed color mapping
                # Create a fixed categorical mapping for all possible classes
                new_scatter.color(by="class", map=class_color_map)
            else:
                # Fallback to density-based opacity
                new_scatter.opacity(by="density")

            # Connect selection callback
            try:
                if hasattr(new_scatter, "widget") and hasattr(
                    new_scatter.widget, "selection"
                ):
                    new_scatter.widget.observe(on_selection_change, names=["selection"])
                elif hasattr(new_scatter, "selection"):
                    new_scatter.observe(on_selection_change, names=["selection"])
            except Exception as e:
                print(f"Warning: Could not connect selection callback: {e}")

            # Update the plot widget in the container
            new_plot_widget = new_scatter.show()

            # Update container children
            container_children = list(container.children)
            # Find and replace the plot widget (it should be the last one)
            for i in range(len(container_children) - 1, -1, -1):
                if (
                    hasattr(container_children[i], "children")
                    or str(type(container_children[i])).find("jscatter") >= 0
                ):
                    container_children[i] = new_plot_widget
                    break

            container.children = container_children

            # Update references
            current_scatter = new_scatter
            current_plot_widget = new_plot_widget
            scatter = new_scatter

        except Exception as e:
            print(f"Error updating plot: {e}")

    # Connect dropdown callback
    class_dropdown.observe(update_plot, names="value")

    # Create layout with plot and dropdown
    plot_widget = scatter.show()
    current_plot_widget = plot_widget

    # Create filter controls
    filter_controls = HBox([class_dropdown])

    if title:
        # Create container widget with title
        title_widget = widgets.HTML(f"<h3>{title}</h3>")
        container = VBox([title_widget, filter_controls, plot_widget])
    elif x_label and y_label and not title:
        title_widget = widgets.HTML(f"<h3>{y_label} vs. {x_label}</h3>")
        container = VBox([title_widget, filter_controls, plot_widget])
    else:
        container = VBox([filter_controls, plot_widget])

    # Make the main container responsive
    if hasattr(container, "layout"):
        container.layout.width = "100%"
        container.layout.padding = "18px"  # Add padding to container

    # Display the container
    display(container)

    # Create selection handle that returns currently selected data
    def get_selection():
        """Return DataFrame of currently selected points, empty if none selected"""
        return selected_data.clone() if len(selected_data) > 0 else pl.DataFrame()

    return ScatterplotResult(
        scatter=scatter,
        merged_data=merged_data,
        container=container,
        selection=get_selection,
        class_dropdown=class_dropdown,
    )
