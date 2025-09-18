from typing import Any, Callable, NamedTuple

import ipywidgets as widgets
import jscatter
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from ipywidgets import HBox, VBox
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


class ScatterplotResult(NamedTuple):
    """Named tuple for scatterplot function return value."""

    scatter: Any
    merged_data: pl.DataFrame
    container: Any
    selection: Callable[[], pl.DataFrame]
    class_dropdown: Any


def kde(bandwidth=1.0):
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


def radius(radius=1.0):
    def calculate_radius_density(points):
        """Calculate density based on points within radius"""
        tree = KDTree(points)
        counts = tree.query_radius(points, r=radius, count_only=True)
        return np.array(counts, dtype=float)

    return calculate_radius_density


def create_class_legend(unique_classes, class_color_map):
    """
    Create a legend widget showing class colors.

    Parameters:
    -----------
    unique_classes : list
        List of unique class names
    class_color_map : dict
        Mapping of class names to colors

    Returns:
    --------
    ipywidgets.VBox
        Legend widget
    """
    legend_items = []
    for cls in unique_classes:
        color = class_color_map[cls]
        legend_items.append(
            widgets.HTML(
                f'<div style="display: flex; align-items: center; margin: 2px 0;">'
                f'<div style="width: 12px; height: 12px; background-color: {color}; '
                f'border: 1px solid #ccc; margin-right: 8px; border-radius: 2px;"></div>'
                f'<span style="font-size: 12px;">{cls}</span></div>'
            )
        )

    legend_box = VBox(legend_items)
    legend_box.layout.padding = "8px"
    legend_box.layout.border = "1px solid #ddd"
    legend_box.layout.border_radius = "4px"
    legend_box.layout.background_color = "#f9f9f9"
    legend_box.layout.width = "auto"
    legend_box.layout.max_width = "200px"

    legend_title = widgets.HTML(
        '<div style="font-weight: bold; margin-bottom: 4px; font-size: 13px;">cCRE Class Legend</div>'
    )
    return VBox([legend_title, legend_box])


def scatterplot(
    x: pl.DataFrame,
    y: pl.DataFrame,
    metadata: pl.DataFrame,
    join_column: str,
    category_column: str = "class",
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    colormap: Callable | None = None,
    default_category: str = "All",
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
        Metadata describing the cCREs with columns including the category column
    join_column : str
        Column name to join the datasets on
    category_column : str, default "class"
        Column name in metadata to use for categorical coloring and filtering
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
    default_category : str, default "All"
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
    required_metadata_cols = ["rDHS", "cCRE", "chr", "start", "end"]
    if category_column not in required_metadata_cols:
        required_metadata_cols.append(category_column)

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
    unique_classes = sorted(full_merged_data[category_column].unique().to_list())
    available_classes = ["All"] + unique_classes

    # Create interpolated colormap based on number of unique categories
    def create_interpolated_colormap(n_categories: int) -> tuple[list[str], dict[str, str]]:
        """Create an interpolated colormap with exactly n_categories colors."""
        # Base color palette for interpolation
        base_colors = [
            "#8f2be7",  # Purple
            "#fb4fd9",  # Pink
            "#e9162d",  # Red
            "#f28200",  # Orange
            "#ffdb28",  # Yellow
            "#1fb819",  # Green
            "#00e1da",  # Cyan
            "#007bd8",  # Blue
        ]

        if n_categories <= len(base_colors):
            # Use base colors directly if we have enough
            selected_colors = base_colors[:n_categories]
        else:
            # Create a custom colormap and interpolate
            # Create colormap from base colors
            cmap = LinearSegmentedColormap.from_list(
                "custom", base_colors, N=n_categories
            )
            # Sample colors evenly across the colormap
            selected_colors = [
                plt.colors.rgb2hex(cmap(i / (n_categories - 1) if n_categories > 1 else 0))
                for i in range(n_categories)
            ]

        return selected_colors

    # Generate colors for exact number of categories
    n_categories = len(unique_classes)
    class_color_list = create_interpolated_colormap(n_categories)

    # Create mapping of class to color for legend
    class_color_map = {}
    for i, cls in enumerate(unique_classes):
        class_color_map[cls] = class_color_list[i]

    # Validate default_category
    if default_category not in available_classes:
        print(
            f"Warning: default_category '{default_category}' not found in data. Available classes: {available_classes}"
        )
        if available_classes:
            default_category = available_classes[0]
        else:
            raise ValueError("No class data available for filtering")

    # Initially filter by default class
    if default_category == "All":
        merged_data = full_merged_data
    else:
        merged_data = full_merged_data.filter(pl.col(category_column) == default_category)

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
    all_class_data = full_merged_clean[category_column].to_numpy()

    # Create complete plot DataFrame with properly formatted categorical data
    complete_plot_df = pd.DataFrame(
        {"x_data": all_x_coords, "y_data": all_y_coords, category_column: all_class_data}
    )

    # Ensure categorical column has consistent categories for all possible classes
    complete_plot_df[category_column] = pd.Categorical(
        complete_plot_df[category_column],
        categories=unique_classes,
        ordered=True
    )

    if colormap is not None:
        # Use density-based coloring for all points
        points = np.column_stack([all_x_coords, all_y_coords])
        complete_plot_df["colormap"] = colormap(points)

    # Set default labels using actual column names from the data
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col

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
            plot_df = complete_plot_df[complete_plot_df[category_column] == selected_class]

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
    plot_result = create_plot(default_category)
    if plot_result is None:
        raise ValueError("No valid data points to plot")

    plot_df, axis_min, axis_max = plot_result

    # Create scatter plot using jscatter.Scatter (not jscatter.plot)
    try:
        # plot_df is already prepared with all necessary data and colors

        # Create diagonal line annotation extending far beyond data range
        # Use 100x the data range to ensure it remains visible during panning
        data_range = axis_max - axis_min
        line_extend = data_range * 50  # Extend 50x beyond each side
        line_min = axis_min - line_extend
        line_max = axis_max + line_extend
        # Style to match typical grid lines: light gray, thin
        diagonal_line = jscatter.Line(
            [(line_min, line_min), (line_max, line_max)],
            line_color="#e0e0e0",
            line_width=1,
        )

        # Create scatter plot using the correct API with square aspect ratio
        if colormap is not None and "colormap" in plot_df.columns:
            # Use density-based coloring
            scatter = jscatter.Scatter(
                data=plot_df,
                x="x_data",
                y="y_data",
                x_label=x_label,
                y_label=y_label,
                width=500,
                height=500,
                aspect_ratio=1.0,
                axes=True,
                axes_grid=True,
                annotations=[diagonal_line],
                color_by="colormap",
                color_map="viridis"
            )
        else:
            # Use class-based coloring with predefined color list
            scatter = jscatter.Scatter(
                data=plot_df,
                x="x_data",
                y="y_data",
                x_label=x_label,
                y_label=y_label,
                width=500,
                height=500,
                aspect_ratio=1.0,
                axes=True,
                axes_grid=True,
                annotations=[diagonal_line],
                color_by=category_column,
                color_map=class_color_list
            )

        # Set identical axis ranges using the scale parameter
        shared_range = (axis_min, axis_max)
        scatter.x("x_data", scale=shared_range)
        scatter.y("y_data", scale=shared_range)

        # Set axis labels using the axes() method
        scatter.axes(axes=True, grid=True, labels=[x_label, y_label])

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

    scatter.widget.observe(on_selection_change, names=["selection"])

    # Create class filter dropdown
    class_dropdown = widgets.Dropdown(
        options=available_classes,
        value=default_category,
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

        # Update existing scatter plot in-place instead of recreating widget
        try:
            # Use scatter.data() now that we're using list-based color mapping
            # Disable animation and keep existing scales
            scatter.data(new_plot_df, reset_scales=False, animate=False)

            # Keep the original axis ranges static - don't recalculate based on filtered data
            # This prevents zooming/panning animation between class changes
            original_shared_range = (axis_min, axis_max)
            scatter.x("x_data", scale=original_shared_range)
            scatter.y("y_data", scale=original_shared_range)

            print(f"Updated plot in-place for class: {selected_class}")

        except Exception as e:
            raise
            print(f"Error updating plot in-place: {e}")
            print("Falling back to widget recreation...")

            # Fallback to original method if in-place update fails
            try:
                # Create diagonal line annotation extending far beyond data range
                new_data_range = new_axis_max - new_axis_min
                new_line_extend = new_data_range * 50  # Extend 50x beyond each side
                new_line_min = new_axis_min - new_line_extend
                new_line_max = new_axis_max + new_line_extend
                # Style to match typical grid lines: light gray, thin
                new_diagonal_line = jscatter.Line(
                    [(new_line_min, new_line_min), (new_line_max, new_line_max)],
                    line_color="#e0e0e0",
                    line_width=1,
                )

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
                    annotations=[new_diagonal_line],
                )

                # Set axis ranges
                shared_range = (new_axis_min, new_axis_max)
                new_scatter.x("x_data", scale=shared_range)
                new_scatter.y("y_data", scale=shared_range)

                # Configure coloring
                if colormap is not None and "colormap" in new_plot_df.columns:
                    new_scatter.color(by="colormap", map="viridis")
                else:
                    new_scatter.color(by=category_column, map=class_color_list)

                # Connect selection callback
                if hasattr(new_scatter, "widget") and hasattr(
                    new_scatter.widget, "selection"
                ):
                    new_scatter.widget.observe(on_selection_change, names=["selection"])

                # Replace widget
                new_plot_widget = new_scatter.show()
                container_children = list(container.children)

                for i in range(len(container_children) - 1, -1, -1):
                    if hasattr(container_children[i], "children"):
                        plot_legend_children = list(container_children[i].children)
                        if len(plot_legend_children) > 0:
                            container_children[i].children = [
                                new_plot_widget
                            ] + plot_legend_children[1:]
                        break
                    elif str(type(container_children[i])).find("jscatter") >= 0:
                        container_children[i] = new_plot_widget
                        break

                container.children = container_children

                # Update references
                scatter = new_scatter
                current_scatter = new_scatter
                current_plot_widget = new_plot_widget

            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")

    # Connect dropdown callback
    class_dropdown.observe(update_plot, names="value")

    # Create layout with plot and dropdown
    plot_widget = scatter.show()
    current_plot_widget = plot_widget

    # Create legend for class colors (only when using class-based coloring)
    legend_widget = None
    if colormap is None:  # Only show legend for class-based coloring
        legend_widget = create_class_legend(unique_classes, class_color_map)

    # Create filter controls (just dropdown, legend will be separate)
    filter_controls = HBox([class_dropdown])

    # Create plot area with legend positioned to the right
    if legend_widget:
        # Create a container with plot on left and legend on right
        plot_with_legend = HBox([plot_widget, legend_widget])
    else:
        plot_with_legend = plot_widget

    if title:
        # Create container widget with user-specified title
        title_widget = widgets.HTML(f"<h3>{title}</h3>")
        container = VBox([title_widget, filter_controls, plot_with_legend])
    else:
        # No title - just the controls and plot
        container = VBox([filter_controls, plot_with_legend])

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
