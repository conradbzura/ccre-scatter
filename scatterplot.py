from typing import Any, Callable, NamedTuple

import ipywidgets as widgets
import jscatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from ipywidgets import HBox, VBox
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors
from ipydatagrid import DataGrid


class ScatterplotResult(NamedTuple):
    """Named tuple for scatterplot function return value."""

    scatter: Any
    merged_data: pl.DataFrame
    container: Any
    selection: Callable[[], pl.DataFrame]
    class_dropdown: Any
    metadata_table: Any
    datagrid: Any


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


def _handle_selection_change(
    change, datagrid, metadata_table_widget, apply_button, status_message
):
    """Handle metadata table selection changes."""
    selections = change["new"]
    datagrid.append(change)

    # Calculate total number of rows selected across all selections
    total_rows_selected = 0
    for selection in selections:
        rows_in_selection = abs(selection["r2"] - selection["r1"]) + 1
        total_rows_selected += rows_in_selection

    # Change selection styling and button state based on number of selected rows
    if total_rows_selected > 2:
        # More than 2 rows selected - use warning colors
        warning_style = {
            "selection_fill_color": "rgba(255, 0, 0, 0.2)",
            "selection_border_color": "rgb(255, 0, 0)",
        }
        if metadata_table_widget:
            metadata_table_widget.grid_style = warning_style
        apply_button.disabled = True
        status_message.value = f"<span style='color: #dc3545;'>Too many rows selected ({total_rows_selected}). Select exactly 2 rows.</span>"
    elif total_rows_selected == 2:
        # Exactly 2 rows - use success colors
        success_style = {
            "selection_fill_color": "rgba(0, 128, 0, 0.2)",
            "selection_border_color": "rgb(0, 128, 0)",
        }
        if metadata_table_widget:
            metadata_table_widget.grid_style = success_style
        apply_button.disabled = False
        status_message.value = (
            "<span style='color: #28a745;'>✓ Ready to apply selection</span>"
        )
    else:
        # Less than 2 rows - use default styling
        default_style = {
            "selection_fill_color": "rgba(0, 123, 255, 0.2)",
            "selection_border_color": "rgb(0, 123, 255)",
        }
        if metadata_table_widget:
            metadata_table_widget.grid_style = default_style
        apply_button.disabled = True
        if total_rows_selected == 0:
            status_message.value = "<span style='color: #666; font-style: italic;'>Select exactly 2 rows to enable</span>"
        else:
            status_message.value = f"<span style='color: #ffc107;'>Need more rows selected ({total_rows_selected}/2)</span>"


def _create_plot_data(
    full_merged_data,
    x_col,
    y_col,
    category_column,
    unique_classes,
    join_column,
    selected_class="All",
):
    """Create plot data for the selected class, returning pandas DataFrame and current plot data for selection."""
    # Apply class filtering on the Polars DataFrame
    if selected_class == "All":
        filtered_data = full_merged_data
    else:
        filtered_data = full_merged_data.filter(
            pl.col(category_column) == selected_class
        )

    if len(filtered_data) == 0:
        print(f"No valid data points for class: {selected_class}")
        return None, None, None, None

    # Extract coordinates for axis calculations
    x_coords = filtered_data[x_col].to_numpy().ravel()
    y_coords = filtered_data[y_col].to_numpy().ravel()

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

    # Convert to pandas DataFrame for jscatter
    class_data = filtered_data[category_column].to_numpy()
    join_data = filtered_data[join_column].to_numpy()
    plot_df = pd.DataFrame(
        {
            "x_data": x_coords,
            "y_data": y_coords,
            category_column: class_data,
            join_column: join_data,
        }
    )

    # Ensure categorical column has consistent categories
    plot_df[category_column] = pd.Categorical(
        plot_df[category_column], categories=unique_classes, ordered=True
    )

    # Add colormap if it exists in the polars data
    if "colormap" in filtered_data.columns:
        colormap_values = filtered_data["colormap"].to_numpy()
        plot_df["colormap"] = colormap_values

    # Reset index to ensure proper correspondence with jscatter selection
    plot_df.reset_index(drop=True, inplace=True)

    # Convert plot_df back to polars for consistency with selection handling
    plot_data_for_selection = pl.from_pandas(plot_df)

    return plot_df, axis_min, axis_max, plot_data_for_selection


def _handle_apply_click(
    metadata_table_widget,
    sample_metadata_df,
    available_columns,
    complete_data_scan,
    join_column,
    metadata,
    category_column,
    debug_output,
):
    """Handle apply button click to update plot with selected samples"""
    with debug_output:
        print("=" * 50)
        print("🔥 BUTTON CLICKED! 🔥")

        # Get selected rows
        if not metadata_table_widget or not hasattr(
            metadata_table_widget, "selections"
        ):
            print("❌ No metadata table available")
            return None

        selections = metadata_table_widget.selections
        print(f"📋 Current selections: {selections}")

        if not selections or len(selections) == 0:
            print("❌ No rows selected")
            return None

        sample_metadata_pandas = sample_metadata_df.to_pandas()
        print(f"📊 Sample metadata columns: {list(sample_metadata_pandas.columns)}")

        selected_biosamples = []
        for selection in selections:
            print(f"🔍 Processing selection: {selection}")
            for row_idx in range(selection["r1"], selection["r2"] + 1):
                if "biosample" in sample_metadata_pandas.columns:
                    biosample_value = sample_metadata_pandas.iloc[row_idx]["biosample"]
                else:
                    # Try the first column if biosample doesn't exist
                    first_col = sample_metadata_pandas.columns[0]
                    biosample_value = sample_metadata_pandas.iloc[row_idx][first_col]
                    print(
                        f"⚠️  'biosample' column not found, using '{first_col}' instead"
                    )

                selected_biosamples.append(biosample_value)
                print(f"✅ Row {row_idx}: {biosample_value}")

        print(f"🎯 Selected biosamples: {selected_biosamples}")

        # Create new x and y dataframes with only selected biosample columns
        if len(selected_biosamples) == 2:
            biosample1, biosample2 = selected_biosamples[0], selected_biosamples[1]

            # Create new x and y dataframes from the complete data file using lazy loading
            if biosample1 in available_columns:
                new_x = complete_data_scan.select([join_column, biosample1]).collect()
                print(f"✅ Created new x dataframe with columns: {new_x.columns}")
            else:
                print(f"❌ ERROR: {biosample1} not found in complete data columns")
                print(f"Available columns: {available_columns}")
                return None

            if biosample2 in available_columns:
                new_y = complete_data_scan.select([join_column, biosample2]).collect()
                print(f"✅ Created new y dataframe with columns: {new_y.columns}")
            else:
                print(f"❌ ERROR: {biosample2} not found in complete data columns")
                print(f"Available columns: {available_columns}")
                return None

            # Merge the new datasets with metadata
            print("🔗 Merging new datasets...")
            merged_xy = new_x.join(new_y, on=join_column, how="inner", suffix="_y")
            new_full_merged_data = merged_xy.join(metadata, on=join_column, how="inner")

            if len(new_full_merged_data) == 0:
                print("❌ ERROR: No matching records found between new datasets")
                return None

            # Find the correct column names after join
            if biosample1 in new_full_merged_data.columns:
                actual_x_col = biosample1
            elif biosample1 + "_y" in new_full_merged_data.columns:
                actual_x_col = biosample1 + "_y"
            else:
                print(f"❌ ERROR: Cannot find column for {biosample1}")
                return None

            if biosample2 in new_full_merged_data.columns:
                actual_y_col = biosample2
            elif biosample2 + "_y" in new_full_merged_data.columns:
                actual_y_col = biosample2 + "_y"
            else:
                print(f"❌ ERROR: Cannot find column for {biosample2}")
                return None

            print(f"📈 Using columns: X={actual_x_col}, Y={actual_y_col}")

            # Filter out NaN values
            print("🧹 Filtering NaN values...")
            new_merged_clean = new_full_merged_data.filter(
                (~pl.col(actual_x_col).is_nan()) & (~pl.col(actual_y_col).is_nan())
            )
            print(f"   Clean data: {len(new_merged_clean)} rows")

            # Return the new data to replace the global state
            return {
                "full_merged_data": new_merged_clean,
                "x_col": actual_x_col,
                "y_col": actual_y_col,
                "x_label": biosample1,
                "y_label": biosample2,
            }
        else:
            print(
                f"❌ ERROR: Expected exactly 2 biosamples, got {len(selected_biosamples)}"
            )
            return None


def _initialize_metadata_table(
    sample_metadata_file, sample_id_column, data_file, join_column
):
    """Initialize metadata table widgets and data structures if all required parameters are provided.

    Returns:
        Tuple of (sample_metadata_df, metadata_table_widget, datagrid, apply_button,
                 status_message, debug_output, complete_data_scan, available_columns,
                 selection_handler, apply_handler)
    """
    sample_metadata_df = None
    metadata_table_widget = None
    datagrid = []
    apply_button = None
    status_message = None
    debug_output = None
    complete_data_scan = None
    available_columns = []
    selection_handler = None
    apply_handler = None

    if sample_metadata_file and sample_id_column and data_file:
        try:
            # Load sample metadata
            sample_metadata_df = pl.read_parquet(sample_metadata_file)
            if sample_id_column not in sample_metadata_df.columns:
                raise ValueError(
                    f"Column '{sample_id_column}' not found in sample metadata"
                )

            # Get column names from the data file without loading into memory
            data_scan = pl.scan_parquet(data_file)
            available_columns = data_scan.columns

            if join_column not in available_columns:
                raise ValueError(f"Column '{join_column}' not found in data file")

            print(f"Data file has {len(available_columns)} columns available")

            # Store the lazy frame for later use - we'll only load specific columns when needed
            complete_data_scan = data_scan

            # Create ipydatagrid widget for the metadata
            sample_metadata_pandas = sample_metadata_df.to_pandas()
            metadata_table_widget = DataGrid(
                sample_metadata_pandas,
                base_row_size=30,
                base_column_size=120,
                selection_mode="row",
                layout={"height": "400px", "width": "100%"},
            )

            # Create apply button and status message
            apply_button = widgets.Button(
                description="Apply Selection", button_style="primary", disabled=True
            )
            status_message = widgets.HTML(
                value="<span style='color: #666; font-style: italic;'>Select exactly 2 rows to enable</span>"
            )

            # Create output widget to capture debug messages from button clicks
            debug_output = widgets.Output()
            debug_output.layout.height = "300px"
            debug_output.layout.width = "100%"
            debug_output.layout.border = "1px solid #ddd"
            debug_output.layout.overflow = "auto"  # Make it scrollable
            debug_output.layout.padding = "10px"

            # Create selection handler using partial application
            def limit_selection(change):
                _handle_selection_change(
                    change,
                    datagrid,
                    metadata_table_widget,
                    apply_button,
                    status_message,
                )

            selection_handler = limit_selection

            # Create apply handler using partial application
            def on_apply_click_wrapper(
                metadata,
                category_column,
                colormap,
                update_globals_callback,
            ):
                def on_apply_click(b):
                    result = _handle_apply_click(
                        metadata_table_widget,
                        sample_metadata_df,
                        available_columns,
                        complete_data_scan,
                        join_column,
                        metadata,
                        category_column,
                        debug_output,
                    )
                    if result is not None and update_globals_callback is not None:
                        with debug_output:
                            print("🔄 Updating plot with new biosamples...")

                            # Extract the new data
                            new_full_merged_data = result["full_merged_data"]
                            x_col = result["x_col"]
                            y_col = result["y_col"]
                            x_label = result["x_label"]
                            y_label = result["y_label"]

                            # Add colormap to the new polars data if needed
                            if colormap is not None:
                                x_coords = (
                                    new_full_merged_data[x_col].to_numpy().ravel()
                                )
                                y_coords = (
                                    new_full_merged_data[y_col].to_numpy().ravel()
                                )
                                points = np.column_stack([x_coords, y_coords])
                                colormap_values = colormap(points)
                                new_full_merged_data = (
                                    new_full_merged_data.with_columns(
                                        pl.lit(colormap_values).alias("colormap")
                                    )
                                )

                            # Update all global variables - this will trigger plot refresh
                            update_globals_callback(
                                new_full_merged_data, x_col, y_col, x_label, y_label
                            )

                return on_apply_click

            apply_handler = on_apply_click_wrapper

            print("Added selection callback")
            metadata_table_widget.observe(selection_handler, names="selections")

        except Exception as e:
            print(f"Warning: Could not load sample metadata: {e}")
            sample_metadata_df = None
            metadata_table_widget = None

    return (
        sample_metadata_df,
        metadata_table_widget,
        datagrid,
        apply_button,
        status_message,
        debug_output,
        complete_data_scan,
        available_columns,
        selection_handler,
        apply_handler,
    )


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
    sample_metadata_file: str | None = None,
    sample_id_column: str | None = None,
    data_file: str | None = None,
) -> ScatterplotResult:
    """
    Create a JScatter scatterplot with two datasets and interactive selection.

    Note: This function uses Polars DataFrames for efficient data processing, but
    internally converts to Pandas DataFrames where needed for jscatter compatibility.

    Parameters:
    -----------
    x : pl.DataFrame
        Dataset for X-axis values
    y : pl.DataFrame
        Dataset for Y-axis values
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
    sample_metadata_file : str, optional
        Path to parquet file containing sample metadata to display as interactive table
    sample_id_column : str, optional
        Column name in sample metadata that contains sample IDs/names for matching
    data_file : str, optional
        Path to parquet file containing complete dataset with all biosample columns
        If not provided, metadata table and biosample selection will be disabled

    Returns:
    --------
    ScatterplotResult
        Named tuple containing:
        - scatter: The jscatter plot object
        - merged_data: The merged dataset used for plotting
        - container: The plot container widget
        - selection: Function that returns DataFrame of currently selected points
        - class_dropdown: The class filter dropdown widget
        - metadata_table: ITables widget for sample metadata (None if not provided)
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

    # Initialize metadata table and related widgets
    (
        sample_metadata_df,
        metadata_table_widget,
        datagrid,
        apply_button,
        status_message,
        debug_output,
        complete_data_scan,
        available_columns,
        selection_handler,
        apply_handler,
    ) = _initialize_metadata_table(
        sample_metadata_file, sample_id_column, data_file, join_column
    )

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
    def create_interpolated_colormap(
        n_categories: int,
    ) -> list[str]:
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
                mcolors.rgb2hex(
                    cmap(i / (n_categories - 1) if n_categories > 1 else 0)
                )
                for i in range(n_categories)
            ]

        return selected_colors

    # Generate colors for exact number of categories
    n_categories = len(unique_classes)
    class_color_list = create_interpolated_colormap(n_categories)

    # Create mapping of class to color for legend
    class_color_map = dict(zip(unique_classes, class_color_list))

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
        merged_data = full_merged_data.filter(
            pl.col(category_column) == default_category
        )

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
    x_col = x_cols[0]
    if x_col in merged_data.columns and x_col + "_y" in merged_data.columns:
        x_col = x_col  # Use the original column from x
    elif x_col + "_y" in merged_data.columns:
        # Column was renamed during join, but we want the one from x (no suffix in Polars join)
        pass

    y_col = y_cols[0]
    if y_col + "_y" in merged_data.columns:
        y_col = y_col + "_y"

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

    # Add colormap column to polars dataframe if needed
    if colormap is not None:
        # Extract coordinates for colormap calculation
        all_x_coords = full_merged_clean[x_col].to_numpy().ravel()
        all_y_coords = full_merged_clean[y_col].to_numpy().ravel()
        points = np.column_stack([all_x_coords, all_y_coords])
        colormap_values = colormap(points)

        # Add colormap column to the polars dataframe
        full_merged_clean = full_merged_clean.with_columns(
            pl.lit(colormap_values).alias("colormap")
        )

    # Set default labels using actual column names from the data
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col

    # Create selection tracking state container
    class SelectionState:
        def __init__(self):
            self.selected_ids = set()  # Will hold selected point IDs from join_column
            self.current_plot_data = (
                merged_data  # Track the currently displayed data for selection
            )
            self.current_full_data = full_merged_clean  # Track the current full merged data

    selection_state = SelectionState()

    # Initial plot creation
    plot_result = _create_plot_data(
        full_merged_clean,
        x_col,
        y_col,
        category_column,
        unique_classes,
        join_column,
        default_category,
    )
    if plot_result[0] is None:
        raise ValueError("No valid data points to plot")

    plot_df, axis_min, axis_max, selection_state.current_plot_data = plot_result

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
            [(line_min, line_min), (line_max, line_max)],  # type: ignore
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
                color_map="viridis",
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
                color_map=class_color_list,
            )

        # Set identical axis ranges using the scale parameter
        shared_range = (axis_min, axis_max)
        scatter.x("x_data", scale=shared_range)
        scatter.y("y_data", scale=shared_range)

        # Set axis labels using the axes() method
        # Try to configure axes with better spacing
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
        # Fallback: try basic version with minimal plot_df
        basic_plot_df = pd.DataFrame(
            {"x_data": plot_df["x_data"], "y_data": plot_df["y_data"]}
        )
        scatter = jscatter.Scatter(data=basic_plot_df, x="x_data", y="y_data")

    # Set up selection callback for lasso selection
    def on_selection_change(change):
        """Callback for when selection changes in the scatter plot."""
        # Get the new selection indices
        selected_indices = change.get("new", [])
        if selected_indices is not None and len(selected_indices) > 0:
            # Convert to list if it's a numpy array
            if hasattr(selected_indices, "tolist"):
                selected_indices = selected_indices.tolist()
            # Convert indices to IDs using the current plot data
            selected_rows = selection_state.current_plot_data[selected_indices]
            selected_ids = set(selected_rows[join_column].to_list())
            selection_state.selected_ids = selected_ids
            print(f"Selected {len(selected_indices)} points")
        else:
            # No selection - empty set
            selection_state.selected_ids = set()
            print("No points selected")

    scatter.widget.observe(on_selection_change, names=["selection"])

    # Create class filter dropdown
    class_dropdown = widgets.Dropdown(
        options=available_classes,
        value=default_category,
        description="Class:",
        disabled=False,
    )

    def update_plot(change, data_df, x_column, y_column):
        """Update the plot when class filter changes"""

        selected_class = change["new"]
        print(f"Filtering by class: {selected_class}")

        # Create new plot with filtered data
        plot_result = _create_plot_data(
            data_df, x_column, y_column, category_column, unique_classes, join_column, selected_class
        )

        if plot_result[0] is None:
            return

        new_plot_df, _, _, new_current_data = plot_result

        # Update the selection state with new filtered data
        selection_state.current_plot_data = new_current_data

        # Use scatter.data() now that we're using list-based color mapping
        # Disable animation and keep existing scales
        scatter.data(new_plot_df, reset_scales=False, animate=False)

        # Keep the original axis ranges static - don't recalculate based on filtered data
        # This prevents zooming/panning animation between class changes
        original_shared_range = (axis_min, axis_max)
        scatter.x("x_data", scale=original_shared_range)
        scatter.y("y_data", scale=original_shared_range)

        print(f"Updated plot in-place for class: {selected_class}")

    # Connect dropdown callback with wrapper to pass current data
    def on_class_change(change):
        update_plot(change, full_merged_clean, x_col, y_col)

    class_dropdown.observe(on_class_change, names="value")

    # Set up apply button callback if metadata table is available
    if apply_handler is not None and apply_button is not None:
        # Create callback to update global variables
        def update_globals_callback(
            new_full_merged_data, new_x_col, new_y_col, new_x_label, new_y_label
        ):
            nonlocal full_merged_clean, x_col, y_col, x_label, y_label

            # Update global state
            full_merged_clean = new_full_merged_data
            x_col = new_x_col
            y_col = new_y_col
            x_label = new_x_label
            y_label = new_y_label

            # Update selection state with new data
            selection_state.current_full_data = new_full_merged_data

            # Trigger plot refresh with new data
            update_plot({"new": class_dropdown.value}, full_merged_clean, x_col, y_col)

            # Update the scatter plot axis labels
            scatter.axes(axes=True, grid=True, labels=[new_x_label, new_y_label])

        on_apply_click = apply_handler(
            metadata,
            category_column,
            colormap,
            update_globals_callback,
        )
        apply_button.on_click(on_apply_click)

    # Create layout with plot and dropdown
    plot_widget = scatter.show()

    # Add some right padding to the plot to give axis labels more space
    if hasattr(plot_widget, "layout"):
        plot_widget.layout.padding = "0 20px 0 0"  # Right padding for axis labels

    # Create legend for class colors (only when using class-based coloring)
    legend_widget = None
    if colormap is None:  # Only show legend for class-based coloring
        legend_widget = create_class_legend(unique_classes, class_color_map)

    # Create filter controls (just dropdown, legend will be separate)
    filter_controls = HBox([class_dropdown])

    # Create plot area with legend positioned to the right
    if legend_widget:
        # Add spacing between plot and legend
        legend_widget.layout.margin = "0 0 0 20px"  # Left margin for spacing from plot
        # Create a container with plot on left and legend on right
        plot_with_legend = HBox([plot_widget, legend_widget])
        # Ensure proper spacing in the HBox
        plot_with_legend.layout.align_items = "flex-start"
    else:
        plot_with_legend = plot_widget

    # Create the main container components
    container_elements = []

    if title:
        # Create container widget with user-specified title
        title_widget = widgets.HTML(f"<h3>{title}</h3>")
        container_elements.append(title_widget)

    container_elements.extend([filter_controls, plot_with_legend])

    # Add metadata table below the plot if provided
    if metadata_table_widget and apply_button and status_message:
        metadata_title = widgets.HTML("<h4>Sample Metadata</h4>")

        # Create horizontal box for button and message (using already created widgets)
        apply_controls = HBox([apply_button, status_message])
        apply_controls.layout.align_items = "center"
        apply_controls.layout.margin = "10px 0"

        container_elements.extend(
            [metadata_title, apply_controls, metadata_table_widget]
        )

        # Add debug output if it exists
        if debug_output:
            debug_title = widgets.HTML("<h5>Debug Output</h5>")
            container_elements.extend([debug_title, debug_output])

    container = VBox(container_elements)

    # Make the main container responsive
    if hasattr(container, "layout"):
        container.layout.width = "100%"
        container.layout.padding = "18px"  # Add padding to container

    # Display the container
    display(container)

    # Create selection handle that returns currently selected data
    def get_selection():
        """Return DataFrame of currently selected points with full metadata, empty if none selected"""
        if len(selection_state.selected_ids) > 0:
            # Filter the current full data by selected IDs to get complete metadata
            return selection_state.current_full_data.filter(pl.col(join_column).is_in(list(selection_state.selected_ids)))
        else:
            return pl.DataFrame()

    return ScatterplotResult(
        scatter=scatter,
        merged_data=merged_data,
        container=container,
        selection=get_selection,
        class_dropdown=class_dropdown,
        metadata_table=metadata_table_widget,
        datagrid=datagrid,
    )
