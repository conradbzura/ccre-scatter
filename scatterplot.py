from typing import Any, Callable, NamedTuple

import ipywidgets as widgets
import jscatter
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from ipywidgets import HBox, VBox
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors
from ipydatagrid import DataGrid

SCALE = (-10, 10)


class ScatterplotResult(NamedTuple):
    """Named tuple for scatterplot function return value."""

    scatter: Any
    plot_data: pd.DataFrame
    container: Any
    selection: Callable[[], pl.DataFrame]
    category_dropdown: Any
    metadata_table: Any
    datagrid: Any


class SelectionState:
    def __init__(self, plot_data):
        self.selected_ids = set()
        self.plot_data = plot_data


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


def _create_category_legend(unique_categories, category_color_map):
    """
    Create a legend widget showing category colors.

    Parameters:
    -----------
    unique_categories : list
        List of unique category names
    category_color_map : dict
        Mapping of category names to colors

    Returns:
    --------
    ipywidgets.VBox
        Legend widget
    """
    legend_items = []
    for cls in unique_categories:
        color = category_color_map[cls]
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
        status_message.value = f"<span style='color: #dc3545; font-style: italic;'> Too many rows selected ({total_rows_selected}) - select exactly 2 rows</span>"
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
            "<span style='color: #28a745; font-style: italic;'> ✓ Ready to apply selection</span>"
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
            status_message.value = "<span style='color: #666; font-style: italic;'> Select 2 rows</span>"
        else:
            status_message.value = "<span style='color: #007bff; font-style: italic;'> Select one more row</span>"


def _create_plot_data(
    all_data,
    x_column,
    y_column,
    category_column,
    unique_categories,
    join_column,
    selected_category="All",
):
    """Create plot data for the selected category, returning pandas DataFrame and current plot data for selection."""
    if selected_category == "All":
        filtered_data = all_data
    else:
        filtered_data = all_data.filter(pl.col(category_column) == selected_category)

    if len(filtered_data) == 0:
        print(f"No valid data points for category: {selected_category}")
        return None, None, None, None

    # Extract coordinates for axis calculations
    x_coords = filtered_data[x_column].to_numpy().ravel()
    y_coords = filtered_data[y_column].to_numpy().ravel()

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
    category_data = filtered_data[category_column].to_numpy()
    join_data = filtered_data[join_column].to_numpy()
    plot_df = pd.DataFrame(
        {
            "x_data": x_coords,
            "y_data": y_coords,
            category_column: pd.Categorical(
                category_data, categories=unique_categories, ordered=True
            ),
            join_column: join_data,
        }
    ).set_index(join_column)

    return plot_df, axis_min, axis_max


def _handle_apply_click(
    metadata_table_widget,
    sample_metadata_df,
    available_columns,
    all_data_scan,
    join_column,
    metadata,
    debug_output,
):
    """Handle apply button click to update plot with selected samples"""
    with debug_output:
        print("=" * 50)
        print("🔥 Apply button clicked")

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
                    print("❌ 'biosample' column not found")
                    return None

                selected_biosamples.append(biosample_value)
                print(f"✅ Row {row_idx}: {biosample_value}")

        print(f"🎯 Selected biosamples: {selected_biosamples}")

        biosample1, biosample2, *extra = selected_biosamples
        if extra:
            print(
                f"❌ ERROR: Expected exactly 2 biosamples, got {len(selected_biosamples)}"
            )
            return None

        if biosample1 not in available_columns:
            print(f"❌ ERROR: {biosample1} not found in complete data columns")
            print(f"Available columns: {available_columns}")
            return None

        if biosample2 not in available_columns:
            print(f"❌ ERROR: {biosample2} not found in complete data columns")
            print(f"Available columns: {available_columns}")
            return None

        # Merge the new datasets with metadata
        print("🔗 Merging new datasets...")
        new_all_data = (
            all_data_scan.select([join_column, biosample1, biosample2])
            .collect()
            .join(metadata, on=join_column, how="inner")
        )

        if len(new_all_data) == 0:
            print("❌ ERROR: No matching records found between new datasets")
            return None

        print(f"📈 Using columns: X={biosample1}, Y={biosample2}")

        # Filter out NaN values
        print("🧹 Filtering NaN values...")
        new_all_data = new_all_data.filter(
            (~pl.col(biosample1).is_nan()) & (~pl.col(biosample2).is_nan())
        )
        print(f"   Clean data: {len(new_all_data)} rows")

        # Return the new data to replace the global state
        return {
            "all_data": new_all_data,
            "x_column": biosample1,
            "y_column": biosample2,
            "x_label": biosample1,
            "y_label": biosample2,
        }


def _initialize_metadata_table(
    sample_metadata_file, sample_id_column, data_file, join_column
):
    """Initialize metadata table widgets and data structures if all required parameters are provided.

    Returns:
        Tuple of (sample_metadata_df, metadata_table_widget, datagrid, apply_button,
                 status_message, debug_output, all_data_scan, available_columns,
                 selection_handler, apply_handler)
    """
    sample_metadata_df = None
    metadata_table_widget = None
    datagrid = []
    apply_button = None
    status_message = None
    debug_output = None
    all_data_scan = None
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
            available_columns = data_scan.collect_schema().names()

            if join_column not in available_columns:
                raise ValueError(f"Column '{join_column}' not found in data file")

            print(f"Data file has {len(available_columns)} columns available")

            # Store the lazy frame for later use - we'll only load specific columns when needed
            all_data_scan = data_scan

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
                value="<span style='color: #666; font-style: italic;'> Select 2 rows</span>"
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
            def on_click_wrapper(
                metadata,
                category_column,
                colormap,
                update_globals_callback,
            ):
                def on_click(b):
                    result = _handle_apply_click(
                        metadata_table_widget,
                        sample_metadata_df,
                        available_columns,
                        all_data_scan,
                        join_column,
                        metadata,
                        debug_output,
                    )
                    if result is not None and update_globals_callback is not None:
                        with debug_output:
                            print("🔄 Updating plot with new biosamples...")

                            # Extract the new data
                            new_all_data = result["all_data"]
                            x_column = result["x_column"]
                            y_column = result["y_column"]
                            x_label = result["x_label"]
                            y_label = result["y_label"]

                            # Add colormap to the new polars data if needed
                            if colormap is not None:
                                x_coords = new_all_data[x_column].to_numpy().ravel()
                                y_coords = new_all_data[y_column].to_numpy().ravel()
                                points = np.column_stack([x_coords, y_coords])
                                colormap_values = colormap(points)
                                new_all_data = new_all_data.with_columns(
                                    pl.lit(colormap_values).alias("colormap")
                                )

                            # Update all global variables - this will trigger plot refresh
                            update_globals_callback(
                                new_all_data, x_column, y_column, x_label, y_label
                            )

                return on_click

            apply_handler = on_click_wrapper

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
        all_data_scan,
        available_columns,
        selection_handler,
        apply_handler,
    )


def create_plot(
    plot_data,
    category_column,
    category_color_list,
):
    diagonal_line = jscatter.Line(
        [(-500, -500), (500, 500)],  # type: ignore
        line_color="#e0e0e0",
        line_width=1,
    )
    scatter = jscatter.Scatter(
        data=plot_data,
        x="x_data",
        y="y_data",
        width=500,
        height=500,
        aspect_ratio=1.0,
        axes=True,
        axes_grid=True,
        annotations=[diagonal_line],
        color_by=category_column,
        color_map=category_color_list,
        use_index=True,
    )
    scatter.x("x_data", scale=SCALE)
    scatter.y("y_data", scale=SCALE)
    return scatter


def init_plot(
    scatter,
    all_data,
    x_column,
    y_column,
    x_label,
    y_label,
    category_column,
    unique_categories,
    selected_category,
    join_column,
    category_color_list,
):
    """Update the plot when category filter changes"""

    print(f"Filtering by category: {selected_category}")

    # Create new plot with filtered data
    plot_result = _create_plot_data(
        all_data,
        x_column,
        y_column,
        category_column,
        unique_categories,
        join_column,
        selected_category,
    )

    if plot_result[0] is None:
        return

    new_plot_df, _, _ = plot_result

    if not scatter:
        scatter = create_plot(new_plot_df, category_column, category_color_list)
    else:
        scatter.data(new_plot_df, reset_scales=False, animate=False, use_index=True)
    scatter.axes(axes=True, grid=True, labels=[x_label, y_label])

    print(f"Updated plot in-place for category: {selected_category}")

    return scatter


def scatterplot(
    x: pl.DataFrame,
    y: pl.DataFrame,
    metadata: pl.DataFrame,
    join_column: str,
    category_column: str = "category",
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    colormap: Callable | None = None,
    default_category: str = "All",
    sample_metadata_file: str | None = None,
    sample_id_column: str | None = None,
    data_file: str | None = None,
    debug: bool = False,
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
    category_column : str, default "category"
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
        - None: Use category-based coloring (default)
    default_category : str, default "All"
        Default category to filter by on initial display. Use "All" to show all categories.
        A dropdown will allow changing the category filter.
    sample_metadata_file : str, optional
        Path to parquet file containing sample metadata to display as interactive table
    sample_id_column : str, optional
        Column name in sample metadata that contains sample IDs/names for matching
    data_file : str, optional
        Path to parquet file containing complete dataset with all biosample columns
        If not provided, metadata table and biosample selection will be disabled
    debug : bool, default False
        Whether to show the debug output window below the metadata table

    Returns:
    --------
    ScatterplotResult
        Named tuple containing:
        - scatter: The jscatter plot object
        - plot_data: The merged dataset used for plotting
        - container: The plot container widget
        - selection: Function that returns DataFrame of currently selected points
        - category_dropdown: The category filter dropdown widget
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
        all_data_scan,
        available_columns,
        selection_handler,
        apply_handler,
    ) = _initialize_metadata_table(
        sample_metadata_file, sample_id_column, data_file, join_column
    )

    all_data = x.join(y, on=join_column, how="inner", suffix="_y").join(
        metadata, on=join_column, how="inner"
    )
    if len(all_data) == 0:
        raise ValueError("No matching records found between datasets")

    # Get unique categories for dropdown
    unique_categories = sorted(all_data[category_column].unique().to_list())
    available_categories = ["All"] + unique_categories

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
                mcolors.rgb2hex(cmap(i / (n_categories - 1) if n_categories > 1 else 0))
                for i in range(n_categories)
            ]

        return selected_colors

    # Generate colors for exact number of categories
    category_color_list = create_interpolated_colormap(len(unique_categories))

    # Create mapping of category to color for legend
    category_color_map = dict(zip(unique_categories, category_color_list))

    # Validate default_category
    if default_category not in available_categories:
        print(
            f"Warning: default_category '{default_category}' not found in data. Available categories: {available_categories}"
        )
        if available_categories:
            default_category = available_categories[0]
        else:
            raise ValueError("No category data available for filtering")

    # Prepare data for plotting
    # Assume we want to plot the first numeric column from each dataset
    # (excluding the join column)
    x_column, *extra_x = [
        c for c in x.columns if c != join_column and x[c].dtype.is_numeric()
    ]
    y_column, *extra_y = [
        c for c in y.columns if c != join_column and y[c].dtype.is_numeric()
    ]
    if extra_x:
        raise ValueError("Expected a single data column for x")
    if extra_y:
        raise ValueError("Expected a single data column for y")

    all_data = all_data.filter(
        (~pl.col(x_column).is_nan()) & (~pl.col(y_column).is_nan())
    )

    # Set default labels using actual column names from the data
    if x_label is None:
        x_label = x_column
    if y_label is None:
        y_label = y_column

    selection_state = SelectionState(all_data)

    scatter = init_plot(
        None,
        all_data,
        x_column,
        y_column,
        x_label,
        y_label,
        category_column,
        unique_categories,
        default_category,
        join_column,
        category_color_list,
    )

    # Workaround for jupyter-scatter Button widget bug
    # Add missing _dblclick_handler attribute to prevent AttributeError
    def fix_button_widgets(widget):
        """Recursively fix Button widgets missing _dblclick_handler attribute"""
        try:
            # Check if this widget needs the fix
            if (
                hasattr(widget, "_click_handler")
                and not hasattr(widget, "_dblclick_handler")
                and hasattr(widget, "__category__")
                and "Button" in str(widget.__category__)
            ):
                widget._dblclick_handler = None

            # Recursively check children
            if hasattr(widget, "children"):
                for child in widget.children:
                    fix_button_widgets(child)
        except Exception as e:
            pass  # Silently ignore errors in fix attempts

    try:
        if hasattr(scatter, "widget"):
            fix_button_widgets(scatter.widget)
    except Exception as e:
        print(f"Warning: Could not apply Button widget fix: {e}")

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
            selected_rows = selection_state.plot_data[selected_indices]
            selected_ids = selected_rows[join_column].to_list()
            selection_state.selected_ids = selected_ids
            print(f"Selected {len(selected_indices)} points")
        else:
            # No selection - empty set
            selection_state.selected_ids = set()
            print("No points selected")

    scatter.widget.observe(on_selection_change, names=["selection"])

    # Create category filter dropdown
    category_dropdown = widgets.Dropdown(
        options=available_categories,
        value=default_category,
        description="Class:",
        disabled=False,
    )

    # Connect dropdown callback with wrapper to pass current data
    def on_category_change(change):
        init_plot(
            scatter,
            all_data,
            x_column,
            y_column,
            x_label,
            y_label,
            category_column,
            unique_categories,
            change["new"],
            join_column,
            colormap,
        )

    category_dropdown.observe(on_category_change, names="value")

    # Set up apply button callback if metadata table is available
    if apply_handler is not None and apply_button is not None:
        # Create callback to update global variables
        def update_globals_callback(
            new_all_data, new_x_column, new_y_col, new_x_label, new_y_label
        ):
            # Update global state
            all_data = new_all_data
            x_column = new_x_column
            y_column = new_y_col
            x_label = new_x_label
            y_label = new_y_label

            # Update selection state with new data
            selection_state.plot_data = new_all_data

            # Trigger plot refresh with new data
            init_plot(
                scatter,
                all_data,
                x_column,
                y_column,
                x_label,
                y_label,
                category_column,
                unique_categories,
                category_dropdown.value,
                join_column,
                category_color_list,
            )

        on_click = apply_handler(
            metadata,
            category_column,
            colormap,
            update_globals_callback,
        )
        apply_button.on_click(on_click)

    # Create layout with plot and dropdown
    plot_widget = scatter.show()

    # Add some right padding to the plot to give axis labels more space
    if hasattr(plot_widget, "layout"):
        plot_widget.layout.padding = "0 20px 0 0"  # Right padding for axis labels

    # Create legend for category colors (only when using category-based coloring)
    legend_widget = None
    if colormap is None:  # Only show legend for category-based coloring
        legend_widget = _create_category_legend(unique_categories, category_color_map)

    # Create filter controls (just dropdown, legend will be separate)
    filter_controls = HBox([category_dropdown])

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

        # Add debug output if it exists and debug is True
        if debug and debug_output:
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
            return selection_state.plot_data.filter(
                pl.col(join_column).is_in(list(selection_state.selected_ids))
            )
        else:
            return pl.DataFrame()

    return ScatterplotResult(
        scatter=scatter,
        plot_data=None,
        container=container,
        selection=get_selection,
        category_dropdown=category_dropdown,
        metadata_table=metadata_table_widget,
        datagrid=datagrid,
    )
