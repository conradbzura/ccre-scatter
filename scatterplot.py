from typing import Optional, Dict, Any
import pandas as pd
import jscatter
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import VBox, HBox


def scatterplot(
    x: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    join_column: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: str | None = None,
) -> Dict[str, Any]:
    """
    Create a JScatter scatterplot with two datasets and interactive metadata table.

    Parameters:
    -----------
    x : pd.DataFrame
        Dataset for Y-axis values
    y : pd.DataFrame
        Dataset for X-axis values
    metadata : pd.DataFrame
        Metadata describing the cCREs with columns: rDHS, cCRE, chrom, start, end, class
    join_column : str, default "cCRE"
        Column name to join the datasets on
    x_name : str, default "Dataset A"
        Name for dataset A (Y-axis)
    y_name : str, default "Dataset B"
        Name for dataset B (X-axis)
    x_label : str, optional
        Custom label for X-axis
    y_label : str, optional
        Custom label for Y-axis
    **scatter_kwargs : Any
        Additional keyword arguments passed to jscatter.plot()

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'scatter': The jscatter plot object
        - 'metadata_table': The interactive metadata table widget
        - 'merged_data': The merged dataset used for plotting
        - 'container': The complete widget container
    """

    # Validate inputs
    if join_column not in x.columns:
        raise ValueError(f"Column '{join_column}' not found in x")
    if join_column not in y.columns:
        raise ValueError(f"Column '{join_column}' not found in y")
    if join_column not in metadata.columns:
        raise ValueError(f"Column '{join_column}' not found in metadata")

    # Validate metadata columns
    required_metadata_cols = ["rDHS", "cCRE", "chrom", "start", "end", "class"]
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

    # Create metadata table widget
    metadata_display = merged_data[required_metadata_cols].copy()
    metadata_table = widgets.HTML()

    def update_metadata_table(selected_indices=None):
        """Update the metadata table based on selection."""
        if selected_indices is not None and len(selected_indices) > 0:
            display_data = metadata_display.iloc[selected_indices]
            table_title = (
                f"<h4>Selected Points Metadata ({len(selected_indices)} points)</h4>"
            )
        else:
            display_data = metadata_display
            table_title = (
                f"<h4>All Points Metadata ({len(metadata_display)} points)</h4>"
            )

        # Convert to HTML table
        html_table = display_data.to_html(
            index=False,
            classes="table table-striped table-hover",
            table_id="metadata-table",
        )

        # Add responsive styling with minimum width and flexible width
        styled_html = f"""
        <style>
        .metadata-container {{
            width: 99%;
            min-width: 400px;
            height: 500px;
            border: 1px solid #ddd;
            background-color: white;
            flex: 1;
            margin-left: 10px;
        }}
        .metadata-title {{
            padding: 10px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #ddd;
            margin: 0;
            font-size: 16px;
            font-weight: bold;
        }}
        .metadata-table-container {{
            height: 450px;
            overflow-y: auto;
            overflow-x: auto;
            padding: 0;
        }}
        #metadata-table {{
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            min-width: 380px;
            margin: 0;
        }}
        #metadata-table th, #metadata-table td {{
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #eee;
            font-size: 12px;
            white-space: nowrap;
            min-width: fit-content;
        }}
        #metadata-table th {{
            background-color: #f2f2f2;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        #metadata-table tr:hover {{
            background-color: #f5f5f5;
        }}
        /* Column-specific widths */
        #metadata-table th:nth-child(1), #metadata-table td:nth-child(1) {{ min-width: 80px; }} /* rDHS */
        #metadata-table th:nth-child(2), #metadata-table td:nth-child(2) {{ min-width: 80px; }} /* cCRE */
        #metadata-table th:nth-child(3), #metadata-table td:nth-child(3) {{ min-width: 60px; }} /* chrom */
        #metadata-table th:nth-child(4), #metadata-table td:nth-child(4) {{ min-width: 80px; }} /* start */
        #metadata-table th:nth-child(5), #metadata-table td:nth-child(5) {{ min-width: 80px; }} /* end */
        #metadata-table th:nth-child(6), #metadata-table td:nth-child(6) {{ min-width: 80px; }} /* class */
        </style>
        <div class="metadata-container">
            <div class="metadata-title">{table_title.replace("<h4>", "").replace("</h4>", "")}</div>
            <div class="metadata-table-container">
                {html_table}
            </div>
        </div>
        """

        metadata_table.value = styled_html

    # Initialize metadata table
    update_metadata_table()

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

    # Create scatter plot using jscatter.Scatter (not jscatter.plot)
    try:
        # Prepare data for jscatter.Scatter - it expects a DataFrame
        plot_df = pd.DataFrame({"x_data": x_coords, "y_data": y_coords})

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
        print(f"Selection change detected: {change}")  # Debug
        # Get the new selection indices
        selected_indices = change.get("new", [])
        print(f"Selected indices: {selected_indices}")  # Debug
        if selected_indices is not None and len(selected_indices) > 0:
            # Convert to list if it's a numpy array
            if hasattr(selected_indices, "tolist"):
                selected_indices = selected_indices.tolist()
            update_metadata_table(selected_indices)
        else:
            # No selection - show all data
            update_metadata_table()

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

    # Create flexible HBox layout
    plot_widget = scatter.show()
    # plot_widget.layout.width = '500px'  # Fixed width for plot (updated to match scatter dimensions)
    # plot_widget.layout.flex = '0 0 500px'  # Don't grow or shrink
    # plot_widget.layout.margin = '10px'  # Add margin around plot

    metadata_table.layout.flex = "1 1 400px"  # Grow to fill space, min 400px
    metadata_table.layout.min_width = "400px"
    # metadata_table.layout.margin = '10px'  # Add margin around table

    plot_and_table = HBox([plot_widget, metadata_table])

    # Make the HBox fill available width
    plot_and_table.layout.width = "100%"
    plot_and_table.layout.align_items = "flex-start"

    if title:
        # Create container widget with side-by-side layout
        title_widget = widgets.HTML(f"<h3>{title}</h3>")
        container = VBox([title_widget, plot_and_table])
    elif x_label and y_label and not title:
        title_widget = widgets.HTML(f"<h3>{y_label} vs. {x_label}</h3>")
        container = VBox([title_widget, plot_and_table])
    else:
        container = plot_and_table

    # Make the main container responsive
    container.layout.width = "100%"
    container.layout.padding = "18px"  # Add padding to container

    # Display the container
    display(container)

    return {
        "scatter": scatter,
        "metadata_table": metadata_table,
        "merged_data": merged_data,
        "container": container,
        "update_metadata_callback": update_metadata_table,
    }
