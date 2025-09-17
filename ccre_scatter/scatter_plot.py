"""Utility functions for creating JScatter scatterplots with cCRE datasets."""

from typing import Optional, Dict, Any
import pandas as pd
import jscatter
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import VBox


def create_ccre_scatterplot(
    x: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    join_column: str = "cCRE",
    x_name: str = "Dataset A",
    y_name: str = "Dataset B",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    **scatter_kwargs: Any,
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
    merged_ab = pd.merge(x, y, on=join_column, how="inner", suffixes=("_a", "_b"))

    # Then merge with metadata
    merged_data = pd.merge(merged_ab, metadata, on=join_column, how="inner")

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
    if y_col + "_a" in merged_data.columns:
        y_col = y_col + "_a"

    x_col = y_cols[0]
    if x_col + "_b" in merged_data.columns:
        x_col = x_col + "_b"

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
        x_label = f"{y_name} ({x_col})"
    if y_label is None:
        y_label = f"{x_name} ({y_col})"

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

        # Add some basic styling
        styled_html = f"""
        <style>
        #metadata-table {{
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            max-height: 400px;
            overflow-y: auto;
            display: block;
        }}
        #metadata-table th, #metadata-table td {{
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        #metadata-table th {{
            background-color: #f2f2f2;
            position: sticky;
            top: 0;
        }}
        #metadata-table tr:hover {{
            background-color: #f5f5f5;
        }}
        </style>
        {table_title}
        <div style="max-height: 400px; overflow-y: auto;">
        {html_table}
        </div>
        """

        metadata_table.value = styled_html

    # Initialize metadata table
    update_metadata_table()

    # Debug information
    print(f"Data shapes: x_coords={x_coords.shape}, y_coords={y_coords.shape}")
    print(f"Data types: x_coords={x_coords.dtype}, y_coords={y_coords.dtype}")
    print(f"Sample x values: {x_coords[:5] if len(x_coords) > 0 else 'empty'}")
    print(f"Sample y values: {y_coords[:5] if len(y_coords) > 0 else 'empty'}")

    # Prepare scatter plot arguments - try without selection first
    scatter_args = {
        "x": x_coords,
        "y": y_coords,
        "x_label": x_label,
        "y_label": y_label,
        **scatter_kwargs,
    }

    # Create scatter plot using jscatter.Scatter (not jscatter.plot)
    try:
        # Prepare data for jscatter.Scatter - it expects a DataFrame
        plot_df = pd.DataFrame({
            'x_data': x_coords,
            'y_data': y_coords
        })

        # Create scatter plot using the correct API
        scatter = jscatter.Scatter(
            data=plot_df,
            x='x_data',
            y='y_data',
            x_label=x_label,
            y_label=y_label
        )

    except Exception as e:
        print(f"Error creating plot: {e}")
        # Fallback: try basic version
        plot_df = pd.DataFrame({'x_data': x_coords, 'y_data': y_coords})
        scatter = jscatter.Scatter(data=plot_df, x='x_data', y='y_data')

    # Set up selection callback for lasso selection
    def on_selection_change(change):
        """Callback for when selection changes in the scatter plot."""
        print(f"Selection change detected: {change}")  # Debug
        # Get the new selection indices
        selected_indices = change.get('new', [])
        print(f"Selected indices: {selected_indices}")  # Debug
        if selected_indices is not None and len(selected_indices) > 0:
            # Convert to list if it's a numpy array
            if hasattr(selected_indices, 'tolist'):
                selected_indices = selected_indices.tolist()
            update_metadata_table(selected_indices)
        else:
            # No selection - show all data
            update_metadata_table()

    # Connect selection callback for jupyter-scatter
    try:
        # Based on jupyter-scatter documentation, use scatter.widget.selection
        if hasattr(scatter, 'widget') and hasattr(scatter.widget, 'selection'):
            print("Connecting to scatter.widget.selection trait")
            scatter.widget.observe(on_selection_change, names=['selection'])
        elif hasattr(scatter, 'selection'):
            print("Connecting to scatter.selection trait")
            scatter.observe(on_selection_change, names=['selection'])
        else:
            # Fallback: check if scatter itself has the selection trait
            print(f"Available traits on scatter: {scatter.trait_names() if hasattr(scatter, 'trait_names') else 'no trait_names method'}")
            print("Could not find selection trait - selection updates may not work")

    except Exception as e:
        print(f"Warning: Could not connect selection callback: {e}")
        print("Lasso selection updates may not work automatically")

    # Create container widget - use scatter.show() for the plot
    container = VBox(
        [
            widgets.HTML(f"<h3>cCRE Scatter Plot: {x_name} vs {y_name}</h3>"),
            scatter.show(),
            metadata_table,
        ]
    )

    # Display the container
    display(container)

    return {
        "scatter": scatter,
        "metadata_table": metadata_table,
        "merged_data": merged_data,
        "container": container,
        "update_metadata_callback": update_metadata_table,
    }
