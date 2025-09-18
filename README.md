# cCRE Scatter Plot

Interactive scatter plots for analyzing candidate cis-regulatory elements (cCREs) with JScatter, featuring dynamic class filtering, density-based coloring, and diagonal reference lines.

## Features

- **Interactive Scatter Plots**: Built on JScatter for high-performance visualization
- **Dynamic Class Filtering**: Switch between categories without plot flickering
- **Flexible Coloring**: Categorical coloring with interpolated palettes or density-based coloring (KDE, KNN, radius)
- **Diagonal Reference Line**: Automatically included diagonal line with gridline styling
- **Persistent Legend**: Legend remains visible during class filtering
- **Selection Support**: Interactive point selection with callback functions
- **Polars Integration**: Efficient data processing with Polars DataFrames

## Installation

### Using uv (recommended)

```bash
# Clone the repo
git clone https://github.com/conradbzura/ccre-scatter.git
cd ccre-scatter

# Create a virtual environment and install dependencies
uv sync
```

### Using pip

```bash
# Clone the repo
git clone https://github.com/conradbzura/ccre-scatter.git
cd ccre-scatter

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install .
```

## API Reference

### `scatterplot(x, y, metadata, join_column, category_column="class", ...)`

Creates an interactive scatter plot with categorical coloring and filtering.

**Parameters:**

- `x` (pl.DataFrame): Dataset for X-axis values
- `y` (pl.DataFrame): Dataset for Y-axis values
- `metadata` (pl.DataFrame): Metadata with categorical information
- `join_column` (str): Column name to join datasets on
- `category_column` (str, default="class"): Column for categorical coloring/filtering
- `x_label` (str, optional): Custom X-axis label
- `y_label` (str, optional): Custom Y-axis label
- `title` (str, optional): Custom plot title
- `colormap` (Callable, optional): Density function for coloring (kde(), knn(), radius())
- `default_category` (str, default="All"): Initial class filter

**Returns:**

`ScatterplotResult` named tuple with:
- `scatter`: JScatter plot object
- `merged_data`: Combined dataset (Polars DataFrame)
- `container`: Plot container widget
- `selection`: Function returning selected points
- `class_dropdown`: Class filter dropdown widget

### Density Functions

#### `kde(bandwidth=1.0)`
Kernel density estimation coloring.

#### `knn(k=100)`
K-nearest neighbors density coloring.

#### `radius(radius=1.0)`
Points-within-radius density coloring.

## Example Usage

See [example_usage.ipynb](example_usage.ipynb) for detailed examples including:

- Basic categorical scatter plots
- Density-based coloring
- Custom category columns
- Programmatic interaction with plots
- Data selection and filtering

## Color Palette

The function automatically generates an interpolated colormap based on the number of unique categories:

- **≤8 categories**: Uses base colors directly
- **>8 categories**: Creates smooth interpolation across the palette

Base colors:
- Purple (#8f2be7)
- Pink (#fb4fd9)
- Red (#e9162d)
- Orange (#f28200)
- Yellow (#ffdb28)
- Green (#1fb819)
- Cyan (#00e1da)
- Blue (#007bd8)

## Requirements

- Python ≥3.9
- jupyter-scatter
- polars
- pandas
- numpy
- scikit-learn
- matplotlib
- ipywidgets

## License

MIT License