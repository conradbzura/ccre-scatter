"""Pure data-processing helpers for ccre-scatter.

No ipywidgets, jscatter, or IPython dependencies — only Polars, NumPy,
pandas, scikit-learn, and matplotlib (for colormap generation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors


SCALE = (-10, 10)

# Default base color palette used for category coloring
BASE_COLORS = [
    "#06DA93",  # CA
    "#00B0F0",  # CA-CTCF
    "#ffaaaa",  # CA-H3K4me3
    "#be28e5",  # CA-TF
    "#FF0000",  # PLS
    "#d876ec",  # TF
    "#FFCD00",  # dELS
    "#FFA700",  # pELS
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tsv_for_scatterplot(
    tsv_file: str,
    x_column: str = "N2",
    y_column: str = "N1",
    join_column: str = "cCRE",
    category_column: str = "cCRE_type",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load a TSV file and split into (x_data, y_data, metadata) DataFrames."""
    df = pl.read_csv(tsv_file, separator="\t")

    required_cols = [x_column, y_column, join_column, category_column]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in TSV: {missing}")

    x_data = df.select([join_column, x_column])
    y_data = df.select([join_column, y_column])

    metadata_cols = [col for col in df.columns if col not in [x_column, y_column]]
    metadata = df.select(metadata_cols)

    return x_data, y_data, metadata


# ---------------------------------------------------------------------------
# Density estimators (return callables that accept an (N, 2) array)
# ---------------------------------------------------------------------------

def kde(bandwidth: float = 1.0):
    def calculate_kde_density(points):
        kd = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(points)
        return np.exp(kd.score_samples(points))
    return calculate_kde_density


def knn(k: int = 100):
    def calculate_knn_density(points):
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        return 1 / (distances[:, 1:].mean(axis=1) + 1e-10)
    return calculate_knn_density


def radius(radius: float = 1.0):
    def calculate_radius_density(points):
        tree = KDTree(points)
        counts = tree.query_radius(points, r=radius, count_only=True)
        return np.array(counts, dtype=float)
    return calculate_radius_density


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def create_interpolated_colormap(n_categories: int) -> list[str]:
    """Return *n_categories* hex colours interpolated from BASE_COLORS."""
    if n_categories <= len(BASE_COLORS):
        return BASE_COLORS[:n_categories]

    cmap = LinearSegmentedColormap.from_list("custom", BASE_COLORS, N=n_categories)
    return [
        mcolors.rgb2hex(cmap(i / (n_categories - 1) if n_categories > 1 else 0))
        for i in range(n_categories)
    ]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def merge_datasets(
    x: pl.DataFrame,
    y: pl.DataFrame,
    metadata: pl.DataFrame,
    join_column: str,
) -> pl.DataFrame:
    """Inner-join x, y, and metadata on *join_column*."""
    return (
        x.join(y, on=join_column, how="inner", suffix="_y")
        .join(metadata, on=join_column, how="inner")
    )


def get_numeric_column(df: pl.DataFrame, join_column: str) -> str:
    """Return the single numeric column in *df* (excluding *join_column*).

    Raises ValueError when there is not exactly one such column.
    """
    cols = [c for c in df.columns if c != join_column and df[c].dtype.is_numeric()]
    if len(cols) != 1:
        raise ValueError(
            f"Expected exactly 1 numeric column (excluding '{join_column}'), "
            f"found {len(cols)}: {cols}"
        )
    return cols[0]


def bin_scatter(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    category_col: str,
    id_col: str,
    n_bins: int = 200,
    x_range: tuple[float, float] = (-12.0, 9.0),
    y_range: tuple[float, float] = (-12.0, 9.0),
) -> pl.DataFrame:
    """Bin points into an n_bins x n_bins grid and return one row per non-empty bin.

    Returns a DataFrame with columns:
        cx, cy   – bin centre coordinates (float)
        count    – number of points in the bin (uint32)
        category – majority category in the bin (str)
        ids      – list of original IDs in the bin (list[str])
    """
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    x_edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    y_edges = np.linspace(y_range[0], y_range[1], n_bins + 1)

    xi = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)

    binned = pl.DataFrame({
        "xi": xi.astype(np.int32),
        "yi": yi.astype(np.int32),
        "cat": df[category_col].to_list(),
        "id": df[id_col].to_list(),
    })

    # Majority category per bin: count per (xi, yi, cat), then pick the cat
    # with the highest count (ties broken alphabetically via stable sort on cat asc).
    majority = (
        binned
        .group_by(["xi", "yi", "cat"])
        .len()
        .sort(["xi", "yi", "cat"])           # stable: alphabetical cat for ties
        .sort(["xi", "yi", "len"], descending=[False, False, True])
        .group_by(["xi", "yi"], maintain_order=True)
        .first()
        .select(["xi", "yi", pl.col("cat").alias("category")])
    )

    # Aggregates per bin: total count + list of IDs
    agg = binned.group_by(["xi", "yi"]).agg([
        pl.len().alias("count"),
        pl.col("id").alias("ids"),
    ])

    result = agg.join(majority, on=["xi", "yi"], how="inner")

    # Compute bin centres
    x_step = (x_range[1] - x_range[0]) / n_bins
    y_step = (y_range[1] - y_range[0]) / n_bins
    result = result.with_columns([
        (pl.col("xi").cast(pl.Float64) * x_step + x_range[0] + x_step / 2).alias("cx"),
        (pl.col("yi").cast(pl.Float64) * y_step + y_range[0] + y_step / 2).alias("cy"),
    ])

    return result.select(["cx", "cy", "count", "category", "ids"])


def create_plot_data(
    all_data: pl.DataFrame,
    x_column: str,
    y_column: str,
    category_column: str,
    unique_categories: list[str],
    join_column: str,
    selected_category: str = "All",
) -> tuple[pd.DataFrame, float, float] | tuple[None, None, None]:
    """Filter *all_data* by category and convert to a pandas DataFrame ready for plotting.

    Returns (plot_df, axis_min, axis_max) or (None, None, None) when the
    filter yields zero rows.
    """
    if selected_category == "All":
        filtered_data = all_data
    else:
        filtered_data = all_data.filter(pl.col(category_column) == selected_category)

    if len(filtered_data) == 0:
        return None, None, None

    x_coords = filtered_data[x_column].to_numpy().ravel()
    y_coords = filtered_data[y_column].to_numpy().ravel()

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    overall_min = min(x_min, y_min)
    overall_max = max(x_max, y_max)

    padding = (overall_max - overall_min) * 0.05
    axis_min = overall_min - padding
    axis_max = overall_max + padding

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
