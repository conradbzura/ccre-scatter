"""Replay app.py's data-loading and plot-building pipeline for memory profiling.

Usage:
    memray run -o profile_output.bin profile_memory.py
    memray flamegraph profile_output.bin -o memray_flamegraph.html
    memray table profile_output.bin
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scatterplot_helpers import create_interpolated_colormap

# ── Configuration (from app.py config cell) ──────────────────────────────
DATAFILE = "https://users.abdenlab.org/conrad/itcr/ccre-scatter/comb_healthy_scatac_tcga_encode.parquet"
TSV_FILE = "https://users.abdenlab.org/conrad/itcr/ccre-scatter/cCREs_N1_N2_zoonomia_replicate_447_wgroups.tsv"
SAMPLE_METADATA_FILE = "https://users.abdenlab.org/conrad/itcr/ccre-scatter/sample_metadata.pq"
JOIN_COLUMN = "cCRE"
CATEGORY_COLUMN = "class"
INITIAL_X = "LGGx-TCGA-DU-6395-02A-11-A644-42-X037-S06"
INITIAL_Y = "LGGx-TCGA-F6-A8O3-01A-31-A617-42-X013-S07"
N_BINS = 150


def main():
    # ── Step 1: load_static ──────────────────────────────────────────────
    print("Loading sample metadata parquet...")
    sample_metadata_df = pl.read_parquet(SAMPLE_METADATA_FILE)
    print(f"  sample_metadata: {sample_metadata_df.shape}")

    print("Skipping schema scan (VALIDATE_COLUMNS=False)...")

    # ── Step 2: load + bin N-data ───────────────────────────────────────
    print("Loading and binning TSV (N-data)...")
    n_x_col = "N2"
    n_y_col = "N1"
    _n_x_range, _n_y_range = (0.0, 450.0), (0.0, 450.0)
    _n_x_step = (_n_x_range[1] - _n_x_range[0]) / N_BINS
    _n_y_step = (_n_y_range[1] - _n_y_range[0]) / N_BINS
    n_all_data = (
        pl.read_csv(TSV_FILE, separator="\t")
        .rename({"cCRE_type": CATEGORY_COLUMN})
        .with_columns([
            ((pl.col(n_x_col).cast(pl.Float64) - _n_x_range[0]) / _n_x_step)
                .floor().cast(pl.Int32).clip(0, N_BINS - 1).alias("xi"),
            ((pl.col(n_y_col).cast(pl.Float64) - _n_y_range[0]) / _n_y_step)
                .floor().cast(pl.Int32).clip(0, N_BINS - 1).alias("yi"),
        ])
        .group_by([CATEGORY_COLUMN, "xi", "yi"])
        .agg([pl.len().alias("count"), pl.col(JOIN_COLUMN).alias("ids")])
        .with_columns([
            (pl.col("xi").cast(pl.Float64) * _n_x_step + _n_x_range[0] + _n_x_step / 2).alias("cx"),
            (pl.col("yi").cast(pl.Float64) * _n_y_step + _n_y_range[0] + _n_y_step / 2).alias("cy"),
        ])
    )
    print(f"  n_all_data (binned): {n_all_data.shape}")

    # ── Step 3: merge_comparison (streaming, no ids) ─────────────────────
    x_col_name, y_col_name = INITIAL_X, INITIAL_Y
    _x_range, _y_range = (-12.0, 9.0), (-12.0, 9.0)
    _x_step = (_x_range[1] - _x_range[0]) / N_BINS
    _y_step = (_y_range[1] - _y_range[0]) / N_BINS

    print(f"Streaming comparison binning: {x_col_name}, {y_col_name}...")
    comp_binned = (
        pl.scan_parquet(DATAFILE)
        .select([CATEGORY_COLUMN, x_col_name, y_col_name])
        .filter(pl.col(x_col_name).is_not_nan() & pl.col(y_col_name).is_not_nan())
        .with_columns([
            ((pl.col(x_col_name) - _x_range[0]) / _x_step)
                .floor().cast(pl.Int32).clip(0, N_BINS - 1).alias("xi"),
            ((pl.col(y_col_name) - _y_range[0]) / _y_step)
                .floor().cast(pl.Int32).clip(0, N_BINS - 1).alias("yi"),
        ])
        .group_by([CATEGORY_COLUMN, "xi", "yi"])
        .agg(pl.len().alias("count"))
        .with_columns([
            (pl.col("xi").cast(pl.Float64) * _x_step + _x_range[0] + _x_step / 2).alias("cx"),
            (pl.col("yi").cast(pl.Float64) * _y_step + _y_range[0] + _y_step / 2).alias("cy"),
        ])
        .collect(engine="streaming")
    )
    print(f"  comp_binned: {comp_binned.shape}")

    unique_categories = sorted(comp_binned[CATEGORY_COLUMN].unique().to_list())
    category_color_map = dict(
        zip(unique_categories, create_interpolated_colormap(len(unique_categories)))
    )
    print(f"  {len(unique_categories)} categories")

    # ── Step 4: filter pre-binned comparison data per category ───────────
    print("Filtering pre-binned comparison data per category...")
    cat_bins = []
    for cat in unique_categories:
        cat_binned = comp_binned.filter(pl.col(CATEGORY_COLUMN) == cat)
        if len(cat_binned) == 0:
            continue
        cat_bins.append((cat, cat_binned))
        print(f"  {cat}: {len(cat_binned)} bins")

    # ── Step 5: build comparison Plotly figure ───────────────────────────
    print("Building comparison Plotly figure...")
    all_counts = np.concatenate([b["count"].to_numpy() for _, b in cat_bins])
    all_log = np.log1p(all_counts)
    lmin, lmax = all_log.min(), all_log.max()

    comp_fig = go.Figure()
    offset = 0
    for cat, cat_binned in cat_bins:
        n = len(cat_binned)
        cat_log = all_log[offset:offset + n]
        offset += n
        if lmax > lmin:
            cat_sizes = 2 + (cat_log - lmin) / (lmax - lmin) * 5
        else:
            cat_sizes = np.full_like(cat_log, 2.0)
        comp_fig.add_trace(
            go.Scattergl(
                x=cat_binned["cx"].to_numpy(),
                y=cat_binned["cy"].to_numpy(),
                mode="markers",
                marker=dict(size=cat_sizes, opacity=0.4, color=category_color_map[cat], line=dict(width=0)),
                name=cat,
            )
        )

    _ = comp_fig.update_layout(
        width=600, height=600,
        xaxis=dict(range=[-12, 9], scaleanchor="y", scaleratio=1, title=x_col_name),
        yaxis=dict(range=[-12, 9], title=y_col_name),
        dragmode="select",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        title="Comparison Plot",
    )
    _ = comp_fig.add_shape(
        type="line", x0=-500, y0=-500, x1=500, y1=500,
        line=dict(color="lightgray", width=1),
    )
    print(f"  Comparison figure: {sum(len(b) for _, b in cat_bins)} total bins")

    # ── Step 6: filter pre-binned N-data per category ──────────────────
    print("Filtering pre-binned N-data per category...")
    n_cat_bins = []
    for cat in unique_categories:
        cat_binned = n_all_data.filter(pl.col(CATEGORY_COLUMN) == cat)
        if len(cat_binned) == 0:
            continue
        n_cat_bins.append((cat, cat_binned))
        print(f"  {cat}: {len(cat_binned)} bins")

    # ── Step 7: build N-plot Plotly figure ───────────────────────────────
    print("Building N-plot Plotly figure...")
    n_all_counts = np.concatenate([b["count"].to_numpy() for _, b in n_cat_bins])
    n_all_log = np.log1p(n_all_counts)
    n_lmin, n_lmax = n_all_log.min(), n_all_log.max()

    n_fig = go.Figure()
    offset = 0
    for cat, cat_binned in n_cat_bins:
        n = len(cat_binned)
        cat_log = n_all_log[offset:offset + n]
        offset += n
        if n_lmax > n_lmin:
            cat_sizes = 2 + (cat_log - n_lmin) / (n_lmax - n_lmin) * 5
        else:
            cat_sizes = np.full_like(cat_log, 2.0)
        n_fig.add_trace(
            go.Scattergl(
                x=cat_binned["cx"].to_numpy(),
                y=cat_binned["cy"].to_numpy(),
                mode="markers",
                marker=dict(size=cat_sizes, opacity=0.4, color=category_color_map[cat], line=dict(width=0)),
                name=cat,
            )
        )

    _ = n_fig.update_layout(
        width=600, height=600,
        xaxis=dict(range=[0, 450], scaleanchor="y", scaleratio=1, title="N2"),
        yaxis=dict(range=[0, 450], title="N1"),
        dragmode="select",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        title="N Plot",
    )
    print(f"  N-plot figure: {sum(len(b) for _, b in n_cat_bins)} total bins")

    print("Done. All data loaded and figures built.")


if __name__ == "__main__":
    main()
