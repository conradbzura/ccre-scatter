import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors

    return mo, np, pl, go, LinearSegmentedColormap, mcolors


@app.cell
def helpers(np, pl, LinearSegmentedColormap, mcolors):
    _BASE_COLORS = [
        "#06DA93",  # CA
        "#00B0F0",  # CA-CTCF
        "#ffaaaa",  # CA-H3K4me3
        "#be28e5",  # CA-TF
        "#FF0000",  # PLS
        "#d876ec",  # TF
        "#FFCD00",  # dELS
        "#FFA700",  # pELS
    ]

    def create_interpolated_colormap(n_categories: int) -> list[str]:
        """Return *n_categories* hex colours interpolated from _BASE_COLORS."""
        if n_categories <= len(_BASE_COLORS):
            return _BASE_COLORS[:n_categories]

        cmap = LinearSegmentedColormap.from_list("custom", _BASE_COLORS, N=n_categories)
        return [
            mcolors.rgb2hex(cmap(i / (n_categories - 1) if n_categories > 1 else 0))
            for i in range(n_categories)
        ]

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
        """Bin points into an n_bins x n_bins grid and return one row per non-empty bin."""
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        x_edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
        y_edges = np.linspace(y_range[0], y_range[1], n_bins + 1)

        xi = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
        yi = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)

        binned = pl.DataFrame(
            {
                "xi": xi.astype(np.int32),
                "yi": yi.astype(np.int32),
                "cat": df[category_col].to_list(),
                "id": df[id_col].to_list(),
            }
        )

        majority = (
            binned.group_by(["xi", "yi", "cat"])
            .len()
            .sort(["xi", "yi", "cat"])
            .sort(["xi", "yi", "len"], descending=[False, False, True])
            .group_by(["xi", "yi"], maintain_order=True)
            .first()
            .select(["xi", "yi", pl.col("cat").alias("category")])
        )

        agg = binned.group_by(["xi", "yi"]).agg(
            [
                pl.len().alias("count"),
                pl.col("id").alias("ids"),
            ]
        )

        result = agg.join(majority, on=["xi", "yi"], how="inner")

        x_step = (x_range[1] - x_range[0]) / n_bins
        y_step = (y_range[1] - y_range[0]) / n_bins
        result = result.with_columns(
            [
                (
                    pl.col("xi").cast(pl.Float64) * x_step + x_range[0] + x_step / 2
                ).alias("cx"),
                (
                    pl.col("yi").cast(pl.Float64) * y_step + y_range[0] + y_step / 2
                ).alias("cy"),
            ]
        )

        return result.select(["cx", "cy", "count", "category", "ids"])

    return (create_interpolated_colormap, bin_scatter)


# ── Configuration ────────────────────────────────────────────────────────
@app.cell
def config():
    DATAFILE_UINT8 = "https://users.abdenlab.org/conrad/itcr/ccre-scatter/ccre_zscore_uint8.parquet"
    TSV_FILE = "https://users.abdenlab.org/conrad/itcr/ccre-scatter/cCREs_N1_N2_zoonomia_replicate_447_wgroups.tsv"
    SAMPLE_METADATA_FILE = (
        "https://users.abdenlab.org/conrad/itcr/ccre-scatter/sample_metadata.pq"
    )
    JOIN_COLUMN = "cCRE"
    CATEGORY_COLUMN = "class"
    INITIAL_X = "LGGx-TCGA-DU-6395-02A-11-A644-42-X037-S06"
    INITIAL_Y = "LGGx-TCGA-F6-A8O3-01A-31-A617-42-X013-S07"
    N_BINS = 256
    # Mapping constants from preprocessing (global z-score range → uint8)
    ZSCORE_MIN = -13.92728
    ZSCORE_MAX = 10.77115249633789
    ZSCORE_STEP = (ZSCORE_MAX - ZSCORE_MIN) / N_BINS
    VALIDATE_COLUMNS = False
    return (
        DATAFILE_UINT8,
        TSV_FILE,
        SAMPLE_METADATA_FILE,
        JOIN_COLUMN,
        CATEGORY_COLUMN,
        INITIAL_X,
        INITIAL_Y,
        N_BINS,
        ZSCORE_MIN,
        ZSCORE_MAX,
        ZSCORE_STEP,
        VALIDATE_COLUMNS,
    )


# ── Biosample state ─────────────────────────────────────────────────────
@app.cell
def biosample_state(mo, INITIAL_X, INITIAL_Y):
    get_biosamples, set_biosamples = mo.state((INITIAL_X, INITIAL_Y))
    return get_biosamples, set_biosamples


# ── Selection state (one per direction) ────────────────────────────────
@app.cell
def selection_state(mo):
    get_comp_to_n, set_comp_to_n = mo.state(set())
    get_n_to_comp, set_n_to_comp = mo.state(set())
    return get_comp_to_n, set_comp_to_n, get_n_to_comp, set_n_to_comp


# ── Load metadata + N-data (runs once) ──────────────────────────────────
@app.cell
def load_static(DATAFILE_UINT8, SAMPLE_METADATA_FILE, VALIDATE_COLUMNS, pl):
    sample_metadata_df = pl.read_parquet(SAMPLE_METADATA_FILE)
    available_columns = (
        pl.scan_parquet(DATAFILE_UINT8).collect_schema().names()
        if VALIDATE_COLUMNS
        else None
    )
    return sample_metadata_df, available_columns


# ── Merge comparison data (reactive to biosample selection) ─────────────
@app.cell
def merge_comparison(
    get_biosamples,
    DATAFILE_UINT8,
    CATEGORY_COLUMN,
    ZSCORE_MIN,
    ZSCORE_STEP,
    pl,
    create_interpolated_colormap,
):
    x_col_name, y_col_name = get_biosamples()

    comp_binned = (
        pl.scan_parquet(DATAFILE_UINT8)
        .select([CATEGORY_COLUMN, x_col_name, y_col_name])
        .filter(pl.col(x_col_name).is_not_null() & pl.col(y_col_name).is_not_null())
        .rename({x_col_name: "xi", y_col_name: "yi"})
        .cast({"xi": pl.Int32, "yi": pl.Int32})
        .group_by([CATEGORY_COLUMN, "xi", "yi"])
        .agg(pl.len().alias("count"))
        .with_columns(
            [
                (
                    pl.col("xi").cast(pl.Float64) * ZSCORE_STEP
                    + ZSCORE_MIN
                    + ZSCORE_STEP / 2
                ).alias("cx"),
                (
                    pl.col("yi").cast(pl.Float64) * ZSCORE_STEP
                    + ZSCORE_MIN
                    + ZSCORE_STEP / 2
                ).alias("cy"),
            ]
        )
        .collect(engine="streaming")
    )
    unique_categories = sorted(comp_binned[CATEGORY_COLUMN].unique().to_list())
    category_color_map = dict(
        zip(unique_categories, create_interpolated_colormap(len(unique_categories)))
    )
    return comp_binned, x_col_name, y_col_name, unique_categories, category_color_map


# ── Load + bin N-data ────────────────────────────────────────────────────
@app.cell
def load_n_data(TSV_FILE, CATEGORY_COLUMN, JOIN_COLUMN, N_BINS, pl):
    n_x_col = "N2"
    n_y_col = "N1"
    _x_range, _y_range = (0.0, 450.0), (0.0, 450.0)
    _x_step = (_x_range[1] - _x_range[0]) / N_BINS
    _y_step = (_y_range[1] - _y_range[0]) / N_BINS

    n_all_data = (
        pl.read_csv(TSV_FILE, separator="\t")
        .rename({"cCRE_type": CATEGORY_COLUMN})
        .with_columns(
            [
                ((pl.col(n_x_col).cast(pl.Float64) - _x_range[0]) / _x_step)
                .floor()
                .cast(pl.Int32)
                .clip(0, N_BINS - 1)
                .alias("xi"),
                ((pl.col(n_y_col).cast(pl.Float64) - _y_range[0]) / _y_step)
                .floor()
                .cast(pl.Int32)
                .clip(0, N_BINS - 1)
                .alias("yi"),
            ]
        )
        .group_by([CATEGORY_COLUMN, "xi", "yi"])
        .agg([pl.len().alias("count"), pl.col(JOIN_COLUMN).alias("ids")])
        .with_columns(
            [
                (
                    pl.col("xi").cast(pl.Float64) * _x_step + _x_range[0] + _x_step / 2
                ).alias("cx"),
                (
                    pl.col("yi").cast(pl.Float64) * _y_step + _y_range[0] + _y_step / 2
                ).alias("cy"),
            ]
        )
    )
    return (n_all_data,)


# ── Category dropdown ───────────────────────────────────────────────────
@app.cell
def category_filter(unique_categories, mo):
    category_dropdown = mo.ui.dropdown(
        options=["All"] + unique_categories,
        value="All",
        label="Class",
    )
    return (category_dropdown,)


# ── Filter both datasets by selected category ──────────────────────────
@app.cell
def filter_data(category_dropdown, comp_binned, n_all_data, CATEGORY_COLUMN, pl):
    _selected = category_dropdown.value

    if _selected == "All":
        comp_filtered = comp_binned
        n_filtered = n_all_data
    else:
        comp_filtered = comp_binned.filter(pl.col(CATEGORY_COLUMN) == _selected)
        n_filtered = n_all_data.filter(pl.col(CATEGORY_COLUMN) == _selected)

    return comp_filtered, n_filtered


# ── Comparison scatter plot ─────────────────────────────────────────────
@app.cell
def comparison_plot(
    comp_filtered,
    x_col_name,
    y_col_name,
    CATEGORY_COLUMN,
    unique_categories,
    category_color_map,
    get_n_to_comp,
    DATAFILE_UINT8,
    JOIN_COLUMN,
    ZSCORE_MIN,
    ZSCORE_MAX,
    ZSCORE_STEP,
    np,
    pl,
    go,
    mo,
):
    # Filter pre-binned data per category
    _cat_bins = []
    for _cat in unique_categories:
        _cat_binned = comp_filtered.filter(pl.col(CATEGORY_COLUMN) == _cat)
        if len(_cat_binned) == 0:
            continue
        _cat_bins.append((_cat, _cat_binned))

    # Global size scaling across all categories
    _all_counts = np.concatenate([b["count"].to_numpy() for _, b in _cat_bins])
    _all_log = np.log1p(_all_counts)
    _lmin, _lmax = _all_log.min(), _all_log.max()

    _fig = go.Figure()
    comp_bins: list[tuple[int, int]] = []
    _offset = 0
    for _cat, _cat_binned in _cat_bins:
        _n = len(_cat_binned)
        _cat_log = _all_log[_offset : _offset + _n]
        _offset += _n
        if _lmax > _lmin:
            _cat_sizes = 2 + (_cat_log - _lmin) / (_lmax - _lmin) * 5
        else:
            _cat_sizes = np.full_like(_cat_log, 2.0)
        comp_bins.extend(zip(_cat_binned["xi"].to_list(), _cat_binned["yi"].to_list()))
        _fig.add_trace(
            go.Scattergl(
                x=_cat_binned["cx"].to_numpy(),
                y=_cat_binned["cy"].to_numpy(),
                mode="markers",
                marker=dict(
                    size=_cat_sizes,
                    opacity=0.4,
                    color=category_color_map[_cat],
                    line=dict(width=0),
                ),
                name=_cat,
            )
        )

    _ = _fig.update_layout(
        width=600,
        height=600,
        xaxis=dict(
            range=[ZSCORE_MIN, ZSCORE_MAX],
            scaleanchor="y",
            scaleratio=1,
            title=x_col_name,
        ),
        yaxis=dict(range=[ZSCORE_MIN, ZSCORE_MAX], title=y_col_name),
        dragmode="select",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        title="Comparison Plot",
    )
    _ = _fig.add_shape(
        type="line",
        x0=-500,
        y0=-500,
        x1=500,
        y1=500,
        line=dict(color="lightgray", width=1),
    )

    # Highlight bins containing cCREs selected from N-plot (re-scan parquet)
    _n_ids = get_n_to_comp()
    if _n_ids:
        _hl_binned = (
            pl.scan_parquet(DATAFILE_UINT8)
            .select([JOIN_COLUMN, x_col_name, y_col_name])
            .filter(
                pl.col(x_col_name).is_not_null()
                & pl.col(y_col_name).is_not_null()
                & pl.col(JOIN_COLUMN).is_in(list(_n_ids))
            )
            .rename({x_col_name: "xi", y_col_name: "yi"})
            .cast({"xi": pl.Int32, "yi": pl.Int32})
            .group_by(["xi", "yi"])
            .agg(pl.len().alias("hl_count"))
            .with_columns(
                [
                    (
                        pl.col("xi").cast(pl.Float64) * ZSCORE_STEP
                        + ZSCORE_MIN
                        + ZSCORE_STEP / 2
                    ).alias("cx"),
                    (
                        pl.col("yi").cast(pl.Float64) * ZSCORE_STEP
                        + ZSCORE_MIN
                        + ZSCORE_STEP / 2
                    ).alias("cy"),
                ]
            )
            .collect(engine="streaming")
        )
        if len(_hl_binned) > 0:
            _hl_counts = _hl_binned["hl_count"].to_numpy()
            _hl_log = np.log1p(_hl_counts)
            _hl_lmin, _hl_lmax = _hl_log.min(), _hl_log.max()
            if _hl_lmax > _hl_lmin:
                _hl_sizes = 2 + (_hl_log - _hl_lmin) / (_hl_lmax - _hl_lmin) * 5
            else:
                _hl_sizes = np.full_like(_hl_log, 2.0)
            _fig.add_trace(
                go.Scattergl(
                    x=_hl_binned["cx"].to_numpy(),
                    y=_hl_binned["cy"].to_numpy(),
                    mode="markers",
                    marker=dict(
                        size=_hl_sizes, color="red", opacity=0.4, line=dict(width=0)
                    ),
                    name="Selected",
                    showlegend=False,
                )
            )

    comp_chart = mo.ui.plotly(_fig)
    return comp_chart, comp_bins


# ── Resolve selection from comparison chart → cCRE IDs ─────────────────
@app.cell
def resolve_comp_selection(
    comp_chart,
    comp_bins,
    set_comp_to_n,
    set_n_to_comp,
    DATAFILE_UINT8,
    JOIN_COLUMN,
    x_col_name,
    y_col_name,
    pl,
):
    _selected_bins = set()
    _indices = comp_chart.indices
    if _indices is not None and len(_indices) > 0:
        for _i in _indices:
            if 0 <= _i < len(comp_bins):
                _selected_bins.add(comp_bins[_i])
    if _selected_bins:
        _bin_df = pl.DataFrame(
            {
                "xi": pl.Series([b[0] for b in _selected_bins], dtype=pl.Int32),
                "yi": pl.Series([b[1] for b in _selected_bins], dtype=pl.Int32),
            }
        )
        _ids = (
            pl.scan_parquet(DATAFILE_UINT8)
            .select([JOIN_COLUMN, x_col_name, y_col_name])
            .filter(
                pl.col(x_col_name).is_not_null()
                & pl.col(y_col_name).is_not_null()
            )
            .rename({x_col_name: "xi", y_col_name: "yi"})
            .cast({"xi": pl.Int32, "yi": pl.Int32})
            .join(_bin_df.lazy(), on=["xi", "yi"], how="inner")
            .select(JOIN_COLUMN)
            .collect(engine="streaming")[JOIN_COLUMN]
            .to_list()
        )
        set_comp_to_n(set(_ids))
        set_n_to_comp(set())


# ── N-data scatter plot (highlights selection from comparison) ──────────
@app.cell
def n_data_plot(
    n_filtered,
    CATEGORY_COLUMN,
    unique_categories,
    category_color_map,
    get_comp_to_n,
    np,
    pl,
    go,
    mo,
):
    # Filter pre-binned data per category
    _cat_bins = []
    for _cat in unique_categories:
        _cat_binned = n_filtered.filter(pl.col(CATEGORY_COLUMN) == _cat)
        if len(_cat_binned) == 0:
            continue
        _cat_bins.append((_cat, _cat_binned))

    # Global size scaling across all categories
    _all_counts = np.concatenate([b["count"].to_numpy() for _, b in _cat_bins])
    _all_log = np.log1p(_all_counts)
    _lmin, _lmax = _all_log.min(), _all_log.max()

    _fig = go.Figure()
    n_ids: list[list[str]] = []
    _offset = 0
    for _cat, _cat_binned in _cat_bins:
        _n = len(_cat_binned)
        _cat_log = _all_log[_offset : _offset + _n]
        _offset += _n
        if _lmax > _lmin:
            _cat_sizes = 2 + (_cat_log - _lmin) / (_lmax - _lmin) * 5
        else:
            _cat_sizes = np.full_like(_cat_log, 2.0)
        n_ids.extend(_cat_binned["ids"].to_list())
        _fig.add_trace(
            go.Scattergl(
                x=_cat_binned["cx"].to_numpy(),
                y=_cat_binned["cy"].to_numpy(),
                mode="markers",
                marker=dict(
                    size=_cat_sizes,
                    opacity=0.4,
                    color=category_color_map[_cat],
                    line=dict(width=0),
                ),
                name=_cat,
            )
        )

    _ = _fig.update_layout(
        width=600,
        height=600,
        xaxis=dict(range=[0, 450], scaleanchor="y", scaleratio=1, title="N2"),
        yaxis=dict(range=[0, 450], title="N1"),
        dragmode="select",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        title="N Plot",
    )

    _comp_ids = get_comp_to_n()
    if _comp_ids:
        _hl_binned = (
            n_filtered.explode("ids")
            .filter(pl.col("ids").is_in(list(_comp_ids)))
            .group_by(["xi", "yi", "cx", "cy"])
            .agg(pl.len().alias("hl_count"))
        )
        if len(_hl_binned) > 0:
            _hl_counts = _hl_binned["hl_count"].to_numpy()
            _hl_log = np.log1p(_hl_counts)
            _hl_lmin, _hl_lmax = _hl_log.min(), _hl_log.max()
            if _hl_lmax > _hl_lmin:
                _hl_sizes = 2 + (_hl_log - _hl_lmin) / (_hl_lmax - _hl_lmin) * 5
            else:
                _hl_sizes = np.full_like(_hl_log, 2.0)
            _fig.add_trace(
                go.Scattergl(
                    x=_hl_binned["cx"].to_numpy(),
                    y=_hl_binned["cy"].to_numpy(),
                    mode="markers",
                    marker=dict(
                        size=_hl_sizes, color="red", opacity=0.4, line=dict(width=0)
                    ),
                    name="Selected",
                    showlegend=False,
                )
            )

    n_chart = mo.ui.plotly(_fig)
    return n_chart, n_ids


# ── Resolve selection from N chart → cCRE IDs ─────────────────────────
@app.cell
def resolve_n_selection(n_chart, n_ids, set_n_to_comp, set_comp_to_n):
    _selected = set()
    _indices = n_chart.indices
    if _indices is not None and len(_indices) > 0:
        for _i in _indices:
            if 0 <= _i < len(n_ids):
                _selected.update(n_ids[_i])
    if _selected:
        set_n_to_comp(_selected)
        set_comp_to_n(set())


# ── Legend ───────────────────────────────────────────────────────────────
@app.cell
def legend(unique_categories, category_color_map, mo):
    _rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<div style="width:12px;height:12px;background:{category_color_map[_cat]};'
        f'border:1px solid #ccc;margin-right:8px;border-radius:2px;flex-shrink:0;"></div>'
        f'<span style="font-size:12px;white-space:nowrap;">{_cat}</span></div>'
        for _cat in unique_categories
    )
    legend_html = mo.Html(
        f'<div style="display:flex;flex-direction:column;">'
        f'<strong style="margin-bottom:4px;">cCRE Class Legend</strong>'
        f"{_rows}</div>"
    )
    return (legend_html,)


# ── Layout: plots + legend + dropdown ───────────────────────────────────
@app.cell
def layout(category_dropdown, comp_chart, n_chart, legend_html, mo):
    _row = mo.hstack(
        [comp_chart, n_chart, legend_html],
        align="start",
        gap=1,
    )
    mo.vstack([category_dropdown, _row])


# ── Selection info + export ────────────────────────────────────────────
@app.cell
def selection_summary(get_comp_to_n, get_n_to_comp, mo):
    _comp = get_comp_to_n()
    _n = get_n_to_comp()
    _all_ids = sorted(_comp | _n)
    _count = len(_all_ids)

    if _count > 0:
        _csv = "\n".join(["cCRE"] + _all_ids)
        _download = mo.download(
            data=_csv.encode(),
            filename="selected_ccres.csv",
            mimetype="text/csv",
            label=f"Export {_count} cCREs",
        )
        mo.hstack([mo.md(f"**Selected cCREs:** {_count}"), _download], align="center", gap=1)
    else:
        mo.md("**Selected cCREs:** 0")


# ── Metadata table ──────────────────────────────────────────────────────
@app.cell
def metadata_table(sample_metadata_df, mo):
    table = mo.ui.table(
        sample_metadata_df,
        selection="multi",
        page_size=20,
        label="Sample Metadata — select exactly 2 rows, then click Apply",
    )
    return (table,)


# ── Apply biosample selection ───────────────────────────────────────────
@app.cell
def apply_section(table, set_biosamples, available_columns, mo):
    _rows = table.value

    _n = len(_rows) if _rows is not None else 0
    _ok = _n == 2

    if _ok and "biosample" in _rows.columns:
        _bs = _rows["biosample"].to_list()
        _valid = (
            all(b in available_columns for b in _bs)
            if available_columns is not None
            else True
        )
    else:
        _bs = []
        _valid = False

    if _n == 0:
        _msg = mo.callout(
            mo.md("Select 2 samples from the table below to compare"), kind="neutral"
        )
    elif _n == 1:
        _msg = mo.callout(mo.md("Select one more sample"), kind="warn")
    elif _ok and _valid:
        _msg = mo.callout(
            mo.md(f"Ready to apply: **{_bs[0]}** vs **{_bs[1]}**"), kind="success"
        )
    elif _ok and not _valid:
        _msg = mo.callout(
            mo.md("Selected samples not found in data file columns"), kind="danger"
        )
    else:
        _msg = mo.callout(
            mo.md(f"Too many samples selected ({_n}) — select exactly 2"), kind="danger"
        )

    _btn = mo.ui.button(
        label="Apply Selection",
        on_click=lambda _: set_biosamples(tuple(_bs)) if _ok and _valid else None,
        disabled=not (_ok and _valid),
        kind="success" if (_ok and _valid) else "neutral",
    )

    mo.vstack([_msg, table, _btn])


if __name__ == "__main__":
    app.run()
