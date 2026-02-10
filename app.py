import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from scatterplot_helpers import (
        load_tsv_for_scatterplot,
        merge_datasets,
        get_numeric_column,
        create_interpolated_colormap,
        bin_scatter,
    )
    return (
        mo, np, pl, go,
        load_tsv_for_scatterplot, merge_datasets, get_numeric_column,
        create_interpolated_colormap, bin_scatter,
    )


# ── Configuration ────────────────────────────────────────────────────────
@app.cell
def config():
    DATAFILE = "/Users/conrad/Projects/vedatonuryilmaz/simplex-scatter/comb_healthy_scatac_tcga_encode.parquet"
    TSV_FILE = "cCREs_N1_N2_zoonomia_replicate_447_wgroups.tsv"
    SAMPLE_METADATA_FILE = "sample_metadata.pq"
    JOIN_COLUMN = "cCRE"
    CATEGORY_COLUMN = "class"
    INITIAL_X = "LGGx-TCGA-DU-6395-02A-11-A644-42-X037-S06"
    INITIAL_Y = "LGGx-TCGA-F6-A8O3-01A-31-A617-42-X013-S07"
    return (
        DATAFILE, TSV_FILE, SAMPLE_METADATA_FILE,
        JOIN_COLUMN, CATEGORY_COLUMN,
        INITIAL_X, INITIAL_Y,
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
def load_static(DATAFILE, TSV_FILE, SAMPLE_METADATA_FILE, JOIN_COLUMN, CATEGORY_COLUMN, pl, load_tsv_for_scatterplot):
    metadata = pl.read_parquet(
        DATAFILE, columns=[JOIN_COLUMN, "rDHS", "chr", "start", "end", CATEGORY_COLUMN]
    )

    n2_df, n1_df, _n_meta = load_tsv_for_scatterplot(
        TSV_FILE,
        x_column="N2", y_column="N1",
        join_column=JOIN_COLUMN, category_column="cCRE_type",
    )

    sample_metadata_df = pl.read_parquet(SAMPLE_METADATA_FILE)

    _scan = pl.scan_parquet(DATAFILE)
    available_columns = _scan.collect_schema().names()

    return metadata, n1_df, n2_df, sample_metadata_df, available_columns


# ── Merge comparison data (reactive to biosample selection) ─────────────
@app.cell
def merge_comparison(
    get_biosamples, DATAFILE, JOIN_COLUMN, CATEGORY_COLUMN, metadata, pl,
    merge_datasets, create_interpolated_colormap,
):
    x_col_name, y_col_name = get_biosamples()

    _x = pl.read_parquet(DATAFILE, columns=[JOIN_COLUMN, x_col_name])
    _y = pl.read_parquet(DATAFILE, columns=[JOIN_COLUMN, y_col_name])

    all_data = merge_datasets(_x, _y, metadata, JOIN_COLUMN)
    all_data = all_data.filter(
        (~pl.col(x_col_name).is_nan()) & (~pl.col(y_col_name).is_nan())
    )

    unique_categories = sorted(all_data[CATEGORY_COLUMN].unique().to_list())
    category_color_map = dict(
        zip(unique_categories, create_interpolated_colormap(len(unique_categories)))
    )

    return all_data, x_col_name, y_col_name, unique_categories, category_color_map


# ── Merge N-data ────────────────────────────────────────────────────────
@app.cell
def merge_n_data(n1_df, n2_df, metadata, JOIN_COLUMN, merge_datasets, get_numeric_column, pl):
    n_all_data = merge_datasets(n2_df, n1_df, metadata, JOIN_COLUMN)

    n_x_col = get_numeric_column(n2_df, JOIN_COLUMN)
    n_y_col = get_numeric_column(n1_df, JOIN_COLUMN)

    n_all_data = n_all_data.filter(
        (~pl.col(n_x_col).is_nan()) & (~pl.col(n_y_col).is_nan())
    )
    return n_all_data, n_x_col, n_y_col


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
def filter_data(category_dropdown, all_data, n_all_data, CATEGORY_COLUMN, pl):
    _selected = category_dropdown.value

    if _selected == "All":
        comp_filtered = all_data
        n_filtered = n_all_data
    else:
        comp_filtered = all_data.filter(pl.col(CATEGORY_COLUMN) == _selected)
        n_filtered = n_all_data.filter(pl.col(CATEGORY_COLUMN) == _selected)

    return comp_filtered, n_filtered


# ── Comparison scatter plot ─────────────────────────────────────────────
@app.cell
def comparison_plot(
    comp_filtered, x_col_name, y_col_name, CATEGORY_COLUMN, JOIN_COLUMN,
    unique_categories, category_color_map, get_n_to_comp,
    np, pl, go, mo, bin_scatter,
):
    # Bin each category independently, collect counts for global scaling
    _cat_bins = []
    for _cat in unique_categories:
        _cat_data = comp_filtered.filter(pl.col(CATEGORY_COLUMN) == _cat)
        if len(_cat_data) == 0:
            continue
        _cat_binned = bin_scatter(
            _cat_data, x_col_name, y_col_name,
            category_col=CATEGORY_COLUMN, id_col=JOIN_COLUMN,
            n_bins=200, x_range=(-12.0, 9.0), y_range=(-12.0, 9.0),
        )
        _cat_bins.append((_cat, _cat_binned))

    # Global size scaling across all categories
    _all_counts = np.concatenate([b["count"].to_numpy() for _, b in _cat_bins])
    _all_log = np.log1p(_all_counts)
    _lmin, _lmax = _all_log.min(), _all_log.max()

    _fig = go.Figure()
    comp_ids: list[list[str]] = []
    _offset = 0
    for _cat, _cat_binned in _cat_bins:
        _n = len(_cat_binned)
        _cat_log = _all_log[_offset:_offset + _n]
        _offset += _n
        if _lmax > _lmin:
            _cat_sizes = 2 + (_cat_log - _lmin) / (_lmax - _lmin) * 5
        else:
            _cat_sizes = np.full_like(_cat_log, 2.0)
        comp_ids.extend(_cat_binned["ids"].to_list())
        _fig.add_trace(
            go.Scattergl(
                x=_cat_binned["cx"].to_numpy(),
                y=_cat_binned["cy"].to_numpy(),
                mode="markers",
                marker=dict(size=_cat_sizes, opacity=0.4, color=category_color_map[_cat], line=dict(width=0)),
                name=_cat,
            )
        )

    _ = _fig.update_layout(
        width=600, height=600,
        xaxis=dict(range=[-12, 9], scaleanchor="y", scaleratio=1, title=x_col_name),
        yaxis=dict(range=[-12, 9], title=y_col_name),
        dragmode="select",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        title="Comparison Plot",
    )
    _ = _fig.add_shape(
        type="line", x0=-500, y0=-500, x1=500, y1=500,
        line=dict(color="lightgray", width=1),
    )

    _n_ids = get_n_to_comp()
    if _n_ids:
        _hl = comp_filtered.filter(pl.col(JOIN_COLUMN).is_in(list(_n_ids)))
        if len(_hl) > 0:
            _hl_binned = bin_scatter(
                _hl, x_col_name, y_col_name,
                category_col=CATEGORY_COLUMN, id_col=JOIN_COLUMN,
                n_bins=200, x_range=(-12.0, 9.0), y_range=(-12.0, 9.0),
            )
            _hl_counts = _hl_binned["count"].to_numpy()
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
                    marker=dict(size=_hl_sizes, color="red", opacity=0.4, line=dict(width=0)),
                    name="Selected",
                    showlegend=False,
                )
            )

    comp_chart = mo.ui.plotly(_fig)
    return comp_chart, comp_ids


# ── Resolve selection from comparison chart → cCRE IDs ─────────────────
@app.cell
def resolve_comp_selection(comp_chart, comp_ids, set_comp_to_n, set_n_to_comp):
    _selected = set()
    _indices = comp_chart.indices
    if _indices is not None and len(_indices) > 0:
        for _i in _indices:
            if 0 <= _i < len(comp_ids):
                _selected.update(comp_ids[_i])
    if _selected:
        set_comp_to_n(_selected)
        set_n_to_comp(set())


# ── N-data scatter plot (highlights selection from comparison) ──────────
@app.cell
def n_data_plot(
    n_filtered, n_x_col, n_y_col, CATEGORY_COLUMN, JOIN_COLUMN,
    unique_categories, category_color_map,
    get_comp_to_n, np, pl, go, mo, bin_scatter,
):
    # Bin each category independently, collect counts for global scaling
    _cat_bins = []
    for _cat in unique_categories:
        _cat_data = n_filtered.filter(pl.col(CATEGORY_COLUMN) == _cat)
        if len(_cat_data) == 0:
            continue
        _cat_binned = bin_scatter(
            _cat_data, n_x_col, n_y_col,
            category_col=CATEGORY_COLUMN, id_col=JOIN_COLUMN,
            n_bins=200, x_range=(0.0, 450.0), y_range=(0.0, 450.0),
        )
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
        _cat_log = _all_log[_offset:_offset + _n]
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
                marker=dict(size=_cat_sizes, opacity=0.4, color=category_color_map[_cat], line=dict(width=0)),
                name=_cat,
            )
        )

    _ = _fig.update_layout(
        width=600, height=600,
        xaxis=dict(range=[0, 450], scaleanchor="y", scaleratio=1, title="N2"),
        yaxis=dict(range=[0, 450], title="N1"),
        dragmode="select",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        title="N Plot",
    )

    _comp_ids = get_comp_to_n()
    if _comp_ids:
        _hl = n_filtered.filter(pl.col(JOIN_COLUMN).is_in(list(_comp_ids)))
        if len(_hl) > 0:
            _hl_binned = bin_scatter(
                _hl, n_x_col, n_y_col,
                category_col=CATEGORY_COLUMN, id_col=JOIN_COLUMN,
                n_bins=200, x_range=(0.0, 450.0), y_range=(0.0, 450.0),
            )
            _hl_counts = _hl_binned["count"].to_numpy()
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
                    marker=dict(size=_hl_sizes, color="red", opacity=0.4, line=dict(width=0)),
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
        f'{_rows}</div>'
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


# ── Selection info ──────────────────────────────────────────────────────
@app.cell
def selection_summary(get_comp_to_n, get_n_to_comp, mo):
    _comp = get_comp_to_n()
    _n = get_n_to_comp()
    _count = len(_comp) + len(_n)
    mo.md(f"**Selected cCREs:** {_count}")


# ── Metadata table ──────────────────────────────────────────────────────
@app.cell
def metadata_table(sample_metadata_df, mo):
    table = mo.ui.table(
        sample_metadata_df,
        selection="multi",
        page_size=20,
        label="Sample Metadata — select exactly 2 rows, then click Apply",
    )
    table
    return (table,)


# ── Apply biosample selection ───────────────────────────────────────────
@app.cell
def apply_section(table, set_biosamples, available_columns, mo):
    _rows = table.value

    _n = len(_rows) if _rows is not None else 0
    _ok = _n == 2

    if _ok and "biosample" in _rows.columns:
        _bs = _rows["biosample"].to_list()
        _valid = all(b in available_columns for b in _bs)
    else:
        _bs = []
        _valid = False

    if _n == 0:
        _msg = mo.md("*Select 2 rows from the table above*")
    elif _n == 1:
        _msg = mo.md("*Select one more row*")
    elif _ok and _valid:
        _msg = mo.md(f"Ready to apply: **{_bs[0]}** vs **{_bs[1]}**")
    elif _ok and not _valid:
        _msg = mo.md("**Error:** selected biosamples not found in data file columns")
    else:
        _msg = mo.md(f"*Too many rows selected ({_n}) — select exactly 2*")

    _btn = mo.ui.button(
        label="Apply Selection",
        on_click=lambda _: set_biosamples(tuple(_bs)) if _ok and _valid else None,
        disabled=not (_ok and _valid),
        kind="success" if (_ok and _valid) else "neutral",
    )

    mo.hstack([_btn, _msg], align="center", gap=1)


if __name__ == "__main__":
    app.run()
