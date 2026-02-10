import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    from scatterplot_helpers import (
        merge_datasets,
        get_numeric_column,
        create_interpolated_colormap,
    )
    return (
        mo, pl, go,
        merge_datasets, get_numeric_column, create_interpolated_colormap,
    )


@app.cell
def config():
    DATAFILE = "/Users/conrad/Projects/vedatonuryilmaz/simplex-scatter/comb_healthy_scatac_tcga_encode.parquet"
    JOIN_COLUMN = "cCRE"
    CATEGORY_COLUMN = "class"
    X_BIOSAMPLE = "LGGx-TCGA-DU-6395-02A-11-A644-42-X037-S06"
    Y_BIOSAMPLE = "LGGx-TCGA-F6-A8O3-01A-31-A617-42-X013-S07"
    return DATAFILE, JOIN_COLUMN, CATEGORY_COLUMN, X_BIOSAMPLE, Y_BIOSAMPLE


@app.cell
def load_data(DATAFILE, JOIN_COLUMN, CATEGORY_COLUMN, X_BIOSAMPLE, Y_BIOSAMPLE, pl):
    x = pl.read_parquet(DATAFILE, columns=[JOIN_COLUMN, X_BIOSAMPLE])
    y = pl.read_parquet(DATAFILE, columns=[JOIN_COLUMN, Y_BIOSAMPLE])
    metadata = pl.read_parquet(
        DATAFILE, columns=[JOIN_COLUMN, CATEGORY_COLUMN]
    )
    return x, y, metadata


@app.cell
def prepare(x, y, metadata, JOIN_COLUMN, CATEGORY_COLUMN, pl, merge_datasets, get_numeric_column, create_interpolated_colormap):
    all_data = merge_datasets(x, y, metadata, JOIN_COLUMN)

    x_col = get_numeric_column(x, JOIN_COLUMN)
    y_col = get_numeric_column(y, JOIN_COLUMN)

    all_data = all_data.filter(
        (~pl.col(x_col).is_nan()) & (~pl.col(y_col).is_nan())
    )

    unique_categories = sorted(all_data[CATEGORY_COLUMN].unique().to_list())
    category_color_list = create_interpolated_colormap(len(unique_categories))
    category_color_map = dict(zip(unique_categories, category_color_list))

    return all_data, x_col, y_col, unique_categories, category_color_map


@app.cell
def scatter_plot(all_data, x_col, y_col, CATEGORY_COLUMN, unique_categories, category_color_map, go, pl, mo):
    fig = go.Figure()

    for cat in unique_categories:
        subset = all_data.filter(pl.col(CATEGORY_COLUMN) == cat)
        fig.add_trace(
            go.Scattergl(
                x=subset[x_col].to_numpy(),
                y=subset[y_col].to_numpy(),
                mode="markers",
                marker=dict(size=2, opacity=0.5, color=category_color_map[cat]),
                name=cat,
            )
        )

    _ = fig.update_layout(
        width=700,
        height=700,
        xaxis=dict(range=[-12, 9], scaleanchor="y", scaleratio=1, title=x_col),
        yaxis=dict(range=[-12, 9], title=y_col),
        dragmode="select",
    )
    _ = fig.add_shape(
        type="line", x0=-500, y0=-500, x1=500, y1=500,
        line=dict(color="lightgray", width=1),
    )

    chart = mo.ui.plotly(fig)
    return (chart,)


@app.cell
def show(chart, all_data, mo):
    mo.vstack([
        mo.md(f"## Plotly POC — comparison scatter plot (WebGL)\n\n*{len(all_data):,} points*"),
        chart,
    ])


@app.cell
def selection_info(chart, mo):
    sel = chart.value
    count = len(sel) if sel is not None and len(sel) > 0 else 0
    mo.md(f"**Selected points:** {count}")


if __name__ == "__main__":
    app.run()
