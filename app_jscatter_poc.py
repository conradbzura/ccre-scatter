import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import polars as pl
    import jscatter
    from scatterplot_helpers import (
        load_tsv_for_scatterplot,
        merge_datasets,
        get_numeric_column,
        create_plot_data,
        create_interpolated_colormap,
    )

    return (
        create_interpolated_colormap,
        create_plot_data,
        get_numeric_column,
        jscatter,
        merge_datasets,
        mo,
        pl,
    )


@app.cell
def config():
    DATAFILE = "/Users/conrad/Projects/vedatonuryilmaz/simplex-scatter/comb_healthy_scatac_tcga_encode.parquet"
    TSV_FILE = "cCREs_N1_N2_zoonomia_replicate_447_wgroups.tsv"
    JOIN_COLUMN = "cCRE"
    CATEGORY_COLUMN = "class"
    X_BIOSAMPLE = "LGGx-TCGA-DU-6395-02A-11-A644-42-X037-S06"
    Y_BIOSAMPLE = "LGGx-TCGA-F6-A8O3-01A-31-A617-42-X013-S07"
    return CATEGORY_COLUMN, DATAFILE, JOIN_COLUMN, X_BIOSAMPLE, Y_BIOSAMPLE


@app.cell
def load_data(
    CATEGORY_COLUMN,
    DATAFILE,
    JOIN_COLUMN,
    X_BIOSAMPLE,
    Y_BIOSAMPLE,
    pl,
):
    x = pl.read_parquet(DATAFILE, columns=[JOIN_COLUMN, X_BIOSAMPLE])
    y = pl.read_parquet(DATAFILE, columns=[JOIN_COLUMN, Y_BIOSAMPLE])
    metadata = pl.read_parquet(
        DATAFILE, columns=[JOIN_COLUMN, "rDHS", "chr", "start", "end", CATEGORY_COLUMN]
    )
    return metadata, x, y


@app.cell
def prepare(
    CATEGORY_COLUMN,
    JOIN_COLUMN,
    create_interpolated_colormap,
    create_plot_data,
    get_numeric_column,
    merge_datasets,
    metadata,
    pl,
    x,
    y,
):
    all_data = merge_datasets(x, y, metadata, JOIN_COLUMN)

    x_col = get_numeric_column(x, JOIN_COLUMN)
    y_col = get_numeric_column(y, JOIN_COLUMN)

    all_data = all_data.filter(
        (~pl.col(x_col).is_nan()) & (~pl.col(y_col).is_nan())
    )

    unique_categories = sorted(all_data[CATEGORY_COLUMN].unique().to_list())
    category_color_list = create_interpolated_colormap(len(unique_categories))

    plot_df, axis_min, axis_max = create_plot_data(
        all_data, x_col, y_col, CATEGORY_COLUMN, unique_categories, JOIN_COLUMN
    )
    return category_color_list, plot_df


@app.cell
def scatter_plot(CATEGORY_COLUMN, category_color_list, jscatter, mo, plot_df):
    scatter = jscatter.Scatter(
        data=plot_df,
        x="x_data",
        y="y_data",
        width=600,
        height=600,
        aspect_ratio=1.0,
        axes=True,
        axes_grid=True,
        color_by=CATEGORY_COLUMN,
        color_map=category_color_list,
        use_index=True,
    )
    scatter.x("x_data", scale=(-10, 10))
    scatter.y("y_data", scale=(-10, 10))

    widget = mo.ui.anywidget(scatter.widget)
    return (widget,)


@app.cell
def show(mo):
    mo.md("""
    ## jscatter POC — comparison scatter plot
    """)
    return


@app.cell
def selection_info(mo, widget):
    sel = widget.value
    mo.md(f"**Selected points:** {sel}")
    return


if __name__ == "__main__":
    app.run()
