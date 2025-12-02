import pandas as pd
import plotly.express as px
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'supervised_classification'


def plot_scatter():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    outpath = base_dir / VIGNETTE_DIR / 'scatterplot.html'
    title = "Energy variables showing different built_age type"
    fig = scatter_onecat(df, 'built_age', title)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(outpath)


def scatter_onecat(df, cat_column, title):
    # Ensure the categorical column exists
    if cat_column not in df.columns:
        raise ValueError(f"Column '{cat_column}' not found in dataframe")
    # Select numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in dataframe to plot.")
    # Create an index for x-axis (just row index)
    df = df.copy()
    df['__index__'] = range(len(df))
    # Melt numeric columns to long format
    long_df = df.melt(
        id_vars=['__index__', cat_column],
        value_vars=numeric_cols,
        var_name='variable',
        value_name='value'
    )
    # Build scatter with facet per variable
    fig = px.scatter(
        long_df,
        x='__index__',
        y='value',
        color=cat_column,
        facet_col='variable',
        facet_col_wrap=3,
        title=title,
        labels={'__index__': 'Observation index', 'value': 'Value'}
    )
    # Make layout a bit nicer
    fig.update_layout(
        legend_title=cat_column,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def get_frequencies(df, cat_column):
    return df[cat_column].value_counts()


def get_grouped_stats(df, cat_column):
    numeric_cols = df.select_dtypes(include='number').columns
    grouped_stats = df.groupby(cat_column)[numeric_cols].describe()
    grouped_stats.columns = ['{}_{}'.format(var, stat) for var, stat in grouped_stats.columns]
    return grouped_stats.transpose()


def get_summary_stats():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'energy.csv')
    cat_column = 'built_age'
    frequencies = get_frequencies(df, cat_column)
    outpath_f = base_dir / VIGNETTE_DIR / 'frequencies.csv'
    frequencies.to_csv(outpath_f)
    summary_stats = get_grouped_stats(df, cat_column)
    outpath_s = base_dir / VIGNETTE_DIR / 'grouped_stats.csv'
    summary_stats.to_csv(outpath_s)
