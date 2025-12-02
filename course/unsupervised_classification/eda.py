import pandas as pd
import plotly.express as px
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def plot_scatter():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    outpath = base_dir / VIGNETTE_DIR / 'scatterplot.html'
    title = "Crash types in each Local Authority"
    fig = _scatter(df, title)
    fig.write_html(outpath)


def _scatter(df, title):
    """Return a scatter-matrix Plotly figure over all numeric columns with a title."""
    fig = px.scatter_matrix(df)
    fig.update_layout(title=title)
    return fig
