from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def hcluster_analysis():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    outpath = base_dir / VIGNETTE_DIR / 'dendrogram.html'
    fig = _plot_dendrogram(df_scaled)
    fig.write_html(outpath)


def hierarchical_groups(height):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    linked = _fit_dendrogram(df_scaled)
    clusters = _cutree(linked, height)  # adjust this value based on dendrogram scale
    df_plot = _pca(df_scaled)
    df_plot['cluster'] = clusters.astype(str)  # convert to string for color grouping
    outpath = base_dir / VIGNETTE_DIR / 'hscatter.html'
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)


def _fit_dendrogram(df):
    Z = linkage(df if not isinstance(df, pd.DataFrame) else df.values, method="ward")
    return Z


def _plot_dendrogram(df):
    """Return a Plotly dendrogram figure for the given data (DataFrame or ndarray)."""
    data = df.values if isinstance(df, pd.DataFrame) else df
    fig = ff.create_dendrogram(
        data,
        linkagefun=lambda x: linkage(x, method="ward")
    )
    fig.update_layout(title="Interactive Hierarchical Clustering Dendrogram")
    return fig


def _cutree(tree, height):
    """Cut the tree at that height and return a DataFrame with one column 'cluster'."""
    labels = fcluster(tree, t=height, criterion="distance")
    return pd.DataFrame({"cluster": labels})


def _pca(df):
    """Return a DataFrame with the first two PCA components named 'PC1' and 'PC2'."""
    p = PCA(n_components=2, random_state=0)
    comps = p.fit_transform(df if not isinstance(df, pd.DataFrame) else df.values)
    return pd.DataFrame(comps, columns=["PC1", "PC2"])


def _scatter_clusters(df):
    """Return a Plotly scatter (PC1 vs PC2) colored by 'cluster'."""
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="PCA Scatter Plot Colored by Cluster Labels"
    )
    return fig
