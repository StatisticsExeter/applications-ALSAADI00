import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from course.utils import find_project_root
from course.unsupervised_classification.tree import _scatter_clusters, _pca

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def _dbscan(df, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(df)
    return model


def dbscan(eps=0.5, min_samples=5):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    model = _dbscan(df_scaled, eps, min_samples)
    clusters = model.labels_

    df_plot = _pca(df_scaled)
    df_plot['cluster'] = clusters.astype(str)
    df_plot.loc[df_plot['cluster'] == '-1', 'cluster'] = 'Noise'

    outpath = base_dir / VIGNETTE_DIR / 'dbscan_scatter.html'
    fig = _scatter_clusters(df_plot)
    fig.update_layout(
        title=f"DBSCAN Clusters (eps={eps}, min_samples={min_samples})"
    )
    fig.write_html(outpath)

