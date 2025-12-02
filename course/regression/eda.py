from typing import Union
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path("data_cache") / "vignettes" / "regression"
VIGNETTE_DIR.mkdir(parents=True, exist_ok=True)


def _boxplot(df: pd.DataFrame,
             x_var: Union[str, None],
             y_var: str,
             title: str) -> Figure:
    """
    Given a DataFrame df, return a Plotly boxplot Figure of y_var,
    grouped by x_var if provided. The figure title must equal `title`.
    """
    fig = px.box(df, x=x_var, y=y_var, points="outliers") if x_var else px.box(df, y=y_var, points="outliers")
    fig.update_layout(title=title)
    return fig


def boxplot_age():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / "data_cache" / "la_energy.csv")
    fig = _boxplot(df, "age", "shortfall", "Shortfall by Age Category")
    fig.write_html(VIGNETTE_DIR / "boxplot_age.html")


def boxplot_rooms():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / "data_cache" / "la_energy.csv")
    fig = _boxplot(df, "n_rooms", "shortfall", "Shortfall by Number of rooms")
    fig.write_html(VIGNETTE_DIR / "boxplot_rooms.html")
