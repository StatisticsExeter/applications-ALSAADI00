from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import pandas as pd
import joblib
from course.utils import find_project_root
from pathlib import Path
from sklearn.linear_model import LogisticRegression


def fit_classifier(X_train_path, y_train_path, model_path, classifier):
    # 1) Normalize paths (safer across OSes)
    X_train_path = Path(X_train_path)
    y_train_path = Path(y_train_path)
    model_path   = Path(model_path)
    # 2) Read CSVs
    X_train = pd.read_csv(X_train_path)
    y_df    = pd.read_csv(y_train_path)
    y_train = y_train = y_df['built_age'] if 'built_age' in y_df.columns else y_df.squeeze()
    classifier.fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_path)
    return classifier


def fit_lda():
    # 1) Get the project root as a Path
    base_dir = Path(find_project_root())
    # 2) Build training
    X_train_path = base_dir / 'data_cache' / 'energy_X_train.csv'
    y_train_path = base_dir / 'data_cache' / 'energy_y_train.csv'
    model_path = base_dir / 'data_cache' / 'models' / 'lda_model.joblib'
    # 3) Create the classifier
    classifier = LinearDiscriminantAnalysis()
    # 4) Train + save
    fit_classifier(X_train_path, y_train_path, model_path, classifier)


def fit_qda():
    # 1) Get the project root as a Path
    base_dir = Path(find_project_root())
    # 2) Build training paths
    X_train_path = base_dir / 'data_cache' / 'energy_X_train.csv'
    y_train_path = base_dir / 'data_cache' / 'energy_y_train.csv'
    model_path = base_dir / 'data_cache' / 'models' / 'qda_model.joblib'
    # 3) Create the classifier
    classifier = QuadraticDiscriminantAnalysis()
    # 4) Train + save
    fit_classifier(X_train_path, y_train_path, model_path, classifier)

def fit_logreg():
    base_dir = find_project_root()
    X_train_path = base_dir / "data_cache" / "energy_X_train.csv"
    y_train_path = base_dir / "data_cache" / "energy_y_train.csv"
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)["built_age"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    model_path = base_dir / "data_cache" / "models" / "logreg_model.joblib"
    joblib.dump(model, model_path)
