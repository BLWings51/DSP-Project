import os
import pickle
import joblib
import numpy as np
import pandas as pd
import shap
from tensorflow.keras.models import load_model, Model
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from utils import load_column_mapping, preprocess_single_transaction, KerasBinaryClassifier

class EnsembleProbaWrapper:
    """Picklable wrapper that always returns an (n,2) proba matrix."""
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        if hasattr(X, "data"):
            X_arr = np.asarray(X.data, dtype=np.float64)
        elif hasattr(X, "values"):
            X_arr = np.asarray(X.values, dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim > 2:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)
        out = self.model.predict_proba(X_arr)
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.shape[1] == 1:
            out = np.hstack([1 - out, out])
        return out

class PredictProbaWrapper:
    """Ensures SHAP always gets a clean 2-D numpy array back."""
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        if hasattr(X, "data"):
            X_arr = np.asarray(X.data, dtype=np.float64)
        elif hasattr(X, "values"):
            X_arr = np.asarray(X.values, dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim > 2:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X_arr)
            if p.ndim == 2 and p.shape[1] == 2:
                p = p[:, 1]
            else:
                p = p.reshape(-1)
        else:
            p = self.model.predict(X_arr, verbose=0).reshape(-1)
        return np.column_stack([1 - p, p])

def load_background_data(n_samples=100, n_features=None):
    """Returns a (n_samples × n_features) float64 array from banksimData.csv."""
    df = pd.read_csv('banksimData.csv')
    mapping = load_column_mapping()
    rows = []
    for _, row in df.iterrows():
        proc = preprocess_single_transaction(row, mapping)
        arr  = np.array(proc, dtype=np.float64).reshape(-1)
        rows.append(arr)
    data = np.vstack(rows)
    if n_features is not None and data.shape[1] != n_features:
        data = data[:, :n_features]
    if len(data) > n_samples:
        idx = np.random.choice(len(data), n_samples, replace=False)
        data = data[idx]
    return data

def preload_explainers(models_dir='models', background_samples=100):
    """
    Loads RF, NN, and three ensemble variants; builds and caches SHAP explainers.
    """
    # Load base models
    rf_model      = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
    raw_keras_net = load_model(os.path.join(models_dir, 'nn_model.h5'))

    ## Load each ensemble variant from disk
    ensemble_models = {}
    for kind in ['voting', 'adaboost', 'stacking']:
        path = os.path.join(models_dir, f'ensemble_{kind}.joblib')
        ensemble_models[kind] = joblib.load(path)

    #  Determine feature count
    n_features = rf_model.n_features_in_

    #  Wrap the Keras net
    keras_wrapper = KerasBinaryClassifier(model=raw_keras_net, n_features=n_features)

    # Build a background dataset
    background = load_background_data(background_samples, n_features)
    print(f"[preload_explainers] Background of {background.shape} ready")

    explainers = {}

    # === Random Forest ===
    print("[preload_explainers] Building TreeExplainer for Random Forest…")
    explainers['random_forest'] = shap.TreeExplainer(rf_model)
    print("[preload_explainers] working ! Random Forest explainer complete")

    # === Neural Network ===
    print("[preload_explainers] Building KernelExplainer for Neural Network…")
    nn_wrapper = PredictProbaWrapper(keras_wrapper)
    explainers['neural_network'] = shap.KernelExplainer(
        nn_wrapper,
        background,
        output_names=['Not Fraud','Fraud']
    )
    print("[preload_explainers] working ! Neural Network explainer complete")

    # === Ensemble variants ===
    for kind, model in ensemble_models.items():
        key = kind  # use 'voting', 'adaboost', or 'stacking'
        title = kind.title() + " Ensemble"

        if isinstance(model, (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier)):
            print(f"[preload_explainers] Building TreeExplainer for {title}…")
            expl = shap.TreeExplainer(model)
            explainers[key] = expl
            print(f"[preload_explainers] working ! {title} TreeExplainer complete")

        elif isinstance(model, (VotingClassifier, StackingClassifier)):
            print(f"[preload_explainers] Building KernelExplainer for {title}…")
            wrapper = EnsembleProbaWrapper(model)
            expl = shap.KernelExplainer(
                wrapper,
                background,
                output_names=['Not Fraud','Fraud']
            )
            explainers[key] = expl
            print(f"[preload_explainers] working ! {title} KernelExplainer complete")

        else:
            print(f"[preload_explainers] Building fallback KernelExplainer for {title}…")
            wrapper = EnsembleProbaWrapper(model)
            expl = shap.KernelExplainer(
                wrapper,
                background,
                output_names=['Not Fraud','Fraud']
            )
            explainers[key] = expl
            print(f"[preload_explainers] working ! {title} fallback explainer complete")

    # Cache to disk
    cache_path = os.path.join(models_dir, 'explainers_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(explainers, f)
    print(f"[preload_explainers] All explainers cached to {cache_path}")

    return explainers

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    preload_explainers()
    print("preload_explainers finished.")
