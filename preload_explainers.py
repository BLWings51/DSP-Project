# preload_explainers.py

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import shap
from tensorflow.keras.models import load_model, Model
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from utils import load_column_mapping, preprocess_single_transaction, KerasBinaryClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)

import numpy as np

class EnsembleProbaWrapper:
    """Picklable wrapper that always returns an (n,2) proba matrix."""
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        # 1) unwrap SHAP MaskedData or pandas → ndarray
        if hasattr(X, "data"):
            X_arr = np.asarray(X.data, dtype=np.float64)
        elif hasattr(X, "values"):
            X_arr = np.asarray(X.values, dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        # 2) flatten any extra dims → (n_samples, n_features)
        if X_arr.ndim > 2:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)

        # 3) get predict_proba output
        out = self.model.predict_proba(X_arr)
        out = np.asarray(out, dtype=np.float64)

        # 4) force shape → (n_samples,2)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.shape[1] == 1:
            out = np.hstack([1 - out, out])

        return out


def _safe_ensemble_proba(model, X):
    # 1) unwrap SHAP MaskedData or pandas or memoryview → raw numpy
    if hasattr(X, "data"):
        X_arr = np.asarray(X.data, dtype=np.float64)
    elif hasattr(X, "values"):
        X_arr = np.asarray(X.values, dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)

    # 2) flatten any extra dims → (n_samples, n_features)
    if X_arr.ndim > 2:
        X_arr = X_arr.reshape(X_arr.shape[0], -1)

    # 3) get whatever predict_proba gives you
    out = model.predict_proba(X_arr)
    out = np.asarray(out, dtype=np.float64)

    # 4) now force it into (n_samples, 2)
    if out.ndim == 1:
        # got back (n_samples,), make it column
        out = out.reshape(-1, 1)
    if out.shape[1] == 1:
        # got back single‐col P(fraud), build [P(not),P(fraud)]
        out = np.hstack([1 - out, out])

    return out


class PredictProbaWrapper:
    """Ensures SHAP always gets a clean 2-D numpy array back."""
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        # 1) Unwrap SHAP MaskedData, pandas DataFrame, or memoryview → plain ndarray
        if hasattr(X, "data"):
            X_arr = np.asarray(X.data, dtype=np.float64)
        elif hasattr(X, "values"):
            X_arr = np.asarray(X.values, dtype=np.float64)
        elif isinstance(X, memoryview):
            X_arr = np.asarray(X, dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        # 2) Flatten any extra dims → (n_samples, n_features)
        if X_arr.ndim > 2:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)

        # 3) Get raw fraud-probabilities
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X_arr)
            # binary → take the positive class
            if p.ndim == 2 and p.shape[1] == 2:
                p = p[:, 1]
            else:
                p = p.reshape(-1)
        else:
            p = self.model.predict(X_arr, verbose=0).reshape(-1)

        # 4) Return a proper (n_samples, 2) array for SHAP
        return np.column_stack([1 - p, p])



def load_background_data(n_samples=100, n_features=None):
    """Returns a (n_samples × n_features) float64 array from banksimData.csv."""
    df = pd.read_csv('banksimData.csv')
    mapping = load_column_mapping()
    rows = []
    for _, row in df.iterrows():
        proc = preprocess_single_transaction(row, mapping)   # → shape (1, n_raw)
        proc = np.array(proc, dtype=np.float64).reshape(-1)  # → shape (n_raw,)
        rows.append(proc)
    data = np.vstack(rows)                                  # → (n_rows, n_raw)
    if n_features is not None and data.shape[1] != n_features:
        data = data[:, :n_features]
    if len(data) > n_samples:
        idx = np.random.choice(len(data), n_samples, replace=False)
        data = data[idx]
    return data

def preload_explainers(models_dir='models', background_samples=100):
    """
    Loads RF, NN (raw .h5), and Voting ensemble from disk,
    rebuilds the ensemble so that every member is either:
      - a RandomForestClassifier, or
      - a KerasBinaryClassifier wrapper around a tf.keras.Model,
    then creates TreeExplainer for RF and KernelExplainer for the others,
    and finally pickles them to explainers_cache.pkl.
    """
    # 1) Load your saved models
    rf_model       = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
    raw_keras_net  = load_model(os.path.join(models_dir, 'nn_model.h5'))
    ensemble_model = joblib.load(os.path.join(models_dir, 'ensemble_model.joblib'))

    # 2) How many features your RF expects
    n_features = rf_model.n_features_in_

    # 3) A clean KerasBinaryClassifier for your .h5 network
    keras_wrapper = KerasBinaryClassifier(model=raw_keras_net,
                                          n_features=n_features)

    # 4) Build a unified background
    background = load_background_data(background_samples,
                                      n_features=n_features)
    print(f"Loaded background: {background.shape}")

    # 5) Reconstruct every estimator in the VotingClassifier
    if isinstance(ensemble_model, VotingClassifier):
        rebuilt = {}
        for name, member in ensemble_model.named_estimators_.items():
            # a) RandomForest? keep it
            if isinstance(member, RandomForestClassifier):
                rebuilt[name] = member

            # b) If it's a tf.keras.Model instance, wrap it
            elif isinstance(member, Model):
                wrap = KerasBinaryClassifier(model=member,
                                             n_features=n_features)
                wrap.fit(background)  # dummy fit so n_features_in_ is set
                rebuilt[name] = wrap

            # c) If it's already a KerasBinaryClassifier, reuse it
            elif isinstance(member, KerasBinaryClassifier):
                member.fit(background)
                rebuilt[name] = member

            # d) Otherwise assume any sklearn-style estimator with predict_proba
            else:
                member.fit(background, np.zeros(len(background)))
                rebuilt[name] = member

        # overwrite the ensemble's estimators
        ensemble_model.named_estimators_ = rebuilt
        ensemble_model.estimators_       = list(rebuilt.values())
        ensemble_model._n_features_in    = n_features

    # 6) Now build your SHAP explainers
    explainers = {}
    # Random Forest → Tree
    explainers['random_forest'] = shap.TreeExplainer(rf_model)

    # Neural net → Kernel
    nn_wrapper = PredictProbaWrapper(keras_wrapper)
    explainers['neural_network'] = shap.KernelExplainer(
        nn_wrapper,
        background,
        output_names=['Not Fraud', 'Fraud']
    )

    # Ensemble → pick best explainer by type
    if isinstance(ensemble_model, (RandomForestClassifier,
                                   AdaBoostClassifier,
                                   GradientBoostingClassifier)):
        # tree-based ensemble
        explainers['ensemble'] = shap.TreeExplainer(ensemble_model)

    elif isinstance(ensemble_model, StackingClassifier):
        # heterogeneous stack → Kernel on predict_proba
        ensemble_wrapper = EnsembleProbaWrapper(ensemble_model)
        explainers['ensemble'] = shap.KernelExplainer(
            ensemble_wrapper,
            background,
            output_names=['Not Fraud','Fraud']
        )

    elif isinstance(ensemble_model, VotingClassifier):
        # voting classifier → Kernel
        ensemble_wrapper = EnsembleProbaWrapper(ensemble_model)
        explainers['ensemble'] = shap.KernelExplainer(
            ensemble_wrapper,
            background,
            output_names=['Not Fraud','Fraud']
        )

    else:
        # fallback → Kernel on predict_proba
        ensemble_wrapper = EnsembleProbaWrapper(ensemble_model)
        explainers['ensemble'] = shap.KernelExplainer(
            ensemble_wrapper,
            background,
            output_names=['Not Fraud','Fraud']
        )
    
    # 7) Cache to disk
    with open(os.path.join(models_dir, 'explainers_cache.pkl'), 'wb') as f:
        pickle.dump(explainers, f)

    return explainers

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    explainers = preload_explainers()
    print("✅ Preloading done, explainers cached to disk.")
